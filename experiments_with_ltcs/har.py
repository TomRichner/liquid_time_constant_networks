import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
from srnn_model import SRNNCell
from io_masks import generate_neuron_partition, make_input_row_mask, make_output_mask
from sequence_looping import palindrome_loop, palindrome_loop_labels, compute_n_loops, random_window
from time_stretch import stretch_batch, random_stretch_factor
import argparse

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

class HarData:

    def __init__(self,seq_len=16):
        train_x = np.loadtxt("data/har/UCI HAR Dataset/train/X_train.txt")
        train_y = (np.loadtxt("data/har/UCI HAR Dataset/train/y_train.txt")-1).astype(np.int32)
        test_x = np.loadtxt("data/har/UCI HAR Dataset/test/X_test.txt")
        test_y = (np.loadtxt("data/har/UCI HAR Dataset/test/y_test.txt")-1).astype(np.int32)

        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        test_x,test_y = cut_in_sequences(test_x,test_y,seq_len,inc=8)
        print("Total number of training sequences: {}".format(train_x.shape[1]))
        permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,permutation[:valid_size]]
        self.valid_y = train_y[:,permutation[:valid_size]]
        self.train_x = train_x[:,permutation[valid_size:]]
        self.train_y = train_y[:,permutation[valid_size:]]

        self.test_x = test_x
        self.test_y = test_y
        self.seq_len = seq_len
        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self, batch_size=16, rng=None, stretch_lo=1.0,
                      stretch_hi=1.0, min_loops=5, min_loop_len=500):
        """Yield training batches with time-stretch and palindrome looping.

        Returns: (batch_x, batch_y, readout_idx, bptt_start_idx)
            batch_x: (win_len, batch, features)
            batch_y: (win_len, batch) labels (palindromed)
            readout_idx: int — index within window for readout
            bptt_start_idx: int — where to start BPTT
        """
        if rng is None:
            rng = np.random.RandomState()

        total_seqs = self.train_x.shape[1]
        permutation = rng.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]

            # 1. Time stretch (before palindrome looping)
            if abs(stretch_lo - stretch_hi) > 1e-6 or abs(stretch_lo - 1.0) > 1e-6:
                factor = random_stretch_factor(stretch_lo, stretch_hi, rng)
                batch_x, batch_y = stretch_batch(
                    batch_x, batch_y, factor, per_timestep_labels=True)

            # 2. Palindrome looping
            effective_seq_len = batch_x.shape[0]
            n_loops = compute_n_loops(effective_seq_len, min_loop_len, min_loops)
            x_looped = palindrome_loop(batch_x, n_loops)
            y_looped = palindrome_loop_labels(batch_y, n_loops, per_timestep=True)

            # 3. Random windowing
            loop_len = 2 * effective_seq_len
            x_win, y_win, readout_idx, bptt_start_idx = random_window(
                x_looped, y_looped, loop_len, rng)

            yield (x_win, y_win, readout_idx, bptt_start_idx)

    def iterate_eval(self, which="valid", batch_size=None, min_loops=5,
                     min_loop_len=500):
        """Yield eval batches (no stretch, no random windowing).

        Palindrome loops the full sequence, readout at end.
        """
        if which == "valid":
            x, y = self.valid_x, self.valid_y
        elif which == "test":
            x, y = self.test_x, self.test_y
        else:
            raise ValueError(f"Unknown split: {which}")

        n_loops = compute_n_loops(x.shape[0], min_loop_len, min_loops)
        x_looped = palindrome_loop(x, n_loops)
        y_looped = palindrome_loop_labels(y, n_loops, per_timestep=True)
        readout_idx = x_looped.shape[0] - 1  # last timestep

        if batch_size is None or batch_size >= x_looped.shape[1]:
            yield (x_looped, y_looped, readout_idx)
        else:
            n_batches = x_looped.shape[1] // batch_size
            for i in range(n_batches):
                s = i * batch_size
                e = s + batch_size
                yield (x_looped[:, s:e], y_looped[:, s:e], readout_idx)
            if e < x_looped.shape[1]:
                yield (x_looped[:, e:], y_looped[:, e:], readout_idx)


class HarModel:

    def __init__(self, model_type, model_size, learning_rate=0.001,
                 batch_size=16, seed=None, solver="semi_implicit",
                 h=0.0025):
        self.model_type = model_type
        self.constrain_op = None
        self.batch_size = batch_size

        # ── I/O masks (shared across all models for a given seed) ──
        mask_seed = seed if seed is not None else 0
        input_idx, inter_idx, output_idx = generate_neuron_partition(
            model_size, mask_seed)
        W_in_mask_np = make_input_row_mask(model_size, input_idx)
        W_out_mask_np = make_output_mask(model_size, output_idx)

        # Convert to TF constants
        self._W_in_mask = tf.constant(W_in_mask_np, dtype=tf.float32,
                                       name="W_in_mask")
        self._W_out_mask = tf.constant(W_out_mask_np, dtype=tf.float32,
                                        name="W_out_mask")
        print(f"  I/O masks: {len(input_idx)} input neurons, "
              f"{len(inter_idx)} interneurons, {len(output_idx)} output neurons")

        # ── Placeholders ──
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 561])
        self.target_y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.readout_idx = tf.placeholder(dtype=tf.int32, shape=[], name="readout_idx")
        self.bptt_start_idx = tf.placeholder(dtype=tf.int32, shape=[], name="bptt_start_idx")

        self.model_size = model_size
        head = self.x

        # ── Model construction (pass W_in_mask to all models) ──
        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            self.wm = ltc.LTCCell(model_size, W_in_mask=self._W_in_mask)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1,W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1,W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True,W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "hopf"):
            self.fused_cell = SRNNCell(model_size, n_E=model_size,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-per-neuron"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1, dales=True, per_neuron=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-echo"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-no-adapt"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-no-adapt-no-dales"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0, dales=False,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-sfa-only"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=0, n_b_E=0, n_b_I=0, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-std-only"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=0, n_a_I=0, n_b_E=1, n_b_I=0, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-E-only"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-e-only-echo"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-e-only-per-neuron"):
            n_E = model_size // 2
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0, dales=True, per_neuron=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        # ── Single-timestep readout ──
        # head is (T, batch, model_size) time-major
        # Gather single timestep at readout_idx
        head_at_readout = head[self.readout_idx]  # (batch, model_size)

        # Apply W_out_mask (zero non-output neurons)
        masked_state = head_at_readout * self._W_out_mask  # (batch, model_size)

        # Dense readout
        self.y = tf.layers.Dense(6, activation=None, name="dense_readout")(masked_state)
        print("logit shape: ",str(self.y.shape))

        # ── Loss (single timestep) ──
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.target_y,
            logits=self.y,
        ))
        self.lr_var = tf.Variable(1e-8, trainable=False, dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(self.lr_var)
        if model_type in ("srnn-echo", "srnn-e-only-echo"):
            trainable = [v for v in tf.trainable_variables()
                         if "W_in" in v.name or "dense" in v.name]
            self.train_step = optimizer.minimize(self.loss, var_list=trainable)
        else:
            self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(model_prediction, tf.cast(self.target_y, tf.int64)),
            tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","har","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/har")):
            os.makedirs("results/har")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","har","{}".format(model_type))
        if(not os.path.exists("tf_sessions/har")):
            os.makedirs("tf_sessions/har")

        self.saver = tf.train.Saver(max_to_keep=None)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def save_named(self, suffix):
        self.saver.save(self.sess, self.checkpoint_path + suffix)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self, gesture_data, epochs, verbose=True, log_period=50,
            seed=None, stretch_lo=1.0, stretch_hi=1.0,
            min_loops=5, min_loop_len=500):
        from lr_schedule import warmup_hold_cosine_lr

        rng = np.random.RandomState(seed)
        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        batches_per_epoch = gesture_data.train_x.shape[1] // self.batch_size
        total_steps = epochs * batches_per_epoch
        global_step = 0
        self.save_named("_init")
        self.save()
        for e in range(epochs):
            if(e%log_period == 0):
                # Evaluate on validation and test (palindromed, readout at end)
                valid_accs, valid_losses = [], []
                for vx, vy, v_readout in gesture_data.iterate_eval(
                        "valid", min_loops=min_loops, min_loop_len=min_loop_len):
                    va, vl = self.sess.run(
                        [self.accuracy, self.loss],
                        {self.x: vx, self.target_y: vy[v_readout],
                         self.readout_idx: v_readout,
                         self.bptt_start_idx: 0})
                    valid_accs.append(va); valid_losses.append(vl)
                valid_acc = np.mean(valid_accs)
                valid_loss = np.mean(valid_losses)

                test_accs, test_losses = [], []
                for tx, ty, t_readout in gesture_data.iterate_eval(
                        "test", min_loops=min_loops, min_loop_len=min_loop_len):
                    ta, tl = self.sess.run(
                        [self.accuracy, self.loss],
                        {self.x: tx, self.target_y: ty[t_readout],
                         self.readout_idx: t_readout,
                         self.bptt_start_idx: 0})
                    test_accs.append(ta); test_losses.append(tl)
                test_acc = np.mean(test_accs)
                test_loss = np.mean(test_losses)

                # Accuracy metric -> higher is better
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x, batch_y, readout_idx, bptt_start_idx in gesture_data.iterate_train(
                    batch_size=self.batch_size, rng=rng,
                    stretch_lo=stretch_lo, stretch_hi=stretch_hi,
                    min_loops=min_loops, min_loop_len=min_loop_len):
                lr = warmup_hold_cosine_lr(global_step, total_steps)
                self.sess.run(self.lr_var.assign(lr))

                # For training: target_y is the label at the readout index
                target = batch_y[readout_idx]  # (batch,)

                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x,
                     self.target_y: target,
                     self.readout_idx: readout_idx,
                     self.bptt_start_idx: bptt_start_idx})
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)
                global_step += 1

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and e % 10 == 0):
                self.save_named("_epoch{}".format(e))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.save_named("_last")
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--batch_size',default=16,type=int)
    parser.add_argument('--seed',default=None,type=int)
    # New refactor args
    parser.add_argument('--solver',default="semi_implicit",type=str,
                        help="ODE solver: semi_implicit, explicit, rk4, exponential")
    parser.add_argument('--h',default=0.0025,type=float,
                        help="ODE step size")
    parser.add_argument('--min_loops',default=5,type=int,
                        help="Min palindrome fwd+bwd loop pairs")
    parser.add_argument('--min_loop_len',default=500,type=int,
                        help="Min total looped sequence length")
    parser.add_argument('--stretch_lo',default=1.0,type=float,
                        help="Min time-stretch factor (1.0 = disabled)")
    parser.add_argument('--stretch_hi',default=1.0,type=float,
                        help="Max time-stretch factor (1.0 = disabled)")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    har_data = HarData()
    model = HarModel(model_type=args.model, model_size=args.size,
                     learning_rate=args.lr, batch_size=args.batch_size,
                     seed=args.seed, solver=args.solver, h=args.h)

    model.fit(har_data, epochs=args.epochs, log_period=args.log,
              seed=args.seed, stretch_lo=args.stretch_lo,
              stretch_hi=args.stretch_hi, min_loops=args.min_loops,
              min_loop_len=args.min_loop_len)
