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
from training_utils import wrap_train_batch, wrap_eval_batch, setup_lyapunov_ops, run_lyapunov_if_due
import argparse
import pandas as pd

def read_file(filename):
    df = pd.read_csv(filename)                                    

    data_x = np.stack([
        df['Temperature'].values,
        df['Humidity'].values,
        df['Light'].values,
        df['CO2'].values,
        df['HumidityRatio'].values,
        ],axis=-1)
    data_y = df['Occupancy'].values.astype(np.int32)
    return data_x,data_y

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

class OccupancyData:

    def __init__(self,seq_len=16):
        train_x,train_y = read_file("data/occupancy/datatraining.txt")
        test0_x,test0_y = read_file("data/occupancy/datatest.txt")
        test1_x,test1_y = read_file("data/occupancy/datatest2.txt")

        mean_x = np.mean(train_x,axis=0)
        std_x = np.std(train_x,axis=0)
        train_x = (train_x-mean_x)/std_x
        test0_x = (test0_x-mean_x)/std_x
        test1_x = (test1_x-mean_x)/std_x

        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        test0_x,test0_y = cut_in_sequences(test0_x,test0_y,seq_len,inc=8)
        test1_x,test1_y = cut_in_sequences(test1_x,test1_y,seq_len,inc=8)
        print("Total number of training sequences: {}".format(train_x.shape[1]))
        permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,permutation[:valid_size]]
        self.valid_y = train_y[:,permutation[:valid_size]]
        self.train_x = train_x[:,permutation[valid_size:]]
        self.train_y = train_y[:,permutation[valid_size:]]

        self.test_x = np.concatenate([test0_x,test1_x],axis=1)
        self.test_y = np.concatenate([test0_y,test1_y],axis=1)
        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

        

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

class OccupancyModel:

    def __init__(self,model_type,model_size,sparsity_level=0.0,learning_rate = 0.001,batch_size=16):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.batch_size = batch_size

        # ── I/O masks ──
        mask_seed = seed if seed is not None else 0
        input_idx, inter_idx, output_idx = generate_neuron_partition(
            model_size, mask_seed)
        W_in_mask_np = make_input_row_mask(model_size, input_idx)
        W_out_mask_np = make_output_mask(model_size, output_idx)
        self._W_in_mask = tf.constant(W_in_mask_np, dtype=tf.float32, name="W_in_mask")
        self._W_out_mask = tf.constant(W_out_mask_np, dtype=tf.float32, name="W_out_mask")
        print(f"  I/O masks: {len(input_idx)} input, {len(inter_idx)} inter, {len(output_idx)} output")
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,5])
        self.target_y = tf.placeholder(dtype=tf.int32, shape=[None])


        self.readout_idx = tf.placeholder(dtype=tf.int32, shape=[], name="readout_idx")
        self.bptt_start_idx = tf.placeholder(dtype=tf.int32, shape=[], name="bptt_start_idx")
        self.model_size = model_size
        head = self.x
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
            n_E = model_size // 2  # 50% excitatory
            self.fused_cell = SRNNCell(model_size, n_E=n_E,
                n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1, dales=True,
                solver=solver, h=h, W_in_mask=self._W_in_mask)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "srnn-per-neuron"):
            n_E = model_size // 2  # 50% excitatory
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
        elif(model_type == "srnn-e-only"):
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

        if(self.sparsity_level > 0):
            if self.constrain_op is None:
                self.constrain_op = self.get_sparsity_ops()
            else:
                self.constrain_op.extend(self.get_sparsity_ops())

        # ── Single-timestep readout ──
        head_at_readout = head[self.readout_idx]
        masked_state = head_at_readout * self._W_out_mask
        self.y = tf.layers.Dense(2, activation=None, name="dense_readout")(masked_state)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        self.lr_var = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(self.lr_var)
        if model_type in ("srnn-echo", "srnn-e-only-echo"):
            trainable = [v for v in tf.trainable_variables()
                         if "W_in" in v.name or "dense" in v.name]
            self.train_step = optimizer.minimize(self.loss, var_list=trainable)
        else:
            self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","occupancy","{}_{}_{:02d}.csv".format(model_type,model_size,int(100*self.sparsity_level)))
        if(not os.path.exists("results/occupancy")):
            os.makedirs("results/occupancy")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","occupancy","{}".format(model_type))
        if(not os.path.exists("tf_sessions/occupancy")):
            os.makedirs("tf_sessions/occupancy")
            
        self.saver = tf.train.Saver(max_to_keep=None)

        # ── Lyapunov placeholders (single-step ops) ──
        state_dim = self.fused_cell.state_size if hasattr(self, 'fused_cell') else (
            self.wm.state_size if hasattr(self, 'wm') else model_size)
        if hasattr(state_dim, '__len__'):
            state_dim = sum(state_dim)
        self._lya_x_ph, self._lya_s_ph = setup_lyapunov_ops(
            self.fused_cell if hasattr(self, 'fused_cell') else self.wm,
            5, state_dim)

    def get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def save_named(self, suffix):
        self.saver.save(self.sess, self.checkpoint_path + suffix)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):
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
            if(verbose and e%log_period == 0):
                _x_eval, _y_eval, _ri_eval = wrap_eval_batch(
                    gesture_data.test_x, gesture_data.test_y,
                    min_loops=min_loops, min_loop_len=min_loop_len,
                    per_timestep_labels=True)
                test_acc,test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: _x_eval, self.target_y: _y_eval,
                     self.readout_idx: _ri_eval, self.bptt_start_idx: 0})
                _x_eval, _y_eval, _ri_eval = wrap_eval_batch(
                    gesture_data.valid_x, gesture_data.valid_y,
                    min_loops=min_loops, min_loop_len=min_loop_len,
                    per_timestep_labels=True)
                valid_acc,valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: _x_eval, self.target_y: _y_eval,
                     self.readout_idx: _ri_eval, self.bptt_start_idx: 0})
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
            for raw_x, raw_y in gesture_data.iterate_train(batch_size=self.batch_size):
                batch_x, batch_y, readout_idx, bptt_start_idx = wrap_train_batch(
                    raw_x, raw_y, rng, stretch_lo=stretch_lo, stretch_hi=stretch_hi,
                    min_loops=min_loops, min_loop_len=min_loop_len,
                    per_timestep_labels=True)
                lr = warmup_hold_cosine_lr(global_step, total_steps)
                self.sess.run(self.lr_var.assign(lr))
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x: batch_x, self.target_y: batch_y[readout_idx],
                     self.readout_idx: readout_idx, self.bptt_start_idx: bptt_start_idx})
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
                self.save_named("_epochgesture_data".format(e))
                # Lyapunov at checkpoint
                checkpoint_epochs = set(range(10, epochs+1, 10))
                lya_cell = self.fused_cell if hasattr(self, "fused_cell") else self.wm
                run_lyapunov_if_due(
                    e, checkpoint_epochs, self.sess, lya_cell,
                    self._lya_x_ph, self._lya_s_ph,
                    gesture_data.valid_x, os.path.join("lyapunov", "occupancy"),
                    seed=seed if seed is not None else 42)
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
    parser.add_argument('--sparsity',default=0.0,type=float)

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

    occ_data = OccupancyData()
    model = OccupancyModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity,learning_rate=args.lr,batch_size=args.batch_size)

    model.fit(occ_data,epochs=args.epochs,log_period=args.log,
              seed=args.seed, stretch_lo=args.stretch_lo,
              stretch_hi=args.stretch_hi, min_loops=args.min_loops,
              min_loop_len=args.min_loop_len)

