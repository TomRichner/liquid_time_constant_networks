# training_utils.py — Shared training loop helpers
#
# Wraps old-style (batch_x, batch_y) batches with palindrome looping,
# time-stretching, and random windowing so that each experiment doesn't
# need to reimplement these.

import numpy as np
from sequence_looping import (palindrome_loop, palindrome_loop_labels,
                              compute_n_loops, random_window)
from time_stretch import stretch_batch, random_stretch_factor


def wrap_train_batch(batch_x, batch_y, rng,
                     stretch_lo=1.0, stretch_hi=1.0,
                     min_loops=5, min_loop_len=500,
                     per_timestep_labels=True):
    """Apply time-stretch + palindrome looping + random windowing to a batch.

    Args:
        batch_x: (T, batch, features) time-major.
        batch_y: Labels. If per_timestep: (T, batch). If not: (batch,).
        rng: numpy RandomState.
        stretch_lo/hi: Time-stretch range.
        min_loops: Min palindrome loop pairs.
        min_loop_len: Min total looped sequence length.
        per_timestep_labels: Whether labels have a time dimension.

    Returns:
        x_win: (win_len, batch, features) windowed sequence.
        y_win: (win_len, batch) or (batch,) windowed labels.
        readout_idx: int — index within window for readout.
        bptt_start_idx: int — where to start BPTT.
    """
    # 1. Time stretch (before palindrome looping)
    if abs(stretch_lo - stretch_hi) > 1e-6 or abs(stretch_lo - 1.0) > 1e-6:
        factor = random_stretch_factor(stretch_lo, stretch_hi, rng)
        batch_x, batch_y = stretch_batch(
            batch_x, batch_y, factor, per_timestep_labels=per_timestep_labels)

    if not per_timestep_labels:
        # For single-label tasks (smnist), palindrome the input only
        effective_seq_len = batch_x.shape[0]
        n_loops = compute_n_loops(effective_seq_len, min_loop_len, min_loops)
        x_looped = palindrome_loop(batch_x, n_loops)
        loop_len = 2 * effective_seq_len

        # Random window on x only
        T_total = x_looped.shape[0]
        n_full_loops = T_total // loop_len
        bptt_loops = min(2, n_full_loops)
        bptt_start = max(0, T_total - bptt_loops * loop_len)

        # Random readout in last loop
        last_loop_start = T_total - loop_len
        readout_idx = rng.randint(last_loop_start, T_total)

        return x_looped, batch_y, readout_idx, bptt_start
    else:
        # Per-timestep labels: palindrome both x and y
        effective_seq_len = batch_x.shape[0]
        n_loops = compute_n_loops(effective_seq_len, min_loop_len, min_loops)
        x_looped = palindrome_loop(batch_x, n_loops)
        y_looped = palindrome_loop_labels(batch_y, n_loops, per_timestep=True)

        # Random windowing
        loop_len = 2 * effective_seq_len
        x_win, y_win, readout_idx, bptt_start_idx = random_window(
            x_looped, y_looped, loop_len, rng)

        return x_win, y_win, readout_idx, bptt_start_idx


def wrap_eval_batch(batch_x, batch_y,
                    min_loops=5, min_loop_len=500,
                    per_timestep_labels=True):
    """Palindrome loop for eval (no stretch, no random windowing).

    Args:
        batch_x: (T, batch, features).
        batch_y: Labels.
        min_loops/min_loop_len: Palindrome config.
        per_timestep_labels: Whether labels have a time dimension.

    Returns:
        x_looped: (T_looped, batch, features).
        y_at_readout: (batch,) or (batch, n_classes) — label at readout.
        readout_idx: int — last timestep.
    """
    effective_seq_len = batch_x.shape[0]
    n_loops = compute_n_loops(effective_seq_len, min_loop_len, min_loops)
    x_looped = palindrome_loop(batch_x, n_loops)
    readout_idx = x_looped.shape[0] - 1

    if per_timestep_labels:
        y_looped = palindrome_loop_labels(batch_y, n_loops, per_timestep=True)
        y_at_readout = y_looped[readout_idx]
    else:
        y_at_readout = batch_y

    return x_looped, y_at_readout, readout_idx


def setup_lyapunov_ops(cell, input_dim, state_dim):
    """Build single-step TF ops needed for Lyapunov computation.

    Call once after the main graph is built (but before session.run).

    Args:
        cell: The RNNCell used in the model.
        input_dim: Input feature dimension.
        state_dim: Full state dimension of the cell.

    Returns:
        x_ph: (1, input_dim) placeholder for single-timestep input.
        s_ph: (1, state_dim) placeholder for state.
    """
    import tensorflow.compat.v1 as tf

    x_ph = tf.placeholder(tf.float32, [1, input_dim], name="lya_input")
    s_ph = tf.placeholder(tf.float32, [1, state_dim], name="lya_state")
    return x_ph, s_ph


def run_lyapunov_if_due(epoch, checkpoint_epochs, sess, cell,
                        lya_x_ph, lya_s_ph,
                        val_x, save_dir,
                        n_palindrome_loops=20, save_last_n_loops=3,
                        seed=42):
    """Compute Lyapunov exponent at checkpoint epochs.

    Call from fit() after saving a checkpoint. Does nothing if epoch is
    not in checkpoint_epochs.

    Args:
        epoch: Current epoch.
        checkpoint_epochs: Set or list of epochs to trigger on.
        sess: tf.Session.
        cell: The RNNCell (must have state_size).
        lya_x_ph: (1, input_dim) placeholder from setup_lyapunov_ops.
        lya_s_ph: (1, state_dim) placeholder from setup_lyapunov_ops.
        val_x: (T, N, features) validation data. A random sequence is picked.
        save_dir: Where to save HDF5 results.
        n_palindrome_loops: Number of fwd+bwd loops for LE computation.
        save_last_n_loops: Loops to save trajectories for.
        seed: Random seed.

    Returns:
        LLE or None if not a checkpoint epoch.
    """
    if epoch not in checkpoint_epochs:
        return None

    from lyapunov import compute_lyapunov_at_checkpoint

    # Pick a random validation sequence
    rng = np.random.RandomState(seed + epoch)
    seq_idx = rng.randint(0, val_x.shape[1])
    val_seq = val_x[:, seq_idx, :]  # (T, features)

    # Get current initial state (zeros for now — will be IC when integrated)
    state_dim = cell.state_size
    if hasattr(state_dim, '__len__'):
        # LSTMStateTuple or similar
        state_dim = sum(state_dim)
    initial_state = np.zeros(state_dim, dtype=np.float32)

    LLE = compute_lyapunov_at_checkpoint(
        sess, cell, lya_x_ph, lya_s_ph,
        val_seq, initial_state,
        save_dir=save_dir, epoch=epoch,
        n_palindrome_loops=n_palindrome_loops,
        save_last_n_loops=save_last_n_loops,
        seed=seed)

    return LLE
