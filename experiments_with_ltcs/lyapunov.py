# lyapunov.py — Benettin's algorithm for largest Lyapunov exponent (LLE)
#
# Port from Intersect-LNNs-SRNNs/JuliaLang/src/lyapunov.jl.
# Works with any TF1 RNNCell by stepping through a pre-recorded input
# sequence.
#
# Usage:
#   from lyapunov import compute_lyapunov_at_checkpoint
#
#   lle, local_lya = compute_lyapunov_at_checkpoint(
#       sess, cell, x_placeholder, state_placeholder,
#       val_batch_x, initial_state_np,
#       save_dir="lyapunov", epoch=10)

import os
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def benettin_lle_numpy(state_trajectory, step_fn, steps_per_interval,
                       tau_interval=1.0, d0=1e-3, skip_intervals=0,
                       seed=42):
    """Compute largest Lyapunov exponent using Benettin's algorithm.

    Pure numpy implementation — the step_fn runs one macro-step.

    Args:
        state_trajectory: (n_total_steps+1, state_dim) — reference states.
            Row 0 is the initial state, row k is state after k macro-steps.
        step_fn: Callable(state, step_idx) → new_state.
            Steps the perturbed trajectory one macro-step.
            step_idx is 1-based global step index.
        steps_per_interval: Number of macro-steps per renormalization interval.
        tau_interval: Real-time duration of one interval (for units of 1/time).
        d0: Initial perturbation magnitude.
        skip_intervals: Number of initial intervals to skip (transient burn-in).
        seed: Random seed for perturbation direction.

    Returns:
        LLE: Largest Lyapunov exponent (time-averaged, 1/time).
        local_lya: Per-interval instantaneous exponent.
        finite_lya: Running time-average exponent (NaN during burn-in).
    """
    n_total_steps = state_trajectory.shape[0] - 1
    state_dim = state_trajectory.shape[1]

    n_intervals = n_total_steps // steps_per_interval
    if n_intervals < 1:
        raise ValueError(
            f"Trajectory too short: {n_total_steps} steps, "
            f"need at least {steps_per_interval}.")

    local_lya = np.zeros(n_intervals, dtype=np.float32)
    finite_lya = np.full(n_intervals, np.nan, dtype=np.float32)
    sum_log_stretching = 0.0
    accumulated_time = 0.0

    rng = np.random.RandomState(seed)
    rnd_dir = rng.randn(state_dim).astype(np.float32)
    pert = (rnd_dir / np.linalg.norm(rnd_dir)) * d0

    for k in range(n_intervals):
        ref_start_idx = k * steps_per_interval
        ref_end_idx = (k + 1) * steps_per_interval

        S_ref_start = state_trajectory[ref_start_idx]
        S_ref_end = state_trajectory[ref_end_idx]

        # Perturbed trajectory
        S_pert = S_ref_start + pert

        for step in range(steps_per_interval):
            global_step_idx = k * steps_per_interval + step + 1  # 1-based
            S_pert = step_fn(S_pert, global_step_idx)

        # Measure divergence
        delta = S_pert - S_ref_end
        d_k = np.linalg.norm(delta)

        if d_k == 0 or not np.isfinite(d_k):
            local_lya = local_lya[:k]
            finite_lya = finite_lya[:k]
            valid = finite_lya[~np.isnan(finite_lya)]
            LLE = valid[-1] if len(valid) > 0 else 0.0
            return float(LLE), local_lya, finite_lya

        local_lya[k] = np.log(d_k / d0) / tau_interval

        # Renormalize
        pert = (delta / d_k) * d0

        if k >= skip_intervals:
            sum_log_stretching += np.log(d_k / d0)
            accumulated_time += tau_interval
            finite_lya[k] = sum_log_stretching / accumulated_time

    valid = finite_lya[~np.isnan(finite_lya)]
    LLE = float(valid[-1]) if len(valid) > 0 else 0.0

    return LLE, local_lya, finite_lya


def collect_reference_trajectory(sess, cell, x_placeholder, state_placeholder,
                                 input_sequence, initial_state):
    """Run cell forward and save states at each macro-step.

    Args:
        sess: tf.Session
        cell: RNNCell (already built)
        x_placeholder: (batch=1, input_dim) placeholder for one timestep
        state_placeholder: (batch=1, state_dim) placeholder
        input_sequence: (T, input_dim) numpy array
        initial_state: (state_dim,) numpy array

    Returns:
        trajectory: (T+1, state_dim) numpy array
    """
    import tensorflow.compat.v1 as tf

    T = input_sequence.shape[0]
    state_dim = initial_state.shape[0]
    trajectory = np.zeros((T + 1, state_dim), dtype=np.float32)
    trajectory[0] = initial_state

    # Build step op if needed
    if not hasattr(cell, '_lya_output_op'):
        with tf.variable_scope("lyapunov_step", reuse=tf.AUTO_REUSE):
            cell._lya_output_op, cell._lya_state_op = cell(
                x_placeholder, state_placeholder)

    state = initial_state[np.newaxis, :]  # (1, state_dim)
    for t in range(T):
        inp = input_sequence[t:t+1, :]  # (1, input_dim)
        _, state = sess.run(
            [cell._lya_output_op, cell._lya_state_op],
            feed_dict={x_placeholder: inp, state_placeholder: state})
        trajectory[t + 1] = state[0]

    return trajectory


def _make_step_fn(sess, cell, x_placeholder, state_placeholder, input_sequence):
    """Create a numpy step function that replays the input sequence."""
    def step_fn(state_np, step_idx):
        # step_idx is 1-based; wrap for input_sequence
        t = (step_idx - 1) % input_sequence.shape[0]
        inp = input_sequence[t:t+1, :]  # (1, input_dim)
        state_batch = state_np[np.newaxis, :]  # (1, state_dim)
        _, new_state = sess.run(
            [cell._lya_output_op, cell._lya_state_op],
            feed_dict={x_placeholder: inp,
                       state_placeholder: state_batch})
        return new_state[0]
    return step_fn


def save_lyapunov_hdf5(filepath, epoch, LLE, local_lya, finite_lya,
                       ref_traj_last=None, pert_traj_last=None):
    """Save Lyapunov results to HDF5 with compression.

    Args:
        filepath: Path to .h5 file.
        epoch: Epoch number.
        LLE: Scalar largest Lyapunov exponent.
        local_lya: Per-interval local LE.
        finite_lya: Running average LE.
        ref_traj_last: Reference trajectory for last N loops (state_dim, T_save).
        pert_traj_last: Perturbed trajectory for last N loops.
    """
    if not HAS_H5PY:
        print("WARNING: h5py not available, skipping Lyapunov HDF5 save")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.attrs['epoch'] = epoch
        f.attrs['LLE'] = LLE

        f.create_dataset('local_lya', data=local_lya.astype(np.float32),
                         compression='gzip', compression_opts=1, shuffle=True)
        f.create_dataset('finite_lya', data=finite_lya.astype(np.float32),
                         compression='gzip', compression_opts=1, shuffle=True)

        if ref_traj_last is not None:
            # Convert to float16, orient as (state_dim, T_save)
            data = ref_traj_last.T.astype(np.float16)
            state_dim, T_save = data.shape
            chunks = (state_dim, min(256, T_save))
            f.create_dataset('ref_trajectory', data=data,
                             compression='gzip', compression_opts=1,
                             shuffle=True, chunks=chunks)

        if pert_traj_last is not None:
            data = pert_traj_last.T.astype(np.float16)
            state_dim, T_save = data.shape
            chunks = (state_dim, min(256, T_save))
            f.create_dataset('pert_trajectory', data=data,
                             compression='gzip', compression_opts=1,
                             shuffle=True, chunks=chunks)


def compute_lyapunov_at_checkpoint(
        sess, cell, x_placeholder, state_placeholder,
        val_batch_x, initial_state_np,
        save_dir, epoch,
        n_palindrome_loops=20, save_last_n_loops=3,
        steps_per_interval=None, tau_interval=None,
        seed=42):
    """Compute LLE at a checkpoint and save results.

    Args:
        sess: tf.Session
        cell: RNNCell
        x_placeholder: (1, input_dim) placeholder
        state_placeholder: (1, state_dim) placeholder
        val_batch_x: (T, input_dim) validation input (single sequence)
        initial_state_np: (state_dim,) numpy initial state
        save_dir: Directory for HDF5 output
        epoch: Current epoch
        n_palindrome_loops: Number of fwd+bwd loops for LE computation
        save_last_n_loops: Number of last loops to save trajectories for
        steps_per_interval: Steps per renormalization interval
            (default: one fwd+bwd loop = 2*T)
        tau_interval: Real-time per interval
            (default: steps_per_interval * h * unfolds if available)
        seed: Random seed

    Returns:
        LLE: Largest Lyapunov exponent
    """
    from sequence_looping import palindrome_loop

    T = val_batch_x.shape[0]

    # Build palindrome-looped input
    # val_batch_x is (T, input_dim) — add dummy batch dim then remove
    x_looped = palindrome_loop(
        val_batch_x[:, np.newaxis, :], n_palindrome_loops)[:, 0, :]
    # x_looped is (T_total, input_dim)
    T_total = x_looped.shape[0]

    # Default steps_per_interval = one fwd+bwd loop
    if steps_per_interval is None:
        steps_per_interval = 2 * T

    # Default tau_interval
    if tau_interval is None:
        if hasattr(cell, '_h') and hasattr(cell, '_ode_solver_unfolds'):
            tau_interval = float(
                steps_per_interval * cell._h * cell._ode_solver_unfolds)
        elif hasattr(cell, '_delta_t') and hasattr(cell, '_unfolds'):
            tau_interval = float(
                steps_per_interval * cell._delta_t * cell._unfolds)
        else:
            tau_interval = float(steps_per_interval)

    # Collect reference trajectory
    ref_traj = collect_reference_trajectory(
        sess, cell, x_placeholder, state_placeholder,
        x_looped, initial_state_np)

    # Build step function
    step_fn = _make_step_fn(
        sess, cell, x_placeholder, state_placeholder, x_looped)

    # Run Benettin
    LLE, local_lya, finite_lya = benettin_lle_numpy(
        ref_traj, step_fn, steps_per_interval,
        tau_interval=tau_interval, d0=1e-3,
        skip_intervals=2, seed=seed)

    # Extract last N loops of trajectories for saving
    save_steps = save_last_n_loops * 2 * T
    ref_traj_last = ref_traj[-save_steps:] if save_steps <= ref_traj.shape[0] else ref_traj

    # Re-run perturbed trajectory for last N loops to get its trajectory
    # (Benettin doesn't save full perturbed traj — just re-run from the
    #  appropriate starting point with the final perturbation direction)
    # For simplicity, save None for pert_traj — can be added later if needed
    pert_traj_last = None

    # Save
    h5_path = os.path.join(save_dir, f"lyapunov_epoch_{epoch:04d}.h5")
    save_lyapunov_hdf5(
        h5_path, epoch, LLE, local_lya, finite_lya,
        ref_traj_last=ref_traj_last, pert_traj_last=pert_traj_last)

    print(f"  Lyapunov: LLE = {LLE:.4f} (epoch {epoch}, "
          f"{len(local_lya)} intervals, saved to {h5_path})")

    return LLE
