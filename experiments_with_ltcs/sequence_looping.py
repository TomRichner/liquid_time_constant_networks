# sequence_looping.py — Palindrome looping and random windowing
#
# Constructs forward/backward palindrome loops from input sequences,
# applies random start/end windowing for constant-length output,
# and provides random readout index within the last loop.
#
# Usage:
#   from sequence_looping import palindrome_loop, compute_n_loops, random_window
#
#   n_loops = compute_n_loops(seq_len=16, min_loop_len=500, min_loops=5)
#   x_looped = palindrome_loop(batch_x, n_loops)         # (T_new, batch, feat)
#   y_looped = palindrome_loop_labels(batch_y, n_loops)   # labels mirrored
#   x_win, y_win, readout_idx, bptt_start = random_window(
#       x_looped, y_looped, loop_len=2*seq_len, rng=rng)

import numpy as np


def compute_n_loops(seq_len, min_loop_len=500, min_loops=5):
    """Compute number of palindrome fwd+bwd loop pairs.

    Each loop pair = one forward pass + one backward pass = 2 * seq_len steps.

    Args:
        seq_len: Length of the original sequence.
        min_loop_len: Minimum total looped length in timesteps.
        min_loops: Minimum number of fwd+bwd loop pairs.

    Returns:
        n_loops: Number of fwd+bwd pairs (each pair is 2*seq_len long).
    """
    loops_for_length = int(np.ceil(min_loop_len / (2.0 * seq_len)))
    return max(min_loops, loops_for_length)


def palindrome_loop(x, n_loops):
    """Create palindrome-looped sequence from input.

    Constructs: [fwd, bwd, fwd, bwd, ...] × n_loops pairs.

    Args:
        x: Input array, time-major (T, batch, features) or (T, batch).

    Returns:
        x_looped: (n_loops * 2 * T, batch, features) time-major.
    """
    # Forward = x, Backward = x reversed along time axis
    x_fwd = x
    x_bwd = x[::-1]

    pieces = []
    for _ in range(n_loops):
        pieces.append(x_fwd)
        pieces.append(x_bwd)

    return np.concatenate(pieces, axis=0)


def palindrome_loop_labels(y, n_loops, per_timestep=True):
    """Mirror labels to match palindrome-looped sequence.

    Args:
        y: Labels. If per_timestep: (T, batch) or (T, batch, n_classes).
           If not per_timestep: (batch,) — single label per sequence.
        n_loops: Number of fwd+bwd pairs.
        per_timestep: Whether labels are per-timestep.

    Returns:
        y_looped: Mirrored labels matching palindrome structure.
            If per_timestep: same shape pattern as palindrome_loop output.
            If not per_timestep: returns y unchanged (single label).
    """
    if not per_timestep:
        return y

    y_fwd = y
    y_bwd = y[::-1]

    pieces = []
    for _ in range(n_loops):
        pieces.append(y_fwd)
        pieces.append(y_bwd)

    return np.concatenate(pieces, axis=0)


def random_window(x_looped, y_looped, loop_len, rng, n_bptt_loops=2,
                  per_timestep_labels=True):
    """Extract a constant-length window from palindrome-looped data.

    Picks a random start offset within the first loop, trims symmetrically
    from both ends to produce (n_loops - 1) * loop_len timesteps.

    Also returns the readout index (random point in last loop) and the
    BPTT boundary (where to insert tf.stop_gradient).

    Args:
        x_looped: Palindrome-looped input (T_total, batch, features).
        y_looped: Palindrome-looped labels (T_total, batch) or (batch,).
        loop_len: Length of one fwd+bwd loop pair (2 * seq_len).
        rng: numpy RandomState for reproducibility.
        n_bptt_loops: Number of loops at the end to backpropagate through.
        per_timestep_labels: Whether labels have a time dimension.

    Returns:
        x_win: Windowed input (win_len, batch, features).
        y_win: Windowed labels (matching).
        readout_idx: Random index within the last loop of the window.
        bptt_start_idx: Index in x_win where BPTT should begin
            (earlier timesteps get stop_gradient).
    """
    T_total = x_looped.shape[0]
    n_total_loops = T_total // loop_len

    if n_total_loops < 2:
        # Not enough loops to window; return as-is
        readout_idx = T_total - 1
        return x_looped, y_looped, readout_idx, 0

    # Random start offset within first loop
    i = rng.randint(0, loop_len)

    # Window: from i to T_total - (loop_len - i)
    # This gives exactly (n_total_loops - 1) * loop_len timesteps
    end = T_total - (loop_len - i)
    win_len = end - i  # = (n_total_loops - 1) * loop_len

    x_win = x_looped[i:end]
    if per_timestep_labels:
        y_win = y_looped[i:end]
    else:
        y_win = y_looped

    # Readout index: random point within the last loop of the window
    last_loop_start = win_len - loop_len
    readout_idx = rng.randint(last_loop_start, win_len)

    # BPTT boundary: only backpropagate through last n_bptt_loops
    bptt_len = n_bptt_loops * loop_len
    bptt_start_idx = max(0, win_len - bptt_len)

    return x_win, y_win, readout_idx, bptt_start_idx
