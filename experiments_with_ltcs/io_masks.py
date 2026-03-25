# io_masks.py — Neuron partition and I/O masks for SRNN experiments
#
# Partitions n neurons into 3 non-overlapping roles:
#   - Input neurons  (1/4): receive external input
#   - Interneurons   (1/2): no direct I/O, purely recurrent
#   - Output neurons (1/4): feed the readout Dense layer
#
# Masks are seeded per-experiment so all models share the same partition
# for a given seed, enabling fair comparison.
#
# Usage:
#   input_idx, inter_idx, output_idx = generate_neuron_partition(32, seed=42)
#   W_in_row_mask = make_input_row_mask(32, input_idx)   # (32,) binary
#   W_out_mask    = make_output_mask(32, output_idx)      # (32,) binary

import numpy as np


def generate_neuron_partition(n, seed, frac_input=0.25, frac_output=0.25):
    """Randomly partition n neurons into input, interneuron, and output roles.

    Args:
        n: Total number of neurons.
        seed: Random seed for reproducibility.
        frac_input: Fraction of neurons receiving external input (default 1/4).
        frac_output: Fraction of neurons feeding the output layer (default 1/4).

    Returns:
        input_indices: ndarray of indices for input neurons.
        inter_indices: ndarray of indices for interneurons.
        output_indices: ndarray of indices for output neurons.
    """
    assert frac_input + frac_output <= 1.0, \
        "frac_input + frac_output must be <= 1.0"

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_input = max(1, int(round(frac_input * n)))
    n_output = max(1, int(round(frac_output * n)))
    # Remaining neurons are interneurons
    n_inter = n - n_input - n_output
    assert n_inter >= 0, \
        f"Not enough neurons: n={n}, n_input={n_input}, n_output={n_output}"

    input_indices = np.sort(perm[:n_input])
    output_indices = np.sort(perm[n_input:n_input + n_output])
    inter_indices = np.sort(perm[n_input + n_output:])

    return input_indices, inter_indices, output_indices


def make_input_row_mask(n, input_indices):
    """Create a binary (n,) mask: 1 for input neurons, 0 otherwise.

    Used to zero out rows of W_in (SRNN) or columns of W_step (CTRNN/NODE)
    for non-input neurons.

    Args:
        n: Total number of neurons.
        input_indices: Array of neuron indices that receive input.

    Returns:
        mask: (n,) float32 binary array.
    """
    mask = np.zeros(n, dtype=np.float32)
    mask[input_indices] = 1.0
    return mask


def make_output_mask(n, output_indices):
    """Create a binary (n,) mask: 1 for output neurons, 0 otherwise.

    Applied as: Dense(n_classes)(state * W_out_mask)
    Zeros out input neurons and interneurons from the readout.

    Args:
        n: Total number of neurons.
        output_indices: Array of neuron indices that feed the output.

    Returns:
        mask: (n,) float32 binary array.
    """
    mask = np.zeros(n, dtype=np.float32)
    mask[output_indices] = 1.0
    return mask
