# time_stretch.py — PCHIP time-stretching data augmentation
#
# Resamples sequences using Piecewise Cubic Hermite Interpolation (PCHIP)
# to train timescale-invariant representations. The ODE step size h remains
# fixed — only the input spacing changes.
#
# Usage:
#   from time_stretch import stretch_batch, random_stretch_factor
#
#   factor = random_stretch_factor(lo=0.25, hi=4.0, rng=rng)
#   x_stretched, y_stretched = stretch_batch(batch_x, batch_y, factor)

import numpy as np
from scipy.interpolate import PchipInterpolator


def random_stretch_factor(lo=0.25, hi=4.0, rng=None):
    """Sample a random time-stretch factor (log-uniform).

    Args:
        lo: Minimum stretch factor.
        hi: Maximum stretch factor.
        rng: numpy RandomState (default: np.random).

    Returns:
        factor: float in [lo, hi], log-uniformly sampled.
    """
    if rng is None:
        rng = np.random
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return float(np.exp(rng.uniform(log_lo, log_hi)))


def pchip_stretch(x, stretch_factor):
    """Resample a time-major array using PCHIP interpolation.

    Args:
        x: Input array, time-major. Shape (T, ...) where ... can be
           (batch, features) or (batch,) or any trailing dims.
        stretch_factor: Factor > 0. >1 = slower (more steps),
            <1 = faster (fewer steps).

    Returns:
        x_stretched: Resampled array (T_new, ...) where T_new = round(T * factor).
    """
    T = x.shape[0]
    T_new = max(2, int(round(T * stretch_factor)))

    # Original sample positions (normalized to [0, 1])
    t_orig = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, T_new)

    # Reshape to (T, -1) for interpolation, then reshape back
    orig_shape = x.shape
    x_flat = x.reshape(T, -1)  # (T, M)
    M = x_flat.shape[1]

    x_new_flat = np.empty((T_new, M), dtype=x.dtype)
    for col in range(M):
        interp = PchipInterpolator(t_orig, x_flat[:, col])
        x_new_flat[:, col] = interp(t_new).astype(x.dtype)

    # Reshape back to original trailing dimensions
    new_shape = (T_new,) + orig_shape[1:]
    return x_new_flat.reshape(new_shape)


def stretch_batch(batch_x, batch_y, stretch_factor, per_timestep_labels=True):
    """Apply time stretch to a batch of sequences and their labels.

    Args:
        batch_x: Input (T, batch, features), time-major.
        batch_y: Labels. If per_timestep: (T, batch) or (T, batch, n_classes).
            If not: (batch,) — unchanged.
        stretch_factor: Stretch factor (float).
        per_timestep_labels: Whether labels have a time dimension.

    Returns:
        x_stretched: (T_new, batch, features).
        y_stretched: If per_timestep: (T_new, batch, ...).
            If not: (batch,) unchanged.
    """
    if abs(stretch_factor - 1.0) < 1e-6:
        return batch_x, batch_y

    x_stretched = pchip_stretch(batch_x, stretch_factor)

    if per_timestep_labels:
        y_stretched = pchip_stretch_labels(batch_y, stretch_factor)
    else:
        y_stretched = batch_y

    return x_stretched, y_stretched


def pchip_stretch_labels(y, stretch_factor):
    """Resample per-timestep labels via nearest-neighbor.

    For classification labels (integers), PCHIP interpolation doesn't make
    sense — use nearest-neighbor instead.

    For regression targets (floats), use PCHIP.

    Args:
        y: Labels, time-major (T, ...).
        stretch_factor: Factor > 0.

    Returns:
        y_stretched: Resampled labels (T_new, ...).
    """
    T = y.shape[0]
    T_new = max(2, int(round(T * stretch_factor)))

    if np.issubdtype(y.dtype, np.integer):
        # Classification: nearest-neighbor
        t_orig = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, T_new)
        indices = np.searchsorted(t_orig, t_new, side='right') - 1
        indices = np.clip(indices, 0, T - 1)
        return y[indices]
    else:
        # Regression: PCHIP interpolation
        return pchip_stretch(y, stretch_factor)
