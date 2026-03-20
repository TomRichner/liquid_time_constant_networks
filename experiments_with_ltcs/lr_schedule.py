"""Warmup-Hold-Cosine learning rate schedule.

Schedule phases (by fraction of total training batches):
    0%  - 20%: Linear warmup from start_lr to max_lr
    20% - 70%: Hold at max_lr
    70% - 100%: Cosine decay from max_lr to end_lr
"""

import math


def warmup_hold_cosine_lr(step, total_steps,
                          start_lr=1e-8, max_lr=5e-3, end_lr=None):
    """Compute learning rate for a given batch step.

    Args:
        step: Current batch step (0-indexed).
        total_steps: Total number of training batches across all epochs.
        start_lr: Initial learning rate at step 0.
        max_lr: Peak learning rate after warmup.
        end_lr: Final learning rate after cosine decay (default: max_lr / 20).

    Returns:
        Learning rate for this step.
    """
    if end_lr is None:
        end_lr = max_lr / 20.0

    warmup_end = int(0.20 * total_steps)
    hold_end = int(0.70 * total_steps)

    if step < warmup_end:
        # Linear warmup
        return start_lr + (max_lr - start_lr) * step / max(warmup_end, 1)
    elif step < hold_end:
        # Hold at max_lr
        return max_lr
    else:
        # Cosine decay from max_lr to end_lr
        decay_steps = total_steps - hold_end
        progress = (step - hold_end) / max(decay_steps, 1)
        return end_lr + 0.5 * (max_lr - end_lr) * (1 + math.cos(math.pi * progress))
