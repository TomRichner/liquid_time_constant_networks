# Learning Rate Schedule

## Original (Hasani et al. 2021)

In the original paper and codebase, a **fixed learning rate** was used throughout training:

- **Most models** (LSTM, CTRNN, NODE, CTGRU): `lr = 0.001` (Adam optimizer)
- **LTC**: `lr = 0.01` — hardcoded override because the LTC architecture needed a higher LR
- No warmup, no decay, no schedule

This meant the LTC model was trained at 10× the learning rate of all other models, making direct comparisons less clean.

## Our Runs

### `full200` (first full run)
Used `--ltc_lr_for_srnn` flag: both LTC and SRNN variants ran at `lr=0.01`, all others at `lr=0.001`.

### `lr001` (uniform LR run)
Used `--same_lr_ltc` flag: all models (including LTC) ran at `lr=0.001`. This was the first run with a uniform LR.

### Current (warmup-hold-cosine schedule)

All models now use the same schedule defined in `lr_schedule.py`. The `--same_lr_ltc` and `--ltc_lr_for_srnn` flags have been removed.

```
LR
 5e-4 ────────────────────────────────┐
      /                                \
     /          hold (50%)              \ cosine decay (30%)
    / warmup                             \
   /  (20%)                               \_____ 2.5e-5
1e-8                                        
   0%          20%          70%         100%  → total batches
```

**Parameters:**
- `start_lr = 1e-8` — near-zero initial LR
- `max_lr = 5e-4` — peak LR after warmup
- `end_lr = max_lr / 20 = 2.5e-5` — final LR after cosine decay

**Phases (by fraction of total training batches):**
1. **Warmup (0–20%):** Linear increase from `1e-8` to `5e-4`
2. **Hold (20–70%):** Constant at `5e-4`
3. **Cosine decay (70–100%):** Smooth decay from `5e-4` to `2.5e-5`

**Implementation:** LR is a `tf.Variable` updated per batch via `warmup_hold_cosine_lr(step, total_steps)`. Total steps = `epochs × batches_per_epoch`, so the schedule adapts to each experiment's data size and batch size.

**Why this schedule:**
- **Warmup** prevents large initial gradient updates from destabilizing randomly initialized weights
- **Hold** gives the optimizer time at peak LR to find good basins
- **Cosine decay** smoothly reduces LR for fine-grained convergence, avoiding the sharp drops of step-based schedules
