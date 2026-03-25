# SRNN Training Pipeline Refactor — Status & Notes

**Started:** 2025-03-25  
**Branch:** (current working branch)

## What's Done

### Phase 1: Infrastructure Modules (✅ complete)
All in `experiments_with_ltcs/`:
- **`io_masks.py`** — Neuron partition into 3 roles: 1/4 input, 1/2 interneuron, 1/4 output. Non-overlapping. Seeded per-experiment for fair cross-model comparison. Forces signal through at least one recurrent connection.
- **`trainable_ic.py`** — 30-second burn-in with zero input → stores settled state as `tf.Variable(trainable=True)`. Works with any RNNCell. ⚠️ Not yet integrated into experiment scripts.
- **`sequence_looping.py`** — Palindrome forward/backward looping, random start/end windowing for constant-length sequences, random readout index in last loop, BPTT boundary computation.
- **`time_stretch.py`** — PCHIP time-stretch augmentation, log-uniform sampling in [0.25, 4.0]. Nearest-neighbor for classification labels, PCHIP for regression targets.

### Phase 2: Model Changes (✅ complete)
- **`srnn_model.py`** — Added `rk4` and `exponential` solvers alongside existing `semi_implicit` and `explicit`. Added `_compute_rhs_flat()` helper. Added `W_in_mask` parameter (zeros rows for non-input neurons).
- **`ctrnn_model.py`** — Added `W_in_mask` to CTRNN, NODE, and CTGRU. Applied after input Dense projection.
- **`ltc_model.py`** — Added `W_in_mask` to LTCCell. Applied to sensory weight activations.

### Phase 3: Lyapunov Module (✅ complete)
- **`lyapunov.py`** — Benettin's algorithm ported from Julia. Numpy-based. HDF5 save with float16 + gzip(level=1) + shuffle compression. ⚠️ Not yet integrated into experiment scripts.

### Phase 4: Experiment Integration (🔶 partial)
- **`har.py`** — Fully refactored as template:
  - I/O masks (W_in_mask passed to all model constructors, W_out_mask applied before Dense readout)
  - Single-timestep readout via `head[readout_idx] * W_out_mask → Dense(6)`
  - `iterate_train()` now yields `(batch_x, batch_y, readout_idx, bptt_start_idx)` with time-stretch + palindrome looping + random windowing
  - `iterate_eval()` with palindrome loops, readout at end
  - New CLI args: `--solver`, `--h`, `--min_loops`, `--min_loop_len`, `--stretch_lo`, `--stretch_hi`
  - `target_y` placeholder changed from `[None,None]` to `[None]` (single timestep)

---

## What's Remaining

### Phase 4 continued: Port to remaining 8 experiments
Same pattern as har.py. Key differences per experiment:
| Experiment | Task | Input dim | Classes/Targets | Notes |
|---|---|---|---|---|
| gesture | classification | 32 | 5 | Per-timestep labels |
| occupancy | classification | 5 | 2 | Per-timestep labels |
| smnist | classification | 28 | 10 | **Single label** per sequence — same label at any readout |
| traffic | regression (MAE) | 4 | 1 | Per-timestep targets (float) |
| power | regression (MAE) | 7 | 1 | Per-timestep targets (float) |
| ozone_fixed | classification | 72 | 2 | Per-timestep labels |
| person | classification | 24 | 7 | Per-timestep labels, per-person time series |
| cheetah | regression (MSE/MAE) | 17 | 17 | Vector autoregression |

### Phase 4b: Integrate trainable IC + Lyapunov calls
- Add `create_trainable_ic()` call after session init, pass IC to `dynamic_rnn`
- Add `compute_lyapunov_at_checkpoint()` call at checkpoints in `fit()`
- BPTT truncation via `tf.stop_gradient` at `bptt_start_idx` — needs TF graph-level work

### Phase 5: VM Image Rebuild
- Add `h5py` and `scipy` to pip install in `cloud/build_image.sh`
- Rebuild `srnn-python` VM image

### Phase 6: Smoke Test
- 3 epochs, har, seed 1
- 4 models: `ltc`, `ctrnn`, `srnn-no-adapt-no-dales`, `srnn-E-only`
- Verify: training_log, checkpoints, palindrome looping behavior

---

## Design Decisions

1. **Readout:** Dense on full state at one timestep only (random during training, final during eval). No per-timestep readout.
2. **I/O masks:** 1/4 input, 1/2 interneuron, 1/4 output. Non-overlapping. Seeded.
3. **Palindrome:** fwd→bwd→fwd→bwd... with configurable loop count. Random start/end trimming for constant length.
4. **BPTT:** Truncated to last 2 full loops. Earlier loops provide warm-up only.
5. **Time stretch:** PCHIP resampling, fixed ODE step h. Applied before palindrome looping.
6. **Solvers:** semi_implicit (default), explicit, rk4, exponential Euler.
