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

### Phase 4: Experiment Integration (✅ complete)
- **`har.py`** — Fully refactored as template (see above)
- **All 8 remaining experiments** ported with same pattern: gesture, occupancy, smnist, traffic, power, ozone_fixed, person, cheetah
  - Each has: I/O masks, W_in_mask on all constructors, single-timestep readout with W_out_mask, solver/h CLI args
  - Regression experiments (traffic, power, cheetah) use `tf.squeeze` after Dense(1) for scalar output

### Phase 4b: Lyapunov + Batch Wrapping (✅ complete)
- **`training_utils.py`** — `wrap_train_batch()`, `wrap_eval_batch()`, `setup_lyapunov_ops()`, `run_lyapunov_if_due()`
- All 9 experiments: `fit()` now applies palindrome looping, time-stretch, random windowing via shared wrappers
- Lyapunov LLE computed at every 10-epoch checkpoint, saved as HDF5 to `lyapunov/<experiment>/`
- ⚠️ **Trainable IC deferred** — requires restructuring Model `__init__` to build graph with `initial_state`

---

## What's Remaining

### Trainable IC (deferred)
- Requires restructuring all Model constructors: burn-in must run before dynamic_rnn graph is built

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
