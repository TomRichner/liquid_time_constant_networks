# SRNN Extension

Port of the Stable Recurrent Neural Network (SRNN) from Julia (`Intersect-LNNs-SRNNs/JuliaLang/src/models/srnn.jl`) to a TF1 `RNNCell` for direct comparison with LTC and other models on all Hasani 2021 benchmark tasks.

## Files

- **`experiments_with_ltcs/srnn_model.py`** — `SRNNCell(tf.nn.rnn_cell.RNNCell)` implementing:
  - Piecewise sigmoid activation (port of `activations.jl`)
  - E/I neuron split with Dale's law
  - Spike-Frequency Adaptation (SFA) with multi-timescale adaptation variables
  - Short-Term Depression (STD)
  - Semi-implicit (fused) solver (unconditionally stable)
  - Explicit Euler solver option
  - Readout modes: synaptic (default), rate, dendritic

## Model Variants

| `--model` flag | Description | State size (n=32) |
|---|---|---|
| `hopf` | Vanilla rate RNN — all excitatory, no SFA/STD. Essentially a Hopfield-style network with learnable τ_d. | 32 |
| `srnn` | Full SRNN — 75% E / 25% I, 3 SFA timescales, STD on both populations. | 160 |

## Usage

```bash
cd experiments_with_ltcs

# Vanilla Hopfield baseline
uv run python3 har.py --model hopf --epochs 200 --log 1 --size 32

# Full SRNN with SFA + STD + E/I
uv run python3 har.py --model srnn --epochs 200 --log 1 --size 32
```

Both `hopf` and `srnn` are available in all 9 experiment scripts: `har`, `gesture`, `occupancy`, `smnist`, `traffic`, `power`, `ozone`, `cheetah`, `person`.

## Smoke Test Results (2 epochs, size=32)

| Experiment | Model | Test Metric | Status |
|---|---|---|---|
| HAR | srnn | 82% accuracy | ✅ |
| HAR | srnn (10 ep) | 96.5% accuracy | ✅ |
| HAR | hopf | 89% accuracy | ✅ |
| Occupancy | srnn | 95.2% accuracy | ✅ |
| Traffic | srnn | 0.51 MAE | ✅ |
| Gesture | srnn | 42.7% accuracy | ✅ |
| Power | srnn | — | untested |
| Ozone | srnn | — | untested |
| SMnist | srnn | — | untested |
| Cheetah | srnn | — | untested |
| Person | srnn | — | untested |

## Key Differences from LTC

| | LTC | SRNN |
|---|---|---|
| **ODE form** | `dx/dt = -(1/τ + f)·x + f·A` (liquid τ) | `dx/dt = (-x + W·b·r + u) / τ_d` (fixed τ per neuron) |
| **Nonlinearity** | Gaussian-gated synapses: `σ((x-μ)·σ) · W · erev` | Piecewise sigmoid on dendritic voltage |
| **Synaptic model** | Reversal potentials (erev) | SFA + STD (biologically motivated) |
| **Neuron types** | Homogeneous | Excitatory / Inhibitory split |
| **Speed** | ~seconds/epoch (HAR) | ~seconds/epoch (HAR), comparable |
| **Constraint op** | Clipping (W, gleak, cm) | None needed (softplus parameterization) |

## Relationship to Julia Implementation

The Python `SRNNCell` is a faithful port of `srnn.jl`'s `SRNNCell` struct. Both use:
- Identical state layout: `[a_E, a_I, b_E, b_I, x]`
- Same semi-implicit solver equations
- Same `softplus` parameterization for time constants
- Same `_make_tau_range` for log-spaced SFA timescales
- Same readout modes
