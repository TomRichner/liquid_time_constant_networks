# Models

## Overview

All models are 32 neurons (default `--size 32`), trained with Adam optimizer and the warmup-hold-cosine LR schedule.

| Model | Key | Source | Description |
|-------|-----|--------|-------------|
| LSTM | `lstm` | Hasani 2021 | Standard LSTM baseline |
| LTC | `ltc` | Hasani 2021 | Liquid Time-Constant (semi-implicit solver) |
| CTRNN | `ctrnn` | Hasani 2021 | Continuous-Time RNN |
| CT-GRU | `ctgru` | Hasani 2021 | Continuous-Time GRU |
| Neural ODE | `node` | Hasani 2021 | Neural ODE |
| SRNN | `srnn` | Ours | Full SRNN (Dale's law, SFA+STD on E and I) |
| SRNN Per-Neuron | `srnn-per-neuron` | Ours | SRNN with per-neuron dynamics params |
| SRNN Echo | `srnn-echo` | Ours | SRNN reservoir: only W_in + Dense are trained |
| SRNN No-Adapt | `srnn-no-adapt` | Ours | SRNN with no adaptation (ablation) |
| SRNN SFA-Only | `srnn-sfa-only` | Ours | SRNN with SFA on E neurons only (ablation) |
| SRNN STD-Only | `srnn-std-only` | Ours | SRNN with STD on E neurons only (ablation) |
| SRNN E-Only | `srnn-E-only` | Ours | SRNN with both SFA+STD on E neurons only (ablation) |
| SRNN E-Only Echo | `srnn-e-only-echo` | Ours | E-only reservoir: only W_in + Dense trained |
| SRNN E-Only Per-Neuron | `srnn-e-only-per-neuron` | Ours | E-only with per-neuron dynamics params |

Previously available (removed): `ltc_rk` (Runge-Kutta LTC), `ltc_ex` (Explicit LTC).

## Baseline Models (from Hasani et al. 2021)

These are the reference models from the original LTC paper. They use standard TF1 RNN cells.

- **LSTM**: Standard long short-term memory. No continuous-time dynamics.
- **LTC**: Liquid Time-Constant networks with semi-implicit ODE solver. Has a `constrain_op` that clips parameters.
- **CTRNN**: Continuous-Time RNN with global feedback and no cell clipping.
- **CT-GRU**: Continuous-Time GRU variant with no cell clipping.
- **NODE**: Neural ODE with cell clipping at 10.

## SRNN (Structured Recurrent Neural Network)

All SRNN variants use `SRNNCell` from `srnn_model.py` with `dales=True` and 50% E / 50% I neurons (`n_E = model_size // 2`).

### Core SRNN architecture

- **Dale's Law**: Excitatory neurons can only have positive outgoing weights; inhibitory neurons can only have negative outgoing weights. Enforced via `softplus` (E) and `-softplus` (I) transformations on the raw weight matrix `W`.
- **Recurrent weights**: Initialized via RMT (Random Matrix Theory) method (`generate_rmt_matrix`), which controls spectral radius.
- **Input weights** (`W_in`): Initialized as `N(0, 0.1)`.
- **SFA** (Spike-Frequency Adaptation): Subtractive adaptation with `n_a` timescale variables per population. Timescales are log-uniform between `tau_lo` and `tau_hi`.
- **STD** (Short-Term Depression): Multiplicative synaptic depression with recovery (`tau_b_rec`) and release (`tau_b_rel`) timescale.

### SRNN Variants

#### `srnn` — Full model
The complete SRNN with all adaptation mechanisms on both populations:
- `n_a_E=3, n_a_I=3`: 3 SFA timescales for E and I
- `n_b_E=1, n_b_I=1`: 1 STD variable for E and I

#### `srnn-per-neuron` — Per-neuron dynamics
Same as `srnn` but with `per_neuron=True`: each neuron gets its own dynamics parameters instead of sharing within E/I populations.

#### `srnn-echo` — Echo state / reservoir
Same cell as `srnn`, but only `W_in` (input weights) and the output Dense layer are trained. All internal dynamics parameters and `W` are frozen at initialization. Tests whether the random SRNN dynamics are useful as a fixed feature extractor.

#### `srnn-e-only-echo` — E-only echo state
Same as `srnn-e-only-echo` uses `srnn-E-only` cell (SFA+STD on E only), but only `W_in` and Dense are trained. Tests whether the E-only random dynamics work as a fixed feature extractor.

#### `srnn-e-only-per-neuron` — E-only per-neuron dynamics
Same as `srnn-E-only` but with `per_neuron=True`: each E neuron gets its own SFA/STD parameters. I neurons have no adaptation.

### Ablation Models

These test the contribution of individual adaptation mechanisms. All use Dale's law, 50% E/I, but vary which adaptation is enabled:

| Variant | SFA (E) | SFA (I) | STD (E) | STD (I) | Purpose |
|---------|---------|---------|---------|---------|---------|
| `srnn` | 3 | 3 | 1 | 1 | Full model (reference) |
| `srnn-E-only` | 3 | 0 | 1 | 0 | Both adapt, E only |
| `srnn-e-only-per-neuron` | 3 | 0 | 1 | 0 | E-only, per-neuron params |
| `srnn-sfa-only` | 3 | 0 | 0 | 0 | SFA alone |
| `srnn-std-only` | 0 | 0 | 1 | 0 | STD alone |
| `srnn-no-adapt` | 0 | 0 | 0 | 0 | No adaptation (just Dale's E/I) |
| `srnn-e-only-echo` | 3 | 0 | 1 | 0 | E-only reservoir (frozen) |

The ablation ladder answers: Does adaptation help? Is SFA or STD more important? Does I-neuron adaptation matter?

### Trainable Parameters by Variant

| Parameter | `srnn` | `srnn-E-only` | `srnn-sfa-only` | `srnn-std-only` | `srnn-no-adapt` |
|-----------|--------|---------------|-----------------|-----------------|-----------------|
| `tau_d` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `a_0` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `W`, `W_in` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `tau_a_E_lo/hi` | ✓ | ✓ | ✓ | – | – |
| `c_E`, `c_0_E` | ✓ | ✓ | ✓ | – | – |
| `tau_a_I_lo/hi` | ✓ | – | – | – | – |
| `c_I`, `c_0_I` | ✓ | – | – | – | – |
| `tau_b_E_rec/rel` | ✓ | ✓ | – | ✓ | – |
| `tau_b_I_rec/rel` | ✓ | – | – | – | – |
