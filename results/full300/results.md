---
title: "Experiment Results — full300"
date: "2026-03-20 02:57 – 2026-03-22 14:47 UTC"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: full300

Test performance at best validation epoch. Seeds per cell: n=5.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

- **Started:** 2026-03-20 02:57 UTC
- **Completed:** 2026-03-22 14:47 UTC
- **Wall-clock elapsed:** 2d 11h 50m
- **Total CPU-hours:** 1636h (256 cells)

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | accuracy | 96.43% ± 1.24 | 96.31% ± 0.43 | **96.72% ± 0.36** | 95.46% ± 0.74 | 96.07% ± 0.65 | 96.58% ± 0.29 | 96.25% ± 0.45 | 94.86% ± 0.53 | 96.30% ± 0.23 | 96.48% ± 0.38 | 96.13% ± 0.64 | 96.35% ± 0.40 | 96.37% ± 0.28 | 93.69% ± 0.81 | 96.63% ± 0.44 |
| gesture | accuracy | 61.37% ± 2.07 | 61.27% ± 1.96 | 48.60% ± 3.33 | **71.40% ± 1.34** | 62.88% ± 3.36 | 57.43% ± 1.41 | 57.64% ± 1.05 | 56.35% ± 0.80 | 57.71% ± 1.89 | 58.12% ± 1.81 | 58.70% ± 1.36 | 57.25% ± 2.32 | 58.44% ± 1.98 | 56.10% ± 0.57 | 57.44% ± 0.49 |
| occupancy | accuracy | 93.85% ± 1.11 | 90.75% ± 2.75 | 92.61% ± 2.28 | 91.98% ± 3.12 | 93.68% ± 1.06 | 94.29% ± 0.82 | 93.72% ± 1.09 | **94.82% ± 2.34** | 93.04% ± 2.24 | 94.00% ± 2.02 | 94.57% ± 0.68 | 93.80% ± 1.40 | 93.26% ± 2.59 | 92.09% ± 1.89 | 93.74% ± 1.35 |
| smnist | accuracy | **98.50% ± 0.18** | 96.07% ± 0.36 | 97.78% ± 0.12 | 97.74% ± 0.24 | 95.54% ± 1.12 | 96.72% ± 0.24 | 97.58% (n=1) | — | — | — | — | — | — | — | — |
| traffic | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | F1-score | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | MSE | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Table: full300 — 5 seed(s) per cell.*

## Median Wall-Clock Duration

| Dataset | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | 28m | 32m | 54m | 34m | 1.3h | 1.2h | 1.0h | 59m | 36m | 34m | 49m | 42m | 55m | 54m | 48m |
| gesture | 9m | 9m | 14m | 12m | 15m | 36m | 32m | 27m | 13m | 11m | 23m | 15m | 19m | 17m | 23m |
| occupancy | 38m | 36m | 48m | 36m | 1.1h | 1.3h | 1.2h | 1.2h | 44m | 45m | 53m | 46m | 58m | 56m | 57m |
| smnist | 44.6h | 42.8h | 46.6h | 46.7h | 49.9h | 52.9h | 53.8h | — | — | — | — | — | — | — | — |
| traffic | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Durations are median across seeds.*

## Median CPU-Hours

| Dataset | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | 0.47 | 0.53 | 0.91 | 0.57 | 1.33 | 1.16 | 1.04 | 0.98 | 0.60 | 0.57 | 0.81 | 0.70 | 0.91 | 0.90 | 0.80 |
| gesture | 0.15 | 0.16 | 0.24 | 0.19 | 0.25 | 0.61 | 0.54 | 0.45 | 0.22 | 0.19 | 0.38 | 0.25 | 0.32 | 0.29 | 0.38 |
| occupancy | 0.64 | 0.60 | 0.80 | 0.59 | 1.07 | 1.31 | 1.21 | 1.15 | 0.73 | 0.76 | 0.89 | 0.77 | 0.97 | 0.93 | 0.95 |
| smnist | 44.6 | 42.8 | 46.6 | 46.7 | 49.9 | 52.9 | 53.8 | — | — | — | — | — | — | — | — |
| traffic | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*CPU-hours are median across seeds (1 vCPU per cell).*

