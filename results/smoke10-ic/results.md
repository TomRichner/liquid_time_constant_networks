---
title: "Experiment Results — smoke10-ic"
date: "2026-03-26 03:03 – 2026-03-26 05:50 UTC"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: smoke10-ic

Test performance at best validation epoch. Seeds per cell: n=1.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

- **Started:** 2026-03-26 03:03 UTC
- **Completed:** 2026-03-26 05:50 UTC
- **Wall-clock elapsed:** 2h 47m
- **Total CPU-hours:** 4h (4 cells)

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gesture | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| occupancy | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| smnist | accuracy | **84.27%** | — | — | — | 19.53% | — | — | — | — | — | — | — | — | — | 34.32% |
| traffic | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | F1-score | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | MSE | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Table: smoke10-ic — 1 seed(s) per cell.*

## Median Wall-Clock Duration

| Dataset | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gesture | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| occupancy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| smnist | 8m | — | — | — | 2.8h | — | — | — | — | — | — | — | — | — | 1.4h |
| traffic | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Durations are median across seeds.*

## Median CPU-Hours

| Dataset | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gesture | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| occupancy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| smnist | 0.14 | — | — | — | 2.77 | — | — | — | — | — | — | — | — | — | 1.44 |
| traffic | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*CPU-hours are median across seeds (1 vCPU per cell).*

