---
title: "Experiment Results — lr001"
date: "2026-03-19 09:32"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: lr001

Test performance at best validation epoch. Seeds per cell: n=1.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

| Dataset | Metric | ltc | srnn | srnn-per-neuron |
|---|---|---:|---:|---:|
| har | accuracy | 96.00% | 96.85% | **97.55%** |
| gesture | accuracy | 53.89% | **57.36%** | 55.31% |
| occupancy | accuracy | 91.32% | 93.44% | **94.11%** |
| smnist | accuracy | 72.81% | 94.72% | **96.68%** |
| traffic | squared error | **0.0633** | 0.101 | 0.0790 |
| power | squared error | **0.000633** | 0.000672 | 0.000665 |
| ozone_fixed | F1-score | **0.255** | 0.205 | 0.211 |
| person | accuracy | **73.66%** | 67.24% | 70.52% |
| cheetah | MSE | **2.30** | 2.94 | 2.58 |

*Table: lr001 — 1 seed(s) per cell.*

