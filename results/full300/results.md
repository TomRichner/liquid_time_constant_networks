---
title: "Experiment Results — full300"
date: "2026-03-21 10:12"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: full300

Test performance at best validation epoch. Seeds per cell: n=5.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-no-adapt-no-dales | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | accuracy | 96.43% ± 1.24 | 96.31% ± 0.43 | **96.72% ± 0.36** | 95.46% ± 0.74 | 96.07% ± 0.65 | 96.58% ± 0.29 | 96.25% ± 0.45 | 94.86% ± 0.53 | 96.30% ± 0.23 | 96.48% ± 0.38 | 96.13% ± 0.64 | 96.35% ± 0.40 | 96.37% ± 0.28 | 93.69% ± 0.81 | 96.63% ± 0.44 |
| gesture | accuracy | 61.37% ± 2.07 | 61.27% ± 1.96 | 48.60% ± 3.33 | **71.40% ± 1.34** | 62.88% ± 3.36 | 57.43% ± 1.41 | 57.64% ± 1.05 | 56.35% ± 0.80 | 57.71% ± 1.89 | 58.12% ± 1.81 | 58.70% ± 1.36 | 57.25% ± 2.32 | 58.44% ± 1.98 | 56.10% ± 0.57 | 57.44% ± 0.49 |
| occupancy | accuracy | 93.85% ± 1.11 | 90.75% ± 2.75 | 92.61% ± 2.28 | 91.98% ± 3.12 | 93.68% ± 1.06 | 94.29% ± 0.82 | 93.72% ± 1.09 | **94.82% ± 2.34** | 93.04% ± 2.24 | 94.00% ± 2.02 | 94.57% ± 0.68 | 93.80% ± 1.40 | 93.26% ± 2.59 | 92.09% ± 1.89 | 93.74% ± 1.35 |
| smnist | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| traffic | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | F1-score | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | MSE | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Table: full300 — 5 seed(s) per cell.*

