---
title: "Experiment Results — run25ep"
date: "2026-03-17 13:33"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: run25ep

Test performance at best validation epoch. Seeds per cell: n=1.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn |
|---|---|---:|---:|---:|---:|---:|---:|
| har | accuracy | **97.33%** | 96.22% | 96.22% | 96.36% | 96.22% | 95.40% |
| gesture | accuracy | **60.73%** | 53.96% | 44.13% | 59.97% | 53.96% | 56.84% |
| occupancy | accuracy | 94.21% | 90.55% | 93.61% | 92.43% | 93.92% | **95.57%** |
| smnist | accuracy | **98.21%** | 95.99% | 96.66% | 97.81% | 92.44% | 93.73% |
| traffic | squared error | 0.0997 | 0.113 | 0.199 | 0.102 | **0.0618** | 0.148 |
| power | squared error | 0.000669 | 0.000782 | 0.00124 | **0.000640** | 0.000746 | 0.000902 |
| ozone | F1-score | 0.157 | 0.121 | **0.167** | 0.120 | 0.124 | 0.157 |
| person | accuracy | 81.04% | 76.37% | 70.58% | 82.94% | **83.07%** | 64.79% |
| cheetah | MSE | 3.73 | 4.12 | 5.56 | 3.33 | **2.22** | 2.58 |

*Table: run25ep — 1 seed(s) per cell.*

