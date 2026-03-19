---
title: "Experiment Results — ckpt20"
date: "2026-03-19 13:19"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: ckpt20

Test performance at best validation epoch. Seeds per cell: n=5.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-sfa-only | srnn-std-only | srnn-E-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | accuracy | 96.03% | 93.38% | 88.91% | **96.80%** | 94.24% | 83.67% | 84.43% | 70.71% | 91.81% | 92.44% | 91.43% | 92.10% |
| gesture | accuracy | — | — | — | — | — | — | — | — | — | — | — | — |
| occupancy | accuracy | — | — | — | — | — | — | — | — | — | — | — | — |
| smnist | accuracy | — | — | — | — | — | — | — | — | — | — | — | — |
| traffic | squared error | — | — | — | — | — | — | — | — | — | — | — | — |
| power | squared error | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | F1-score | — | — | — | — | — | — | — | — | — | — | — | — |
| person | accuracy | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | MSE | — | — | — | — | — | — | — | — | — | — | — | — |

*Table: ckpt20 — 5 seed(s) per cell.*

