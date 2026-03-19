---
title: "Experiment Results — cpt20b"
date: "2026-03-19 14:26"
geometry: landscape,margin=1.5cm
fontsize: 10pt
---

# Results: cpt20b

Test performance at best validation epoch. Seeds per cell: n=5.
Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).

| Dataset | Metric | lstm | ctrnn | node | ctgru | ltc | srnn | srnn-per-neuron | srnn-echo | srnn-no-adapt | srnn-sfa-only | srnn-std-only | srnn-E-only | srnn-e-only-echo | srnn-e-only-per-neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| har | accuracy | 96.03% | 93.38% | 88.91% | **96.80%** | 94.24% | 83.48% | 82.71% | 70.54% | 93.14% | 92.44% | 91.76% | 91.94% | 89.19% | 91.33% |
| gesture | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| occupancy | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| smnist | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| traffic | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| power | squared error | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| ozone_fixed | F1-score | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| person | accuracy | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cheetah | MSE | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Table: cpt20b — 5 seed(s) per cell.*

