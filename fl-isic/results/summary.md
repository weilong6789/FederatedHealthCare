## Final test scores (mean ± std over seeds 0, 1, 2)

`local` is each hospital's own model scored on its own test set; it has no `all` row. `all` is the 4650 test images together.

Balanced accuracy:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.763 ± 0.023 | 0.753 ± 0.009 | 0.737 ± 0.003 |
| 1 | 791 | 0.719 ± 0.032 | 0.689 ± 0.031 | 0.645 ± 0.014 |
| 2 | 672 | 0.676 ± 0.016 | 0.700 ± 0.100 | 0.639 ± 0.002 |
| 3 | 452 | 0.675 ± 0.033 | 0.642 ± 0.029 | 0.473 ± 0.013 |
| 4 | 164 | 0.630 ± 0.067 | 0.653 ± 0.033 | 0.440 ± 0.036 |
| 5 | 88 | 0.303 ± 0.026 | 0.680 ± 0.052 | 0.767 ± 0.081 |
| all | 4650 | - | 0.748 ± 0.006 | 0.684 ± 0.006 |

Accuracy:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.737 ± 0.041 | 0.744 ± 0.023 | 0.776 ± 0.008 |
| 1 | 791 | 0.962 ± 0.026 | 0.954 ± 0.016 | 0.966 ± 0.003 |
| 2 | 672 | 0.767 ± 0.021 | 0.765 ± 0.037 | 0.781 ± 0.007 |
| 3 | 452 | 0.677 ± 0.002 | 0.634 ± 0.008 | 0.587 ± 0.007 |
| 4 | 164 | 0.654 ± 0.059 | 0.671 ± 0.018 | 0.488 ± 0.037 |
| 5 | 88 | 0.856 ± 0.007 | 0.837 ± 0.017 | 0.765 ± 0.063 |
| all | 4650 | - | 0.771 ± 0.020 | 0.780 ± 0.004 |

AUC:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.957 ± 0.008 | 0.957 ± 0.003 | 0.966 ± 0.001 |
| 1 | 791 | 0.950 ± 0.013 | 0.965 ± 0.015 | 0.920 ± 0.031 |
| 2 | 672 | 0.940 ± 0.010 | 0.958 ± 0.011 | 0.965 ± 0.001 |
| 3 | 452 | 0.934 ± 0.005 | 0.920 ± 0.006 | 0.892 ± 0.013 |
| 4 | 164 | 0.849 ± 0.011 | 0.851 ± 0.002 | 0.753 ± 0.010 |
| 5 | 88 | 0.779 ± 0.131 | 0.907 ± 0.014 | 0.897 ± 0.057 |
| all | 4650 | - | 0.962 ± 0.003 | 0.955 ± 0.003 |

## Test-set label mix per hospital (share of each class)

The hospitals see very different mixes of the 8 classes, which is what pulls FedAvg away from pooled here.

| hospital | class 0 | class 1 | class 2 | class 3 | class 4 | class 5 | class 6 | class 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.23 | 0.34 | 0.23 | 0.05 | 0.09 | 0.01 | 0.01 | 0.04 |
| 1 | 0.01 | 0.94 | 0.00 | 0.00 | 0.04 | 0.01 | 0.01 | 0.00 |
| 2 | 0.22 | 0.54 | 0.06 | 0.00 | 0.14 | 0.02 | 0.03 | 0.00 |
| 3 | 0.15 | 0.34 | 0.16 | 0.05 | 0.19 | 0.02 | 0.00 | 0.09 |
| 4 | 0.30 | 0.40 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 |
| 5 | 0.10 | 0.84 | 0.00 | 0.00 | 0.05 | 0.00 | 0.01 | 0.00 |

## FedAvg minus pooled on all test images (paired bootstrap)

Test images resampled with replacement within each hospital, 1000 times; both methods scored on the same resamples, averaged over seeds. `*` marks an interval that excludes 0.

| run | metric | FedAvg − pooled [95% CI] |
|---|---|---|
| fedavg T=100 | balanced accuracy | -0.064 [-0.081, -0.047]* |
| fedavg T=100 | accuracy | +0.009 [-0.001, +0.018] |
| fedavg T=100 | AUC | -0.007 [-0.011, -0.004]* |

## Run settings

| run | seeds | optimizer | lr | epochs | longest run (min) |
|---|---|---|---|---|---|
| local | 0, 1, 2 | adam | 0.0005 | 20.0 | 6 |
| pooled | 0, 1, 2 | adam | 0.0005 | 20.0 | 5 |
| fedavg T=100 | 0, 1, 2 | adam | 0.0005 | 20.3 | 84 |
