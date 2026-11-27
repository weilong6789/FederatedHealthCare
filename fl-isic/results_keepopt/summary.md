## Final test scores (mean ± std over seeds 0, 1, 2)

`local` is each hospital's own model scored on its own test set; it has no `all` row. `all` is the 4650 test images together.

Balanced accuracy:

| hospital | test n | pooled | fedavg T=1 | fedavg T=5 | fedavg T=20 |
|---|---|---|---|---|---|
| 0 | 2483 | 0.753 ± 0.009 | 0.745 ± 0.040 | 0.760 ± 0.037 | 0.756 ± 0.013 |
| 1 | 791 | 0.689 ± 0.031 | 0.702 ± 0.064 | 0.679 ± 0.072 | 0.690 ± 0.045 |
| 2 | 672 | 0.700 ± 0.100 | 0.631 ± 0.031 | 0.694 ± 0.077 | 0.740 ± 0.097 |
| 3 | 452 | 0.642 ± 0.029 | 0.532 ± 0.020 | 0.545 ± 0.011 | 0.510 ± 0.051 |
| 4 | 164 | 0.653 ± 0.033 | 0.424 ± 0.019 | 0.441 ± 0.018 | 0.444 ± 0.031 |
| 5 | 88 | 0.680 ± 0.052 | 0.746 ± 0.019 | 0.718 ± 0.061 | 0.686 ± 0.076 |
| all | 4650 | 0.748 ± 0.006 | 0.693 ± 0.023 | 0.704 ± 0.022 | 0.702 ± 0.013 |

Accuracy:

| hospital | test n | pooled | fedavg T=1 | fedavg T=5 | fedavg T=20 |
|---|---|---|---|---|---|
| 0 | 2483 | 0.744 ± 0.023 | 0.765 ± 0.031 | 0.768 ± 0.029 | 0.767 ± 0.035 |
| 1 | 791 | 0.954 ± 0.016 | 0.921 ± 0.027 | 0.920 ± 0.045 | 0.934 ± 0.027 |
| 2 | 672 | 0.765 ± 0.037 | 0.763 ± 0.003 | 0.775 ± 0.020 | 0.771 ± 0.005 |
| 3 | 452 | 0.634 ± 0.008 | 0.591 ± 0.021 | 0.621 ± 0.017 | 0.584 ± 0.018 |
| 4 | 164 | 0.671 ± 0.018 | 0.474 ± 0.021 | 0.488 ± 0.016 | 0.492 ± 0.021 |
| 5 | 88 | 0.837 ± 0.017 | 0.761 ± 0.052 | 0.723 ± 0.097 | 0.723 ± 0.135 |
| all | 4650 | 0.771 ± 0.020 | 0.764 ± 0.020 | 0.770 ± 0.028 | 0.768 ± 0.024 |

AUC:

| hospital | test n | pooled | fedavg T=1 | fedavg T=5 | fedavg T=20 |
|---|---|---|---|---|---|
| 0 | 2483 | 0.957 ± 0.003 | 0.963 ± 0.008 | 0.964 ± 0.009 | 0.964 ± 0.010 |
| 1 | 791 | 0.965 ± 0.015 | 0.935 ± 0.029 | 0.947 ± 0.027 | 0.931 ± 0.003 |
| 2 | 672 | 0.958 ± 0.011 | 0.952 ± 0.006 | 0.959 ± 0.009 | 0.962 ± 0.002 |
| 3 | 452 | 0.920 ± 0.006 | 0.893 ± 0.018 | 0.901 ± 0.013 | 0.902 ± 0.005 |
| 4 | 164 | 0.851 ± 0.002 | 0.764 ± 0.006 | 0.777 ± 0.020 | 0.762 ± 0.012 |
| 5 | 88 | 0.907 ± 0.014 | 0.870 ± 0.048 | 0.887 ± 0.013 | 0.888 ± 0.042 |
| all | 4650 | 0.962 ± 0.003 | 0.946 ± 0.008 | 0.950 ± 0.009 | 0.951 ± 0.007 |

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
| fedavg T=1 | balanced accuracy | -0.055 [-0.070, -0.040]* |
| fedavg T=1 | accuracy | -0.007 [-0.016, +0.002] |
| fedavg T=1 | AUC | -0.015 [-0.019, -0.011]* |
| fedavg T=5 | balanced accuracy | -0.044 [-0.060, -0.028]* |
| fedavg T=5 | accuracy | -0.001 [-0.011, +0.007] |
| fedavg T=5 | AUC | -0.012 [-0.015, -0.009]* |
| fedavg T=20 | balanced accuracy | -0.047 [-0.063, -0.031]* |
| fedavg T=20 | accuracy | -0.003 [-0.012, +0.006] |
| fedavg T=20 | AUC | -0.011 [-0.014, -0.007]* |

## Still improving at epoch 20? (all test images)

Final score, and the gain per 10 epochs from a straight-line fit to each seed's scores over epochs 10-20 (mean over seeds; `n/3 up` counts seeds with a positive slope; `-` when a run has fewer than 3 evaluations in that window, e.g. FedAvg with very few rounds).

| run | metric | final | gain per 10 epochs | seeds rising |
|---|---|---|---|---|
| pooled | balanced accuracy | 0.748 ± 0.006 | +0.025 | 2/3 up |
| pooled | accuracy | 0.771 ± 0.020 | +0.040 | 3/3 up |
| pooled | AUC | 0.962 ± 0.003 | +0.009 | 3/3 up |
| fedavg T=1 | balanced accuracy | 0.693 ± 0.023 | +0.007 | 2/3 up |
| fedavg T=1 | accuracy | 0.764 ± 0.020 | +0.016 | 2/3 up |
| fedavg T=1 | AUC | 0.946 ± 0.008 | +0.006 | 3/3 up |
| fedavg T=5 | balanced accuracy | 0.704 ± 0.022 | -0.005 | 1/3 up |
| fedavg T=5 | accuracy | 0.770 ± 0.028 | +0.010 | 3/3 up |
| fedavg T=5 | AUC | 0.950 ± 0.009 | +0.001 | 2/3 up |
| fedavg T=20 | balanced accuracy | 0.702 ± 0.013 | -0.001 | 2/3 up |
| fedavg T=20 | accuracy | 0.768 ± 0.024 | +0.017 | 3/3 up |
| fedavg T=20 | AUC | 0.951 ± 0.007 | +0.007 | 3/3 up |

## FedAvg local steps per round (T), all test images

Final score, mean ± std over seeds. Rounds = steps / T, so T = 1 averages after every step and the largest T averages once at the end. T values that don't divide the budget run a few extra steps (see `steps`).

| run | rounds | steps | balanced accuracy | accuracy | AUC |
|---|---|---|---|---|---|
| pooled | - | 5820 | 0.748 ± 0.006 | 0.771 ± 0.020 | 0.962 ± 0.003 |
| fedavg T=1 | 5820 | 5820 | 0.693 ± 0.023 | 0.764 ± 0.020 | 0.946 ± 0.008 |
| fedavg T=5 | 1164 | 5820 | 0.704 ± 0.022 | 0.770 ± 0.028 | 0.950 ± 0.009 |
| fedavg T=20 | 291 | 5820 | 0.702 ± 0.013 | 0.768 ± 0.024 | 0.951 ± 0.007 |

## Run settings

| run | seeds | optimizer | lr | epochs | longest run (min) |
|---|---|---|---|---|---|
| pooled | 0, 1, 2 | adam | 0.0005 | 20.0 | 5 |
| fedavg T=1 | 0, 1, 2 | adam-keep | 0.0005 | 20.0 | 59 |
| fedavg T=5 | 0, 1, 2 | adam-keep | 0.0005 | 20.0 | 71 |
| fedavg T=20 | 0, 1, 2 | adam-keep | 0.0005 | 20.0 | 85 |
