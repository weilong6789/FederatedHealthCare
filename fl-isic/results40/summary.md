## Final test scores (mean ± std over seeds 0, 1, 2)

`local` is each hospital's own model scored on its own test set; it has no `all` row. `all` is the 4650 test images together.

Balanced accuracy:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.754 ± nan | 0.771 ± 0.032 | 0.759 ± 0.020 |
| 1 | 791 | 0.645 ± nan | 0.751 ± 0.063 | 0.615 ± 0.002 |
| 2 | 672 | 0.705 ± nan | 0.779 ± 0.105 | 0.672 ± 0.067 |
| 3 | 452 | 0.637 ± nan | 0.663 ± 0.052 | 0.451 ± 0.060 |
| 4 | 164 | 0.693 ± nan | 0.690 ± 0.031 | 0.434 ± 0.012 |
| 5 | 88 | 0.417 ± nan | 0.698 ± 0.016 | 0.753 ± 0.016 |
| all | 4650 | - | 0.767 ± 0.031 | 0.686 ± 0.006 |

Accuracy:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.766 ± nan | 0.763 ± 0.029 | 0.809 ± 0.018 |
| 1 | 791 | 0.850 ± nan | 0.949 ± 0.019 | 0.957 ± 0.014 |
| 2 | 672 | 0.814 ± nan | 0.765 ± 0.019 | 0.790 ± 0.015 |
| 3 | 452 | 0.622 ± nan | 0.645 ± 0.033 | 0.594 ± 0.029 |
| 4 | 164 | 0.713 ± nan | 0.699 ± 0.025 | 0.486 ± 0.013 |
| 5 | 88 | 0.875 ± nan | 0.860 ± 0.007 | 0.841 ± 0.039 |
| all | 4650 | - | 0.783 ± 0.023 | 0.800 ± 0.003 |

AUC:

| hospital | test n | local | pooled | fedavg T=100 |
|---|---|---|---|---|
| 0 | 2483 | 0.962 ± nan | 0.961 ± 0.008 | 0.967 ± 0.003 |
| 1 | 791 | 0.918 ± nan | 0.965 ± 0.015 | 0.939 ± 0.020 |
| 2 | 672 | 0.953 ± nan | 0.961 ± 0.004 | 0.967 ± 0.007 |
| 3 | 452 | 0.913 ± nan | 0.925 ± 0.016 | 0.897 ± 0.018 |
| 4 | 164 | 0.879 ± nan | 0.852 ± 0.015 | 0.755 ± 0.022 |
| 5 | 88 | 0.629 ± nan | 0.886 ± 0.013 | 0.896 ± 0.018 |
| all | 4650 | - | 0.963 ± 0.007 | 0.956 ± 0.005 |

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
| fedavg T=100 | balanced accuracy | -0.081 [-0.099, -0.063]* |
| fedavg T=100 | accuracy | +0.016 [+0.007, +0.026]* |
| fedavg T=100 | AUC | -0.008 [-0.012, -0.003]* |

## Still improving at epoch 40? (all test images)

Final score, and the gain per 10 epochs from a straight-line fit to each seed's scores over epochs 20-40 (mean over seeds; `n/3 up` counts seeds with a positive slope).

| run | metric | final | gain per 10 epochs | seeds rising |
|---|---|---|---|---|
| pooled | balanced accuracy | 0.767 ± 0.031 | +0.007 | 3/3 up |
| pooled | accuracy | 0.783 ± 0.023 | +0.016 | 3/3 up |
| pooled | AUC | 0.963 ± 0.007 | +0.003 | 3/3 up |
| fedavg T=100 | balanced accuracy | 0.686 ± 0.006 | -0.002 | 1/3 up |
| fedavg T=100 | accuracy | 0.800 ± 0.003 | +0.006 | 3/3 up |
| fedavg T=100 | AUC | 0.956 ± 0.005 | +0.001 | 2/3 up |

## Run settings

| run | seeds | optimizer | lr | epochs | longest run (min) |
|---|---|---|---|---|---|
| local | 2 | adam | 0.0005 | 40.0 | 11 |
| pooled | 0, 1, 2 | adam | 0.0005 | 40.0 | 11 |
| fedavg T=100 | 0, 1, 2 | adam | 0.0005 | 40.2 | 168 |
