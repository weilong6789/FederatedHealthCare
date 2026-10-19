## Per-hospital test accuracy (mean ± std over seeds)

| hospital       | test n | majority class | local         | pooled        | fedavg        |
|----------------|--------|----------------|---------------|---------------|---------------|
| 0 Cleveland    | 104    | 0.538          | 0.737 ± 0.005 | 0.771 ± 0.008 | 0.773 ± 0.009 |
| 1 Hungary      | 89     | 0.629          | 0.719 ± 0.000 | 0.764 ± 0.000 | 0.789 ± 0.017 |
| 2 Switzerland  | 16     | 0.938          | 0.938 ± 0.000 | 0.750 ± 0.000 | 0.662 ± 0.056 |
| 3 Long Beach V | 45     | 0.778          | 0.764 ± 0.012 | 0.653 ± 0.012 | 0.604 ± 0.029 |

`majority class` is the accuracy of always predicting that hospital's more
common label. A method at or below it has learned nothing useful there.

## Accuracy minus majority-class baseline

| hospital       | local  | pooled | fedavg |
|----------------|--------|--------|--------|
| 0 Cleveland    | +0.198 | +0.233 | +0.235 |
| 1 Hungary      | +0.090 | +0.135 | +0.160 |
| 2 Switzerland  | +0.000 | -0.188 | -0.275 |
| 3 Long Beach V | -0.013 | -0.124 | -0.173 |

## Per-hospital test balanced accuracy (mean ± std over seeds)

| hospital       | local         | pooled        | fedavg        |
|----------------|---------------|---------------|---------------|
| 0 Cleveland    | 0.733 ± 0.006 | 0.773 ± 0.009 | 0.770 ± 0.009 |
| 1 Hungary      | 0.715 ± 0.000 | 0.781 ± 0.000 | 0.794 ± 0.014 |
| 2 Switzerland  | 0.500 ± 0.000 | 0.867 ± 0.000 | 0.820 ± 0.030 |
| 3 Long Beach V | 0.634 ± 0.008 | 0.706 ± 0.008 | 0.674 ± 0.019 |

## Per-hospital test AUC (mean ± std over seeds)

| hospital       | local         | pooled        | fedavg        |
|----------------|---------------|---------------|---------------|
| 0 Cleveland    | 0.811 ± 0.003 | 0.858 ± 0.002 | 0.858 ± 0.008 |
| 1 Hungary      | 0.833 ± 0.005 | 0.876 ± 0.002 | 0.875 ± 0.004 |
| 2 Switzerland  | 0.427 ± 0.329 | 0.933 ± 0.000 | 0.920 ± 0.030 |
| 3 Long Beach V | 0.609 ± 0.004 | 0.759 ± 0.005 | 0.742 ± 0.009 |

Always predicting one label scores 0.500 on balanced accuracy and AUC at
every hospital, however skewed its test set, so 0.500 is the bar to beat.

## Summary

Accuracy:

| method | mean over hospitals | weighted by test n | worst hospital | mean lift over baseline | hospitals beaten |
|--------|---------------------|--------------------|----------------|-------------------------|------------------|
| local  | 0.789               | 0.748              | 0.719          | +0.069                  | 2 of 4           |
| pooled | 0.735               | 0.746              | 0.653          | +0.014                  | 2 of 4           |
| fedavg | 0.707               | 0.742              | 0.604          | -0.014                  | 2 of 4           |

Balanced accuracy:

| method | mean over hospitals | weighted by test n | worst hospital |
|--------|---------------------|--------------------|----------------|
| local  | 0.645               | 0.694              | 0.500          |
| pooled | 0.782               | 0.770              | 0.706          |
| fedavg | 0.765               | 0.765              | 0.674          |

AUC:

| method | mean over hospitals | weighted by test n | worst hospital |
|--------|---------------------|--------------------|----------------|
| local  | 0.670               | 0.759              | 0.427          |
| pooled | 0.856               | 0.851              | 0.759          |
| fedavg | 0.849               | 0.847              | 0.742          |

`mean over hospitals` weights every hospital equally, so hospital 2 (16 test
patients) counts as much as hospital 0 (104). `weighted by test n` weights each
hospital by its test-set size, which is the per-patient score.

## 95% confidence intervals (bootstrap over test patients)

Each hospital's test patients resampled with replacement, 1000 times, keeping
its label counts; scores averaged over seeds. Unlike the ± above, the interval
shows how much a score could move with a different draw of test patients.

Accuracy:

| hospital            | local                | pooled               | fedavg               |
|---------------------|----------------------|----------------------|----------------------|
| 0 Cleveland         | 0.737 [0.652, 0.821] | 0.771 [0.688, 0.848] | 0.773 [0.694, 0.848] |
| 1 Hungary           | 0.719 [0.629, 0.809] | 0.764 [0.674, 0.843] | 0.789 [0.710, 0.861] |
| 2 Switzerland       | 0.938 [0.938, 0.938] | 0.750 [0.562, 0.938] | 0.662 [0.438, 0.875] |
| 3 Long Beach V      | 0.764 [0.662, 0.867] | 0.653 [0.520, 0.787] | 0.604 [0.476, 0.733] |
| mean over hospitals | 0.789 [0.745, 0.828] | 0.735 [0.662, 0.803] | 0.707 [0.634, 0.775] |
| weighted by test n  | 0.748 [0.691, 0.798] | 0.746 [0.692, 0.798] | 0.742 [0.689, 0.793] |

Balanced accuracy:

| hospital            | local                | pooled               | fedavg               |
|---------------------|----------------------|----------------------|----------------------|
| 0 Cleveland         | 0.733 [0.648, 0.817] | 0.773 [0.690, 0.850] | 0.770 [0.692, 0.847] |
| 1 Hungary           | 0.715 [0.618, 0.808] | 0.781 [0.689, 0.860] | 0.794 [0.710, 0.868] |
| 2 Switzerland       | 0.500 [0.500, 0.500] | 0.867 [0.767, 0.967] | 0.820 [0.700, 0.933] |
| 3 Long Beach V      | 0.634 [0.487, 0.799] | 0.706 [0.556, 0.840] | 0.674 [0.529, 0.806] |
| mean over hospitals | 0.645 [0.593, 0.697] | 0.782 [0.727, 0.834] | 0.765 [0.710, 0.816] |
| weighted by test n  | 0.694 [0.635, 0.752] | 0.770 [0.717, 0.823] | 0.765 [0.716, 0.814] |

AUC:

| hospital            | local                | pooled               | fedavg               |
|---------------------|----------------------|----------------------|----------------------|
| 0 Cleveland         | 0.811 [0.725, 0.888] | 0.858 [0.780, 0.928] | 0.858 [0.780, 0.927] |
| 1 Hungary           | 0.833 [0.743, 0.908] | 0.876 [0.797, 0.940] | 0.875 [0.792, 0.940] |
| 2 Switzerland       | 0.427 [0.320, 0.547] | 0.933 [0.800, 1.000] | 0.920 [0.773, 1.000] |
| 3 Long Beach V      | 0.609 [0.393, 0.822] | 0.759 [0.557, 0.931] | 0.742 [0.547, 0.916] |
| mean over hospitals | 0.670 [0.604, 0.738] | 0.856 [0.790, 0.917] | 0.849 [0.784, 0.908] |
| weighted by test n  | 0.759 [0.698, 0.819] | 0.851 [0.795, 0.904] | 0.847 [0.792, 0.900] |

## FedAvg minus each other method (paired bootstrap)

`*` marks a difference whose 95% interval excludes 0. Without a `*`, the
test sets are too small to say which method is better there.

Accuracy:

| hospital            | fedavg − local           | fedavg − pooled          |
|---------------------|--------------------------|--------------------------|
| 0 Cleveland         | +0.037 [-0.033, +0.108]  | +0.002 [-0.046, +0.050]  |
| 1 Hungary           | +0.070 [-0.011, +0.155]  | +0.025 [-0.018, +0.074]  |
| 2 Switzerland       | -0.275 [-0.500, -0.062]* | -0.088 [-0.213, +0.000]  |
| 3 Long Beach V      | -0.160 [-0.293, -0.035]* | -0.049 [-0.102, -0.009]* |
| mean over hospitals | -0.082 [-0.153, -0.012]* | -0.027 [-0.065, +0.005]  |
| weighted by test n  | -0.006 [-0.058, +0.044]  | -0.005 [-0.033, +0.023]  |

Balanced accuracy:

| hospital            | fedavg − local           | fedavg − pooled          |
|---------------------|--------------------------|--------------------------|
| 0 Cleveland         | +0.037 [-0.035, +0.111]  | -0.003 [-0.049, +0.044]  |
| 1 Hungary           | +0.079 [-0.016, +0.176]  | +0.012 [-0.026, +0.054]  |
| 2 Switzerland       | +0.320 [+0.200, +0.433]* | -0.047 [-0.113, +0.000]  |
| 3 Long Beach V      | +0.040 [-0.123, +0.219]  | -0.031 [-0.066, -0.006]* |
| mean over hospitals | +0.119 [+0.058, +0.177]* | -0.017 [-0.041, +0.005]  |
| weighted by test n  | +0.070 [+0.013, +0.123]* | -0.005 [-0.030, +0.019]  |

AUC:

| hospital            | fedavg − local           | fedavg − pooled          |
|---------------------|--------------------------|--------------------------|
| 0 Cleveland         | +0.047 [-0.009, +0.105]  | +0.000 [-0.018, +0.018]  |
| 1 Hungary           | +0.042 [-0.012, +0.100]  | -0.001 [-0.020, +0.019]  |
| 2 Switzerland       | +0.493 [+0.280, +0.653]* | -0.013 [-0.040, +0.000]  |
| 3 Long Beach V      | +0.133 [-0.070, +0.365]  | -0.017 [-0.053, +0.013]  |
| mean over hospitals | +0.179 [+0.101, +0.246]* | -0.008 [-0.021, +0.003]  |
| weighted by test n  | +0.088 [+0.039, +0.139]* | -0.004 [-0.016, +0.007]  |


## Data caveats

- 2 Switzerland: every training patient has the same label, so a model trained there alone can only predict that label. Its `local` accuracy is the majority-class baseline (0.938), not learning.
- 2 Switzerland: only 16 test patients, so its accuracy moves in steps of 0.062.
- 3 Long Beach V: only 45 test patients, so its accuracy moves in steps of 0.022.
- 2 Switzerland: the test set has 1 patient(s) with label 0 and 15 with label 1. Half of its balanced accuracy rests on the 1 rarer one(s), and its AUC on 15 label-0/label-1 pairs (steps of 0.067).
- 2 Switzerland: the bootstrap keeps label counts fixed, so every resample contains the same 1 rarer patient(s). Its intervals understate the real uncertainty; don't draw conclusions from this hospital alone.

## Run settings

| method | seeds         | lr    | server rounds | epochs (per round if federated) |
|--------|---------------|-------|---------------|---------------------------------|
| local  | 0, 1, 2, 3, 4 | 0.1   | -             | 50                              |
| pooled | 0, 1, 2, 3, 4 | 0.003 | -             | 50                              |
| fedavg | 0, 1, 2, 3, 4 | 0.003 | 50            | 1                               |
