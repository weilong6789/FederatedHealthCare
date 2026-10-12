## Per-hospital test accuracy (mean ± std over seeds)

| hospital       | test n | majority class | local         | pooled        | fedavg        |
|----------------|--------|----------------|---------------|---------------|---------------|
| 0 Cleveland    | 104    | 0.538          | 0.737 ± 0.005 | 0.773 ± 0.005 | 0.790 ± 0.008 |
| 1 Hungary      | 89     | 0.629          | 0.742 ± 0.008 | 0.764 ± 0.000 | 0.775 ± 0.008 |
| 2 Switzerland  | 16     | 0.938          | 0.938 ± 0.000 | 0.750 ± 0.000 | 0.688 ± 0.000 |
| 3 Long Beach V | 45     | 0.778          | 0.724 ± 0.025 | 0.667 ± 0.000 | 0.560 ± 0.010 |

`majority class` is the accuracy of always predicting that hospital's more
common label. A method at or below it has learned nothing useful there.

## Accuracy minus majority-class baseline

| hospital       | local  | pooled | fedavg |
|----------------|--------|--------|--------|
| 0 Cleveland    | +0.198 | +0.235 | +0.252 |
| 1 Hungary      | +0.112 | +0.135 | +0.146 |
| 2 Switzerland  | +0.000 | -0.188 | -0.250 |
| 3 Long Beach V | -0.053 | -0.111 | -0.218 |

## Summary

| method | mean over hospitals | weighted by test n | worst hospital | mean lift over baseline | hospitals beaten |
|--------|---------------------|--------------------|----------------|-------------------------|------------------|
| local  | 0.785               | 0.749              | 0.724          | +0.064                  | 2 of 4           |
| pooled | 0.738               | 0.750              | 0.667          | +0.018                  | 2 of 4           |
| fedavg | 0.703               | 0.738              | 0.560          | -0.017                  | 2 of 4           |

`mean over hospitals` weights every hospital equally, so hospital 2 (16 test
patients) counts as much as hospital 0 (104). `weighted by test n` weights each
hospital by its test-set size, which is the per-patient accuracy.

## Data caveats

- 2 Switzerland: every training patient has the same label, so a model trained there alone can only predict that label. Its `local` accuracy is the majority-class baseline (0.938), not learning.
- 2 Switzerland: only 16 test patients, so its accuracy moves in steps of 0.062.
- 3 Long Beach V: only 45 test patients, so its accuracy moves in steps of 0.022.

## Run settings

| method | seeds         | lr   | server rounds | epochs (per round if federated) |
|--------|---------------|------|---------------|---------------------------------|
| local  | 0, 1, 2, 3, 4 | 0.01 | -             | 50                              |
| pooled | 0, 1, 2, 3, 4 | 0.01 | -             | 50                              |
| fedavg | 0, 1, 2, 3, 4 | 0.01 | 50            | 1                               |
