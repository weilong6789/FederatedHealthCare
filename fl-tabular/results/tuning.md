## Hyperparameter tuning (validation split, not test)

Validation AUC, mean over hospitals (Switzerland's validation slice has one label, so it has no AUC and is left out), mean over seeds 0, 1, 2. Local and pooled train 50 epochs; FedAvg runs 50 rounds.

| lr | pooled | local |
|---|---|---|
| 0.003 | 0.780 ← | 0.749 |
| 0.01 | 0.772 | 0.757 |
| 0.03 | 0.766 | 0.760 |
| 0.1 | 0.755 | 0.761 ← |

| lr | FedAvg, 1 local epoch(s) | FedAvg, 5 local epoch(s) |
|---|---|---|
| 0.003 | 0.844 ← | 0.838 |
| 0.01 | 0.839 | 0.836 |
| 0.03 | 0.838 | 0.833 |
| 0.1 | 0.839 | 0.830 |

Chosen (`←`, also in `results/tuned.env`):

- pooled: lr 0.003
- local: lr 0.1
- fedavg: lr 0.003, 1 local epoch(s) per round

The validation slices are small (98 patients, 17 at Long Beach VA), so gaps
under about 0.01 between settings are noise: the choice then barely matters,
and what tuning buys is that the test set played no part in it.

FedAvg convergence: validation AUC changed by +0.0028 over the last 10 rounds. It has levelled off.
