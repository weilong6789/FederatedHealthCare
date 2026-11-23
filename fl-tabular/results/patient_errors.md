## Majority-label baseline vs models (test accuracy)

| hospital | test n | dominant label | guess dominant label | local | pooled | fedavg |
|---|---|---|---|---|---|---|
| Cleveland | 104 | healthy (56/104) | 0.538 | 0.737 | 0.771 | 0.773 |
| Hungary | 89 | healthy (56/89) | 0.629 | 0.719 | 0.764 | 0.789 |
| Switzerland | 16 | sick (15/16) | 0.938 | 0.938 | 0.750 | 0.662 |
| Long Beach VA | 45 | sick (35/45) | 0.778 | 0.764 | 0.653 | 0.604 |
| all patients | 254 | each hospital's own | 0.638 | 0.748 | 0.746 | 0.742 |
| all patients | 254 | sick for everyone (131/254) | 0.516 | | | |

## Same patients wrong across seeds? (tuned baselines, 5 seeds)

Patients by the number of seeds in which they are misclassified.

**local**

| hospital | 0 of 5 | 1 of 5 | 2 of 5 | 3 of 5 | 4 of 5 | 5 of 5 |
|---|---|---|---|---|---|---|
| Cleveland | 74 | 1 | 2 | 1 | 1 | 25 |
| Hungary | 64 | 0 | 0 | 0 | 0 | 25 |
| Switzerland | 15 | 0 | 0 | 0 | 0 | 1 |
| Long Beach VA | 33 | 1 | 1 | 0 | 0 | 10 |

**pooled**

| hospital | 0 of 5 | 2 of 5 | 3 of 5 | 4 of 5 | 5 of 5 |
|---|---|---|---|---|---|
| Cleveland | 79 | 1 | 1 | 1 | 22 |
| Hungary | 68 | 0 | 0 | 0 | 21 |
| Switzerland | 12 | 0 | 0 | 0 | 4 |
| Long Beach VA | 29 | 0 | 1 | 0 | 15 |

**fedavg**

| hospital | 0 of 5 | 1 of 5 | 2 of 5 | 3 of 5 | 4 of 5 | 5 of 5 |
|---|---|---|---|---|---|---|
| Cleveland | 75 | 4 | 1 | 2 | 4 | 18 |
| Hungary | 62 | 5 | 5 | 2 | 2 | 13 |
| Switzerland | 10 | 0 | 0 | 1 | 1 | 4 |
| Long Beach VA | 26 | 0 | 1 | 1 | 1 | 16 |

## Always-wrong patients by true label (tuned baselines)

Healthy called sick = true label 0; sick called healthy = true label 1.

| hospital | method | always wrong | healthy called sick | sick called healthy |
|---|---|---|---|---|
| Cleveland | local | 25 | 11 | 14 |
| Cleveland | pooled | 22 | 14 | 8 |
| Cleveland | fedavg | 18 | 8 | 10 |
| Hungary | local | 25 | 15 | 10 |
| Hungary | pooled | 21 | 16 | 5 |
| Hungary | fedavg | 13 | 8 | 5 |
| Switzerland | local | 1 | 1 | 0 |
| Switzerland | pooled | 4 | 0 | 4 |
| Switzerland | fedavg | 4 | 0 | 4 |
| Long Beach VA | local | 10 | 6 | 4 |
| Long Beach VA | pooled | 15 | 2 | 13 |
| Long Beach VA | fedavg | 16 | 2 | 14 |

## Same patients wrong across K? (2440-step budget, 5 seeds)

| run | always wrong | same as pooled | not in pooled's set | in pooled's set but right here |
|---|---|---|---|---|
| pooled | 54 | 54 | 0 | 0 |
| FedAvg K=1 | 54 | 54 | 0 | 0 |
| FedAvg K=8 | 54 | 54 | 0 | 0 |
| FedAvg K=61 | 54 | 54 | 0 | 0 |
| FedAvg K=305 | 54 | 54 | 0 | 0 |
| FedAvg K=2440 | 47 | 47 | 0 | 7 |

Pooled's always-wrong patients sit on average 0.21 from the 0.5 cutoff (mean predicted probability over seeds), so most are not borderline.
