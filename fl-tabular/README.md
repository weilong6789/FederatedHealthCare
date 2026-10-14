# fl-tabular — FedAvg on Fed-Heart-Disease

Four real hospitals train a shared heart-disease classifier without pooling
their data. The data is FLamby's Fed-Heart-Disease: 740 patients, 13 features,
binary label, already split by the hospital each patient came from.

| hospital | where | train | test |
|---|---|---|---|
| 0 | Cleveland | 199 | 104 |
| 1 | Hungary | 172 | 89 |
| 2 | Switzerland | 30 | 16 |
| 3 | VA Long Beach | 85 | 45 |

The model is FLamby's reference `Baseline`: logistic regression, 14 parameters.
Tiny on purpose: one federated run takes about 40 seconds on a laptop.

## What gets compared

| method | what it is |
|---|---|
| `local` | each hospital trains alone on its own patients |
| `pooled` | one model trained on everyone's data in one place (what FL is not allowed to do) |
| `fedavg` | hospitals train locally, a server averages their weights, repeat |

Every method uses the same data splits, model, optimizer, and evaluation, all
from `fltab/harness.py`. Each is scored on every hospital's own test set.

## Metrics

Every evaluation logs three numbers per hospital:

| metric | what it is | always predicting one label scores |
|---|---|---|
| `accuracy` | share of patients classified right (threshold 0.5); FLamby's metric | that hospital's majority-class share (0.938 at Switzerland) |
| `balanced_accuracy` | mean of the share of label-1 patients caught and the share of label-0 patients caught | 0.5 everywhere |
| `auc` | chance a random label-1 patient gets a higher score than a random label-0 one; ignores the threshold | 0.5 everywhere |

Accuracy alone flatters a model that has collapsed to one answer, which is
exactly how models fail under heavy DP noise. Read balanced accuracy and AUC
alongside it. Switzerland's test set has a single label-0 patient, so its
balanced accuracy and AUC are still noisy; `summary.md` flags this.

## Setup (once)

Everything runs in the conda env `flamby` (Python 3.10).

```bash
conda create -n flamby python=3.10
conda activate flamby
git clone https://github.com/owkin/FLamby.git ~/FLamby
pip install -e "$HOME/FLamby[heart]"
cd ~/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Then, from this folder:

```bash
pip install -e .
flwr federation simulation-config --num-supernodes 4
```

The last line tells Flower to simulate 4 clients, one per hospital. Flower's
default is 10; the 6 extra ones have no hospital and crash every round (the
results are unaffected, but the log fills with errors).

## Run

Always run from this folder, with the env active: results are written to
`results/` relative to where you launch from.

```bash
conda activate flamby
./run_all.sh
```

That runs every method for seeds 0-4 and rebuilds the table and plots. The
previous `results/results.csv` is renamed, not overwritten. The pieces, if you
want them one at a time:

```bash
python baselines.py --seeds 0 1 2 3 4 --lr 0.01 --epochs 50      # pooled + local
flwr run . --stream --run-config "seed=0"                         # fedavg
python summarize.py                                              # table + plots
```

Settings you can override with `--run-config` (defaults in `pyproject.toml`
under `[tool.flwr.app.config]`): `num-server-rounds`, `local-epochs`, `lr`,
`seed`.

## Files

| file | what it does |
|---|---|
| `fltab/harness.py` | everything shared: data loaders, model, SGD training loop, evaluation, CSV logging. No Flower imports. |
| `fltab/client_app.py` | hospital side: load global weights, train locally, send weights back |
| `fltab/server_app.py` | coordinator: hand out weights, average what comes back, evaluate, repeat |
| `baselines.py` | the `pooled` and `local` comparisons. No Flower. |
| `summarize.py` | turns `results/results.csv` into `results/summary.md` and plots per metric |
| `run_all.sh` | runs all of the above for 5 seeds |
| `pyproject.toml` | dependencies + Flower app config |
| `results/` | the log (`results.csv`), the table, the plots |

`results/summary.md` is generated: edit `summarize.py`, not the file.

## Where this goes next

The point of these baselines is differential privacy, which is not in the
code yet:

1. **DP** — wrap the strategy in
   `DifferentialPrivacyServerSideFixedClipping`. Plot ε against AUC.
2. **Correlated noise** — compare against the independent-noise baseline from 1.
