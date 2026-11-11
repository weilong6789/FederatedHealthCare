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
python tune.py      # once: pick hyperparameters on a validation split (~10 min)
./run_all.sh
```

`tune.py` trains every method on 80% of each hospital's training data, scores
it on the other 20%, and writes the best setting per method to
`results/tuned.env` (grid in `results/tuning.md`, FedAvg convergence in
`results/tuning_curve.png`). The test set plays no part in choosing settings.

`run_all.sh` then retrains every method on the full training data with those
settings, for seeds 0-4, and rebuilds the tables and plots. The previous
`results.csv` and `predictions.csv` are renamed, not overwritten. The pieces,
if you want them one at a time:

```bash
python baselines.py --seeds 0 1 2 3 4 --pooled-lr 0.01 --local-lr 0.01 --epochs 50
flwr run . --stream --run-config "seed=0"                         # fedavg
python summarize.py                                              # tables + plots
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
| `local_steps.py` | FedAvg with T local SGD steps per round vs pooled, same step budget |
| `lr_T_grid.py` | FedAvg − pooled over a learning rate × T grid |
| `fltab/privacy.py` | the trainer every in-process experiment uses; DP-SGD goes in `DPSGDTrainer` here |
| `OPACUS_HANDOFF.md` | where DP-SGD (Opacus) plugs into this code |
| `tune.py` | picks each method's learning rate (and FedAvg's local epochs) on a validation split |
| `summarize.py` | turns `results/results.csv` into `results/summary.md` and plots per metric |
| `run_all.sh` | runs all of the above for 5 seeds |
| `pyproject.toml` | dependencies + Flower app config |
| `results/` | the logs (`results.csv`, `predictions.csv`), tuning output, tables, plots |

`results/summary.md` is generated: edit `summarize.py`, not the file.

## Local steps T vs pooled

```bash
python local_steps.py      # T = 1 ... 2440, 20 epochs, seeds 0-4 (~8 min)
```

FedAvg where each hospital takes T minibatch SGD steps per round (not
epochs), compared with pooled training on the same budget: every run takes
the same number of SGD steps (20 pooled-equivalent epochs = 20 × 122 steps),
so FedAvg with T steps runs budget / T rounds. With T = 1 the server's
average is one SGD step on a gradient averaged over all hospitals, which is
pooled SGD with a bigger batch; larger T lets each hospital drift toward its
own data between averages but needs far fewer rounds. Same lr for every run
(`--lr`, default 0.003), so T is the only thing that changes.

The default sweep covers the two endpoints and 10 values between them, spaced
evenly on a log scale. Each T divides the 2440-step budget, so every run takes
exactly the same number of steps:

| T | 1 | 2 | 4 | 8 | 20 | 40 | 61 | 122 | 305 | 610 | 1220 | 2440 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rounds | 2440 | 1220 | 610 | 305 | 122 | 61 | 40 | 20 | 8 | 4 | 2 | 1 |

T = 1 is pooled SGD with a bigger batch. T = 2440 is a single round: each
hospital trains alone for the whole budget (Switzerland's 30 patients get about
300 passes), and the server averages the four models once at the end. A
different `--epochs` picks its own divisors; `--Ts` overrides the list.

The budget is 20 epochs because that's where pooled has finished: in the
earlier 50-epoch run its test balanced accuracy peaked at 0.746 around epoch
10-15 and slid to 0.734 by 50. Training past that only adds overfitting on top
of whatever T changes.

Pooled here is not quite `baselines.py`'s pooled. FLamby scales features with
each hospital's own mean/std, but with `pooled=True` it uses all hospitals'
statistics instead; FedAvg can only use the per-hospital version. So for this
comparison pooled trains on the four per-hospital datasets concatenated, and
both are scored on the 254 test patients scaled per hospital
(`pooled_steps` and `all_patients_test` in `fltab/harness.py`).

Outputs in `results/`: `local_steps_curve_<metric>.png` (score vs epochs, one
line per T, pooled dashed), `local_steps_rounds_<metric>.png` (score vs
communication rounds), `local_steps_tradeoff.png` (final score and rounds
needed vs T), `local_steps.md` (table) and `local_steps.csv` (raw).
Each final model is also scored on every hospital's own test set
(`local_steps_hospitals.png`, `local_steps_hospitals.csv`, tables at the end
of `local_steps.md`), to check whether a large T leaves one hospital behind
even when the overall score holds.
Try `--lr 0.1` too: drift grows with lr × T, so a large lr is where a gap
from pooled would show first.

### Learning rate × T grid

```bash
python lr_T_grid.py        # lr 0.003-1.0 × T 1-500, 20 epochs, seeds 0-2 (~30 min)
```

Heatmap of FedAvg's final score minus pooled's, same lr and step budget in
each cell (`results/lr_T_grid_<metric>.png`, table in `lr_T_grid.md`). Drift
grows with lr × T, so this shows where "FedAvg ≈ pooled" stops holding. The
table also lists pooled's own score per lr, since a very large lr can break
pooled too.

## Confidence intervals

The ± in `summary.md` only covers retraining with different seeds; the test
patients never change. With 16 test patients at Switzerland and 45 at Long
Beach VA, which patients landed in the test set matters far more. So every
final model also saves its per-patient test predictions
(`results/predictions.csv`), and `summarize.py` bootstraps them: resample each
hospital's test patients with replacement 1000 times, re-score, and report the
middle 95% as the interval.

The resampling keeps each hospital's label counts fixed, so every resample has
both labels. All methods are scored on the same resamples, which gives paired
intervals for the differences (FedAvg minus pooled, FedAvg minus local). A
difference whose interval contains 0 is not evidence that one method is better.

## Differential privacy

`baselines.py`, `local_steps.py` and `lr_T_grid.py` all take the same DP flags:

```bash
python local_steps.py --dp --noise-multiplier 1.0 --max-grad-norm 1.0 --delta 1e-3
```

Each hospital (and pooled) then trains with DP-SGD, ε per hospital goes in the
tables, and output files get a `_dp_s<σ>_c<C>` suffix so the non-DP results
stay put. The DP-SGD step itself is not implemented yet: it goes in
`DPSGDTrainer` in `fltab/privacy.py`, and until then `--dp` exits with an
error. `OPACUS_HANDOFF.md` has the details.

## Where this goes next

The point of these baselines is differential privacy, which is not in the
code yet:

1. **DP** — wrap the strategy in
   `DifferentialPrivacyServerSideFixedClipping`. Plot ε against AUC.
2. **Correlated noise** — compare against the independent-noise baseline from 1.
