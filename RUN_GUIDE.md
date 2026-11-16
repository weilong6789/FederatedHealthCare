# FederatedHealthCare Run Guide

How to set up, run and read every part of the code.

## 1. The project

Two experiments on FLamby datasets. The goal is to measure what federated training costs, then add differential privacy (the privacy code is not written yet).

| experiment | data | model | where it runs |
|---|---|---|---|
| `fl-tabular` | Fed-Heart-Disease: 4 hospitals, 740 patients, 13 features | logistic regression, 14 parameters | laptop, about 40 s per federated run |
| `fl-isic` | Fed-ISIC2019: 6 hospitals, about 23,000 dermoscopy images, 8 classes | EfficientNet-b0 | GPU, Slurm jobs |

### Three methods, compared everywhere

| method | what it is |
|---|---|
| `local` | each hospital trains alone on its own patients (the floor) |
| `pooled` | one model trained on everyone's data in one place (the ceiling, and what federated learning is not allowed to do) |
| `fedavg` | hospitals train locally, a server averages their weights, repeat |

All three share the same data splits, model, optimizer and scoring. Each is scored on every hospital's own test set.

### Repo map

| path | what it is |
|---|---|
| `README.md` | one-paragraph project description |
| `.devcontainer/` | container that sets up fl-tabular automatically |
| `fl-tabular/` | heart-disease experiments (Flower + PyTorch) |
| `fl-tabular/claude/` | `baseline_guide.html`, a standalone explainer page |
| `fl-isic/` | skin-lesion experiments, run as GPU jobs on Slurm |
| `results/` folders | logged CSVs, tables and plots, tracked in git |

---

## 2. fl-tabular (Fed-Heart-Disease)

### Setup, option A: dev container

Open the repo in GitHub Codespaces or VS Code Dev Containers. It builds Python 3.10 and runs `.devcontainer/setup.sh`, which:

- clones FLamby and installs it with the heart extra
- pins `numpy<2` (FLamby breaks on NumPy 2)
- downloads the heart-disease dataset
- installs the `fltab` package from `fl-tabular`
- tells Flower to simulate 4 clients

### Setup, option B: by hand (once)

```bash
conda create -n flamby python=3.10
conda activate flamby
git clone https://github.com/owkin/FLamby.git ~/FLamby
pip install -e "$HOME/FLamby[heart]"
pip install "numpy<2"
cd ~/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Then, from the `fl-tabular` folder:

```bash
pip install -e .
flwr federation simulation-config --num-supernodes 4
```

### Two rules before the first run

- **Run from `fl-tabular/`.** Results are written to `results/` relative to where you launch. `flwr run` executes a packaged copy of the code, so launching elsewhere sends output to the wrong place. If you must, set `FLTAB_ROOT` to the project folder.
- **Four clients, not ten.** Flower simulates 10 clients by default. The 6 extra have no hospital and crash every round. The `simulation-config` command above fixes it. Results are correct either way, but the log fills with errors.

Keep the conda env `flamby` active for every command.

### Run the full pipeline

```bash
conda activate flamby
cd fl-tabular
python tune.py        # once, about 10 minutes
./run_all.sh          # every method, 5 seeds, tables and plots
```

- `tune.py` trains each method on 80% of every hospital's training data and scores it on the other 20%. It writes the best settings to `results/tuned.env`. The test set plays no part.
- `run_all.sh` retrains on the full training data with those settings for seeds 0 to 4, then builds the tables. Old `results.csv` and `predictions.csv` are renamed, not overwritten.

### Run the pieces one at a time

```bash
python baselines.py --seeds 0 1 2 3 4 --pooled-lr 0.01 --local-lr 0.01 --epochs 50
flwr run . --stream --run-config "seed=0"
python summarize.py
```

| `--run-config` key | default | what it sets |
|---|---|---|
| `num-server-rounds` | 50 | FedAvg rounds |
| `local-epochs` | 1 | epochs each hospital trains per round |
| `lr` | 0.01 | learning rate |
| `seed` | 0 | random seed |

Defaults live in `pyproject.toml` under `[tool.flwr.app.config]`.

### The `fltab` package

| file | what it does |
|---|---|
| `fltab/harness.py` | everything shared: data loaders, model, SGD loop, evaluation, CSV logging. No Flower imports |
| `fltab/client_app.py` | hospital side: load global weights, train locally, send weights back |
| `fltab/server_app.py` | coordinator: hand out weights, average what returns, evaluate, repeat |
| `pyproject.toml` | dependencies and the Flower app config |

Change training or scoring in `harness.py` once and every method follows.

### The scripts

| script | what it does | time |
|---|---|---|
| `baselines.py` | pooled and local comparisons, no Flower | |
| `tune.py` | picks learning rates (and FedAvg local epochs) on a validation split | ~10 min |
| `run_all.sh` | all methods, 5 seeds, then summaries | |
| `local_steps.py` | FedAvg with T local SGD steps per round vs pooled, same step budget | ~8 min |
| `lr_T_grid.py` | FedAvg minus pooled over a learning rate by T grid | ~30 min |
| `summarize.py` | tables, plots and bootstrap intervals | |
| `OPACUS_HANDOFF.md` | where DP-SGD (Opacus) plugs into the code | |

### Reading `results/`

| file | what it holds |
|---|---|
| `results.csv`, `predictions.csv` | every evaluation, and per-patient test predictions |
| `summary.md` | tables per metric, generated by `summarize.py` |
| `tuned.env`, `tuning.md`, `tuning_curve.png` | chosen settings, full grid, FedAvg convergence |
| `*_vs_round.png`, `accuracy_vs_bytes.png` | score curves per method |
| `local_steps*`, `lr_T_grid*` | outputs of the two sweeps |

Read balanced accuracy and AUC next to accuracy: a model that always says one label scores 0.5 on both. Edit `summarize.py`, never `summary.md`.

---

## 3. fl-isic (Fed-ISIC2019)

### Setup on the ICE cluster (once)

```bash
module load anaconda3 && conda activate ~/scratch/envs/flamby
pip install -e "$HOME/FLamby[isic2019]" "albumentations==1.3.1" "numpy<2"
```

- `albumentations` must be 1.3.1. FLamby's augmentations use `Flip` and old `CoarseDropout` arguments that 2.x removed, so loading the training set fails on a newer version.
- Training needs a GPU node. Plan for about 10 GB of disk for the images.

### Download the images

ISIC moved its files, so point FLamby's download script at the new bucket first:

```bash
cd ~/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts
sed -i 's#isic-challenge-data.s3.amazonaws.com/2019/#isic-archive.s3.amazonaws.com/challenges/2019/#' download_isic.py
python download_isic.py --output-folder ~/scratch/isic2019
python resize_images.py
```

### Smoke test first, on a GPU node

```bash
salloc -N1 --cpus-per-task=8 --gres=gpu:1 -t 0:30:00
module load anaconda3 && conda activate ~/scratch/envs/flamby
cd ~/FederatedHealthCare/fl-isic
python isic.py --method fedavg --T 5 --epochs 1 --max-images 256 --out results_smoke
```

Takes a couple of minutes. It caps every train and test set at 256 images and writes to `results_smoke`, which git ignores.

### Real runs as Slurm jobs

```bash
mkdir -p logs
for s in 0 1 2; do
  sbatch isic.sbatch --method pooled --seed $s
  sbatch isic.sbatch --method fedavg --T 100 --seed $s
  sbatch isic.sbatch --method local --seed $s
done
squeue -u $USER
tail -f logs/isic-<jobid>.out
```

One job per method and seed. Each asks for 1 GPU, 8 CPUs, 32 GB and 8 hours. Every run appends to `results/isic.csv`, so jobs can run at the same time.

### `isic.py` arguments

| argument | default | what it is |
|---|---|---|
| `--method` | required | `pooled`, `fedavg` or `local` |
| `--epochs` | 20 | budget in pooled-equivalent epochs |
| `--T` | 100 | FedAvg local steps per round |
| `--opt`, `--lr` | adam, 5e-4 | same for every method |
| `--eval-every` | 1 | evaluate about this often, in epochs |
| `--max-images` | none | cap every train and test set, for quick tests |
| `--out` | `results` | folder for the output CSVs |

One pooled epoch is 291 steps of batch 64, so 20 epochs is 5820 steps. FedAvg runs 5820 / T rounds.

### Longer runs and the sweep of T

A different budget goes to its own folder so it does not mix with the 20-epoch rows. FedAvg at 40 epochs takes about 3 hours.

```bash
sbatch isic.sbatch --method fedavg --epochs 40 --out results40
python summarize.py results40
```

`sweep_T.sh` runs FedAvg at T = 1, 5, 20, 97, 291, 970 and 5820, seeds 0 to 2: 21 jobs. Time a short T = 1 run first, since its 5820 rounds must fit in 8 hours.

```bash
bash sweep_T.sh
TS="1 5" SEEDS="0" bash sweep_T.sh
```

### The files

| file | what it does |
|---|---|
| `isic.py` | pooled, local and FedAvg on one GPU; writes the CSVs |
| `isic.sbatch` | Slurm wrapper: 1 GPU, 32 GB, 8 hours |
| `sweep_T.sh` | submits the T sweep, one job per T and seed |
| `summarize.py` | tables, plots, paired bootstrap; about 2 minutes |
| `plot_T_sweep.py` | sweep figures, in seconds, from finished runs only |
| `results/`, `results40/` | tracked outputs; `results_*/` and `logs/` are git-ignored |

### Reading the results

| file | what it holds |
|---|---|
| `results/isic.csv` | one row per evaluation per hospital, plus `all` |
| `results/isic_predictions.csv` | final class probabilities per test image |
| `results/summary.md` | final scores, label mix, bootstrap intervals |
| `isic_curve_<metric>.png` | score on all test images vs epochs |
| `T_*.png`, `isic_T_sweep.png` | sweep figures |

Read balanced accuracy first. At 20 epochs, FedAvg with T = 100 scores 0.684 against pooled's 0.748 on all test images. Plain accuracy hides that gap.

---

## 4. Troubleshooting

| symptom | fix |
|---|---|
| errors every round in the log | `flwr federation simulation-config --num-supernodes 4` |
| NumPy errors from FLamby | `pip install "numpy<2"` |
| ISIC training set fails to load | `pip install "albumentations==1.3.1"` |
| ISIC download fails | apply the `sed` fix to `download_isic.py` |
| results in the wrong folder | run from `fl-tabular/` or set `FLTAB_ROOT` |

## 5. What is next: differential privacy

- All SGD goes through `train_sgd` (whole epochs) and `train_steps` (fixed steps) in `fltab/harness.py`.
- Each hospital needs one Opacus `PrivacyEngine` that lives across all rounds, or epsilon is counted per round instead of in total.
- Opacus prefixes `state_dict` keys with `_module.`, so send weights through `model._module`.
- Hospitals differ in size, so the same noise costs different epsilon. Pick delta below 1/n of the smallest hospital.
- Planned: server-side clipping with an epsilon vs AUC plot, then correlated noise.

Details are in `fl-tabular/OPACUS_HANDOFF.md`.

## 6. Cheat sheet

```bash
# fl-tabular, after setup
cd fl-tabular
python tune.py && ./run_all.sh

# fl-isic, on a GPU cluster, after setup
cd fl-isic
sbatch isic.sbatch --method fedavg --T 100 --seed 0
python summarize.py
```

Check `results/summary.md` in each folder.
