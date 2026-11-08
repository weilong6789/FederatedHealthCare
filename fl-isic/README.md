# fl-isic — FedAvg on Fed-ISIC2019

FLamby's Fed-ISIC2019: dermoscopy images of skin lesions, 8 diagnosis
classes, split by the hospital (or archive) each image came from.

| hospital | train | test |
|---|---|---|
| 0 | 9930 | 2483 |
| 1 | 3163 | 791 |
| 2 | 2691 | 672 |
| 3 | 1807 | 452 |
| 4 | 655 | 164 |
| 5 | 351 | 88 |

Unlike Fed-Heart-Disease, the hospitals' label mixes differ a lot, so this is
where FedAvg can actually fall behind pooled training. Model, loss and
augmentations are FLamby's: EfficientNet-b0 pretrained on ImageNet (about 4M
parameters), weighted focal loss, batch 64, random 200 px crops. Needs a GPU.

## Setup on ICE (once)

```bash
module load anaconda3 && conda activate ~/scratch/envs/flamby
pip install -e "$HOME/FLamby[isic2019]" "albumentations==1.3.1" "numpy<2"
```

`albumentations` must be 1.3.1: FLamby's augmentations use `Flip` and the old
`CoarseDropout` arguments, which 2.x removed, so `FedIsic2019(train=True)`
fails on a newer version.

Data (about 10 GB download). ISIC moved its files, so point FLamby's script at
the new bucket first:

```bash
cd ~/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts
sed -i 's#isic-challenge-data.s3.amazonaws.com/2019/#isic-archive.s3.amazonaws.com/challenges/2019/#' download_isic.py
python download_isic.py --output-folder ~/scratch/isic2019
python resize_images.py
```

## Run

Smoke test first, on a GPU node (a couple of minutes):

```bash
salloc -N1 --cpus-per-task=8 --gres=gpu:1 -t 0:30:00
module load anaconda3 && conda activate ~/scratch/envs/flamby
cd ~/FederatedHealthCare/fl-isic
python isic.py --method fedavg --T 5 --epochs 1 --max-images 256 --out results_smoke
```

Real runs as batch jobs, one per method and seed:

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

Every run appends to `results/isic.csv`, so jobs running at once can share
it. Arguments (see `python isic.py -h`):

| argument | default | what it is |
|---|---|---|
| `--method` | | `pooled`, `fedavg` or `local` |
| `--epochs` | 20 | training budget in pooled-equivalent epochs (FLamby's pooled default) |
| `--T` | 100 | FedAvg local steps per round |
| `--opt`, `--lr` | adam, 5e-4 | same for every method (FLamby's pooled setting) |
| `--eval-every` | 1 | evaluate about this often, in epochs |
| `--max-images` | | cap every train/test set, for quick tests |

## Budget

One pooled epoch is 291 steps of batch 64, so the default 20 epochs is 5820
steps. FedAvg spends the same steps on the global model (5820 / T rounds), but
all six hospitals train each round, so it does six times the compute of
pooled. Local trains each hospital for 20 passes over its own data, about the
same compute as pooled.

## Output

`results/isic.csv`: one row per evaluation per hospital, plus `client = all`
(all 4650 test images together). Metrics: `balanced_accuracy` (FLamby's
metric for ISIC, the mean recall over the 8 classes), `accuracy`, and `auc`
(one-vs-rest, averaged over the classes present). Accuracy is mostly a
measure of the common classes; read balanced accuracy first.

`results/isic_predictions.csv`: the final model's 8 class probabilities for
every test image, for bootstrap intervals later.
