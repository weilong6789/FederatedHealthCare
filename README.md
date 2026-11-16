# FederatedHealthCare
Federated learning and differential privacy.
Using the Flamby datasets to evaluate the privacy costs of federated learning.
Finding ways to improve privacy.

## Progress log

Notation: K is FedAvg local steps per round, T is total communication rounds (T = budget / K). Before Nov 13 the code called K "T".

### Nov 9 (Mon)
- ISIC: table and plot of whether scores were still rising at epoch 20 (pooled and FedAvg K=100, straight-line fit over epochs 10-20). They were, so the run was repeated at 40 epochs.

### Nov 10 (Tue)
- `fl-isic/summarize.py` takes a results folder (40-epoch runs go in `results40/`), fits the trend over the second half of training, and skips rows spliced together by jobs writing at once.
- ISIC 40 epochs: pooled still rising at epoch 40 on all three metrics (3/3 seeds), balanced accuracy 0.767. FedAvg K=100 flat on balanced accuracy (0.686), slightly ahead on accuracy (0.800 vs 0.783).
- Heart disease: `local_steps.py` sweeps K from 1 step per round to a single round, with every run taking the same number of SGD steps. Default budget is 20 epochs (2440 steps), where pooled has peaked. Every K lands at 0.744-0.749 accuracy, same as pooled.
- ISIC: `sweep_K.sh` (then `sweep_T.sh`) submits FedAvg at 20 epochs for K = 1, 5, 20, 97, 291, 970, 5820, seeds 0-2. `isic.py` locks the CSVs while appending.
- ISIC K sweep, balanced accuracy on all test images (pooled 0.748): K=1 0.485, K=5 0.538, K=20 0.609, K=97 0.689, K=291 0.714, K=970 0.682, K=5820 (one round) 0.350. FedAvg is below pooled everywhere. K=291 is closest (-0.034). Accuracy at K=97 and K=100 is statistically tied with pooled.

### Nov 11 (Wed)
- ISIC K=20 seed 1 hit its time limit and was rerun. Added `plot_K_sweep.py` (score vs K, curves per K, per hospital, per class).
- Added a DP-SGD slot (`fltab/privacy.py`) behind `--dp` in `baselines.py`, `local_steps.py` and `lr_T_grid.py`. Plain SGD results are unchanged. `--dp` goes to a stub `DPSGDTrainer` for Opacus.

### Nov 12 (Thu)
- `fl-tabular`: removed the bytes-uploaded plot and column (redundant with accuracy vs round) and the stale SCAFFOLD guide.

### Nov 13 (Fri)
- Renamed the FedAvg local-steps parameter T to K in `fl-isic` and `fl-tabular` (flag, CSV column, run tags, scripts, result files). ISIC figures regenerated.

### Nov 16 (Mon)
- Added `RUN_GUIDE.md` (setup, running, outputs for both experiments).
- Heart disease: `budget_sweep.py` repeats the K sweep at budgets from 2440 steps (20 epochs) down to 2, with a paired bootstrap over test patients. Scores hold pooled-at-20-epochs down to about 600 steps (4.9 epochs) on accuracy and balanced accuracy, and 840 steps (6.9 epochs) on AUC. `budget_sweep.md` still labels runs "FedAvg T=" for local steps (old naming).

### Open
- Whether pooled keeps improving past 40 epochs on ISIC.
- Whether the ISIC K sweep needs a 40-epoch version.
