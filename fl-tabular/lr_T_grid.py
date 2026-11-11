"""Where does FedAvg stop matching pooled? A grid over learning rate × T.

    python lr_T_grid.py                       # defaults below, ~30 min
    python lr_T_grid.py --lrs 0.01 0.1 1 --Ts 1 10 100 --seeds 0   # quick look
    python lr_T_grid.py --dp --noise-multiplier 1.0 --max-grad-norm 1.0  # DP-SGD everywhere

Each cell runs FedAvg with T local SGD steps per round and pooled SGD at the
same lr, both for the same number of SGD steps (`--epochs` pooled-equivalent
epochs, see local_steps.py), from the same initial model. The cell shows
FedAvg's final test score minus pooled's, averaged over seeds.

How far a hospital drifts toward its own data between averages grows with
lr × T, so the gap should be ~0 in the bottom-left (small lr, small T) and open
up toward the top-right. Very large lr can make pooled itself unstable; the
pooled row in the table shows when that happens. A blue (positive) cell at
high lr and small T is not drift: averaging 4 hospitals' steps is like pooled
SGD with a 16-patient batch, which is less noisy than pooled's batch of 4.

With --dp both FedAvg and pooled train with DP-SGD (fltab/privacy.py) and the
outputs get a `_dp_s<σ>_c<C>` suffix.

Writes:
    results/lr_T_grid.csv           final scores per lr, T, seed
    results/lr_T_grid_<metric>.png  heatmap of FedAvg − pooled
    results/lr_T_grid.md            pooled scores per lr and the gap table
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fltab.harness import (METRICS, ROOT, all_patients_test, evaluate, fedavg_steps,
                           pooled_steps, steps_per_pooled_epoch)
from fltab.privacy import add_privacy_args, privacy_from_args

p = argparse.ArgumentParser()
p.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
p.add_argument("--Ts", type=int, nargs="+", default=[1, 5, 25, 100, 500])
p.add_argument("--epochs", type=int, default=20, help="training budget, pooled-equivalent epochs")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
add_privacy_args(p)
args = p.parse_args()
PRIV = privacy_from_args(args)
TAG = PRIV.tag() if PRIV else ""
DESC = f", {PRIV.describe()}" if PRIV else ""

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
TOTAL = args.epochs * steps_per_pooled_epoch()
TEST = all_patients_test()
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "auc": "AUC"}

rows = []
for lr in args.lrs:
    for seed in args.seeds:
        model, eps = pooled_steps(seed, lr, TOTAL, privacy=PRIV)
        rows.append({"method": "pooled", "lr": lr, "T": 0, "seed": seed,
                     **evaluate(model, TEST), "epsilon": max(eps) if PRIV else None})
        for T in args.Ts:
            model, eps = fedavg_steps(seed, T, lr, TOTAL, privacy=PRIV)
            rows.append({"method": "fedavg", "lr": lr, "T": T, "seed": seed,
                         **evaluate(model, TEST), "epsilon": max(eps) if PRIV else None})
    print(f"lr {lr} done")

with open(OUT / f"lr_T_grid{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

df = pd.DataFrame(rows)
pooled = df[df["method"] == "pooled"].set_index(["lr", "seed"])[METRICS]
fed = df[df["method"] == "fedavg"].set_index(["lr", "seed"])
gap = fed[METRICS].sub(pooled.reindex(fed.index)).assign(T=fed["T"]).reset_index()
gap_mean = gap.groupby(["lr", "T"])[METRICS].mean()   # paired: same seed, same lr
Ts, lrs = sorted(args.Ts), sorted(args.lrs)

for k in METRICS:
    grid = gap_mean[k].unstack("T").reindex(index=lrs, columns=Ts).to_numpy()
    lim = max(0.01, np.nanmax(np.abs(grid)))
    fig, ax = plt.subplots(figsize=(1.2 * len(Ts) + 2.5, 0.6 * len(lrs) + 2))
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    for i in range(len(lrs)):
        for j in range(len(Ts)):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(Ts)), Ts)
    ax.set_yticks(range(len(lrs)), lrs)
    ax.set_xlabel("T, local SGD steps per round")
    ax.set_ylabel("learning rate")
    ax.set_title(f"FedAvg − pooled, final test {LABELS[k]}\n"
                 f"{args.epochs} pooled-equivalent epochs ({TOTAL} steps), "
                 f"{len(args.seeds)} seed(s){DESC}", fontsize=9)
    fig.colorbar(im, ax=ax, label="red: FedAvg worse, blue: better")
    fig.tight_layout()
    fig.savefig(OUT / f"lr_T_grid{TAG}_{k}.png", dpi=150)
    plt.close(fig)

lines = [f"## FedAvg − pooled over learning rate × T ({args.epochs} pooled-equivalent "
         f"epochs{DESC}, seeds {', '.join(map(str, args.seeds))})", "",
         "Pooled final test scores per lr (if these drop at high lr, pooled itself is "
         "unstable there and the gap below means less):", "",
         "| lr | " + " | ".join(LABELS[k] for k in METRICS) + " |",
         "|---|" + "---|" * len(METRICS)]
if PRIV:
    eps = df.groupby("method")["epsilon"].max()
    lines[1:1] = ["", f"ε at δ = {PRIV.delta:g} (worst cell; it depends on σ, C and the "
                  f"number of steps, not on lr, and T only through rounding up the last round): pooled {eps['pooled']:.2f}, "
                  f"FedAvg worst hospital {eps['fedavg']:.2f}."]
pm = pooled.groupby("lr").mean()
for lr in lrs:
    lines.append(f"| {lr} | " + " | ".join(f"{pm.loc[lr, k]:.3f}" for k in METRICS) + " |")
for k in METRICS:
    lines += ["", f"FedAvg − pooled, {LABELS[k]} (rows lr, columns T):", "",
              "| lr | " + " | ".join(f"T={T}" for T in Ts) + " |",
              "|---|" + "---|" * len(Ts)]
    for lr in lrs:
        lines.append(f"| {lr} | " + " | ".join(f"{gap_mean.loc[(lr, T), k]:+.3f}"
                                               for T in Ts) + " |")
(OUT / f"lr_T_grid{TAG}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
