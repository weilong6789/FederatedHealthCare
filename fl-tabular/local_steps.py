"""FedAvg with T local SGD steps per round, against pooled training.

    python local_steps.py                  # T = 1 2 5 10 25 50 100 250, 50 epochs, seeds 0-4
    python local_steps.py --Ts 1 10 100 --epochs 20 --seeds 0 1

Each round, every hospital starts from the global model, takes T minibatch SGD
steps on its own data, and the server averages the results weighted by
hospital size (as Flower's FedAvg does). With T = 1 that average is one SGD
step on a gradient averaged over all hospitals, i.e. pooled SGD with a bigger
batch. As T grows, each hospital drifts toward its own data between averages,
but the server needs fewer rounds (less communication) for the same training.

Every run gets the same training budget, counted in pooled-equivalent epochs:
one epoch = the number of SGD steps one pass over the pooled training data
takes (486 patients / batch 4 = 122 steps). FedAvg with T local steps runs
budget / T rounds, so every T and pooled take the same number of SGD steps on
the model. Same lr for every run, so T is the only thing that changes.

Writes:
    results/local_steps.csv               test scores along the way
    results/local_steps_curve_<metric>.png   score vs epochs, one line per T
    results/local_steps_tradeoff.png      final score and rounds needed vs T
    results/local_steps.md                the table behind the tradeoff plot

Scores are on all 254 test patients together (the "pooled" test set), so one
patient counts the same at every hospital.
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fltab.harness import (METRICS, ROOT, eval_loaders, evaluate, fedavg_steps,
                           pooled_steps, steps_per_pooled_epoch)

p = argparse.ArgumentParser()
p.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 250],
               help="local SGD steps per round")
p.add_argument("--epochs", type=int, default=50, help="training budget, pooled-equivalent epochs")
p.add_argument("--lr", type=float, default=0.003, help="same for every run (tuned pooled/FedAvg lr)")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--eval-every", type=float, default=1.0,
               help="evaluate about this often, in epochs")
args = p.parse_args()

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
STEPS_PER_EPOCH = steps_per_pooled_epoch()
TOTAL = args.epochs * STEPS_PER_EPOCH
EVAL_EVERY = max(1, round(args.eval_every * STEPS_PER_EPOCH))
TEST = eval_loaders("test")["pooled"]
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "auc": "AUC"}

rows = []


def recorder(method, T, seed):
    def record(round_, steps, model):
        rows.append({"method": method, "T": T, "seed": seed, "round": round_,
                     "steps": steps, "epochs": steps / STEPS_PER_EPOCH,
                     **evaluate(model, TEST)})
    return record


for seed in args.seeds:
    pooled_steps(seed, args.lr, TOTAL, EVAL_EVERY, recorder("pooled", 0, seed))
    for T in args.Ts:
        fedavg_steps(seed, T, args.lr, TOTAL, EVAL_EVERY, recorder("fedavg", T, seed))
    print(f"seed {seed} done")

with open(OUT / "local_steps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- plots
df = pd.DataFrame(rows)
df["run"] = np.where(df["method"] == "pooled", "pooled", "T=" + df["T"].astype(str))
Ts = sorted(args.Ts)
cmap = plt.get_cmap("viridis")
color = {T: cmap(i / max(1, len(Ts) - 1)) for i, T in enumerate(Ts)}


def curve(g, k):
    """Mean and std over seeds at each evaluated epoch. Large T is evaluated at
    round ends only, so epochs are binned to the nearest evaluation point."""
    g = g.assign(x=(g["epochs"] / args.eval_every).round() * args.eval_every)
    s = g.groupby("x")[k]
    return s.mean().index.to_numpy(), s.mean().to_numpy(), s.std().fillna(0).to_numpy()


for k in METRICS:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for T in Ts:
        x, m, s = curve(df[(df["method"] == "fedavg") & (df["T"] == T)], k)
        ax.plot(x, m, color=color[T], label=f"FedAvg, T={T}")
        ax.fill_between(x, m - s, m + s, color=color[T], alpha=0.12)
    x, m, s = curve(df[df["method"] == "pooled"], k)
    ax.plot(x, m, "k--", linewidth=2, label="pooled")
    ax.set_xlabel(f"total training, pooled-equivalent epochs ({STEPS_PER_EPOCH} SGD steps each)")
    ax.set_ylabel(f"test {LABELS[k]}, all 254 test patients")
    ax.set_title(f"FedAvg with T local steps per round vs pooled (lr {args.lr})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"local_steps_curve_{k}.png", dpi=150)
    plt.close(fig)

    # Same runs against communication: how many rounds each T needs to get there
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for T in Ts:
        g = df[(df["method"] == "fedavg") & (df["T"] == T) & (df["round"] > 0)]
        s = g.groupby("round")[k]
        ax.plot(s.mean().index, s.mean(), color=color[T], marker=".", label=f"FedAvg, T={T}")
    ax.axhline(df[df["method"] == "pooled"].groupby("seed").last()[k].mean(),
               color="k", linestyle="--", label="pooled, final")
    ax.set_xscale("log")
    ax.set_xlabel("communication rounds")
    ax.set_ylabel(f"test {LABELS[k]}, all 254 test patients")
    ax.set_title(f"Score vs communication rounds (lr {args.lr})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"local_steps_rounds_{k}.png", dpi=150)
    plt.close(fig)

final = df[df["steps"] == df.groupby(["run", "seed"])["steps"].transform("max")]
agg = final.groupby(["method", "T"])[METRICS].agg(["mean", "std"])
pooled_final = agg.loc[("pooled", 0)]

fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4 * (len(METRICS) + 1), 4))
for ax, k in zip(axes, METRICS):
    m = [agg.loc[("fedavg", T), (k, "mean")] for T in Ts]
    s = [agg.loc[("fedavg", T), (k, "std")] for T in Ts]
    ax.errorbar(Ts, m, yerr=s, marker="o", capsize=3, label="FedAvg")
    ax.axhline(pooled_final[(k, "mean")], color="k", linestyle="--", label="pooled")
    ax.set_xscale("log")
    ax.set_xlabel("T, local SGD steps per round")
    ax.set_ylabel(f"final test {LABELS[k]}")
    ax.legend(fontsize=8)
ax = axes[-1]
ax.plot(Ts, [math.ceil(TOTAL / T) for T in Ts], marker="o")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("T, local SGD steps per round")
ax.set_ylabel("communication rounds needed")
fig.suptitle(f"Same budget for every T: {args.epochs} pooled-equivalent epochs "
             f"({TOTAL} SGD steps), lr {args.lr}")
fig.tight_layout()
fig.savefig(OUT / "local_steps_tradeoff.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- table
lines = [f"## FedAvg with T local steps vs pooled ({args.epochs} pooled-equivalent epochs, "
         f"lr {args.lr}, seeds {', '.join(map(str, args.seeds))})", "",
         "Final test scores on all 254 test patients, mean ± std over seeds. Every row "
         f"takes the same {TOTAL} SGD steps; FedAvg spreads them over rounds × T.", "",
         "| run | rounds | " + " | ".join(LABELS[k] for k in METRICS) + " |",
         "|---|---|" + "---|" * len(METRICS)]
cell = lambda r, k: f"{agg.loc[r, (k, 'mean')]:.3f} ± {agg.loc[r, (k, 'std')]:.3f}"
lines.append("| pooled | - | " + " | ".join(cell(("pooled", 0), k) for k in METRICS) + " |")
for T in Ts:
    lines.append(f"| FedAvg T={T} | {math.ceil(TOTAL / T)} | "
                 + " | ".join(cell(("fedavg", T), k) for k in METRICS) + " |")
(OUT / "local_steps.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
