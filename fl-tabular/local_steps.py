"""FedAvg with T local SGD steps per round, against pooled training.

    python local_steps.py                  # T from 1 to the whole budget, 20 epochs, seeds 0-4
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

By default the sweep runs between two endpoints: T = 1 (one step per round,
as many rounds as steps) and T = the whole budget (a single round: every
hospital trains alone for all of it, and the server averages once at the end).
In between are about a dozen T values spaced evenly on a log scale, each a
divisor of the budget so every run takes exactly the same number of steps.

Writes:
    results/local_steps.csv               test scores along the way
    results/local_steps_curve_<metric>.png   score vs epochs, one line per T
    results/local_steps_tradeoff.png      final score and rounds needed vs T
    results/local_steps.md                the table behind the tradeoff plot

Scores are on all 254 test patients together, so one patient counts the same
at every hospital. Pooled and FedAvg both train and test on features scaled
per hospital (FLamby's per-center scaling), so they see the same inputs; see
pooled_steps in fltab/harness.py.
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fltab.harness import (METRICS, NUM_CLIENTS, ROOT, all_patients_test, eval_loaders,
                           evaluate, fedavg_steps, pooled_steps, steps_per_pooled_epoch)

p = argparse.ArgumentParser()
p.add_argument("--Ts", type=int, nargs="+",
               help="local SGD steps per round (default: --n-T values from 1 to the whole budget)")
p.add_argument("--n-T", type=int, default=12, help="how many T values in the default sweep")
p.add_argument("--epochs", type=int, default=20,
               help="training budget, pooled-equivalent epochs (pooled peaks around 10-20)")
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


def default_Ts(total, n):
    """About n divisors of `total`, log-spaced from 1 (total rounds) to total (1 round)."""
    divisors = [d for d in range(1, total + 1) if total % d == 0]
    picks = [min(divisors, key=lambda d: abs(math.log(d / t)))
             for t in np.logspace(0, math.log10(total), n)]
    return sorted(set(picks))


if args.Ts is None:
    args.Ts = default_Ts(TOTAL, args.n_T)
uneven = [T for T in args.Ts if TOTAL % T]
if uneven:
    print(f"note: T = {uneven} don't divide {TOTAL}; those runs take ceil({TOTAL}/T) "
          f"rounds, so slightly more steps than pooled")
print(f"budget {TOTAL} steps; T = {args.Ts}; rounds = {[math.ceil(TOTAL / T) for T in args.Ts]}")
TEST = all_patients_test()
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "auc": "AUC"}
# Each hospital's own test set, scaled with its own statistics like everything else here
HOSPITALS = ["Cleveland", "Hungary", "Switzerland", "Long Beach VA"]
HOSP_TEST = {str(i): eval_loaders("test")[str(i)] for i in range(NUM_CLIENTS)}

rows = []
hosp_rows = []        # final model only, scored at each hospital


def recorder(method, T, seed):
    def record(round_, steps, model):
        rows.append({"method": method, "T": T, "seed": seed, "round": round_,
                     "steps": steps, "epochs": steps / STEPS_PER_EPOCH,
                     **evaluate(model, TEST)})
        if steps >= TOTAL:
            for c, dl in HOSP_TEST.items():
                hosp_rows.append({"method": method, "T": T, "seed": seed, "client": c,
                                  "hospital": HOSPITALS[int(c)], **evaluate(model, dl)})
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
with open(OUT / "local_steps_hospitals.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hosp_rows[0].keys())
    w.writeheader()
    w.writerows(hosp_rows)

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
# ---------------------------------------------------------------- per hospital
# Does a large T leave one hospital behind even when the overall score holds?
hdf = pd.DataFrame(hosp_rows)
hagg = hdf.groupby(["hospital", "method", "T"])[METRICS].mean()
names = [h for h in HOSPITALS if h in hdf["hospital"].unique()]
hcolor = {h: plt.get_cmap("tab10")(i) for i, h in enumerate(names)}

fig, axes = plt.subplots(1, len(METRICS), figsize=(4.5 * len(METRICS), 4))
for ax, k in zip(axes, METRICS):
    for h in names:
        ax.plot(Ts, [hagg.loc[(h, "fedavg", T), k] for T in Ts], marker="o",
                color=hcolor[h], label=h)
        ax.axhline(hagg.loc[(h, "pooled", 0), k], color=hcolor[h], linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("T, local SGD steps per round")
    ax.set_ylabel(f"final test {LABELS[k]}, that hospital")
    ax.set_title(LABELS[k])
axes[0].legend(fontsize=8)
fig.suptitle(f"Each hospital's own test set: FedAvg (lines) vs pooled (dashed), "
             f"lr {args.lr}, mean of {len(args.seeds)} seeds")
fig.tight_layout()
fig.savefig(OUT / "local_steps_hospitals.png", dpi=150)
plt.close(fig)

for k in METRICS:
    lines += ["", f"Final test {LABELS[k]} at each hospital, mean over seeds:", "",
              "| hospital | pooled | " + " | ".join(f"T={T}" for T in Ts) + " |",
              "|---|---|" + "---|" * len(Ts)]
    for h in names:
        lines.append(f"| {h} | {hagg.loc[(h, 'pooled', 0), k]:.3f} | "
                     + " | ".join(f"{hagg.loc[(h, 'fedavg', T), k]:.3f}" for T in Ts) + " |")
lines += ["", "Switzerland has 16 test patients, one of them label 0, so its accuracy moves "
          "in steps of 0.062 and its AUC rests on that one patient. Long Beach VA has 45."]
(OUT / "local_steps.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
