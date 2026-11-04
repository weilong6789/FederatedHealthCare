"""Communication cost before convergence: FedAvg with T local steps, short budgets.

    python comm_budget.py                   # lr 0.003-0.1 × T 1-250, seeds 0-4 (~20 min)
    python comm_budget.py --lrs 0.01 --Ts 1 10 100 --seeds 0    # quick look

local_steps.py gives every run 6100 SGD steps (50 pooled-equivalent epochs).
Pooled reaches its best test AUC by about 10 epochs at lr 0.003, so by the end
every T has converged to the same model and the final scores say nothing about
how much communication it took to get there. This script looks at the part
before convergence instead:

1. Score vs communication rounds, one line per T, for each lr. Runs go to
   `--epochs` pooled-equivalent epochs or `--min-rounds` rounds, whichever is
   more SGD steps, and are scored after every round.
2. Rounds needed to get within `--tol` of pooled's converged score (pooled's
   score after `--epochs` epochs at the same lr, mean over seeds).
3. FedAvg minus pooled at the same number of SGD steps, for short budgets
   (`--budgets`, in pooled-equivalent epochs) where neither has converged. This
   is where client drift would show if it slowed training down.

One round = every hospital downloads the global model and uploads its update:
14 float32 parameters each way, 4 hospitals, so 448 bytes per round.

Writes:
    results/comm_budget.csv                    every evaluation (method, lr, T, seed, round, steps)
    results/comm_rounds_<metric>.png           score vs rounds per lr, pooled converged dashed
    results/comm_steps_<metric>.png            score vs SGD steps, FedAvg vs pooled, short budgets
    results/comm_rounds_to_target.png          rounds to reach pooled's converged score vs T
    results/comm_equal_steps_<metric>.png      FedAvg − pooled at equal steps, short budgets
    results/comm_budget.md                     the tables behind the plots

Same data, model, scaling and test set (all 254 test patients, scaled per
hospital) as local_steps.py.
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

p = argparse.ArgumentParser()
p.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
p.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 250])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--epochs", type=float, default=20,
               help="run length and pooled's 'converged' point, pooled-equivalent epochs")
p.add_argument("--min-rounds", type=int, default=30, help="every T runs at least this many rounds")
p.add_argument("--tol", type=float, default=0.01, help="'reached' = within this of pooled converged")
p.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4],
               help="short budgets for the equal-steps comparison, pooled-equivalent epochs")
args = p.parse_args()

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SPE = steps_per_pooled_epoch()
RUN_STEPS = round(args.epochs * SPE)
TEST = all_patients_test()
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "auc": "AUC"}
BYTES_PER_ROUND = 2 * 14 * 4 * 4    # up + down, 14 float32 params, 4 hospitals
Ts, lrs = sorted(args.Ts), sorted(args.lrs)

rows = []


def recorder(method, lr, T, seed):
    def record(round_, steps, model):
        rows.append({"method": method, "lr": lr, "T": T, "seed": seed, "round": round_,
                     "steps": steps, **evaluate(model, TEST)})
    return record


for lr in lrs:
    for seed in args.seeds:
        pooled_steps(seed, lr, RUN_STEPS, 1, recorder("pooled", lr, 0, seed))
        for T in Ts:
            total = max(RUN_STEPS, args.min_rounds * T)
            fedavg_steps(seed, T, lr, total, 1, recorder("fedavg", lr, T, seed))
    print(f"lr {lr} done")

with open(OUT / "comm_budget.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

df = pd.DataFrame(rows)
pooled = df[df["method"] == "pooled"]
fed = df[df["method"] == "fedavg"]
# Pooled's converged score per lr: after RUN_STEPS steps, mean over seeds
conv = pooled[pooled["steps"] == RUN_STEPS].groupby("lr")[METRICS].mean()
fed_mean = fed.groupby(["lr", "T", "round"])[METRICS].mean()
cmap = plt.get_cmap("viridis")
color = {T: cmap(i / max(1, len(Ts) - 1)) for i, T in enumerate(Ts)}


def panels():
    n = len(lrs)
    cols = min(n, 2)
    rows_ = -(-n // cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(6.5 * cols, 4.3 * rows_), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


# ---------------------------------------------------------------- score vs rounds
for k in METRICS:
    fig, axes = panels()
    for ax, lr in zip(axes, lrs):
        for T in Ts:
            g = fed_mean.loc[(lr, T)][k]
            g = g[g.index > 0]
            ax.plot(g.index, g.to_numpy(), color=color[T], label=f"FedAvg, T={T}")
        c = conv.loc[lr, k]
        ax.axhline(c, color="k", linestyle="--", label=f"pooled, {args.epochs:g} epochs")
        ax.axhspan(c - args.tol, c + args.tol, color="k", alpha=0.07)
        ax.set_xscale("log")
        ax.set_xlabel(f"communication rounds ({BYTES_PER_ROUND} bytes each)")
        ax.set_ylabel(f"test {LABELS[k]}")
        ax.set_title(f"lr {lr}")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Score vs communication rounds, mean of {len(args.seeds)} seeds "
                 f"(band: within {args.tol} of pooled)")
    fig.tight_layout()
    fig.savefig(OUT / f"comm_rounds_{k}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- score vs steps, short budgets
max_b = round(max(args.budgets) * SPE)
for k in METRICS:
    fig, axes = panels()
    for ax, lr in zip(axes, lrs):
        for T in Ts:
            g = fed[(fed["lr"] == lr) & (fed["T"] == T) & (fed["steps"] <= max_b)]
            s = g.groupby("steps")[k].mean()
            ax.plot(s.index / SPE, s.to_numpy(), color=color[T], marker="." if T > 25 else None,
                    label=f"FedAvg, T={T}")
        s = pooled[(pooled["lr"] == lr) & (pooled["steps"] <= max_b)].groupby("steps")[k].mean()
        ax.plot(s.index / SPE, s.to_numpy(), "k--", linewidth=2, label="pooled")
        ax.axhline(conv.loc[lr, k], color="gray", linestyle=":", label="pooled converged")
        ax.set_xlabel(f"SGD steps, pooled-equivalent epochs ({SPE} steps each)")
        ax.set_ylabel(f"test {LABELS[k]}")
        ax.set_title(f"lr {lr}")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Same SGD steps, short budgets: FedAvg vs pooled, mean of {len(args.seeds)} seeds")
    fig.tight_layout()
    fig.savefig(OUT / f"comm_steps_{k}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- rounds to target
def rounds_to(lr, T, k):
    g = fed_mean.loc[(lr, T)][k]
    hit = g[(g.index > 0) & (g >= conv.loc[lr, k] - args.tol)]
    return int(hit.index[0]) if len(hit) else np.nan


r2t = {k: pd.DataFrame([[rounds_to(lr, T, k) for T in Ts] for lr in lrs], index=lrs, columns=Ts)
       for k in METRICS}
fig, axes = plt.subplots(1, len(METRICS), figsize=(4.8 * len(METRICS), 4.2))
lcolor = {lr: plt.get_cmap("plasma")(i / max(1, len(lrs) - 0.5)) for i, lr in enumerate(lrs)}
for ax, k in zip(axes, METRICS):
    for lr in lrs:
        ax.plot(Ts, r2t[k].loc[lr], marker="o", color=lcolor[lr], label=f"lr {lr}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T, local SGD steps per round")
    ax.set_ylabel("rounds to reach target")
    ax.set_title(f"{LABELS[k]} within {args.tol} of pooled")
axes[0].legend(fontsize=8)
fig.suptitle("Communication rounds needed to match pooled's converged score "
             f"(mean of {len(args.seeds)} seeds; missing point = never reached)")
fig.tight_layout()
fig.savefig(OUT / "comm_rounds_to_target.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- equal steps, short budgets
# FedAvg after floor(B / T) rounds vs pooled after the same number of steps,
# paired by seed. T larger than the budget can't fit one round and is left out.
pk = pooled.set_index(["lr", "seed", "steps"])[METRICS]
fk = fed.set_index(["lr", "T", "seed", "round"])[METRICS]
gap = {}
for k in METRICS:
    for lr in lrs:
        for b in args.budgets:
            B = round(b * SPE)
            for T in Ts:
                R = B // T
                if R == 0 or R * T > RUN_STEPS:   # no pooled score that far
                    continue
                d = [fk.loc[(lr, T, s, R), k] - pk.loc[(lr, s, R * T), k] for s in args.seeds]
                gap[(k, lr, b, T)] = (np.mean(d), np.std(d))

for k in METRICS:
    fig, axes = panels()
    vals = [v[0] for key, v in gap.items() if key[0] == k]
    lim = max(0.01, np.nanmax(np.abs(vals)))
    for ax, lr in zip(axes, lrs):
        grid = np.array([[gap.get((k, lr, b, T), (np.nan,))[0] for T in Ts] for b in args.budgets])
        ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(len(args.budgets)):
            for j in range(len(Ts)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(Ts)), Ts)
        ax.set_yticks(range(len(args.budgets)),
                      [f"{b:g} ({round(b * SPE)})" for b in args.budgets])
        ax.set_xlabel("T, local SGD steps per round")
        ax.set_ylabel("budget, epochs (SGD steps)")
        ax.set_title(f"lr {lr}")
    fig.suptitle(f"FedAvg − pooled, test {LABELS[k]}, same SGD steps "
                 f"(blue: FedAvg ahead; blank: T > budget)")
    fig.tight_layout()
    fig.savefig(OUT / f"comm_equal_steps_{k}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- tables
fmt = lambda v: "-" if np.isnan(v) else f"{int(v)}"
lines = [f"## Communication cost before convergence (lr {', '.join(map(str, lrs))}; "
         f"seeds {', '.join(map(str, args.seeds))})", "",
         f"Test scores on all 254 test patients. One round = {BYTES_PER_ROUND} bytes "
         "(14 float32 parameters up and down, 4 hospitals).", "",
         f"Pooled converged = pooled after {args.epochs:g} pooled-equivalent epochs "
         f"({RUN_STEPS} SGD steps) at the same lr, mean over seeds:", "",
         "| lr | " + " | ".join(LABELS[k] for k in METRICS) + " |",
         "|---|" + "---|" * len(METRICS)]
for lr in lrs:
    lines.append(f"| {lr} | " + " | ".join(f"{conv.loc[lr, k]:.3f}" for k in METRICS) + " |")

for k in METRICS:
    lines += ["", f"Rounds until FedAvg's mean test {LABELS[k]} is within {args.tol} of pooled "
              "converged (`-` = never, within the run):", "",
              "| lr | " + " | ".join(f"T={T}" for T in Ts) + " |", "|---|" + "---|" * len(Ts)]
    for lr in lrs:
        lines.append(f"| {lr} | " + " | ".join(fmt(v) for v in r2t[k].loc[lr]) + " |")

R_SHOW = [1, 2, 5, 10, 25]
for lr in lrs:
    lines += ["", f"Test AUC after R rounds, lr {lr} (pooled converged {conv.loc[lr, 'auc']:.3f}), "
              "mean ± std over seeds:", "",
              "| T | " + " | ".join(f"R={R}" for R in R_SHOW) + " |", "|---|" + "---|" * len(R_SHOW)]
    for T in Ts:
        g = fed[(fed["lr"] == lr) & (fed["T"] == T)].groupby("round")["auc"]
        m, s = g.mean(), g.std()
        lines.append(f"| {T} | " + " | ".join(f"{m[R]:.3f} ± {s[R]:.3f}" if R in m.index else "-"
                                               for R in R_SHOW) + " |")

for k in METRICS:
    lines += ["", f"FedAvg − pooled, test {LABELS[k]}, same SGD steps, mean ± std over seeds "
              "(rows: budget in epochs; `-`: T larger than the budget):"]
    for lr in lrs:
        lines += ["", f"lr {lr}:", "", "| budget | " + " | ".join(f"T={T}" for T in Ts) + " |",
                  "|---|" + "---|" * len(Ts)]
        for b in args.budgets:
            cells = [f"{gap[(k, lr, b, T)][0]:+.3f} ± {gap[(k, lr, b, T)][1]:.3f}"
                     if (k, lr, b, T) in gap else "-" for T in Ts]
            lines.append(f"| {b:g} ({round(b * SPE)} steps) | " + " | ".join(cells) + " |")
(OUT / "comm_budget.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
