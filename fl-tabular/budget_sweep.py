"""How small can the training budget get before scores drop? Same setup as
local_steps.py (FedAvg with T local SGD steps per round, from T = 1 to a
single round, against pooled), repeated at shrinking budgets.

    python budget_sweep.py                       # 2440 steps (20 epochs) down to 2, seeds 0-4
    python budget_sweep.py --budgets 2440 305 38 5 --seeds 0 1

local_steps.py found that at 20 pooled-equivalent epochs (2440 SGD steps) T
makes no difference: every T lands on pooled's score. Here the budget B
(total SGD steps, counted as in local_steps.py) goes from 2440 down to 2
in factors of about sqrt(2).
At each B, FedAvg runs with about --n-T values of T from 1 (B rounds) to B
(one round), each a divisor of B, plus pooled with B steps. Same lr for
everything.

A score counts as a drop when it is below the reference, pooled at the full
2440 steps, by at least --tol (default 0.01) and significantly so, with a paired
bootstrap over the 254 test patients: the same resampled patients score both
models (each the mean over seeds), and the 95% interval of the difference has
to lie below 0. Paired differences between near-identical models are tiny, so
without --tol a 0.002 dip can come out significant. The
"break point" for a run type is the smallest budget at which it, and every
larger budget, still holds the reference.

Writes:
    results/budget_sweep.csv              final test scores, one row per run and seed
    results/budget_sweep_predictions.csv  per-patient test probabilities, final models
    results/budget_sweep_<metric>.png     final score vs budget, one line per T position
    results/budget_sweep_heatmap.png      drop vs reference over budget x T
    results/budget_sweep.md               tables, bootstrap intervals and break points
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fltab.harness import (METRICS, ROOT, all_patients_test, evaluate, fedavg_steps,
                           pooled_steps, predict, steps_per_pooled_epoch)

p = argparse.ArgumentParser()
p.add_argument("--budgets", type=int, nargs="+",
               help="total SGD steps per run (default: 20 epochs down to 2 steps, factors of sqrt 2)")
p.add_argument("--n-T", type=int, default=5, help="how many T values per budget")
p.add_argument("--lr", type=float, default=0.003, help="same as local_steps.py")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--tol", type=float, default=0.01,
               help="smallest drop that counts, on top of significance (0 = significance alone)")
args = p.parse_args()

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SPE = steps_per_pooled_epoch()
FULL = 20 * SPE


def n_divisors(b):
    return sum(b % d == 0 for d in range(1, b + 1))


def default_budgets():
    """FULL down to 2 steps in factors of sqrt(2). Each is nudged (within 8%) to
    the number with the most divisors, so every budget gets a spread of T."""
    out = {FULL}
    for k in range(1, 21):
        t = FULL * 2 ** (-k / 2)
        near = range(max(2, math.floor(t * 0.92)), math.ceil(t * 1.08) + 1)
        out.add(max(near, key=lambda b: (n_divisors(b), -abs(b - t))))
    return sorted(out, reverse=True)


if args.budgets is None:
    args.budgets = default_budgets()
REF_B = max(args.budgets)
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "auc": "AUC"}


def default_Ts(total, n):
    """About n divisors of `total`, log-spaced from 1 (total rounds) to total (1 round)."""
    divisors = [d for d in range(1, total + 1) if total % d == 0]
    picks = [min(divisors, key=lambda d: abs(math.log(d / t)))
             for t in np.logspace(0, math.log10(total), n)]
    return sorted(set(picks))


GRID = {B: default_Ts(B, args.n_T) for B in args.budgets}
for B, Ts in GRID.items():
    print(f"budget {B} steps ({B / SPE:.3g} epochs): T = {Ts}")

TEST = all_patients_test()
rows, preds = [], []
for seed in args.seeds:
    for B, Ts in GRID.items():
        runs = [("pooled", 0, pooled_steps(seed, args.lr, B))]
        runs += [("fedavg", T, fedavg_steps(seed, T, args.lr, B)) for T in Ts]
        for method, T, model in runs:
            rows.append({"method": method, "T": T, "budget": B, "epochs": B / SPE,
                         "rounds": -(-B // T) if T else 0, "seed": seed,
                         **evaluate(model, TEST)})
            y, prob = predict(model, TEST)
            preds += [{"method": method, "T": T, "budget": B, "seed": seed,
                       "patient": i, "label": int(yi), "prob": float(pi)}
                      for i, (yi, pi) in enumerate(zip(y, prob))]
    print(f"seed {seed} done")

df = pd.DataFrame(rows)
pr = pd.DataFrame(preds)
df.to_csv(OUT / "budget_sweep.csv", index=False)
pr.to_csv(OUT / "budget_sweep_predictions.csv", index=False, float_format="%.6f")

# ---------------------------------------------------------------- bootstrap vs reference
# Metrics are computed per seed and then averaged, as in the tables, on the same
# resampled patients for the run and the reference.
key = ["method", "T", "budget"]
P = {k: g.pivot(index="seed", columns="patient", values="prob").to_numpy()
     for k, g in pr.groupby(key)}
y = pr[pr.seed == args.seeds[0]].drop_duplicates("patient").sort_values("patient").label.to_numpy()
rng = np.random.default_rng(0)
n = len(y)
idx = rng.integers(0, n, size=(args.n_boot, n))
idx = idx[[len(np.unique(y[i])) == 2 for i in idx]]


def scores(probs, yy):
    """Mean over seeds of each metric, `probs` is (seeds, patients)."""
    hard = probs > 0.5
    acc = (hard == yy).mean(1)
    tpr = (hard & (yy == 1)).sum(1) / (yy == 1).sum()
    tnr = (~hard & (yy == 0)).sum(1) / (yy == 0).sum()
    auc = [roc_auc_score(yy, pp) for pp in probs]
    return {"accuracy": acc.mean(), "balanced_accuracy": ((tpr + tnr) / 2).mean(),
            "auc": float(np.mean(auc))}


REF = ("pooled", 0, REF_B)
ref_boot = [scores(P[REF][:, i], y[i]) for i in idx]
ref_point = scores(P[REF], y)
stats = []
for k, probs in P.items():
    point = scores(probs, y)
    boot = ref_boot if k == REF else [scores(probs[:, i], y[i]) for i in idx]
    rec = dict(zip(key, k))
    for m in METRICS:
        d = np.array([b[m] - r[m] for b, r in zip(boot, ref_boot)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rec |= {m: point[m], f"{m}_diff": point[m] - ref_point[m],
                f"{m}_lo": lo, f"{m}_hi": hi,
                f"{m}_drop": bool(hi < 0 and point[m] - ref_point[m] <= -args.tol)}
    stats.append(rec)
st = pd.DataFrame(stats).sort_values(["budget", "method", "T"], ascending=[False, False, True])
spread = df.groupby(key)[METRICS].std()


def position(T, B):
    """Which end of the T range a run sits at, so budgets line up in plots."""
    if T == 0:
        return "pooled"
    if T == 1:
        return "FedAvg T=1"
    if T == B:
        return "FedAvg 1 round"
    return None


st["pos"] = [position(T, B) for T, B in zip(st["T"], st["budget"])]


def break_point(g, m):
    """Smallest budget at which g, and every larger budget, has no drop."""
    ok = g.sort_values("budget", ascending=False)
    keep = None
    for _, r in ok.iterrows():
        if r[f"{m}_drop"]:
            break
        keep = r["budget"]
    return keep


# ---------------------------------------------------------------- plots
Bs = sorted(args.budgets)
style = {"pooled": dict(color="k", ls="--", marker="s"),
         "FedAvg T=1": dict(color="#1f77b4", ls="-", marker="o"),
         "FedAvg 1 round": dict(color="#d62728", ls="-", marker="^")}
for m in METRICS:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pos, sty in style.items():
        g = st[st.pos == pos].sort_values("budget")
        ax.plot(g.budget, g[m], label=pos, **{k: v for k, v in sty.items() if k != "marker"})
        for _, r in g.iterrows():
            ax.plot(r.budget, r[m], sty["marker"], color=sty["color"], ms=7,
                    mfc="white" if r[f"{m}_drop"] else sty["color"])
    mid = st[st.pos.isna()]
    ax.scatter(mid.budget, mid[m], s=14, color="grey", alpha=0.6, zorder=1,
               label="FedAvg, T in between")
    ax.axhline(ref_point[m], color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(f"training budget, total SGD steps ({SPE} = 1 pooled epoch)")
    ax.set_ylabel(f"final test {LABELS[m]}, all 254 test patients")
    ax.set_title(f"Shrinking the budget (lr {args.lr}, mean of {len(args.seeds)} seeds)\n"
                 f"hollow = at least {args.tol} and significantly below pooled at {REF_B} steps")
    top = ax.secondary_xaxis("top", functions=(lambda s: s / SPE, lambda e: e * SPE))
    top.set_xlabel("pooled-equivalent epochs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"budget_sweep_{m}.png", dpi=150)
    plt.close(fig)

# drop vs reference over budget x (T as a fraction of the budget)
fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.2))
fed = st[st.method == "fedavg"]
for ax, m in zip(axes, METRICS):
    sc = ax.scatter(fed.budget, fed["T"] / fed.budget, c=fed[f"{m}_diff"], cmap="RdBu",
                    vmin=-0.15, vmax=0.15, s=90, edgecolors=np.where(fed[f"{m}_drop"], "k", "none"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget, total SGD steps")
    ax.set_ylabel("T / budget  (1 = one round)")
    ax.set_title(f"{LABELS[m]} minus pooled at {REF_B} steps")
    fig.colorbar(sc, ax=ax)
fig.suptitle(f"FedAvg, black ring = drop (at least {args.tol}, significant)")
fig.tight_layout()
fig.savefig(OUT / "budget_sweep_heatmap.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- tables
lines = [f"## Shrinking the training budget (lr {args.lr}, seeds {', '.join(map(str, args.seeds))})",
         "",
         f"Reference: pooled at {REF_B} steps ({REF_B / SPE:.3g} epochs): "
         + ", ".join(f"{LABELS[m]} {ref_point[m]:.3f}" for m in METRICS) + ".",
         f"Δ is the run minus the reference (both mean over seeds); [lo, hi] is the 95% paired "
         f"bootstrap interval over the 254 test patients ({len(idx)} resamples). "
         f"**Bold** = drop: Δ ≤ -{args.tol} and hi < 0.", "",
         "### Break points", "",
         "Smallest budget where the run, and every larger budget, has no drop.", "",
         "| run | " + " | ".join(LABELS[m] for m in METRICS) + " |",
         "|---|" + "---|" * len(METRICS)]
for pos in style:
    g = st[st.pos == pos]
    cells = []
    for m in METRICS:
        b = break_point(g, m)
        cells.append("never holds" if b is None else f"{int(b)} steps ({b / SPE:.2g} ep)")
    lines.append(f"| {pos} | " + " | ".join(cells) + " |")
fed_all = st[st.method == "fedavg"]
cells = []
for m in METRICS:
    worst = fed_all.groupby("budget")[f"{m}_drop"].any().reset_index()
    b = break_point(worst, m)
    cells.append("never holds" if b is None else f"{int(b)} steps ({b / SPE:.2g} ep)")
lines.append("| every FedAvg T at that budget | " + " | ".join(cells) + " |")

lines += ["", "### All runs", "",
          "| budget (steps) | epochs | run | rounds | "
          + " | ".join(f"{LABELS[m]} (Δ [lo, hi])" for m in METRICS) + " |",
          "|---|---|---|---|" + "---|" * len(METRICS)]
for _, r in st.iterrows():
    name = "pooled" if r.method == "pooled" else f"FedAvg T={r['T']}"
    rounds = "-" if r.method == "pooled" else str(-(-r.budget // r["T"]))
    cells = []
    for m in METRICS:
        c = (f"{r[m]:.3f} ± {spread.loc[(r.method, r['T'], r.budget), m]:.3f} "
             f"({r[f'{m}_diff']:+.3f} [{r[f'{m}_lo']:+.3f}, {r[f'{m}_hi']:+.3f}])")
        cells.append(f"**{c}**" if r[f"{m}_drop"] else c)
    lines.append(f"| {r.budget} | {r.budget / SPE:.3g} | {name} | {rounds} | " + " | ".join(cells) + " |")
lines += ["", "± is the std over seeds."]
(OUT / "budget_sweep.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
