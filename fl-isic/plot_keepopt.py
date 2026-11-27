"""Fresh Adam every round vs each hospital keeping its Adam state, ISIC.

    python plot_keepopt.py      # reads results/ and results_keepopt/, writes results_keepopt/keepopt_vs_fresh.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

POOLED, FRESH, KEEP = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "figure.facecolor": "white"})


def final(folder):
    d = pd.read_csv(Path(folder) / "isic.csv", dtype=str)
    d = d[d.method.isin(["pooled", "fedavg"]) & (d.client == "all")].copy()
    for c in ["T", "seed", "steps", "balanced_accuracy"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["seed", "steps", "balanced_accuracy"])
    return d.loc[d.groupby(["method", "T", "seed"], dropna=False).steps.idxmax()]


fresh, keep = final("results"), final("results_keepopt")
pooled = fresh[fresh.method == "pooled"].balanced_accuracy
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axhspan(pooled.mean() - pooled.std(), pooled.mean() + pooled.std(), color=POOLED, alpha=0.12, lw=0)
ax.axhline(pooled.mean(), color=POOLED, ls="--", lw=1.5, label="pooled")
for d, color, name in [(fresh, FRESH, "FedAvg, fresh Adam every round"),
                       (keep, KEEP, "FedAvg, each hospital keeps its Adam")]:
    g = d[(d.method == "fedavg") & (d["T"] != 100)]
    ax.scatter(g["T"], g.balanced_accuracy, s=14, color=color, alpha=0.35, lw=0)
    m = g.groupby("T").balanced_accuracy.mean()
    ax.plot(m.index, m, color=color, lw=2, marker="o", ms=6, label=name)
ax.set_xscale("log")
ax.set_xlabel("local steps per round, K (log scale)")
ax.set_ylabel("balanced accuracy")
ax.set_title("ISIC, all 4650 test images, 5820 steps per run")
ax.grid(alpha=0.25, lw=0.6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="lower center")
fig.text(0.01, 0.01, "Dots: each seed. Lines: mean over 3 seeds. Pooled band ± 1 std.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("results_keepopt/keepopt_vs_fresh.png", dpi=200)
