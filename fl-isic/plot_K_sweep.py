"""Figures for the FedAvg local-steps (K) sweep on ISIC.

    python plot_K_sweep.py              # reads and writes results/
    python plot_K_sweep.py results40    # another results folder

Only runs that finished (wrote final predictions) are used, so a job that hit
its time limit doesn't drag down its K. Writes:
    K_final.png       final balanced accuracy, accuracy and AUC vs K, against pooled
    K_curves.png      balanced accuracy vs epochs, one panel per K, pooled for reference
    K_hospitals.png   final balanced accuracy at each hospital vs K, against pooled and local
    K_classes.png     recall of each of the 8 classes, pooled and every K
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore", message="y_pred contains classes")

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
METRICS = {"balanced_accuracy": "balanced accuracy", "accuracy": "accuracy", "auc": "AUC"}
CLASSES = ["melanoma", "nevus", "basal cell carcinoma", "actinic keratosis",
           "benign keratosis", "dermatofibroma", "vascular lesion", "squamous cell carcinoma"]
POOLED, FEDAVG, LOCAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "figure.facecolor": "white"})


def tidy(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- data
d = pd.read_csv(OUT / "isic.csv", dtype=str)
d = d[d.method.isin(["pooled", "fedavg", "local"])].copy()
for c in ["K", "seed", "round", "steps", "epochs", *METRICS]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["seed", "steps", "epochs", *METRICS])
d["seed"] = d.seed.astype(int)
d["tag"] = np.where(d.method == "fedavg", "fedavg_K" + d["K"].fillna(0).astype(int).astype(str),
                    d.method)
d = d.drop_duplicates(["tag", "seed", "steps", "client"], keep="last")

pred = pd.read_csv(OUT / "isic_predictions.csv", dtype=str)
pred = pred[pred.method != "method"].copy()
P = [f"p{k}" for k in range(8)]
pred[P] = pred[P].astype(float)
pred[["seed", "image", "label"]] = pred[["seed", "image", "label"]].astype(int)
pred = pred.drop_duplicates(["method", "seed", "client", "image"], keep="last")
pred["pred"] = pred[P].to_numpy().argmax(1)

finished = set(pred.groupby(["method", "seed"]).size().index)
dropped = sorted({(t, s) for t, s in d[["tag", "seed"]].itertuples(index=False)} - finished)
if dropped:
    print("left out, no final predictions (unfinished):", dropped)
d = d[[(t, s) in finished for t, s in zip(d.tag, d.seed)]]

Ks = sorted(int(t[len("fedavg_K"):]) for t in d.tag.unique() if t.startswith("fedavg_K"))
TAGS = [f"fedavg_K{K}" for K in Ks]
BUDGET = int(d[d.tag == "pooled"].steps.max())
ROUNDS = {K: -(-BUDGET // K) for K in Ks}
final = d.loc[d.groupby(["tag", "seed", "client"]).steps.idxmax()]
fin_all = final[final.client == "all"]
n_seeds = {t: g.seed.nunique() for t, g in fin_all.groupby("tag")}


def seed_note():
    short = [f"K={t[len('fedavg_K'):]}" for t in TAGS if n_seeds.get(t, 0) < max(n_seeds.values())]
    return f" {', '.join(short)}: fewer seeds (unfinished runs left out)." if short else ""


# ---------------------------------------------------------------- 1. final score vs K
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (m, name) in zip(axes, METRICS.items()):
    pm, ps = fin_all[fin_all.tag == "pooled"][m].agg(["mean", "std"])
    ax.axhspan(pm - ps, pm + ps, color=POOLED, alpha=0.12, lw=0)
    ax.axhline(pm, color=POOLED, ls="--", lw=1.5, label="pooled")
    g = fin_all[fin_all.tag.isin(TAGS)]
    ax.scatter(g["K"], g[m], s=14, color=FEDAVG, alpha=0.35, lw=0, label="FedAvg, each seed")
    mean = g.groupby("K")[m].mean().loc[Ks]
    ax.plot(Ks, mean, color=FEDAVG, lw=2, marker="o", ms=6, label="FedAvg, mean")
    ax.set_xscale("log")
    ax.set_xlabel("local steps per round, K (log scale)")
    ax.set_title(f"{name}, all 4650 test images")
    top = ax.secondary_xaxis("top")
    # K values that don't divide the budget (e.g. 100 next to 97) would crowd the labels
    even = [K for K in Ks if BUDGET % K == 0] or Ks
    top.set_xticks(even, [str(ROUNDS[K]) for K in even], fontsize=8)
    top.tick_params(length=0, colors=MUTED)
    top.set_xlabel("rounds", fontsize=8.5, color=MUTED)
    tidy(ax)
best = int(fin_all[fin_all.tag.isin(TAGS)].groupby("K").balanced_accuracy.mean().idxmax())
axes[0].annotate(f"best: K={best}", (best, fin_all[fin_all["K"] == best].balanced_accuracy.mean()),
                 xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower center")
fig.text(0.01, 0.01, f"Final model after {BUDGET} steps for every run. Pooled: dashed line, band "
         f"± 1 std over seeds.{seed_note()}", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "K_final.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. curves, one panel per K
a = d[d.client == "all"].copy()
a["ep"] = a.epochs.round(1)
pooled_curve = a[a.tag == "pooled"].groupby("ep").balanced_accuracy.mean()
ncol = 4
nrow = -(-len(Ks) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
for ax, K, tag in zip(axes, Ks, TAGS):
    ax.plot(pooled_curve.index, pooled_curve, color=POOLED, ls="--", lw=1.3, label="pooled")
    g = a[a.tag == tag].groupby("ep").balanced_accuracy.agg(["mean", "std"])
    few = len(g) < 8
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=FEDAVG,
                    alpha=0.18, lw=0)
    ax.plot(g.index, g["mean"], color=FEDAVG, lw=2, marker="o" if few else None, ms=5,
            ls=":" if few else "-", label="FedAvg")
    ax.set_title(f"K={K}  ({ROUNDS[K]} round{'s' if ROUNDS[K] > 1 else ''})")
    tidy(ax)
for ax in axes[len(Ks):]:
    ax.axis("off")
for ax in axes[::ncol]:
    ax.set_ylabel("balanced accuracy")
for ax in axes[-ncol:]:
    ax.set_xlabel("pooled-equivalent epochs")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "All 4650 test images; mean over seeds, band ± 1 std. FedAvg is scored at "
         "the end of each round, so runs with few rounds have few points (dotted).",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(OUT / "K_curves.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- 3. per hospital
def bal(g):
    return balanced_accuracy_score(g.label, g.pred)


hosp = (pred.groupby(["method", "seed", "client"]).apply(bal, include_groups=False)
        .rename("bal").reset_index())
hosp = hosp[[(t, s) in finished for t, s in zip(hosp.method, hosp.seed)]]
n_test = pred[pred.method == "pooled"].drop_duplicates(["client", "image"]).groupby("client").size()
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
for ax, c in zip(axes.ravel(), sorted(hosp.client.unique())):
    h = hosp[hosp.client == c]
    for tag, color, ls, label in [("pooled", POOLED, "--", "pooled"),
                                  ("local", LOCAL, ":", "local (own data only)")]:
        v = h[h.method == tag].bal.mean()
        ax.axhline(v, color=color, ls=ls, lw=1.6, label=label)
    f = h[h.method.isin(TAGS)].copy()
    f["K"] = f.method.str[len("fedavg_K"):].astype(int)
    m = f.groupby("K").bal.mean().loc[Ks]
    ax.plot(Ks, m, color=FEDAVG, lw=2, marker="o", ms=5, label="FedAvg")
    ax.set_xscale("log")
    ax.set_title(f"hospital {c}  ({n_test[c]} test images)")
    tidy(ax)
for ax in axes[:, 0]:
    ax.set_ylabel("balanced accuracy")
for ax in axes[-1]:
    ax.set_xlabel("local steps per round, K")
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower center")
fig.text(0.01, 0.01, "Final model, mean over seeds, each hospital's own test set. Balanced "
         "accuracy averages recall over the classes present at that hospital.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(OUT / "K_hospitals.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. per-class recall
cols = ["pooled", *TAGS]
pr = pred[pred.method.isin(cols)]
pr = pr[[(t, s) in finished for t, s in zip(pr.method, pr.seed)]]
rec = (pr.assign(hit=pr.pred == pr.label).groupby(["method", "seed", "label"]).hit.mean()
       .groupby(["method", "label"]).mean().unstack(0)[cols])
share = pr[pr.method == "pooled"].drop_duplicates(["client", "image"]).label.value_counts(
    normalize=True).sort_index()
fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3.2, 4.6))
im = ax.imshow(rec.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(rec.shape[0]):
    for j in range(rec.shape[1]):
        v = rec.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if v > 0.6 else INK)
ax.set_xticks(range(len(cols)), ["pooled", *[f"K={K}" for K in Ks]])
ax.set_yticks(range(8), [f"{k} {CLASSES[k]} ({share[k]:.0%})" for k in rec.index])
ax.axvline(0.5, color="white", lw=3)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("recall per class, all test images (share of test images in brackets)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("recall", color=MUTED)
fig.text(0.01, 0.01, "Final model, mean over seeds. Balanced accuracy is the mean of each column.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "K_classes.png", dpi=200)
plt.close(fig)

print("wrote", ", ".join(str(OUT / f) for f in
                         ["K_final.png", "K_curves.png", "K_hospitals.png", "K_classes.png"]))
