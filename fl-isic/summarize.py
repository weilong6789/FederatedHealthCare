"""Tables and plots from results/isic.csv and results/isic_predictions.csv.

    python summarize.py              # reads and writes results/
    python summarize.py results40    # same for another results folder

Writes <folder>/summary.md, <folder>/isic_curve_<metric>.png, <folder>/isic_trend.png
and, once there is more than one FedAvg T, <folder>/isic_T_sweep.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
METRICS = {"balanced_accuracy": "balanced accuracy", "accuracy": "accuracy", "auc": "AUC"}
CLIENTS = [str(i) for i in range(6)]
N_BOOT = 1000

d = pd.read_csv(OUT / "isic.csv", dtype=str)
d = d[d.method != "method"]          # header repeated by jobs that started together
for c in ["seed", "round", "steps", "epochs", "n", "seconds", *METRICS]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
# Jobs appending at the same moment can splice two lines into one; drop those.
ok = d.method.isin(["pooled", "fedavg", "local"]) & d.client.isin(CLIENTS + ["all"]) \
    & d[["seed", "steps", "epochs", "n", *METRICS]].notna().all(axis=1)
if (~ok).any():
    print(f"skipping {(~ok).sum()} malformed rows of {OUT / 'isic.csv'}:")
    print(d[~ok].to_string())
d = d[ok].copy()
d[["seed", "round", "steps", "n"]] = d[["seed", "round", "steps", "n"]].astype(int)
d["T"] = pd.to_numeric(d["T"], errors="coerce")
d["run"] = d.method + np.where(d.method == "fedavg",
                               " T=" + d["T"].fillna(0).astype(int).astype(str), "")
d = d.drop_duplicates(["run", "seed", "steps", "client"], keep="last")
final = d.loc[d.groupby(["run", "seed", "client"]).steps.idxmax()]
FED = sorted((r for r in set(d.run) if r.startswith("fedavg")), key=lambda r: int(r[len("fedavg T="):]))
RUNS = [r for r in ["local", "pooled"] if r in set(d.run)] + FED
# pooled blue; FedAvg orange when there is one T, light to dark by T when there are several
COLORS = {"pooled": "#2a78d6"}
_cmap = plt.get_cmap("YlOrRd")
for i, r in enumerate(FED):
    COLORS[r] = "#eb6834" if len(FED) == 1 else _cmap(0.35 + 0.6 * i / (len(FED) - 1))
lines = []


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    lines.append("")


# ---------------------------------------------------------------- final scores
seeds = sorted(final.seed.unique())
lines.append(f"## Final test scores (mean ± std over seeds {', '.join(map(str, seeds))})\n")
lines.append("`local` is each hospital's own model scored on its own test set; it has no "
             "`all` row. `all` is the 4650 test images together.\n")
n_test = final[final.run == "pooled"].groupby("client").n.first()
for m, name in METRICS.items():
    lines.append(f"{name[0].upper() + name[1:]}:\n")
    g = final.groupby(["client", "run"])[m].agg(["mean", "std"])
    rows = []
    for c in CLIENTS + ["all"]:
        cells = [f"{g.loc[(c, r), 'mean']:.3f} ± {g.loc[(c, r), 'std']:.3f}"
                 if (c, r) in g.index else "-" for r in RUNS]
        rows.append([c, str(int(n_test.get(c, 0))), *cells])
    table(["hospital", "test n", *RUNS], rows)

# ---------------------------------------------------------------- label mix
pred = pd.read_csv(OUT / "isic_predictions.csv", dtype={"client": str})
pred = pred[pred.method != "method"]
P = [f"p{k}" for k in range(8)]
pred[P] = pred[P].astype(float)
pred[["seed", "image", "label"]] = pred[["seed", "image", "label"]].astype(int)
mix = pd.crosstab(pred[pred.method == "pooled"].drop_duplicates(["client", "image"]).client,
                  pred[pred.method == "pooled"].drop_duplicates(["client", "image"]).label,
                  normalize="index")
lines.append("## Test-set label mix per hospital (share of each class)\n")
lines.append("The hospitals see very different mixes of the 8 classes, which is what "
             "pulls FedAvg away from pooled here.\n")
table(["hospital", *[f"class {k}" for k in mix.columns]],
      [[c, *[f"{v:.2f}" for v in mix.loc[c]]] for c in mix.index])


# ---------------------------------------------------------------- bootstrap
def score(m, y, p):
    if m == "balanced_accuracy":
        return balanced_accuracy_score(y, p.argmax(1))
    if m == "accuracy":
        return float((p.argmax(1) == y).mean())
    aucs = [roc_auc_score(y == c, p[:, c]) for c in range(8) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs))


def arrays(run_tag):
    """Per seed: labels and probabilities for all test images, hospitals in order."""
    out = {}
    for s, g in pred[pred.method == run_tag].groupby("seed"):
        g = g.sort_values(["client", "image"])
        out[s] = (g.label.to_numpy(), g[P].to_numpy(), g.client.to_numpy())
    return out


tags = {r: r.replace(" T=", "_T") for r in RUNS if r != "local"}
arr = {r: arrays(t) for r, t in tags.items()}
fed = [r for r in tags if r.startswith("fedavg")]
if "pooled" in arr and fed:
    y0, _, cl = next(iter(arr["pooled"].values()))
    rng = np.random.default_rng(0)
    idx_by_client = [np.where(cl == c)[0] for c in CLIENTS]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_client])
             for _ in range(N_BOOT)]
    lines.append("## FedAvg minus pooled on all test images (paired bootstrap)\n")
    lines.append(f"Test images resampled with replacement within each hospital, {N_BOOT} "
                 "times; both methods scored on the same resamples, averaged over seeds. "
                 "`*` marks an interval that excludes 0.\n")
    rows = []
    for f in fed:
        for m, name in METRICS.items():
            def mean_score(run, ix):
                return np.mean([score(m, y[ix], p[ix]) for y, p, _ in arr[run].values()])
            diffs = np.array([mean_score(f, ix) - mean_score("pooled", ix) for ix in boots])
            full = np.arange(len(y0))
            point = mean_score(f, full) - mean_score("pooled", full)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else ""
            rows.append([f, name, f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"])
    table(["run", "metric", "FedAvg − pooled [95% CI]"], rows)

# ---------------------------------------------------------------- curves
for m, name in METRICS.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in [r for r in RUNS if r != "local"]:
        a = d[(d.run == r) & (d.client == "all")].copy()
        a["ep"] = a.epochs.round(1)
        g = a.groupby("ep")[m].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], label=r, color=COLORS[r], ls="--" if r == "pooled" else "-",
                marker="o" if len(g) < 6 else None, ms=4)   # few-round runs: show the points
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=COLORS[r], alpha=0.15, lw=0)
    ax.set_xlabel("pooled-equivalent epochs")
    ax.set_ylabel(f"test {name}, all hospitals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"isic_curve_{m}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- still improving?
# Straight-line fit to each seed's score over the second half of training; the
# slope says how much a run was still gaining when training stopped.
END = round(d.epochs.max())
LATE = END / 2
curve_runs = [r for r in RUNS if r != "local"]
lines.append(f"## Still improving at epoch {END}? (all test images)\n")
lines.append(f"Final score, and the gain per 10 epochs from a straight-line fit to each "
             f"seed's scores over epochs {LATE:g}-{END} (mean over seeds; `n/3 up` counts seeds "
             f"with a positive slope; `-` when a run has fewer than 3 evaluations in that "
             f"window, e.g. FedAvg with very few rounds).\n")
rows, slopes = [], {}
for r in curve_runs:
    a = d[(d.run == r) & (d.client == "all")]
    fin = final[(final.run == r) & (final.client == "all")]
    for m, name in METRICS.items():
        sl = [np.polyfit(s.epochs, s[m], 1)[0] for _, s in a[a.epochs >= LATE].groupby("seed")
              if len(s) >= 3]
        slopes[(r, m)] = np.mean(sl) if sl else np.nan
        rows.append([r, name, f"{fin[m].mean():.3f} ± {fin[m].std():.3f}",
                     f"{10 * np.mean(sl):+.3f}" if sl else "-",
                     f"{sum(s > 0 for s in sl)}/{len(sl)} up" if sl else "-"])
table(["run", "metric", "final", "gain per 10 epochs", "seeds rising"], rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, m in zip(axes, ["accuracy", "balanced_accuracy"]):
    for r in curve_runs:
        a = d[(d.run == r) & (d.client == "all")].copy()
        a["ep"] = a.epochs.round(0)
        g = a.groupby("ep")[m].agg(["mean", "std"])
        g = g[g.index >= 1]                           # epoch 0 is the untrained model
        c = COLORS[r]
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=c, alpha=0.15, lw=0)
        ax.plot(g.index, g["mean"], color=c, lw=2, label=r, marker="o" if len(g) < 6 else None, ms=4)
        late = a[a.epochs >= LATE]
        # fitted line and label only when there are few runs, or the plot gets unreadable
        if len(curve_runs) > 3 or late.epochs.nunique() < 3:
            continue
        k, b = np.polyfit(late.epochs, late[m], 1)
        xs = np.array([LATE, late.epochs.max()])
        ax.plot(xs, k * xs + b, color=c, lw=1.5, ls=(0, (4, 3)))
        ax.annotate(f"{r}: {10 * k:+.3f} / 10 ep", (xs[1], k * xs[1] + b),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#52514e")
    ax.axvspan(LATE, END + 0.3, color="#000000", alpha=0.04, lw=0)
    ax.set_title(f"test {METRICS[m]}, all 4650 images", fontsize=11, loc="left")
    ax.set_xlabel("pooled-equivalent epochs")
    ax.set_xlim(0.5, END * (1.35 if len(curve_runs) <= 3 else 1.02))
    ax.set_xticks(range(0, END + 1, 5 if END <= 20 else 10))
    ax.grid(alpha=0.25, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
fig.text(0.01, 0.01, "Lines: mean over seeds, band ± 1 std. " + (
    f"Dashed: straight-line fit over epochs {LATE:g}-{END} (shaded), labelled with its gain "
    "per 10 epochs." if len(curve_runs) <= 3 else
    f"Shaded: epochs {LATE:g}-{END}; slopes are in summary.md."),
         fontsize=8.5, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "isic_trend.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- T sweep
if len(FED) > 1:
    fin = final[final.client == "all"]
    pooled_fin = fin[fin.run == "pooled"]
    lines.append("## FedAvg local steps per round (T), all test images\n")
    lines.append("Final score, mean ± std over seeds. Rounds = steps / T, so T = 1 averages "
                 "after every step and the largest T averages once at the end. T values that "
                 "don't divide the budget run a few extra steps (see `steps`).\n")
    rows = []
    if len(pooled_fin):
        rows.append(["pooled", "-", str(int(pooled_fin.steps.max())), *[
            f"{pooled_fin[m].mean():.3f} ± {pooled_fin[m].std():.3f}" for m in METRICS]])
    for r in FED:
        f_ = fin[fin.run == r]
        rows.append([r, str(int(f_["round"].max())), str(int(f_.steps.max())), *[
            f"{f_[m].mean():.3f} ± {f_[m].std():.3f}" for m in METRICS]])
    table(["run", "rounds", "steps", *METRICS.values()], rows)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.8))
    Ts = np.array([int(r[len("fedavg T="):]) for r in FED])
    for ax, (m, name) in zip(axes, METRICS.items()):
        g = fin[fin.run.isin(FED)].groupby("run")[m].agg(["mean", "std"]).loc[FED]
        ax.errorbar(Ts, g["mean"], yerr=g["std"], color="#eb6834", marker="o", ms=4,
                    lw=1.8, capsize=3, label="FedAvg")
        if len(pooled_fin):
            mu, sd = pooled_fin[m].mean(), pooled_fin[m].std()
            ax.axhline(mu, color=COLORS["pooled"], ls="--", lw=1.5, label="pooled")
            ax.axhspan(mu - sd, mu + sd, color=COLORS["pooled"], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("local steps per round (T)")
        ax.set_title(f"test {name}, all 4650 images", fontsize=11, loc="left")
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Final model, mean ± 1 std over seeds. Pooled: dashed line and band.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "isic_T_sweep.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- settings
lines.append("## Run settings\n")
s = d.groupby("run").agg(seeds=("seed", lambda x: ", ".join(map(str, sorted(x.unique())))),
                         lr=("lr", "first"), opt=("opt", "first"),
                         epochs=("epochs", "max"), minutes=("seconds", lambda x: x.max() / 60))
table(["run", "seeds", "optimizer", "lr", "epochs", "longest run (min)"],
      [[r, s.loc[r, "seeds"], s.loc[r, "opt"], s.loc[r, "lr"], f"{s.loc[r, 'epochs']:.1f}",
        f"{s.loc[r, 'minutes']:.0f}"] for r in RUNS])

(OUT / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
