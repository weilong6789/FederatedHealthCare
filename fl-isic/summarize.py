"""Tables and plots from results/isic.csv and results/isic_predictions.csv.

    python summarize.py

Writes results/summary.md and results/isic_curve_<metric>.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

OUT = Path("results")
METRICS = {"balanced_accuracy": "balanced accuracy", "accuracy": "accuracy", "auc": "AUC"}
CLIENTS = [str(i) for i in range(6)]
N_BOOT = 1000

d = pd.read_csv(OUT / "isic.csv", dtype=str)
d = d[d.method != "method"]          # header repeated by jobs that started together
for c in ["seed", "round", "steps", "epochs", "n", "seconds", *METRICS]:
    d[c] = pd.to_numeric(d[c])
d["run"] = d.method + np.where(d.method == "fedavg", " T=" + d["T"].fillna(""), "")
d = d.drop_duplicates(["run", "seed", "steps", "client"], keep="last")
final = d.loc[d.groupby(["run", "seed", "client"]).steps.idxmax()]
RUNS = [r for r in ["local", "pooled"] if r in set(d.run)] + sorted(
    r for r in set(d.run) if r.startswith("fedavg"))
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
        ax.plot(g.index, g["mean"], label=r, ls="--" if r == "pooled" else "-")
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
    ax.set_xlabel("pooled-equivalent epochs")
    ax.set_ylabel(f"test {name}, all hospitals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"isic_curve_{m}.png", dpi=150)
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
