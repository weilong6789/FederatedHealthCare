"""Turn results/results.csv into tables (results/summary.md) and plots.

    python summarize.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Balanced accuracy and AUC score 0.5 for always predicting one label, on any
# test set. Older logs only have accuracy; their other columns are skipped.
LABELS = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy",
          "auc": "AUC"}


# ---------- majority-class baseline, read from the data itself
def class_balance():
    """Per hospital: test-set size and label counts, the accuracy of always
    predicting that hospital's more common label, and whether its training set
    has one label only (a model trained there alone can then only learn to say
    that label)."""
    from flamby.datasets.fed_heart_disease import NUM_CLIENTS, FedHeartDisease
    out = {}
    for i in range(NUM_CLIENTS):
        y = {t: np.array([ds[j][1].item() for j in range(len(ds))])
             for t in (True, False)
             for ds in [FedHeartDisease(center=i, train=t, pooled=False)]}
        out[str(i)] = {"n_test": len(y[False]),
                       "n_test_1": int(y[False].sum()),
                       "n_test_0": int(len(y[False]) - y[False].sum()),
                       "baseline": max(y[False].mean(), 1 - y[False].mean()),
                       "one_class_train": y[True].min() == y[True].max()}
    return out


try:
    balance = class_balance()
except Exception as exc:          # still summarise if FLamby isn't importable
    balance = None
    print(f"no baseline columns ({exc})")

df = pd.read_csv("results/results.csv", dtype={"client": str})
hospitals = df[df["client"] != "pooled"]
metrics = [m for m in LABELS if m in df.columns]

# ---------- final-round tables
final = hospitals[hospitals["round"] ==
                  hospitals.groupby(["method", "seed"])["round"].transform("max")]
by = final.groupby(["client", "method"])
means = {m: by[m].mean().unstack("method") for m in metrics}
stds = {m: by[m].std().unstack("method").fillna(0) for m in metrics}
mean, std = means["accuracy"], stds["accuracy"]
order = ["local", "pooled", "fedavg"]
methods = [m for m in order if m in mean.columns] + \
          [m for m in mean.columns if m not in order]

# FLamby's centers, in partition-id order (dataset.py: centers_number)
HOSPITALS = ["Cleveland", "Hungary", "Switzerland", "Long Beach V"]


def name(client):
    """'2' -> '2 Switzerland'. Falls back to the bare id if it isn't a hospital."""
    return f"{client} {HOSPITALS[int(client)]}" if client.isdigit() and \
        int(client) < len(HOSPITALS) else client


def table(header, rows):
    """Markdown table with padded columns, so it also reads well as plain text."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return [fmt(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + \
           [fmt(r) for r in rows]


lines = ["## Per-hospital test accuracy (mean ± std over seeds)", ""]
extra = ["test n", "majority class"] if balance else []
lines += table(["hospital"] + extra + methods,
               [[name(c)] + ([balance[c]["n_test"], f"{balance[c]['baseline']:.3f}"] if balance else [])
                + [f"{mean.loc[c, m]:.3f} ± {std.loc[c, m]:.3f}" for m in methods]
                for c in mean.index])

if balance:
    base = pd.Series({c: balance[c]["baseline"] for c in mean.index})
    lift = mean.sub(base, axis=0)
    lines += ["", "`majority class` is the accuracy of always predicting that hospital's more",
              "common label. A method at or below it has learned nothing useful there.", "",
              "## Accuracy minus majority-class baseline", ""]
    lines += table(["hospital"] + methods,
                   [[name(c)] + [f"{lift.loc[c, m]:+.3f}" for m in methods] for c in lift.index])

for k in metrics[1:]:
    lines += ["", f"## Per-hospital test {LABELS[k]} (mean ± std over seeds)", ""]
    lines += table(["hospital"] + methods,
                   [[name(c)] + [f"{means[k].loc[c, m]:.3f} ± {stds[k].loc[c, m]:.3f}"
                                 for m in methods] for c in means[k].index])
if len(metrics) > 1:
    lines += ["", "Always predicting one label scores 0.500 on balanced accuracy and AUC at",
              "every hospital, however skewed its test set, so 0.500 is the bar to beat."]


def summary(k):
    """Mean, test-n-weighted mean and worst hospital for metric k, per method."""
    head = ["method", "mean over hospitals", "worst hospital"]
    rows = [[m, f"{means[k][m].mean():.3f}", f"{means[k][m].min():.3f}"] for m in methods]
    if balance:
        # Weighted by test-set size: one patient counts the same everywhere, so
        # a 16-patient hospital stops counting as much as a 104-patient one.
        n = pd.Series({c: balance[c]["n_test"] for c in means[k].index})
        head.insert(2, "weighted by test n")
        for row, m in zip(rows, methods):
            v = means[k][m].dropna()
            row.insert(2, f"{(v * n[v.index]).sum() / n[v.index].sum():.3f}")
        if k == "accuracy":
            head += ["mean lift over baseline", "hospitals beaten"]
            for row, m in zip(rows, methods):
                row += [f"{lift[m].mean():+.3f}",
                        f"{int((lift[m] > 0).sum())} of {len(lift)}"]
    return table(head, rows)


lines += ["", "## Summary", ""]
for k in metrics:
    if len(metrics) > 1:
        lines += [f"{LABELS[k].capitalize()}:", ""]
    lines += summary(k) + [""]
if balance:
    lines += ["`mean over hospitals` weights every hospital equally, so hospital 2 (16 test",
              "patients) counts as much as hospital 0 (104). `weighted by test n` weights each",
              "hospital by its test-set size, which is the per-patient score."]

if balance:
    lines += ["", "## Data caveats", ""]
    for c in mean.index:
        if balance[c]["one_class_train"]:
            lines.append(
                f"- {name(c)}: every training patient has the same label, so a model trained "
                f"there alone can only predict that label. Its `local` accuracy is the "
                f"majority-class baseline ({balance[c]['baseline']:.3f}), not learning.")
    for c in mean.index:
        if balance[c]["n_test"] < 50:
            lines.append(
                f"- {name(c)}: only {balance[c]['n_test']} test patients, so its accuracy "
                f"moves in steps of {1 / balance[c]['n_test']:.3f}.")
    for c in mean.index:
        n0, n1 = balance[c]["n_test_0"], balance[c]["n_test_1"]
        if len(metrics) > 1 and 0 < min(n0, n1) < 5:
            rare = min(n0, n1)
            lines.append(
                f"- {name(c)}: the test set has {n0} patient(s) with label 0 and {n1} with "
                f"label 1. Half of its balanced accuracy rests on the {rare} rarer one(s), and "
                f"its AUC on {n0 * n1} label-0/label-1 pairs (steps of {1 / (n0 * n1):.3f}).")

# ---------- what was run, read back from the log so it can't go stale
def values(g, col):
    return ", ".join(str(v) for v in sorted(g[col].unique()))

settings = []
for m in methods:
    g = final[final["method"] == m]
    rounds = int(g["round"].max())
    settings.append([m, values(g, "seed"), values(g, "lr"),
                     rounds if rounds >= 0 else "-", values(g, "local_epochs")])
lines += ["", "## Run settings", ""]
lines += table(["method", "seeds", "lr", "server rounds", "epochs (per round if federated)"],
               settings)

out = "\n".join(lines)
print(out)
with open("results/summary.md", "w") as f:
    f.write(out + "\n")

# ---------- curves: mean over hospitals, averaged over seeds
fed = hospitals[hospitals["round"] >= 0]
plots = [("accuracy", "round", "server round"),
         ("accuracy", "bytes_uploaded", "total bytes uploaded")] + \
        [(k, "round", "server round") for k in metrics[1:]]

for k, x, label in plots:
    curves = fed.groupby(["method", "seed", "round", "bytes_uploaded"])[k].mean() \
                .groupby(["method", "round", "bytes_uploaded"]).mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in curves.groupby("method"):
        ax.plot(g[x], g[k], label=m)
    for m, style in [("pooled", "--"), ("local", ":")]:
        if m in means[k].columns:
            ax.axhline(means[k][m].mean(), linestyle=style, color="grey",
                       label=f"{m} (final)")
    # a curve below this line is no better than always saying one label
    if k != "accuracy":
        ax.axhline(0.5, linestyle="-.", color="firebrick", label="chance")
    elif balance:
        ax.axhline(base.mean(), linestyle="-.", color="firebrick",
                   label="majority class")
    ax.set_xlabel(label)
    ax.set_ylabel(f"mean test {LABELS[k]} over hospitals")
    ax.set_title("Fed-Heart-Disease")
    ax.legend()
    fig.tight_layout()
    fname = f"{k}_vs_{'bytes' if x == 'bytes_uploaded' else x}.png"
    fig.savefig(f"results/{fname}", dpi=150)
    plt.close(fig)
    print(f"saved results/{fname}")
