"""Turn results/results.csv into a table (results/summary.md) and two plots.

    python summarize.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- majority-class baseline, read from the data itself
def class_balance():
    """Per hospital: test-set size, the accuracy of always predicting that
    hospital's more common label, and whether its training set has one label
    only (a model trained there alone can then only learn to say that label)."""
    from flamby.datasets.fed_heart_disease import NUM_CLIENTS, FedHeartDisease
    out = {}
    for i in range(NUM_CLIENTS):
        y = {t: np.array([ds[j][1].item() for j in range(len(ds))])
             for t in (True, False)
             for ds in [FedHeartDisease(center=i, train=t, pooled=False)]}
        out[str(i)] = {"n_test": len(y[False]),
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

# ---------- final-round table
final = hospitals[hospitals["round"] ==
                  hospitals.groupby(["method", "seed"])["round"].transform("max")]
mean = final.groupby(["client", "method"])["accuracy"].mean().unstack("method")
std = final.groupby(["client", "method"])["accuracy"].std().unstack("method").fillna(0)
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

lines += ["", "## Summary", ""]
head = ["method", "mean over hospitals", "worst hospital"]
rows = [[m, f"{mean[m].mean():.3f}", f"{mean[m].min():.3f}"] for m in methods]
if balance:
    # Weighted by test-set size: one patient counts the same everywhere, so a
    # 16-patient hospital stops counting as much as a 104-patient one.
    n = pd.Series({c: balance[c]["n_test"] for c in mean.index})
    head.insert(2, "weighted by test n")
    head += ["mean lift over baseline", "hospitals beaten"]
    for row, m in zip(rows, methods):
        acc = mean[m].dropna()
        row.insert(2, f"{(acc * n[acc.index]).sum() / n[acc.index].sum():.3f}")
        row += [f"{lift[m].mean():+.3f}", f"{int((lift[m] > 0).sum())} of {len(lift)}"]
lines += table(head, rows)
if balance:
    lines += ["",
              "`mean over hospitals` weights every hospital equally, so hospital 2 (16 test",
              "patients) counts as much as hospital 0 (104). `weighted by test n` weights each",
              "hospital by its test-set size, which is the per-patient accuracy."]

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

# ---------- curves: mean accuracy over hospitals, averaged over seeds
fed = hospitals[hospitals["round"] >= 0]
curves = fed.groupby(["method", "seed", "round", "bytes_uploaded"])["accuracy"].mean() \
            .groupby(["method", "round", "bytes_uploaded"]).mean().reset_index()

for x, fname, label in [("round", "accuracy_vs_round.png", "server round"),
                        ("bytes_uploaded", "accuracy_vs_bytes.png", "total bytes uploaded")]:
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in curves.groupby("method"):
        ax.plot(g[x], g["accuracy"], label=m)
    for m, style in [("pooled", "--"), ("local", ":")]:
        if m in mean.columns:
            ax.axhline(mean[m].mean(), linestyle=style, color="grey", label=f"{m} (final)")
    if balance:  # a curve below this line is worse than guessing the common label
        ax.axhline(base.mean(), linestyle="-.", color="firebrick",
                   label="majority class")
    ax.set_xlabel(label)
    ax.set_ylabel("mean test accuracy over hospitals")
    ax.set_title("Fed-Heart-Disease")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results/{fname}", dpi=150)
    print(f"saved results/{fname}")
