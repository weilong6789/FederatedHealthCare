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
        lines += [f"{LABELS[k][0].upper()}{LABELS[k][1:]}:", ""]
    lines += summary(k) + [""]
if balance:
    lines += ["`mean over hospitals` weights every hospital equally, so hospital 2 (16 test",
              "patients) counts as much as hospital 0 (104). `weighted by test n` weights each",
              "hospital by its test-set size, which is the per-patient score."]


# ---------- bootstrap intervals over test patients
# The ± above only covers retraining (seeds) on the same test patients. These
# intervals also cover which patients happened to land in each test set:
# resample a hospital's test patients with replacement, re-score, repeat.
B = 1000


def boot_scores(y, P, pos_idx, neg_idx):
    """Seed-averaged scores on each resample.

    y: labels (n,). P: predictions (seeds, n). pos_idx / neg_idx: (B, n1) and
    (B, n0) resampled indices of the label-1 / label-0 patients.
    Returns metric -> (B,) array.
    """
    sp, sn = P[:, pos_idx], P[:, neg_idx]              # (seeds, B, n1|n0)
    n1, n0 = pos_idx.shape[1], neg_idx.shape[1]
    tp, tn = (sp > 0.5).sum(-1), (sn <= 0.5).sum(-1)   # 0.5 cutoff, as in evaluate()
    pairs = sp[..., :, None] - sn[..., None, :]        # every label-1/label-0 pair
    auc = (pairs > 0).mean((-1, -2)) + 0.5 * (pairs == 0).mean((-1, -2))
    return {"accuracy": ((tp + tn) / (n1 + n0)).mean(0),
            "balanced_accuracy": (0.5 * (tp / n1 + tn / n0)).mean(0),
            "auc": auc.mean(0)}


def bootstrap(preds):
    """method -> hospital -> metric -> (point estimate, (B,) resampled scores).

    Stratified: label-1 and label-0 patients are resampled separately, keeping
    each hospital's label counts, so every resample has both labels and AUC is
    always defined. All methods share the same resamples, so differences
    between methods are paired: they don't pick up which patients were drawn.
    """
    rng = np.random.default_rng(0)
    out = {}
    for c, g in preds.groupby("client"):
        probs = g.pivot_table(index=["method", "seed"], columns="patient", values="prob")
        y = g.groupby("patient")["label"].first().reindex(probs.columns).to_numpy()
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_idx = rng.choice(pos, (B, len(pos)))
        neg_idx = rng.choice(neg, (B, len(neg)))
        for m in probs.index.get_level_values("method").unique():
            P = probs.loc[m].to_numpy()
            point = boot_scores(y, P, pos[None], neg[None])
            boot = boot_scores(y, P, pos_idx, neg_idx)
            out.setdefault(m, {})[c] = {k: (point[k][0], boot[k]) for k in boot}
    return out


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return lo, hi


try:
    preds = pd.read_csv("results/predictions.csv", dtype={"client": str})
except FileNotFoundError:
    preds = None
    print("no results/predictions.csv: skipping bootstrap intervals")

if preds is not None:
    boot = bootstrap(preds)
    bmethods = [m for m in methods if m in boot]
    clients = sorted(set.intersection(*(set(boot[m]) for m in bmethods)))
    n_test = {c: len(preds[(preds["client"] == c) & (preds["method"] == bmethods[0]) &
                           (preds["seed"] == preds["seed"].min())]) for c in clients}

    def rows_for(k, value):
        """Per-hospital rows plus mean and test-n-weighted mean over hospitals.
        `value(m, c)` returns (point, (B,) samples) for metric k."""
        rows = [(name(c), {m: value(m, c) for m in bmethods}) for c in clients]
        w = np.array([n_test[c] for c in clients], float)
        for label, wts in [("mean over hospitals", np.ones_like(w)),
                           ("weighted by test n", w)]:
            agg = {}
            for m in bmethods:
                pts, smp = zip(*(value(m, c) for c in clients))
                agg[m] = (np.average(pts, weights=wts),
                          np.average(np.stack(smp), axis=0, weights=wts))
            rows.append((label, agg))
        return rows

    lines += ["", "## 95% confidence intervals (bootstrap over test patients)", "",
              f"Each hospital's test patients resampled with replacement, {B} times, keeping",
              "its label counts; scores averaged over seeds. Unlike the ± above, the interval",
              "shows how much a score could move with a different draw of test patients.", ""]
    for k in metrics:
        lines += [f"{LABELS[k][0].upper()}{LABELS[k][1:]}:", ""]
        rows = rows_for(k, lambda m, c: boot[m][c][k])
        lines += table(["hospital"] + bmethods,
                       [[r] + ["{:.3f} [{:.3f}, {:.3f}]".format(v[m][0], *ci(v[m][1]))
                               for m in bmethods] for r, v in rows]) + [""]

    if "fedavg" in bmethods and len(bmethods) > 1:
        others = [m for m in bmethods if m != "fedavg"]
        lines += ["## FedAvg minus each other method (paired bootstrap)", "",
                  "`*` marks a difference whose 95% interval excludes 0. Without a `*`, the",
                  "test sets are too small to say which method is better there.", ""]
        for k in metrics:
            lines += [f"{LABELS[k][0].upper()}{LABELS[k][1:]}:", ""]
            rows = []
            for r, v in rows_for(k, lambda m, c: boot[m][c][k]):
                cells = []
                for o in others:
                    lo, hi = ci(v["fedavg"][1] - v[o][1])
                    star = "*" if lo > 0 or hi < 0 else " "
                    cells.append(f"{v['fedavg'][0] - v[o][0]:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
                rows.append([r] + cells)
            lines += table(["hospital"] + [f"fedavg − {o}" for o in others], rows) + [""]

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
            if preds is not None:
                lines.append(
                    f"- {name(c)}: the bootstrap keeps label counts fixed, so every resample "
                    f"contains the same {rare} rarer patient(s). Its intervals understate the "
                    f"real uncertainty; don't draw conclusions from this hospital alone.")

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
for k in metrics:
    curves = fed.groupby(["method", "seed", "round"])[k].mean() \
                .groupby(["method", "round"]).mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in curves.groupby("method"):
        ax.plot(g["round"], g[k], label=m)
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
    ax.set_xlabel("server round")
    ax.set_ylabel(f"mean test {LABELS[k]} over hospitals")
    ax.set_title("Fed-Heart-Disease")
    ax.legend()
    fig.tight_layout()
    fname = f"{k}_vs_round.png"
    fig.savefig(f"results/{fname}", dpi=150)
    plt.close(fig)
    print(f"saved results/{fname}")
