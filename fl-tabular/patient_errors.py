"""Which test patients each method gets wrong, per hospital.

    python patient_errors.py      # reads results/, writes results/patient_errors.md

Uses the saved per-patient predictions, so nothing is retrained:
    results/predictions.csv               local, pooled, FedAvg (tuned baselines), 5 seeds
    results/budget_sweep_predictions.csv  pooled and FedAvg at several K, 20-epoch budget

A patient is "always wrong" for a run if it is misclassified (threshold 0.5)
in every seed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
HOSP = ["Cleveland", "Hungary", "Switzerland", "Long Beach VA"]
TEST_N = [104, 89, 16, 45]  # test patients per hospital, in FLamby's order
out = []

# ---------------------------------------------------------------- baselines
d = pd.read_csv(R / "predictions.csv")
d["wrong"] = (d.prob >= 0.5).astype(int) != d.label
n_seeds = d.seed.nunique()
w = d.pivot_table(index=["client", "patient", "label"], columns="method",
                  values="wrong", aggfunc="sum").reset_index()
methods = ["local", "pooled", "fedavg"]

out.append("## Majority-label baseline vs models (test accuracy)\n")
out.append("| hospital | test n | dominant label | guess dominant label | "
           + " | ".join(methods) + " |")
out.append("|---|---|---|---|" + "---|" * len(methods))
acc = d.groupby(["method", "client"]).apply(lambda g: 1 - g.wrong.mean()).unstack(0)
maj_correct = 0
for c in range(4):
    lab = w[w.client == c].label
    k = lab.value_counts()
    top = k.idxmax()
    maj_correct += k.max()
    name = "sick" if top == 1 else "healthy"
    out.append(f"| {HOSP[c]} | {len(lab)} | {name} ({k.max()}/{len(lab)}) | {k.max() / len(lab):.3f} | "
               + " | ".join(f"{acc.loc[c, m]:.3f}" for m in methods) + " |")
n_all = len(w)
allacc = {m: 1 - d[d.method == m].wrong.mean() for m in methods}
out.append(f"| all patients | {n_all} | each hospital's own | {maj_correct / n_all:.3f} | "
           + " | ".join(f"{allacc[m]:.3f}" for m in methods) + " |")
k = w.label.value_counts()
out.append(f"| all patients | {n_all} | {'sick' if k.idxmax() == 1 else 'healthy'} for everyone "
           f"({k.max()}/{n_all}) | {k.max() / n_all:.3f} | | | |\n")

out.append(f"## Same patients wrong across seeds? (tuned baselines, {n_seeds} seeds)\n")
out.append("Patients by the number of seeds in which they are misclassified.\n")
for m in methods:
    t = pd.crosstab(w.client, w[m]).reindex(range(4))
    t.index = HOSP
    out.append(f"**{m}**\n")
    out.append("| hospital | " + " | ".join(f"{int(c)} of {n_seeds}" for c in t.columns) + " |")
    out.append("|---|" + "---|" * len(t.columns))
    for h, row in t.iterrows():
        out.append(f"| {h} | " + " | ".join(str(v) for v in row) + " |")
    out.append("")

out.append("## Always-wrong patients by true label (tuned baselines)\n")
out.append("Healthy called sick = true label 0; sick called healthy = true label 1.\n")
out.append("| hospital | method | always wrong | healthy called sick | sick called healthy |")
out.append("|---|---|---|---|---|")
for c in range(4):
    for m in methods:
        s = w[(w.client == c) & (w[m] == n_seeds)]
        out.append(f"| {HOSP[c]} | {m} | {len(s)} | {(s.label == 0).sum()} | {(s.label == 1).sum()} |")
out.append("")

# ---------------------------------------------------------------- K sweep
b = pd.read_csv(R / "budget_sweep_predictions.csv")
b = b[b.budget == b.budget.max()].copy()
b["client"] = np.repeat(range(4), TEST_N)[b.patient]
b["wrong"] = (b.prob >= 0.5).astype(int) != b.label
b["run"] = np.where(b.method == "fedavg", "FedAvg K=" + b["T"].astype(str), "pooled")
s = b.groupby(["run", "patient"]).agg(w=("wrong", "all"), c=("client", "first"),
                                      p=("prob", "mean"), l=("label", "first")).reset_index()
sets = {r: set(g.loc[g.w, "patient"]) for r, g in s.groupby("run")}
ref = sets["pooled"]
out.append(f"## Same patients wrong across K? ({int(b.budget.max())}-step budget, "
           f"{b.seed.nunique()} seeds)\n")
out.append("| run | always wrong | same as pooled | not in pooled's set | in pooled's set but right here |")
out.append("|---|---|---|---|---|")
order = ["pooled"] + sorted((r for r in sets if r != "pooled"), key=lambda r: int(r.split("=")[1]))
for r in order:
    x = sets[r]
    out.append(f"| {r} | {len(x)} | {len(x & ref)} | {len(x - ref)} | {len(ref - x)} |")
pw = s[(s.run == "pooled") & s.w]
out.append(f"\nPooled's always-wrong patients sit on average {(pw.p - 0.5).abs().mean():.2f} "
           "from the 0.5 cutoff (mean predicted probability over seeds), so most are not borderline.\n")

(R / "patient_errors.md").write_text("\n".join(out))
print("\n".join(out))
