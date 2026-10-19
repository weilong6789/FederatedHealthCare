"""Pick each method's hyperparameters on a validation split, never on the test set.

    python tune.py

Every method trains on 80% of each hospital's training data ("fit") and is
scored on the other 20% ("val"); see VAL_FRACTION in fltab/harness.py. Each
method gets its own best setting, picked by validation AUC averaged over
hospitals and seeds. Writes:

    results/tuned.env         the chosen settings; run_all.sh reads it
    results/tuning.md         the full grid
    results/tuning_curve.png  FedAvg validation AUC per round, for convergence

FedAvg runs in-process here rather than through Flower: 50 rounds x the grid x
seeds through `flwr run` would take hours. `simulate_fedavg` repeats what
client_app.py and Flower's FedAvg do (same shuffling seeds, same
example-weighted average), so it reaches the same model.
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fltab.harness import (NUM_CLIENTS, ROOT, eval_loaders, evaluate, evaluate_all,
                              make_model, set_seed, train_loader, train_sgd)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
p.add_argument("--local-epochs", type=int, nargs="+", default=[1, 5],
               help="FedAvg local epochs per round")
p.add_argument("--epochs", type=int, default=50, help="local and pooled training epochs")
p.add_argument("--rounds", type=int, default=50, help="FedAvg server rounds")
args = p.parse_args()

HOSPITALS = [str(i) for i in range(NUM_CLIENTS)]


def val_auc(results):
    """Mean validation AUC over hospitals. A hospital whose validation slice has
    one label only (Switzerland) has no AUC and is left out."""
    return float(np.nanmean([results[h]["auc"] for h in HOSPITALS if h in results]))


def simulate_fedavg(seed, lr, epochs, rounds, part="fit", on_round=None):
    """FedAvg exactly as client_app.py + Flower's FedAvg run it."""
    set_seed(seed)
    model = make_model()
    for r in range(1, rounds + 1):
        states, sizes = [], []
        for pid in range(NUM_CLIENTS):
            shuffle_seed = seed * 1000 + r * 10 + pid     # as in client_app.py
            set_seed(shuffle_seed)
            loader = train_loader(center=pid, seed=shuffle_seed, part=part)
            local = make_model()
            local.load_state_dict(model.state_dict())
            train_sgd(local, loader, lr, epochs)
            states.append(local.state_dict())
            sizes.append(len(loader.dataset))
        model.load_state_dict({k: sum(s[k] * n for s, n in zip(states, sizes)) / sum(sizes)
                               for k in states[0]})
        if on_round:
            on_round(r, model)
    return model


def tune_pooled(lr):
    scores = []
    for seed in args.seeds:
        set_seed(seed)
        m = train_sgd(make_model(), train_loader(pooled=True, seed=seed, part="fit"),
                      lr, args.epochs)
        scores.append(val_auc(evaluate_all(m, split="val")))
    return float(np.mean(scores))


def tune_local(lr):
    scores = []
    for seed in args.seeds:
        res = {}
        for i in range(NUM_CLIENTS):
            set_seed(seed)
            m = train_sgd(make_model(), train_loader(center=i, seed=seed, part="fit"),
                          lr, args.epochs)
            res[str(i)] = evaluate(m, eval_loaders("val")[str(i)])
        scores.append(val_auc(res))
    return float(np.mean(scores))


def tune_fedavg(lr, epochs):
    """Final-round score, plus the per-round curve averaged over seeds."""
    curves = []
    for seed in args.seeds:
        curve = []
        simulate_fedavg(seed, lr, epochs, args.rounds,
                        on_round=lambda r, m: curve.append(val_auc(evaluate_all(m, "val"))))
        curves.append(curve)
    curve = np.mean(curves, axis=0)
    return float(curve[-1]), curve


grid = {"pooled": {}, "local": {}, "fedavg": {}}
curves = {}
for lr in args.lrs:
    grid["pooled"][(lr,)] = tune_pooled(lr)
    grid["local"][(lr,)] = tune_local(lr)
    print(f"lr={lr}: pooled {grid['pooled'][(lr,)]:.3f}  local {grid['local'][(lr,)]:.3f}")
for lr, e in itertools.product(args.lrs, args.local_epochs):
    grid["fedavg"][(lr, e)], curves[(lr, e)] = tune_fedavg(lr, e)
    print(f"lr={lr} local-epochs={e}: fedavg {grid['fedavg'][(lr, e)]:.3f}")

best = {m: max(g, key=g.get) for m, g in grid.items()}

out = ROOT / "results"
out.mkdir(exist_ok=True)
(out / "tuned.env").write_text(
    "# written by tune.py: chosen on the validation split, not the test set\n"
    f"POOLED_LR={best['pooled'][0]}\n"
    f"LOCAL_LR={best['local'][0]}\n"
    f"FEDAVG_LR={best['fedavg'][0]}\n"
    f"FEDAVG_EPOCHS={best['fedavg'][1]}\n")

# Convergence: is the chosen FedAvg still improving at the last round?
c = curves[best["fedavg"]]
tail = max(1, len(c) // 5)
gain = c[-1] - c[-1 - tail]
converged = abs(gain) < 0.005

lines = ["## Hyperparameter tuning (validation split, not test)", "",
         f"Validation AUC, mean over hospitals (Switzerland's validation slice has one "
         f"label, so it has no AUC and is left out), mean over seeds "
         f"{', '.join(map(str, args.seeds))}. Local and pooled train {args.epochs} epochs; "
         f"FedAvg runs {args.rounds} rounds.", "",
         "| lr | pooled | local |", "|---|---|---|"]
for (lr,) in grid["pooled"]:
    mark = lambda m: " ←" if best[m] == (lr,) else ""
    lines.append(f"| {lr} | {grid['pooled'][(lr,)]:.3f}{mark('pooled')} | "
                 f"{grid['local'][(lr,)]:.3f}{mark('local')} |")
lines += ["", "| lr | " + " | ".join(f"FedAvg, {e} local epoch(s)" for e in args.local_epochs) + " |",
          "|---|" + "---|" * len(args.local_epochs)]
for lr in args.lrs:
    cells = [f"{grid['fedavg'][(lr, e)]:.3f}" + (" ←" if best["fedavg"] == (lr, e) else "")
             for e in args.local_epochs]
    lines.append(f"| {lr} | " + " | ".join(cells) + " |")
lines += ["", "Chosen (`←`, also in `results/tuned.env`):", "",
          f"- pooled: lr {best['pooled'][0]}",
          f"- local: lr {best['local'][0]}",
          f"- fedavg: lr {best['fedavg'][0]}, {best['fedavg'][1]} local epoch(s) per round", "",
          f"FedAvg convergence: validation AUC changed by {gain:+.4f} over the last {tail} "
          f"rounds. " + ("It has levelled off." if converged else
                         "It is still moving: consider more rounds (--rounds).")]
(out / "tuning.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6, 4))
for (lr, e), curve in curves.items():
    ax.plot(range(1, len(curve) + 1), curve, label=f"lr {lr}, {e} ep",
            linewidth=2.5 if (lr, e) == best["fedavg"] else 1,
            alpha=1 if (lr, e) == best["fedavg"] else 0.5)
ax.set_xlabel("server round")
ax.set_ylabel("validation AUC, mean over hospitals")
ax.set_title("FedAvg tuning (bold = chosen)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / "tuning_curve.png", dpi=150)
print(f"saved {out / 'tuning_curve.png'}")
