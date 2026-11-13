"""Shared experiment harness for Fed-Heart-Disease.

Every method (pooled, local, FedAvg) uses the SAME data splits,
model, optimizer, evaluation, and logging from this file, so the
comparisons are fair.
"""
import csv
import os
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)

# `flwr run` executes a packaged copy of this file elsewhere, so a path
# relative to __file__ would point at that copy, not the project. Flower keeps
# the working directory, though, so run everything from the project root
# (or set FLTAB_ROOT to it).
ROOT = Path(os.environ.get("FLTAB_ROOT", Path.cwd()))
RESULTS_PATH = ROOT / "results" / "results.csv"
PREDICTIONS_PATH = ROOT / "results" / "predictions.csv"
METRICS = ["accuracy", "balanced_accuracy", "auc"]
FIELDS = ["timestamp", "commit", "method", "seed", "round", "client",
          *METRICS, "bytes_uploaded", "lr", "local_epochs"]


# ---------------------------------------------------------------- setup
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_model():
    return Baseline()  # FLamby's reference model: identical for every method


# ---------------------------------------------------------------- data
# Hyperparameters are tuned on a validation slice of each hospital's training
# data, never on the test set. The slice is fixed (VAL_SEED), independent of
# the run seed, so every method and seed tunes on the same patients.
VAL_FRACTION = 0.2
VAL_SEED = 12345


def _split(ds, part):
    """Indices of `ds` in `part`: "full" (all of it), "fit" (80% of each
    hospital's patients) or "val" (the other 20%). Stratified by label where a
    hospital has both labels. A hospital's k-th patient is the same patient in
    its own dataset and in the pooled one, so both see the same split."""
    if part == "full":
        return list(range(len(ds)))
    by_center = {}
    for i, c in enumerate(ds.centers):
        by_center.setdefault(c, []).append(i)
    keep = []
    for idx in by_center.values():
        y = [int(ds[i][1].item()) for i in idx]
        fit, val = train_test_split(idx, test_size=VAL_FRACTION, random_state=VAL_SEED,
                                    stratify=y if len(set(y)) == 2 else None)
        keep += fit if part == "fit" else val
    return sorted(keep)


@lru_cache(maxsize=None)  # FLamby re-reads and re-normalises the CSVs on every load
def _dataset(center, pooled, train, part="full"):
    if pooled:  # all 4 hospitals concatenated
        ds = FedHeartDisease(train=train, pooled=True)
    else:
        ds = FedHeartDisease(center=center, train=train, pooled=False)
    return ds if part == "full" else Subset(ds, _split(ds, part))


def train_loader(center=None, pooled=False, seed=0, part="full"):
    """Training data. part="fit" holds out the validation slice (for tuning);
    the default "full" trains on all of it (for the final runs)."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_dataset(center, pooled, True, part), batch_size=BATCH_SIZE,
                      shuffle=True, generator=g)


@lru_cache(maxsize=2)
def eval_loaders(split="test"):
    """Each hospital's own test (or validation) set, plus the pooled one.
    Never shuffled, so patient k is the same patient for every model."""
    def loader(center, pooled):
        ds = (_dataset(center, pooled, False) if split == "test"
              else _dataset(center, pooled, True, "val"))
        return DataLoader(ds, batch_size=BATCH_SIZE)
    loaders = {str(i): loader(i, False) for i in range(NUM_CLIENTS)}
    loaders["pooled"] = loader(None, True)
    return loaders


def test_loaders():
    return eval_loaders("test")


# ---------------------------------------------------------------- training
def train_sgd(model, loader, lr, epochs):
    """Plain SGD. Used by every method."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = BaselineLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def batches(loader):
    """Endless minibatches from `loader`, reshuffled at every pass."""
    while True:
        yield from loader


def train_steps(model, batch_iter, lr, steps):
    """Plain SGD for a fixed number of minibatch steps (not epochs), drawing
    from `batch_iter` (see `batches`). Same optimizer and loss as train_sgd."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = BaselineLoss()
    model.train()
    for _ in range(steps):
        xb, yb = next(batch_iter)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    return model


# FLamby scales each patient's features with their own hospital's training
# mean/std, except with pooled=True, which uses all hospitals' training data
# together. FedAvg can only ever see the per-hospital version. To compare
# FedAvg with pooled on the same inputs, pooled_steps trains on the four
# per-hospital datasets concatenated, and both are scored on
# all_patients_test(): the 254 test patients, each scaled by their hospital.
@lru_cache(maxsize=None)
def _all_patients(train):
    return ConcatDataset([_dataset(i, False, train) for i in range(NUM_CLIENTS)])


def all_patients_test():
    """All 254 test patients, each scaled with their own hospital's statistics."""
    return DataLoader(_all_patients(False), batch_size=BATCH_SIZE)


def pooled_steps(seed, lr, total, eval_every=None, on_eval=None):
    """Pooled SGD for `total` minibatch steps on all hospitals' training data,
    scaled per hospital (see above). Calls on_eval(round=0, steps, model) at
    step 0, every `eval_every` steps and at the end."""
    set_seed(seed)
    model = make_model()
    it = batches(DataLoader(_all_patients(True), batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(seed)))
    if on_eval:
        on_eval(0, 0, model)
    done, chunk = 0, eval_every or total
    while done < total:
        n = min(chunk, total - done)
        train_steps(model, it, lr, n)
        done += n
        if on_eval:
            on_eval(0, done, model)
    return model


def fedavg_steps(seed, K, lr, total, eval_every=None, on_eval=None):
    """FedAvg where every hospital takes K minibatch SGD steps per round, for
    ceil(total / K) rounds, averaged weighted by hospital size (as Flower's
    FedAvg does). Same initial model as pooled_steps with the same seed. Calls
    on_eval(round, steps, model) at round 0, at the first round end past each
    multiple of `eval_every` steps, and at the last round."""
    set_seed(seed)
    model = make_model()
    loaders = [train_loader(center=i, seed=seed * 1000 + i) for i in range(NUM_CLIENTS)]
    its = [batches(dl) for dl in loaders]            # each hospital keeps its place
    sizes = [len(dl.dataset) for dl in loaders]
    local = make_model()
    rounds = -(-total // K)
    if on_eval:
        on_eval(0, 0, model)
    next_eval = eval_every or total
    for r in range(1, rounds + 1):
        states = []
        for it in its:
            local.load_state_dict(model.state_dict())
            train_steps(local, it, lr, K)
            states.append({k: v.clone() for k, v in local.state_dict().items()})
        model.load_state_dict({k: sum(s[k] * n for s, n in zip(states, sizes)) / sum(sizes)
                               for k in states[0]})
        steps = r * K
        if on_eval and (steps >= next_eval or r == rounds):
            on_eval(r, steps, model)
            next_eval = (steps // (eval_every or total) + 1) * (eval_every or total)
    return model


def steps_per_pooled_epoch():
    """SGD steps in one pass over the pooled training data (122 at batch 4)."""
    return -(-len(_all_patients(True)) // BATCH_SIZE)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, loader):
    """True labels and predicted P(disease), one entry per patient."""
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        ys.append(yb)
        preds.append(model(xb))
    return (torch.cat(ys).numpy().reshape(-1).astype(int),
            torch.cat(preds).numpy().reshape(-1))


def evaluate(model, loader):
    """Accuracy, balanced accuracy and AUC on one test set.

    The model outputs P(disease). Accuracy (FLamby's `metric`) and balanced
    accuracy threshold it at 0.5; AUC ranks patients by it directly. Always
    predicting one label scores 0.5 on both balanced accuracy and AUC, however
    skewed the test set is. AUC is NaN on a test set with only one label.
    """
    y, p = predict(model, loader)
    with warnings.catch_warnings():  # one-label sets (Switzerland) warn every call
        warnings.simplefilter("ignore")
        bal = float(balanced_accuracy_score(y, (p > 0.5).astype(int)))
    return {
        "accuracy": float(metric(y, p)),
        "balanced_accuracy": bal,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }


def evaluate_all(model, split="test"):
    """All metrics on every hospital's test (or validation) set and the pooled one."""
    return {name: evaluate(model, dl) for name, dl in eval_loaders(split).items()}


def predict_hospitals(model):
    """Per-patient test predictions at every hospital (for bootstrap intervals)."""
    return {str(i): predict(model, eval_loaders("test")[str(i)])
            for i in range(NUM_CLIENTS)}


def array_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


# ---------------------------------------------------------------- logging
def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def log(method, seed, results, round_=-1, bytes_uploaded=0, lr=None, local_epochs=None):
    """Append one row per client to results/results.csv.

    `results` maps client -> the metrics dict from `evaluate`.
    """
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for client, metrics in results.items():
            w.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": _git_commit(), "method": method, "seed": seed,
                "round": round_, "client": client,
                **{k: round(metrics[k], 4) for k in METRICS},
                "bytes_uploaded": bytes_uploaded, "lr": lr,
                "local_epochs": local_epochs,
            })


def save_predictions(method, seed, preds):
    """Append per-patient test predictions to results/predictions.csv.

    `preds` maps hospital -> (labels, probabilities) from `predict`. Only final
    models are saved; summarize.py bootstraps over these patients.
    """
    PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
    new_file = not PREDICTIONS_PATH.exists()
    with open(PREDICTIONS_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["method", "seed", "client", "patient", "label", "prob"])
        for client, (y, p) in preds.items():
            for k, (yk, pk) in enumerate(zip(y, p)):
                w.writerow([method, seed, client, k, int(yk), f"{pk:.6f}"])
