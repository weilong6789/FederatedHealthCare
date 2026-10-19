"""Shared experiment harness for Fed-Heart-Disease.

Every method (pooled, local, FedAvg) uses the SAME data splits,
model, optimizer, evaluation, and logging from this file, so the
comparisons are fair.
"""
import csv
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
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
    return {
        "accuracy": float(metric(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, (p > 0.5).astype(int))),
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
