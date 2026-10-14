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
from torch.utils.data import DataLoader
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
def train_loader(center=None, pooled=False, seed=0):
    if pooled:  # all 4 hospitals' training data concatenated
        ds = FedHeartDisease(train=True, pooled=True)
    else:
        ds = FedHeartDisease(center=center, train=True, pooled=False)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)


@lru_cache(maxsize=1)
def test_loaders():
    """Each hospital's own test set, plus the pooled test set."""
    loaders = {
        str(i): DataLoader(FedHeartDisease(center=i, train=False, pooled=False),
                           batch_size=BATCH_SIZE)
        for i in range(NUM_CLIENTS)
    }
    loaders["pooled"] = DataLoader(FedHeartDisease(train=False, pooled=True),
                                   batch_size=BATCH_SIZE)
    return loaders


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
def evaluate(model, loader):
    """Accuracy, balanced accuracy and AUC on one test set.

    The model outputs P(disease). Accuracy (FLamby's `metric`) and balanced
    accuracy threshold it at 0.5; AUC ranks patients by it directly. Always
    predicting one label scores 0.5 on both balanced accuracy and AUC, however
    skewed the test set is. AUC is NaN on a test set with only one label.
    """
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        ys.append(yb)
        preds.append(model(xb))
    y = torch.cat(ys).numpy().reshape(-1).astype(int)
    p = torch.cat(preds).numpy().reshape(-1)
    return {
        "accuracy": float(metric(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, (p > 0.5).astype(int))),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }


def evaluate_all(model):
    """All metrics on every hospital's test set and on the pooled test set."""
    return {name: evaluate(model, dl) for name, dl in test_loaders().items()}


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
