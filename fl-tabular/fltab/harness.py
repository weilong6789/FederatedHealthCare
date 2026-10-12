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
FIELDS = ["timestamp", "commit", "method", "seed", "round", "client",
          "accuracy", "bytes_uploaded", "lr", "local_epochs"]


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
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        ys.append(yb)
        preds.append(model(xb))
    return float(metric(torch.cat(ys).numpy(), torch.cat(preds).numpy()))


def evaluate_all(model):
    """Accuracy on every hospital's test set and on the pooled test set."""
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


def log(method, seed, accs, round_=-1, bytes_uploaded=0, lr=None, local_epochs=None):
    """Append one row per client to results/results.csv."""
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for client, acc in accs.items():
            w.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": _git_commit(), "method": method, "seed": seed,
                "round": round_, "client": client, "accuracy": round(acc, 4),
                "bytes_uploaded": bytes_uploaded, "lr": lr,
                "local_epochs": local_epochs,
            })
