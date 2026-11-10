"""Pooled, local and FedAvg on FLamby's Fed-ISIC2019, on one GPU.

    python isic.py --method pooled --seed 0
    python isic.py --method fedavg --T 100 --seed 0
    python isic.py --method local --seed 0
    python isic.py --method pooled --epochs 1 --max-images 512    # quick smoke test

Six hospitals, 8 skin-lesion classes, about 23k dermoscopy images. Model,
loss, batch size and augmentations are FLamby's: EfficientNet-b0 pretrained on
ImageNet, weighted focal loss, batch 64, random crops of 200 px.

Training budget is counted in pooled-equivalent epochs, as in
fl-tabular/local_steps.py: one epoch = the SGD steps one pass over the pooled
training data takes. Pooled runs that many steps. FedAvg runs budget / T
rounds; each round every hospital starts from the global model, takes T steps
on its own data, and the server averages the six models weighted by hospital
size. Local trains one model per hospital for --epochs passes over its own
data only.

Every method uses the same optimizer (--opt, default Adam at FLamby's lr
5e-4). FedAvg starts a fresh optimizer at each hospital every round, so Adam's
moment estimates never cross between hospitals.

Writes, under results/:
    isic.csv              test scores along the way, per hospital and over all patients
    isic_predictions.csv  final model's class probabilities per test image
"""
import argparse
import csv
import fcntl
import os
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from flamby.datasets.fed_isic2019 import (BATCH_SIZE, LR, NUM_CLIENTS, Baseline,
                                          BaselineLoss, FedIsic2019)

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["pooled", "fedavg", "local"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=float, default=20, help="pooled-equivalent epochs (FLamby: 20)")
p.add_argument("--T", type=int, default=100, help="FedAvg local steps per round")
p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
p.add_argument("--lr", type=float, default=LR)
p.add_argument("--eval-every", type=float, default=1.0, help="evaluate about this often, in epochs")
p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)),
               help="data-loading processes (shared by all hospitals' loaders)")
p.add_argument("--max-images", type=int, default=None,
               help="cap every train and test set at this many images (smoke tests)")
p.add_argument("--data-path", default=None,
               help="folder holding ISIC_2019_Training_Input_preprocessed "
                    "(default: where download_isic.py recorded it)")
p.add_argument("--out", default="results")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
TAG = args.method + (f"_T{args.T}" if args.method == "fedavg" else "")
print(f"{TAG} seed {args.seed} on {DEVICE}"
      + (f" ({torch.cuda.get_device_name()})" if DEVICE.type == "cuda" else ""), flush=True)

random.seed(args.seed)  # FLamby picks its CoarseDropout hole count with `random`
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ---------------------------------------------------------------- data
def dataset(center, train):
    ds = FedIsic2019(center=center, train=train, pooled=False, data_path=args.data_path)
    if args.max_images and len(ds) > args.max_images:
        ds = Subset(ds, range(args.max_images))
    return ds


TRAIN = [dataset(i, True) for i in range(NUM_CLIENTS)]
TEST = [dataset(i, False) for i in range(NUM_CLIENTS)]
SIZES = [len(d) for d in TRAIN]
print("train images per hospital:", SIZES, " test:", [len(d) for d in TEST], flush=True)

# FedAvg keeps all six hospitals' loaders open at once, so the workers are
# split between them; pooled and local only ever have one open.
n_loaders = NUM_CLIENTS if args.method == "fedavg" else 1
WORKERS = max(1, args.workers // n_loaders)


def train_loader(ds, seed):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                      num_workers=WORKERS, persistent_workers=True, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


TEST_LOADERS = [DataLoader(d, batch_size=128, num_workers=args.workers, pin_memory=True)
                for d in TEST]


def batches(loader):
    """Endless minibatches, reshuffled at every pass."""
    while True:
        yield from loader


# ---------------------------------------------------------------- model
def make_model():
    torch.manual_seed(args.seed)  # same initial head for every method at a given seed
    return Baseline(pretrained=True).to(DEVICE)


def make_opt(model):
    if args.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


LOSS = BaselineLoss()


def train_steps(model, it, opt, steps):
    model.train()
    for _ in range(steps):
        xb, yb = next(it)
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        LOSS(model(xb), yb).backward()
        opt.step()


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def scores(y, p):
    """Balanced accuracy is FLamby's metric for ISIC. AUC is one-vs-rest,
    averaged over the classes that have both positives and negatives here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y, p.argmax(1))
    aucs = [roc_auc_score(y == c, p[:, c]) for c in range(p.shape[1])
            if 0 < (y == c).sum() < len(y)]
    return {"accuracy": float((p.argmax(1) == y).mean()), "balanced_accuracy": float(bal),
            "auc": float(np.mean(aucs)) if aucs else float("nan"), "n": len(y)}


@contextmanager
def locked(path):
    """Append to `path` holding an exclusive lock, so jobs running at once
    don't splice their lines together. Falls back to an unlocked append on a
    filesystem without locks; summarize.py drops any spliced rows."""
    with open(path, "a", newline="") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
        except OSError:
            pass
        f.seek(0, os.SEEK_END)
        yield f
        f.flush()
    # closing the file releases the lock


FIELDS = ["method", "T", "seed", "round", "steps", "epochs", "client",
          "accuracy", "balanced_accuracy", "auc", "n", "lr", "opt", "seconds"]
RESULTS = OUT / "isic.csv"
T0 = time.time()


def log(model, round_, steps, epochs, hospitals=range(NUM_CLIENTS), final=False):
    preds = {i: predict(model, TEST_LOADERS[i]) for i in hospitals}
    rows = [{"client": str(i), **scores(*preds[i])} for i in hospitals]
    if len(preds) > 1:
        y = np.concatenate([preds[i][0] for i in preds])
        pr = np.concatenate([preds[i][1] for i in preds])
        rows.append({"client": "all", **scores(y, pr)})
    with locked(RESULTS) as f:
        w = csv.DictWriter(f, FIELDS)
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow({"method": args.method, "T": args.T if args.method == "fedavg" else "",
                        "seed": args.seed, "round": round_, "steps": steps,
                        "epochs": round(epochs, 3), "lr": args.lr, "opt": args.opt,
                        "seconds": round(time.time() - T0), **r})
    last = rows[-1]
    print(f"  epoch {epochs:6.2f}  round {round_:4d}  {last['client']:>4}: "
          f"bal acc {last['balanced_accuracy']:.3f}  auc {last['auc']:.3f}  "
          f"({time.time() - T0:.0f}s)", flush=True)
    if final:
        with locked(OUT / "isic_predictions.csv") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["method", "seed", "client", "image", "label",
                            *[f"p{c}" for c in range(8)]])
            for i, (y, pr) in preds.items():
                for k, (yk, pk) in enumerate(zip(y, pr)):
                    w.writerow([TAG, args.seed, i, k, int(yk), *[f"{v:.5f}" for v in pk]])


# ---------------------------------------------------------------- methods
STEPS_PER_EPOCH = -(-sum(SIZES) // BATCH_SIZE)
TOTAL = max(1, round(args.epochs * STEPS_PER_EPOCH))
EVAL_EVERY = max(1, round(args.eval_every * STEPS_PER_EPOCH))
print(f"{STEPS_PER_EPOCH} steps per pooled epoch, {TOTAL} steps in total", flush=True)

if args.method == "pooled":
    model = make_model()
    opt = make_opt(model)
    it = batches(train_loader(ConcatDataset(TRAIN), args.seed))
    log(model, 0, 0, 0)
    done = 0
    while done < TOTAL:
        n = min(EVAL_EVERY, TOTAL - done)
        train_steps(model, it, opt, n)
        done += n
        log(model, 0, done, done / STEPS_PER_EPOCH, final=done >= TOTAL)

elif args.method == "fedavg":
    model = make_model()
    local = make_model()
    its = [batches(train_loader(d, args.seed * 1000 + i)) for i, d in enumerate(TRAIN)]
    rounds = -(-TOTAL // args.T)
    weights = [n / sum(SIZES) for n in SIZES]
    print(f"{rounds} rounds of {args.T} local steps", flush=True)
    log(model, 0, 0, 0)
    next_eval = EVAL_EVERY
    for r in range(1, rounds + 1):
        avg = None
        for it, wgt in zip(its, weights):
            local.load_state_dict(model.state_dict())
            train_steps(local, it, make_opt(local), args.T)
            sd = local.state_dict()
            if avg is None:
                avg = {k: v.float() * wgt for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    avg[k] += v.float() * wgt
        # BatchNorm's running stats are averaged like the weights; its integer
        # batch counter is cast back to its own type.
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in avg.items()})
        steps = r * args.T
        if steps >= next_eval or r == rounds:
            log(model, r, steps, steps / STEPS_PER_EPOCH, final=r == rounds)
            next_eval = (steps // EVAL_EVERY + 1) * EVAL_EVERY

else:  # local: each hospital alone, --epochs passes over its own data
    for i, d in enumerate(TRAIN):
        model = make_model()
        opt = make_opt(model)
        it = batches(train_loader(d, args.seed * 1000 + i))
        own_epoch = -(-len(d) // BATCH_SIZE)
        total = max(1, round(args.epochs * own_epoch))
        print(f"hospital {i}: {total} steps", flush=True)
        done = 0
        while done < total:
            n = min(max(1, round(args.eval_every * own_epoch)), total - done)
            train_steps(model, it, opt, n)
            done += n
            log(model, 0, done, done / own_epoch, hospitals=[i], final=done >= total)
        it.close()  # shuts down this hospital's loader workers before the next one starts

print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)
