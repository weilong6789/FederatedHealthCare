"""Pooled (ceiling) and local-only (floor) baselines. No Flower needed.

    python baselines.py --seeds 0 1 2 3 4 --lr 0.01 --epochs 50
"""
import argparse

from fltab.harness import (NUM_CLIENTS, evaluate, evaluate_all, log, make_model,
                              set_seed, test_loaders, train_loader, train_sgd)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--tag", default="", help="suffix for method names, e.g. _lr0.1 while tuning")
args = p.parse_args()

for seed in args.seeds:
    # Pooled: one model trained on every hospital's data, tested on each hospital
    set_seed(seed)
    model = train_sgd(make_model(), train_loader(pooled=True, seed=seed), args.lr, args.epochs)
    accs = evaluate_all(model)
    log("pooled" + args.tag, seed, accs, lr=args.lr, local_epochs=args.epochs)
    print(f"seed {seed} pooled: " + ", ".join(f"{k}={v:.3f}" for k, v in accs.items()))

    # Local-only: each hospital trains alone, tested on its own test set
    for i in range(NUM_CLIENTS):
        set_seed(seed)
        model = train_sgd(make_model(), train_loader(center=i, seed=seed), args.lr, args.epochs)
        acc = evaluate(model, test_loaders()[str(i)])
        log("local" + args.tag, seed, {str(i): acc}, lr=args.lr, local_epochs=args.epochs)
        print(f"seed {seed} local client {i}: {acc:.3f}")
