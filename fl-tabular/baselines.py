"""Pooled (ceiling) and local-only (floor) baselines. No Flower needed.

    python baselines.py --seeds 0 1 2 3 4 --pooled-lr 0.01 --local-lr 0.01 --epochs 50
    python baselines.py --dp --noise-multiplier 1.0 --max-grad-norm 1.0   # DP-SGD

With --dp both train through DPSGDTrainer (fltab/privacy.py); method names get a
`_dp_s<σ>_c<C>` suffix, so summarize.py shows them next to the non-DP ones, and
each model's ε is printed.
"""
import argparse

from fltab.harness import (NUM_CLIENTS, evaluate, evaluate_all, log, make_model,
                              predict, predict_hospitals, save_predictions,
                              set_seed, test_loaders, train_loader)
from fltab.privacy import add_privacy_args, make_trainer, privacy_from_args, steps_per_epoch

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--lr", type=float, default=0.01, help="default for both methods")
p.add_argument("--pooled-lr", type=float, help="overrides --lr for pooled")
p.add_argument("--local-lr", type=float, help="overrides --lr for local")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--tag", default="", help="suffix for method names, e.g. _lr0.1")
add_privacy_args(p)
args = p.parse_args()
PRIV = privacy_from_args(args)
TAG = args.tag + (PRIV.tag() if PRIV else "")


def train(loader, lr):
    """args.epochs passes over `loader`'s data, plain SGD or DP-SGD."""
    model = make_model()
    trainer = make_trainer(loader, lr, PRIV)
    trainer.train(model, args.epochs * steps_per_epoch(loader))
    eps = trainer.epsilon()
    return model, ("" if eps is None else f" ε={eps:.2f} (δ={PRIV.delta:g})")


pooled_lr = args.pooled_lr or args.lr
local_lr = args.local_lr or args.lr

for seed in args.seeds:
    # Pooled: one model trained on every hospital's data, tested on each hospital
    set_seed(seed)
    model, eps = train(train_loader(pooled=True, seed=seed), pooled_lr)
    results = evaluate_all(model)
    log("pooled" + TAG, seed, results, lr=pooled_lr, local_epochs=args.epochs)
    save_predictions("pooled" + TAG, seed, predict_hospitals(model))
    print(f"seed {seed} pooled{eps}: " + ", ".join(
        f"{k} acc={v['accuracy']:.3f} auc={v['auc']:.3f}" for k, v in results.items()))

    # Local-only: each hospital trains alone, tested on its own test set
    for i in range(NUM_CLIENTS):
        set_seed(seed)
        model, eps = train(train_loader(center=i, seed=seed), local_lr)
        res = evaluate(model, test_loaders()[str(i)])
        log("local" + TAG, seed, {str(i): res}, lr=local_lr, local_epochs=args.epochs)
        save_predictions("local" + TAG, seed,
                         {str(i): predict(model, test_loaders()[str(i)])})
        print(f"seed {seed} local client {i}{eps}: acc={res['accuracy']:.3f} "
              f"auc={res['auc']:.3f}")
