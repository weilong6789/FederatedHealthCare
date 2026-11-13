# Adding DP-SGD (Opacus)

Notes for whoever adds differential privacy. Setup and the experiments that
already exist are in `README.md`; this is only about where DP goes.

## Where training happens

All SGD in the project goes through two functions in `fltab/harness.py`:

| function | used by | trains for |
|---|---|---|
| `train_sgd(model, loader, lr, epochs)` | `baselines.py`, `tune.py`, `client_app.py` (Flower FedAvg) | whole epochs |
| `train_steps(model, batch_iter, lr, steps)` | `pooled_steps` / `fedavg_steps` (used by `local_steps.py`, `lr_K_grid.py`) | a fixed number of minibatch steps |

Both build a fresh `torch.optim.SGD` on every call, which is fine for plain
SGD (it has no state) but not for Opacus: the privacy accountant lives with
the `PrivacyEngine`, so each hospital needs one engine that persists across
all its rounds, or ε will be counted per round instead of in total.

Model: FLamby's `Baseline`, a single `nn.Linear(13, 1)` + sigmoid, which
Opacus supports as is. Loss: `BaselineLoss` (BCE, mean reduction). Batch
size: 4 (FLamby's `BATCH_SIZE`).

## Suggested shape

Per hospital, once, before round 1:

```python
engine = PrivacyEngine()
model, opt, loader = engine.make_private(
    module=make_model(), optimizer=torch.optim.SGD(params, lr=lr),
    data_loader=train_loader(center=i, seed=...),
    noise_multiplier=sigma, max_grad_norm=C)
```

Each round: load the global weights into it, train, send weights back. Two
things to watch:

- `make_private` wraps the model in a `GradSampleModule`, so its
  `state_dict()` keys gain a `_module.` prefix. Load and send through
  `model._module.state_dict()` / `model._module.load_state_dict(...)` so the
  server's averaging (keys `fc.weight`, `fc.bias`) still lines up.
- Opacus's loader uses Poisson sampling: batch sizes vary around 4 and can be
  0. `batches()` / `train_steps` work with it unchanged.

Report `engine.get_epsilon(delta)` per hospital after the last round. The
hospitals are very different sizes (train n = 199, 172, 30, 85), so the same
noise and steps cost very different ε: Switzerland's sample rate is 4/30 vs
4/199 at Cleveland. Pick δ below 1/n for the smallest hospital (e.g. 1e-3
for n = 30, or use one δ well below that everywhere).

## In Flower

`client_app.py` is a stateless function called once per round, so an engine
created inside it would reset every round. Either keep engines in a
module-level dict keyed by partition-id (works in the simulation, since each
client runs in a long-lived process, but check this), or run DP experiments
in-process like `tune.py` / `local_steps.py` do (`simulate_fedavg` there
reproduces Flower's FedAvg exactly), which is simpler and much faster for a
grid of ε values.

Server-side DP (Flower's `DifferentialPrivacyServerSideFixedClipping`,
mentioned in `README.md`) is a different guarantee: it clips and noises each
hospital's whole update once per round, protecting a hospital's dataset
rather than individual patients. It needs no Opacus at all.

## What to keep fixed so results compare

- Same splits, model, evaluation and logging: use `harness.py`, don't copy it.
- Log through `harness.log(...)` with a new method name (e.g. `fedavg_dp`)
  and the ε in the tag, so `summarize.py` picks it up next to the baselines.
- Tune lr with DP on, on the validation split (`part="fit"` / `"val"`, see
  `tune.py`). The best lr without noise is usually not the best with it.
- Read balanced accuracy and AUC, not just accuracy: under heavy noise the
  model tends to collapse to one label, which accuracy hides (0.938 at
  Switzerland for always saying "disease").
