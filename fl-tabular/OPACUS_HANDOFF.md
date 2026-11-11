# Adding DP-SGD (Opacus)

Notes for whoever adds differential privacy. Setup and the experiments that
already exist are in `README.md`; this is only about where DP goes.

## Where it goes: `fltab/privacy.py`

Fill in `DPSGDTrainer` in `fltab/privacy.py`. That's the only code to write:
the scripts already call it when given `--dp`, and its docstring lists what
each method has to do. Until it's filled in, `--dp` stops with
`NotImplementedError`.

Every in-process experiment trains through a trainer, one per hospital (or one
for pooled), created once before round 1 and kept for the whole run:

```python
trainer = make_trainer(loader, lr, privacy)   # SGDTrainer, or DPSGDTrainer with --dp
trainer.train(model, steps)                   # each round: train `model` in place
trainer.epsilon()                             # after the last round
```

Because the trainer lives for the whole run, the `PrivacyEngine` inside it
counts every step the hospital takes across all rounds. `privacy` carries σ
(`--noise-multiplier`), C (`--max-grad-norm`) and δ (`--delta`).

| script | with `--dp` |
|---|---|
| `local_steps.py` | T sweep, FedAvg and pooled both DP; ε per hospital in `local_steps_dp_s<σ>_c<C>.md` |
| `lr_T_grid.py` | lr × T grid, both DP; ε in `lr_T_grid_dp_s<σ>_c<C>.md` |
| `baselines.py` | pooled and local, DP; logged as `pooled_dp_s<σ>_c<C>` etc. so `summarize.py` puts them next to the non-DP rows; ε printed |

Outputs get the `_dp_s<σ>_c<C>` suffix, so DP runs never overwrite the non-DP
results. Without `--dp`, every script trains exactly as before (checked
bit-for-bit against the previous code).

Not wired yet: `tune.py` (rebuilds each hospital's loader every round to match
Flower's seeding, so it has no per-hospital object to keep an accountant in)
and the Flower apps (see below).

Model: FLamby's `Baseline`, a single `nn.Linear(13, 1)` + sigmoid, which
Opacus supports as is. Loss: `BaselineLoss` (BCE, mean reduction). Batch
size: 4 (FLamby's `BATCH_SIZE`).

## Opacus details

- `make_private` wraps the model in a `GradSampleModule`, so its
  `state_dict()` keys gain a `_module.` prefix. Load and send through
  `model._module.state_dict()` / `model._module.load_state_dict(...)` so the
  server's averaging (keys `fc.weight`, `fc.bias`) still lines up.
- Opacus's loader uses Poisson sampling: batch sizes vary around 4 and can be
  0. Don't skip the empty ones: call backward and `opt.step()` anyway (skipping
  backward makes `opt.step()` raise "Per sample gradient is not initialized").
- The Poisson batches come from the loader's own seeded generator and the
  noise from torch's global RNG, which `set_seed` seeds, so a run is
  reproducible per seed.

ε is reported per hospital after the last round (`trainer.epsilon()`). The
hospitals are very different sizes (train n = 199, 172, 30, 85), so the same
noise and steps cost very different ε: Switzerland's sample rate is 4/30 vs
4/199 at Cleveland. Pick δ below 1/n for the smallest hospital (e.g. 1e-3
for n = 30, or use one δ well below that everywhere).

## In Flower

`client_app.py` still trains with plain `train_sgd`. It is a stateless
function called once per round, so a trainer created inside it would reset
every round. Either keep engines in a
module-level dict of trainers keyed by partition-id (works in the simulation, since each
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
- `baselines.py --dp` already logs through `harness.log(...)` with the DP
  settings in the method name, so `summarize.py` picks it up next to the
  baselines. Do the same for anything new.
- Tune lr with DP on. The best lr without noise is usually not the best with
  it. `lr_T_grid.py --dp` shows the trend quickly but scores on the test set;
  pick final settings on the validation split (`part="fit"` / `"val"`, see
  `tune.py`).
- Read balanced accuracy and AUC, not just accuracy: under heavy noise the
  model tends to collapse to one label, which accuracy hides (0.938 at
  Switzerland for always saying "disease").
