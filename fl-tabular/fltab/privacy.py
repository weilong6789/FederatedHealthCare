"""The one place differential privacy plugs in.

Every in-process experiment (baselines.py, local_steps.py, lr_T_grid.py) trains
through a *trainer*: one object per hospital (or one for pooled), created once
before training and kept for the whole run, so a privacy accountant inside it
counts every step the hospital ever takes, not one round's worth.

    trainer = make_trainer(loader, lr, privacy)   # once per hospital
    trainer.train(model, steps)                   # every round: train `model` in place
    trainer.epsilon()                             # after the last round

`privacy` is None for plain SGD, or a `Privacy` with the DP-SGD settings. The
scripts build it from command-line flags (see `add_privacy_args`), so

    python local_steps.py --dp --noise-multiplier 1.0 --max-grad-norm 1.0

runs the T sweep with DP-SGD at every hospital and in pooled, writes ε next to
every score, and keeps the non-DP outputs (the files get a `_dp_...` suffix).

To add DP, fill in `DPSGDTrainer` below. Nothing else needs to change. Notes on
Opacus specifics (the `_module.` prefix, Poisson batches, choosing δ) are in
OPACUS_HANDOFF.md.
"""
from dataclasses import dataclass

import torch  # noqa: F401  (torch, BaselineLoss, make_model: for DPSGDTrainer)

from fltab.harness import BATCH_SIZE, BaselineLoss, batches, make_model, train_steps  # noqa: F401


@dataclass(frozen=True)
class Privacy:
    noise_multiplier: float    # σ: Gaussian noise std, as a multiple of max_grad_norm
    max_grad_norm: float       # C: each patient's gradient is clipped to this L2 norm
    delta: float = 1e-3        # δ for reporting ε; keep it below 1/n of the smallest hospital (30)

    def tag(self):
        """Suffix for method names and output files, e.g. `_dp_s1.0_c1.0`."""
        return f"_dp_s{self.noise_multiplier:g}_c{self.max_grad_norm:g}"

    def describe(self):
        return f"DP-SGD σ={self.noise_multiplier:g}, C={self.max_grad_norm:g}, δ={self.delta:g}"


def steps_per_epoch(loader):
    """Minibatch steps in one pass over `loader`'s data at the nominal batch size.
    Use this rather than len(loader): a DP loader's length is its expected
    number of Poisson batches, which a trainer is free to change."""
    return -(-len(loader.dataset) // BATCH_SIZE)


class SGDTrainer:
    """Plain minibatch SGD, exactly what the project ran before DP existed.

    Keeps its place in `loader` between calls, so T steps then T more steps
    draw the same batches as 2T steps in one go."""

    def __init__(self, loader, lr):
        self.lr = lr
        self.batches = batches(loader)

    def train(self, model, steps):
        """Train `model` in place for `steps` minibatch SGD steps."""
        train_steps(model, self.batches, self.lr, steps)
        return model

    def epsilon(self):
        """Privacy spent so far. None: no DP guarantee at all."""
        return None


class DPSGDTrainer:
    """DP-SGD with Opacus. TODO: implement; until then `--dp` stops with an error.

    Must behave like SGDTrainer from the outside:

    __init__(loader, lr, privacy)
        Runs once per hospital, before round 1. Build the private model,
        optimizer and loader here and keep them, along with the PrivacyEngine
        (its accountant must see every step of every round):

            self.engine = PrivacyEngine()
            self.model, self.opt, dp_loader = self.engine.make_private(
                module=make_model(),
                optimizer=torch.optim.SGD(<that model's params>, lr=lr),
                data_loader=loader,
                noise_multiplier=privacy.noise_multiplier,
                max_grad_norm=privacy.max_grad_norm)
            self.batches = batches(dp_loader)

    train(model, steps)
        Copy `model`'s weights into the private model
        (self.model._module.load_state_dict(model.state_dict())), take `steps`
        DP-SGD steps on self.batches with BaselineLoss, copy the weights back
        into `model` (model.load_state_dict(self.model._module.state_dict())).
        `model` itself must stay a plain `Baseline`: the server averages and
        evaluates it.
        Opacus's loader samples each patient independently (Poisson), so some
        batches are empty: still call backward and opt.step() on them, the
        noise is added either way and the accountant expects every step.

    epsilon()
        self.engine.get_epsilon(self.privacy.delta)
    """

    def __init__(self, loader, lr, privacy):
        raise NotImplementedError(
            "DP-SGD is not implemented yet: fill in DPSGDTrainer in fltab/privacy.py "
            "(see its docstring and OPACUS_HANDOFF.md)")

    def train(self, model, steps):
        raise NotImplementedError

    def epsilon(self):
        raise NotImplementedError


def make_trainer(loader, lr, privacy=None):
    """One trainer per hospital (or for pooled), kept for the whole run."""
    if privacy is None:
        return SGDTrainer(loader, lr)
    return DPSGDTrainer(loader, lr, privacy)


# ---------------------------------------------------------------- command line
def add_privacy_args(parser):
    g = parser.add_argument_group("differential privacy (needs DPSGDTrainer in fltab/privacy.py)")
    g.add_argument("--dp", action="store_true", help="train with DP-SGD instead of plain SGD")
    g.add_argument("--noise-multiplier", type=float, default=1.0, help="σ")
    g.add_argument("--max-grad-norm", type=float, default=1.0, help="C, per-patient clipping norm")
    g.add_argument("--delta", type=float, default=1e-3, help="δ for reporting ε")


def privacy_from_args(args):
    """A Privacy from the flags above, or None without --dp."""
    if not args.dp:
        return None
    return Privacy(args.noise_multiplier, args.max_grad_norm, args.delta)
