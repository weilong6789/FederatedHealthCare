# Roadmap: privacy cost of federated learning on Fed-Heart-Disease

## Design goal

Measure how much model quality four hospitals give up to get a formal privacy
guarantee, then test whether a better noise design gets some of it back.

One sentence for the PI: **for FedAvg on Fed-Heart-Disease, plot AUC against
ε under record-level and hospital-level DP, and show whether correlated noise
beats independent noise at the same ε.**

### Research questions

1. **Cost.** How far does AUC fall from the non-private FedAvg result
   (pooled ≈ FedAvg ≈ 0.805 overall AUC) as ε shrinks?
2. **Unit of privacy.** Record-level DP (protects one patient, DP-SGD at each
   hospital) vs hospital-level DP (protects a whole hospital's dataset, noise
   on the server). Which one costs less at a comparable ε, and which one is
   the right promise for a hospital consortium?
3. **Who pays.** The hospitals have 199 / 172 / 30 / 85 training patients.
   Does the same noise hurt Switzerland and Long Beach VA far more than
   Cleveland?
4. **Improvement.** Does correlated noise across rounds (matrix-factorization
   / DP-FTRL style) reach higher AUC than independent per-round noise at the
   same ε and δ?
5. **Reality check.** Does a membership inference attack get weaker as ε
   shrinks, and does the measured attack strength stay under what ε allows?

Questions 1 and 2 are the minimum deliverable. 3 is nearly free once 1 runs.
4 is the novel part. 5 is the validation that makes the claims credible.

## Where we are

Done (see `README.md`, `results/`):

- Data, model, harness, and metrics shared by every method (`fltab/harness.py`).
- `local`, `pooled`, `fedavg` baselines with tuned hyperparameters, 5 seeds,
  bootstrap CIs.
- FedAvg ≈ pooled across local steps T and lr × T (`local_steps.py`,
  `lr_T_grid.py`): the non-private reference is solid.
- A written plan for where Opacus plugs in (`OPACUS_HANDOFF.md`).

Not done: any DP code. Every privacy number in the project is still zero.

Known limits to plan around:

- Switzerland has 16 test patients, one of them label-0. Its balanced
  accuracy and AUC are noise. Report overall and per-hospital, and lean on
  the pooled 254-patient score for headline curves.
- Accuracy hides collapse to one label under heavy noise. Headline metric is
  AUC; balanced accuracy is the second check.
- The model is 14 parameters. That keeps runs fast but means conclusions are
  about a very small model. State this in the writeup rather than claim it
  generalizes to deep models.

## Fixed experimental protocol

Decide these once so every phase is comparable. Changing one later means
re-running everything after it.

| item | choice |
|---|---|
| ε grid | 0.25, 0.5, 1, 2, 4, 8, ∞ (no noise) |
| δ | 1e-3 at record level (below 1/30, the smallest hospital); 1e-3 at hospital level (4 hospitals, so δ matters differently, see Phase 3) |
| accountant | Opacus RDP accountant for DP-SGD; same accountant family for server-side so ε is comparable |
| clipping | tune C on the validation split, never on test |
| tuning | lr, rounds, C tuned per ε on `part="fit"/"val"` as in `tune.py` |
| seeds | 0–4, same splits as baselines |
| headline metric | AUC on all 254 test patients, plus per-hospital AUC and balanced accuracy |
| logging | `harness.log(...)` with new method names (`fedavg_dp_record`, `fedavg_dp_client`, `fedavg_corr`) and ε in the tag, so `summarize.py` picks them up |

## Phases

Each phase has a deliverable and an exit check. Do not start the next phase
until the check passes.

### Phase 1: privacy definition and threat model (≈ 1 week)

- Write half a page: what is being protected (a patient record, or a
  hospital's dataset), from whom (the server, other hospitals, an outside
  observer of the final model), what the adversary sees (every round's
  update, or only the final model).
- Fix the ε grid and δ above and get the PI's sign-off.
- Check the Opacus and Flower versions install together in the `flamby` env.

Deliverable: `PRIVACY_SPEC.md`. Exit check: PI agrees with the privacy unit
and the ε/δ choices.

### Phase 2: record-level DP-SGD (≈ 2 weeks)

Follow `OPACUS_HANDOFF.md`.

- One `PrivacyEngine` per hospital that persists across all rounds, so ε
  accumulates over the run instead of resetting.
- Run in-process like `local_steps.py` (`simulate_fedavg`) rather than
  through Flower, for speed over the ε grid.
- Log each hospital's own ε. Small hospitals get larger ε for the same noise;
  decide whether the target ε is per hospital or the worst case, and say which
  in every plot.
- Tune lr and C per ε on the validation split.

Deliverables: `dp_record.py`, `results/dp_record.md`, ε-vs-AUC plot with the
non-private line, per-hospital ε table.
Exit check: at ε = ∞ (noise 0, clip huge) the code reproduces the existing
FedAvg result within seed noise. If it does not, the DP code has a bug.

### Phase 3: hospital-level DP through Flower (≈ 2 weeks)

- Wrap the strategy in `DifferentialPrivacyServerSideFixedClipping`
  (already named in `README.md`).
- With only 4 clients, hospital-level DP is hard: each round's update is one
  of four, so noise that protects one hospital swamps the average. Expect
  this to cost much more than record level. That is a result, not a failure;
  report it.
- Compute ε over rounds with the same accountant family as Phase 2.

Deliverables: `dp_client.py`, ε-vs-AUC plot overlaid on Phase 2's.
Exit check: the Phase 2 and Phase 3 curves are on the same axes with the
privacy unit stated in the legend.

### Phase 4: who pays (≈ 1 week)

- Per-hospital AUC and balanced accuracy at each ε, for both privacy units.
- Compare against the `local` baseline at each hospital: at what ε does
  joining the federation stop beating training alone?

Deliverable: `results/dp_per_hospital.md` and a heatmap (hospital × ε).
Exit check: a clear statement of the form "below ε = x, hospital y is better
off training alone."

### Phase 5: correlated noise (≈ 3 weeks, the novel part)

- Implement a correlated-noise mechanism across rounds (start with the
  simplest: a banded matrix-factorization / DP-FTRL tree mechanism applied to
  the sequence of clipped updates).
- Independent-noise results from Phases 2–3 are the baseline. Same ε, δ,
  clip, rounds, seeds.
- Sweep rounds, since correlated noise should help most when there are many
  rounds.

Deliverables: `dp_corr.py`, ε-vs-AUC plot with independent vs correlated.
Exit check: either correlated noise wins at some ε by more than the paired
bootstrap CI, or we report that it does not for a model this small. Both are
publishable outcomes; the check is that we can say which.

### Phase 6: empirical privacy audit (≈ 2 weeks)

- Run a membership inference attack (loss-threshold first, shadow-model
  version if time allows) against the final model at each ε.
- Plot attack AUC against ε. It should shrink as ε shrinks and stay below the
  bound ε implies.

Deliverable: `audit.py`, `results/audit.md`.
Exit check: no point on the curve exceeds the theoretical bound; if one
does, find the accounting bug before anything else.

### Phase 7: writeup (≈ 2 weeks)

- Figures: the AUC-vs-ε plots from Phases 2, 3, 5; per-hospital heatmap;
  audit curve.
- One table: method × privacy unit × ε → AUC with CI.
- Limitations section: tiny model, 4 hospitals, one dataset, 16-patient
  Switzerland test set.

Deliverable: report/paper draft. Exit check: PI review.

## Timeline

≈ 13 weeks from Phase 1 start if phases run back to back. Phases 4 and 6 can
overlap the phase before them once their inputs exist.

| phase | weeks | blocks |
|---|---|---|
| 1 privacy spec | 1 | everything |
| 2 record-level DP | 2 | 4, 5, 6 |
| 3 hospital-level DP | 2 | 4 |
| 4 who pays | 1 | 7 |
| 5 correlated noise | 3 | 7 |
| 6 audit | 2 | 7 |
| 7 writeup | 2 | |

If time is short, cut in this order: Phase 6 shrinks to loss-threshold
only, then Phase 5 drops to a smaller sweep. Phases 1–4 and 7 are the core.

## Open decisions for the PI

1. Is the headline privacy unit the patient or the hospital? This changes
   which plot is the main figure.
2. Is a target ε per hospital acceptable, or must one ε hold for the whole
   consortium?
3. Is a single dataset enough, or should another FLamby tabular dataset be
   added as a second case (extra cost: about 2 weeks)?
4. Is the 14-parameter logistic regression acceptable as the only model?
