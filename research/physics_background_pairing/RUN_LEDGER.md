# Run ledger

Last updated: 2026-08-04 (Europe/Zurich)

## Contract and inputs

- Plan: `research/physics_background_pairing/PLAN.md`
- Pair artifacts:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183`
- Validation/test prefix per source: 163,840 events (10 x 16,384)
- Pairings found for both splits: `flat_physical`, `physics_summary`,
  `typed_sliced_wasserstein`, `jetclr`
- Frozen debug campaign commit: `6d10cd5601ffa86111a12415b521c4e104c261e6`.
- Superseded pilot campaigns at `ebe2ce1` and `bb0f981` are archived in scratch;
  neither is eligible for production submission.

## Status

| Phase | State | Evidence / next action |
|---|---|---|
| Scientific contract | complete | `PLAN.md` |
| Implementation audit | complete | Model/data/HPO settings match `dev/patrick`; requested metric scope, score routing, CDF, retraining, and padding semantics audited |
| Code and configs | complete | Refactored to six shared model studies with 56 logical fronts, per-strategy checkpoint selection, and score-aware retraining |
| Tests and preflight | complete | 45 targeted tests pass, all six search/retrain configs compose, all changed-file hooks pass, and all 8 pair tables are authenticated |
| Pilots | pending | Earlier isolated pilots passed from `6d10cd5`, but they do not validate the new unified callback/evaluator path |
| Full searches | pending | Six persistent studies, 600 trials each; no full-scale job is authorized or submitted yet |
| Retraining/evaluation | pending | Offline 56-front freeze, unique 200-epoch retraining, and native/OAS threshold separation are implemented; validation pending |
| Aggregation/report | pending | Score-aware downstream oracle aggregation is implemented; awaits validation and results |

## Decisions

- The ordinary combined validation data remains available and unchanged; source-specific
  loaders are additional model-selection views.
- W1 is run once per model because it is invariant to a bijective reordering of the same
  two score samples.
- Primary drift direction is 2025E to 2025G. The reverse direction is a diagnostic.
- AE residual OAS state must be checkpointed and fitted from a deterministic bounded
  training-normal sample so it is practical during every validation epoch.
- AE and VAE each retain the paper-native anomaly score as a direct control: AE MSE
  versus residual OAS, and VAE latent KL versus residual OAS.
- All logical strategies for one model share the same 600 sampled configurations. This
  gives six trained studies rather than 56 and enables direct strategy comparisons;
  logical two-dimensional fronts are reconstructed offline.

## Verification log

- 2026-08-04: implemented the six shared 600-trial model studies. Every trial now
  returns all native/OAS CAP, W1, drift, and native-secondary objective pairs in a
  frozen order; the 56 logical two-dimensional fronts are reconstructed offline.
- 2026-08-04: added independent native/OAS validation thresholds for downstream AE/VAE
  evaluation, restricted search validation to normal plus the two source backgrounds,
  and kept all 20 anomaly datasets held out until 200-epoch retraining.
- 2026-08-04: the new shared evaluator, campaign, score thresholds, configurations,
  pairing paths, and masked OAS contracts passed 45 targeted tests. All six final
  configs recomposed after formatting, and every changed-file pre-commit hook passed.

- 2026-08-04: model, trainer, data, and HPO settings match the `dev/patrick` physics
  setup at `9c2c4ec`; the later posterior-consistency callback is intentionally absent
  because this campaign's requested primary metrics are CAP, W1, and drift. The six
  HPO spaces otherwise match, with the SQLite lock timeout supplied by the campaign.
- 2026-08-04: added paper-native/OAS score factoring for AE and VAE, an empirical-CDF
  CAP control, and preserved the selected pairing during Pareto retraining.
- 2026-08-04: OAS fitting and scoring now treat padded coordinates as missing, using
  observed-only locations, neutral mean imputation for covariance fitting, and
  observed-feature-only event energies.
- 2026-08-04: expanded targeted suite passed (40 tests), including composition of all
  56 primary search cells; pre-commit passed on every changed file.
- 2026-08-04: merged the newer `origin/research/main` JetCLR and mapping-table work,
  preserved the CDF/native/OAS campaign contract, and passed the expanded post-merge
  suite (49 tests).
- 2026-08-04: a real-data pilot exposed one structurally absent physics coordinate.
  Residual OAS now fits only the globally observed feature subspace, embeds inactive
  precision rows/columns as zero, and retains observed-feature event normalization.
  The expanded targeted suite passed 52 tests and all pre-commit hooks.
- 2026-08-04: fresh frozen-commit debug pilots passed end to end: AE residual-OAS CAP
  with the authenticated `physics_summary` table (`3000424`, 9m59s), VAE
  residual-OAS CAP with the table-free CDF control (`3000425`, 10m12s), and AE
  residual-OAS W1 (`3000500`, 7m58s). Each produced its expected selected checkpoint,
  evaluator artifacts, optimized-metric artifact, and success marker.

- 2026-08-03: all six paper-model configurations composed with the JetCLR variant.
- 2026-08-03: CAP, Wasserstein, and drift search-only overlays composed independently.
- 2026-08-03: targeted callback, AE-score, pairing, configuration, provenance, and
  campaign suite passed (25 tests).
- 2026-08-03: pre-commit passed on every changed file.
- 2026-08-03: validation and test tables for all four strategies authenticated against
  the exact ordered 163,840-event source-loader tensors. See `preflight.json`.
- 2026-08-03: initial pilot submission was rejected with
  `QOSMaxSubmitJobPerUserLimit`; no job was created and no unrelated queued work was
  cancelled. Pilot and chunked search submission scripts are ready for the first free
  scheduler slots.
- 2026-08-03: paper protocol rechecked from the supplied PDF. Search evaluation now
  retains all three required checkpoint candidates. The campaign freezes every Pareto
  trial, retrains it for 200 epochs, and performs the paper's explicitly labelled
  downstream-oracle selection only after the Pareto manifest is immutable.
