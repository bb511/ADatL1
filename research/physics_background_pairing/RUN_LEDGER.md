# Run ledger

Last updated: 2026-08-04 (Europe/Zurich)

## Contract and inputs

- Plan: `research/physics_background_pairing/PLAN.md`
- Pair artifacts:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183`
- Validation/test prefix per source: 163,840 events (10 x 16,384)
- Pairings found for both splits: `flat_physical`, `physics_summary`,
  `typed_sliced_wasserstein`, `jetclr`
- Superseded pilot campaign commit: `ebe2ce1`; the expanded campaign will be
  reinitialized from the post-audit commit before any search is submitted.

## Status

| Phase | State | Evidence / next action |
|---|---|---|
| Scientific contract | complete | `PLAN.md` |
| Implementation audit | complete | Physics bases and HPO spaces match `origin/dev/patrick`; score routing, CDF, retraining, and padding semantics audited |
| Code and configs | complete | Expanded 56-study matrix and mask-aware AE/VAE OAS are implemented and validated after the remote merge |
| Tests and preflight | complete | 49 post-merge tests passed, including all 56 configurations; all 8 pair tables are authenticated |
| Pilots | superseded | Three pre-mask AE/OAS pilots completed; fresh pilots from the audited commit are required |
| Full searches | pending | 56 persistent studies, 600 trials each; five CAP pairings including CDF, with AE MSE/OAS and VAE KL/OAS as separate cells |
| Retraining/evaluation | pending | Automated freeze gate and 200-epoch indexed retraining are implemented |
| Aggregation/report | pending | Authenticated paper-style downstream oracle aggregation is implemented; awaits results |

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

## Verification log

- 2026-08-04: all six base physics experiment configs are byte-identical to
  `origin/dev/patrick` at `9c2c4ec`; all six HPO search spaces are identical apart
  from the campaign's explicit SQLite lock timeout.
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
