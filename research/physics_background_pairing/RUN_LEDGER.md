# Run ledger

Last updated: 2026-08-03 (Europe/Zurich)

## Contract and inputs

- Plan: `research/physics_background_pairing/PLAN.md`
- Pair artifacts:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183`
- Validation/test prefix per source: 163,840 events (10 x 16,384)
- Pairings found for both splits: `flat_physical`, `physics_summary`,
  `typed_sliced_wasserstein`, `jetclr`
- Current branch at initialization: `research/main`, commit `9b6f14b`

## Status

| Phase | State | Evidence / next action |
|---|---|---|
| Scientific contract | complete | `PLAN.md` |
| Implementation audit | complete | Confirmed prior prototype covered only mapped VAE CAP |
| Code and configs | complete | Six model overlays; CAP/W1/E-to-G drift search overlays; AE canonical residual OAS score |
| Tests and preflight | complete | 22 targeted tests passed; all 8 pair tables authenticated in `preflight.json` |
| Pilots | queued for capacity | First submission attempt was rejected by the per-user QOS job cap; resumable three-metric pilot array is ready |
| Full searches | pending | 36 persistent studies, 600 trials each |
| Retraining/evaluation | pending | Freeze Pareto candidates first |
| Aggregation/report | pending | Must cover full matrix |

## Decisions

- The ordinary combined validation data remains available and unchanged; source-specific
  loaders are additional model-selection views.
- W1 is run once per model because it is invariant to a bijective reordering of the same
  two score samples.
- Primary drift direction is 2025E to 2025G. The reverse direction is a diagnostic.
- AE residual OAS state must be checkpointed and fitted from a deterministic bounded
  training-normal sample so it is practical during every validation epoch.

## Verification log

- 2026-08-03: all six paper-model configurations composed with the JetCLR variant.
- 2026-08-03: CAP, Wasserstein, and drift search-only overlays composed independently.
- 2026-08-03: targeted callback, AE-score, pairing, configuration, and campaign suite
  passed (22 tests).
- 2026-08-03: pre-commit passed on every changed file.
- 2026-08-03: validation and test tables for all four strategies authenticated against
  the exact ordered 163,840-event source-loader tensors. See `preflight.json`.
- 2026-08-03: initial pilot submission was rejected with
  `QOSMaxSubmitJobPerUserLimit`; no job was created and no unrelated queued work was
  cancelled. Pilot and chunked search submission scripts are ready for the first free
  scheduler slots.
