# Experiment Implementation Status

This file separates implementation readiness from scientific evidence. A passing
smoke test means the code path is executable; it does not replace a full-data,
multi-seed result.

## Current State

| Area | Implementation | Evidence currently available | What is still required |
| --- | --- | --- | --- |
| Analytical (`analytical.md`) | Implemented and locally smoke-tested, with `paper` and `smoke` profiles and a declared artifact inventory | The smoke profile generates 21 declared artifacts plus metadata; regression tests cover theory transitions and controls | Run the `paper` profile once in the final environment and freeze the resulting bundle |
| Causal Chamber (`cchamber.md`) | Implemented and locally smoke-tested for AE, VAE, SVDD, RealNVP; metadata, random, and frozen-encoder pairing paths exist | All 20 paper configs compose; all four model demos and a production-shaped AE training/evaluation run pass on the public Causal Chamber CSVs | Run the shared candidate pool, at least three retraining seeds, all real interventions, and final aggregation |
| Pairing (`pairing.md`) | Implemented with a versioned, fingerprinted table contract and distinct validation/test consumption | Controlled smoke: 77 validation/80 test pairs. Real Causal Chamber canary: 1,000 validation/1,000 test pairs, training CAP, evaluator validation/test, and one real intervention | Build and audit final Causal Chamber and physics tables across encoder seeds; report coverage, balance, distances, and overlap |
| Physics L1 | Config/generation/evaluation implementation is present | Local checkpoints are smoke/debug artifacts only | Supply `RAW_DATA_DIR`, run the generated matrix on the cluster, and aggregate multi-seed intervention results |
| CIFAR-10 | Config/generation implementation is present | Demo training works and downloads automatically | Run all intended normal-class conditions and reporting seeds |
| RobustAD | Config/generation implementation is present | Local artifacts are smoke/debug only | Run all intended subsets and reporting seeds; do not treat the saturated PCB smoke as final evidence |
| Paper selection/reporting | Implemented as explicit CSV/JSON contracts | Selection, checkpoint resolution, callback collection, paired aggregation, plots, and provenance are regression tested | Populate the contracts from completed cluster studies |
| Cloud handoff | Portable generated scripts and a strict preflight gate are implemented | All 76 exact sweep compositions and all 76 generated Slurm shell scripts pass locally | Pass the gate from a clean target-cluster checkout with real L1 parquet data and real Causal Chamber pair tables; then run a one-job GPU canary |

The generator currently contains 76 core experiment specifications when the
optional CVaR-10 ablation is excluded: 24 physics, 20 Causal Chamber, 16 CIFAR-10,
and 16 RobustAD.

## What Is Reproducible Locally

Run:

```bash
make analytical-smoke
make smoke
make preflight-local
uv run pytest -k "not slow"
```

`make smoke` combines the analytical artifact smoke, the four-model/three-seed
checkpoint-and-reporting smoke, and the full pairing producer/consumer smoke. The
model smoke uses the same training callbacks, four checkpoint branches,
validation/test evaluator, raw metric CSV contract, seed-aware aggregation, and
plot/report code as the cluster workflow.

The smoke-test numbers are diagnostic only. Short training can select equivalent
weights for CAP, drift, Wasserstein, and last-epoch branches; this verifies the
handoff but says nothing about which criterion wins on real data.

## Cluster Workflow

1. Generate and run sweeps with `scripts/generation.py`.
2. Export a long-form candidate table with the exact columns documented in
   `cchamber.md`. Reuse the same candidate IDs across compared label-free
   strategies.
3. Run `paper_pipeline.py select`. It rejects downstream/oracle strategies,
   inconsistent parameters, duplicate rows, and non-shared candidate pools.
4. Generate retraining scripts from `retrain_manifest.json` and run at least three
   paired seeds.
5. Resolve the selected strategy checkpoints, generate final evaluation scripts,
   and evaluate every intervention.
6. Create a collection manifest pointing at callback `values.csv` files.
7. Run `collect` and `aggregate`. The output includes normalized long-form data,
   seed summaries, intervention summaries, paired strategy differences,
   deterministic bootstrap intervals, figures, a report, and SHA-256 provenance.

## Next Actions, In Order

1. Run `make smoke`, the non-slow test suite, and `make preflight-local` on the
   commit intended for deployment.
2. Freeze the analytical paper bundle and record the commit plus metadata hash.
3. Run the Causal Chamber shared-pool study first. It needs no private physics
   data and is the quickest real validation of the claimed pairing ordering.
4. Inspect Causal Chamber pair quality before looking at anomaly performance.
   Reject encoder tables with poor coverage/balance or unstable seed overlap.
5. Run `make preflight-cloud` from a clean target checkout, followed by one
   generated GPU canary job. This gate requires both Causal Chamber pair tables
   and every configured L1 parquet directory.
6. Run the physics pairing encoder and fixed-table audit once `RAW_DATA_DIR` is
   available.
7. Launch the selected physics, CIFAR-10, and RobustAD retraining/evaluation
   matrices with paired seeds.
8. Treat the generated `paper/report.md` files as the evidence gate: do not
   promote smoke checkpoints or one-seed summaries into the manuscript.
