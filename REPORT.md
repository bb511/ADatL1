# Physics background-pairing confirmation: 200-epoch primary results

## Executive answer

The complete primary campaign contains **84/84 authenticated 200-epoch retraining
runs**: 28 model/selector configurations, three fixed seeds, and 20 downstream
anomaly datasets. Efficiency is mean signal efficiency at 250 Hz; AUPRC is mean
average precision. The optional two-seed extension was cancelled for cost after
partial runs reached epochs 40–80. Those partial trajectories are excluded from
every number below.

1. **Does CAP beat W1 and drift?** CDF-CAP is directionally better in **11/12
   efficiency contrasts**, with an equal-model macro advantage of +1.16 pp over
   drift and +0.80 pp over W1. It is not universally better: after multiplicity
   adjustment only SVDD CAP versus drift is a meaningful efficiency improvement.
   AUPRC is mixed: CAP is better in **5/12 contrasts**, with DSVAE CAP versus drift
   meaningfully better and AE CAP versus W1 meaningfully worse.
2. **Does learned pairing improve CAP over CDF pairing?** **Not overall.** Only
   6/24 model/pairing efficiency contrasts and 9/24 AUPRC contrasts are positive;
   none is an adjusted meaningful improvement. Every fixed learned pairing has a
   negative macro-average delta versus CDF on both endpoints.
3. **Does OAS help AE/VAE?** No. Padding-masked OAS improves 0/14 matched
   efficiency cells and 2/14 AUPRC cells. The native MSE/KL scores remain better.
4. **Are results better than the paper?** The changed background0/background1
   system is directionally higher in **11/18 method-matched efficiency cells**.
   This compares two complete systems; it does not causally isolate the dataset
   change. The paper provides no matched physics AUPRC.

The strongest defensible conclusion is therefore: **CDF-CAP has a broad
directional efficiency advantage over W1/drift, but learned pairings do not
improve CDF-CAP robustly, and CAP does not dominate AUPRC.**

## Complete primary matrix

Each cell is `efficiency / AUPRC`, averaged over three seeds and 20 anomalies.
Bold marks the efficiency winner and AUPRC winner independently within each row.
Flat, Summary, Typed, JetCLR, and CDF are CAP pairings; W1 and Drift are unpaired
selectors.

| Model | Flat CAP | Summary CAP | Typed CAP | JetCLR CAP | CDF CAP | W1 | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE | 4.11% / 0.611 | 4.11% / 0.611 | 4.11% / 0.611 | **5.24%** / 0.602 | 4.37% / 0.592 | 3.73% / **0.620** | 4.11% / 0.611 |
| VAE | **4.75%** / 0.254 | 4.58% / 0.534 | 4.64% / 0.239 | 4.58% / 0.534 | 4.46% / 0.592 | 4.52% / **0.618** | 4.41% / 0.607 |
| DSAE | 7.53% / 0.589 | 5.71% / 0.591 | 5.71% / 0.591 | 5.71% / 0.591 | **7.71%** / 0.588 | 7.53% / 0.589 | 5.39% / **0.594** |
| DSVAE | 2.90% / 0.505 | **5.14%** / 0.068 | 1.64% / 0.348 | 1.02% / 0.219 | 4.66% / **0.657** | 3.24% / 0.544 | 4.18% / 0.522 |
| DeepSVDD | 5.46% / 0.253 | 6.09% / 0.263 | 6.09% / 0.263 | 6.09% / 0.263 | **6.63% / 0.278** | 5.07% / 0.255 | 4.54% / 0.257 |
| RealNVP | **3.96%** / 0.369 | 1.95% / 0.359 | 2.43% / **0.390** | 2.29% / 0.176 | **3.96%** / 0.369 | 2.94% / 0.342 | 2.20% / 0.385 |

The apparent DSVAE Summary efficiency winner has AUPRC 0.068, illustrating why
the two endpoints must be reported together rather than cherry-picking efficiency.

## Pairing versus the CDF control

Equal-model macro deltas for each fixed learned pairing relative to CDF-CAP:

| Learned pairing | Efficiency delta | AUPRC delta | Interpretation |
|---|---:|---:|---|
| Flat physical | -0.513 pp | -0.082 | Worse on average |
| Physics summary | -0.702 pp | -0.108 | Worse on average |
| Typed sliced-Wasserstein | -1.194 pp | -0.106 | Worse on average |
| JetCLR | -1.143 pp | -0.115 | Worse on average |

There are isolated directional gains—notably AE JetCLR efficiency (+0.872 pp)
and DSVAE Summary efficiency (+0.477 pp)—but their adjusted intervals remain
inconclusive. No learned pairing establishes a robust improvement over CDF for
any model/endpoint contrast. The result does **not** support the general claim
that learned pairing improves CAP.

## CDF-CAP versus W1 and drift

The controlled macro contrasts average models equally:

| Contrast | Endpoint | Mean delta | Adjusted classification |
|---|---|---:|---|
| CDF-CAP − drift | Efficiency | +1.162 pp | Positive, not multiplicity-adjusted meaningful |
| CDF-CAP − W1 | Efficiency | +0.796 pp | Positive, not multiplicity-adjusted meaningful |
| CDF-CAP − drift | AUPRC | +0.017 | Positive, not multiplicity-adjusted meaningful |
| CDF-CAP − W1 | AUPRC | +0.018 | Inconclusive |

At model level, the adjusted meaningful findings are:

- SVDD: CDF-CAP improves efficiency over drift by +2.096 pp.
- DSVAE: CDF-CAP improves AUPRC over drift by +0.136.
- AE: CDF-CAP degrades AUPRC versus W1 by -0.029.

Everything else is inconclusive or practically equivalent after the predeclared
multiple-comparison rules. CAP therefore looks promising for low-rate efficiency,
but “CAP always beats W1 and drift” would overstate the evidence.

## Masked OAS diagnostic

OAS was evaluated on the **same native-selected checkpoints**, so this isolates
the anomaly-score change rather than running a second checkpoint oracle. The OAS
residual covariance and score exclude padded/nonexistent physics objects.

| Model | Flat CAP | Summary CAP | Typed CAP | JetCLR CAP | CDF CAP | W1 | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE OAS | **3.05%** / 0.466 | **3.05%** / 0.466 | **3.05%** / 0.466 | 2.70% / **0.488** | 1.93% / 0.480 | 2.04% / 0.476 | **3.05%** / 0.466 |
| VAE OAS | **3.96% / 0.545** | 2.87% / 0.495 | 3.67% / 0.530 | 2.87% / 0.495 | 2.77% / 0.526 | 2.62% / 0.514 | 2.57% / 0.512 |

Relative to native scoring, OAS is higher in 0/14 efficiency cells and only two
VAE AUPRC cells (Flat and Typed), both because their native AUPRC is unusually
poor. It should remain a diagnostic rather than the primary score.

## Method-matched comparison with the paper

The paper values are efficiency-only. CDF-CAP is matched to paper CAP, forward
drift to paper Stable, and W1 to paper W1.

| Model | CAP: paper → confirmation | W1: paper → confirmation | Stable/Drift: paper → confirmation |
|---|---:|---:|---:|
| AE | 5.73% → 4.37% | 3.22% → 3.73% | 4.29% → 4.11% |
| VAE | 4.79% → 4.46% | 4.49% → 4.52% | 4.39% → 4.41% |
| DSAE | 8.55% → 7.71% | 3.75% → 7.53% | 3.57% → 5.39% |
| DSVAE | 5.02% → 4.66% | 0.80% → 3.24% | 4.85% → 4.18% |
| DeepSVDD | 4.03% → 6.63% | 3.25% → 5.07% | 3.56% → 4.54% |
| RealNVP | 3.26% → 3.96% | 1.66% → 2.94% | 3.17% → 2.20% |

Eleven of these 18 cells improve. CAP itself improves for DeepSVDD and RealNVP,
W1 improves for all six models, and Drift improves for VAE, DSAE, and DeepSVDD.
The datasets, search formulation, and selection system differ from the paper, so
this is a full-system benchmark—not proof that background0/background1 alone
caused the gains.

## Why the 200-epoch result supersedes the earlier report

Of 72 predeclared selector/pairing endpoint contrasts, only 40 retained their
50-epoch direction and 28 reversed at the three-seed 200-epoch endpoint. Validation
objectives improved favorably from epoch 50 to 200 in only 13/42 panels. The old
50-epoch screening results were useful for selecting candidates, but they were not
stable enough to support the final conclusions above.

## Scope, limitations, and artifacts

- Primary evidence: 84 sealed runs, 126 checkpoint selections, 5,040 native and
  1,680 OAS endpoint rows; seeds 271828, 314159, and 161803.
- The reserve rule requested two additional seeds for AE, DSVAE, DeepSVDD, and
  RealNVP. Those jobs were stopped for cost; partial epoch-40–80 results are
  excluded. This is therefore complete for the three-seed primary campaign, but
  not the originally planned five-seed confirmatory extension.
- The random-CAP control in the full bundle is an authenticated discovery
  diagnostic only. A new confirmation random-CAP selector was not run after the
  cost stop, and no primary conclusion depends on it.
- Downstream anomalies were inspected during discovery, so this establishes
  200-epoch and seed stability rather than performance on a wholly untouched
  external holdout.
- Paper AUPRC is unavailable and is not inferred.

The detailed confidence intervals, per-model contrasts, robustness statistics,
training curves, random-control diagnostics, and machine-readable evidence are in
[the full report](results/physics_background_pairing_confirmation/primary_cost_stopped_4d031ab/REPORT.md)
and [report bundle](results/physics_background_pairing_confirmation/primary_cost_stopped_4d031ab/).

Provenance:

- Confirmation analysis SHA-256: `11c104ee1a65d4950eb47308e20d61c85329b06830a49d54760c51f48fb03bbd`
- Report summary SHA-256: `e04907395ceacb42fcde57738fae3038b1d43b3fa5e10194eacb8aa2b377a20b`
- Reserve cost-stop receipt SHA-256: `543dbfa2c113c6201b1b1644fcf119b2578ae5df1b2922df6236734e67eb6b85`
- Report generator commit: `4d031abe6e4057f803602a4073839ef7495330e8`
