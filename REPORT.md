# Background-pairing experiment: efficiency and AUPRC results

## Executive answer

The two research questions now have endpoint-complete answers. Here, **efficiency**
is mean signal efficiency at the 250 Hz operating point, and **AUPRC** is mean
average precision over the same 20 downstream anomaly datasets.

1. **Does CAP beat W1 and drift?** With the controlled CDF pairing and native scores,
   CAP has the best macro-average on both endpoints: **5.99% efficiency / 0.537
   AUPRC**, versus **4.12% / 0.496** for W1 and **4.66% / 0.512** for drift. Per
   algorithm, CDF-CAP wins AUPRC for **5/6** models and efficiency for **3/6**;
   W1 wins efficiency for the other three, while drift wins AUPRC only for DSVAE.
2. **Does learned pairing improve CAP?** Not universally. Relative to CDF-CAP,
   the best learned pairing improves efficiency for VAE, DSVAE, DeepSVDD, and
   RealNVP, ties DSAE, and hurts AE. It improves AUPRC for DSVAE and DeepSVDD,
   ties DSAE and RealNVP, and hurts AE and VAE. The clearest joint gain is
   DeepSVDD; DSVAE also improves both endpoints, but with different best pairings.
3. **Does OAS help AE/VAE?** No overall. Every one of the 14 matched OAS results
   has lower efficiency than its native-score counterpart, and 12/14 also have
   lower AUPRC. Native MSE/KL remains the primary choice.

The main scientific result is therefore more nuanced than the efficiency-only
result: **CAP is the strongest controlled selector overall, while learned pairing
can improve the low-rate operating point without necessarily improving global
ranking quality.**

## Complete result table

Each campaign cell is **efficiency % / AUPRC**. The four learned-pairing columns
and CDF are CAP selectors. W1 and drift do not consume event pairings. Paper
references are efficiency-only (`CAP / W1 / Stable`); the paper did not report a
directly comparable physics AUPRC, so no paper AUPRC is invented here.

| Model | Score | Paper efficiency: CAP / W1 / Stable | Flat CAP | Physics-summary CAP | Typed-sliced CAP | JetCLR CAP | CDF CAP | W1 | Drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AE | native MSE | 5.73 / 3.22 / 4.29% | 4.13% / 0.617 | 4.13% / 0.617 | 4.13% / 0.617 | 4.21% / 0.623 | 4.71% / 0.628 | 4.77% / 0.619 | 4.13% / 0.617 |
| AE | masked OAS | not reported | 2.64% / 0.501 | 2.75% / 0.501 | 2.66% / 0.498 | 3.09% / 0.481 | 2.64% / 0.501 | 2.37% / 0.452 | 2.37% / 0.462 |
| VAE | native latent KL | 4.79 / 4.49 / 4.39% | 5.36% / 0.512 | 5.21% / 0.308 | 5.36% / 0.512 | 5.21% / 0.308 | 4.53% / 0.634 | 4.67% / 0.632 | 4.67% / 0.632 |
| VAE | masked OAS | not reported | 2.78% / 0.457 | 2.78% / 0.457 | 2.78% / 0.457 | 2.78% / 0.457 | 2.91% / 0.494 | 4.25% / 0.544 | 2.82% / 0.483 |
| DSAE | native reconstruction | 8.55 / 3.75 / 3.57% | 6.10% / 0.601 | 5.35% / 0.585 | 5.35% / 0.585 | 5.35% / 0.585 | 6.10% / 0.601 | 6.15% / 0.599 | 5.35% / 0.585 |
| DSVAE | native variational | 5.02 / 0.80 / 4.85% | 6.92% / 0.626 | 6.01% / 0.039 | 7.40% / 0.633 | 6.78% / 0.649 | 6.84% / 0.627 | 0.84% / 0.533 | 6.39% / 0.672 |
| DeepSVDD | native center distance | 4.03 / 3.25 / 3.56% | 8.28% / 0.315 | 7.23% / 0.290 | 7.23% / 0.290 | 7.23% / 0.290 | 8.27% / 0.303 | 5.48% / 0.256 | 4.65% / 0.234 |
| RealNVP | native likelihood | 3.26 / 1.66 / 3.17% | 5.50% / 0.431 | 7.57% / 0.387 | 3.57% / 0.406 | 3.32% / 0.333 | 5.50% / 0.431 | 2.78% / 0.333 | 2.78% / 0.333 |

These are not 32 isolated configurations. The analysis authenticates **421 unique
physical checkpoints**, **601 checkpoint/score evaluations**, and **591 Pareto-front
memberships** across 56 logical fronts. The table reports one efficiency-selected
checkpoint per model, score, and selector/pairing combination.

## Research question 1: CAP versus W1 and drift

### Controlled comparison: CDF-CAP versus unpaired selectors

CDF is the prespecified control and therefore the cleanest comparison, without a
best-of-four pairing advantage.

| Native-score model | Best efficiency selector | Best AUPRC selector | CDF-CAP | W1 | Drift |
|---|---|---|---:|---:|---:|
| AE | W1 | CDF-CAP | 4.71% / 0.628 | 4.77% / 0.619 | 4.13% / 0.617 |
| VAE | W1 | CDF-CAP | 4.53% / 0.634 | 4.67% / 0.632 | 4.67% / 0.632 |
| DSAE | W1 | CDF-CAP | 6.10% / 0.601 | 6.15% / 0.599 | 5.35% / 0.585 |
| DSVAE | CDF-CAP | drift | 6.84% / 0.627 | 0.84% / 0.533 | 6.39% / 0.672 |
| DeepSVDD | CDF-CAP | CDF-CAP | 8.27% / 0.303 | 5.48% / 0.256 | 4.65% / 0.234 |
| RealNVP | CDF-CAP | CDF-CAP | 5.50% / 0.431 | 2.78% / 0.333 | 2.78% / 0.333 |
| **Macro mean** | **CDF-CAP** | **CDF-CAP** | **5.99% / 0.537** | **4.12% / 0.496** | **4.66% / 0.512** |

The efficiency result is a tie in model wins but not in macro performance: CAP's
large gains on DSVAE, DeepSVDD, and RealNVP outweigh W1's small gains on AE, VAE,
and DSAE. The AUPRC result is stronger: CDF-CAP wins five models, with DSVAE the
only exception.

If the best learned pairing is allowed after looking at efficiency, paired CAP wins
efficiency for 4/6 native models. That is a useful system result, but it is
exploratory because CAP receives four learned candidates while W1 and drift each
receive one.

## Research question 2: does pairing improve CAP?

The table below compares the best learned pairing with CDF separately for each
endpoint. `Delta efficiency` is in percentage points; `delta AUPRC` is absolute.
The best pairing can differ between endpoints.

| Model/score | Best efficiency pairing | Delta efficiency vs CDF | Best AUPRC pairing | Delta AUPRC vs CDF |
|---|---|---:|---|---:|
| AE native | JetCLR | -0.51 pp | JetCLR | -0.005 |
| AE OAS | JetCLR | +0.45 pp | physics-summary | +0.000 |
| VAE native | flat-physical | +0.83 pp | flat-physical | -0.122 |
| VAE OAS | all tied | -0.13 pp | all tied | -0.036 |
| DSAE native | flat-physical | 0.00 pp | flat-physical | 0.000 |
| DSVAE native | typed-sliced | +0.56 pp | JetCLR | +0.022 |
| DeepSVDD native | flat-physical | +0.01 pp | flat-physical | +0.011 |
| RealNVP native | physics-summary | +2.06 pp | flat-physical | 0.000 |

Interpretation by pairing:

- **Flat-physical is the safest learned pairing.** It is neutral or positive on
  both endpoints for DSAE, DeepSVDD, and RealNVP. It improves VAE efficiency by
  0.83 pp but reduces VAE AUPRC by 0.122.
- **Typed-sliced is the best DSVAE efficiency pairing**, raising efficiency by
  0.56 pp and AUPRC by 0.006 relative to CDF.
- **JetCLR gives the best DSVAE AUPRC**, +0.022 over CDF, but slightly lowers its
  efficiency. It is also the least-bad learned pairing for native AE.
- **Physics-summary produces the largest efficiency gain**, +2.06 pp for RealNVP,
  but lowers its AUPRC by 0.044. Its DSVAE checkpoint is the sharpest endpoint
  conflict: 6.01% efficiency but only 0.039 AUPRC. The per-anomaly outputs and
  checkpoint hash were verified; this is not a missing-row or join artifact.

Across native models, applying any one fixed learned pairing globally is not a good
policy. Compared with CDF, the macro changes are:

| Fixed pairing | Delta efficiency | Delta AUPRC |
|---|---:|---:|
| flat-physical | +0.06 pp | -0.020 |
| physics-summary | -0.08 pp | -0.166 |
| typed-sliced | -0.49 pp | -0.030 |
| JetCLR | -0.64 pp | -0.073 |

Thus pairing should be treated as model-dependent, and claims of improvement must
name the endpoint. The present evidence does **not** support “pairing improves both
efficiency and AUPRC in general.”

## OAS versus native AE/VAE scores

Masked OAS is an additional anomaly score evaluated on the same models; it is not a
new training run. Its padding-aware implementation excludes nonexistent physics
objects.

- AE's best native efficiency/AUPRC across the table is **4.77% / 0.628**; its best
  OAS result is **3.09% efficiency** and **0.501 AUPRC** (from different selectors).
- VAE's best native result is **5.36% efficiency** and **0.634 AUPRC**; its best OAS
  result is **4.25% / 0.544** with W1.
- In all 14 like-for-like selector/pairing cells, OAS lowers efficiency. It lowers
  AUPRC in 12; the two exceptions are VAE physics-summary and JetCLR, where the
  native checkpoints themselves have unusually poor AUPRC.

OAS is therefore useful as a diagnostic but is not competitive with the native
scores in this campaign.

## Comparison with the paper

The paper comparison is possible only for efficiency. Using each native model's
best learned-pairing CAP result:

| Model | Best paired CAP efficiency | Paper CAP efficiency | Difference | Paired-checkpoint AUPRC |
|---|---:|---:|---:|---:|
| AE | 4.21% | 5.73% | -1.52 pp | 0.623 |
| VAE | 5.36% | 4.79% | +0.57 pp | 0.512 |
| DSAE | 6.10% | 8.55% | -2.45 pp | 0.601 |
| DSVAE | 7.40% | 5.02% | +2.38 pp | 0.633 |
| DeepSVDD | 8.28% | 4.03% | +4.25 pp | 0.315 |
| RealNVP | 7.57% | 3.26% | +4.31 pp | 0.387 |

The paired setup is numerically above the paper CAP efficiency for four of six
models, especially DSVAE, DeepSVDD, and RealNVP. This is promising, but it is not a
formal paper-level superiority claim: these are 50-epoch search checkpoints rather
than independently retrained 200-epoch endpoints; the selection datasets changed
from zero-bias/background to background0/background1; the campaign used shared
multi-objective NSGA-II; and the reported checkpoint within each front was selected
using downstream mean efficiency.

The AUPRC column above characterizes those exact checkpoints. It is not a comparison
with the paper because the paper provides no matching physics AUPRC reference.

## Selection rule and evidence integrity

The primary candidate was retained exactly as in the efficiency analysis: one
checkpoint was selected from each strategy-specific Pareto front using mean
downstream 250 Hz efficiency, and AUPRC was then attached to that **same checkpoint**.
No second AUPRC oracle was used. This makes the endpoint comparison honest: a model
that looks good at 250 Hz is allowed to look poor under AUPRC.

The frozen evaluation contains all 20 anomalies and passed exact joins on model,
trial, checkpoint path, checkpoint SHA-256, anomaly, score, selector, and pairing.
There are 11,820 rows per endpoint and 23,640 joined rows. Machine-readable results
are in:

`/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c/results/endpoint_analysis/`

Key files are `primary_front_summary.csv`, `controlled_selector_comparison.csv`,
`fixed_pairing_vs_cdf.csv`, `oas_panel.csv`, `combined_long.csv`, and
`endpoint_analysis.json`. The authenticated AUPRC aggregate is:

`/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c/auprc_evaluation/aggregate.json`

Its SHA-256 is
`ffaa4e66c67408cbed0975ec50363789db655010bb66251c87826ea69b9b0225`.
