# Background-pairing experiment: results

## Executive answer

At the evaluated 250 Hz operating point:

1. **CAP is the best model-selection metric for four of six algorithms using their
   paper-native anomaly scores.** W1 is narrowly better for AE and DSAE; drift is never
   best. With OAS, CAP wins for AE and W1 wins for VAE.
2. **Learned event pairing improves CAP for VAE, DSVAE, and especially RealNVP.** It
   hurts AE and is effectively neutral for DSAE and DeepSVDD. Pairing therefore works,
   but must be chosen per algorithm.
3. **The best paired-CAP result is numerically above the paper's CAP result for four
   of six algorithms:** VAE, DSVAE, DeepSVDD, and RealNVP. AE and DSAE remain below.
4. **OAS is not competitive with the native AE/VAE anomaly scores.** It lowers the
   best overall efficiency by 35% for AE and 21% for VAE.

The completed evaluation measured fixed-rate signal efficiency, which is the physics
endpoint reported below. It did not persist AUPRC, so no AUPRC claim is made.

## Native anomaly scores: complete per-algorithm comparison

All values are mean signal efficiency over the same 20 downstream anomaly datasets.
“Paired CAP” is the best of the four learned pairings; CDF is the unlearned control.

| Algorithm | Best paired CAP | CDF CAP | W1 | Drift | Overall winner | Paired CAP vs paper CAP |
|---|---:|---:|---:|---:|---|---:|
| AE | JetCLR: 4.21% | 4.71% | **4.77%** | 4.13% | **W1** | **-1.52 pp** |
| VAE | flat-physical: **5.36%** | 4.53% | 4.67% | 4.67% | **paired CAP** | **+0.57 pp** |
| DSAE | flat-physical: 6.10% | 6.10% | **6.15%** | 5.35% | **W1** | **-2.45 pp** |
| DSVAE | typed-sliced: **7.40%** | 6.84% | 0.84% | 6.39% | **paired CAP** | **+2.38 pp** |
| DeepSVDD | flat-physical: **8.28%** | 8.27% | 5.48% | 4.65% | **paired CAP** | **+4.25 pp** |
| RealNVP | physics-summary: **7.57%** | 5.50% | 2.78% | 2.78% | **paired CAP** | **+4.31 pp** |

### Does CAP beat the other metrics?

Yes for VAE, DSVAE, DeepSVDD, and RealNVP. No for AE and DSAE, where W1 wins by only
0.06 and 0.05 percentage points respectively. Drift never wins.

This is not merely an average effect. CAP is decisively stronger for:

- DSVAE: 7.40% versus 6.39% drift and 0.84% W1;
- DeepSVDD: 8.28% versus 5.48% W1 and 4.65% drift;
- RealNVP: 7.57% versus 2.78% for both alternatives.

For VAE the advantage is smaller but still clear: 5.36% versus approximately 4.67%.

### Does learned pairing improve CAP?

Compared with CDF CAP:

- RealNVP improves by **+2.06 pp** with physics-summary pairing.
- VAE improves by **+0.83 pp** with flat-physical pairing.
- DSVAE improves by **+0.56 pp** with typed-sliced pairing.
- DeepSVDD improves by only **+0.01 pp**, effectively a tie.
- DSAE is exactly tied.
- AE becomes **0.51 pp worse**; CDF remains preferable for AE's native score.

The correct scientific conclusion is therefore not “pairing always improves CAP.” It
is: **pairing creates substantial gains for particular model geometries, with the
optimal pairing depending on the algorithm.**

Pairing does not apply to W1 or drift. W1 uses only the two score marginals, so a
one-to-one reordering of events cannot change it. Drift compares operating thresholds
between background0 and background1 and likewise has no event-correspondence input.
They are the unpaired metric baselines against which paired CAP is compared.

## AE and VAE with OAS: separate results

OAS is a second anomaly-score view, not a second training run or pairing definition.
It must be compared separately from each model's native score.

| Algorithm and score | Best paired CAP | CDF CAP | W1 | Drift | Winner |
|---|---:|---:|---:|---:|---|
| AE native MSE | 4.21% | 4.71% | **4.77%** | 4.13% | W1 |
| AE masked OAS | **3.09%** | 2.64% | 2.37% | 2.37% | paired CAP |
| VAE native latent KL | **5.36%** | 4.53% | 4.67% | 4.67% | paired CAP |
| VAE masked OAS | 2.78% | 2.91% | **4.25%** | 2.82% | W1 |

The metric ranking changes under OAS, but the absolute result becomes worse:

- AE: the best OAS result is 3.09%, versus 4.77% native (**-1.69 pp; -35%**).
- VAE: the best OAS result is 4.25%, versus 5.36% native (**-1.11 pp; -21%**).
- Across the seven like-for-like CAP/W1/drift comparisons for each model, OAS loses
  every one: **14 losses out of 14 comparisons**.

Pairing itself remains score-dependent. JetCLR pairing improves AE OAS over its CDF
control by +0.45 pp, whereas the best learned VAE OAS pairing is 0.13 pp below CDF.
There is no paper OAS baseline, so OAS cannot be used to claim an improvement over the
paper.

## Does pairing improve the paper result?

Using each algorithm's native score and comparing the best learned-pairing CAP result
with the paper's CAP efficiency:

| Algorithm | Best paired CAP | Paper CAP | Difference |
|---|---:|---:|---:|
| AE | 4.21% | 5.73% | **-1.52 pp** |
| VAE | 5.36% | 4.79% | **+0.57 pp** |
| DSAE | 6.10% | 8.55% | **-2.45 pp** |
| DSVAE | 7.40% | 5.02% | **+2.38 pp** |
| DeepSVDD | 8.28% | 4.03% | **+4.25 pp** |
| RealNVP | 7.57% | 3.26% | **+4.31 pp** |

The result is promising: paired CAP is numerically higher for four algorithms and
substantially higher for DSVAE, DeepSVDD, and RealNVP. It is not evidence that pairing
universally improves the paper: AE and DSAE are below their paper CAP references.

The other closest-form paper comparisons reinforce, but do not prove, the result:

- campaign W1 is numerically above paper W1 for all six algorithms;
- campaign drift is above the paper's closest Stable reference for four of six;
- campaign CDF-CAP is above paper CAP for three of six;
- overall, 13 of 18 closest-form model/metric cells are numerically positive.

## What can be claimed

The evidence supports these statements:

- CAP is usually the strongest selection metric, winning four of six native-score
  algorithm comparisons and one of two OAS comparisons.
- Learned pairing produces real, model-dependent efficiency gains, strongest for
  RealNVP and meaningful for VAE and DSVAE.
- The new paired-CAP setup is numerically promising relative to the paper for four
  algorithms, but not for AE or DSAE.
- Native MSE/KL should remain the primary AE/VAE score; masked OAS is useful only as a
  diagnostic in this experiment.

It does **not** support a formal “outperforms the paper” claim. These are 50-epoch
search checkpoints rather than the paper's 200-epoch retrained endpoints; the
model-selection domains differ; the campaign used shared multi-objective NSGA-II
rather than independent two-objective M-TPE; and each reported checkpoint is an
optimistic oracle chosen on the same 20 downstream anomaly datasets.

## Evidence

The evaluation authenticated 421 physical checkpoints, 601 checkpoint/score passes,
56 logical fronts, 591 front memberships, and complete results for all 20 anomaly
datasets. The machine-readable evidence is under:

`/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c/results/provisional_search50/`

The main files are `front_winners.csv`, `candidate_efficiencies.csv`,
`cross_combination_choices.csv`, `paper_comparison.csv`, and
`artifact_manifest.json`.
