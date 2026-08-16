# Physics background-pairing evaluation: key results

## Bottom line

The experiment gives a clear, useful answer: **pairing background0 and background1
can improve model selection, but the right pairing is model-dependent**. There is no
single learned pairing that dominates CDF for every algorithm.

The strongest evaluated configurations are:

| Model | Score | Best selector | Mean anomaly efficiency | Median | Trial |
|---|---|---|---:|---:|---:|
| DeepSVDD | native | flat-physical CAP | **8.28%** | 1.16% | 177 |
| RealNVP | native | physics-summary CAP | **7.57%** | 0.93% | 377 |
| DSVAE | native | typed-sliced CAP | **7.40%** | 1.40% | 580 |
| DSAE | native | W1 | **6.15%** | 0.83% | 478 |
| VAE | native | flat-physical CAP | **5.36%** | 0.98% | 590 |
| AE | native | W1 | **4.77%** | 0.88% | 418 |
| VAE | masked OAS | W1 | **4.25%** | 1.83% | 149 |
| AE | masked OAS | JetCLR CAP | **3.09%** | 1.36% | 251 |

These means are over the same 20 downstream anomaly datasets at the paper's 250 Hz
operating point. DeepSVDD is the strongest overall candidate, followed by RealNVP and
DSVAE. The much lower medians show that performance varies substantially by anomaly;
none of the methods is uniformly strong across every signal.

## What the pairing experiment says

Among native-score models, the best learned CAP pairing compared with CDF is:

| Model | Best learned pairing | Learned CAP | CDF CAP | Difference |
|---|---|---:|---:|---:|
| AE | JetCLR | 4.21% | **4.71%** | **-0.51 pp** |
| VAE | flat-physical | **5.36%** | 4.53% | **+0.83 pp** |
| DSAE | flat-physical | 6.10% | 6.10% | 0.00 pp |
| DSVAE | typed-sliced | **7.40%** | 6.84% | **+0.56 pp** |
| DeepSVDD | flat-physical | **8.28%** | 8.27% | +0.01 pp |
| RealNVP | physics-summary | **7.57%** | 5.50% | **+2.06 pp** |

The largest and most convincing pairing effect is for RealNVP. VAE and DSVAE also
benefit materially. AE still prefers CDF, while DSAE and DeepSVDD are effectively
tied. Averaged across the six native models, flat-physical CAP is only **0.06
percentage points** above CDF CAP (6.05% versus 5.99%). Therefore:

- use **model-specific pairings** when possible;
- retain **CDF as the control and AE default**;
- do not claim that one learned pairing universally replaces CDF.

Across model-selection metric families, the best CAP variant wins four of six native
models. W1 wins for AE and DSAE. Drift wins for none. W1 remains a valuable cheap
control, but the results support CAP with model-specific pairing for the stronger
DSVAE, DeepSVDD, RealNVP, and VAE candidates.

## OAS result: do not replace the native AE/VAE scores

Masked residual OAS does **not** improve mean anomaly performance:

| Model | Best native result | Best OAS result | OAS change |
|---|---:|---:|---:|
| AE | **4.77%** | 3.09% | **-1.69 pp (-35%)** |
| VAE | **5.36%** | 4.25% | **-1.11 pp (-21%)** |

More importantly, OAS is lower in **all 14 like-for-like comparisons** across AE/VAE
and the seven CAP/W1/drift selectors. Averaged over those matched selectors, the loss
is 1.67 pp for AE and 1.99 pp for VAE. VAE OAS has somewhat better lower-tail behavior
for its selected W1 checkpoint, but its mean remains clearly worse and its worst-case
efficiency is still approximately zero.

The practical conclusion is to keep masked OAS as a diagnostic score view, not as the
default anomaly score for either AE or VAE.

## Comparison with the paper

The closest available comparison uses native CDF-CAP, forward drift as the nearest
counterpart to the paper's Stable selector, and W1. Each cell is
`campaign / paper (campaign - paper)` in percentage points:

| Model | CDF-CAP | Drift / paper Stable | W1 |
|---|---:|---:|---:|
| AE | 4.71 / 5.73 (**-1.02**) | 4.13 / 4.29 (**-0.16**) | 4.77 / 3.22 (**+1.55**) |
| VAE | 4.53 / 4.79 (**-0.26**) | 4.67 / 4.39 (**+0.28**) | 4.67 / 4.49 (**+0.18**) |
| DSAE | 6.10 / 8.55 (**-2.45**) | 5.35 / 3.57 (**+1.78**) | 6.15 / 3.75 (**+2.40**) |
| DSVAE | 6.84 / 5.02 (**+1.82**) | 6.39 / 4.85 (**+1.54**) | 0.84 / 0.80 (**+0.04**) |
| DeepSVDD | 8.27 / 4.03 (**+4.24**) | 4.65 / 3.56 (**+1.09**) | 5.48 / 3.25 (**+2.23**) |
| RealNVP | 5.50 / 3.26 (**+2.24**) | 2.78 / 3.17 (**-0.39**) | 2.78 / 1.66 (**+1.12**) |

The campaign is numerically above the paper in **13 of 18** closest-form cells. W1 is
higher for all six models; CDF-CAP is higher for DSVAE, DeepSVDD, and RealNVP. The
largest positive differences are DeepSVDD CDF-CAP (+4.24 pp), DSAE W1 (+2.40 pp),
RealNVP CDF-CAP (+2.24 pp), and DSVAE CDF-CAP (+1.82 pp). The clearest regressions are
DSAE CDF-CAP (-2.45 pp) and AE CDF-CAP (-1.02 pp).

This is **promising directional evidence, not proof that the new system outperforms
the paper**. The comparison is not like-for-like:

- these are retained 50-epoch search checkpoints, whereas the paper reports
  200-epoch retrained models;
- model-selection data changed from ZeroBias plus SingleNeutrino to two recorded
  background eras;
- the campaign used one shared high-dimensional NSGA-II search per model rather than
  independent two-objective M-TPE studies;
- each reported front winner was chosen using mean performance on the same 20 anomaly
  datasets, so it is an optimistic downstream oracle;
- learned mappings and OAS have no direct paper baseline, and q99 was not exported.

Consequently, no confidence interval, p-value, or formal improvement claim is
supported by this evaluation.

## Recommended interpretation

1. **The background-pairing idea works for some important models.** RealNVP, VAE, and
   DSVAE show material gains over CDF when the pairing matches the model.
2. **DeepSVDD is the strongest candidate overall**, but its learned-pairing advantage
   over CDF is negligible; the gain is mainly in the selected model, not the pairing.
3. **CAP is the preferred selector for four of six models; W1 remains best for AE and
   DSAE; drift is not preferred.**
4. **OAS should not replace native AE/VAE scoring.** It loses every matched comparison.
5. **The paper comparison is encouraging but provisional.** The strongest next step,
   if additional compute is justified, is targeted confirmation of the few specific
   winners above rather than retraining hundreds of configurations indiscriminately.

## Evidence and reproducibility

The evaluation authenticated 346 tasks, 421 physical checkpoints, 601
checkpoint/score evaluations, 56 logical fronts, 591 front memberships, and 11,820
selected checkpoint/anomaly rows. Every front member has all 20 anomaly results, and
all published means reproduce from the raw rows.

Machine-readable evidence is in:

- `results/provisional_search50/front_winners.csv` — best checkpoint in every one of
  the 56 logical fronts;
- `results/provisional_search50/cross_combination_choices.csv` — best mean and tail
  choices per model/score family;
- `results/provisional_search50/candidate_efficiencies.csv` — per-anomaly results;
- `results/provisional_search50/paper_comparison.csv` — exact paper deltas;
- `results/provisional_search50/checkpoint_thresholds.csv` — validation-derived
  operating thresholds and checkpoint identities;
- `results/provisional_search50/artifact_manifest.json` — hashes and integrity record.
