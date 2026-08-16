# Background0-background1 model-selection results

## The two research questions

### 1. Does CAP beat W1 and drift in AUPRC and efficiency?

**Efficiency: usually, but not universally.** The best CAP variant wins for four of
the six native-score models. W1 wins for AE and DSAE. Drift wins for none.

| Model | Best CAP efficiency | W1 efficiency | Drift efficiency | Winner |
|---|---:|---:|---:|---|
| AE | 4.71% | **4.77%** | 4.13% | W1 |
| VAE | **5.36%** | 4.67% | 4.67% | CAP |
| DSAE | 6.10% | **6.15%** | 5.35% | W1 |
| DSVAE | **7.40%** | 0.84% | 6.39% | CAP |
| DeepSVDD | **8.28%** | 5.48% | 4.65% | CAP |
| RealNVP | **7.57%** | 2.78% | 2.78% | CAP |

Across models, the mean of the model-specific best CAP results is 6.57%, compared
with 4.12% for W1 and 4.66% for drift. This aggregate is descriptive: each entry is a
downstream-oracle winner selected using the same 20 anomaly datasets.

**AUPRC: the completed evaluation does not contain it.** It evaluated and persisted
only efficiency at the fixed 250 Hz operating point. Therefore this campaign currently
provides no evidence for whether CAP beats W1 or drift in AUPRC. Any report claiming
otherwise would be inventing a result.

### 2. Does learned pairing improve CAP, and does it affect AUPRC differently from efficiency?

**For efficiency, learned pairing helps some models strongly, but is not a universal
improvement over CDF pairing.**

| Model | Best learned CAP pairing | Learned | CDF control | Learned - CDF |
|---|---|---:|---:|---:|
| AE | JetCLR | 4.21% | **4.71%** | **-0.51 pp** |
| VAE | flat-physical | **5.36%** | 4.53% | **+0.83 pp** |
| DSAE | flat-physical | 6.10% | 6.10% | 0.00 pp |
| DSVAE | typed-sliced | **7.40%** | 6.84% | **+0.56 pp** |
| DeepSVDD | flat-physical | **8.28%** | 8.27% | +0.01 pp |
| RealNVP | physics-summary | **7.57%** | 5.50% | **+2.06 pp** |

The substantive pairing gains are RealNVP (+2.06 percentage points), VAE (+0.83),
and DSVAE (+0.56). AE becomes worse, while DSAE and DeepSVDD are effectively tied.
Across all six models, flat-physical CAP averages only 0.06 percentage points above
CDF, confirming that the useful choice is model-specific rather than universal.

**For AUPRC, this comparison has not yet been measured.** Because AUPRC is absent,
we cannot say whether pairing helps ranking quality more or less than fixed-rate
efficiency.

Pairing itself applies only to CAP. W1 compares the two anomaly-score marginals and is
unchanged by a one-to-one event permutation. Drift compares operating thresholds
between background0 and background1 and likewise has no event-pairing input. Thus the
meaningful second question is:

> For CAP, does each learned event pairing beat the CDF control on downstream AUPRC
> and fixed-rate efficiency? For W1 and drift, which selected checkpoints provide the
> corresponding unpaired baselines?

The efficiency half is answered above; the AUPRC half requires evaluating the same
frozen checkpoints with the AUPRC callback.

## OAS finding

Masked residual OAS should not replace the native AE/VAE scores based on efficiency:

- AE: 3.09% best OAS versus 4.77% native, a 35% relative loss.
- VAE: 4.25% best OAS versus 5.36% native, a 21% relative loss.
- OAS is lower in all 14 like-for-like AE/VAE selector comparisons.

Its AUPRC behavior remains unmeasured for the same reason described above.

## Comparison with the paper

For the endpoint that is available—250 Hz efficiency—the campaign is numerically
above the paper in 13 of 18 closest-form model/metric comparisons. The strongest
positive differences are DeepSVDD CDF-CAP (+4.24 pp), DSAE W1 (+2.40 pp), RealNVP
CDF-CAP (+2.24 pp), and DSVAE CDF-CAP (+1.82 pp). The strongest regressions are DSAE
CDF-CAP (-2.45 pp) and AE CDF-CAP (-1.02 pp).

These are directional comparisons, not proof of improvement: this evaluation uses
50-epoch search checkpoints, background0-background1 rather than the paper's original
model-selection domains, shared multi-objective NSGA-II rather than independent
two-objective M-TPE, and downstream-oracle checkpoint selection. The available paper
reference also does not provide an AUPRC table for a like-for-like comparison.

## What is supported now

1. CAP is the strongest efficiency selector for VAE, DSVAE, DeepSVDD, and RealNVP.
2. W1 is narrowly best for AE and DSAE; drift is never best.
3. Pairing is valuable for RealNVP, VAE, and DSVAE, but pairing choice must be
   model-specific.
4. Native scores beat masked OAS for both AE and VAE on efficiency.
5. No AUPRC conclusion is currently supported; that endpoint must be evaluated before
   answering either research question completely.

## Evidence

The authenticated evaluation covers 421 physical checkpoints, 601 checkpoint/score
passes, 56 logical fronts, and all 20 downstream anomaly datasets. Detailed evidence
is stored under:

`/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c/results/provisional_search50/`

The key files are `front_winners.csv`, `candidate_efficiencies.csv`,
`cross_combination_choices.csv`, `paper_comparison.csv`, and
`artifact_manifest.json`.
