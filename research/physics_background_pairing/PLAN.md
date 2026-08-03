# Physics background-pairing replication

## Scientific question

Repeat the signal-agnostic physics model-selection study from
`28031_Informative_Model_Valida.pdf`, while replacing the model-selection comparison
between recorded ZeroBias and `SingleNeutrino_E-10-gun` simulation with a comparison
between the two held-out recorded ZeroBias sources:

- background0: `ZB_run396102` (2025E)
- background1: `ZB_run398183` (2025G)

This tests selection for robustness across data-taking eras. It is not a
data-versus-simulation agreement test.

## Invariants

1. Training data is unchanged: the ordinary combined ZeroBias training split.
2. The standard combined `normal` validation and test splits are unchanged.
3. Only model-selection metrics receive source-specific background0/background1
   streams. The streams use held-out validation or test events only.
4. Signal datasets, target rate (0.25 kHz), base rate (28608.8064 kHz), preprocessing,
   search spaces, epoch budgets, and downstream evaluation remain those of the paper.
5. AE training still minimizes Huber reconstruction loss, but its anomaly score is
   residual Mahalanobis distance with OAS covariance estimated only from clean
   training-normal residuals. MSE is retained only as a named diagnostic.
6. Pair-table metadata, source names, split, sample count, index bijection, and ordered
   input hashes must pass authentication before CAP is evaluated.
7. Each search trial retains and compares the paper's three checkpoint candidates:
   last epoch, best EMA primary-selection metric, and stable operational score.

## Selection metrics

- CAP: four independent searches, one for each stored one-to-one pairing:
  `flat_physical`, `physics_summary`, `typed_sliced_wasserstein`, and `jetclr`.
- Wasserstein-1: one search on the background0/background1 score marginals. Pair-table
  permutations cannot affect a one-dimensional marginal Wasserstein distance.
- Threshold drift: calibrate the operational threshold on background0 and evaluate it
  on background1. Run background1-to-background0 as a diagnostic after the main study,
  not as an additional primary optimization target.

## Primary experiment matrix

Models: AE, VAE, DSAE, DSVAE, SVDD, and RealNVP. DTE is excluded because it was not
one of the six physics models in the paper.

For each model, run six 600-trial, 50-epoch Optuna searches:

1. CAP / `flat_physical`
2. CAP / `physics_summary`
3. CAP / `typed_sliced_wasserstein`
4. CAP / `jetclr`
5. Wasserstein-1 / E versus G
6. Threshold drift / E calibration to G evaluation

This is 36 studies and at most 21,600 trials. After each study, retain every Pareto
candidate, retrain it for 200 epochs with the paper's fixed reporting seed, and evaluate
all configured physics signals at the unchanged operational rate. As in the paper, the
final reported trial is the retrained Pareto candidate with the best mean downstream
anomaly efficiency; this oracle step is explicitly marked as downstream-label-using.
Search submission may be staged by model and metric based on pilot runtime and cluster
limits, but the matrix does not shrink.

## Execution gates

1. Implement and unit-test cross-dataset drift and AE canonical Mahalanobis scoring.
2. Validate all four pair tables against the exact source-specific loader prefixes.
3. Compose every model/metric/strategy Hydra configuration.
4. Run one-batch CPU configuration tests and one short GPU pilot per metric family.
5. Submit persistent Optuna studies with unique storage and study names.
6. Monitor jobs, retry infrastructure failures, and record job/study state in
   `RUN_LEDGER.md` and machine-readable manifests.
7. Freeze Pareto selections before 200-epoch retraining.
8. Aggregate efficiencies, CAP, rank correlation, Wasserstein, threshold drift,
   provenance, and directional drift diagnostics.

## Completion evidence

Completion requires: green targeted tests; successful config composition for all 36
cells; authenticated pair-table preflight; completed Optuna study manifests showing
600 finished trials per cell (or explicit accounted failed trials followed by enough
replacement trials); completed retraining/evaluation manifests; and a final result
table/report covering every model and selection strategy.
