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
5. AE training still minimizes Huber reconstruction loss. AE searches are repeated
   with the paper-native reconstruction MSE and with residual Mahalanobis distance;
   VAE searches are repeated with the paper-native latent KL score and residual
   Mahalanobis distance. OAS covariance is estimated only from clean training-normal
   residuals using a deterministic bounded sample. Padding is treated as missing:
   feature means use observed entries only, missing entries are mean-imputed for the
   covariance fit, and the event score includes and normalizes over observed features
   only.
6. Pair-table metadata, source names, split, sample count, index bijection, and ordered
   input hashes must pass authentication before CAP is evaluated.
7. Each search trial retains and compares the paper's three checkpoint candidates:
   last epoch, best EMA primary-selection metric, and stable operational score.

## Selection metrics

- CAP: five logical selection views: one for each stored one-to-one pairing
  (`flat_physical`, `physics_summary`, `typed_sliced_wasserstein`, and `jetclr`) plus
  empirical-CDF score-rank pairing as the paper-aligned control.
- Wasserstein-1: one search on the background0/background1 score marginals. Pair-table
  permutations cannot affect a one-dimensional marginal Wasserstein distance.
- Threshold drift: calibrate the operational threshold on background0 and evaluate it
  on background1. Run background1-to-background0 as a diagnostic after the main study,
  not as an additional primary optimization target.

## Primary experiment matrix

Models: AE, VAE, DSAE, DSVAE, SVDD, and RealNVP. DTE is excluded because it was not
one of the six physics models in the paper.

For every trained trial, evaluate seven logical two-objective selection views:

1. CAP / `flat_physical`
2. CAP / `physics_summary`
3. CAP / `typed_sliced_wasserstein`
4. CAP / `jetclr`
5. CAP / empirical-CDF control
6. Wasserstein-1 / E versus G
7. Threshold drift / E calibration to G evaluation

AE and VAE each expose native and OAS views; the other four models retain their native
score. One shared 600-trial pool is trained per model, so the campaign consists of six
Optuna studies and 3,600 trained trials. Each AE/VAE trial returns 28 scalar objectives
(14 ordered pairs), and each other model returns 14 (7 ordered pairs). The 56
strategy-specific two-dimensional Pareto fronts are reconstructed from those shared
pools; the high-dimensional global Optuna front is not used. This deliberately gives
every selection strategy exactly the same sampled configurations, at the cost of using
one joint M-TPE sampling history per model instead of independent paper-style 2D M-TPE
histories.

After all fronts are frozen, every unique front configuration is retrained once for 200
epochs with the paper's fixed reporting seed. Native and OAS downstream scores use
separate validation-derived operating thresholds. As in the paper, the reported trial
for each logical front is the retrained candidate with the best mean downstream anomaly
efficiency; this oracle step is explicitly marked as downstream-label-using.

## Execution gates

1. Implement and unit-test cross-dataset drift and AE canonical Mahalanobis scoring.
2. Validate all four pair tables against the exact source-specific loader prefixes.
3. Compose all six shared-study Hydra configurations and verify their ordered objective
   vectors against the campaign manifest.
4. Run unit/configuration tests and one short unified GPU pilot before any full study.
5. Submit six persistent Optuna studies with unique storage and study names.
   Execute every Slurm phase from a dedicated commit-pinned worktree rather than the
   mutable development checkout.
6. Monitor jobs, retry infrastructure failures, and record job/study state in
   `RUN_LEDGER.md` and machine-readable manifests.
7. Freeze Pareto selections before 200-epoch retraining.
8. Aggregate efficiencies, CAP, rank correlation, Wasserstein, threshold drift,
   provenance, and directional drift diagnostics.

## Completion evidence

Completion requires: green targeted tests; successful composition of all six shared
studies; authenticated pair-table preflight; completed Optuna manifests showing 600
finished trials per model; 56 frozen logical Pareto fronts; completed unique
retraining/evaluation manifests; and a final result table/report covering every model
and selection strategy.
