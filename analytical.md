# Analytical Synthetic Benchmark for CAP

This note defines a controlled synthetic anomaly-detection setting for the CAP
paper. The goal is not to make a realistic event generator. The goal is to
construct a statistically transparent large-N, low-dimensional benchmark that
matches the relevant validation regime of AD@L1:

- features are identifiable;
- normal data are abundant;
- labelled anomalies are unavailable during validation;
- downstream performance is evaluated at a small false-positive rate;
- the model produces a scalar anomaly score.

The benchmark gives closed-form anomaly-score laws and therefore separates three
questions that are conflated in less controlled experiments:

1. Does a score contain any reproducible assignment information on typical data?
2. Does a score transfer stably between two typical domains?
3. Does a score have downstream power against a held-out anomaly family?

CAP addresses the first question, Wasserstein and threshold drift address the
second, and supervised anomaly metrics address the third.

## 1. Gaussian Feature-Identifiable Model

Let the typical event be a low-dimensional feature vector

```math
X \sim P_0 = \mathcal{N}(0, I_d).
```

The second typical domain is

```math
X' \sim P_0^\eta = \mathcal{N}(\eta e_j, I_d),
```

where `eta = 0` gives two identical typical domains and `eta != 0` gives a
controlled nuisance shift in feature `j`. This mimics the role of a real typical
domain and a simulated/reference typical domain.

A held-out anomaly family is

```math
X_\delta \sim P_\delta = \mathcal{N}(\delta e_a, I_d),
```

where feature `a` is the identifiable anomaly-carrying coordinate. Labels from
`P_delta` are never used by CAP, W1, or threshold drift; they are used only to
measure downstream detection power.

## 2. Closed-Form Score Laws

### 2.1 Optimal Linear Score

For testing `P_0` against `P_delta`, the log likelihood ratio is

```math
\log \frac{dP_\delta}{dP_0}(x)
  = \delta x_a - \frac{1}{2}\delta^2.
```

Therefore the Neyman-Pearson anomaly score is any monotone transform of

```math
S_{\mathrm{lin}}(x) = x_a.
```

Under normal and anomalous data,

```math
S_{\mathrm{lin}}(X) \sim \mathcal{N}(0,1),
\qquad
S_{\mathrm{lin}}(X_\delta) \sim \mathcal{N}(\delta,1).
```

At false-positive rate `alpha`, the normal threshold is

```math
\tau_\alpha = \Phi^{-1}(1-\alpha),
```

and the downstream power is

```math
\mathrm{TPR}_{\mathrm{lin}}(\alpha,\delta)
  = \mathbb{P}[S_{\mathrm{lin}}(X_\delta) \ge \tau_\alpha]
  = 1 - \Phi(\Phi^{-1}(1-\alpha)-\delta).
```

The AUROC is also closed form:

```math
\mathrm{AUROC}_{\mathrm{lin}}(\delta)
  = \mathbb{P}[S_{\mathrm{lin}}(X_\delta) > S_{\mathrm{lin}}(X)]
  = \Phi\left(\frac{\delta}{\sqrt{2}}\right).
```

### 2.2 Projection / Autoencoder Residual Score

Let `P` be an orthogonal projection onto the reconstructed subspace and let
`R = I - P` be the residual projection. A linear autoencoder-style anomaly score
is

```math
S_R(x) = \|Rx\|_2^2.
```

Let

```math
r = \mathrm{rank}(R),
\qquad
\lambda = \|R(\delta e_a)\|_2^2.
```

Then

```math
S_R(X) \sim \chi^2_r,
\qquad
S_R(X_\delta) \sim \chi^2_r(\lambda),
```

where `chi^2_r(lambda)` is the noncentral chi-square distribution with `r`
degrees of freedom and noncentrality `lambda`.

The operational threshold and power are therefore

```math
\tau_{\alpha,r} = F^{-1}_{\chi^2_r}(1-\alpha),
```

```math
\mathrm{TPR}_R(\alpha,\delta)
  = 1 - F_{\chi^2_r(\lambda)}(\tau_{\alpha,r}).
```

Two special cases matter:

- If the anomaly coordinate is in the residual subspace, then
  `lambda = delta^2` and the score has power.
- If the anomaly coordinate is fully reconstructed, then `lambda = 0` and
  `TPR_R(alpha, delta) = alpha`; the detector is no better than random
  thresholding at the operating point.

The AUROC has the exact integral form

```math
\mathrm{AUROC}_R
  = \mathbb{P}[\chi^2_r(\lambda) > \chi^2_r]
  = \int_0^\infty F_{\chi^2_r}(s) f_{\chi^2_r(\lambda)}(s)\,ds.
```

This integral is one-dimensional and numerically stable.

## 3. Standard Supervised Anomaly Measures

These quantities use anomaly labels and are therefore evaluation-only in the
paper.

### Fixed-Rate Efficiency / Power

The relevant AD@L1 metric is the true-positive rate at a target false-positive
rate `alpha`:

```math
\mathrm{TPR}(\alpha)
  = \mathbb{P}_{X \sim P_\delta}[S(X) \ge Q_{1-\alpha}(S(P_0))].
```

This is the metric most aligned with the trigger setting because the allowed
background rate fixes the operating quantile.

### ROC and AUROC

For score CDFs `F_0` and `F_1`,

```math
\mathrm{AUROC}
  = \mathbb{P}[S_1 > S_0]
  = \int F_0(s)\,dF_1(s).
```

AUROC is useful as a threshold-integrated ranking measure, but it can hide
differences that only matter at extreme quantiles.

### AUPRC

For anomaly prevalence `pi`, precision at threshold `t` is

```math
\mathrm{Prec}(t)
  = \frac{\pi \, \overline{F}_1(t)}
         {\pi \, \overline{F}_1(t) + (1-\pi)\,\overline{F}_0(t)},
```

where `overline{F}(t)=1-F(t)`. AUPRC integrates precision over recall. It is
prevalence-dependent and should not replace fixed-rate efficiency in the trigger
regime.

### Two-Sample Separability Tests

KS distance, MMD, and normal-vs-anomaly Wasserstein distance can be computed
between labelled normal and anomaly score distributions. These are useful
diagnostics, but they are not label-free validation criteria because they require
the anomaly distribution.

## 4. Signal-Agnostic Metrics on Two Typical Domains

Let `S_theta(x)` be the anomaly score produced by a candidate model `theta`. CAP,
W1, and threshold drift are computed without samples from `P_delta`.

### 4.1 Wasserstein Between Typical Score Distributions

The population W1 score-distance used as a distribution-similarity baseline is

```math
W_1^\theta
  = \int_0^1
      \left|
        Q_{S_\theta(P_0)}(u)
        -
        Q_{S_\theta(P_0^\eta)}(u)
      \right|\,du.
```

If `eta = 0`, then

```math
W_1^\theta = 0
```

for every score function `S_theta`, including collapsed scores and useful
scores. Therefore W1 cannot rank models in the ideal identical-domain limit.
With `eta != 0`, W1 measures sensitivity to the nuisance shift, but it still
does not measure whether the score has downstream anomaly power.

### 4.2 Threshold Drift

Let a normal score sample be split into calibration and evaluation subsets. The
population threshold drift is

```math
L_{\mathrm{thr}}^\theta
  =
  \left|
    \log
    \frac{
      \mathbb{P}[S_\theta(X_E) \ge Q_{1-\alpha}(S_\theta(X_A))]
    }{\alpha}
  \right|.
```

If both subsets are drawn from the same score law and the score distribution is
continuous, then

```math
L_{\mathrm{thr}}^\theta = 0
```

for every score function. Finite-sample drift is dominated by quantile and
binomial noise; for `N_E alpha` expected exceedances it scales roughly as

```math
O_p((N_E \alpha)^{-1/2})
```

after the threshold has been estimated. Like W1, threshold drift measures
stability, not informativeness.

## 5. CAP in the Gaussian Benchmark

For the clean theorem, use the baseline CAP energy

```math
E(p,0)=p,
\qquad
E(p,1)=1-p,
```

where `p = n(S_theta(x))` is a normalized anomaly score. For fixed inverse
temperature `beta`, the induced binary Gibbs posterior is

```math
q_\beta(y=1\mid p)
  =
  \frac{\exp[-\beta(1-p)]}
       {\exp[-\beta p] + \exp[-\beta(1-p)]}
  =
  \sigma(\beta(2p-1)).
```

For a paired typical-domain score pair `(p_1,p_2)`, the CAP kernel is

```math
k_\beta(p_1,p_2)
  =
  \log\left[
    q_\beta(p_1)q_\beta(p_2)
    +
    (1-q_\beta(p_1))(1-q_\beta(p_2))
  \right].
```

The implemented CAP is an empirical average of this kernel, with beta optimized
over a finite procedure. The theorem below uses fixed or bounded beta because an
unconstrained population `sup_beta` can obscure the finite-temperature behavior
that the metric actually exploits.

### Theorem: CAP Separates Collapsed From Informative Typical Scores

Assume:

1. `X` and `X'` are drawn from the same typical distribution `P_0`;
2. the two score samples are paired by population CDF rank;
3. the normalized paired scores satisfy `p_1(U)=p_2(U)=p(U)` for
   `U ~ Uniform(0,1)`;
4. beta is fixed and finite.

Then the population CAP kernel is

```math
\mathrm{CAP}_\beta(S_\theta)
  =
  \mathbb{E}_{U}
  \log\left[
    q_\beta(p(U))^2
    +
    (1-q_\beta(p(U)))^2
  \right].
```

If the score is collapsed, then `p(U)=1/2` almost surely and

```math
\mathrm{CAP}_\beta(S_\theta) = -\log 2.
```

If `p(U)` is nonconstant on a set of nonzero measure, then

```math
\mathrm{CAP}_\beta(S_\theta) > -\log 2.
```

For small beta,

```math
\mathrm{CAP}_\beta(S_\theta)
  =
  -\log 2
  +
  \frac{\beta^2}{4}
  \mathbb{E}\left[(2p(U)-1)^2\right]
  +
  O(\beta^4).
```

Thus CAP measures reproducible assignment capacity: it is low for a stable but
uninformative score and increases when the score supports confident, repeatable
binary assignments across typical domains.

### Proof Sketch

For the baseline energy, direct substitution into the Gibbs posterior gives
`q_beta(y=1|p)=sigma(beta(2p-1))`. Under CDF pairing and identical domains,
paired observations have the same population score quantile, so the agreement
probability is

```math
A(p) = q_\beta(p)^2 + (1-q_\beta(p))^2.
```

The function `A(p)` is minimized at `q_beta(p)=1/2`, equivalently `p=1/2`, and
its minimum is `1/2`. Therefore the log-agreement is at least `-log 2`, with
equality only when `p=1/2` almost surely. Expanding
`sigma(beta z)=1/2 + beta z/4 + O(beta^3)` with `z=2p-1` gives

```math
A(p)=1/2 + beta^2 z^2/8 + O(beta^4),
```

and the stated expansion follows from expanding the logarithm around `1/2`.

## 6. What This Shows and What It Does Not Show

The theorem gives a precise role for CAP:

- W1 and threshold drift are zero in population for any stable score when the
  two typical domains are identical.
- CAP is also signal-agnostic, but it is not blind to collapse: it distinguishes
  stable uninformative scores from stable scores with assignment capacity.
- Supervised metrics such as TPR, AUROC, and AUPRC are still needed for final
  evaluation because they use held-out anomalies.

There is also an important impossibility result. If two candidate scores induce
the same distribution on all available typical domains but differ only in how
they rank an unseen anomaly family, no label-free validation metric can
universally choose between them. In the Gaussian model, a score on the wrong
feature can be as stable and as high-capacity as a score on the anomaly feature.
This is not a failure of CAP; it is the fundamental price of signal-agnostic
validation. CAP should therefore be framed as an informative agnostic selector,
not an oracle for arbitrary unseen anomalies.

The empirical claim to test is narrower and defensible:

```text
In large-N, low-dimensional settings, CAP detects nontrivial reproducible
assignment structure that stability-only and marginal-distribution criteria
cannot detect, while downstream labels confirm when that structure aligns with
the held-out anomaly direction.
```

## 7. Empirical Study Implemented in This Repository

The new Gaussian-subspace datamodule mode is implemented in
`src/data/synthetic.py` with `generator: gaussian_subspace`. A reusable config is
provided at `configs/data/synthetic.yaml`.

The script

```bash
uv run python src/synthetic.py --output-dir results/synthetic_gaussian
```

generates:

- `summary.csv`: CAP, CAP lift over the collapsed baseline, W1, threshold drift,
  TPR at FPR, analytic TPR, AUROC, and AUPRC for each score algorithm;
- `power_grid.csv`: empirical and analytic TPR over anomaly shifts;
- `score_distributions.png`: empirical chi-square and noncentral chi-square
  score laws against their analytic PDFs;
- `agnostic_metrics.png`: CAP/W1/drift comparison on normal-vs-reference data;
- `power_vs_shift.png`: empirical power curves against analytic predictions;
- `cap_vs_power.png`: relationship between CAP and downstream power.

The score algorithms are deliberately simple:

- `constant`: collapsed score;
- `linear_oracle`: optimal Gaussian likelihood-ratio direction;
- `linear_wrong`: a score on an unrelated coordinate;
- `negative_oracle`: correct coordinate with wrong score orientation;
- `residual_oracle_r1`: one-dimensional AE residual containing the anomaly;
- `residual_wrong_r1`: one-dimensional AE residual on the wrong feature;
- `radial_all`: full radial energy over all features.

The expected qualitative outcome is:

- `constant` has CAP close to `-log 2`, zero CAP lift, and no calibrated
  fixed-rate threshold because all events tie; AUROC/AUPRC remain uninformative.
- W1 and threshold drift are near zero for all score algorithms when
  `reference_shift=0`, because both typical domains are identically distributed.
- `linear_oracle` and `residual_oracle_r1` follow the analytic power curves above.
- `linear_wrong` and `residual_wrong_r1` remain at the target FPR.
- `negative_oracle` demonstrates the score-orientation assumption: CAP can detect
  capacity, but downstream power is poor because higher score means less
  anomalous for that artificial score.

This experiment is intentionally simple enough that every plotted trend has a
population-level explanation.
