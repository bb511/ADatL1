# Analytical Experiments for Section 3

This appendix note documents the synthetic experiments used to validate the
Section 3 claims in `theory.tex`. The experiments are deliberately analytical:
the relevant score laws, CAP objective, Wasserstein distance, threshold drift,
and fixed-FPR power all have population formulas.

Run:

```bash
uv run python src/analytical.py --output-dir figures/section3
```

The run writes CSV tables, figures, and
`figures/section3/metadata.json`. The directory was regenerated from scratch
after the update to the stable-nuisance theory.

## Formulation

### Score Channel

For each candidate detector, Section 3 models the standardized score on two
typical validation views as

```math
S_\phi^{(d)}
  =
  \frac{
    a_\phi Z + b_\phi U_\phi + \sigma_\phi \varepsilon_\phi^{(d)}
  }{
    \sqrt{a_\phi^2+b_\phi^2+\sigma_\phi^2}
  },
  \qquad d\in\{1,2\}.
```

`Z` is the anomaly-relevant latent coordinate. `U_phi` is stable nuisance:
it is reproducible across validation views, but future anomalies do not shift
it. The two key reliabilities are

```math
\rho_\phi^Z
  =
  \frac{a_\phi^2}{a_\phi^2+b_\phi^2+\sigma_\phi^2},
\qquad
\rho_\phi^R
  =
  \frac{a_\phi^2+b_\phi^2}{a_\phi^2+b_\phi^2+\sigma_\phi^2}.
```

`\rho_Z` is anomaly-aligned reliability. It controls power:

```math
\operatorname{TPR}_\phi(\alpha)
  =
  \Phi\!\left(
    \delta\sqrt{\rho_\phi^Z}
    -
    \Phi^{-1}(1-\alpha)
  \right).
```

`\rho_R` is total reproducible reliability. It is the paired score correlation
between validation views and controls CAP. The anomaly alignment of the
reproducible component is

```math
\pi_\phi
  =
  \frac{\rho_\phi^Z}{\rho_\phi^R}
  =
  \frac{a_\phi^2}{a_\phi^2+b_\phi^2},
```

when `rho_R > 0`. CAP is guaranteed to select a high-power detector only when
maximizing `rho_R` also maximizes `rho_Z`.

### Linear Gaussian Model

The feature-level experiment instantiates the theorem with

```math
X^{(d)}
  =
  \sqrt{\lambda_Z}\,Z e_0
  +
  \sqrt{\lambda_U}\,U e_1
  +
  \xi^{(d)},
```

where the typical marginal covariance of each view is the identity. Future
anomalies shift only `Z`, so the observed anomaly mean is

```math
\delta\sqrt{\lambda_Z}\,e_0.
```

For the linear sweep

```math
w(\theta)=\cos(\theta)e_0+\sin(\theta)e_1,
```

the population quantities are

```math
\rho_Z(w)=\lambda_Z\cos^2\theta,
\qquad
\rho_U(w)=\lambda_U\sin^2\theta,
\qquad
\rho_R(w)=\rho_Z(w)+\rho_U(w).
```

Thus

```math
\operatorname{TPR}_w(\alpha)
  =
  \Phi\!\left(
    \delta\sqrt{\rho_Z(w)}
    -
    \Phi^{-1}(1-\alpha)
  \right),
```

while CAP is monotone in `rho_R(w)`. The default run uses

```text
d = 6
target FPR alpha = 1e-3
delta = 3.5
lambda_Z = 0.90
lambda_U = 0.12
n_pairs = 60000
n_test = 150000
seed = 123
```

Because `lambda_Z > lambda_U`, the anomaly direction `w* = e0` maximizes both
`rho_Z` and `rho_R`.

![Feature-space geometry](figures/section3/feature_space_geometry.png)

### Empirical CAP

Empirical CAP is computed with the repository
`ApproximationCapacityKernel`, not with a hand-written replacement. To match
the theorem's binary Gibbs posterior, the script passes

```math
p = \frac{s+1}{2}
```

to the baseline-energy kernel. This gives

```math
q_\beta(y=1\mid p)
  =
  \sigma(\beta(2p-1))
  =
  \sigma(\beta s).
```

The raw kernel value is

```math
\widehat C_\beta
  =
  \frac{1}{n}
  \sum_i
  \log
  \left(
    \frac{1+m_\beta(s_i^{(1)})m_\beta(s_i^{(2)})}{2}
  \right).
```

The reported CAP is the lifted objective

```math
\widehat L^\star
  =
  \max_{\beta\in\mathcal B}
  \{\widehat C_\beta+\log 2\}.
```

The CSVs store both `cap_raw_empirical` and `cap_empirical`, so the identity
`cap_empirical = cap_raw_empirical + log(2)` is directly auditable.

CAP uses the row-wise pairing from the two-view validation generator. The row
order itself is exchangeable, but the pairing is not discarded: pair `i` in
view 1 is evaluated with pair `i` in view 2.

### Marginal Criteria

Wasserstein distance and threshold drift compare only the two score marginals.
In the unshifted linear model, every unit-norm linear score has standard normal
marginals in both views, so population W1 and threshold drift are zero for all
directions, even when `rho_R` differs.

For the benign marginal-shift experiment,

```math
X^{(1)}\sim \mathcal N(0,I),
\qquad
X^{(2)}\sim \mathcal N(\eta e_0,I),
```

the linear sweep has

```math
W_1(w)=\eta\cos\theta,
```

and

```math
M_{\mathrm{thr},\alpha}(w)
  =
  1-\Phi(\Phi^{-1}(1-\alpha)-\eta\cos\theta)-\alpha.
```

These marginal diagnostics correctly measure validation-domain score shift,
but smaller shift is not the same objective as higher anomaly power.

## Experiment 1: Reliability Channel

### Setup

This score-level experiment removes nuisance by setting
`rho = rho_Z = rho_R`. For each reliability value,

```math
S^{(d)}
  =
  \sqrt{\rho}\,Z
  +
  \sqrt{1-\rho}\,\varepsilon^{(d)}.
```

Anomaly scores are sampled from

```math
S\mid Y=1\sim \mathcal N(\delta\sqrt{\rho},1).
```

We measure empirical CAP with the approximation-capacity kernel, population CAP
by Gauss-Hermite quadrature, empirical TPR, and population TPR.

### Results

| rho | CAP empirical | CAP population | TPR empirical | TPR population |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.0000 | 0.0000 | 0.0009 | 0.0010 |
| 0.2587 | 0.0321 | 0.0319 | 0.0904 | 0.0951 |
| 0.4850 | 0.1077 | 0.1086 | 0.2602 | 0.2570 |
| 0.7437 | 0.2631 | 0.2603 | 0.4764 | 0.4713 |
| 0.9700 | 0.5264 | 0.5200 | 0.6417 | 0.6394 |

![CAP and TPR reliability experiment](figures/section3/theory_cap_tpr_vs_reliability.png)

### Conclusions

- CAP is monotone in paired reliability.
- TPR is monotone in the same reliability when there is no nuisance.
- Empirical kernel CAP tracks the population quadrature curve.

## Experiment 2: Linear Stable-Nuisance Sweep

### Setup

This is the direct linear-Gaussian Section 3 experiment. We rotate from the
anomaly direction to a stable nuisance direction:

```math
w(\theta)=\cos(\theta)e_0+\sin(\theta)e_1.
```

For each direction, we measure CAP on paired typical validation views, TPR
against shifted anomalies, and marginal W1/threshold drift on typical score
histograms. This tests the sufficient condition `lambda_Z > lambda_U`: nuisance
can be stable, but it is not stable enough to maximize `rho_R`.

### Results

| angle | c(w) | rho_Z | rho_U | rho_R | CAP emp. | CAP pop. | TPR emp. | TPR pop. | W1 emp. | W1 pop. | drift emp. | drift pop. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 1.0000 | 0.9000 | 0.0000 | 0.9000 | 0.4158 | 0.4171 | 0.5895 | 0.5910 | 0.0041 | 0.0000 | 0.0001 | 0.0000 |
| 30.3750 | 0.8627 | 0.6699 | 0.0307 | 0.7006 | 0.2292 | 0.2286 | 0.4072 | 0.4107 | 0.0064 | 0.0000 | 0.0002 | 0.0000 |
| 59.6250 | 0.5057 | 0.2301 | 0.0893 | 0.3194 | 0.0472 | 0.0481 | 0.0794 | 0.0791 | 0.0069 | 0.0000 | 0.0000 | 0.0000 |
| 90.0000 | 0.0000 | 0.0000 | 0.1200 | 0.1200 | 0.0066 | 0.0071 | 0.0009 | 0.0010 | 0.0057 | 0.0000 | 0.0000 | 0.0000 |

![Linear direction sweep](figures/section3/linear_direction_sweep.png)

![Population metric landscape](figures/section3/population_metric_landscape.png)

### Conclusions

- `w*` maximizes `rho_Z`, so it maximizes TPR.
- Since `lambda_Z > lambda_U`, `w*` also maximizes `rho_R`, so CAP selects it.
- The orthogonal nuisance direction has nonzero reproducibility, but not enough
  to beat the anomaly direction.
- Population W1 and threshold drift are zero for every unshifted direction
  because the two score marginals are identical; empirical values are finite
  validation-sample noise.

## Experiment 3: Score Distributions

### Setup

This experiment visualizes the laws behind the power formulas. It compares the
oracle linear score, an orthogonal linear score, and residual-style scores.
For residual scores

```math
S_R(X)=\sum_{j\in R}X_j^2,
```

the null law is central chi-square. If the residual set contains the anomaly
coordinate, the anomaly law is noncentral chi-square with

```math
\mathrm{nc}=\delta^2\lambda_Z.
```

If the residual excludes the anomaly coordinate, the noncentrality is zero.

### Results

![Linear score distributions](figures/section3/score_distributions.png)

![Score-family distributions](figures/section3/score_family_distributions.png)

### Conclusions

- The oracle linear score shifts by `delta sqrt(lambda_Z)`.
- The orthogonal linear score does not shift under anomalies.
- Residual scores have power only when their residual subspace includes the
  anomaly coordinate.
- The chi-square controls verify the non-linear score behavior independently of
  the linear theorem.

## Experiment 4: Diverse Score Families

### Setup

This experiment evaluates linear, residual, radial, wrong-orientation, and
collapsed scores. The goal is to show that the conclusions are not caused by a
single hand-picked linear candidate.

For linear scores, the table reports `rho_Z`, `rho_U`, `rho_R`, and
`pi = rho_Z/rho_R`. For non-linear residual scores those linear reliability
columns are not defined, but their null/anomaly laws are still analytic.

### Results

| score | rho_Z | rho_U | rho_R | pi | CAP | TPR pop. | TPR emp. | shifted W1 emp. | shifted W1 pop. | drift emp. | drift pop. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_oracle_w_star | 0.900000 | 0.000000 | 0.900000 | 1.000000 | 0.415708 | 0.591016 | 0.590113 | 0.997588 | 1.000000 | 0.015733 | 0.017298 |
| linear_mixed_45deg | 0.450000 | 0.060000 | 0.510000 | 0.882353 | 0.121271 | 0.228934 | 0.223307 | 0.707833 | 0.707107 | 0.006817 | 0.007583 |
| linear_noise_dim | 0.000000 | 0.120000 | 0.120000 | 0.000000 | 0.007398 | 0.001000 | 0.001040 | 0.010249 | 0.000000 | 0.000033 | 0.000000 |
| linear_negative_oracle | 0.900000 | 0.000000 | 0.900000 | 1.000000 | 0.415708 | 7.25e-11 | 0.000000 | 0.997588 | 1.000000 | 0.000967 | 0.000978 |
| residual_oracle_r1 | NA | NA | NA | NA | 0.337630 | 0.511913 | 0.512867 | 0.719967 | 0.706995 | 0.008533 | 0.010004 |
| residual_mixed_r2 | NA | NA | NA | NA | 0.079145 | 0.400405 | 0.405160 | 0.517946 | 0.499939 | 0.005767 | 0.005852 |
| radial_all | NA | NA | NA | NA | 0.010259 | 0.215634 | 0.228413 | 0.298619 | 0.288656 | 0.002317 | 0.002388 |
| residual_noise_r1 | NA | NA | NA | NA | 0.000019 | 0.001000 | 0.001007 | 0.013019 | 0.000000 | 0.000183 | 0.000000 |
| residual_without_anomaly | NA | NA | NA | NA | 0.000090 | 0.001000 | 0.001060 | 0.006143 | 0.000000 | 0.000033 | 0.000000 |
| constant_collapse | NA | NA | NA | NA | 0.000000 | NA | NA | 0.000000 | 0.000000 | 0.999000 | NA |

![Score-family comparison](figures/section3/score_family_comparison.png)

### Conclusions

- Oracle and mixed linear scores follow their `rho_Z` and `rho_R` values.
- The noise coordinate has reproducible nuisance variation but no anomaly
  alignment, so its TPR stays at the FPR target.
- The negative oracle has high CAP but nearly zero TPR because it violates the
  required score orientation.
- Residual and radial scores behave according to their chi-square laws.
- The collapsed score is a degenerate control, not a useful validation score.

## Experiment 5: Benign Marginal-Shift Trap

### Setup

The validation domains are shifted along the true anomaly direction:

```math
X^{(1)}\sim\mathcal N(0,I),
\qquad
X^{(2)}\sim\mathcal N(\eta e_0,I),
\qquad
\eta=1.
```

We evaluate TPR, W1, and threshold drift over the same linear direction sweep.
This isolates what marginal criteria are measuring: they prefer scores whose
validation histograms move less, even when those scores have lower anomaly
power.

### Results

| c(w) | TPR emp. | TPR pop. | W1 emp. | W1 pop. | threshold drift emp. | threshold drift pop. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000000 | 0.589233 | 0.591016 | 0.989093 | 1.000000 | 0.014883 | 0.017298 |
| 0.747508 | 0.271047 | 0.271523 | 0.738055 | 0.747508 | 0.006633 | 0.008572 |
| 0.505657 | 0.077740 | 0.079085 | 0.498458 | 0.505657 | 0.003767 | 0.003875 |
| 0.252492 | 0.011813 | 0.012166 | 0.247867 | 0.252492 | 0.000617 | 0.001272 |
| 0.000000 | 0.001033 | 0.001000 | 0.004509 | 0.000000 | 0.000167 | 0.000000 |

![Marginal shift trap](figures/section3/marginal_shift_trap.png)

### Conclusions

- TPR is maximized at `c(w)=1`.
- W1 and threshold drift are minimized near `c(w)=0`.
- The marginal diagnostics correctly detect less validation shift, but that
  selection would choose an almost powerless anomaly score.

## Experiment 6: Alignment Stress Test

### Setup

This population-only test compares two covariance regimes:

- aligned: `lambda_Z = 0.90`, `lambda_U = 0.12`;
- nuisance-dominated: `lambda_Z = 0.12`, `lambda_U = 0.90`.

The second case violates the sufficient condition
`lambda_Z > ||Gamma_U||_op`. It creates a reliable direction that is not the
anomaly direction.

### Results

| case | CAP argmax | rho_Z at CAP argmax | rho_R at CAP argmax | TPR argmax | rho_Z at TPR argmax | rho_R at TPR argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 0.0000 | 0.9000 | 0.9000 | 0.0000 | 0.9000 | 0.9000 |
| nuisance-dominated | 90.0000 | 0.0000 | 0.9000 | 0.0000 | 0.1200 | 0.1200 |

![Alignment assumption check](figures/section3/alignment_assumption_check.png)

### Conclusions

- In the aligned regime, CAP and TPR agree at `w*`.
- In the nuisance-dominated regime, CAP selects the stable nuisance direction
  because it maximizes `rho_R`.
- TPR still selects `w*`, because it depends on `rho_Z`.
- This is the empirical counterpart of the theory's alignment condition.

## Reproducibility Checklist

- All randomness uses `numpy.random.default_rng(seed)`.
- All defaults are CLI arguments in `src/analytical.py`.
- The script does not tune values after seeing outputs; it samples, computes
  metrics, and writes population formulas next to empirical estimates.
- Empirical CAP is computed through `ApproximationCapacityKernel`.
- Population CAP is deterministic Gauss-Hermite quadrature on the same beta
  grid.
- CSVs store `cap_raw_empirical` and `cap_empirical`, making the `+log(2)`
  lift checkable.
- W1 and threshold drift plots include empirical and population values.
- Residual scores use central and noncentral chi-square formulas.

## Overall Conclusions

- CAP is a measure of paired reproducible reliability `rho_R`.
- Fixed-FPR anomaly power is controlled by anomaly-aligned reliability `rho_Z`.
- CAP works for anomaly validation when the most reproducible directions are
  also the most anomaly-aligned directions.
- Stable nuisance is allowed, but it must not dominate total reproducible
  reliability.
- Marginal W1 and threshold drift cannot identify paired reliability from score
  histograms alone, and under benign shifts they can prefer lower-power scores.
