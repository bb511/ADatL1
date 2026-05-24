# Analytical Experiments for Section 3

This note documents the controlled experiments used to illustrate the theory in
`theory.tex`. The goal is deliberately narrow: instantiate the Gaussian
score-channel and linear stable-nuisance model, compute the quantities that the
theory discusses, and compare finite-sample estimates with the available
population formulas.

Run:

```bash
uv run python src/analytical.py --output-dir figures/section3
```

The script writes CSV files, figures, and `metadata.json` under
`figures/section3`.

## Model

The score-channel model is

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

`Z` is the anomaly-relevant coordinate. `U_phi` is reproducible nuisance:
future anomalies do not shift it. The two reliabilities are

```math
\rho_\phi^Z
  =
  \frac{a_\phi^2}{a_\phi^2+b_\phi^2+\sigma_\phi^2},
\qquad
\rho_\phi^R
  =
  \frac{a_\phi^2+b_\phi^2}{a_\phi^2+b_\phi^2+\sigma_\phi^2}.
```

The theory says fixed-FPR power is monotone in `rho_Z`, while CAP is monotone in
the paired reproducible reliability `rho_R`.

The feature-level experiment uses

```math
X^{(d)}
  =
  \sqrt{\lambda_Z}\,Z e_0
  +
  \sqrt{\lambda_U}\,U e_1
  +
  \xi^{(d)},
```

with unit marginal covariance for each validation view. Future anomalies shift
only `Z`, so the anomaly mean is

```math
\delta\sqrt{\lambda_Z}\,e_0.
```

For

```math
w(\theta)=\cos(\theta)e_0+\sin(\theta)e_1,
```

the population reliabilities are

```math
\rho_Z(w)=\lambda_Z\cos^2\theta,
\qquad
\rho_U(w)=\lambda_U\sin^2\theta,
\qquad
\rho_R(w)=\rho_Z(w)+\rho_U(w).
```

The default run uses

```text
d = 6
target FPR alpha = 1e-3
delta = 3.5
lambda_Z = 0.90
lambda_U = 0.12
n_pairs = 60000
n_test = 150000
n_match_features = 8
match_noise = 0.05
seed = 123
```

Because `lambda_Z > lambda_U`, the anomaly direction `w* = e0` maximizes both
`rho_Z` and `rho_R`.

## CAP Computation

Empirical CAP is computed with the repository
`ApproximationCapacityKernel`. The experiment uses the baseline binary energy
from the theory,

```math
E(p,y)=y(1-p)+(1-y)p.
```

To recover the theorem's special posterior, the script passes

```math
p = \frac{s+1}{2}
```

to the baseline-energy kernel. Then

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

The reported CAP lift is

```math
\widehat L^\star
  =
  \max_{\beta\in\mathcal B}
  \{\widehat C_\beta+\log 2\}.
```

Population CAP is deterministic Gauss-Hermite quadrature of the same
baseline-energy objective.

## Pairing

The empirical feature-level CAP uses static nearest-neighbor pairing in a
model-independent matching descriptor. Each synthetic object has an observed
descriptor

```math
M^{(d)} = M + \tau \zeta^{(d)},
```

with `tau = match_noise`. CAP pairs each object in validation view 1 to its
nearest neighbor in validation view 2 using this descriptor, then evaluates the
candidate scores on those pairs. The descriptor is not a candidate score and is
not changed across directions or epochs.

In the default run, the NN recovery rate is about `0.9999`, so empirical CAP is
effectively computed on the intended paired validation channel. This is the
controlled analogue of using a fixed reference representation or metadata-based
matching in real data.

## Experiment 1: Score-Channel Reliability

This experiment removes nuisance by setting `rho = rho_Z = rho_R`:

```math
S^{(d)}
  =
  \sqrt{\rho}\,Z
  +
  \sqrt{1-\rho}\,\varepsilon^{(d)}.
```

Anomaly scores follow

```math
S\mid Y=1\sim\mathcal N(\delta\sqrt{\rho},1).
```

| rho | cap_empirical | cap_theory | tpr_empirical | tpr_theory |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.0000 | 0.0000 | 0.0009 | 0.0010 |
| 0.2587 | 0.0321 | 0.0319 | 0.0904 | 0.0951 |
| 0.4850 | 0.1077 | 0.1086 | 0.2602 | 0.2570 |
| 0.7437 | 0.2631 | 0.2603 | 0.4764 | 0.4713 |
| 0.9700 | 0.5264 | 0.5200 | 0.6417 | 0.6394 |

![CAP and TPR reliability experiment](figures/section3/theory_cap_tpr_vs_reliability.png)

This validates the basic theorem: CAP increases with paired reliability, and
when there is no nuisance, the same parameter controls TPR.

## Experiment 2: Linear Stable-Nuisance Sweep

This experiment rotates the score direction from the anomaly coordinate to the
stable nuisance coordinate. CAP is evaluated on NN-paired typical validation
views. TPR is evaluated against shifted anomalies.

| angle_deg | cos_to_anomaly | rho_z | rho_u | rho_r | nn_pair_accuracy | cap_empirical | cap_theory | tpr_empirical | tpr_theory | wasserstein_empirical | wasserstein_theory | threshold_drift_empirical | threshold_drift_theory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 1.0000 | 0.9000 | 0.0000 | 0.9000 | 0.9999 | 0.4158 | 0.4171 | 0.5897 | 0.5910 | 0.0041 | 0.0000 | 0.0001 | 0.0000 |
| 30.3750 | 0.8627 | 0.6699 | 0.0307 | 0.7006 | 0.9999 | 0.2292 | 0.2286 | 0.4089 | 0.4107 | 0.0064 | 0.0000 | 0.0002 | 0.0000 |
| 59.6250 | 0.5057 | 0.2301 | 0.0893 | 0.3194 | 0.9999 | 0.0472 | 0.0481 | 0.0790 | 0.0791 | 0.0069 | 0.0000 | 0.0000 | 0.0000 |
| 90.0000 | 0.0000 | 0.0000 | 0.1200 | 0.1200 | 0.9999 | 0.0066 | 0.0071 | 0.0010 | 0.0010 | 0.0057 | 0.0000 | 0.0000 | 0.0000 |

![Linear direction sweep](figures/section3/linear_direction_sweep.png)

![Population metric landscape](figures/section3/population_metric_landscape.png)

The anomaly direction maximizes `rho_Z`, hence TPR. Since `lambda_Z >
lambda_U`, it also maximizes `rho_R`, hence CAP. Marginal W1 and threshold drift
are zero in population because every unit-norm linear score has standard normal
marginals in both unshifted validation views.

## Experiment 3: Alignment Stress Test

The theorem requires the most reproducible direction to also be
anomaly-aligned. This population-only stress test compares:

- aligned: `lambda_Z = 0.90`, `lambda_U = 0.12`;
- nuisance-dominated: `lambda_Z = 0.12`, `lambda_U = 0.90`.

| case | CAP argmax | rho_Z at CAP | rho_R at CAP | TPR argmax | rho_Z at TPR | rho_R at TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 0.0000 | 0.9000 | 0.9000 | 0.0000 | 0.9000 | 0.9000 |
| nuisance-dominated | 90.0000 | 0.0000 | 0.9000 | 0.0000 | 0.1200 | 0.1200 |

![Alignment assumption check](figures/section3/alignment_assumption_check.png)

This is the key limitation made explicit: CAP selects the most reproducible
score. If stable nuisance dominates reproducibility, CAP selects nuisance and
does not maximize TPR.

The same transition is clearer when sweeping the ratio
`\lambda_U/\lambda_Z`. Here `lambda_Z = 0.45` is fixed and `lambda_U` is varied.
The vertical threshold is the equality point `lambda_U = lambda_Z`.

| lambda_U/lambda_Z | selected | angle min | angle max | CAP max | CAP at w* | CAP at nuisance | TPR selected min | TPR optimal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2410 | anomaly | 0.0000 | 0.0000 | 0.0937 | 0.0937 | 0.0058 | 0.2289 | 0.2289 |
| 0.7411 | anomaly | 0.0000 | 0.0000 | 0.0937 | 0.0937 | 0.0524 | 0.2289 | 0.2289 |
| 1.0000 | tie | 0.0000 | 90.0000 | 0.0937 | 0.0937 | 0.0937 | 0.0010 | 0.2289 |
| 1.2527 | nuisance | 90.0000 | 90.0000 | 0.1462 | 0.0937 | 0.1462 | 0.0010 | 0.2289 |
| 2.0000 | nuisance | 90.0000 | 90.0000 | 0.4171 | 0.0937 | 0.4171 | 0.0010 | 0.2289 |

![Alignment ratio sweep](figures/section3/alignment_ratio_sweep.png)

Below the threshold, the maximum CAP value is attained at the anomaly direction.
At equality, CAP is flat across the two-dimensional span and cannot identify the
anomaly direction. Above the threshold, the maximum CAP value is attained at the
nuisance direction; the CAP-selected TPR collapses to the target FPR even though
the population CAP value increases.

## Experiment 4: CAP Versus Marginal Stability Under Benign Shift

This is the main counterexample to marginal stability. The two validation
domains are still paired through the same latent Gaussian structure, but the
second domain has a benign offset along the anomaly coordinate:

```math
X^{(1)}
  =
  \sqrt{\lambda_Z} Z e_0
  +
  \sqrt{\lambda_U} U e_1
  +
  \xi^{(1)},
\qquad
X^{(2)}
  =
  \sqrt{\lambda_Z} Z e_0
  +
  \sqrt{\lambda_U} U e_1
  +
  \eta e_0
  +
  \xi^{(2)},
\qquad
\eta=1.
```

For `w(theta)`, the paired reliability remains
`\rho_R(w)=\lambda_Z\cos^2\theta+\lambda_U\sin^2\theta`, so CAP is maximized
at the anomaly direction because `lambda_Z > lambda_U`. The marginal baselines
see only the score offset `eta cos(theta)`, so W1 and threshold drift are
minimized by the orthogonal direction, which has no anomaly power.

| angle_deg | cos_to_anomaly | cap_empirical | cap_theory | tpr_empirical | tpr_theory | wasserstein_empirical | wasserstein_theory | threshold_drift_empirical | threshold_drift_theory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000000 | 1.000000 | 0.164204 | 0.165945 | 0.590893 | 0.591016 | 1.000750 | 1.000000 | 0.017683 | 0.017298 |
| 30.375000 | 0.862734 | 0.119611 | 0.120705 | 0.409927 | 0.410750 | 0.860849 | 0.862734 | 0.013150 | 0.011957 |
| 60.750000 | 0.488621 | 0.034888 | 0.035698 | 0.071173 | 0.071077 | 0.484617 | 0.488621 | 0.004333 | 0.003639 |
| 90.000000 | 0.000000 | 0.006858 | 0.007059 | 0.000873 | 0.001000 | 0.007747 | 0.000000 | 0.000067 | 0.000000 |

| criterion | angle_deg | cos_to_anomaly | cap_theory | tpr_theory | wasserstein_theory | threshold_drift_theory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max CAP | 0.000000 | 1.000000 | 0.165945 | 0.591016 | 1.000000 | 0.017298 |
| max TPR | 0.000000 | 1.000000 | 0.165945 | 0.591016 | 1.000000 | 0.017298 |
| min W1 | 90.000000 | 0.000000 | 0.007059 | 0.001000 | 0.000000 | 0.000000 |
| min threshold drift | 90.000000 | 0.000000 | 0.007059 | 0.001000 | 0.000000 | 0.000000 |

![Marginal shift trap](figures/section3/marginal_shift_trap.png)

The conclusion is sharp: CAP and TPR select `w*`, while minimizing W1 or
threshold drift selects the orthogonal direction. The marginal criteria are not
wrong about marginal stability; they are measuring the wrong property for
signal-agnostic anomaly validation in this paired setting.

The same comparison can be repeated for every benign shift direction

```math
\eta\{\cos(\psi)e_0+\sin(\psi)e_1\},
\qquad
\psi\in[0,\pi/2].
```

For each shift angle `psi`, the experiment records only the selected score
angle. CAP selects the maximizer of the population paired objective. W1 and
threshold drift are stability criteria, so they select the minimizer. In this
default model, evaluating the population CAP objective over all score angles
selects the anomaly direction for every shift angle,

```math
\theta_{\mathrm{CAP}}
  \in
  \arg\max_\theta L_{\mathrm{CAP}}(\theta;\psi)
  =
  \{0\},
```

matching the maximizer of the paired reliability
`\rho_R(\theta)`. In contrast, the marginal score shift is

```math
\eta\cos(\theta-\psi).
```

Thus the marginal selectors choose the score direction farthest from the shift,
not the direction with anomaly power.

| shift angle | shift Z | shift U | CAP selected | W1 selected | threshold selected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 1.0000 | 0.0000 | 0.0000 | 90.0000 | 90.0000 |
| 30.3750 | 0.8627 | 0.5057 | 0.0000 | 90.0000 | 90.0000 |
| 45.0000 | 0.7071 | 0.7071 | 0.0000 | 0.0000--90.0000 | 0.0000--90.0000 |
| 59.6250 | 0.5057 | 0.8627 | 0.0000 | 0.0000 | 0.0000 |
| 90.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

![Marginal shift selector sweep](figures/section3/marginal_shift_selector_sweep.png)

## Reproducibility Notes

- All randomness uses `numpy.random.default_rng(seed)`.
- All defaults are CLI arguments in `src/analytical.py`.
- Empirical CAP uses the repository `ApproximationCapacityKernel`.
- Empirical feature-level CAP uses static NN pairing in the matching descriptor.
- Population CAP uses deterministic Gauss-Hermite quadrature for the same
  baseline-energy objective.
- CSVs store both `cap_raw_empirical` and `cap_empirical`, so the `+log(2)` lift
  is auditable.

## Overall Conclusion

The experiment supports the precise theory claim:

- CAP measures paired reproducible reliability `rho_R`;
- fixed-FPR anomaly power is controlled by anomaly-aligned reliability `rho_Z`;
- CAP is useful for signal-agnostic validation when the most reproducible score
  directions are also anomaly-aligned;
- stable nuisance is allowed, but it must not dominate;
- marginal histogram stability is not enough to identify paired reliability or
  anomaly power.
