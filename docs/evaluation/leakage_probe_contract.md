# Leakage Probe Scientific Contract

## Status and scope

**Protocol version:** `fet-et-four-probe-v6`

**Status:** Frozen for the FET.Et proof-of-concept study.

This document defines the leakage measurement used to compare ADatL1 autoencoder
configurations on the validation Pareto front. It fixes the target, representations,
probe family, data splits, metrics, controls, and invalid-run behavior before the
hyperparameter results are inspected.

Every run included in one comparison must use this exact protocol version. A change to
any primary choice below requires a new protocol version and a new comparison; results
from different protocol versions must not be combined into one Pareto analysis.

This protocol concerns post-training leakage probes only. It does not define the MI
loss, collapse threshold, correlation aggregation, signal-efficiency constraint, or
final Pareto-ranking weights.

The primary scientific claim supported by this protocol is:

> On held-out data, the selected autoencoder configuration shows low recoverability of
> raw ordered FET.Et from both the pre-Bernoulli latent logits and the reconstructed
> feature vector under the fixed MLP and linear-regression probe families defined here.

This is evidence about the stated probe family and evaluation distribution. It is not
proof of statistical independence.

## 1. Sensitive target

### 1.1 Current proof-of-concept target

The leakage probes predict:

\[
S_{\mathrm{raw}} = \mathrm{FET.Et}.
\]

The target contract is:

| Property | Frozen value |
|---|---|
| Variable | `FET.Et` |
| Semantic type | Ordered continuous regression target |
| Source tensor | `control_x` |
| Reduction | `first` |
| Evaluation representation | Denormalized physical value |
| Unit | GeV |
| Expected shape | One scalar per event |

The probe target is deliberately different from the target representation used by the
discrete MI estimator:

\[
S_{\mathrm{bin}}
  = \operatorname{bin}(S_{\mathrm{raw}}; B_{\mathrm{train}}).
\]

`S_bin` is used only by the MI training loss. All four leakage probes predict the
same `S_raw`, regardless of `algorithm.mi_sensitive_num_bins`. The probe implementation
must request denormalized values explicitly and must not inherit the binner's
`mi_sensitive_use_denormalization` setting implicitly.

The target must remain available in the control tensor but must be absent from the
autoencoder model-input feature map and from the reconstruction feature map. This
absence must be checked at runtime.

### 1.2 Future pileup target

The final pileup study will replace the target with the float-valued field
`nPV_True`. That change requires a new protocol version. The regression formulation
remains appropriate because `nPV_True` is ordered and numerical.

The final pileup protocol must additionally preserve `run` and `lumi` as control-only
metadata and use group-disjoint splitting by `(run, lumi)`. No luminosity section may
appear in more than one probe split.

## 2. Probe representations

### 2.1 Primary latent representation

The primary latent representation is named `latent_logits` and is defined as the
encoder output before Bernoulli sampling. It is the same tensor supplied to the MI
estimator during autoencoder training.

The primary latent probe is:

\[
g_Z: Z_{\mathrm{logits}} \rightarrow S_{\mathrm{raw}}.
\]

Stored metrics and artifacts must use `z_logits` or `latent_logits` in their names.
They must not use the ambiguous name `z` without qualification.

### 2.2 Reconstruction representation

The primary reconstruction representation is named `reconstructed_data` and is the
complete decoder output corresponding to the autoencoder model-input layout.

The primary reconstruction probe is:

\[
g_{\hat X}: \hat X \rightarrow S_{\mathrm{raw}}.
\]

Its feature matrix contains only `reconstructed_data`. The following must never be
appended to it:

- `control_x` or `control_mask`;
- the model-input mask;
- event or dataset labels;
- FET.Et or any transformation of FET.Et;
- MI bin labels;
- run, luminosity, or event metadata.

### 2.3 Binary latent representation

The deterministic hard Bernoulli code produced in evaluation mode is named
`latent_sample`. It is consumed by the decoder and may be used by collapse diagnostics,
whose definitions are outside this contract. It is not an input to a leakage probe.

## 3. Autoencoder checkpoint and inference behavior

Every leakage evaluation uses exactly one autoencoder checkpoint:

```text
loss_total.ckpt
```

It is the checkpoint with the minimum complete-epoch normal-validation objective

\[
\mathrm{val/loss\_total}
  = \mathrm{val/loss\_reco}
  + \gamma\,\mathrm{val/loss\_mi}.
\]

Before representation extraction, the evaluator must:

1. load `loss_total.ckpt` explicitly;
2. set the autoencoder to evaluation mode;
3. freeze its parameters;
4. run under `torch.inference_mode()`;
5. verify that Bernoulli sampling produces its deterministic evaluation-time hard
   code;
6. detach representations and targets before transferring them to CPU.

No probe operation may update autoencoder parameters or influence checkpoint
selection.

## 4. Probe family and preprocessing

The primary measurement uses four independent scikit-learn regressors:

1. `MLPRegressor`: `latent_logits -> FET.Et`;
2. `MLPRegressor`: `reconstructed_data -> FET.Et`;
3. `LinearRegression`: `latent_logits -> FET.Et`;
4. `LinearRegression`: `reconstructed_data -> FET.Et`.

The fixed MLP configuration is:

| Parameter | Frozen value |
|---|---|
| `hidden_layer_sizes` | `(64, 32)` |
| `activation` | `relu` |
| `solver` | `adam` |
| `alpha` | `1e-4` |
| `learning_rate` | `constant` |
| `learning_rate_init` | `1e-3` |
| `max_iter` | `500` |
| `shuffle` | `true` |
| `early_stopping` | `true` |
| `validation_fraction` | `0.1` |
| `n_iter_no_change` | `10` |
| `tol` | `1e-4` |
| `beta_1` | `0.9` |
| `beta_2` | `0.999` |
| `epsilon` | `1e-8` |
| Initialization seeds | `[10, 123, 500]` |

Each probe uses a separate feature `StandardScaler` fitted only on its probe-training
features. The MLP target is standardized using parameters fitted only on the
corresponding probe-training target. MLP predictions are inverse-transformed before
MAE is calculated. The linear probes fit the physical target directly. Consequently,
MAE for all four probes remains in GeV.

All four probes have separate scalers, estimators, and fitted parameters. The two MLPs
also have independent seed selection. The probe families and MLP training budget are
fixed measurement instruments and are not autoencoder Pareto hyperparameters.

## 5. Data-splitting and model-selection protocol

### 5.1 Outer split

The current study has no separate development dataset. It therefore uses two explicit
evaluation modes:

| Mode | Probe development and fitting | Held-out scoring | Use |
|---|---|---|---|
| `validation` | `train` | `valid` | Every run used to construct the validation Pareto front |
| `final_test` | `train + valid` | `test` | Only the final selected configuration |

The validation split is held out from probe fitting during hyperparameter selection.
After one autoencoder configuration has been selected, validation is no longer an
unseen selection set and is added to the final probe-development pool. The test split
remains untouched until that decision and is never used for scaling, MLP seed
selection, early stopping, or any probe hyperparameter.

The physics sweep configuration fixes `test=false` and `mode=validation`. After the
final configuration is selected, enable the complete final evaluation explicitly:

```text
test=true \
evaluation.leakage_probes.mode=final_test \
evaluation.leakage_probes.run_shuffled_target_controls=true
```

### 5.2 Inner probe split

The active probe-development pool (`train` in validation mode and `train + valid` in
final-test mode) is divided deterministically into:

- 80% `probe_fit`;
- 20% `probe_inner_validation`.

The inner split seed is `12345`.

For each primary representation independently, the MLP procedure is:

1. fit one MLP for each frozen initialization seed on `probe_fit`;
2. evaluate each candidate on `probe_inner_validation`;
3. choose the seed with the highest raw inner-validation R2;
4. refit that selected seed and unchanged MLP configuration on the complete active
   probe-development pool;
5. evaluate the refitted probe once on the mode's held-out split.

For each primary representation independently, fit one `LinearRegression` on the
complete active probe-development pool and evaluate it once on the same held-out
split. Linear regression has no probe-seed selection.

The active held-out target must not influence scaling, MLP seed selection, early
stopping, or any probe hyperparameter.

### 5.3 Event-set consistency

Probe loaders must be unshuffled. If representations are subsampled, the evaluator
must use a deterministic index manifest generated with sample seed `12345` and reuse
the identical event positions for every autoencoder configuration and seed.

For protocol version `fet-et-four-probe-v6`, `max_samples` is `null`: all available
events in the relevant split are used. Introducing a sample cap requires a new
protocol version unless the cap is fixed before any comparable run is evaluated and
all earlier runs are reevaluated with the same manifest.

The event manifest is a SHA-256 hash of the actual ordered cached input, padding-mask,
and L1-bit tensor content. It is independent of dataloader batch boundaries and detects
a changed or reordered dataset even when the event count is unchanged. The evaluator
also records the resolved cache identity, source split names, event count, sample seed,
and sample cap for both the development and held-out pools. Paired autoencoder seeds
may be aggregated only when these comparable provenance fields are identical.

## 6. Primary leakage metrics

Calculate held-out R2 for each family and representation:

\[
R^2_{\mathrm{MLP},Z},\quad
R^2_{\mathrm{MLP},\hat X},\quad
R^2_{\mathrm{linear},Z},\quad
R^2_{\mathrm{linear},\hat X}.
\]

Raw R2 values, including negative values, must be retained. Define four nonnegative
leakage components:

\[
L_{f,r}=\max(0,R^2_{f,r}),
\qquad
f\in\{\mathrm{MLP},\mathrm{linear}\},
\quad
r\in\{Z,\hat X\}.
\]

The primary run-level leakage objective is:

\[
L=\max\left(
L_{\mathrm{MLP},Z},
L_{\mathrm{MLP},\hat X},
L_{\mathrm{linear},Z},
L_{\mathrm{linear},\hat X}
\right).
\]

The linear scores are included even though an adequately optimized MLP can represent
linear relationships. The MLP has a finite architecture and non-convex training
procedure, whereas ordinary least squares provides a deterministic check for directly
recoverable linear signal. Including both families prevents a weak MLP fit from making
the reported leakage artificially optimistic.

Lower is better. The maximum is required because low leakage is required for both
probe families at both locations; averaging the components is not permitted. Exact
ties are resolved in the displayed order.

`L = 0` means that none of the four probes explains positive held-out variance relative
to the standard R2 reference. It does not mean negative information and does not prove
independence.

For all four primary probes, also report MAE in GeV. MAE is supporting information and
is not combined with R2 or with `L`.

## 7. Diagnostics and optional negative controls

Every primary MLP initialization records its convergence warning, iteration count, and
final loss. Shuffled-target MLP controls are an optional audit mode rather than a
requirement for every comparable run.

### 7.1 Purpose, implementation, and use

The shuffled-target controls are negative controls for the probe pipeline. They test
whether an MLP can obtain suspiciously positive held-out R2 after the association
between each training representation and its target has deliberately been destroyed.
They can reveal problems such as train/validation overlap, target misalignment, or an
unstable finite-sample result. Passing the controls supports the integrity of the
measurement pipeline; it does not establish that the primary representations contain
no sensitive information.

The implementation is controlled by
`evaluation.leakage_probes.run_shuffled_target_controls` in
`configs/experiment/physics/ae.yaml` and propagated by `src/train.py` to the leakage
evaluator. Its behavior is:

- `false` (the physics default): run and persist the four primary probes, skip all
  shuffled-target fits and metrics, record
  `diagnostics.shuffled_targets.enabled=false`, and do not apply the shuffled-target
  guardrail;
- `true`: shuffle the complete training target deterministically, run the same MLP
  seed-selection and refit procedure for both primary representations, persist the
  control results separately, log the two `probe/shuffled/*` metrics, and apply the
  guardrail below.

One enabled control run adds eight MLP fits: three seed candidates plus one refit for
each of the two representations. The controls never alter the four primary results or
enter `probe/leakage_worst`.

Use `false` for broad hyperparameter sweeps, where repeating this audit for every run
adds substantial cost without changing the Pareto objective. Use `true`:

- on an initial end-to-end run to validate the evaluation pipeline;
- on a representative run after changes to data loading, event splitting, target
  extraction, or probe fitting;
- for the final selected configuration before reporting its leakage result.

Enable audit mode by appending this Hydra override to the normal training command:

```text
evaluation.leakage_probes.run_shuffled_target_controls=true
```

The shuffled-target control uses a NumPy `RandomState` MT19937
permutation with seed `12345`. The permutation is applied only to
the complete probe-development target. The held-out target is never
shuffled. Both primary representations reuse the identical
permuted training-target vector.

### 7.2 Shuffled-target guardrail when enabled

For both primary representations, clipped shuffled-target validation leakage must be
at most 0.02. A larger value marks the probe measurement invalid and requires an audit
of splitting, target alignment, and finite-sample behavior.

## 8. Invalid-run behavior

A leakage evaluation is invalid if any of the following occurs:

- `loss_total.ckpt` is missing, corrupt, or cannot be loaded strictly;
- target or representation values contain NaN or infinity;
- representation and target row counts differ;
- the target is constant or has fewer than two distinct finite values;
- the requested physical target cannot be denormalized;
- FET.Et or a direct transformation of it appears in either primary feature matrix;
- the probe-fit, inner-validation, and held-out partitions overlap;
- comparable autoencoder configurations use different held-out event samples;
- every MLP initialization raises an exception or produces non-finite predictions;
- the selected MLP has non-finite weights, loss, or predictions;
- the shuffled-target guardrail fails when the controls are enabled;
- a required primary metric cannot be calculated.

A scikit-learn `ConvergenceWarning` caused solely by reaching `max_iter=500` is logged
but does not automatically invalidate the result if the fitted estimator and all
metrics are finite and, when enabled, the shuffled-target guardrail passes.

Invalid evaluations remain in the run table with `probe_valid=false` and an explicit,
machine-readable rejection reason. Missing or invalid scores must never be replaced by
zero leakage.

## 9. Required metric and artifact names

The following MLflow metric names are fixed:

```text
probe/mlp/z_logits/r2_raw
probe/mlp/z_logits/r2_clipped
probe/mlp/z_logits/mae_gev
probe/mlp/reconstruction/r2_raw
probe/mlp/reconstruction/r2_clipped
probe/mlp/reconstruction/mae_gev

probe/linear/z_logits/r2_raw
probe/linear/z_logits/r2_clipped
probe/linear/z_logits/mae_gev
probe/linear/reconstruction/r2_raw
probe/linear/reconstruction/r2_clipped
probe/linear/reconstruction/mae_gev

probe/leakage_worst

probe/shuffled/z_logits/r2_raw          # audit mode only
probe/shuffled/reconstruction/r2_raw    # audit mode only
```

The only leakage value supplied to the Pareto analysis is:

```text
probe/leakage_worst
```

Each validation-mode run must write the machine-readable artifact:

```text
plots/val/loss_total/probes/leakage_probes.json
```

A final-test run writes the same schema at:

```text
plots/test/loss_total/probes/leakage_probes.json
```

It contains at least:

- `leakage_probe_protocol_version`;
- `run.autoencoder_seed` and the seed-independent `run.configuration_id`;
- `evaluation.mode`, development-data provenance, and held-out-data provenance;
- `worst_probe` and `leakage_worst`;
- the representation name and dimension for each of the four probes;
- raw and clipped R2 plus MAE for each of the four probes;
- development and held-out event counts for each of the four probes;
- the selected seed and convergence information for each MLP;
- `diagnostics.shuffled_targets.enabled`, plus the shuffled-control results when
  audit mode is enabled.

## 10. Cross-run aggregation

The probe evaluator produces one primary `L` per trained autoencoder run and seed.
Probe initialization seeds belong to the measurement procedure; they are not
autoencoder replicates and must not be pooled with autoencoder seeds.

For a hyperparameter configuration, aggregate run-level leakage only across the
predeclared paired autoencoder seeds. The frozen invalid-run policy is
`reject_configuration`: if any expected autoencoder seed is missing or has
`probe_valid=false`, the configuration has no aggregate leakage score and cannot enter
the Pareto front. Individual valid and invalid run records remain in the aggregate
artifact; an invalid score is never replaced with zero or averaged away.

For a complete valid seed set, report the mean, sample standard deviation, standard
error, and normal-approximation 95% confidence interval of `leakage_worst`. The
aggregation tool also rejects mixed protocol versions, configuration identities,
evaluation modes, cache identities, event manifests, and sampling protocols.

Run the aggregator with the explicitly predeclared autoencoder seeds:

```text
python scripts/aggregate_leakage_probes.py \
  --expected-seeds <seed-1> <seed-2> ... \
  --output <configuration>/leakage_probe_aggregate.json \
  <run-1>/plots/val/loss_total/probes/leakage_probes.json \
  <run-2>/plots/val/loss_total/probes/leakage_probes.json ...
```

Runs with different `leakage_probe_protocol_version` values, target definitions,
sample manifests, or outer split identities must not be aggregated together.

## 11. Definition of done

An implementation conforms to `fet-et-four-probe-v6` only when all of the following
are true:

- both `latent_logits` and `reconstructed_data` are evaluated by independent primary
  MLP and linear regressors, producing four independent fitted probes;
- the target is denormalized FET.Et in GeV and is identical across all MI bin counts;
- FET.Et is absent from both primary feature matrices;
- validation-mode probe fitting and seed selection use only `train`;
- reported Pareto leakage uses only the held-out autoencoder validation split;
- the test split is not evaluated during hyperparameter selection;
- final-test mode uses `train + valid` for probe development and untouched `test` for
  scoring only after one configuration has been selected;
- raw and clipped R2 plus MAE are stored for all four primary probes;
- `L` is the maximum of all four clipped primary R2 values;
- the artifact records whether shuffled-target controls ran; when enabled, their
  diagnostics are stored separately and cannot enter `L`;
- `leakage_probes.json` records all four results, `worst_probe`, and `leakage_worst` at
  the required checkpoint-relative path;
- invalid measurements fail visibly and are never converted to zero;
- actual cached event content and selection are identical across comparable
  autoencoder runs;
- paired-seed aggregation rejects the complete configuration when an expected seed is
  missing or invalid;
- every output records protocol version `fet-et-four-probe-v6`.
