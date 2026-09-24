# The FET.Et validation Pareto study

**Study** `fet-et-pareto-v1` · **selection protocol** `fet-et-pareto-v1` · **probe protocol** `fet-et-four-probe-v10`
**Results** `outputs/pareto_studies/fet-et-pareto-v1/` · **figures** `phase4/`

---

## 1. The question

The anomaly detector is a Bernoulli autoencoder over L1 trigger objects. Its latent code
is an 8-bit bottleneck and the reconstruction error is the anomaly score. The concern is
that such a detector may fire preferentially on high-energy events rather than on
genuinely anomalous ones: if the latent code carries the total transverse energy
`FET.Et`, the trigger degenerates into an expensive energy threshold, and its rate is
neither predictable nor physics-motivated.

Two things are done about it. `FET.Et` is removed from the model input outright, so the
encoder sees 116 features rather than 117; and a Bernoulli mutual-information penalty
weighted by $\gamma$ pushes the code towards statistical independence from a
quantile-binned `FET.Et`. Because `FET.Et` is not an input, any dependence that survives
is *indirect*, arriving through its correlations with the other 116 features.

Removing that dependence is not free — a code that knows nothing about energy may also
detect less. The study therefore asks a trade-off question, not an optimisation question:
**across penalty strengths, binning resolutions and encoder architectures, which
configurations are not beaten on every axis at once?** The answer is a Pareto front, not
a winner.

## 2. Notation

| symbol | meaning |
|---|---|
| $N$ | number of events in the split under consideration |
| $x \in \mathbb{R}^{116}$ | model input for one event, `FET.Et` excluded |
| $\hat{x} \in \mathbb{R}^{116}$ | the autoencoder's reconstruction of $x$ |
| $s \in \mathbb{R}$ | the sensitive variable, raw `FET.Et` in GeV |
| $S \in \{0,\dots,B_{\text{eff}}-1\}$ | $s$ after quantile binning; the MI target |
| $z \in \mathbb{R}^{8}$ | latent logits, the encoder output before sampling |
| $b \in \{0,1\}^{8}$ | the hard Bernoulli latent sample (the 8-bit code) |
| $j = 1..8$ | index over latent units |
| $\gamma$ | weight of the MI penalty in the loss |
| $B$, $B_{\text{eff}}$ | requested and effective number of quantile bins |
| $T = 6$ | sigmoid temperature in the MI estimator |
| $L$, $E$, $\epsilon_{\text{med}}$ | the three Pareto objectives |
| $k$ | seed index; $k \in \{123, 500\}$ |

## 3. Configuration versus run

A **configuration** is a point on the grid: one triple
$(\gamma, B, \text{architecture})$. A **run** is one training of an autoencoder at that
configuration with one seed. Configurations are the unit of comparison, runs the unit of
training. The identifier `configuration_id` deliberately excludes the seed, which is what
makes two runs aggregate into one configuration automatically:

```
fet-et-pareto-v1__gamma-0.1__bins-50__arch-h64_64_32
```

Every run leaves a `run_manifest.yaml` beside its checkpoint, so a directory of
autoencoders describes itself and the front can be thrown over whatever has been trained,
whenever and on whatever resources.

## 4. The grid

| axis | values | note |
|---|---|---|
| $\gamma$ | 0.05, 0.10, 0.15, 0.20, 0.25 | penalty strength |
| $B$ | 40, 50, 60 | requested quantile bins |
| architecture | `h64_32` (64,32,8), `h128_64` (128,64,8), `h64_64_32` (64,64,32,8) | latent width fixed at 8 |
| seeds | 123, 500 | both required |

That is $5 \times 3 \times 3 = 45$ regularised configurations plus one $\gamma = 0$
baseline per architecture. Binning is meaningless without a penalty, so the baselines use
a single canonical 50-bin setting rather than three redundant copies — 48 configurations,
96 training runs.

The latent width is frozen at 8 across the grid
(`candidate_architecture_contract: encoder_nodes[-1] == 8`), so the architectures differ
in encoder capacity, not in bottleneck size. All three compress to the same 8 bits.

## 5. Definitions

### 5.1 Quantile binning of the sensitive variable

The MI penalty needs a discrete target. Bin edges are the $B-1$ interior quantiles of
$s$ over the full training split, computed once before training and frozen into the
checkpoint:

$$q_k = \mathrm{Quantile}\!\left(s_{\text{train}},\ \tfrac{k}{B}\right), \qquad k = 1,\dots,B-1$$

$$S(s) = \sum_{k=1}^{B-1} \mathbb{1}[\,s > q_k\,]$$

Quantiles rather than equal-width bins because the energy spectrum is steeply falling:
equal-width bins would put almost every event in the first bin and leave the rest nearly
empty, which makes the conditional entropy in section 5.2 an average over bins containing
a handful of events. Equal-occupancy bins give every conditional term comparable
statistics.

**Effective bins.** Duplicate quantile edges are collapsed, so
$B_{\text{eff}} = |\{\text{unique } q_k\}| + 1 \le B$. For `FET.Et` this bites: a request
for $B = 60$ yields $B_{\text{eff}} = 53$. The manifest and the `configuration_id` record
the *requested* $B$; the effective count is recoverable from the checkpoint buffer
`sensitive_bin_edges_count` $+\,1$.

### 5.2 The Bernoulli mutual-information penalty

This is the training objective, not an evaluation metric. Each latent logit is read as a
Bernoulli logit and mapped to an activation probability with a sharpening temperature.
Averaging over a batch gives a per-unit activation rate:

$$\theta_j = \frac{1}{N}\sum_{i=1}^{N}\sigma(T\,z_{ij}), \qquad T = 6$$

With $H_b(\theta) = -\theta\log_2\theta - (1-\theta)\log_2(1-\theta)$ the binary entropy,
the estimator sums the per-unit entropies and subtracts the class-conditional average:

$$\hat{H}(L) = \sum_{j=1}^{8} H_b(\theta_j), \qquad
\hat{H}(L\mid S=s) = \sum_{j=1}^{8} H_b\!\left(\theta_j^{(s)}\right)$$

$$\hat{I}(L;S) = \hat{H}(L) - \sum_{s} p(s)\,\hat{H}(L \mid S = s), \qquad p(s) = \frac{N_s}{N}$$

where $\theta_j^{(s)}$ is the activation rate of unit $j$ over the events in bin $s$. The
total loss is

$$\mathcal{L} = \mathcal{L}_{\text{reco}} + \gamma\,\hat{I}(L;S)$$

**This estimator is factorised, and that matters.** $\hat{H}(L)$ is the *sum of per-unit
marginal entropies*, not the joint entropy of the 8-bit code, so $\hat{I}$ is
$\sum_j I(L_j; S)$ — the total of the eight per-bit mutual informations. Any information
about $S$ carried *jointly* across bits, in a way no single bit reveals, is invisible to
the penalty. The leakage probes in section 5.3 see all eight logits at once and would
find exactly that information. This is a concrete candidate mechanism for the null result
in section 10 and should be checked before the grid is blamed.

The choice of estimator is inherited from `hepinfo` for compatibility, including the
float64 accumulation, the $10^{-20}$ logarithm guard and the float32 return.

### 5.3 Leakage $L$

How well the sensitive variable can be recovered from what the model produces. Four
probes are fitted on the training split ($N = 12{,}532{,}307$) and scored once on the
held-out validation split ($N = 4{,}177{,}435$): an MLP and an ordinary least-squares
regression, each on the latent logits $z$ (8 features) and on the reconstruction
$\hat{x}$ (116 features).

The target is **raw $s$ in GeV**, taken before binning. Each probe $p$ predicts
$\hat{s}^{(p)}_i$ on held-out events, and is scored by the coefficient of determination

$$R^2_p = 1 - \frac{\sum_i \left(s_i - \hat{s}^{(p)}_i\right)^2}{\sum_i \left(s_i - \bar{s}\right)^2},
\qquad R^{2,\text{clip}}_p = \max\!\left(0,\ R^2_p\right)$$

$$L = \max_{p \in \{\text{mlp},\,\text{lin}\} \times \{z,\,\hat{x}\}} R^{2,\text{clip}}_p$$

Four choices are worth defending.

*Why the maximum over probes.* A quantity recoverable by **any** probe is leaked. Taking
the worst case removes the need to argue which probe is the right one, and makes $L$ a
statement about the representation rather than about a particular estimator.

*Why both an MLP and a linear probe.* The linear probe finds affine dependence exactly,
cheaply and without hyperparameters; the MLP can find nonlinear structure the linear one
cannot. Reporting the maximum means neither has to be trusted alone. In practice they
agree closely here, which is itself informative — the leakage that exists is almost
entirely linear.

*Why clipping at zero.* $R^2 < 0$ means the probe predicts worse than the constant mean,
which is not "negative information" but no information. Clipping makes $L$ a bounded
quantity in $[0,1]$ that reads as "the fraction of variance in $s$ recoverable from the
representation", which is what the ranking in section 8 assumes.

*Why the raw target rather than the binned one.* It keeps $L$ independent of the binning
axis of the grid, so the three grid axes stay orthogonal in the results. Probing against
$S$ instead would confound leakage with binning resolution.

The MLP is a (64, 32) ReLU network trained with Adam ($\text{lr} = 10^{-3}$,
$\alpha = 10^{-4}$, batch 16384, $\le 500$ epochs, early stopping on an internal 10%
split with $\text{tol} = 10^{-4}$ and patience 10), from one frozen initialisation
(seed 123); there is no seed search and no inner partition. Features and target are
standardised on the development pool only. The linear probe is solved in float64 by a
blocked Householder QR followed by an SVD with dimension-aware rank truncation — the
reconstruction features are a deterministic function of an 8-bit latent and therefore
strongly collinear, and a float32 or normal-equations solve is numerically unsafe on
them.

### 5.4 Residual correlation $E$

$L$ asks whether energy is *recoverable* from the representation. $E$ asks whether the
model's output is still *ordered* by energy — a detector can pass the first and fail the
second, which is why $E$ is a Pareto objective and not a dashboard check.

Over the normal validation set, for each of the 21 configured variables $v$ other than
`FET.Et` itself, the correlation is taken between the **true** sensitive value $s$ and
the **reconstructed** value $\hat{x}_v$:

$$\bar{\rho}_{\text{P}} = \frac{1}{|V|}\sum_{v \in V}\left|\rho_{\text{Pearson}}(s,\ \hat{x}_v)\right|,
\qquad
\bar{\rho}_{\text{S}} = \frac{1}{|V|}\sum_{v \in V}\left|\rho_{\text{Spearman}}(s,\ \hat{x}_v)\right|$$

$$E = \max\!\left(\max(0, \bar{\rho}_{\text{P}}),\ \max(0, \bar{\rho}_{\text{S}})\right)$$

Absolute values because sign is irrelevant — an anti-correlation with energy is just as
much a dependence. Pearson **and** Spearman because Pearson catches linear dependence and
Spearman catches any monotone dependence; taking the maximum again avoids having to
choose. Both components lie in $[0,1]$ by construction, so no candidate-set-dependent
rescaling is needed.

**A naming warning.** The column is called `residual_correlation` and the code's Pareto
symbol is `C`, but `include_residual` is `false` in the configuration: this is computed in
the *reconstruction* space, not on residuals $\hat{x} - x$. The name is misleading and
should be read as "correlation of the reconstruction with the sensitive variable".

### 5.5 Signal efficiency $\epsilon_{\text{med}}$

The trigger runs at a fixed rate, so efficiency is measured at a fixed operating point
rather than swept. The target false-positive rate is the ratio of the target to the base
trigger rate,

$$\text{FPR} = \frac{r_{\text{target}}}{r_{\text{base}}}$$

and the threshold is the corresponding upper quantile of the anomaly score on the normal
(background) validation sample:

$$t = \mathrm{Quantile}\!\left(a_{\text{bkg}},\ 1 - \text{FPR}\right)$$

computed with `interpolation="higher"`, so the realised FPR never exceeds the target. The
efficiency of signal sample $d$ is the exceedance fraction at that same threshold,

$$\epsilon_d = \frac{1}{N_d}\sum_{i=1}^{N_d}\mathbb{1}\!\left[a_i^{(d)} > t\right]$$

and the objective is the median over the 21 signal samples,
$\epsilon_{\text{med}} = \mathrm{median}_d\,\epsilon_d$.

The threshold is computed once during the fit-time validation loop and stored *in the
checkpoint*, so every later stage scores against the identical operating point.

*Why the median.* It is the typical signal rather than the best or worst, and it is
robust to one unusually easy or unusually hard sample among the 21. The mean would let a
single trivially-detected signal carry a configuration; the minimum would make the
objective a single-sample statistic. The minimum is not discarded — it becomes the
feasibility gate in section 7, where "did this configuration abandon a signal class
entirely?" is a yes/no question rather than a quantity to trade against.

Two further summaries are computed and stored but are not objectives:
$\epsilon_{\text{mean}}$, and the 25% conditional value at risk
$\epsilon_{\text{CVaR25}} = \frac{1}{k}\sum \text{(the } k = \lceil 0.25 N_{\text{sig}}\rceil
\text{ smallest } \epsilon_d)$, the average efficiency over the worst quarter of signals.

### 5.6 Latent-collapse statistics

**Why this exists at all.** The leakage objective can be gamed. A latent code that has
collapsed to a constant carries no information about anything, so its mutual information
with `FET.Et` is exactly zero and every probe in section 5.3 returns $R^2 \le 0$, giving
$L = 0$ — a perfect score on the study's primary axis, achieved by a dead autoencoder.
Minimising $L$ alone therefore has a degenerate optimum. The collapse gate is what makes
the leakage axis meaningful: it disqualifies the degenerate solution before the comparison
begins, rather than relying on the efficiency objective to notice.

**What is measured.** The hard 8-bit code $b \in \{0,1\}^8$ produced by the Bernoulli
layer — not the continuous logits — because that is the representation the deployed
trigger would actually transmit, and because the logits would require an arbitrary
discretisation choice of their own. At evaluation time the Bernoulli sampling is
deterministic (threshold 0.5), and the callback asserts this by re-running the first
batch and requiring an identical code. The statistics are accumulated over the `normal`
dataset of the validation split, from `loss_total.ckpt`.

**How it is accumulated.** Two running sufficient statistics, so that no per-event codes
are retained: a vector of bit sums, giving the per-bit activation rate
$\pi_j = \Pr(b_j = 1)$, and a counter over codewords packed into one byte with
`np.packbits`, giving the empirical codeword distribution $p(c)$ over at most
$2^8 = 256$ possible values.

$$H_{\text{marg}} = \sum_{j=1}^{8} H_b(\pi_j) \in [0, 8],
\qquad
H_{\text{joint}} = -\sum_{c} p(c)\log_2 p(c) \in \left[0,\ \log_2\min(256, N)\right]$$

**Why the joint entropy is the gate and the marginal one is not.** By subadditivity
$H_{\text{joint}} \le H_{\text{marg}}$, with equality only if the eight bits are
independent, and the gap between them can be the whole scale. Consider a code in which
all eight bits are perfect copies of one another, each firing half the time. Every
marginal is maximal, so $H_{\text{marg}} = 8$ bits — the code looks perfectly healthy
bit by bit. But the code takes only two values, so $H_{\text{joint}} = 1$ bit: the
bottleneck is a single coin flip wearing eight hats. Only the joint entropy detects this,
and it is exactly the failure mode an MI penalty could plausibly induce, since the penalty
acts on the bits individually (section 5.2) and is indifferent to whether they duplicate
each other.

**The two derived counts.** The perplexity

$$n_{\text{effective}} = 2^{H_{\text{joint}}}$$

is the number of *equally likely* codewords that would produce the same entropy, which is
easier to judge than a figure in bits: "4.3 effective codes out of 256" says immediately
what "2.1 bits" does not. Alongside it, $n_{\text{observed}} = |\{c : p(c) > 0\}|$ counts
codewords used at least once. Since a uniform distribution maximises entropy for a given
support, $n_{\text{effective}} \le n_{\text{observed}}$ always, and the ratio measures how
unequally the used codes are used — 200 codes observed with an effective count of 3 means
197 of them are essentially never seen.

**The thresholds.**

$$H_{\text{joint}}^{(k)} \ge 1.0\ \text{bit}
\qquad\text{and}\qquad
H_{\text{joint}}^{(k)} \ge 0.5\,H_{\text{joint}}^{(k),\,\gamma=0}$$

The absolute floor of 1.0 bit is the weakest possible non-degenerate code — two effective
states. The relative condition exists because architectures differ in how much of the code
space they naturally use, and the question the study asks is what the *penalty* cost, not
what the architecture achieves in absolute terms: a configuration that keeps half the
joint entropy of its own unregularised twin has not collapsed, whatever that twin's
absolute level was. Both are evaluated per seed, and the configuration-level rule requires
every seed to pass. If a baseline is itself below the absolute minimum, the baseline and
every candidate depending on it are rejected together, because the relative test would
otherwise be measured against a broken reference.

**Estimator bias.** $H_{\text{joint}}$ is the plug-in (maximum-likelihood) estimate from
empirical frequencies, which is biased low; the leading Miller–Madow correction is
$(K-1)/(2N\ln 2)$ bits for $K$ observed codewords. With $K \le 256$ and $N$ of order
$10^6$ this is below $10^{-4}$ bits, four orders of magnitude under the 1.0-bit threshold,
so no correction is applied.

### 5.7 AUROC and partial AUROC (diagnostic only)

**The construction.** Sweeping a threshold $t$ over the anomaly score gives, for each
signal sample, a true-positive rate $\text{TPR}(t)$ — the fraction of signal events scored
above $t$ — and a false-positive rate $\text{FPR}(t)$ over the normal sample. The ROC
curve is $\text{TPR}$ as a function of $\text{FPR}$, and

$$\text{AUROC} = \int_0^1 \text{TPR}(u)\,du
= \Pr\!\left(a_{\text{signal}} > a_{\text{background}}\right)$$

the probability that a randomly chosen signal event outranks a randomly chosen background
event. It is threshold-free and independent of class prevalence, which is what makes it
the conventional summary.

**Why it is the wrong summary here.** The trigger runs at a single, extremely tight
operating point: $\text{FPR} = 8.74\times10^{-6}$. AUROC integrates uniformly over the
entire $[0,1]$ range of FPR, so more than 99.999% of its support lies at operating points
the trigger will never occupy. Two detectors with identical AUROC can differ by orders of
magnitude in the only region that matters. The measured values make the point: median
AUROC across this grid spans 0.9825 to 0.9855 — a range of 0.003 — while the quantity the
trigger actually experiences, $\epsilon_{\text{med}}$, varies by 9% across the same
configurations.

**Partial AUROC.** The integral is therefore restricted to the operating region and
rescaled:

$$\text{pAUC}_{\text{raw}} = \int_0^{m} \text{TPR}(u)\,du,
\qquad
\text{pAUC} = \frac{\text{pAUC}_{\text{raw}}}{m},
\qquad m = \frac{r_{\text{target}}}{r_{\text{base}}}$$

computed by the trapezoid rule on the full ROC (`drop_intermediate=False`), with
$\text{TPR}$ linearly interpolated exactly at $u = m$ so that the endpoint is not a step
artefact of wherever the last ROC vertex happened to fall.

The normalised quantity is the **mean TPR over all operating points at least as strict as
the nominal one**. That is why it tracks the efficiency objective so closely:
$\epsilon_{\text{med}}$ is $\text{TPR}$ evaluated exactly at $u = m$, while pAUC is its
average over $[0, m]$. The measured ratio bears this out — median pAUC is 0.00271 against
$\epsilon_{\text{med}} = 0.00526$, a ratio of 0.51, which is what one expects if TPR rises
approximately linearly from zero across the interval.

**A documentation error worth correcting.** `configs/pareto_study/fet_et.yaml` records
`random_ranking_reference: 0.5`. That is wrong for the normalisation actually implemented.
Under random ranking $\text{TPR}(u) = u$, so
$\text{pAUC}_{\text{raw}} = m^2/2$ and $\text{pAUC} = m/2 = 4.37\times10^{-6}$ — not 0.5.
The value 0.5 would be correct only under the McClish standardisation
$\tfrac{1}{2}\left[1 + \frac{A - A_{\min}}{A_{\max} - A_{\min}}\right]$, which is not what
the code computes. The implemented number is well defined and interpretable; the
annotation should be changed to $m/2$ so that nobody compares 0.0027 against 0.5 and
concludes the detector is worse than random.

**Both are summarised per signal by minimum and median**, and one of those four numbers is
dead: `min_partial_auroc` is **identically zero for all 48 configurations**. At least one
of the 21 signal samples has no events at all above threshold anywhere in $[0, m]$, so the
minimum is pinned at the floor and carries no information. It should either be dropped or
replaced by a statistic with resolution in this regime.

The `min_auroc` values are more informative and worth reading alongside: they span 0.541
to 0.583 while the median AUROC is 0.985. The hardest of the 21 signals is barely better
than a coin flip, which is the context in which $\epsilon_{\min}$ was made a feasibility
gate rather than an objective — the question there is whether a configuration abandoned a
signal class outright, not how far it moved a number that is already near chance.

### 5.8 Seed aggregation

Each metric $m$ is computed per run. For a configuration with seeds $k = 1..n$ (here
$n = 2$):

$$\bar{m} = \frac{1}{n}\sum_k m_k, \qquad
\sigma = \sqrt{\frac{1}{n-1}\sum_k (m_k - \bar{m})^2}, \qquad
\mathrm{SE} = \frac{\sigma}{\sqrt{n}}$$

$$\text{CI}_{95} = \bar{m} \pm t_{0.975,\;n-1}\,\mathrm{SE}$$

Student's $t$ rather than 1.96, because with two paired seeds there is one degree of
freedom and $t_{0.975,1} = 12.71$; the normal quantile understates the interval by a
factor of 6.5. Aggregation refuses to proceed with fewer than two finite seed values.

At $n = 2$ these collapse to simple forms worth knowing when reading the figures:
$\sigma = |m_1 - m_2|/\sqrt{2}$ and $\mathrm{SE} = |m_1 - m_2|/2$, so the standard error
is exactly half the gap between the two seeds and $\bar{m} \pm \mathrm{SE}$ lands exactly
on them. The figures draw $\bar{m} \pm \sigma$ and label it as the seed spread, not as a
confidence interval — two points cannot support an inferential claim.

## 6. Hyperparameters

Everything below is frozen across the grid except the three searched axes and the seed.
The authoritative source is each run's `resolved_config.yaml`, written beside its
checkpoint before training; the values here are that configuration.

### 6.1 Autoencoder

| parameter | value | where |
|---|---|---|
| model input | 116 features | `FET.Et` removed; control tensor retains 117 |
| encoder nodes | `[64,32,8]` / `[128,64,8]` / `[64,64,32,8]` | searched |
| decoder nodes | mirror of the encoder, out_dim 116 | derived |
| activation | ReLU | encoder and decoder |
| latent width | 8, frozen | `encoder_nodes[-1] == 8` |
| Bernoulli sampling | threshold 0.5, 10 samples, std 1.0 | between logits and decoder |
| reconstruction loss | Huber, $\delta = 3.0$ | per-event, then mean |
| anomaly score | per-event MSE reconstruction error | separate from the training loss |
| input noise | $\sigma = 10^{-4}$, training only, masked | denoising regulariser |
| optimizer | AdamW | |
| learning rate | $1.3029941778 \times 10^{-3}$ | constant, no scheduler |
| weight decay | $10^{-3}$ | |
| Adam $\beta$, $\epsilon$ | $(0.9,\ 0.999)$, $10^{-8}$ | |
| gradient clipping | norm, value 0.0 | i.e. disabled |
| batch size | 16384 | 765 batches per epoch |
| epochs | 30 | capped from the manifest's 200 |
| validation batches | 10 | `max_val_batches` |
| accelerator | CPU | |
| normaliser | robust | fitted on the training split |
| checkpoint | monitor `val/loss_total`, mode min, top-1 | saved as `loss_total.ckpt` |
| seeds | 123, 500 | both required per configuration |

The reconstruction loss and the anomaly score are deliberately different functions. Huber
with $\delta = 3$ trains the decoder without letting a few extreme events dominate the
gradient; the score is plain MSE, because at scoring time a large residual *is* the
signal and should not be down-weighted.

The learning rate is carried over from an earlier Optuna search and is stated to that
precision because that is the value in the config, not because the digits are meaningful.

### 6.2 Operating point

| parameter | value |
|---|---|
| target rate $r_{\text{target}}$ | 0.25 kHz |
| base rate $r_{\text{base}}$ | 28608.8064 kHz (bunch-crossing rate) |
| background FPR | $r_{\text{target}}/r_{\text{base}} = 8.739\times10^{-6}$ |
| threshold quantile | $1 - \text{FPR} = 0.99999126$, `interpolation="higher"` |

This is an extremely tight operating point — roughly one background event in 114,000 — and
it is the reason $\epsilon_{\text{med}}$ sits near 0.005 rather than near 1. Efficiencies
of that scale are expected here and are not a sign of a broken detector.

### 6.3 Mutual-information penalty

| parameter | value | note |
|---|---|---|
| $\gamma$ | 0, 0.05, 0.10, 0.15, 0.20, 0.25 | searched |
| requested bins $B$ | 40, 50, 60 | searched; $B_{\text{eff}} \le B$ |
| sensitive variable | `FET.Et` | excluded from the model input |
| sigmoid temperature $T$ | 6.0 | |
| entropy dtype | float64 | float32 on MPS only |
| log guard $\varepsilon$ | $10^{-20}$ | inside $\log_2$ |
| quantised sigmoid | off (8 bits if enabled) | hardware-emulation path, unused |
| denormalised target | false | binning on the normalised value |
| reduction | `first` | one scalar per event |
| input-leak guard | `forbid_sensitive_variable_in_input: true` | asserted at fit start |

### 6.4 Leakage probes

| parameter | value |
|---|---|
| development pool | train split, 12,532,307 events, uncapped |
| held-out pool | valid split, 4,177,435 events |
| representations probed | $z$ (8 features), $\hat{x}$ (116 features) |
| regression target | raw `FET.Et` in GeV, denormalised |
| frozen probe seed | 123 (no seed search, no inner split) |
| MLP hidden layers | (64, 32), ReLU |
| MLP solver | Adam, $\text{lr} = 10^{-3}$ constant |
| MLP L2 $\alpha$ | $10^{-4}$ |
| MLP batch size | 16384 (v9; previously scikit-learn's default of 200) |
| MLP max epochs | 500 |
| MLP early stopping | on, 10% internal split, $\text{tol}=10^{-4}$, patience 10 |
| MLP Adam $\beta$, $\epsilon$ | $(0.9,\ 0.999)$, $10^{-8}$ |
| feature scaling | `StandardScaler` fitted on the development pool, one per probe |
| target scaling | standardised for the MLP; raw GeV for the linear probe |
| linear solver | float64 blocked Householder QR (65,536-row blocks) then SVD |
| linear rank rule | dimension-aware truncation on the development matrix |
| streaming chunk | 65,536 rows |
| shuffled-target controls | disabled (guardrail would be $R^{2,\text{clip}} \le 0.02$) |

Each of the four probes owns its own scaler and estimator; the aggregator asserts that the
four scaler objects and the four estimators are distinct instances, so no state is shared
between them.

### 6.5 Aggregation and selection

| parameter | value |
|---|---|
| seeds per configuration | 2, both required |
| interval | $t_{0.975,\,n-1}$, i.e. 12.71 at $n=2$ |
| collapse gate | $H_{\text{joint}} \ge 1.0$ bit and $\ge 0.5 \times$ baseline |
| efficiency gate | $\epsilon_{\min} \ge 0.95 \times$ baseline |
| seed pass fraction required | 1.0 |
| ranking weights | $(1/3,\ 1/3,\ 1/3)$ |
| tie-break | `configuration_id` ascending |

## 7. Feasibility gates

Two conditions are gates rather than objectives. A configuration failing either is kept
in the results, marked `infeasible`, and never competes. Both are measured **relative to
the configuration's own $\gamma = 0$ twin at the same architecture and seed**, and both
require every expected seed to pass — one bad seed rejects the configuration.

$$H_{\text{joint}}^{(k)} \ge 1.0\ \text{bit}
\quad\text{and}\quad
H_{\text{joint}}^{(k)} \ge 0.5\,H_{\text{joint}}^{(k),\,\gamma=0}$$

$$\epsilon_{\min}^{(k)} \ge 0.95\,\epsilon_{\min}^{(k),\,\gamma=0}$$

Relative rather than absolute thresholds because the quantity of interest is what the
penalty *cost*, not what the architecture happens to achieve. The consequence is worth
stating plainly: the $\gamma = 0$ baselines are load-bearing, and if a baseline is
anomalous every configuration measured against it inherits that.

## 8. Selection

Selection reads only the aggregated configuration-level table. It never reopens a run
artifact and never recomputes a metric, which keeps validation-set selection structurally
separate from training and from any later test evaluation.

Configuration $a$ **dominates** $b$ when

$$L_a \le L_b \ \wedge\ E_a \le E_b \ \wedge\ \epsilon_a \ge \epsilon_b$$

with at least one inequality strict. The non-dominated set is the Pareto front; exact
ties deliberately remain on it.

The front has no internal order. To name a single recommendation anyway, each member is
given a cost vector $(L,\ E,\ 1 - \epsilon_{\text{med}})$ — all three then "lower is
better", all three in $[0,1]$ — and ranked by distance to the ideal point:

$$D = \sqrt{\frac{L^2 + E^2 + (1-\epsilon_{\text{med}})^2}{3}}$$

with ties broken on `configuration_id` ascending. **This ranking is a convention, not a
result.** Equal weights assume the three objectives are equally important *and*
commensurable on their raw scales; in practice whichever axis spans the widest numerical
range dominates $D$.

## 9. The pipeline

Four stages, each independently resubmittable, because a 96-run grid will not survive
being one job. Stage 1 trains one autoencoder and runs the ordinary evaluation, writing
`loss_total.ckpt`, the evaluation artifacts and `run_manifest.yaml`. Stage 2 fits the four
leakage probes against that frozen checkpoint. Stage 3 exists as a repair tool for the
remaining metrics; in this study they were produced inside stage 1. Stage 4 collects
everything, applies the gates, builds the front and draws the figures.

Stage 4 writes `phase2/` (per-configuration `pareto_metrics.json` with per-seed metrics
under `runs[]`, plus flat `pareto_metrics.csv`/`.parquet`), `phase3/`
(`pareto_candidates.csv`, `pareto_front.csv`, `pareto_selection.json`) and `phase4/` (the
figures). Note that "phase 2" and "phase 3" here are sub-steps of stage 4 and are
unrelated to pipeline stages 2 and 3 — an unfortunate collision worth knowing before
reading the tree.

## 10. Results

All 96 runs trained and probed successfully. Of the 48 configurations, 8 are infeasible
(5 failing on seed 123, 3 on seed 500, all through the paired gates), 28 are dominated
and 12 are on the front. The ranking names
`fet-et-pareto-v1__gamma-0.1__bins-50__arch-h64_64_32`, and `selection_uncertain` is
`true`: 9 of the other 11 front members overlap it on all three objectives.

Mean objective values by penalty strength over eligible configurations:

| $\gamma$ | n | $L$ | $E$ | $\epsilon_{\text{med}}$ | median AUROC |
|---|---|---|---|---|---|
| 0.00 | 3 | 0.11357 | 0.10914 | 0.00500 | 0.98345 |
| 0.05 | 7 | 0.11027 | 0.10049 | 0.00519 | 0.98444 |
| 0.10 | 8 | 0.10981 | 0.10147 | 0.00521 | 0.98459 |
| 0.15 | 8 | 0.10927 | 0.09906 | 0.00524 | 0.98444 |
| 0.20 | 7 | 0.10598 | 0.11198 | 0.00524 | 0.98470 |
| 0.25 | 7 | 0.10840 | 0.11651 | 0.00524 | 0.98462 |

**The headline is a null result and should be read before anything else here is used.**

Turning the penalty from off to maximum moves $L$ from 0.1136 to 0.1084 — a reduction of
0.0052, or 4.6% relative — and the trend is not monotone, since $\gamma = 0.20$ beats
$\gamma = 0.25$. Against that, the median difference in $L$ between the **two seeds of the
same configuration** is 0.00516. One seed's worth of noise is the size of the entire
effect. Rank correlation with $\gamma$ is only $-0.44$.

The other two objectives move in directions inconsistent with a trade-off:
$\rho(\gamma, E) = +0.32$ although the penalty exists to reduce that dependence, and
$\rho(\gamma, \epsilon_{\text{med}}) = +0.51$, so no cost is visibly being paid for the
regularisation. The $\gamma = 0$ baselines are simultaneously the worst configurations on
leakage *and* on efficiency, which is not what a trade-off looks like.

Three readings are consistent with this, and all are cheaper to test than to argue.
Either the grid's $\gamma$ range is one to two orders of magnitude too small for the MI
term to contribute meaningfully to the loss — readable directly from the stage-1 loss
decomposition; or the factorised estimator of section 5.2 is decorrelating each bit
marginally while leaving joint information intact, which the probes then find; or an
8-bit latent trained on inputs with `FET.Et` already removed simply carries about
$R^2 = 0.11$ of indirect leakage that this penalty cannot reach. The third would be a real
and publishable finding; it is not the finding the study was shaped to report. Extending
$\gamma$ to 0.5, 1, 2, 5 at one architecture and one binning with two seeds is eight
training runs and distinguishes the first from the rest.

## 11. What was deliberately not used

**AUROC and partial AUROC** carry `role: diagnostic_only`. They are not objectives
because a trigger does not operate at a threshold chosen to optimise the whole ROC curve;
it operates at a fixed rate, where $\epsilon_{\text{med}}$ measures performance. AUROC
also barely moves across the grid (0.9825 to 0.9855), so promoting it would add an axis
with no resolving power. It is retained because a collapse in AUROC would indicate
something broken upstream.

**Shuffled-target negative controls** exist in the probe code and are disabled
(`run_shuffled_target_controls: false`). They refit the probes against a permuted target
to confirm a non-zero $L$ is not an artefact of the fitting procedure, and roughly double
stage 2's cost. Given that the conclusion now rests on $L$ being *flat* rather than large,
this is the omission I would most want back: a control run would establish whether
$L \approx 0.11$ is real signal or a floor of the probe protocol.

**The alternative efficiency statistics** $\epsilon_{\min}$, $\epsilon_{\text{mean}}$ and
$\epsilon_{\text{CVaR25}}$ are aggregated and stored but only the median is an objective.
$\epsilon_{\min}$ is not decorative — it is the quantity the minimum-efficiency gate acts
on. The others are the evidence for *why* a configuration was gated, without which one
would have to reopen the per-run JSONs.

**Capping the probe sample** is implemented (`max_samples`) and unused: the probes ran on
the full 12.5M-event training split. The cap is deliberately wired as a smoke-test switch
that marks its output non-reportable, so using it for real results would require a
protocol change. It remains the largest available saving — the fits are dominated by
sample count and the standard error on $R^2$ at 4M events is already about $5\times10^{-4}$
— but it changes what the protocol means.

**`latent_sample`** is extracted and retained alongside the logits but no probe uses it;
the probes read $z$ and $\hat{x}$. It is kept for the collapse diagnostics of section 5.6
and for the extraction-time determinism check.

**The `fet_et_adhoc` config variant** carries the identical policy with the candidate
interpolated *from* the algorithm config rather than declared literally, so a one-off AE
run can be collected into the study later. The grid uses the literal variant, because
having both interpolation directions active at once is a cycle Hydra cannot resolve.

**`data.load_aux_in_fit`** was added to skip the auxiliary signal datasets during
training, cutting stage 1 from 15.3 GB to 10.3 GB, and was reverted. The fit-time
validation loop computes the operational threshold of section 5.5 and stores it in the
checkpoint; without it the evaluator raises "Threshold has not been set", and a checkpoint
trained that way carries no threshold at all, so every later stage inherits a failure that
cannot be repaired without retraining.

**Three seeds.** The protocol requires two. Two is the minimum that permits any dispersion
estimate at all, and a weak one.

**Probe design options considered and rejected.** Sharing one scaled feature matrix
between the MLP and linear probes of a representation would have saved 5.8 GB and was
rejected to keep the four probes procedurally independent. Probe parallelism was rejected
because it raises peak memory, the constraint being solved, and because once the fits
became BLAS-bound the cores were already busy. Solving the linear probe by normal
equations was tested and rejected on numerical grounds: the Gram approach disagreed with
the reference solution by $8\times10^{-4}$ in $R^2$ — the size of the effects this study
must resolve — because squaring the condition number is fatal on decoder-generated
features. Blocked Householder QR reproduces the reference to $10^{-12}$ at negligible
memory.

## 12. Known weaknesses

These are stated so the results can be weighed, not because they are believed to change
section 10.

**The confidence intervals in the current artifacts are too narrow.** `_summary()` used
1.96 standard errors; the correct multiplier at $n = 2$ is $t_{0.975,1} = 12.71$, so every
stored interval understates by 6.5. This is fixed in code but the tables in this study
predate the fix, and `pareto_selection.json` therefore already reports
`selection_uncertain: true` on intervals that are far too optimistic. Regenerating stage 4
corrects it; the figures are unaffected, since they draw $\bar{m} \pm \sigma$.

**Dominance ignores uncertainty.** The front is computed on seed means, so a
configuration can be marked `dominated` by a rival well inside its spread. Much of the
28/12 split is therefore not statistically meaningful.

**Requested bins are not effective bins** (section 5.1), and the tables record the
requested value, so two nominally distinct binning levels may be closer together than the
grid suggests.

**The ideal-point ranking uses equal weights on incommensurable scales.** $L$ spans about
0.013 across the grid while $1 - \epsilon_{\text{med}}$ spans about 0.0004, so $D$ is
driven by whichever axis is numerically widest. The rank-1 configuration should not be
reported as "the best" without decomposing its distance.

**`E` is named for residuals but computed on the reconstruction** (section 5.4).

**No `dominated_by` record.** A dominated configuration carries no note of which
configuration beat it, so the front cannot be audited row by row.

**A probe hyperparameter was wrong until v9.** scikit-learn defaults `batch_size` to
$\min(200, N)$; on a 12.5M-event pool that is 62,661 optimizer steps per epoch on
$200 \times 116$ matmuls, using one of seven allocated cores for 108 minutes. It is now
16384, matching the autoencoder's own batch size. Any $L$ produced before v9 is not
comparable with these results.

## 13. Reproducing

```bash
condor_submit batch/runae_pareto.sub      # stage 1: 96 training runs
condor_submit batch/runprobes_pareto.sub  # stage 2: four probes per run
condor_submit batch/runcollect.sub        # stage 4: collect, select, draw
```

Stage 4 alone re-derives everything downstream of the checkpoints in minutes, and the
figures alone can be redrawn without re-collecting:

```bash
python3 scripts/plot_pareto_study.py \
  --candidates <study>/phase3/pareto_candidates.csv \
  --front      <study>/phase3/pareto_front.csv \
  --output-dir <study>/phase4
```

## 14. The figures

`pareto_L_vs_efficiency_by_architecture.png` and
`pareto_L_vs_correlation_by_architecture.png` facet by architecture, colour by $\gamma$ on
a validated five-step ordinal ramp, size by requested bins, ring the front members and bar
their seed spread. The $\gamma = 0$ baselines are open red squares, set apart from the
ramp because they are the reference the gates are measured against rather than a weak
penalty.

`pareto_L_vs_efficiency_coloured_by_E.png` puts all three objectives in one panel with $E$
on the colourbar. This is the more honest figure for reading the front: dominance is
decided in three dimensions, so a ringed point can sit inside the cloud of a
two-dimensional projection purely by winning on the axis that projection dropped. A
`_plain` variant without rings or bars, and three `_size_*` variants encoding $\gamma$,
bins and architecture as area, are provided for exploration.

`pareto_front_parallel_coordinates.png` shows the front across all three objectives, one
polyline per member, each axis oriented so up is better.

Architecture is a nominal category and area implies a magnitude ordering it does not have,
so the architecture-sized variant is for exploration rather than publication; the faceted
figures are the honest presentation of that axis.
