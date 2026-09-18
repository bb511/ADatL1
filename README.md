[![Email Badge](https://img.shields.io/badge/blah-podagiu%40ethz.ch-blue?style=flat-square&logo=minutemailer&logoColor=white&label=%20&labelColor=grey)](mailto:podagiu@ethz.ch)
[![Python: version](https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Anomaly Detection @ Trigger

## Setup

This repository uses [poetry](https://python-poetry.org/) for package management.
We recommend setting this up using poetry.
However, if you do not want to use poetry, skip to [here](#setup-without-poetry).

Install the dependencies using poetry by running the following command in the repository root:
```
poetry install --no-root
```

To install the dependencies required by the quantisation packages:
```
poetry install --extras quant --no-root
```

## Setup without Poetry

To install the dependencies using `pip`, use
```
pip install -r requirements.txt
```

## Project layout

Run this once. It creates `data/ logs/ outputs/ checkpoints/` and writes the `.env` that
`configs/paths/default.yaml` reads through `${oc.env:PROJECT_ROOT}`. Nothing composes
without it. Re-running never overwrites an existing `.env`.
```
bash scripts/setup.sh
```
To keep those directories on another filesystem, edit `RES_DIR` in `.env` and then run
`bash scripts/symbolink.sh`, which replaces them with symlinks. That step **deletes** any
real directory in the way, so run it before the first training, not after; it asks for
confirmation unless given `--force`.

## Data

### Outputs on EOS / CERNBox

Set these variables inside the LXPLUS container before running Python:

```bash
export PROJECT_ROOT=/eos/user/l/lbehrens/adl1t-stage
export ADL1T_OUTPUT_ROOT=/eos/user/l/lbehrens/adl1t-stage/data/run-outputs
```

Input data remains under `$PROJECT_ROOT/data`. Checkpoints go under
`$ADL1T_OUTPUT_ROOT/checkpoints`, and MLflow and Hydra run outputs under
`$ADL1T_OUTPUT_ROOT/logs`. The raw-data path still needs its normal configuration.
Without `ADL1T_OUTPUT_ROOT`, the existing output locations are preserved.
`python test.py` writes `hello.txt` into this output directory to check EOS writes
and CERNBox synchronization. These settings apply to training outputs using the
central paths configuration; standalone analysis scripts may specify their own paths.

The LHC L1 AD data is produced by [this code](https://github.com/bb511/adl1t_datamaker).
See more details about it there.
We recommend using the HuggingFace variant [`podagiu/anomaly_detection_cmsl1t`](https://huggingface.co/datasets/podagiu/anomaly_detection_cmsl1t) of the dataloader of the LHC L1 AD data.

## Licence

The code is MIT (see `LICENSE`). The dataset is released separately under CC0 1.0.

## Usage

Training runs from `src/train.py`, selected by an experiment config:
```
python src/train.py \
    experiment=physics/ae \
    paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
    trainer=gpu trainer.devices=[0]
```
Add `data=basis_hf` to read the published record instead, in which case no data path is needed.
Domains are `physics/`, `cifar10/` and `robustad/`; `*_agnostic` variants validate with the
label-free objectives instead of signal efficiency. Hyperparameter searches use
`--multirun hparams_search=physics/ae_optuna`.

The exact commands behind every result in the paper are in [`scripts/`](scripts/README.md) —
one catalogue of commented, copy-pasteable `src/train.py` invocations per model and domain,
plus the tooling that generated them, submitted them to slurm, and harvested the results.
The experiment configs already carry the hyperparameter values reported in the paper.

## The four-stage pipeline

For the Bernoulli-MI autoencoder and the FET.Et Pareto-front study, training and
analysis are separate processes. This lets every autoencoder be trained before
anything is analysed, and lets each step be scheduled with the resources it
actually needs — the leakage probes alone cost a measured ~27 minutes per run,
independent of epoch count, and used to be paid inside every training job.

| stage | entrypoint | local script | HTCondor | writes |
|---|---|---|---|---|
| 1 train | `src/train.py` | `scripts/physics/runae.sh` | `batch/runae.sub` | `loss_total.ckpt` and the usual AE plots |
| 2 probes | `src/run_probes.py` | `scripts/physics/runprobes.sh` | `batch/runprobes.sub` | `probes/leakage_probes.json` (objective L) |
| 3 metrics | `src/run_eval_metrics.py` | `scripts/physics/runmetrics.sh` | `batch/runmetrics.sub` | `eff/`, `correlation_matrix/`, `latent_collapse/`, `auroc/` summaries |
| 4 front | `scripts/collect_pareto_study.py` + `scripts/select_pareto_front.py` | `scripts/physics/runcollect.sh` | `batch/runcollect.sub` | `phase2/`, `phase3/`, `pareto_front.csv` |

Stages 2 and 3 are independent of each other: both read only the checkpoint and
both write into disjoint subdirectories of the run folder, so they may run
concurrently. Stage 4 is a whole-study step and takes no run name.

### Running one autoencoder

```bash
export RUN_NAME=AE_30ep_gamma0.1
bash scripts/physics/runae.sh        # stage 1  -> loss_total.ckpt
bash scripts/physics/runprobes.sh    # stage 2  (~27 min)
bash scripts/physics/runmetrics.sh   # stage 3
```

Every knob is an environment variable. The shared ones — `RUN_NAME`,
`EXPERIMENT`, `TRAINER`, `CPU_THREADS`, `RAW_DATA_DIR`, `ADL1T_OUTPUT_ROOT` and
the model hyperparameters — live in `scripts/physics/_stage_common.sh`, which all
four scripts source.

### The rule that matters

**Every stage must compose the same config as the stage-1 run it analyses.** The
checkpoint stores weights but not the config, and both the evaluator and the
probe loader call `load_state_dict(..., strict=True)`. A mismatched architecture
fails loudly rather than silently measuring the wrong model — but only after the
data has loaded, which costs minutes. Keeping the hyperparameters in
`_stage_common.sh` rather than repeating them per script is what prevents that
drift.

`ADL1T_OUTPUT_ROOT` needs particular care. `configs/paths/default.yaml` reads it
from the environment, not from an override, and it decides where
`checkpoints/<experiment_name>/<run_name>` lives. On the cluster, stage 1 writes
into the job sandbox and `transfer_output_files` brings the tree home, whereas
stages 2 and 3 must point it at the merged tree on EOS. That is the one setting
that differs between the stage-1 wrapper and the others.

### Stage 3 needs `physics/ae_metrics`

`configs/experiment/physics/ae_metrics.yaml` inherits `physics/ae` unchanged and
only switches on the four summary callbacks. The plain `physics/ae` experiment
leaves the correlation matrix disabled and defines no AUROC or latent-collapse
callback at all, so stage 3 composed with it would exit zero and write nothing.
Both experiments resolve to `experiment_name: physics_ae_models` and therefore
address the same checkpoint directory. For a Pareto-study run use the study's own
experiment instead: `EXPERIMENT=physics/pareto_fet`.

### On HTCondor

Stages 2 and 3 take one run per job, read from `batch/runs.txt`:

```bash
printf '%s\n' AE_LXPLUS_30ep AE_LXPLUS_30ep_gamma0 > batch/runs.txt
condor_submit batch/runprobes.sub     # both may be submitted together
condor_submit batch/runmetrics.sub
```

Neither submit file sets `transfer_output_files`, deliberately: these jobs write
their JSON straight into the existing checkpoint tree on EOS, so there is nothing
in the sandbox to bring home.

### The Pareto study runner

`scripts/physics/run_pareto_fet_ngt.sh` runs three processes per grid point.
`PARETO_STAGES` selects which, defaulting to `train,metrics,probes`. To train the
whole grid first and analyse it afterwards:

```bash
PARETO_STAGES=train          ./scripts/physics/run_pareto_fet_ngt.sh --run
PARETO_STAGES=metrics,probes ./scripts/physics/run_pareto_fet_ngt.sh --run --rerun-incomplete
```

A train-only pass leaves every run marked `incomplete_after_success`, which is
accurate rather than a bug — the analysis has not happened yet.
`--rerun-incomplete` is required for the second pass and is safe there, because
only the stage-1 training callback clears a checkpoint directory.

`src/run_probes.py` exits **3** when the probes ran to completion but the protocol
rejected the result. The runner treats that as incomplete rather than failed, so
`run_status.tsv` keeps the distinction between a broken job and a rejected
configuration. The invalid result is still written to disk.

### Configurations, runs, and the per-run manifest

A **configuration** is a point on the Pareto grid: `(mi_gamma,
mi_sensitive_num_bins, architecture)`. A **run** is one training of an
autoencoder at that configuration with a particular seed. Several runs share a
configuration when they differ only by seed, and that is what lets the
aggregation report a mean and a confidence interval.

At the end of stage 1 every run records itself:

```
checkpoints/<experiment_name>/<run_name>/
    loss_total.ckpt
    run_manifest.yaml        identity, grid point, algorithm fingerprint, mlflow run id
    resolved_config.yaml     the fully resolved config, travelling with the checkpoint
    stage_status/train.yaml
```

It is written after training rather than before, because `ClearRunCheckpointDir`
wipes the run directory when a fit starts — and because a manifest present is
then a truthful claim that stage 1 finished.

Stages 2 and 3 read it and refuse to run when `algorithm_fingerprint` does not
match the config they were given. That closes a gap `strict=True` on
`load_state_dict` cannot: it compares tensor shapes only, so a stage-2 run
composed with the wrong `mi_gamma` would otherwise load perfectly and record a
probe score against the wrong grid point. Pass `manifest_strict=false` for runs
trained before manifests existed; the stage then warns instead of stopping.

Each of stages 2 and 3 writes its own `stage_status/<stage>.yaml` rather than
updating the shared manifest, because they may run concurrently and a
read-modify-write from two processes on EOS loses one of the updates.

### Stage 4 over an experiment directory

Stage 4 consumes a study map listing every configuration and seed. There are two
ways to get one.

The study runner declares the whole grid up front, at
`STUDY_ROOT/study_map.yaml`. If that file exists it is used as is.

Otherwise the map is **built from an experiment directory**, using the
`run_manifest.yaml` each run carries:

```bash
EXPERIMENT_NAME=physics_ae_models bash scripts/physics/runcollect.sh
```

That is the path for autoencoders trained one at a time, whenever and wherever
there was capacity, and collected afterwards. Runs pair into configurations by
`configuration_id`, which excludes the seed, so two runs differing only in seed
aggregate together automatically. The builder prints which configurations are
missing an expected seed — those are rejected by Phase 2 rather than silently
dropped, so that list is the to-do list of runs still to train.

`scripts/build_study_map.py` can also be run on its own. The collector itself is
untouched: it still validates every claim in the map against each run's resolved
manifest and artifacts.

For this to work an ordinary AE run has to carry the study's identity and policy,
which is why `configs/pareto_study/` exists. `fet_et.yaml` holds the study — its
id, protocol versions, search space, constraints and selection policy — and both
`physics/pareto_fet` and `physics/ae` select it. They select different variants
only because of the direction of one link: the study sets
`algorithm.mi_gamma: ${pareto_study.candidate.mi_gamma}`, while an ad-hoc run
needs `candidate.mi_gamma: ${algorithm.mi_gamma}`, and having both at once is an
interpolation cycle. `fet_et_adhoc.yaml` is that second variant and changes
nothing else, so a run trained through `runae.sh` is judged by exactly the same
rules as a grid run.
