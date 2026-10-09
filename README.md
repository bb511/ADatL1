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

## The FET.Et Pareto study

The Bernoulli-MI autoencoder is scanned over `mi_gamma` (strength of the MI
penalty) and `mi_sensitive_num_bins` (binning of the sensitive variable FET.Et).
The study selects the configurations on the Pareto front of leakage L, residual
correlation E and signal efficiency.

### Versions

| | value | defined in |
|---|---|---|
| study id | `fet-et-pareto-v1` | `configs/pareto_study/fet_et.yaml` (`study_id`, prefix of every `configuration_id`) |
| selection protocol | `fet-et-pareto-v3` | `configs/pareto_study/fet_et.yaml` (`protocol_version`, with its history); tag `protocol-v3` in `configs/experiment/physics/pareto_fet.yaml` |
| leakage-probe protocol | `fet-et-four-probe-v10` | `leakage_probe_protocol_version`, `src/evaluation/leakage_probe/constants.py`, [`docs/evaluation/leakage_probe_contract.md`](docs/evaluation/leakage_probe_contract.md) |

Every run records these in its manifest, and stage 3 refuses to put runs of
different protocol versions into one study. `Pareto-Front-260928` and
`Pareto-Front-261002` were trained under v2 and can still be collected on their
own; **v3 runs need a new experiment name** (`EXPERIMENT_NAME`, below).

### The three stages

| stage | what | entrypoint | local script | HTCondor | writes |
|---|---|---|---|---|---|
| 1 | train + validation evaluation | `src/train.py` | `scripts/physics/runae.sh` | `batch/runae_pareto.sub` | `loss_total.ckpt`, `run_manifest.yaml`, and under `plots/val/loss_total/`: `eff/`, `correlation_matrix/` (E), `latent_collapse/`, `auroc/` and the usual AE plots |
| 2 | leakage probes | `src/run_probes.py` | `scripts/physics/runae_pareto_runprobes.sh` | `batch/runprobes_pareto.sub` | `plots/val/loss_total/probes/leakage_probes.json` (L) |
| 3 | collect + select | `src/evaluation/pareto/` (`aggregation`, `selection`, `plots`, `matrices`, `correlation_gamma0`) | `scripts/physics/runae_pareto_runcollect.sh` | `batch/runcollect.sub` | `pareto_studies/<experiment>/phase2/`, `phase3/` (`pareto_front.csv`, `pareto_selection.json`), `phase4/` figures |

- Stages 1 and 2 run once per grid point and stage 3 once per study. Stage 2 needs
  only the checkpoint of stage 1; stage 3 needs stages 1 and 2 of every run.
- Inside stage 3 the steps are called phases (`phase2/` collect, `phase3/`
  select, `phase4/` figures, 4b γ × bins matrices, 4c correlations against the
  γ = 0 run); they are not pipeline stages.
- The grid lives only in `configs/pareto_study/fet_et.yaml`.
  `scripts/physics/runae_pareto_makegrid.sh` turns it into the job list
  `batch/pareto_runs.txt` (`SEED,GAMMA,BINS,ARCH,NODES,RUN_NAME`) that stages 1
  and 2 read on HTCondor. The code of the study is in `src/evaluation/pareto/`,
  one module per phase; each runs on its own as
  `python3 -m src.evaluation.pareto.<module> --help` from the repository root.
  The metric-vs-γ and metric-vs-bins sweeps (`mi_changes`) are not part of
  stage 3 and are only run by hand.
- After the study, for the selected configuration only (and its γ = 0 run):
  `scripts/physics/runae_test.sh` evaluates `loss_total.ckpt` on the test split,
  `scripts/physics/runae_test_comparison.sh` compares those test outputs with the
  γ = 0 run. Test results never feed back into the selection.
- `src/run_eval_metrics.py` re-runs the validation evaluation of a checkpoint
  without retraining, e.g. if the post-fit evaluation of stage 1 crashed (compose
  the same experiment and `run_name` as stage 1).

### Running the stages locally

Needs the staged data under `$PROJECT_ROOT/data/data_2025E+G/{extracted,processed,mlready}`
(default `PROJECT_ROOT` is the checkout; run `bash scripts/setup.sh` once). Every
setting is an environment variable read by `scripts/physics/_stage_common.sh`.
One grid point:

```bash
export EXPERIMENT_NAME=Pareto-Front-local            # checkpoints/<experiment>/, one per study
export PARETO_CANDIDATE=1                            # parameterise through pareto_study.candidate
export SEED=180524 MI_GAMMA=0.1 MI_NUM_BINS=40 ARCHITECTURE_ID=h64_32 ENCODER_NODES='[64,32,8]'
export RUN_NAME=Seed180524_Gamma_0.1_Bins_40_architecture_h64_32_Run01

EXPERIMENT=physics/pareto_fet_train bash scripts/physics/runae.sh            # stage 1 (MAX_EPOCHS=2 for a smoke run)
EXPERIMENT=physics/pareto_fet bash scripts/physics/runae_pareto_runprobes.sh # stage 2, ~27 min
```

Repeat for every grid point, with the run names `runae_pareto_makegrid.sh`
writes, then once for the whole experiment:

```bash
EXPERIMENT_NAME=Pareto-Front-local bash scripts/physics/runae_pareto_runcollect.sh   # stage 3
```

Stage 3 is pandas only (no torch, no data) and takes minutes. Its outputs land
in `pareto_studies/Pareto-Front-local/`.

### Running the stages on lxplus (HTCondor)

The jobs run in the container `/eos/user/l/lbehrens/containers/adl1t_lab-dev.sif`
from the EOS checkout `/eos/user/l/lbehrens/adatl1/ADatL1`, read the data in
`/eos/user/l/lbehrens/adl1t-stage` and write to
`/eos/user/l/lbehrens/adatl1/ADatL1/outputs/` (`checkpoints/`, `logs/mlflow/`,
`pareto_studies/`).

```bash
# on the laptop: job list from configs/pareto_study/fet_et.yaml (needs omegaconf,
# which a bare lxplus shell does not have), then commit and push
bash scripts/physics/runae_pareto_makegrid.sh
git add batch/pareto_runs.txt && git commit -m "pareto: job list" && git push

# on lxplus
cd /eos/user/l/lbehrens/adatl1/ADatL1
git pull                           # never while a job is using this checkout
module load lxbatch/eossubmit      # every new shell; standard schedds reject /eos paths
kinit                              # a job must finish, output transfer included, within ~24 h
mkdir -p batch/logs

# stage 1: one job per run; outputs come back to outputs/ on EOS when each job ends
condor_submit EXPERIMENT_NAME=Pareto-Front-<date> batch/runae_pareto.sub

# stage 2: after every stage-1 job has finished and its outputs are on EOS
condor_submit EXPERIMENT_NAME=Pareto-Front-<date> batch/runprobes_pareto.sub

# stage 3: once, after every stage-2 job has finished
condor_submit EXPERIMENT_NAME=Pareto-Front-<date> batch/runcollect.sub
```

- **Same name everywhere.** `EXPERIMENT_NAME` must be identical in all three
  submits. Left empty, it falls back to `PARETO_EXPERIMENT_NAME` in
  `batch/_stage_env.sh` (stages 1 and 2) and to the default in
  `batch/runcollect.sh` (stage 3), currently `Pareto-Front-261002`, a v2 study.
- **Subsets and retrains:** `condor_submit RUNS=batch/my_runs.txt ...` with lines
  from `runae_pareto_makegrid.sh --attempt 02 --only <filter> --output batch/my_runs.txt`.
- **Epochs:** `MAX_EPOCHS=<n>` overrides the 200 epochs of
  `configs/experiment/physics/pareto_fet_train.yaml` for stage 1. A 200-epoch run
  takes about 5 h. Above about 12 h, append `-append '+MaxRuntime = 86400'`.
- **Monitoring:** `condor_q -name bigbird103.cern.ch` (or `condor_q -global`), and
  logs in `batch/logs/<job>.<cluster>.*`. The HTCondor `.log` can contain NUL
  bytes, so use `grep -a`.
- **Resources** are set in the submit files: stage 1 has 6 cpu / 18 GB, stage 2
  has 8 cpu / 24 GB, stage 3 has 2 cpu / 6 GB. Exit code 3 of stage 2 is a
  rejected probe result, not a crash, and is not retried.
- **Test split, after the selection:** put the selected run and its γ = 0 / 50-bin
  run in `batch/test_runs.txt`, then
  `condor_submit EXPERIMENT_NAME=... batch/runae_test.sub` and afterwards
  `condor_submit EXPERIMENT_NAME=... batch/runae_test_comparison.sub`.

### The rule that matters

**Every stage must compose the same config as the stage-1 run it analyses.** The
checkpoint stores weights but not the config, and both the evaluator and the
probe loader call `load_state_dict(..., strict=True)`. A mismatched architecture
fails loudly rather than silently measuring the wrong model, but only after the
data has loaded, which costs minutes. This is why stages 1 and 2 read the same
job list on HTCondor and share `_stage_common.sh` locally.

`ADL1T_OUTPUT_ROOT` decides where `checkpoints/<experiment_name>/<run_name>`
lives (`configs/paths/default.yaml` reads it from the environment). On the
cluster, stage 1 writes into the job sandbox and `transfer_output_files` brings
the tree home, whereas stage 2, stage 3 and the test evaluation point it at the
merged tree on EOS.

### Configurations, runs, and the per-run manifest

A **configuration** is a point on the Pareto grid: `(mi_gamma,
mi_sensitive_num_bins, architecture)`. A **run** is one training of an
autoencoder at that configuration. The study is single-seed
(`pareto_study.candidate.autoencoder_seed`), so every configuration is exactly
one run and Phase 2 reports that run's metrics directly.

At the end of stage 1 every run records itself:

```
checkpoints/<experiment_name>/<run_name>/
    loss_total.ckpt          best val/loss_total among epochs whose latent has not collapsed
    loss_total_guard.json    per-epoch latent code entropy H(L) and the guard's decision
    run_manifest.yaml        identity, grid point, algorithm fingerprint, mlflow run id
    resolved_config.yaml     the fully resolved config, travelling with the checkpoint
    stage_status/train.yaml
```

`loss_total.ckpt` comes from `CollapseGuardedModelCheckpoint`
(`src/callbacks/checkpointing/collapse_guard.py`). After every validation epoch it
computes the joint entropy H(L) of the hard latent codes on the normal validation
split (logged as `val/latent_joint_code_entropy_bits`). Epochs with H(L) below
`min_joint_code_entropy_bits` (0.05 bits, i.e. full collapse) cannot become
`loss_total.ckpt`, however low their `val/loss_total`. If every epoch collapsed,
the best collapsed epoch is kept and `loss_total_guard.json` reports
`status: no_non_collapsed_epoch`; stage 3's collapse rule then rejects the run.

The manifest is written after training rather than before, because `ClearRunCheckpointDir`
wipes the run directory when a fit starts — and because a manifest present is
then a truthful claim that stage 1 finished.

Stage 2 and the re-evaluation entrypoint read it and refuse to run when `algorithm_fingerprint` does not
match the config they were given. That closes a gap `strict=True` on
`load_state_dict` cannot: it compares tensor shapes only, so a stage-2 run
composed with the wrong `mi_gamma` would otherwise load perfectly and record a
probe score against the wrong grid point. Pass `manifest_strict=false` for runs
trained before manifests existed; the stage then warns instead of stopping.

Every stage after training (the probes, the test evaluation) writes its own
`stage_status/<stage>.yaml` rather than updating the shared manifest, because
they may run concurrently and a read-modify-write from two processes on EOS loses
one of the updates.

### Stage 3 over an experiment directory

Stage 3 consumes a study map listing the one run of every configuration. There
are two ways to get one.

An existing `STUDY_ROOT/study_map.yaml` (for example one edited by hand to leave
a run out) is used as is, unless runs were added to the experiment directory
since it was written; then it is moved aside and rebuilt.

Otherwise the map is **built from an experiment directory**, using the
`run_manifest.yaml` each run carries:

```bash
EXPERIMENT_NAME=Pareto-Front-260928 bash scripts/physics/runae_pareto_runcollect.sh
```

That is the path for autoencoders trained one at a time, whenever and wherever
there was capacity, and collected afterwards. The builder refuses a directory
that holds more than one run of the same `configuration_id` (a retrain, or a
leftover second seed) and names them, so the front never depends on which of
two runs happened to be picked.

`python3 -m src.evaluation.pareto.study_map` can also be run on its own. The collector itself is
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
