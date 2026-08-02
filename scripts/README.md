# `scripts/`

Everything outside `src/` that produced the paper: the exact commands the
experiments were run with, the tooling that generated and submitted them, and
the harvests that turn finished runs into the notebooks' inputs.

Run everything from the repository root.

| Directory | What lives there |
| --- | --- |
| `physics/` `cifar10/` `robustad/` | The **command catalogues** — one file per model, per domain. Not runnable scripts; see below. |
| `cluster/` | Submitting those catalogues to slurm (clariden), to an NGT pod, and checking what is still missing. |
| `optuna/` | Reading the Optuna study databases: Pareto fronts out, retraining catalogues in. |
| `analysis/` | Turning finished MLflow runs and the notebooks into the paper's numbers and figures. |
| `runtime_study/` | Rebuilding `tab:compute_resources`. Self-contained — see [its README](runtime_study/README.md). |
| `publish_l1data/` | Packaging the L1AD data for Zenodo and HuggingFace. Self-contained — see [its README](publish_l1data/README.md). |
| `setup.sh` `symbolink.sh` | One-time bootstrap. `setup.sh` is required before anything composes. |

## The catalogues are documentation, not programs

Every `run<model>[_q99]_{search,pareto}.sh` is **100% commented out**. A file is a
list of `python3 src/train.py` blocks, each self-contained and copy-pasteable:

- `*_search.sh` — one hyperparameter-search driver per validation strategy
  (cvar25, cvar10, CAP, stability, Wasserstein, consistency). Hand-written; these
  are the canonical record of the sweeps' overrides.
- `*_pareto.sh` — one retraining per point on each strategy's Pareto front.
  **Generated** by `optuna/make_pareto_scripts.py`; regenerate, do not hand-edit.
- `rundte_cache.sh` — the exception: live code. A 2-epoch default-config run whose
  purpose is to materialise the data caches a DTE sweep needs. There are no DTE
  Pareto CSVs, so there is no DTE pareto catalogue.
- `_q99` is the physics-only q = 0.99 background-rate tier; the plain files are
  the 250 Hz tier.

The submitters in `cluster/` parse these blocks and run them; `cluster/lib.sh`
holds the parser, which is the one thing that must stay in step with the
generator's output layout.

Two properties of the catalogues are **records, not bugs**: the physics
ae/dsae/dsvae/realnvp/vae searches carry `hydra/launcher=submitit_local` because
they were run that way on olqti (only dte/svdd and all of cifar10/robustad went to
clariden), and several of them set `n_startup_trials` at or above `n_trials`, so
those studies were effectively random search. `submit_search.sh --list` prints the
launcher per block so this is visible before you launch anything.

## The pipeline

```
logs/optuna/<domain>/<model>.db          the sweeps' only durable record
  |  optuna/fetch_optuna_pareto.py --all
  v
notebooks/paretos/<domain>/*.csv         trials + an is_pareto flag
  |  optuna/make_pareto_scripts.py
  v
<domain>/run<model>_pareto.sh            one commented block per front point
  |  cluster/submit_pareto.sh  (slurm)   or  cluster/submit_pareto_ngt.sh  (pod)
  v
MLflow runs named <strategy>_t<trial>
  |  analysis/harvest_pareto_effs.py
  v
notebooks/pareto_effs/<experiment>.csv   -> notebooks/effs_*.nb
```

`optuna/splice_pareto_section.py` is the surgical version of step 3: it inserts
**one** strategy's section into an existing catalogue, because a plain
regeneration would rewrite every already-retrained block. Use it whenever a file
already has sections you care about.

## File by file

### `cluster/`

| File | |
| --- | --- |
| `submit_search.sh` | Launch a `run*_search.sh` file's Optuna drivers. Blocks already carry `-m` and their launcher, so this adds only the data path and a private sweep dir. Pre-creates each sqlite study serially to dodge the alembic bootstrap race. `--list`, `--dry-run` (works off the cluster), `--only`. |
| `submit_pareto.sh` | Submit a `run*_pareto.sh` file to slurm, one job per point. Blocks are bare trainings, so this prepends the submitit launcher and normalises `trainer.devices`. **`--only` is not optional in practice** — a whole physics file is up to 78 jobs at 12 h each. |
| `submit_pareto_ngt.sh` | The same points on an NGT session, which has no batch system: this *is* the scheduler. Fixed slot table, one GPU + 3 CPUs per job, `--shard i/n` across sessions. Expands the cgroup's CPU list, because those pods get a non-contiguous, non-zero-based set. |
| `sweep_status.py` | Every study's COMPLETE trials against its target, whether a driver is still alive (read from the log, not `ps` — clariden rotates login nodes), and the `submit_search.sh` command to finish it. |
| `lib.sh` | Sourced by the three submitters: block parser, `--only` matcher, repo-root and placeholder guards. |

### `optuna/`

| File | |
| --- | --- |
| `fetch_optuna_pareto.py` | Study → CSV with an `is_pareto` column. `--all` sweeps every db under `logs/optuna/`. |
| `make_pareto_scripts.py` | Those CSVs → the `run*_pareto.sh` catalogues, with knee/best/original-pick annotations and knee-window trimming for oversize fronts. |
| `splice_pareto_section.py` | Insert or `--replace` one strategy's section in place. Verifies the splice is purely additive by stripping it back out and requiring a byte-exact match. |
| `downgrade_optuna_db.py` | Reverse an `optuna storage upgrade` so the pinned 2.10.1 sweeper can write to a db again. Works on a copy; `--digest` compares contents across environments. |

Reading the databases needs a modern Optuna (the training env pins 2.10.1); see
each script's docstring.

### `analysis/`

| File | |
| --- | --- |
| `harvest_pareto_effs.py` | MLflow → `notebooks/pareto_effs/*.csv`. `--mode history` reports each run's per-signal efficiency at the epoch its strategy checkpoint monitored and at the last epoch; `--mode eval` flattens the evaluator's test-split scalars. |
| `get_mlflow_corrs.py` | MLflow metric *histories* → `notebooks/exported_metrics/`, which `notebooks/corrs.nb` correlates against signal efficiency. |
| `nb_run.wls` | Run a `.nb` headless with the front end attached, strip volatile cell metadata, log every output. Seeds the RNG so bootstrap CIs reproduce. |
| `nb_compare.wls` | Compare two trees of exported figures: inventory → file hash → rasterised image distance. |

`wolframscript` is not on `PATH` on macOS; invoke it as
`/Applications/Wolfram.app/Contents/MacOS/wolframscript -f scripts/analysis/nb_run.wls ...`.
