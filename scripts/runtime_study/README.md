# Runtime study — rebuilding `tab:compute_resources`

Measures what this project actually cost to run, and builds the paper's
computational-resources table from those measurements.

The table reports **median wall-clock seconds per training epoch** per
(model, validation strategy), measured on a single NVIDIA GH200, and gives each
domain's **total GPU-hours** in its heading.

Output lands in `notebooks/run_times/`:

| file | contents |
|:--|:--|
| `optuna_<domain>_<model>.csv` | one row per hyperparameter-search trial |
| `<domain>_<model>[_q99]_pareto.csv` | one row per Pareto retraining |
| `compute_summary.csv` | one row per table cell, every intermediate number |
| `compute_resources_table.tex` | the table, ready to paste into the paper |
| `compute_resources_table.md` | the same table in Markdown |

## Why epoch cost, and not time per trial

The obvious approach — time a Pareto retraining and call that a trial time —
does not work, for three reasons found while building this:

1. **The sweeps ran 50 epochs, the retrainings 200.** `trainer.max_epochs=50` is
   set by every `scripts/*/run*_search.sh`, not by the `hparams_search` configs,
   so it is easy to miss. Per-epoch cost is invariant to the difference.
2. **The sweeps logged nothing.** They ran with `logger=none`, so no MLflow
   record of a search trial exists. Their durations survive only in the Optuna
   databases, which `scripts/fetch_optuna_pareto.py` does not export.
3. **Total job time is not comparable across strategies.** The post-fit
   evaluation phase is roughly 8x heavier for the semi-supervised runs (they
   evaluate ~24 checkpoints against ~3 for the agnostic ones), which runs in the
   opposite direction to CAP's cost during the fit. Only the fit phase is a
   like-for-like comparison.

So the two stages are measured separately and from their own records: per-epoch
cost from the clean GH200 retrainings, sweep cost from the Optuna databases.

## Reproducing it

Three steps, on three machines. Each script is standalone and dependency-free
(standard library only), so it can be copied to a cluster and run there without
installing anything — which is the point, since the two data sources live on
different clusters.

### 1. Sweep trial durations — on olqti

```bash
rsync -av scripts/runtime_study/harvest_optuna_times.py \
    olqti-gpu-02.cern.ch:/data/deodagiu/adl1t/scripts/runtime_study/

ssh olqti-gpu-02.cern.ch 'bash -lc "conda activate optuna-ui && \
    cd /data/deodagiu/adl1t && \
    python scripts/runtime_study/harvest_optuna_times.py --outdir /tmp/optuna_times"'

rsync -av olqti-gpu-02.cern.ch:/tmp/optuna_times/ notebooks/run_times/
```

Use `optuna-ui`, not `adl1t`: the databases were written by Optuna 4.7.0 and the
training env pins 2.10.1. (The script reads the `trials` table with raw
`sqlite3` so the schema pin cannot bite it either way, but the system Python on
olqti is 3.6 and too old to run it.) Databases are opened `mode=ro`.

### 2. Per-epoch cost of the retrainings — on clariden

```bash
rsync -av scripts/runtime_study/harvest_run_times.py \
    clariden:/users/podagiu/adl1t_lab/scripts/runtime_study/

ssh clariden 'bash -lc "conda activate adl1t && cd /users/podagiu/adl1t_lab && \
    python scripts/runtime_study/harvest_run_times.py \
        --tracking-uri file:/iopsstor/scratch/cscs/podagiu/logs/mlflow/mlruns \
        --outdir /iopsstor/scratch/cscs/podagiu/run_times"'

rsync -av clariden:/iopsstor/scratch/cscs/podagiu/run_times/ notebooks/run_times/
```

Timing comes from the `epoch_idx` metric, logged once per epoch by
`ADLightningModule.on_validation_epoch_end`. It is logged for every model and
strategy regardless of which callbacks are active, so it is comparable across
the whole table, and its absence is what identifies the eval-only
(`train=false`) resubmissions.

The MLflow file store is read directly rather than through `MlflowClient`:
`search_runs` eagerly parses every metric key of every run, and the physics runs
carry one metric directory per validation dataset, which made that route take
hours instead of seconds.

### 3. Build the table — locally

```bash
python scripts/runtime_study/build_compute_table.py
```

## What gets counted

Only the four strategies of the paper's table, at the **250 Hz tier**:

| column | Optuna study | Pareto run name |
|:--|:--|:--|
| Semi | `cvar25eff` | `cvar25_t*` |
| Stable | `drift` | `stability_t*` |
| W1 | `wasserstein` | `wasserstein_t*` |
| CAP | `cap` | `cap_t*` |

Excluded, deliberately:

- **the q99 tier** — physics has it, CIFAR-10 and RobustAD have no counterpart,
  so including it would rest the physics rows on twice the data of the others.
  The `TIER CHECK` diagnostic prints what it would have said.
- **`ae_cap_exploration.db`** — 22 studies optimising the CAP metric's own
  parameters, not the model/strategy sweeps.
- **`cvar10eff`** and **`consistency`** — no column in the table; consistency is
  still work in progress.
- **DTE** — no Pareto retrainings exist.

Retrainings are kept only if they actually trained: an `epoch_idx` history must
exist and must cover the full 200 epochs. Where a run name appears more than
once (duplicate eval waves, `*_retry` scripts) the newest wins. Note that slurm
`State=COMPLETED` is *not* proof a run trained — some jobs died on
`EmptyDatasetError` and still reported success — which is why the epoch history,
not the job state, is the gate.

## How GPU-hours are computed

```
sweep_h   = sum(trial durations) / concurrent processes per GPU
retrain_h = sum(retraining durations)          # one process per GPU on clariden
```

The division matters: three trials sharing one GPU for an hour cost one
GPU-hour, not three. The GPU and concurrency per model are in `HARDWARE` in
`build_compute_table.py`, taken from the paper's existing table.

A consequence worth knowing: the sweeps packed 2–3 trials per GPU and they
barely slowed each other down, because these models are small enough to leave a
GPU underused. Re-running the same sweeps one-job-per-GPU on GH200 would cost
roughly twice as many GPU-hours — that is the `GH200-equiv` column, printed for
comparison but not what the paper quotes.

## Checking the result

`build_compute_table.py` prints five diagnostics. What they should look like:

- **TOTALS** — the headline numbers, split into sweep and retraining.
- **SPREAD** — per-cell trial-time spread. `med/p10` near 1 means tight; the few
  cells above 1.5 are flagged. The last column gives the retrainings' relative
  per-epoch spread, which separates the two causes: if it is equally wide, the
  variation is the search space (different architectures), not GPU contention.
- **CROSS-CHECK** — the same 50 epochs measured two independent ways. Read it for
  *consistency across strategies within a model*, not closeness to 1: the trial
  also carries setup and its own evaluation, and the agnostic sweeps ran with
  `anomaly_eff` disabled while the retrainings kept it.
- **TIER CHECK** — the excluded q99 tier, for evidence the exclusion costs
  precision rather than changing the answer. Currently 24 of 26 cells agree
  within 7%.
- **THIN CELLS** — cells resting on fewer than 3 runs, each compared against the
  same model's other cheap strategies. Currently 16 of 17 are consistent; the
  one marked `SUSPECT` should not be quoted without a second look.

One further check, not automated — the per-run decomposition should close:

```bash
python - <<'EOF'
import csv, glob, statistics
res = []
for f in glob.glob('notebooks/run_times/*_pareto.csv'):
    for r in csv.DictReader(open(f)):
        modelled = (float(r['setup_s'])
                    + (int(r['n_epochs']) - 1) * float(r['sec_per_epoch'])
                    + float(r['post_fit_s']))
        res.append(abs(modelled - float(r['total_s'])) / float(r['total_s']))
print(f"runs={len(res)}  median={statistics.median(res):.2%}  "
      f"p95={sorted(res)[int(0.95 * len(res))]:.2%}")
EOF
```

Expect a median residual well under 1% (currently 0.16%, p95 2.0% over 784
runs). A large residual would mean the per-epoch median is not representative of
the run it came from.

## Caveats to carry into the caption

- Per-epoch costs are GH200; the totals are the compute *actually consumed*,
  most of it on the original mixed hardware at the concurrency listed in
  `HARDWARE`.
- For CIFAR-10 and RobustAD the differences between strategy columns are smaller
  than the hyperparameter-driven spread within a single strategy. Those rows
  should not be read as "strategy X is cheaper"; only the physics CAP column is
  a large, genuine effect.
