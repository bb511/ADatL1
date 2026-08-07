"""Tests for deterministic physics campaign exports."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle  # nosec B403 -- writes only synthetic trusted-local test fixtures
import random
import shutil
import subprocess  # nosec B404 -- fixed Git argv executes only in local test fixtures
from pathlib import Path

import optuna
import pytest
import torch
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

CAMPAIGN_SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
REPORT_SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_report.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


campaign = _load("physics_background_pairing_campaign_for_report", CAMPAIGN_SCRIPT)
report = _load("physics_background_pairing_report", REPORT_SCRIPT)
report.TARGET_TRIALS = 1
REAL_EXECUTION_REPOSITORY_EVIDENCE = report._execution_repository_evidence


def _write_exact_checkpoint(path: Path, epoch: int) -> None:
    """Write a minimal continuation checkpoint with the production RNG contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "total": {
                            "ready": epoch + 1,
                            "started": epoch + 1,
                            "processed": epoch + 1,
                        }
                    }
                }
            },
            "callbacks": {
                report.EXACT_RETRAIN_RNG_STATE_KEY: {
                    "schema_version": 1,
                    "python": random.getstate(),
                    "numpy": {
                        "algorithm": "MT19937",
                        "keys": torch.zeros(624, dtype=torch.uint32),
                        "position": 0,
                        "has_gauss": 0,
                        "cached_gaussian": 0.0,
                    },
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": [torch.get_rng_state()],
                    "datamodule_shuffler": torch.Generator().manual_seed(123).get_state(),
                }
            },
        },
        path,
    )


@pytest.fixture(autouse=True)
def _clean_reporting_checkout(monkeypatch):
    """Synthetic finalization is independent of the developer worktree state."""
    monkeypatch.setattr(report, "_git_state", lambda: ("b" * 40, True, []))
    monkeypatch.setattr(
        report,
        "_execution_repository_evidence",
        lambda root, design, explicit=None: {
            "execution_repository_root": "/synthetic/execution",
            "execution_repository_commit": design["code_commit"],
            "execution_repository_clean": True,
            "execution_uv_lock_sha256": "d" * 64,
        },
    )


def _git(repository: Path, *arguments: str) -> str:
    git = shutil.which("git")
    assert git is not None
    return subprocess.run(  # nosec B603
        [git, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_execution_repository_evidence_uses_campaign_worktree_convention(
    tmp_path,
) -> None:
    """Execution provenance is authenticated from the campaign's immutable worktree."""
    campaign_root = tmp_path / "campaigns" / "campaign-name"
    repository = tmp_path / "worktrees" / campaign_root.name
    campaign_root.mkdir(parents=True)
    repository.mkdir(parents=True)
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "add", "uv.lock")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    commit = _git(repository, "rev-parse", "HEAD")

    evidence = REAL_EXECUTION_REPOSITORY_EVIDENCE(campaign_root, {"code_commit": commit})

    assert evidence == {
        "execution_repository_root": str(repository.resolve()),
        "execution_repository_commit": commit,
        "execution_repository_clean": True,
        "execution_uv_lock_sha256": report._sha256(repository / "uv.lock"),
    }


def test_execution_repository_evidence_rejects_dirty_explicit_worktree(tmp_path) -> None:
    """An explicit execution checkout cannot conceal post-run code changes."""
    repository = tmp_path / "execution"
    repository.mkdir()
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "add", "uv.lock")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    commit = _git(repository, "rev-parse", "HEAD")
    (repository / "uv.lock").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty"):
        REAL_EXECUTION_REPOSITORY_EVIDENCE(
            tmp_path / "campaign", {"code_commit": commit}, repository
        )


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mlflow_run(
    root: Path,
    design: dict,
    *,
    model: str,
    phase: str,
    epochs: int,
    finished: bool = True,
) -> tuple[str, str, str]:
    """Create one compact but contract-complete synthetic FileStore run."""
    store = root / "runtime" / "logs" / "mlflow" / "mlruns"
    client = MlflowClient(tracking_uri=store.resolve().as_uri())
    experiment_name = f"physics_bgpair_{phase}_{model}"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        client.create_experiment(experiment_name)
        if experiment is None
        else str(experiment.experiment_id)
    )
    run_name = f"{phase}-{model}"
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": run_name,
            "mlflow.source.git.commit": design["code_commit"],
        },
    )
    metrics = []
    keys = report._selection_metric_keys(design, model)
    for epoch in range(epochs):
        step = 100 + 17 * epoch
        metrics.append(Metric("epoch_idx", float(epoch), 1_000 + epoch, step))
        metrics.extend(Metric(key, 0.5, 1_000 + epoch, step) for key in keys)
    for start in range(0, len(metrics), 500):
        client.log_batch(run.info.run_id, metrics=metrics[start : start + 500])
    if finished:
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=10_000)
    return str(experiment_id), run.info.run_id, run_name


def _campaign(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "campaign"
    preflight = tmp_path / "preflight.json"
    _json(
        preflight,
        {
            "dataset_1": "ZB_run396102",
            "dataset_2": "ZB_run398183",
            "events_per_source": 163840,
            "tables": [
                {
                    "stage": stage,
                    "strategy": strategy,
                    "authenticated": True,
                    "pairs": 163840,
                }
                for stage in ("validate", "test")
                for strategy in (
                    "flat_physical",
                    "physics_summary",
                    "typed_sliced_wasserstein",
                    "jetclr",
                )
            ],
        },
    )
    studies = campaign._studies()
    design = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "code_commit": "a" * 40,
        "preflight": str(preflight),
        "preflight_sha256": report._sha256(preflight),
        "target_trials_per_study": 1,
        "search_epochs": 50,
        "retrain_epochs": 200,
        "shared_trial_pool": True,
        "models": list(campaign.MODELS),
        "scores_by_model": {
            model: list(scores) for model, scores in campaign.SCORES_BY_MODEL.items()
        },
        "cap_strategies": list(campaign.CAP_STRATEGIES),
        "studies": studies,
    }
    _json(root / "design.json", design)
    slurm_log = root / "runtime" / "slurm" / "synthetic.out"
    slurm_log.parent.mkdir(parents=True, exist_ok=True)
    slurm_log.write_text("synthetic completed job\n", encoding="utf-8")
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "job_history.csv").write_text(
        "job_id,state,commit,log_path,log_sha256\n"
        f"1234,COMPLETED,{design['code_commit']},{slurm_log},{report._sha256(slurm_log)}\n",
        encoding="utf-8",
    )
    (state / "failures.jsonl").write_text("", encoding="utf-8")
    for study in studies:
        database = root / "studies" / f"{study['model']}.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        directions = [
            direction for objective in study["objectives"] for direction in objective["directions"]
        ]
        loaded = optuna.create_study(
            study_name=study["study_name"],
            storage=f"sqlite:///{database}",
            directions=directions,
        )
        loaded.add_trial(optuna.trial.create_trial(values=[0.5] * len(directions)))
        experiment_id, run_id, run_name = _mlflow_run(
            root,
            design,
            model=study["model"],
            phase="search",
            epochs=design["search_epochs"],
        )
        checkpoint_root = root / "runtime" / "checkpoints" / study["model"]
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        thresholds = {"thres_operational": torch.tensor(0.4)}
        if study["model"] in {"ae", "vae"}:
            thresholds["thres_operational__residual_oas"] = torch.tensor(0.4)
        payload = {"state_dict": thresholds}
        torch.save(payload, checkpoint_root / "last.ckpt")
        if study["model"] != "ae":
            stable = checkpoint_root / "single" / "ascore_operational" / "stable"
            stable.mkdir(parents=True)
            torch.save(payload, stable / "epoch.ckpt")
        for objective in study["objectives"]:
            primary = checkpoint_root / report._primary_checkpoint_candidate(objective)
            primary.mkdir(parents=True)
            torch.save(payload, primary / "epoch.ckpt")
        artifact = root / "sweeps" / study["model"] / "stamp" / "0" / "optimized_metric.json"
        objective_order = [objective["id"] for objective in study["objectives"]]
        _json(
            artifact,
            {
                "schema_version": 2,
                "objective_order": objective_order,
                "optimized_metric": [0.5] * len(directions),
                "selections": {
                    objective: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [0.5, 0.5],
                    }
                    for objective in objective_order
                },
                "run_provenance": {
                    "experiment_name": f"physics_bgpair_search_{study['model']}",
                    "run_name": run_name,
                    "hydra_output_dir": str(artifact.parent),
                    "hydra_job_num": 0,
                    "checkpoint_run_dir": str(checkpoint_root),
                    "loggers": [{"experiment_id": experiment_id, "run_id": run_id}],
                },
            },
        )
        hydra = artifact.parent / ".hydra"
        hydra.mkdir()
        (hydra / "config.yaml").write_text("model: synthetic\n", encoding="utf-8")
        (hydra / "overrides.yaml").write_text("[]\n", encoding="utf-8")
        (artifact.parent / "train.log").write_text("synthetic search log\n", encoding="utf-8")
        _json(
            artifact.parents[1] / "trial_bindings.json",
            {
                "schema_version": 1,
                "model": study["model"],
                "study_name": study["study_name"],
                "prior_trial_numbers": [],
                "bindings": [
                    {
                        "model": study["model"],
                        "study_name": study["study_name"],
                        "trial_number": 0,
                        "hydra_job_num": 0,
                        "hydra_run_dir": str(artifact.parent),
                        "optimized_metric_artifact": str(artifact),
                        "optimized_metric_artifact_sha256": report._sha256(artifact),
                        "state": "COMPLETE",
                        "params": {},
                        "values": [0.5] * len(directions),
                    }
                ],
            },
        )
    return root, design


def _mlflow_rows(root: Path, design: dict, *, retrains: bool = False):
    """Return indexed synthetic run rows and their canonical store."""
    _, rows = report._checkpoint_rows(root, design, include_retrains=retrains)
    store = root / "runtime" / "logs" / "mlflow" / "mlruns"
    return rows, store


def _run_store(store: Path, row: dict) -> Path:
    return store / str(row["experiment_id"]) / str(row["run_id"])


def test_snapshot_is_stable_and_selects_one_trial_per_study(tmp_path) -> None:
    """Repeated search snapshots are byte-stable for unchanged sources."""
    root, _ = _campaign(tmp_path)
    output = root / "portable"

    first = report.snapshot(root, output)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    second = report.snapshot(root, output)
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}

    assert first == second
    assert before == after
    with (output / "study_status.csv").open(encoding="utf-8") as handle:
        rows = list(report.csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["selected_for_freeze"] for row in rows} == {"1"}
    with (output / "trials.csv").open(encoding="utf-8") as handle:
        trials = list(report.csv.DictReader(handle))
    assert len(trials) == 6
    assert all(row["selected_for_freeze"] == "True" for row in trials)


def test_search_checkpoint_candidate_audit_allows_only_absent_stable(tmp_path) -> None:
    """Last/primary are mandatory, while the reference stable slot may be absent."""
    root, design = _campaign(tmp_path)
    bindings = report._binding_index(root)

    rows = report._checkpoint_candidate_rows(root, design, bindings, final=True)

    assert len(rows) == 68
    assert sum(bool(row["required"]) for row in rows) == 62
    stable = [row for row in rows if row["slot"] == "stable_native"]
    assert len(stable) == 6
    assert sum(bool(row["available"]) for row in stable) == 5
    assert all(row["checkpoint_sha256"] for row in rows if row["available"])

    ae_objective = design["studies"][0]["objectives"][0]
    primary = (
        root
        / "runtime"
        / "checkpoints"
        / "ae"
        / report._primary_checkpoint_candidate(ae_objective)
    )
    torch.save({"state_dict": {}}, primary / "duplicate.ckpt")
    with pytest.raises(RuntimeError, match="Invalid primary_ema candidate count"):
        report._checkpoint_candidate_rows(root, design, bindings, final=True)
    (primary / "duplicate.ckpt").unlink()
    (primary / "epoch.ckpt").unlink()
    with pytest.raises(RuntimeError, match="Invalid primary_ema candidate count"):
        report._checkpoint_candidate_rows(root, design, bindings, final=True)


def test_snapshot_accepts_active_unhashed_job_log(tmp_path) -> None:
    """An operational snapshot remains usable before a running Slurm log is sealed."""
    root, _ = _campaign(tmp_path)
    ledger = root / "state" / "job_history.csv"
    row = ledger.read_text(encoding="utf-8").splitlines()[1].split(",")
    row[1] = "RUNNING"
    row[-1] = ""
    ledger.write_text(
        ledger.read_text(encoding="utf-8").splitlines()[0] + "\n" + ",".join(row) + "\n",
        encoding="utf-8",
    )

    latest = report.snapshot(root, root / "portable")

    assert latest["ready_to_freeze"] is True


def test_snapshot_does_not_read_downstream_oracle(tmp_path) -> None:
    """Pre-freeze status remains usable even when a downstream file is invalid."""
    root, _ = _campaign(tmp_path)
    oracle = root / "results" / "oracle_winners.json"
    oracle.parent.mkdir(parents=True)
    oracle.write_text("not-json\n", encoding="utf-8")

    status = report.snapshot(root, root / "portable")

    assert status["ready_to_freeze"] is True


def test_snapshot_rejects_binding_that_differs_from_optuna(tmp_path) -> None:
    """Chunk identity cannot be inferred or accepted when its objective vector differs."""
    root, _ = _campaign(tmp_path)
    path = root / "sweeps" / "ae" / "stamp" / "trial_bindings.json"
    binding = json.loads(path.read_text(encoding="utf-8"))
    binding["bindings"][0]["values"][0] = 999.0
    _json(path, binding)

    with pytest.raises(RuntimeError, match="differs from Optuna trial"):
        report.snapshot(root, root / "portable")


def test_final_study_capture_rejects_live_and_extra_usable_trials(tmp_path) -> None:
    """Final publication requires quiescence and exactly the frozen usable budget."""
    root, design = _campaign(tmp_path / "live")
    study = design["studies"][0]
    loaded = report._load_study(root, study)
    loaded.enqueue_trial({})
    with pytest.raises(RuntimeError, match="not quiescent"):
        report._capture_quiescent_studies(root, design)

    root, design = _campaign(tmp_path / "extra")
    study = design["studies"][0]
    loaded = report._load_study(root, study)
    directions = [
        direction for objective in study["objectives"] for direction in objective["directions"]
    ]
    loaded.add_trial(optuna.trial.create_trial(values=[0.4] * len(directions)))
    with pytest.raises(RuntimeError, match="exactly 1"):
        report._capture_quiescent_studies(root, design)


def test_study_evidence_detects_database_change(tmp_path) -> None:
    """Even metadata-only database changes invalidate the reporting read window."""
    root, design = _campaign(tmp_path)
    evidence = report._capture_quiescent_studies(root, design)
    database = Path(evidence["ae"]["path"])
    stat = database.stat()
    os.utime(database, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    with pytest.raises(RuntimeError, match="changed during reporting"):
        report._assert_studies_unchanged(evidence)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "Missing indexed MLflow run"),
        ("experiment", "experiment differs"),
        ("status", "invalid status"),
        ("commit", "execution commit differs"),
        ("artifact", "artifact URI escapes"),
        ("run_id", "run ID differs"),
        ("epoch", "epoch history"),
        ("history", "Invalid MLflow selection history"),
        ("metric_escape", "escapes its canonical run"),
        ("malformed_history", "Malformed MLflow metric history"),
    ),
)
def test_mlflow_final_validation_fails_closed(tmp_path, corruption: str, message: str) -> None:
    """FileStore identity, status, containment, and histories are authoritative."""
    root, design = _campaign(tmp_path)
    rows, store = _mlflow_rows(root, design)
    row = rows[0]
    run_store = _run_store(store, row)
    if corruption == "missing":
        run_store.rename(run_store.with_name(run_store.name + ".missing"))
    elif corruption == "experiment":
        meta = store / str(row["experiment_id"]) / "meta.yaml"
        meta.write_text(
            meta.read_text(encoding="utf-8").replace(
                f"name: {row['experiment_name']}", "name: wrong-experiment"
            ),
            encoding="utf-8",
        )
    elif corruption == "status":
        meta = run_store / "meta.yaml"
        meta.write_text(
            meta.read_text(encoding="utf-8").replace("status: 3", "status: 1"),
            encoding="utf-8",
        )
    elif corruption == "commit":
        (run_store / "tags" / "mlflow.source.git.commit").write_text("wrong", encoding="utf-8")
    elif corruption == "artifact":
        meta = run_store / "meta.yaml"
        lines = [
            "artifact_uri: file:///tmp/escaped" if line.startswith("artifact_uri:") else line
            for line in meta.read_text(encoding="utf-8").splitlines()
        ]
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif corruption == "run_id":
        meta = run_store / "meta.yaml"
        meta.write_text(
            meta.read_text(encoding="utf-8").replace(
                f"run_id: {row['run_id']}", f"run_id: {'f' * 32}"
            ),
            encoding="utf-8",
        )
    elif corruption == "epoch":
        epoch = run_store / "metrics" / "epoch_idx"
        epoch.write_text(
            "\n".join(epoch.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
            encoding="utf-8",
        )
    elif corruption == "history":
        key = "train/ascore_operational"
        (run_store / "metrics" / key).unlink()
    elif corruption == "metric_escape":
        key = "train/ascore_operational"
        metric = run_store / "metrics" / key
        outside = tmp_path / "outside-metric"
        outside.write_text(metric.read_text(encoding="utf-8"), encoding="utf-8")
        metric.unlink()
        metric.symlink_to(outside)
    else:
        key = "train/ascore_operational"
        (run_store / "metrics" / key).write_text("not-a-metric\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        report._validate_mlflow_runs(root, design, rows, final=True)


def test_mlflow_validation_does_not_scan_unrelated_file_store_data(monkeypatch, tmp_path) -> None:
    """Exact bound paths replace generic run and history discovery."""
    root, design = _campaign(tmp_path)
    rows, _ = _mlflow_rows(root, design)
    monkeypatch.setattr(
        MlflowClient,
        "get_run",
        lambda *args, **kwargs: pytest.fail("generic FileStore run scan was used"),
    )
    monkeypatch.setattr(
        MlflowClient,
        "get_metric_history",
        lambda *args, **kwargs: pytest.fail("generic FileStore history scan was used"),
    )

    validated = report._validate_mlflow_runs(root, design, rows, final=True)

    assert len(validated) == len(rows)
    assert all(row["histories_validated"] for row in validated)


def _frozen_results(root: Path, design: dict) -> None:
    memberships = []
    retrains = []
    winners = []
    anomalies = list(report.DOWNSTREAM_DATASETS)
    for retrain_index, study in enumerate(design["studies"]):
        objectives = [objective["id"] for objective in study["objectives"]]
        retrain_row = {
            "model": study["model"],
            "trial_number": 0,
            "params": {},
            "logical_objectives": sorted(objectives),
            "seed": 123,
            "epochs": 200,
            "retrain_index": retrain_index,
        }
        retrains.append(retrain_row)
        artifact = root / "retraining" / "hydra" / f"{retrain_index:04d}" / "optimized_metric.json"
        experiment_id, run_id, run_name = _mlflow_run(
            root,
            design,
            model=study["model"],
            phase="retrain",
            epochs=design["retrain_epochs"],
        )
        _json(
            artifact,
            {
                "schema_version": 2,
                "objective_order": objectives,
                "optimized_metric": [0.5] * (2 * len(objectives)),
                "selections": {
                    objective: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [0.5, 0.5],
                    }
                    for objective in objectives
                },
                "run_provenance": {
                    "experiment_name": f"physics_bgpair_retrain_{study['model']}",
                    "run_name": run_name,
                    "hydra_output_dir": str(artifact.parent),
                    "hydra_job_num": retrain_index,
                    "checkpoint_run_dir": str(root / "runtime" / "checkpoints" / study["model"]),
                    "loggers": [{"experiment_id": experiment_id, "run_id": run_id}],
                },
            },
        )
        hydra = artifact.parent / ".hydra"
        hydra.mkdir()
        callbacks = {"anomaly_eff": {"output_name": "ascore/full"}}
        evaluation_callbacks = {
            "anomaly_efficiency": {"name": "eff_native", "output_name": "ascore/full"}
        }
        if study["model"] in {"ae", "vae"}:
            callbacks["anomaly_eff_oas"] = {
                "output_name": "ascore/residual_oas",
                "threshold_namespace": "residual_oas",
                "metric_suffix": "residual_oas",
            }
            evaluation_callbacks["anomaly_efficiency_oas"] = {
                "name": "eff_residual_oas",
                "output_name": "ascore/residual_oas",
                "threshold_namespace": "residual_oas",
            }
        resolved_config = {
            "data": {"validation_aux_datasets": []},
            "algorithm": {
                "target_rate": report.TARGET_RATE_KHZ,
                "base_rate": report.BASE_RATE_KHZ,
            },
            "callbacks": callbacks,
            "evaluation": {"callbacks": evaluation_callbacks},
        }
        (hydra / "config.yaml").write_text(
            json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
        )
        (hydra / "overrides.yaml").write_text("[]\n", encoding="utf-8")
        (artifact.parent / "train.log").write_text("synthetic retrain log\n", encoding="utf-8")
        checkpoint_root = root / "runtime" / "checkpoints" / study["model"]
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        thresholds = {"thres_operational": torch.tensor(0.4)}
        if study["model"] in {"ae", "vae"}:
            thresholds["thres_operational__residual_oas"] = torch.tensor(0.4)
        torch.save({"state_dict": thresholds}, checkpoint_root / "last.ckpt")
        value_files = []
        for score in design["scores_by_model"][study["model"]]:
            values_path = (
                checkpoint_root / "last" / "plots" / "test" / f"eff_{score}" / "values.csv"
            )
            values_path.parent.mkdir(parents=True, exist_ok=True)
            values_path.write_text(
                "checkpoint,intervention,metric,value\n"
                + "".join(f"last,{anomaly},efficiency_operational,0.5\n" for anomaly in anomalies),
                encoding="utf-8",
            )
            value_files.append({"path": str(values_path), "sha256": report._sha256(values_path)})
            callbacks = [
                (f"cap_{score}_{strategy}_summary", ({"last": 0.5}, {"last": 0.4}))
                for strategy in design["cap_strategies"]
            ]
            callbacks.extend(
                [
                    (f"wasserstein_{score}_summary", ({"last": 0.5},)),
                    (
                        f"drift_{score}_summary",
                        (
                            {"operational": {"last": 0.5}},
                            {"operational": {"last": 0.6}},
                        ),
                    ),
                ]
            )
            for callback, objects in callbacks:
                summary = checkpoint_root / "plots" / "test" / callback / "summary.pkl"
                summary.parent.mkdir(parents=True, exist_ok=True)
                with summary.open("wb") as handle:
                    for value in objects:
                        pickle.dump(value, handle)
        marker = {
            "schema_version": 2,
            "started_at": "2026-01-01T23:00:00+00:00",
            "completed_at": "2026-01-02T00:00:00+00:00",
            "elapsed_seconds": 3600.0,
            "slurm_job_id": "1234",
            "slurm_array_job_id": "1234",
            "slurm_array_task_id": str(retrain_index),
            "row": retrain_row,
            "command": [],
            "optimized_metric_artifact": str(artifact),
            "optimized_metric_artifact_sha256": report._sha256(artifact),
            "checkpoint_root": str(checkpoint_root),
            "value_files": value_files,
        }
        _json(root / "retraining" / "markers" / f"{retrain_index:04d}.json", marker)
        for objective in study["objectives"]:
            membership = {
                "membership_index": len(memberships),
                "model": study["model"],
                "logical_objective": objective["id"],
                "score": objective["score"],
                "metric": objective["metric"],
                "strategy": objective["strategy"],
                "trial_number": 0,
                "objective_values": [0.5, 0.5],
                "retrain_index": retrain_index,
            }
            memberships.append(membership)
            winners.append(
                {
                    **membership,
                    "params": {},
                    "optimized_ckpt_name": "last",
                    "retrained_objective_values": [0.5, 0.5],
                    "mean_downstream_efficiency": 0.5,
                    "downstream_efficiencies": {name: 0.5 for name in anomalies},
                }
            )
    manifest = {
        "schema_version": 2,
        "created_at": "2026-01-02T00:00:00+00:00",
        "code_commit": design["code_commit"],
        "selection_uses_downstream_anomalies": False,
        "front_memberships": memberships,
        "retrain_runs": retrains,
    }
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    _json(manifest_path, manifest)
    for retrain_row in retrains:
        index = int(retrain_row["retrain_index"])
        state_dir = root / "retraining" / "restart_state" / f"{index:04d}"
        segment_rows = []
        attempts = []
        for attempt_number, boundary in enumerate((40, 80, 120, 160, 200), start=1):
            checkpoint = state_dir / "segments" / f"epoch_{boundary:03d}.ckpt"
            config = state_dir / "segments" / f"epoch_{boundary:03d}_config.yaml"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write_exact_checkpoint(checkpoint, epoch=boundary - 1)
            config.write_text(f"trainer:\n  max_epochs: {boundary}\n", encoding="utf-8")
            segment_rows.append(
                {
                    "epoch_boundary": boundary,
                    "attempt_number": attempt_number,
                    "continuation_checkpoint": str(checkpoint.resolve()),
                    "continuation_checkpoint_sha256": report._sha256(checkpoint),
                    "resolved_config": str(config.resolve()),
                    "resolved_config_sha256": report._sha256(config),
                }
            )
            attempts.append(
                {
                    "attempt_number": attempt_number,
                    "started_at": f"2026-01-01T{attempt_number:02d}:00:00+00:00",
                    "completed_at": f"2026-01-01T{attempt_number:02d}:30:00+00:00",
                    "target_epoch_boundary": boundary,
                    "phase": "training",
                }
            )
        attempts.append(
            {
                "attempt_number": 6,
                "started_at": "2026-01-01T23:00:00+00:00",
                "completed_at": "2026-01-02T00:00:00+00:00",
                "target_epoch_boundary": 200,
                "phase": "evaluation",
            }
        )
        model = retrain_row["model"]
        run_name = f"{model}__trial{int(retrain_row['trial_number']):04d}"
        experiment_name = f"physics_bgpair_retrain_{model}"
        mlflow_run_id = next(
            json.loads(path.read_text(encoding="utf-8"))["run_provenance"]["loggers"][0]["run_id"]
            for path in [root / "retraining" / "hydra" / f"{index:04d}" / "optimized_metric.json"]
        )
        identity = {
            "code_commit": design["code_commit"],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": report._sha256(manifest_path),
            "retrain_index": index,
            "row": retrain_row,
            "row_sha256": report._json_sha256(retrain_row),
            "experiment_name": experiment_name,
            "run_name": run_name,
            "hydra_dir": str((root / "retraining" / "hydra" / f"{index:04d}").resolve()),
            "trainer_default_root_dir": str((state_dir / "lightning").resolve()),
            "mlflow_tracking_uri": (
                f"file:{(root / 'runtime' / 'logs' / 'mlflow' / 'mlruns').resolve()}"
            ),
        }
        state_path = state_dir / "state.json"
        _json(
            state_path,
            {
                "schema_version": 1,
                "identity": identity,
                "mlflow_run_id": mlflow_run_id,
                "attempts": attempts,
                "segments": segment_rows,
                "completed_at": "2026-01-02T00:00:00+00:00",
            },
        )
        marker_path = root / "retraining" / "markers" / f"{index:04d}.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.update(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "mlflow_run_id": mlflow_run_id,
                "restart_count": len(attempts) - 1,
                "restart_state_artifact": str(state_path.resolve()),
                "restart_state_artifact_sha256": report._sha256(state_path),
            }
        )
        _json(marker_path, marker)
    _json(
        root / "results" / "oracle_winners.json",
        {
            "schema_version": 2,
            "created_at": "2026-01-03T00:00:00+00:00",
            "oracle_uses_downstream_anomalies": True,
            "winners": winners,
            "all_front_memberships": winners,
        },
    )


def test_finalize_exports_56_winners_and_reverse_drift(tmp_path) -> None:
    """Finalization enforces the full logical matrix and portable row counts."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    output = root / "results"
    (output / "stale.txt").write_text("must disappear\n", encoding="utf-8")
    repository_report = tmp_path / "repository" / "REPORT.md"

    report.finalize(root, output, repository_report=repository_report)

    assert not (output / "stale.txt").exists()
    assert repository_report.read_bytes() == (output / "REPORT.md").read_bytes()
    with (output / "winners.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56
    with (output / "downstream_efficiencies.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56 * 20
    with (output / "primary_metrics.csv").open(encoding="utf-8") as handle:
        metrics = list(report.csv.DictReader(handle))
    assert sum(row["metric_name"] == "reverse_drift_posthoc" for row in metrics) == 8
    assert sum(row["metric_name"] == "rankcorr_test" for row in metrics) == 40
    with (output / "checkpoint_candidates.csv").open(encoding="utf-8") as handle:
        candidate_rows = list(report.csv.DictReader(handle))
    assert len(candidate_rows) == 68
    assert sum(row["available"] == "True" for row in candidate_rows) == 67
    with (output / "checkpoint_candidate_availability.csv").open(encoding="utf-8") as handle:
        availability_rows = list(report.csv.DictReader(handle))
    assert len(availability_rows) == 18
    assert sum(int(row["configured"]) for row in availability_rows) == 68
    analysis_summary = json.loads(
        (output / "analysis" / "analysis_summary.json").read_text(encoding="utf-8")
    )
    assert analysis_summary["fronts"] == 56
    assert analysis_summary["front_candidates"] == 56
    with (output / "artifact_audit" / "checkpoint_families.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56
    with (output / "artifact_audit" / "threshold_audit.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 8
    with (output / "artifact_audit" / "metric_summary_files.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56
    assert (output / "analysis" / "mapping_comparisons.csv").is_file()
    assert (output / "analysis" / "proxy_associations.csv").is_file()
    manifest_lines = (output / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    assert all(not line.endswith("  manifest.sha256") for line in manifest_lines)
    assert all("repository" not in line for line in manifest_lines)
    assert (output / "REPORT.md").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 4
    with (output / "configs" / "index.csv").open(encoding="utf-8") as handle:
        config_rows = list(report.csv.DictReader(handle))
    assert {row["phase"] for row in config_rows} == {"search", "retrain"}
    assert all((output / row["portable_path"]).is_file() for row in config_rows)
    with (output / "logs" / "runs" / "index.csv").open(encoding="utf-8") as handle:
        log_rows = list(report.csv.DictReader(handle))
    assert {row["phase"] for row in log_rows} == {"search", "retrain"}
    assert all((output / row["portable_path"]).is_file() for row in log_rows)
    with (output / "logs" / "slurm" / "index.csv").open(encoding="utf-8") as handle:
        slurm_rows = list(report.csv.DictReader(handle))
    assert len(slurm_rows) == 1
    assert all((output / row["portable_path"]).is_file() for row in slurm_rows)
    with (output / "mlflow_runs.csv").open(encoding="utf-8") as handle:
        mlflow_rows = list(report.csv.DictReader(handle))
    assert len(mlflow_rows) == 12
    assert {row["status"] for row in mlflow_rows} == {"FINISHED"}
    assert {row["histories_validated"] for row in mlflow_rows} == {"True"}
    with (output / "retrain_restart_evidence.csv").open(encoding="utf-8") as handle:
        restart_rows = list(report.csv.DictReader(handle))
    assert len(restart_rows) == 30
    assert {int(row["epoch_boundary"]) for row in restart_rows} == {40, 80, 120, 160, 200}
    with (output / "retrains.csv").open(encoding="utf-8") as handle:
        retrain_rows = list(report.csv.DictReader(handle))
    assert {row["terminal_phase"] for row in retrain_rows} == {"evaluation"}
    assert {int(row["attempt_count"]) for row in retrain_rows} == {6}
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_code_commit"] == design["code_commit"]
    assert provenance["execution_repository_commit"] == design["code_commit"]
    assert provenance["execution_repository_clean"] is True
    assert provenance["execution_uv_lock_sha256"] == "d" * 64
    assert provenance["report_code_commit"] == "b" * 40
    assert provenance["report_repository_root"] == str(report.REPOSITORY_ROOT.resolve())
    assert provenance["report_uv_lock_sha256"] == report._sha256(
        report.REPOSITORY_ROOT / "uv.lock"
    )
    assert "uv_lock_sha256" not in provenance

    before = report._tree_digests(output)
    manifest_before = (output / "manifest.sha256").read_bytes()
    report.finalize(root, output, repository_report=repository_report)
    assert report._tree_digests(output) == before
    assert (output / "manifest.sha256").read_bytes() == manifest_before


def test_finalize_rejects_changed_retrain_artifact(tmp_path) -> None:
    """A marker hash mismatch stops final reporting rather than hiding corruption."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    artifact = root / "retraining" / "hydra" / "0000" / "optimized_metric.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Changed optimized artifact"):
        report.finalize(root, root / "portable")


def test_report_rejects_changed_or_escaped_restart_state(tmp_path) -> None:
    """The marker cannot redirect or conceal changes to its restart-state artifact."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    manifest = json.loads(
        (root / "selection" / "pareto_retrain_manifest.json").read_text(encoding="utf-8")
    )
    state = root / "retraining" / "restart_state" / "0000" / "state.json"
    state.write_text(state.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Changed retrain restart state"):
        report._marker_rows(root, manifest)

    root, design = _campaign(tmp_path / "escaped")
    _frozen_results(root, design)
    manifest = json.loads(
        (root / "selection" / "pareto_retrain_manifest.json").read_text(encoding="utf-8")
    )
    marker_path = root / "retraining" / "markers" / "0000.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    external = tmp_path / "external-state.json"
    external.write_text("{}\n", encoding="utf-8")
    marker["restart_state_artifact"] = str(external)
    marker["restart_state_artifact_sha256"] = report._sha256(external)
    _json(marker_path, marker)
    with pytest.raises(RuntimeError, match="outside its authenticated directory"):
        report._marker_rows(root, manifest)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("identity", "restart identity differs"),
        ("boundaries", "segment boundaries are incomplete"),
        ("segment_hash", "Changed retrain segment artifact"),
        ("rng_state", "lacks exact RNG state"),
        ("terminal_attempt", "terminal attempt is incomplete"),
    ),
)
def test_report_reconstructs_restart_provenance(tmp_path, corruption: str, message: str) -> None:
    """Rehashing a semantically false state does not make it valid for reporting."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    manifest = json.loads(
        (root / "selection" / "pareto_retrain_manifest.json").read_text(encoding="utf-8")
    )
    state_path = root / "retraining" / "restart_state" / "0000" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if corruption == "identity":
        state["identity"]["row_sha256"] = "0" * 64
    elif corruption == "boundaries":
        state["segments"].pop()
    elif corruption == "segment_hash":
        state["segments"][0]["continuation_checkpoint_sha256"] = "0" * 64
    elif corruption == "rng_state":
        checkpoint = Path(state["segments"][0]["continuation_checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["callbacks"].pop(report.EXACT_RETRAIN_RNG_STATE_KEY)
        torch.save(payload, checkpoint)
        state["segments"][0]["continuation_checkpoint_sha256"] = report._sha256(checkpoint)
    else:
        state["attempts"][-1]["phase"] = "training"
    _json(state_path, state)
    marker_path = root / "retraining" / "markers" / "0000.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["restart_state_artifact_sha256"] = report._sha256(state_path)
    _json(marker_path, marker)

    with pytest.raises(RuntimeError, match=message):
        report._marker_rows(root, manifest)


def test_finalize_rejects_missing_checkpoint_threshold(tmp_path) -> None:
    """Final publication fails closed when a selected checkpoint lacks its OAS threshold."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    checkpoint = root / "runtime" / "checkpoints" / "ae" / "last.ckpt"
    torch.save({"state_dict": {"thres_operational": torch.tensor(0.4)}}, checkpoint)

    with pytest.raises(ValueError, match="Threshold buffer keys differ"):
        report.finalize(root, root / "portable")


@pytest.mark.parametrize("corruption", ("empty_jobs", "missing_failures", "log_hash"))
def test_final_ledgers_fail_closed(tmp_path, corruption: str) -> None:
    """A final package never invents ledgers or silently omits recorded Slurm evidence."""
    root, design = _campaign(tmp_path)
    jobs = root / "state" / "job_history.csv"
    if corruption == "empty_jobs":
        jobs.write_text(jobs.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    elif corruption == "missing_failures":
        (root / "state" / "failures.jsonl").unlink()
    else:
        path = Path(jobs.read_text(encoding="utf-8").splitlines()[1].split(",")[-2])
        path.write_text("changed after recording\n", encoding="utf-8")

    with pytest.raises((RuntimeError, FileNotFoundError)):
        report._validated_state_ledgers(root, design, final=True)


def test_dirty_reporting_checkout_is_rejected_before_publication(tmp_path, monkeypatch) -> None:
    """Execution and reporting commits may differ, but reporting must begin clean."""
    root, _ = _campaign(tmp_path)
    output = root / "portable"
    monkeypatch.setattr(report, "_git_state", lambda: ("c" * 40, False, [" M report.py"]))

    with pytest.raises(RuntimeError, match="clean checkout"):
        report.finalize(root, output)
    assert not output.exists()


def test_staged_failure_preserves_previous_canonical_output(tmp_path, monkeypatch) -> None:
    """No build or source-stability failure can expose a partial replacement."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    output = root / "portable"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("previous complete package\n", encoding="utf-8")

    def fail_build(*args, **kwargs):
        raise RuntimeError("injected staged build failure")

    monkeypatch.setattr(report, "_build_final", fail_build)
    with pytest.raises(RuntimeError, match="injected staged build failure"):
        report.finalize(root, output)

    assert sentinel.read_text(encoding="utf-8") == "previous complete package\n"
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_publish_failure_restores_previous_directory(tmp_path, monkeypatch) -> None:
    """A failed stage rename restores the prior canonical directory from backup."""
    output = tmp_path / "results"
    stage = tmp_path / ".results.staging-test"
    output.mkdir()
    stage.mkdir()
    (output / "old.txt").write_text("old\n", encoding="utf-8")
    (stage / "new.txt").write_text("new\n", encoding="utf-8")
    original_rename = Path.rename

    def fail_stage_rename(path: Path, target: Path):
        if path == stage:
            raise OSError("injected rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage_rename)
    with pytest.raises(OSError, match="injected rename failure"):
        report._publish_staged(stage, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert stage.is_dir()


def test_finalize_rejects_tampered_pareto_membership(tmp_path) -> None:
    """Finalization reconstructs fronts instead of trusting the frozen manifest."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["front_memberships"][0]["objective_values"][0] = 999.0
    _json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="differ from reconstructed Optuna fronts"):
        report.finalize(root, root / "portable")


def test_finalize_rejects_tampered_front_member_winner(tmp_path) -> None:
    """A frozen front member is not valid unless its retrain selection is authentic."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    oracle_path = root / "results" / "oracle_winners.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle["winners"][0]["optimized_ckpt_name"] = "tampered"
    _json(oracle_path, oracle)

    with pytest.raises(RuntimeError, match="deterministic downstream argmax"):
        report.finalize(root, root / "portable")


def test_finalize_rejects_tampered_oracle_candidate_efficiency(tmp_path) -> None:
    """Every candidate efficiency must be reconstructed from hashed marker CSVs."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    oracle_path = root / "results" / "oracle_winners.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    anomaly = report.DOWNSTREAM_DATASETS[0]
    oracle["all_front_memberships"][0]["downstream_efficiencies"][anomaly] = 0.9
    _json(oracle_path, oracle)

    with pytest.raises(RuntimeError, match="efficiencies differ from marker"):
        report.finalize(root, root / "portable")


def test_oracle_tie_break_uses_lowest_membership_index() -> None:
    """Exact downstream-mean ties reproduce aggregate's stable manifest order."""
    candidates = [
        {"membership_index": 9, "mean_downstream_efficiency": 0.5},
        {"membership_index": 3, "mean_downstream_efficiency": 0.5},
    ]

    assert report._deterministic_oracle_winner(candidates)["membership_index"] == 3


def test_pickle_reader_rejects_path_outside_authenticated_root(tmp_path) -> None:
    """Reporter pickle loading is confined to authenticated checkpoint trees."""
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    external = tmp_path / "external.pkl"
    with external.open("wb") as handle:
        pickle.dump({}, handle)

    with pytest.raises(RuntimeError, match="outside trusted roots"):
        report._read_pickle_objects(external, [trusted])
