"""Tests for the full-node packed Hydra launcher."""

from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from omegaconf import OmegaConf

from hydra_plugins.physics_packed_srun_launcher import launcher as packed


def test_hydra_accepts_launcher_from_plugin_namespace() -> None:
    """Hydra rejects launchers outside its required top-level namespace."""
    config = OmegaConf.create(
        {"_target_": ("hydra_plugins.physics_packed_srun_launcher.launcher." "PackedSrunLauncher")}
    )
    assert isinstance(Plugins.instance()._instantiate(config), packed.PackedSrunLauncher)


def test_allocation_capacity_counts_all_visible_node_gpus(monkeypatch) -> None:
    """Capacity is the product of allocated nodes and GPUs per node."""
    monkeypatch.setenv("SLURM_NNODES", "12")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    assert packed._allocation_capacity() == 48


def test_allocation_capacity_requires_a_gpu_slurm_job(monkeypatch) -> None:
    """The launcher fails closed outside a GPU allocation."""
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    with pytest.raises(RuntimeError, match="GPU Slurm allocation"):
        packed._allocation_capacity()


def test_launcher_preserves_job_order_and_exact_step_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Concurrent step completion is returned in Optuna ask order."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    commands = []

    class FinishedProcess:
        """Create one synthetic worker result when the step is launched."""

        def __init__(self, command, cwd):
            commands.append((command, cwd))
            payload_path = Path(command[-2])
            result_path = Path(command[-1])
            with payload_path.open("rb") as handle:
                payload = cloudpickle.load(handle)
            result = JobReturn(
                overrides=payload["overrides"],
                status=JobStatus.COMPLETED,
            )
            result.return_value = payload["job_num"]
            with result_path.open("wb") as handle:
                cloudpickle.dump(result, handle)

        def wait(self):
            return 0

    monkeypatch.setattr(packed.subprocess, "Popen", FinishedProcess)
    launcher = packed.PackedSrunLauncher(cpus_per_task=16, mem_gb=96)
    launcher.setup(
        hydra_context=SimpleNamespace(),
        task_function=lambda _: None,
        config=OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}}),
    )
    results = launcher.launch([["x=0"], ["x=1"], ["x=2"], ["x=3"]], 7)

    assert [result.return_value for result in results] == [7, 8, 9, 10]
    assert len(commands) == 4
    for command, cwd in commands:
        assert cwd == Path.cwd()
        assert "--exclusive" in command
        assert "--exact" in command
        assert "--cpus-per-task=16" in command
        assert "--gpus-per-node=1" in command
        assert "--mem=96G" in command


def test_launcher_rejects_more_jobs_than_allocated_gpus(tmp_path, monkeypatch) -> None:
    """An undersized outer allocation cannot silently serialize a batch."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    launcher = packed.PackedSrunLauncher()
    launcher.setup(
        hydra_context=SimpleNamespace(),
        task_function=lambda _: None,
        config=OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}}),
    )
    with pytest.raises(RuntimeError, match="only 4 GPUs"):
        launcher.launch([[f"x={index}"] for index in range(5)], 0)
