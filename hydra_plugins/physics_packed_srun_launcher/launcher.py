"""Hydra launcher for independent single-GPU jobs inside one Slurm allocation."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict

log = logging.getLogger(__name__)


def _cloudpickle() -> Any:
    """Import the serializer already provided by Submitit."""
    import cloudpickle

    return cloudpickle


def _allocation_capacity() -> int:
    """Return the number of GPUs exposed by the enclosing Slurm allocation."""
    nodes = int(os.environ.get("SLURM_NNODES", "0"))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))
    if nodes <= 0 or gpus_per_node <= 0:
        raise RuntimeError("The packed launcher requires a GPU Slurm allocation.")
    return nodes * gpus_per_node


def _failed_return(overrides: Sequence[str], error: Exception) -> JobReturn:
    """Build a Hydra failure result without losing its override identity."""
    result = JobReturn(overrides=list(overrides), status=JobStatus.FAILED)
    result.return_value = error
    return result


def _run_payload(payload_path: Path, result_path: Path) -> None:
    """Execute one serialized Hydra job inside an isolated Slurm step."""
    serializer = _cloudpickle()
    with payload_path.open("rb") as handle:
        payload = serializer.load(handle)

    Singleton.set_state(payload["singleton_state"])
    setup_globals()
    hydra_context = payload["hydra_context"]
    base_config = payload["config"]
    task_function = payload["task_function"]
    overrides = payload["overrides"]
    job_num = int(payload["job_num"])
    sweep_config = hydra_context.config_loader.load_sweep_config(base_config, overrides)
    with open_dict(sweep_config.hydra.job) as job:
        slurm_job = os.environ.get("SLURM_JOB_ID", "unknown")
        slurm_step = os.environ.get("SLURM_STEP_ID", str(job_num))
        job.id = f"{slurm_job}.{slurm_step}"
        job.num = job_num

    result = run_job(
        hydra_context=hydra_context,
        task_function=task_function,
        config=sweep_config,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        serializer.dump(result, handle)
    os.replace(temporary, result_path)


class PackedSrunLauncher(Launcher):
    """Run Hydra jobs concurrently as exclusive one-GPU Slurm steps."""

    def __init__(
        self,
        *,
        cpus_per_task: int = 16,
        mem_gb: int = 96,
        gpus_per_task: int = 1,
    ) -> None:
        if cpus_per_task <= 0 or mem_gb <= 0 or gpus_per_task != 1:
            raise ValueError("Packed jobs require positive CPU/memory and exactly one GPU.")
        self.cpus_per_task = int(cpus_per_task)
        self.mem_gb = int(mem_gb)
        self.gpus_per_task = int(gpus_per_task)
        self.config: DictConfig | None = None
        self.task_function: TaskFunction | None = None
        self.hydra_context: HydraContext | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Retain the controller state serialized into each worker payload."""
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.config = config

    def _command(
        self,
        payload_path: Path,
        result_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> list[str]:
        """Build one resource-exact nested Slurm step command."""
        return [
            "srun",
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--gpus-per-node={self.gpus_per_task}",
            f"--mem={self.mem_gb}G",
            "--unbuffered",
            f"--output={stdout_path}",
            f"--error={stderr_path}",
            sys.executable,
            "-m",
            "hydra_plugins.physics_packed_srun_launcher.launcher",
            "--worker",
            str(payload_path),
            str(result_path),
        ]

    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[JobReturn]:
        """Launch one ordered Optuna batch and return results in trial order."""
        if self.config is None or self.hydra_context is None or self.task_function is None:
            raise RuntimeError("PackedSrunLauncher.setup must run before launch.")
        if not job_overrides:
            raise ValueError("At least one Hydra job override is required.")
        capacity = _allocation_capacity()
        if len(job_overrides) > capacity:
            raise RuntimeError(
                f"Packed batch has {len(job_overrides)} jobs but allocation exposes "
                f"only {capacity} GPUs."
            )

        sweep_dir = Path(str(self.config.hydra.sweep.dir)).resolve()
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is None:
            raise RuntimeError("SLURM_JOB_ID is required for packed-launch provenance.")
        exchange_dir = sweep_dir / ".packed_srun" / slurm_job_id
        exchange_dir.mkdir(parents=True, exist_ok=True)
        serializer = _cloudpickle()
        processes: list[tuple[subprocess.Popen[bytes], int, Sequence[str], Path]] = []

        for offset, overrides in enumerate(job_overrides):
            job_num = initial_job_idx + offset
            log.info("\t#%d : %s", job_num, " ".join(overrides))
            prefix = exchange_dir / f"job_{job_num}"
            payload_path = prefix.with_suffix(".payload.pkl")
            result_path = prefix.with_suffix(".result.pkl")
            with payload_path.open("wb") as handle:
                serializer.dump(
                    {
                        "hydra_context": self.hydra_context,
                        "task_function": self.task_function,
                        "config": self.config,
                        "overrides": list(overrides),
                        "job_num": job_num,
                        "singleton_state": Singleton.get_state(),
                    },
                    handle,
                )
            command = self._command(
                payload_path,
                result_path,
                prefix.with_suffix(".out"),
                prefix.with_suffix(".err"),
            )
            process = subprocess.Popen(command, cwd=Path.cwd())  # nosec B603
            processes.append((process, job_num, overrides, result_path))

        results: list[JobReturn] = []
        for process, job_num, overrides, result_path in processes:
            return_code = process.wait()
            if return_code != 0 or not result_path.is_file():
                results.append(
                    _failed_return(
                        overrides,
                        RuntimeError(
                            f"Packed Slurm step for Hydra job {job_num} failed with "
                            f"exit code {return_code}; inspect {result_path.parent}."
                        ),
                    )
                )
                continue
            with result_path.open("rb") as handle:
                results.append(serializer.load(handle))
        return results


def _parse_worker_args() -> argparse.Namespace:
    """Parse the private worker entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("payload", type=Path)
    parser.add_argument("result", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_worker_args()
    if not args.worker:
        raise SystemExit("This module is invoked only by PackedSrunLauncher.")
    _run_payload(args.payload, args.result)
