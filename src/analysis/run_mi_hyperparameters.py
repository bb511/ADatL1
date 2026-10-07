"""Read the MI hyperparameters of a finished run from what is on disk.

The correlation-matrix plots show the MI weight γ, the requested number of
FET.Et bins and the effective number of bins (the quantile binner merges
coinciding edges, so e.g. 50 requested bins give 48). During evaluation the
callback reads them from the model; redrawing old plots needs them from the
checkpoint folder instead. Sources, first hit wins:

* γ and requested bins: ``<run>/resolved_config.yaml`` (``algorithm.mi_gamma``,
  ``algorithm.mi_sensitive_num_bins``), else the MLflow params of the run named
  in ``<run>/links.yaml``.
* Effective bins: the latest ``<run>/plots/mi_diagnostics/data/epoch_*/
  mi_bin_widths_epoch*.csv``, else the last ``[MI] Effective bins: N`` line of the
  Hydra ``train.log`` files named in ``<run>/links.yaml``.

Missing values stay ``None``; nothing is guessed. Only numpy/pandas/yaml are
needed, no torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from src.analysis.decorrelation import load_effective_bin_count


REPO_ROOT = Path(__file__).resolve().parents[2]
EFFECTIVE_BINS_LOG = re.compile(r"\[MI\] Effective bins: (\d+)")


@dataclass(frozen=True)
class MiHyperparameters:
    """MI settings of one run; ``None`` means not recorded."""

    gamma: Optional[float] = None
    requested_bins: Optional[int] = None
    effective_bins: Optional[int] = None

    @property
    def known(self) -> bool:
        return any(v is not None for v in (self.gamma, self.requested_bins, self.effective_bins))

    def text(self) -> str:
        """One line for a plot subtitle."""

        def show(value: Any, spec: str = "") -> str:
            return "n/a" if value is None else format(value, spec)

        return (
            f"MI: γ = {show(self.gamma, 'g')} · requested bins = {show(self.requested_bins)}"
            f" · effective bins = {show(self.effective_bins)}"
        )


def run_dir_of(path: Path) -> Optional[Path]:
    """Checkpoint run folder that holds ``path`` (the parent of its ``plots`` folder)."""
    parts = Path(path).parts
    if "plots" not in parts:
        return None
    return Path(*parts[: len(parts) - 1 - parts[::-1].index("plots")])


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _as_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _from_resolved_config(run_dir: Path) -> tuple[Optional[float], Optional[int]]:
    algorithm = _read_yaml(run_dir / "resolved_config.yaml").get("algorithm") or {}
    return _as_float(algorithm.get("mi_gamma")), _as_int(algorithm.get("mi_sensitive_num_bins"))


def _mlflow_param_dirs(links: dict, repo_root: Path) -> Iterable[Path]:
    mlflow = links.get("mlflow") or {}
    experiment, run_id = mlflow.get("experiment_id"), mlflow.get("run_id")
    if not experiment or not run_id:
        return []
    roots = [repo_root / "logs" / "mlflow" / "mlruns"]
    if mlflow.get("tracking_dir"):
        roots.append(Path(mlflow["tracking_dir"]) / "mlruns")
    return [root / str(experiment) / str(run_id) / "params" for root in roots]


def _from_mlflow(links: dict, repo_root: Path) -> tuple[Optional[float], Optional[int]]:
    for params in _mlflow_param_dirs(links, repo_root):
        if not params.is_dir():
            continue

        def read(name: str) -> Optional[str]:
            for candidate in (params / "algorithm" / name, params / name):
                if candidate.is_file():
                    return candidate.read_text().strip()
            return None

        return _as_float(read("mi_gamma")), _as_int(read("mi_sensitive_num_bins"))
    return None, None


def _effective_from_logs(links: dict, repo_root: Path) -> Optional[int]:
    hydra = links.get("hydra") or {}
    for entry in [hydra.get("train"), *(hydra.get("other") or [])]:
        if not entry:
            continue
        log = Path(entry) if Path(entry).is_absolute() else repo_root / entry
        log = log / "train.log"
        if log.is_file():
            hits = EFFECTIVE_BINS_LOG.findall(log.read_text(errors="ignore"))
            if hits:
                return int(hits[-1])
    return None


def read_mi_hyperparameters(run_dir: Path, repo_root: Path = REPO_ROOT) -> MiHyperparameters:
    """MI hyperparameters of the checkpoint run folder ``run_dir``."""
    run_dir = Path(run_dir)
    links = _read_yaml(run_dir / "links.yaml")

    gamma, requested = _from_resolved_config(run_dir)
    if gamma is None or requested is None:
        mlflow_gamma, mlflow_requested = _from_mlflow(links, Path(repo_root))
        gamma = gamma if gamma is not None else mlflow_gamma
        requested = requested if requested is not None else mlflow_requested

    try:
        effective: Optional[int] = load_effective_bin_count(run_dir)
    except (FileNotFoundError, ValueError):
        effective = _effective_from_logs(links, Path(repo_root))

    return MiHyperparameters(gamma=gamma, requested_bins=requested, effective_bins=effective)
