"""The per-run manifest that links the four pipeline stages together.

Stage 1 writes one of these into every run's own checkpoint directory:

    checkpoints/<experiment_name>/<run_name>/run_manifest.yaml

It records what was trained -- the grid point, the seed, the architecture, a
fingerprint of the resolved algorithm config -- so that the later stages do not
have to be told again, and so that an experiment directory is self-describing:
drop however many trained runs into it, whenever and from wherever they were
produced, and stage 4 can read the directory rather than a pre-declared plan.

Why one file per run instead of one file per experiment: many runs finish at
once (32 HTCondor shards, say) and appending to a shared file on EOS has no
locking anyone can rely on, so concurrent appends interleave and corrupt it.
Per-run files never collide, they travel with the checkpoint through
``transfer_output_files``, and assembling them is a glob.

Vocabulary, fixed here so the rest of the code can stop guessing:

    configuration   a point on the Pareto grid: (mi_gamma, mi_sensitive_num_bins,
                    architecture). Identified by ``configuration_id``.
    run             one training of an autoencoder at that configuration with a
                    particular seed. Identified by ``run_name``.

Several runs share a configuration when they differ only by seed; that is what
lets the Phase 2 aggregation compute a mean and a confidence interval.

The manifest is written AFTER training, not before. ClearRunCheckpointDir wipes
the run directory when a fit starts, so anything written earlier would not
survive; and a manifest present is then a truthful claim that stage 1 finished.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import os
import socket

from omegaconf import DictConfig, OmegaConf

from src.utils import RankedLogger
from src.evaluation.leakage_probe.provenance import leakage_probe_configuration_id

log = RankedLogger(__name__, rank_zero_only=True)

#: Bump when a field changes meaning, never for an addition.
RUN_MANIFEST_SCHEMA_VERSION = 1

RUN_MANIFEST_FILENAME = "run_manifest.yaml"

#: Stage 1 also drops a copy of the fully resolved Hydra config here. Hydra's own
#: copy lives in the Hydra run directory, which for an ad-hoc run is a timestamped
#: folder under logs/ that does not travel with the checkpoint.
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

#: Stages 2 and 3 may run concurrently, so each records its own completion in its
#: own file rather than updating the shared manifest.
STAGE_STATUS_DIRNAME = "stage_status"


class RunManifestError(RuntimeError):
    """Raised when a manifest is missing, malformed, or describes another run."""


def run_manifest_path(run_ckpts: Path) -> Path:
    return Path(run_ckpts) / RUN_MANIFEST_FILENAME


def _grid_configuration(cfg: DictConfig) -> Dict[str, Any]:
    """Return the Pareto grid coordinates this run was trained at.

    A Pareto-study run states them explicitly under ``pareto_study.candidate``.
    An ad-hoc run trained through runae.sh has no such block, so the coordinates
    are read from the algorithm config itself -- they are the same numbers, just
    not spelled out twice.
    """
    algorithm = cfg.get("algorithm")
    if algorithm is None:
        raise RunManifestError("Cannot describe a run with no algorithm config.")

    encoder = algorithm.get("encoder")
    encoder_nodes = (
        [int(node) for node in encoder.get("nodes")]
        if encoder is not None and encoder.get("nodes") is not None
        else None
    )

    study = cfg.get("pareto_study")
    candidate = study.get("candidate") if study is not None else None

    if candidate is not None:
        architecture_id = candidate.get("architecture_id")
        if candidate.get("encoder_nodes") is not None:
            encoder_nodes = [int(node) for node in candidate.get("encoder_nodes")]
    else:
        architecture_id = None

    if not architecture_id:
        # An ad-hoc run has no study grid, so label the architecture by its
        # encoder shape. Storing the derived value rather than null keeps the
        # manifest self-describing: the pairing downstream compares this field,
        # and a null there would make every run its own architecture.
        architecture_id = (
            "h" + "_".join(str(node) for node in encoder_nodes)
            if encoder_nodes
            else "unknown"
        )

    return {
        "mi_gamma": float(algorithm.get("mi_gamma")),
        "mi_sensitive_num_bins": int(algorithm.get("mi_sensitive_num_bins")),
        "mi_sensitive_variable": str(algorithm.get("mi_sensitive_variable")),
        "architecture_id": architecture_id,
        "encoder_nodes": encoder_nodes,
    }


def _format_number(value: float) -> str:
    """Render a grid coordinate so that 0.1 and 0.10 cannot become two ids."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def derive_configuration_id(cfg: DictConfig, configuration: Mapping[str, Any]) -> str:
    """Return the identifier of the grid point this run sits on.

    A Pareto-study run already carries one, assigned by the study runner, and it
    is used verbatim -- Phase 2 compares it against the study map, so deriving a
    different string here would reject every run.

    For an ad-hoc run the id is built from the coordinates themselves. That is
    deliberate: two runs that differ only by seed then land on the same
    configuration automatically and can be aggregated as paired seeds, which is
    the whole point of separating the two words.
    """
    study = cfg.get("pareto_study")
    if study is not None and study.get("configuration_id"):
        return str(study.get("configuration_id"))

    architecture = configuration.get("architecture_id")
    if not architecture:
        nodes = configuration.get("encoder_nodes") or []
        architecture = "h" + "_".join(str(node) for node in nodes) if nodes else "unknown"

    return (
        f"gamma-{_format_number(configuration['mi_gamma'])}"
        f"__bins-{configuration['mi_sensitive_num_bins']}"
        f"__arch-{architecture}"
    )


def write_run_manifest(
    cfg: DictConfig,
    run_ckpts: Path,
    *,
    mlflow_run_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the per-run manifest and a resolved copy of the config beside it."""
    run_ckpts = Path(run_ckpts)
    run_ckpts.mkdir(parents=True, exist_ok=True)

    configuration = _grid_configuration(cfg)
    configuration_id = derive_configuration_id(cfg, configuration)

    resolved_path = run_ckpts / RESOLVED_CONFIG_FILENAME
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    OmegaConf.save(config=OmegaConf.create(resolved_cfg), f=resolved_path, resolve=True)

    study = cfg.get("pareto_study")

    manifest: Dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "written_by_stage": "train",
        "hostname": socket.gethostname(),
        # Identity. experiment_name + run_name is the address of this run;
        # configuration_id is the grid point several runs may share.
        "experiment_name": str(cfg.experiment_name),
        "run_name": str(cfg.run_name),
        "configuration_id": configuration_id,
        "autoencoder_seed": int(cfg.seed) if cfg.get("seed") is not None else None,
        # The grid point itself, spelled out so a human can read the directory.
        "configuration": configuration,
        # What the later stages check themselves against. A strict load_state_dict
        # only catches a shape change; this catches a different gamma, a different
        # bin count or a different loss setting, none of which change the shapes.
        "algorithm_fingerprint": leakage_probe_configuration_id(cfg.algorithm),
        "paths": {
            "checkpoint_run_dir": str(run_ckpts),
            "resolved_config": str(resolved_path),
            "hydra_output_dir": str(cfg.paths.output_dir),
            "checkpoints_dir": str(cfg.paths.checkpoints_dir),
        },
        "pareto_study_enabled": bool(study is not None and study.get("enabled", False)),
        "mlflow": {
            "experiment_name": str(cfg.experiment_name),
            "run_id": mlflow_run_id,
        },
    }
    if extra:
        manifest.update(dict(extra))

    path = run_manifest_path(run_ckpts)
    _atomic_save(manifest, path)
    log.info(
        f"Wrote run manifest {path} "
        f"(configuration_id={configuration_id}, seed={manifest['autoencoder_seed']})"
    )
    return path


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write via a temporary file so a reader never sees a half-written manifest."""
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    OmegaConf.save(config=OmegaConf.create(dict(payload)), f=tmp, resolve=True)
    os.replace(tmp, path)


def read_run_manifest(run_ckpts: Path) -> Dict[str, Any]:
    """Load the manifest stage 1 wrote for this run."""
    path = run_manifest_path(run_ckpts)
    if not path.is_file():
        raise RunManifestError(
            f"Missing {RUN_MANIFEST_FILENAME} in {run_ckpts}. It is written at the "
            "end of stage 1 (src/train.py); a run trained before this file existed "
            "will not have one."
        )
    loaded = OmegaConf.load(path)
    return OmegaConf.to_container(loaded, resolve=True)


def verify_against_manifest(
    cfg: DictConfig,
    run_ckpts: Path,
    *,
    stage_name: str,
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """Refuse to analyse a checkpoint the composed config does not describe.

    The failure this prevents is quiet rather than loud. ``load_state_dict`` is
    called with ``strict=True`` downstream, but it only compares tensor shapes:
    a stage-2 run composed with the wrong ``mi_gamma`` loads perfectly and then
    records a probe score against the wrong grid point. The fingerprint covers
    the whole resolved algorithm config, so it catches exactly that case.

    ``strict=False`` downgrades a mismatch to a warning, for the runs trained
    before manifests existed.
    """
    try:
        manifest = read_run_manifest(run_ckpts)
    except RunManifestError as error:
        if strict:
            raise
        log.warning(f"[{stage_name}] {error} Continuing without verification.")
        return None

    expected = manifest.get("algorithm_fingerprint")
    actual = leakage_probe_configuration_id(cfg.algorithm)

    if expected and expected != actual:
        message = (
            f"[{stage_name}] This config does not describe the checkpoint in "
            f"{run_ckpts}.\n"
            f"  manifest fingerprint: {expected}\n"
            f"  composed fingerprint: {actual}\n"
            f"  manifest configuration: {manifest.get('configuration')}\n"
            "Compose the same experiment and the same algorithm overrides as the "
            "stage-1 run. Re-run with the hyperparameters recorded in "
            f"{run_manifest_path(run_ckpts)}."
        )
        if strict:
            raise RunManifestError(message)
        log.warning(message)
        return manifest

    log.info(
        f"[{stage_name}] Manifest verified: configuration_id="
        f"{manifest.get('configuration_id')}, seed={manifest.get('autoencoder_seed')}"
    )
    return manifest


def write_stage_status(
    run_ckpts: Path,
    *,
    stage_name: str,
    ok: bool,
    detail: Optional[str] = None,
    artifacts: Optional[Sequence[str]] = None,
) -> Path:
    """Record that a stage finished, in a file only that stage writes.

    Stages 2 and 3 are independent and may run at the same time, so neither may
    update the shared manifest: a read-modify-write from two processes loses one
    of the updates, and on EOS there is no lock to prevent it. One file per stage
    has no such interaction, and stage 4 reads them together.
    """
    status_dir = Path(run_ckpts) / STAGE_STATUS_DIRNAME
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{stage_name}.yaml"
    _atomic_save(
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "stage": stage_name,
            "ok": bool(ok),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "detail": detail,
            "artifacts": list(artifacts) if artifacts else [],
        },
        path,
    )
    log.info(f"[{stage_name}] Recorded stage status at {path} (ok={ok}).")
    return path
