"""Persist the immutable Phase 0 Pareto-study manifest for a run."""

from pathlib import Path

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf


@rank_zero_only
def write_resolved_pareto_manifest(cfg: DictConfig) -> Path | None:
    """Write the full resolved config for enabled Pareto-study runs.

    Hydra retains its ordinary composition under ``.hydra/config.yaml``, where
    interpolations may remain visible.  The scientific study also needs one
    self-contained, resolved manifest so its frozen selection policy can be
    recovered without the original config tree.
    """
    study = cfg.get("pareto_study")
    if study is None or not bool(study.get("enabled", False)):
        return None

    manifest_cfg = study.get("resolved_manifest")
    if manifest_cfg is None:
        raise ValueError(
            "Pareto studies must declare pareto_study.resolved_manifest."
        )

    filename = manifest_cfg.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            "pareto_study.resolved_manifest.filename must be a non-empty string."
        )

    output_dir = Path(str(cfg.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Resolve once before writing.  ``throw_on_missing`` prevents a reportable
    # run from silently recording an incomplete study manifest.
    resolved_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    OmegaConf.save(
        config=OmegaConf.create(resolved_cfg),
        f=output_path,
        resolve=True,
    )
    return output_path
