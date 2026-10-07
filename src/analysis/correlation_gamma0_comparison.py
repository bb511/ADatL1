"""Compare each run's reconstruction correlations with those of its γ = 0 run.

Post-processing (stage 4) step: on the cluster the runs of a study evaluate in
parallel, so the γ = 0 run is usually not finished when another run is evaluated.
For every run of an experiment and every correlation-matrix method folder

    <run>/plots/<split>/<ckpt>/correlation_matrix/<dataset>/<Method>/

this writes into ``<Method>/comparison_gamma0/``:

* ``abs_reconstruction_minus_gamma0_reconstruction_<method>_correlation_matrix``
  ``[_sorted_by_{increase,decrease}][_et_only].png``: ``|r_reco(run)| -
  |r_reco(γ = 0 run)|``, drawn like the self-improvement matrices (FET.Et row
  framed, MI hyperparameters as subtitle), except that an entry of the FET.Et row
  is green if and only if the run's reconstructed correlation is strictly closer
  to 0 than the γ = 0 run's (``corr_plot.closer_to_zero_columns``; Pearson values
  in ``Pearson/``, Spearman values in ``Spearman/``), i.e. where the plotted
  difference is negative;
* ``gamma0_reconstruction_<method>_correlation_matrix.csv``: a copy of the
  reference matrix, so the plots can be redrawn from this folder alone;
* ``reference.json``: which run was used and why.

**Reference.** A γ = 0 run of the *same experiment* with the same seed, encoder
architecture and number of epochs; the number of FET.Et bins does not matter
(without the MI term the binning does not enter training; in Pareto-Front-261002
all ten γ = 0 runs give bit-identical matrices). It must have a reconstruction
matrix for the same split, checkpoint, dataset and method. γ = 0 runs themselves
are skipped. Runs without a matching reference are reported, not guessed.

``migrate_layout`` moves ``|after| - |before|`` PNGs that older evaluator versions
left in the method folder into ``<Method>/self_improvement/``, the layout the
callback now writes. With ``mlruns_root`` the MLflow HTML galleries of the run are
written for every touched folder (``<dataset>_<callback>_<method>[_<subfolder>]``),
for active runs that are not tagged as another training attempt.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

from src.analysis.run_mi_hyperparameters import read_mi_hyperparameters
from src.plot import correlation_matrix as corr_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLBACK = "correlation_matrix"
IDENTICAL_ATOL = 1e-9


@dataclass(frozen=True)
class RunInfo:
    """What decides whether two runs share a γ = 0 reference."""

    run_dir: Path
    gamma: Optional[float]
    seed: Optional[int]
    architecture: Optional[tuple]
    epochs: Optional[int]

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def match_key(self) -> tuple:
        return (self.seed, self.architecture, self.epochs)

    @property
    def identified(self) -> bool:
        return None not in (self.gamma, self.seed, self.architecture, self.epochs)


@dataclass
class Report:
    written: list = field(default_factory=list)          # (method_dir, reference run)
    skipped: Counter = field(default_factory=Counter)    # reason -> count
    details: list = field(default_factory=list)          # (method_dir, reason)
    moved: int = 0
    galleries: int = 0

    def skip(self, method_dir: Path, reason: str) -> None:
        self.skipped[reason] += 1
        self.details.append((method_dir, reason))


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def read_run_info(run_dir: Path) -> RunInfo:
    """Seed, encoder nodes, epochs and γ from resolved_config.yaml / run_manifest.yaml."""
    config = _read_yaml(run_dir / "resolved_config.yaml")
    manifest = _read_yaml(run_dir / "run_manifest.yaml")
    algorithm = config.get("algorithm") or {}
    candidate = manifest.get("configuration") or {}

    seed = config.get("seed", manifest.get("autoencoder_seed"))
    nodes = (algorithm.get("encoder") or {}).get("nodes") or candidate.get("encoder_nodes")
    epochs = (config.get("trainer") or {}).get("max_epochs")
    gamma = algorithm.get("mi_gamma", candidate.get("mi_gamma"))
    if gamma is None:
        gamma = read_mi_hyperparameters(run_dir).gamma
    return RunInfo(
        run_dir=run_dir,
        gamma=None if gamma is None else float(gamma),
        seed=None if seed is None else int(seed),
        architecture=None if not nodes else tuple(int(n) for n in nodes),
        epochs=None if epochs is None else int(epochs),
    )


def method_folders(run_dir: Path, callback: str = CALLBACK) -> dict[tuple, Path]:
    """``(split, ckpt, dataset, Method) -> folder`` of every folder with a reconstruction CSV."""
    folders = {}
    for csv in sorted(run_dir.glob(f"plots/*/*/{callback}/*/*/reconstruction_*_correlation_matrix.csv")):
        folder = csv.parent
        method = folder.name.lower()
        if csv.name != f"{corr_plot.correlation_matrix_stem('reconstruction', method)}.csv":
            continue
        split, ckpt, _, dataset, method_name = folder.relative_to(run_dir / "plots").parts
        folders[(split, ckpt, dataset, method_name)] = folder
    return folders


def _same_matrix(a, b) -> bool:
    return (list(a.index) == list(b.index)
            and np.allclose(a.to_numpy(float), b.to_numpy(float), atol=IDENTICAL_ATOL, equal_nan=True))


def find_reference(run: RunInfo, key: tuple, gamma0_runs: Sequence[RunInfo],
                   callback: str = CALLBACK) -> tuple[Optional[RunInfo], Optional[Path], str]:
    """γ = 0 run (and its reconstruction CSV) for one method folder of ``run``."""
    method = key[3].lower()
    candidates = [
        ref for ref in gamma0_runs
        if ref.run_dir != run.run_dir and ref.match_key == run.match_key
    ]
    if not candidates:
        return None, None, "no γ = 0 run with the same seed, architecture and epochs"
    with_csv = []
    for ref in sorted(candidates, key=lambda r: r.name):
        csv = (ref.run_dir / "plots" / key[0] / key[1] / callback / key[2] / key[3]
               / f"{corr_plot.correlation_matrix_stem('reconstruction', method)}.csv")
        if csv.is_file():
            with_csv.append((ref, csv))
    if not with_csv:
        return None, None, f"γ = 0 run has no {key[3]} matrix for {key[0]}/{key[1]}/{key[2]}"
    ref, csv = with_csv[0]
    note = ""
    first = corr_plot.load_correlation_matrix_csv(csv)
    if any(not _same_matrix(first, corr_plot.load_correlation_matrix_csv(other))
           for _, other in with_csv[1:]):
        note = f"γ = 0 runs disagree; used {ref.name}"
    return ref, csv, note


def migrate_layout(method_dir: Path) -> int:
    """Move ``|after| - |before|`` files still in the method folder into self_improvement/."""
    stems = [corr_plot.correlation_change_stem(method_dir.name.lower(), direction)
             for direction in (None, *corr_plot.SORT_DIRECTIONS)]
    moving = [path for path in sorted(method_dir.iterdir())
              if path.is_file() and path.suffix in {".png", ".csv"}
              and any(path.stem in (stem, f"{stem}{corr_plot.ET_ONLY_SUFFIX}") for stem in stems)]
    if not moving:
        return 0
    target = method_dir / corr_plot.SELF_IMPROVEMENT_DIR
    target.mkdir(exist_ok=True)
    for path in moving:
        path.replace(target / path.name)
    return len(moving)


def write_comparison(method_dir: Path, reference: RunInfo, reference_csv: Path, *,
                     subtitle: Optional[str], note: str = "") -> Path:
    """Draw the six comparison PNGs of one method folder."""
    method = method_dir.name.lower()
    out = method_dir / corr_plot.COMPARISON_GAMMA0_DIR
    out.mkdir(exist_ok=True)
    copy = out / corr_plot.gamma0_reference_csv_name(method)
    shutil.copyfile(reference_csv, copy)

    reconstruction = corr_plot.load_correlation_matrix_csv(
        method_dir / f"{corr_plot.correlation_matrix_stem('reconstruction', method)}.csv")
    gamma0_reconstruction = corr_plot.load_correlation_matrix_csv(copy)
    change = corr_plot.abs_correlation_change(gamma0_reconstruction, reconstruction)
    # Green: the run's reconstructed |r| is closer to 0 than the γ = 0 run's.
    green = corr_plot.closer_to_zero_columns(reconstruction, gamma0_reconstruction)
    for direction, ascending in ((None, None), *corr_plot.SORT_DIRECTIONS.items()):
        corr_plot.write_correlation_matrix_variants(
            change,
            plot_folder=out,
            stem=corr_plot.gamma0_comparison_stem(method, direction),
            title=corr_plot.gamma0_comparison_title(method, direction),
            sort_ascending=ascending,
            green_columns=green,
            subtitle=subtitle,
        )
    (out / "reference.json").write_text(json.dumps({
        "reference_run": reference.name,
        # Relative to the experiment folder, so the record stays valid when the
        # checkpoints are copied from EOS to another machine.
        "reference_csv": str(Path(reference_csv).relative_to(reference.run_dir.parent)),
        "matched_on": {"gamma": 0.0, "seed": reference.seed,
                       "encoder_nodes": list(reference.architecture), "max_epochs": reference.epochs},
        "bins_ignored": True,
        "note": note,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    return out


def _comparison_complete(method_dir: Path) -> bool:
    method = method_dir.name.lower()
    out = method_dir / corr_plot.COMPARISON_GAMMA0_DIR
    return all((out / f"{corr_plot.gamma0_comparison_stem(method, d)}{s}.png").is_file()
               for d in (None, *corr_plot.SORT_DIRECTIONS) for s in corr_plot.VARIANT_FIGURE_SCALES)


# --------------------------------------------------------------------- galleries
def mlflow_runs(mlruns_root: Path, experiment: str, run_name: str) -> list[Path]:
    """Active MLflow runs of ``experiment/run_name`` that are not another training attempt."""
    runs = []
    for meta in Path(mlruns_root).glob("*/meta.yaml"):
        if (_read_yaml(meta).get("name")) != experiment:
            continue
        for run_dir in meta.parent.iterdir():
            tags = run_dir / "tags"
            if not tags.is_dir():
                continue
            name_tag = tags / "mlflow.runName"
            if not name_tag.is_file() or name_tag.read_text().strip() != run_name:
                continue
            if _read_yaml(run_dir / "meta.yaml").get("lifecycle_stage") == "deleted":
                continue
            role = tags / "link.role"
            if role.is_file() and role.read_text().startswith("other training attempt"):
                continue
            runs.append(run_dir)
    return sorted(runs)


def gallery_path(mlflow_run: Path, key: tuple, subfolder: Optional[str], callback: str = CALLBACK) -> Path:
    """Where the evaluator's log_plots_to_mlflow puts the gallery of that folder."""
    split, ckpt, dataset, method_name = key
    base = mlflow_run / "artifacts" / split / ckpt / callback
    if subfolder:
        base = base / method_name
    return base / f"{corr_plot.gallery_name(dataset, callback, method_name.lower(), subfolder)}.html"


def write_galleries(mlflow_run_dirs: Iterable[Path], key: tuple, method_dir: Path,
                    subfolders: Iterable[Optional[str]], callback: str = CALLBACK) -> int:
    from src.plot.gallery import build_gallery_html

    written = 0
    for mlflow_run in mlflow_run_dirs:
        for sub in subfolders:
            folder = method_dir / sub if sub else method_dir
            if not folder.is_dir() or not any(folder.glob("*.png")):
                continue
            target = gallery_path(mlflow_run, key, sub, callback)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_text(build_gallery_html(folder, key[1]), encoding="utf-8")
            temporary.replace(target)
            written += 1
    return written


# --------------------------------------------------------------------------- run
def process_experiment(experiment_dir: Path, *, runs: Optional[Iterable[Path]] = None,
                       mlruns_root: Optional[Path] = None, migrate: bool = False,
                       force: bool = False, dry_run: bool = False,
                       callback: str = CALLBACK) -> Report:
    """Write the γ = 0 comparisons of every run (or of ``runs``) in one experiment."""
    import matplotlib

    matplotlib.use("Agg")
    experiment_dir = Path(experiment_dir)
    all_runs = [read_run_info(d) for d in sorted(experiment_dir.iterdir()) if (d / "plots").is_dir()]
    gamma0_runs = [r for r in all_runs if r.identified and r.gamma == 0.0]
    wanted = None if runs is None else {Path(r).resolve() for r in runs}
    report = Report()

    for run in all_runs:
        if wanted is not None and run.run_dir.resolve() not in wanted:
            continue
        folders = method_folders(run.run_dir, callback)
        if not folders:
            continue
        mlflow_dirs = mlflow_runs(mlruns_root, experiment_dir.name, run.name) if mlruns_root else []
        mi = read_mi_hyperparameters(run.run_dir)
        subtitle = mi.text() if mi.known else None
        for key, method_dir in folders.items():
            touched: list = []
            if migrate:
                moved = 0 if dry_run else migrate_layout(method_dir)
                report.moved += moved
                if moved:
                    touched += [None, corr_plot.SELF_IMPROVEMENT_DIR]
            if not run.identified:
                report.skip(method_dir, "run not identified (seed/architecture/epochs/γ missing)")
            elif run.gamma == 0.0:
                report.skip(method_dir, "γ = 0 run")
            elif _comparison_complete(method_dir) and not force:
                report.skip(method_dir, "comparison exists (use --force)")
            else:
                reference, csv, note = find_reference(run, key, gamma0_runs, callback)
                if reference is None:
                    report.skip(method_dir, note)
                else:
                    if not dry_run:
                        write_comparison(method_dir, reference, csv, subtitle=subtitle, note=note)
                        touched.append(corr_plot.COMPARISON_GAMMA0_DIR)
                    report.written.append((method_dir, reference.name))
                    if note:
                        report.details.append((method_dir, note))
            if touched and mlflow_dirs and not dry_run:
                report.galleries += write_galleries(mlflow_dirs, key, method_dir, touched, callback)
    return report


def _experiment_dirs_from_study_map(study_map: Path, checkpoints_root: Path) -> dict[Path, list[Path]]:
    from src.evaluation.pareto_mi_changes import _local_run_dir

    grouped: dict[Path, list[Path]] = {}
    for run in (_read_yaml(study_map).get("runs") or []):
        run_dir = _local_run_dir(run, checkpoints_root)
        if run_dir is not None:
            grouped.setdefault(run_dir.parent, []).append(run_dir)
    return grouped


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation matrices of each run vs its γ = 0 run.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--experiment-dir", type=Path, action="append",
                       help="checkpoints/<experiment>; every run in it. Repeatable.")
    where.add_argument("--study-map", type=Path,
                       help="study_map.yaml; only its runs (references still come from their experiment).")
    parser.add_argument("--checkpoints-root", type=Path, help="With --study-map: local checkpoints/ dir.")
    parser.add_argument("--mlruns-root", type=Path, help="MLflow file store; writes the HTML galleries.")
    parser.add_argument("--migrate-layout", action="store_true",
                        help="Move |after| - |before| PNGs left in the method folder into self_improvement/.")
    parser.add_argument("--force", action="store_true", help="Redraw comparisons that already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--callback-name", default=CALLBACK)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.study_map is not None:
        if args.checkpoints_root is None:
            raise SystemExit("--study-map needs --checkpoints-root")
        targets = _experiment_dirs_from_study_map(args.study_map, args.checkpoints_root)
    else:
        targets = {Path(d): None for d in args.experiment_dir}
    mlruns = args.mlruns_root if args.mlruns_root and Path(args.mlruns_root).is_dir() else None
    for experiment_dir, runs in targets.items():
        report = process_experiment(experiment_dir, runs=runs, mlruns_root=mlruns,
                                    migrate=args.migrate_layout, force=args.force,
                                    dry_run=args.dry_run, callback=args.callback_name)
        print(f"{experiment_dir.name}: {len(report.written)} method folders compared"
              f"{' (dry run)' if args.dry_run else ''}, {report.moved} files moved to "
              f"{corr_plot.SELF_IMPROVEMENT_DIR}/, {report.galleries} galleries written")
        for reason, count in sorted(report.skipped.items()):
            print(f"  skipped {count}: {reason}")
        for method_dir, note in report.details:
            if note.startswith("γ = 0 runs disagree") or note.startswith("γ = 0 run has no"):
                print(f"  {method_dir}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
