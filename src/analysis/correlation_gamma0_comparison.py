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

It also extends each run's ``<run>/plots/<split>/<ckpt>/correlation_matrix/<dataset>/
mean_correlations.json`` (written by the evaluator in stage 3) with

* ``spaces.reconstruction_gamma0``: the γ = 0 run's reconstruction means, copied
  from its own ``mean_correlations.json`` (``pearson`` / ``spearman``, each with
  ``mean_correlation`` and ``num_other_variables``);
* ``spaces.reconstruction.{pearson,spearman}["mean increase compared to gamma = 0"]``:
  ``100 * (1 - mean_run / mean_γ0)`` in percent, i.e. by how much the run's mean
  |r(FET.Et, ·)| is lower than the γ = 0 run's (negative: higher). ``null`` when the
  γ = 0 mean is 0 or missing.

These are rewritten on every pass, also when the comparison plots exist, so a
stage 3 rerun (which writes the file afresh) is picked up by the next stage 4.

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

MEAN_CORRELATIONS = "mean_correlations.json"
#: ``spaces`` entry with the γ = 0 run's reconstruction means.
GAMMA0_SPACE = "reconstruction_gamma0"
#: Added to ``spaces.reconstruction.<method>``: ``100 * (1 - mean_run / mean_γ0)``.
IMPROVEMENT_KEY = "mean increase compared to gamma = 0"
MEAN_METHODS = ("pearson", "spearman")


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
    means: list = field(default_factory=list)            # (mean_correlations.json, reference run)

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


def method_folders(run_dir: Path, callback: str = CALLBACK,
                   splits: Optional[Sequence[str]] = None) -> dict[tuple, Path]:
    """``(split, ckpt, dataset, Method) -> folder`` of every folder with a reconstruction CSV.

    ``splits`` (e.g. ``["val"]``) keeps only those splits; ``None`` keeps all.
    """
    folders = {}
    for csv in sorted(run_dir.glob(f"plots/*/*/{callback}/*/*/reconstruction_*_correlation_matrix.csv")):
        folder = csv.parent
        method = folder.name.lower()
        if csv.name != f"{corr_plot.correlation_matrix_stem('reconstruction', method)}.csv":
            continue
        split, ckpt, _, dataset, method_name = folder.relative_to(run_dir / "plots").parts
        if splits is not None and split not in splits:
            continue
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


# --------------------------------------------------------- mean_correlations.json
def mean_correlation_files(run_dir: Path, callback: str = CALLBACK,
                           splits: Optional[Sequence[str]] = None) -> dict[tuple, Path]:
    """``(split, ckpt, dataset) -> mean_correlations.json`` of a run (``splits`` as above)."""
    files = {}
    for path in sorted(run_dir.glob(f"plots/*/*/{callback}/*/{MEAN_CORRELATIONS}")):
        split, ckpt, _, dataset, _ = path.relative_to(run_dir / "plots").parts
        if splits is not None and split not in splits:
            continue
        files[(split, ckpt, dataset)] = path
    return files


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reconstruction_means(payload: dict) -> dict:
    """``spaces.reconstruction`` of a mean_correlations.json payload."""
    return ((payload.get("spaces") or {}).get("reconstruction") or {})


def find_mean_reference(run: RunInfo, key: tuple, gamma0_runs: Sequence[RunInfo],
                        callback: str = CALLBACK) -> tuple[Optional[RunInfo], Optional[Path], str]:
    """γ = 0 run (and its mean_correlations.json) for one ``(split, ckpt, dataset)`` of ``run``."""
    split, ckpt, dataset = key
    candidates = [
        ref for ref in gamma0_runs
        if ref.run_dir != run.run_dir and ref.match_key == run.match_key
    ]
    if not candidates:
        return None, None, "no γ = 0 run with the same seed, architecture and epochs"
    usable = []
    for ref in sorted(candidates, key=lambda r: r.name):
        path = ref.run_dir / "plots" / split / ckpt / callback / dataset / MEAN_CORRELATIONS
        means = _reconstruction_means(_read_json(path))
        if all(_finite_mean(means, method) is not None for method in MEAN_METHODS):
            usable.append((ref, path, means))
    if not usable:
        return None, None, f"γ = 0 run has no reconstruction means for {split}/{ckpt}/{dataset}"
    ref, path, means = usable[0]
    note = ""
    if any(abs(_finite_mean(means, m) - _finite_mean(other, m)) > IDENTICAL_ATOL
           for _, _, other in usable[1:] for m in MEAN_METHODS):
        note = f"γ = 0 runs disagree; used {ref.name}"
    return ref, path, note


def _finite_mean(means: dict, method: str) -> Optional[float]:
    value = (means.get(method) or {}).get("mean_correlation")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def improvement_percent(run_mean: Optional[float], gamma0_mean: Optional[float]) -> Optional[float]:
    """``100 * (1 - run_mean / gamma0_mean)``; ``None`` if undefined."""
    if run_mean is None or gamma0_mean is None or gamma0_mean == 0.0:
        return None
    return 100.0 * (1.0 - run_mean / gamma0_mean)


def add_gamma0_means(path: Path, reference_path: Path) -> dict:
    """Write the γ = 0 means and the improvement into one mean_correlations.json."""
    payload = _read_json(path)
    if not payload:
        raise ValueError(f"unreadable {path}")
    gamma0 = _reconstruction_means(_read_json(reference_path))
    spaces = payload.setdefault("spaces", {})
    reconstruction = spaces.get("reconstruction") or {}
    spaces[GAMMA0_SPACE] = {method: dict(gamma0[method]) for method in MEAN_METHODS if method in gamma0}
    for method in MEAN_METHODS:
        if method in reconstruction:
            reconstruction[method][IMPROVEMENT_KEY] = improvement_percent(
                _finite_mean(reconstruction, method), _finite_mean(gamma0, method))
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return payload


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
                       callback: str = CALLBACK,
                       splits: Optional[Sequence[str]] = None) -> Report:
    """Write the γ = 0 comparisons of every run (or of ``runs``) in one experiment.

    ``splits`` restricts the work to those evaluation splits (stage 4: ``["val"]``;
    scripts/physics/runae_test_comparison.sh: ``["test"]``). The reference must
    have outputs for the same split.
    """
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
        if run.identified and run.gamma != 0.0:
            for key, path in mean_correlation_files(run.run_dir, callback, splits).items():
                reference, reference_path, note = find_mean_reference(run, key, gamma0_runs, callback)
                if reference is None:
                    report.skip(path, f"{MEAN_CORRELATIONS}: {note}")
                    continue
                if not dry_run:
                    add_gamma0_means(path, reference_path)
                report.means.append((path, reference.name))
                if note:
                    report.details.append((path, note))
        folders = method_folders(run.run_dir, callback, splits)
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
    parser.add_argument("--split", action="append", dest="splits",
                        help="Only this evaluation split (val, test). Repeatable; default: all.")
    parser.add_argument("--run-name", action="append", dest="run_names",
                        help="Only this run of the experiment(s). Repeatable; references still "
                             "come from the whole experiment.")
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
        if args.run_names:
            names = set(args.run_names)
            candidates = runs if runs is not None else [d for d in Path(experiment_dir).iterdir()]
            runs = [Path(r) for r in candidates if Path(r).name in names]
            if not runs:
                print(f"{Path(experiment_dir).name}: no run named {sorted(names)}")
                continue
        report = process_experiment(experiment_dir, runs=runs, mlruns_root=mlruns,
                                    migrate=args.migrate_layout, force=args.force,
                                    dry_run=args.dry_run, callback=args.callback_name,
                                    splits=args.splits)
        print(f"{experiment_dir.name}: {len(report.written)} method folders compared"
              f"{' (dry run)' if args.dry_run else ''}, {report.moved} files moved to "
              f"{corr_plot.SELF_IMPROVEMENT_DIR}/, {report.galleries} galleries written, "
              f"{len(report.means)} {MEAN_CORRELATIONS} extended")
        for reason, count in sorted(report.skipped.items()):
            print(f"  skipped {count}: {reason}")
        for method_dir, note in report.details:
            if note.startswith("γ = 0 runs disagree") or note.startswith("γ = 0 run has no"):
                print(f"  {method_dir}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
