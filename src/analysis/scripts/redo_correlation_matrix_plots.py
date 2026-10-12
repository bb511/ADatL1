#!/usr/bin/env python3
"""Redraw every existing correlation-matrix PNG with the current layout.

Walks a checkpoint tree for the correlation-matrix PNGs written by
``CorrelationMatrixCallback`` (and the legacy ``abs_correlation_delta`` plots of
``src/analysis/correlation_matrix.py``), redraws each one in place from the CSV
matrices next to it, then refreshes the thumbnails of the matching MLflow HTML
galleries. Only PNGs that already exist are rewritten; no new variants are added.

Source of each PNG (the PNG's own folder; for PNGs in a ``self_improvement/`` or
``comparison_gamma0/`` subfolder, the input/reconstruction CSVs of the method folder
above it):

* The CSV with the PNG's own stem, when it exists. Older evaluator versions wrote
  one per variant; it holds exactly the matrix that was plotted (labels, order and
  NaN rows included). The legacy ``*abs_correlation_delta*.png`` always use it.
* Otherwise ``{space}_{method}_correlation_matrix[_et_only].png`` comes from
  ``{space}_{method}_correlation_matrix.csv``, and
  ``abs_reconstruction_minus_input_{method}_correlation_matrix``
  ``[_sorted_by_{increase,decrease}][_et_only].png`` from ``|reconstruction| -
  |input|`` of those two CSVs, computed, cropped and sorted as the callback does.
* ``abs_reconstruction_minus_gamma0_reconstruction_...`` in ``comparison_gamma0/``:
  ``|reconstruction| - |gamma = 0 reconstruction|`` from the method folder's
  reconstruction CSV and the copy ``gamma0_reconstruction_{method}_correlation_matrix.csv``
  next to the PNG (see src/evaluation/pareto/correlation_gamma0.py).

The subtitle under each title gives the run's MI hyperparameters (γ, requested
and effective FET.Et bins), read from the checkpoint run folder by
``src/analysis/run_mi_hyperparameters.py``; values not on disk show as n/a.

Green FET.Et-row entries (``|r| <= 0.1``) are decided by the plotted matrix for the
before/after PNGs and by ``reconstruction_{method}_correlation_matrix.csv`` for the
self-improvement change PNGs (legacy delta PNGs: method from the file name, default
pearson). In the ``comparison_gamma0/`` PNGs an entry is green if and only if the
run's reconstructed |r| is strictly smaller than the gamma = 0 run's
(``corr_plot.closer_to_zero_columns``).

PNGs without their source CSVs are reported and left untouched. Each PNG is
written to a temporary file first and then swapped in, so an interrupted run never
leaves a truncated image; ``--skip-redrawn-after`` resumes such a run.

Galleries are refreshed only for active primary training runs. Runs tagged as
another training attempt of a checkpoint, and deleted runs, keep their gallery
because the PNGs on disk belong to a different run.

Example::

    python src/analysis/scripts/redo_correlation_matrix_plots.py --jobs 4
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.analysis.run_mi_hyperparameters import (  # noqa: E402
    read_mi_hyperparameters,
    run_dir_of,
)
from src.plot import correlation_matrix as corr_plot  # noqa: E402


DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
DEFAULT_MLRUNS_ROOT = REPO_ROOT / "logs" / "mlflow" / "mlruns"
LEGACY_DELTA_TITLE = "Absolute correlation change"

_SPACE_PNG = re.compile(
    r"^(?P<space>input|reconstruction|residual)_(?P<method>[a-z]+)_correlation_matrix"
    r"(?P<suffix>_et_only)?\.png$"
)
_CHANGE_PNG = re.compile(
    r"^abs_reconstruction_minus_input_(?P<method>[a-z]+)_correlation_matrix"
    r"(?:_sorted_by_(?P<direction>increase|decrease))?(?P<suffix>_et_only)?\.png$"
)
_GAMMA0_PNG = re.compile(
    r"^abs_reconstruction_minus_gamma0_reconstruction_(?P<method>[a-z]+)_correlation_matrix"
    r"(?:_sorted_by_(?P<direction>increase|decrease))?(?P<suffix>_et_only)?\.png$"
)
_LEGACY_PNG = re.compile(
    r"^(?:.+_)?abs_correlation_delta(?:_(?P<method>pearson|spearman|kendall))?(?:_.+)?\.png$"
)
_GALLERY_HTML = re.compile(
    r"^(?P<dataset>.+?)_(?P<callback>correlation_matrix)"
    r"(?:_(?P<method>[a-z]+)(?:_(?P<subfolder>self_improvement|comparison_gamma0))?)?\.html$"
)
_GALLERY_CARD_IMG = re.compile(r"<img loading='lazy' src='(?P<src>[^']*)' alt='(?P<alt>[^']*)'")


@dataclass(frozen=True)
class PlotJob:
    """One existing PNG and how to redraw it."""

    png: Path
    kind: str  # "space", "change" or "legacy"
    method: str | None = None
    space: str | None = None
    direction: str | None = None
    suffix: str = ""


@dataclass(frozen=True)
class PlotResult:
    png: Path
    status: str  # "redrawn", "no_source", "skipped_recent", "failed"
    detail: str = ""


def classify_png(png: Path) -> PlotJob | None:
    """Return the redraw job for a correlation-matrix PNG, or None for other files."""
    name = png.name
    match = _SPACE_PNG.match(name)
    if match:
        return PlotJob(
            png=png,
            kind="space",
            method=match["method"],
            space=match["space"],
            suffix=match["suffix"] or "",
        )
    match = _CHANGE_PNG.match(name)
    if match:
        return PlotJob(
            png=png,
            kind="change",
            method=match["method"],
            direction=match["direction"],
            suffix=match["suffix"] or "",
        )
    match = _GAMMA0_PNG.match(name)
    if match:
        return PlotJob(
            png=png,
            kind="gamma0",
            method=match["method"],
            direction=match["direction"],
            suffix=match["suffix"] or "",
        )
    match = _LEGACY_PNG.match(name)
    if match:
        return PlotJob(png=png, kind="legacy", method=match["method"] or "pearson")
    return None


def discover_plot_jobs(root: Path) -> dict[Path, list[PlotJob]]:
    """Group every correlation-matrix PNG below ``root`` by folder."""
    jobs: dict[Path, list[PlotJob]] = {}
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.endswith(".png") or filename.startswith("."):
                continue
            job = classify_png(Path(directory) / filename)
            if job is not None:
                jobs.setdefault(Path(directory), []).append(job)
    return dict(sorted(jobs.items()))


# Shared with the callback and the gamma = 0 comparison (src/plot/correlation_matrix.py).
_load_matrix = corr_plot.load_correlation_matrix_csv
_exclude_nan_variables = corr_plot.exclude_nan_variables
correlation_change = corr_plot.abs_correlation_change


class _MatrixSources:
    """Lazily loaded CSV matrices of one folder, plus its run's MI subtitle.

    PNGs in a ``self_improvement/`` or ``comparison_gamma0/`` subfolder take the
    input/reconstruction matrices from the method folder above it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.matrix_dir = (
            directory.parent if directory.name in corr_plot.SUBFOLDERS else directory
        )
        self._cache: dict[str, pd.DataFrame] = {}
        self._subtitle: str | None | bool = False

    @property
    def subtitle(self) -> str | None:
        if self._subtitle is False:
            run_dir = run_dir_of(Path(self.directory).resolve())
            mi = read_mi_hyperparameters(run_dir, REPO_ROOT) if run_dir else None
            self._subtitle = mi.text() if mi is not None and mi.known else None
        return self._subtitle

    def space(self, space: str, method: str) -> pd.DataFrame:
        stem = corr_plot.correlation_matrix_stem(space, method)
        if stem not in self._cache:
            path = self.matrix_dir / f"{stem}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"missing {path.name}")
            self._cache[stem] = _load_matrix(path)
        return self._cache[stem]

    def change(self, method: str) -> pd.DataFrame:
        key = f"change:{method}"
        if key not in self._cache:
            self._cache[key] = correlation_change(
                self.space("input", method),
                self.space("reconstruction", method),
            )
        return self._cache[key]

    def gamma0_reconstruction(self, method: str) -> pd.DataFrame:
        """The gamma = 0 reconstruction matrix copied into a comparison folder."""
        key = f"gamma0_reconstruction:{method}"
        if key not in self._cache:
            path = self.directory / corr_plot.gamma0_reference_csv_name(method)
            if not path.is_file():
                raise FileNotFoundError(f"missing {path.name}")
            self._cache[key] = _load_matrix(path)
        return self._cache[key]

    def gamma0_change(self, method: str) -> pd.DataFrame:
        """``|reconstruction| - |gamma = 0 reconstruction|`` of a comparison folder."""
        key = f"gamma0:{method}"
        if key not in self._cache:
            self._cache[key] = correlation_change(
                self.gamma0_reconstruction(method),
                self.space("reconstruction", method),
            )
        return self._cache[key]


def _temporary_path(png: Path) -> Path:
    return png.with_name(f".{png.stem}.redo.png")


def plotted_matrix(job: PlotJob, sources: _MatrixSources) -> pd.DataFrame:
    """The matrix shown in ``job``'s PNG; see the module docstring for the sources."""
    exact = job.png.with_suffix(".csv")
    if exact.is_file():
        return _load_matrix(exact)
    if job.kind == "legacy":
        raise FileNotFoundError(f"missing {exact.name}")

    if job.kind == "space":
        corr = sources.space(job.space, job.method)
    elif job.kind == "gamma0":
        corr = sources.gamma0_change(job.method)
    else:
        corr = sources.change(job.method)
    variant = corr_plot.select_variant(corr, job.suffix)
    if job.direction is not None:
        variant = corr_plot.sort_correlation_change_matrix(
            variant,
            ascending=corr_plot.SORT_DIRECTIONS[job.direction],
        )
    return variant


def decorrelation_reference(
    job: PlotJob,
    sources: _MatrixSources,
    plotted: pd.DataFrame,
) -> pd.DataFrame | None:
    """Matrix whose ``|r| <= 0.1`` entries are printed green in the FET.Et row.

    ``None`` for the ``comparison_gamma0/`` PNGs, which use :func:`green_columns`.
    """
    if job.kind == "space":
        return plotted if job.space in {"input", "reconstruction"} else None
    if job.kind == "gamma0":
        return None
    return sources.space("reconstruction", job.method)


def green_columns(job: PlotJob, sources: _MatrixSources) -> list[str] | None:
    """Green FET.Et-row columns of a ``comparison_gamma0/`` PNG, else ``None``.

    Green where the run's reconstructed |r| is strictly closer to 0 than the
    gamma = 0 run's.
    """
    if job.kind != "gamma0":
        return None
    return corr_plot.closer_to_zero_columns(
        sources.space("reconstruction", job.method),
        sources.gamma0_reconstruction(job.method),
    )


def _draw(job: PlotJob, sources: _MatrixSources, target: Path) -> None:
    """Draw ``job`` into ``target`` (same folder as the job's PNG)."""
    variant = plotted_matrix(job, sources)
    reference = decorrelation_reference(job, sources, variant)
    green = green_columns(job, sources)

    if job.kind == "legacy":
        from src.analysis.correlation_matrix import (
            CorrelationMatrixPlotter,
            CorrelationMatrixSpecs,
        )

        source = job.png.with_suffix(".csv")
        plotter = CorrelationMatrixPlotter(
            CorrelationMatrixSpecs(input_path=source, reconstruction_path=source)
        )
        # matrix.plot switches the global style to CMS; the legacy script never did.
        with matplotlib.style.context("default"):
            plotter._plot_heatmap(
                variant,
                save_path=target,
                title=LEGACY_DELTA_TITLE,
                decorrelation_reference=reference,
                subtitle=sources.subtitle,
            )
        return

    if job.kind == "space":
        title = corr_plot.correlation_matrix_title(job.space, job.method)
    elif job.kind == "gamma0":
        title = corr_plot.gamma0_comparison_title(job.method, job.direction)
    else:
        title = corr_plot.correlation_change_title(job.method, job.direction)
    corr_plot.plot_correlation_matrix(
        variant,
        save_dir=target.parent,
        filename=target.name,
        title=title,
        figure_scale=corr_plot.VARIANT_FIGURE_SCALES[job.suffix],
        decorrelation_reference=reference,
        green_columns=green,
        subtitle=sources.subtitle,
    )


def redraw_directory(
    directory: Path,
    jobs: Sequence[PlotJob],
    skip_redrawn_after: float | None = None,
    dry_run: bool = False,
) -> list[PlotResult]:
    """Redraw every job of one folder in place."""
    sources = _MatrixSources(directory)
    results = []
    for job in jobs:
        if skip_redrawn_after is not None and job.png.stat().st_mtime >= skip_redrawn_after:
            results.append(PlotResult(job.png, "skipped_recent"))
            continue
        target = _temporary_path(job.png)
        try:
            if dry_run:
                decorrelation_reference(job, sources, plotted_matrix(job, sources))
                green_columns(job, sources)
            else:
                _draw(job, sources, target)
                os.replace(target, job.png)
        except FileNotFoundError as error:
            results.append(PlotResult(job.png, "no_source", str(error)))
        except Exception as error:  # noqa: BLE001 - report and continue
            results.append(PlotResult(job.png, "failed", f"{type(error).__name__}: {error}"))
        else:
            results.append(PlotResult(job.png, "redrawn"))
        finally:
            if target.exists():
                target.unlink()
    return results


def redraw_all(
    jobs_by_directory: dict[Path, list[PlotJob]],
    workers: int = 1,
    skip_redrawn_after: float | None = None,
    max_seconds: float | None = None,
    dry_run: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[PlotResult], int]:
    """Redraw all folders; return the results and the number of folders not started."""
    started = time.monotonic()
    items = list(jobs_by_directory.items())
    results: list[PlotResult] = []

    def out_of_time() -> bool:
        return max_seconds is not None and time.monotonic() - started > max_seconds

    if workers <= 1:
        for index, (directory, jobs) in enumerate(items):
            if out_of_time():
                return results, len(items) - index
            results.extend(redraw_directory(directory, jobs, skip_redrawn_after, dry_run))
            if progress:
                progress(index + 1, len(items))
        return results, 0

    pending = iter(items)
    remaining = len(items)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = set()

        def submit_next() -> bool:
            nonlocal remaining
            if out_of_time():
                return False
            try:
                directory, jobs = next(pending)
            except StopIteration:
                return False
            futures.add(
                pool.submit(redraw_directory, directory, jobs, skip_redrawn_after, dry_run)
            )
            remaining -= 1
            return True

        for _ in range(workers * 2):
            if not submit_next():
                break
        while futures:
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                results.extend(future.result())
                done += 1
                if progress:
                    progress(done, len(items))
                submit_next()
    return results, remaining


# ----------------------------------------------------------------------------------
# MLflow galleries
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GalleryResult:
    gallery: Path
    status: str  # "refreshed", "unchanged", "skipped", "failed"
    detail: str = ""


def _read_tag(run_dir: Path, key: str) -> str | None:
    path = run_dir / "tags" / key
    return path.read_text().strip() if path.is_file() else None


def _read_meta(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    prefix = f"{key}:"
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("'\"") or None
    return None


def run_checkpoint_dir(run_dir: Path, checkpoints_root: Path) -> Path | None:
    """Local checkpoint folder of an MLflow run.

    ``link.checkpoint_dir`` holds an absolute path from the machine that wrote it,
    so only its last two parts (experiment and run folder) are used.
    """
    candidates = []
    linked = _read_tag(run_dir, "link.checkpoint_dir")
    if linked:
        linked_path = Path(linked)
        candidates.append(checkpoints_root / linked_path.parent.name / linked_path.name)
    experiment_name = _read_meta(run_dir.parent / "meta.yaml", "name")
    run_name = _read_tag(run_dir, "mlflow.runName") or _read_meta(
        run_dir / "meta.yaml", "run_name"
    )
    if experiment_name and run_name:
        candidates.append(checkpoints_root / experiment_name / run_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def gallery_plot_dir(gallery: Path, run_dir: Path, checkpoint_dir: Path) -> Path | None:
    """Plot folder whose images a correlation-matrix gallery shows."""
    match = _GALLERY_HTML.match(gallery.name)
    if match is None:
        return None
    relative = gallery.relative_to(run_dir / "artifacts").parts
    if len(relative) < 3:
        return None
    split, checkpoint_name = relative[0], relative[1]
    plot_dir = (
        checkpoint_dir
        / "plots"
        / split
        / checkpoint_name
        / match["callback"]
        / match["dataset"]
    )
    if match["method"]:
        plot_dir = plot_dir / match["method"].capitalize()
    if match["subfolder"]:
        plot_dir = plot_dir / match["subfolder"]
    return plot_dir


def discover_galleries(mlruns_root: Path) -> list[Path]:
    galleries = []
    for directory, _, filenames in os.walk(mlruns_root):
        if "artifacts" not in Path(directory).parts:
            continue
        for filename in filenames:
            if _GALLERY_HTML.match(filename):
                galleries.append(Path(directory) / filename)
    return sorted(galleries)


def _run_dir_of(gallery: Path) -> Path:
    parts = gallery.parts
    return Path(*parts[: parts.index("artifacts")])


def refresh_gallery(
    gallery: Path,
    plot_dir: Path,
    redrawn: set[Path],
    thumbnail: Callable[[Path], str],
    dry_run: bool = False,
) -> GalleryResult:
    """Swap the thumbnails of redrawn PNGs inside one gallery; keep everything else."""
    text = gallery.read_text(encoding="utf-8")
    replaced = 0

    def swap(match: re.Match) -> str:
        nonlocal replaced
        png = plot_dir / f"{html.unescape(match['alt'])}.png"
        if png not in redrawn:
            return match.group(0)
        replaced += 1
        src = html.escape(thumbnail(png))
        return f"<img loading='lazy' src='{src}' alt='{match['alt']}'"

    new_text = _GALLERY_CARD_IMG.sub(swap, text)
    if not replaced:
        return GalleryResult(gallery, "unchanged", "no redrawn image in this gallery")
    if not dry_run:
        temporary = gallery.with_name(f".{gallery.name}.redo")
        temporary.write_text(new_text, encoding="utf-8")
        os.replace(temporary, gallery)
    return GalleryResult(gallery, "refreshed", f"{replaced} thumbnails")


def refresh_galleries(
    mlruns_root: Path,
    checkpoints_root: Path,
    redrawn: Iterable[Path],
    thumbnail: Callable[[Path], str] | None = None,
    dry_run: bool = False,
) -> list[GalleryResult]:
    """Refresh every correlation-matrix gallery of an active primary run."""
    if thumbnail is None:
        from src.plot.gallery import generate_thumbnail as thumbnail

    redrawn = {Path(path).resolve() for path in redrawn}
    checkpoints_root = checkpoints_root.resolve()
    results = []
    for gallery in discover_galleries(mlruns_root):
        run_dir = _run_dir_of(gallery)
        role = _read_tag(run_dir, "link.role") or ""
        if _read_meta(run_dir / "meta.yaml", "lifecycle_stage") == "deleted":
            results.append(GalleryResult(gallery, "skipped", "deleted run"))
            continue
        if role.startswith("other training attempt"):
            results.append(GalleryResult(gallery, "skipped", role))
            continue
        checkpoint_dir = run_checkpoint_dir(run_dir, checkpoints_root)
        if checkpoint_dir is None:
            results.append(GalleryResult(gallery, "skipped", "no local checkpoint folder"))
            continue
        plot_dir = gallery_plot_dir(gallery, run_dir, checkpoint_dir)
        if plot_dir is None or not plot_dir.is_dir():
            results.append(GalleryResult(gallery, "skipped", f"no plot folder {plot_dir}"))
            continue
        try:
            results.append(
                refresh_gallery(gallery, plot_dir.resolve(), redrawn, thumbnail, dry_run)
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            results.append(GalleryResult(gallery, "failed", f"{type(error).__name__}: {error}"))
    return results


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw existing correlation-matrix PNGs and refresh their galleries."
    )
    parser.add_argument("--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--mlruns-root", type=Path, default=DEFAULT_MLRUNS_ROOT)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--skip-redrawn-after",
        help="ISO timestamp; skip PNGs modified at or after it (resume an earlier run). "
        "Galleries then also pick up those PNGs.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Stop starting new folders after this many seconds.",
    )
    parser.add_argument("--no-galleries", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="Write a per-file CSV report here.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    skip_after = (
        datetime.fromisoformat(args.skip_redrawn_after).timestamp()
        if args.skip_redrawn_after
        else None
    )

    jobs = discover_plot_jobs(args.checkpoints_root)
    total = sum(len(items) for items in jobs.values())
    print(f"Found {total} correlation-matrix PNGs in {len(jobs)} folders.", flush=True)

    def progress(done: int, count: int) -> None:
        if done % 50 == 0 or done == count:
            print(f"  folders {done}/{count}", flush=True)

    results, not_started = redraw_all(
        jobs,
        workers=args.jobs,
        skip_redrawn_after=skip_after,
        max_seconds=args.max_seconds,
        dry_run=args.dry_run,
        progress=progress,
    )
    counts = Counter(result.status for result in results)
    print(f"PNGs: {dict(sorted(counts.items()))}; folders not started: {not_started}")
    for result in results:
        if result.status in {"no_source", "failed"}:
            print(f"  {result.status.upper()} {result.png}: {result.detail}")

    gallery_results: list[GalleryResult] = []
    if not args.no_galleries and not_started == 0:
        current = [r.png for r in results if r.status in {"redrawn", "skipped_recent"}]
        gallery_results = refresh_galleries(
            args.mlruns_root,
            args.checkpoints_root,
            current,
            dry_run=args.dry_run,
        )
        gallery_counts = Counter(result.status for result in gallery_results)
        print(f"Galleries: {dict(sorted(gallery_counts.items()))}")
        for result in gallery_results:
            if result.status in {"skipped", "failed"}:
                print(f"  {result.status.upper()} {result.gallery}: {result.detail}")
    elif not args.no_galleries:
        print("Galleries not refreshed: rerun with --skip-redrawn-after to finish first.")

    if args.report:
        rows = [
            {"kind": "png", "path": str(r.png), "status": r.status, "detail": r.detail}
            for r in results
        ] + [
            {"kind": "gallery", "path": str(r.gallery), "status": r.status, "detail": r.detail}
            for r in gallery_results
        ]
        pd.DataFrame(rows).to_csv(args.report, index=False)

    failed = counts.get("failed", 0) + sum(r.status == "failed" for r in gallery_results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
