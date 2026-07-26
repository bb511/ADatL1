#!/usr/bin/env python3

"""Export Optuna trials and mark the Pareto-front trials.

For a study, this writes one CSV under ``notebooks/paretos/<domain>/`` with the
columns ``number, values_0, values_1, params_*, state, is_pareto``.
Outputs are grouped by dataset/domain, so e.g. the physics Pareto fronts live in
``notebooks/paretos/physics/``.  These CSVs are consumed by ``notebooks/paretos.nb``.

* single study -- ``fetch_optuna_pareto.py <study_name> <db_path>``;
* sweep -- ``fetch_optuna_pareto.py --all`` loops over every ``logs/optuna/**/
  *.db`` and every study within, exporting one CSV per (model, strategy) so the
  full set needed to train each Pareto point of each model+strategy is produced
  in one run.  Files land in ``paretos/<domain>/<model>_<study>.csv``.

Paths are anchored to the repository root (the directory holding
``.project-root``), so this can be executed from anywhere, e.g.::

    python scripts/fetch_optuna_pareto.py --all
    ./scripts/fetch_optuna_pareto.py --domain physics --model ae --all

Note: the databases were written by Optuna >= 4, whose schema the training env's
pinned Optuna (2.10.1) cannot read.  Run this in the ``optuna-ui`` conda env,
which ships a compatible Optuna.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.project-root``."""
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def find_dbs(
    root: Path,
    domains: list[str] | None = None,
    models: list[str] | None = None,
) -> list[Path]:
    """Find ``*.db`` files under ``root``, optionally filtered by domain/model.

    ``domain`` is the db's parent directory name (physics/cifar10/robustad) and
    ``model`` is the db file stem (ae, vae, dsae, ...).
    """
    dbs = []
    for db in sorted(Path(root).rglob("*.db")):
        if domains and db.parent.name not in domains:
            continue
        if models and db.stem not in models:
            continue
        dbs.append(db)
    return dbs


def study_names(db_path: Path) -> list[str]:
    """List the study names stored in a database."""
    storage_url = f"sqlite:///{Path(db_path).expanduser().resolve()}"
    try:
        return list(optuna.study.get_all_study_names(storage=storage_url))
    except AttributeError:  # very old/new Optuna fallback
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        return [s.study_name for s in summaries]


def export_study(
    study_name: str,
    db_path: Path,
    outdir: Path,
    filename: str | None = None,
    overwrite: bool = True,
) -> tuple[pd.DataFrame, Path, bool]:
    """Load a study and write its trials (with an ``is_pareto`` flag) to CSV.

    :returns: ``(dataframe, output_path, written)`` -- ``written`` is ``False``
        when the file already existed and ``overwrite`` is ``False``.
    """
    storage_url = f"sqlite:///{Path(db_path).expanduser().resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    df = study.trials_dataframe(attrs=("number", "values", "params", "state"))

    # Optuna's Pareto front (multi-objective).
    pareto_numbers = {t.number for t in study.best_trials}
    df["is_pareto"] = df["number"].isin(pareto_numbers)

    output_path = Path(outdir) / (filename or f"{study_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    directions = [d.name for d in study.directions]
    n_pareto = int(df["is_pareto"].sum())

    if output_path.exists() and not overwrite:
        print(
            f"skip (exists): {output_path}  "
            f"[total={len(df)} pareto={n_pareto} dirs={directions}]"
        )
        return df, output_path, False

    df.to_csv(output_path, index=False)
    print(
        f"{study_name}: total={len(df)} pareto={n_pareto} dirs={directions} "
        f"-> {output_path}"
    )
    return df, output_path, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Optuna trials and mark Pareto-front trials."
    )
    parser.add_argument(
        "study_name",
        nargs="?",
        default=None,
        help="Name of the Optuna study to load (single-study mode).",
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        default=None,
        help="Path to the Optuna SQLite .db file (single-study mode).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sweep every study in every db under --root instead of a single study.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="DB search root for --all (default: <repo>/logs/optuna).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Base output dir; CSVs go in <outdir>/<domain>/ "
        "(default: <repo>/notebooks/paretos).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename for single-study mode (default: <study_name>.csv).",
    )
    parser.add_argument(
        "--domain",
        nargs="*",
        default=None,
        help="Restrict --all to these domains, e.g. physics cifar10 robustad.",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Restrict --all to these model tags (db stems), e.g. ae dsvae realnvp.",
    )
    parser.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Restrict --all to studies whose name contains one of these substrings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="In --all, overwrite existing CSVs (default: skip files already present).",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    outdir = Path(args.outdir) if args.outdir else repo / "notebooks" / "paretos"

    if args.all:
        root = Path(args.root) if args.root else repo / "logs" / "optuna"
        dbs = find_dbs(root, args.domain, args.model)
        if not dbs:
            print(f"No .db files found under {root}")
            return

        n_studies = 0
        n_written = 0
        total_pareto = 0
        for db in dbs:
            domain, model = db.parent.name, db.stem
            for study_name in study_names(db):
                if args.studies and not any(s in study_name for s in args.studies):
                    continue
                df, _, written = export_study(
                    study_name,
                    db,
                    outdir / domain,
                    filename=f"{model}_{study_name}.csv",
                    overwrite=args.overwrite,
                )
                n_studies += 1
                n_written += int(written)
                total_pareto += int(df["is_pareto"].sum())

        print(
            f"\nSwept {len(dbs)} dbs, {n_studies} studies, {total_pareto} Pareto "
            f"points total; wrote {n_written} CSVs under {outdir} (grouped by domain)"
        )
        return

    # Single-study mode.
    if not args.study_name or not args.db_path:
        parser.error("study_name and db_path are required unless --all is given")
    db = Path(args.db_path)
    export_study(
        args.study_name,
        db,
        outdir / db.parent.name,
        filename=args.output,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
