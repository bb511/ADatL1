import argparse
from pathlib import Path

import optuna
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Export Optuna trials and mark Pareto-front trials."
    )
    parser.add_argument(
        "study_name",
        help="Name of the Optuna study to load.",
    )
    parser.add_argument(
        "db_path",
        help="Path to the Optuna SQLite .db file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to <study_name>.csv in the current directory.",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    storage_url = f"sqlite:///{db_path}"

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_url,
    )

    df = study.trials_dataframe(attrs=("number", "values", "params", "state"))

    # Optuna’s Pareto front (multi-objective)
    pareto_numbers = {t.number for t in study.best_trials}
    df["is_pareto"] = df["number"].isin(pareto_numbers)

    output_path = Path(args.output) if args.output else Path(f"{args.study_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("total:", len(df), "pareto:", df["is_pareto"].sum())
    print("directions:", study.directions)
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
