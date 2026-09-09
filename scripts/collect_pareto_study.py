"""CLI for the lean Phase 2 Pareto-study collector."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.pareto_aggregation import main


if __name__ == "__main__":
    main()
