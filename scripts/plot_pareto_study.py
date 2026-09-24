"""CLI for the Phase 4 Pareto figures."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.pareto_plots import main


if __name__ == "__main__":
    main()
