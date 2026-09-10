"""CLI for the lean Phase 3 Pareto-front selector."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.pareto_selection import main


if __name__ == "__main__":
    main()
