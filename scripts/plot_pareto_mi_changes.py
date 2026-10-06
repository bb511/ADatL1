"""CLI for the Phase 4c γ / effective-bin sweeps. See src/evaluation/pareto_mi_changes.py."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.pareto_mi_changes import main


if __name__ == "__main__":
    main()
