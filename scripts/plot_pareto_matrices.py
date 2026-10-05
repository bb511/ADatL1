"""CLI for the Phase 4b gamma x bins matrices. See src/evaluation/pareto_matrices.py."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.pareto_matrices import main


if __name__ == "__main__":
    main()
