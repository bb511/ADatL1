"""CLI for the correlation matrices vs the γ = 0 run. See src/analysis/correlation_gamma0_comparison.py."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.analysis.correlation_gamma0_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
