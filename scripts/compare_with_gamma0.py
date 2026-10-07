"""CLI: compare runs' evaluation outputs with their γ = 0 / 50-bin run.

See src/analysis/run_vs_gamma0_comparison.py; run by scripts/physics/runae_test_comparison.sh.
"""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.analysis.run_vs_gamma0_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
