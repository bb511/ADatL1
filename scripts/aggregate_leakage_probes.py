"""CLI for paired-autoencoder-seed leakage aggregation."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.evaluation.leakage_probe.aggregation import main


if __name__ == "__main__":
    main()
