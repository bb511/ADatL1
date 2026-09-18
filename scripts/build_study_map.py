#!/usr/bin/env python3
"""CLI shim: build a Phase 2 study map from an experiment directory.

See src/evaluation/pareto_study_map.py for what it does and why.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.evaluation.pareto_study_map import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
