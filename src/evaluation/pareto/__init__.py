"""Everything that builds the FET.Et validation Pareto front, in one place.

Modules, in pipeline order (nothing is imported here, so importing one module
does not pull in matplotlib, pandas or torch for the others). Every module with a
command line runs as ``python3 -m src.evaluation.pareto.<module> --help`` from the
repository root; the shell scripts in scripts/physics/ call them that way.

    make_grid           expand configs/pareto_study/fet_et.yaml into the stage-1
                        job list (scripts/physics/runae_pareto_makegrid.sh)
    manifest            resolved Pareto manifest written by src/train.py before fitting
    baseline            the single gamma = 0 / 50-bin baseline rule (torch-free)
    study_map           build STUDY_ROOT/study_map.yaml from a checkpoint directory
    aggregation         stage 3 phase 2: collect every run's artifacts
    selection           stage 3 phase 3: feasibility rules and the Pareto front
    plots               stage 3 phase 4: front figures
    matrices            stage 3 phase 4b: gamma x bins matrices, one per metric
    correlation_gamma0  stage 3 phase 4c: correlation matrices vs the gamma = 0 run
                        (also on the test split, scripts/physics/runae_test_comparison.sh)
    mi_changes          metric-vs-gamma and metric-vs-bins sweeps (hand-run)
    run_vs_gamma0       after the study: test outputs of a run vs its gamma = 0 run
                        (scripts/physics/runae_test_comparison.sh)

All stage 3 phases are run by scripts/physics/runae_pareto_runcollect.sh.
"""
