"""Everything that builds the FET.Et validation Pareto front, in one place.

Modules, in pipeline order (nothing is imported here, so importing one module
does not pull in matplotlib, pandas or torch for the others):

    make_grid    expand configs/pareto_study/fet_et.yaml into the stage-1 job list
                 (scripts/physics/runae_pareto_makegrid.sh)
    manifest     resolved Pareto manifest written by src/train.py before fitting
    baseline     the single gamma = 0 / 50-bin baseline rule (torch-free)
    study_map    build STUDY_ROOT/study_map.yaml from a checkpoint directory
    aggregation  stage 4 phase 2: collect every run's artifacts
    selection    stage 4 phase 3: feasibility rules and the Pareto front
    plots        stage 4 phase 4: front figures
    matrices     stage 4 phase 4b: gamma x bins matrices, one per metric
    mi_changes   metric-vs-gamma and metric-vs-bins sweeps (hand-run)
"""
