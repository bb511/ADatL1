"""The single γ = 0 reference of the Pareto study.

Every candidate, whatever its number of FET.Et bins, is compared with the γ = 0
run of its architecture trained with ``GAMMA_ZERO_BASELINE_BINS`` bins: the
collapse rule (joint code entropy >= 0.5 x baseline) and the minimum-efficiency
rule (>= 0.95 x baseline) in ``pareto_aggregation``. The binning does not enter
training at γ = 0, so one baseline serves all bin counts
(``search_space.gamma_zero_baseline`` in ``configs/pareto_study/fet_et.yaml``).

Kept free of torch so that ``scripts/physics/make_pareto_grid.py`` and the
plotting modules can import it.
"""

GAMMA_ZERO_BASELINE_BINS = 50
