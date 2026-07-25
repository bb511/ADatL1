## Highest Priority Fixes

1. Remove the final downstream-oracle selection from the main result.
    The paper says Pareto-front trials are retrained and the final reported trial is chosen by downstream anomaly performance. That weakens the label-free
    claim. The main table should report the model selected entirely by Q' + Q''; put oracle-within-Pareto only in appendix as an upper bound. The evaluator
    already has the relative-change rule in src/evaluation/evaluator.py:239, so formalize and use it end-to-end.
2. Add seeds and real uncertainty.
    Current CIs over anomaly datasets are useful but do not prove robustness to initialization, split, or HPO noise. Run at least 3 seeds for LHC selected
    configs and 5 for image benchmarks. Report paired seed-level deltas, not only per-signal Wilcoxon.
3. De-risk CAP design choices.
    CAP uses CDF pairing + sigmoid + adaptive energy in configs like configs/experiment/physics/ae_agnostic.yaml:84. Since the paper chose this from AE/LHC
    ablations, reviewers may see it as tuned to the main result. Freeze the CAP variant using one dev setting, then evaluate on held-out models/datasets. Add
    stress tests: constant scores, random scores, monotone score transforms, score-scale changes, sample-size reduction, normal-domain mismatch, and anomaly
    contamination in the normal validation split.
4. Strengthen benchmarks.
    For LHC, add run/era transfer beyond ZB vs SingleNeutrino: ZB run A vs ZB run B, 2024 vs 2025, and pileup-rate diagnostics. For CIFAR, run all ten one-vs-
    rest normal classes, not only class 0. For RobustAD, use all subsets, not only PCB; current PCB results saturate. If time allows, add MVTec AD or VisA as a
    stronger industrial benchmark.
5. Add stronger baselines.
    W1 and threshold drift are too few. Add MMD/energy distance/KS over scores, native validation loss, score entropy or collapse penalty, perturbation-
    consistency validation, and synthetic/outlier-exposure as explicitly biased upper-bound baselines.

## Code/Reproducibility Work

- CAP needs hardening before reviewers rely on it. The implementation optimizes beta by stochastic Adam over score batches in src/callbacks/metrics/cap/
metric.py:164, and the kernel uses raw exp/log instead of logsumexp in src/callbacks/metrics/cap/kernel.py:44. Make beta optimization deterministic, clamp
beta after optimizer steps, use logsumexp, and add unit tests for known edge cases.
- The paper claims 600 trials, but repo defaults and scripts disagree: configs/hparams_search/ae_optuna.yaml:29 says 15, while scripts/physics/runae.sh:210
uses 100. Make exact reproduction scripts match the paper.
- uv run pytest -q tests/test_configs.py currently fails because tests instantiate configs without registering the custom reverse resolver at tests/
test_configs.py:19. Fix this before submission.
- pyproject.toml depends on a private SSH capmetric package at pyproject.toml:14. For anonymous reproducibility, vendor it, remove it, or pin an anonymized
public source.