import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks.components.cap import bank
from src.callbacks.components.cap.binary.pairing import NNCoupling, cdf
from src.callbacks.components.cap.kernel import posterior_agreement_curve
from src.callbacks.components.cap.metric import ApproximationCapacity, CAPBank
from src.evaluation.callbacks.components.cap import bank as evaluation_bank
from src.evaluation.callbacks.components.cap.metric import (
    ApproximationCapacity as EvaluationCapacity,
)
from src.evaluation.callbacks.components.cap.metric import CAPBank as EvaluationBank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/bank.yaml"))
PROVISIONAL = OmegaConf.to_container(
    OmegaConf.load(ROOT / "configs/cap/provisional.yaml")
)
SCRIPT = ROOT / "scripts/analysis/cap_beta_fix.py"
BANKS = [bank, evaluation_bank]
MAPS = ["std", "rank", "sq"]
LOG_2 = math.log(2.0)


def _config(**overrides) -> dict:
    return copy.deepcopy(CONFIG) | overrides


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _case(n_1=300, n_2=900, seed=0):
    x_1, x_2 = _randn(n_1, 3, seed=seed), 1.2 * _randn(n_2, 3, seed=seed + 1) + 0.3
    scores_1 = (x_1[:, 0] + 0.5 * _randn(n_1, seed=seed + 2)).exp()
    scores_2 = (x_2[:, 0] + 0.5 * _randn(n_2, seed=seed + 3)).exp()
    return x_1, x_2, _randn(n_1, 3, seed=seed + 4), scores_1, scores_2


def _couplings(module, case, pairings=("cdf", "ot", "nn", "strat")):
    x_1, x_2, x_ref = case[:3]
    config = {name: CONFIG["pairings"][name] for name in pairings}
    if "ot" in config:
        config["ot"] = config["ot"] | {"n_subsamples": 2}
    return module.build_couplings(x_1, x_2, x_ref, config)


def _identity(n, keep=None, partners=None):
    index_1 = torch.arange(n) if keep is None else keep
    index_2 = index_1 if partners is None else partners
    return NNCoupling(index_1, index_2, n, 1.0, 0.0)


def _posterior(z, beta):
    anomalous = torch.sigmoid(beta * z)
    return torch.stack([1.0 - anomalous, anomalous], dim=1)


@pytest.mark.parametrize("name", ["cdf", "nn"])
@pytest.mark.parametrize("map_name", MAPS)
def test_agreement_splits_into_consistency_and_confidence(name, map_name):
    case = _case()
    coupling = _couplings(bank, case, (name,))[name]
    values = bank.cap_bank(case[3], case[4], coupling, CONFIG)

    (pairs,) = coupling.pairs(case[3], case[4])
    beta = CONFIG["beta_fix"][map_name]
    q_1 = _posterior(bank.MAPS[map_name](case[3])[pairs.index_1], beta)
    q_2 = _posterior(bank.MAPS[map_name](case[4])[pairs.index_2], beta)
    norms = q_1.norm(dim=1) * q_2.norm(dim=1)
    consistency = ((q_1 * q_2).sum(dim=1) / norms).log().mean().item()
    confidence = norms.log().mean().item() + LOG_2
    assert values[f"{map_name}_C"] == pytest.approx(consistency, abs=1e-12)
    assert values[f"{map_name}_F"] == pytest.approx(confidence, abs=1e-12)
    a = values[f"{map_name}_A-fix"]
    assert a == pytest.approx(consistency + confidence, abs=1e-12)
    assert consistency <= 0.0 and 0.0 <= confidence < LOG_2


@pytest.mark.parametrize("name", ["cdf", "ot", "nn"])
def test_exact_null_is_the_permutation_mean(name):
    case = _case()
    coupling = _couplings(bank, case, (name,))[name]
    values = bank.cap_bank(case[3], case[4], coupling, _config(null_bins=None))

    z_1, z_2 = bank.MAPS["std"](case[3]), bank.MAPS["std"](case[4])
    beta = torch.tensor([CONFIG["beta_fix"]["std"]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    draws = coupling.null_draws(case[3], case[4], 400, generator)
    agreement = [
        sum(posterior_agreement_curve(-z_1[p[0]], -z_2[p[1]], beta) for p in d) / len(d)
        for d in draws
    ]
    agreement = torch.cat(agreement) + LOG_2
    sigma = agreement.std().item()
    assert abs(values["std_A_pi"] - agreement.mean().item()) < 3 * sigma / 20
    assert values["std_sigma_pi"] == pytest.approx(sigma, rel=0.25)


@pytest.mark.parametrize("name", ["cdf", "ot", "nn"])
def test_binned_null_is_close_to_the_exact_null(name):
    # On Gaussian and lognormal scores at N1 = 1,000 and N2 = 5,400 the error of the
    # 256-bin null stays below 5e-3 sigma_pi for every map (olgpu, 23 Oct 2026);
    # Cauchy-tailed std evidence reaches 0.1 sigma_pi, through its widest tail bins.
    case = _case(1000, 5400)
    coupling = _couplings(bank, case, (name,))[name]
    for scores_1, scores_2 in ((case[3].log(), case[4].log()), case[3:]):
        binned = bank.cap_bank(scores_1, scores_2, coupling, CONFIG)
        exact = bank.cap_bank(scores_1, scores_2, coupling, _config(null_bins=None))
        for m in MAPS:
            error = abs(binned[f"{m}_A_pi"] - exact[f"{m}_A_pi"])
            assert error < 0.01 * exact[f"{m}_sigma_pi"]


@pytest.mark.parametrize("name", ["cdf", "ot"])
@pytest.mark.parametrize("map_name", MAPS)
def test_small_beta_limits_are_the_mean_product_and_the_covariance(name, map_name):
    case = _case()
    coupling = _couplings(bank, case, (name,))[name]
    beta = 1e-3
    config = _config(null_bins=None, beta_fix={m: beta for m in MAPS})
    values = bank.cap_bank(case[3], case[4], coupling, config)

    z_1, z_2 = bank.MAPS[map_name](case[3]), bank.MAPS[map_name](case[4])
    columns = [(z_1[p.index_1], z_2[p.index_2]) for p in coupling.pairs(*case[3:])]
    covariance = sum((a * b).mean() - a.mean() * b.mean() for a, b in columns)
    scale = beta**2 / 4
    a_lim = values[f"{map_name}_A-lim"]
    assert values[f"{map_name}_A-fix"] / scale == pytest.approx(a_lim, rel=1e-3)
    d_fix = values[f"{map_name}_D-fix"]
    assert d_fix / scale == pytest.approx(covariance.item() / len(columns), rel=1e-3)


@pytest.mark.parametrize("map_name", MAPS)
def test_a_sup_is_interior_when_pairs_contradict(map_name):
    scores = _randn(2000)
    noisy = scores + 0.8 * _randn(2000, seed=1)
    contradicting = bank.cap_bank(scores, noisy, _identity(2000), CONFIG)
    agreeing = bank.cap_bank(scores, scores, _identity(2000), CONFIG)

    assert contradicting[f"{map_name}_at_bound"] == 0.0
    beta_star = contradicting[f"{map_name}_beta_star"]
    assert CONFIG["beta_min"] < beta_star < CONFIG["beta_max"]
    assert agreeing[f"{map_name}_at_bound"] == 1.0


def _grid_null(a, b, beta):
    m_1, m_2 = torch.tanh(0.5 * beta * a), torch.tanh(0.5 * beta * b)
    return torch.log1p(m_1[:, None] * m_2[None, :]).mean().item()


def test_null_and_confidence_use_the_retained_pairs_only():
    scores_1, scores_2 = _randn(400), _randn(400, seed=1) + 0.2 * _randn(400, seed=2)
    keep = torch.arange(0, 400, 2)
    # The 100 highest scores of X2, twice, so that the retained column differs from X2.
    top = scores_2.argsort()[-100:]
    partners = torch.cat([top, top])
    coupling = _identity(400, keep, partners)
    values = bank.cap_bank(scores_1, scores_2, coupling, _config(null_bins=None))

    beta = CONFIG["beta_fix"]["std"]
    z_1, z_2 = bank.MAPS["std"](scores_1), bank.MAPS["std"](scores_2)
    retained = _grid_null(z_1[keep], z_2[partners], beta)
    assert values["discarded"] == 0.5
    assert values["std_A_pi"] == pytest.approx(retained, abs=1e-12)
    assert abs(values["std_A_pi"] - _grid_null(z_1, z_2, beta)) > 1e-3
    q_1, q_2 = _posterior(z_1[keep], beta), _posterior(z_2[partners], beta)
    confidence = (q_1.norm(dim=1) * q_2.norm(dim=1)).log().mean().item() + LOG_2
    assert values["std_F"] == pytest.approx(confidence, abs=1e-12)


TRANSFORMS = {
    "std": (lambda s: 2.5 * s + 3.0, lambda s: 0.3 * s - 7.0),
    "sq": (lambda s: 2.5 * s + 3.0, lambda s: 0.3 * s - 7.0),
    "rank": (torch.exp, lambda s: s**3 + s),
}


@pytest.mark.parametrize("map_name", MAPS)
def test_statistics_are_invariant_under_the_maps_invariances(map_name):
    case = _case(200, 600)
    couplings = _couplings(bank, case)
    transform_1, transform_2 = TRANSFORMS[map_name]
    moved = (transform_1(case[3]), transform_2(case[4]))
    for coupling in couplings.values():
        before = bank.cap_bank(case[3], case[4], coupling, CONFIG)
        after = bank.cap_bank(*moved, coupling, CONFIG)
        keys = [k for k in before if k.startswith(f"{map_name}_")]
        assert len(keys) == 13
        for key in keys:
            assert after[key] == pytest.approx(before[key], abs=1e-10), key


def _float32_case(n_1=500, n_2=1200):
    g = torch.Generator().manual_seed(3)
    scores_1 = torch.randn(n_1, generator=g).exp()
    scores_2 = (torch.randn(n_2, generator=g) + 0.1).exp()
    return scores_1, scores_2


@pytest.mark.parametrize(
    "module, metric_cls",
    [(bank, ApproximationCapacity), (evaluation_bank, EvaluationCapacity)],
)
def test_legacy_corner_is_the_neurips_cap_plus_log_2(module, metric_cls):
    scores_1, scores_2 = _float32_case()
    coupling = module.COUPLINGS["cdf"]()
    values = module.cap_bank(scores_1, scores_2, coupling, CONFIG)

    metric = metric_cls(**PROVISIONAL)
    index_1, index_2 = cdf(scores_1, scores_2)
    metric.update(scores_1[index_1], scores_2[index_2])
    assert values["legacy_A-sup"] - LOG_2 == pytest.approx(metric.compute(), abs=1e-10)
    assert values["legacy_beta_star"] == metric.beta_star
    assert values["legacy_at_bound"] == float(metric.at_bound)


def test_bank_metric_flattens_cap_bank_and_the_trees_agree():
    case = _case()
    scores_1, scores_2 = case[3].float(), case[4].float()
    keys = {}
    for module, metric_cls in ((bank, CAPBank), (evaluation_bank, EvaluationBank)):
        couplings = _couplings(module, case)
        metric = metric_cls(**CONFIG)
        metric.update(scores_1, scores_2, couplings)
        keys[module] = metric.compute()
        for name, coupling in couplings.items():
            values = module.cap_bank(scores_1, scores_2, coupling, CONFIG)
            values = {k: v for k, v in values.items() if not k.endswith("_beta_fix")}
            assert {k: keys[module][f"{name}_{k}"] for k in values} == values

    computed = keys[bank]
    assert keys[evaluation_bank] == computed
    assert [computed[f"{m}_beta_fix"] for m in MAPS] == [1.5023, 3.0994, 6.4826]
    assert len(computed) == 3 + 4 * (1 + 3 * 12) + 3
    assert "cdf_legacy_A-sup" in computed and "ot_legacy_A-sup" not in computed


def test_unknown_map_or_statistic_raises():
    scores = _randn(50)
    with pytest.raises(ValueError):
        bank.cap_bank(scores, scores, _identity(50), _config(maps={"minmax": {}}))
    with pytest.raises(ValueError):
        bank.cap_bank(scores, scores, _identity(50), _config(statistics=["A-max"]))


def test_zero_mad_falls_back_to_the_quantiles_and_constant_scores_give_nan():
    scores = torch.zeros(1000, dtype=torch.float64)
    scores[-100:] = torch.linspace(1.0, 2.0, 100, dtype=torch.float64)
    q05, q95 = torch.quantile(scores, torch.tensor([0.05, 0.95], dtype=torch.float64))
    expected = scores * 3.2897 / (q95 - q05)
    torch.testing.assert_close(bank.MAPS["std"](scores), expected, rtol=1e-12, atol=0)

    constant = torch.ones(1000)
    values = bank.cap_bank(constant, _randn(1000), _identity(1000), CONFIG)
    for map_name in ("std", "sq"):
        assert all(math.isnan(values[f"{map_name}_{s}"]) for s in bank.STATISTICS)
    assert values["rank_A-fix"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(values["rank_D-lim"])


def _load_script():
    spec = importlib.util.spec_from_file_location("cap_beta_fix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beta_fix_script_reads_the_last_epoch_of_every_run(tmp_path):
    script = _load_script()
    for run, offset in (("run_a", 0.0), ("run_b", 5.0)):
        directory = tmp_path / run
        directory.mkdir()
        torch.save({}, directory / "meta.pt")
        constant = {"val": {"normal": {"ascore/full": torch.zeros(2001)}}}
        torch.save(constant, directory / "epoch_0000.pt")
        scores = torch.arange(2001, dtype=torch.float32) + offset
        torch.save(
            {"val": {"normal": {"ascore/full": scores}}}, directory / "epoch_0001.pt"
        )
    (tmp_path / "no_meta").mkdir()

    runs = script._runs([tmp_path])
    assert [run.name for run in runs] == ["run_a", "run_b"]
    maps = OmegaConf.load(ROOT / "configs/cap/bank.yaml")["maps"]
    report = script.report(script.quantiles(runs, maps))
    assert (
        "rank: 2 runs (0 non-finite left out), median q = 0.9500, beta_fix = 3.0994"
        in report
    )
    assert report.splitlines()[-1].startswith("beta_fix: { std: ")


@pytest.mark.parametrize("map_name, expected", [("std", 1.5023), ("sq", 6.4826)])
def test_beta_fix_of_gaussian_scores_is_the_placeholder(map_name, expected):
    script = _load_script()
    params = CONFIG["maps"][map_name]
    q = script.evidence_quantile(_randn(200_000), map_name, params)
    assert script.beta_fix(q) == pytest.approx(expected, rel=0.01)
