import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks import threshold_posterior as tp
from src.evaluation.callbacks.threshold_posterior import ThresholdPosterior
from tests.test_threshold_posterior_callback import (
    REFERENCE,
    VIEWS,
    _evaluate,
    _loaders,
    _scores,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/posterior.yaml"))
SMALL = CONFIG | {"n_pairs": 12}
FAST = SMALL | {"n_draws": 20}
ALPHA = CONFIG["alpha"]
# ancap_w's columns before f1_se, se and r2 were added (ANCAP-W as W1 computed it), as
# float.hex: computed on olgpu on 8 Nov 2026 from the working tree on 9f7e3f8 by the
# cases of _case below.
V2 = {
    "small": {
        "ancapw_a": "0x1.f47136f1ef315p-2",
        "ancapw_beta_a": "0x1.66320490e2628p+4",
        "ancapw_conf_a": "0x1.0d1857934abc7p-1",
        "ancapw_cons_a": "-0x1.2dfbc1a5323c8p-5",
        "ancapw_f2_a": "0x1.3a06d3a06d3a0p-3",
        "ancapw_end_a": "0x0.0p+0",
        "ancapw_b": "0x1.9c58532bd587fp-2",
        "ancapw_beta_b": "0x1.1c86531b4ac69p+4",
        "ancapw_conf_b": "0x1.a63da1a8a6c7fp-2",
        "ancapw_cons_b": "-0x1.3ca9cf9a28000p-7",
        "ancapw_f2_b": "0x1.eb851eb851eb8p-4",
        "ancapw_end_b": "0x0.0p+0",
        "ancapw_short": "nan",
        "ancapw_beta_short": "nan",
        "ancapw_conf_short": "nan",
        "ancapw_cons_short": "nan",
        "ancapw_f2_short": "nan",
        "ancapw_end_short": "nan",
        "ancapw_f1": "0x1.4e81b4e81b4e8p-2",
    },
    "registered": {
        "ancapw_g": "0x1.a15b1f81f7cbap+0",
        "ancapw_beta_g": "0x1.aaba136292195p+3",
        "ancapw_conf_g": "nan",
        "ancapw_cons_g": "nan",
        "ancapw_f2_g": "0x1.ccccccccccccdp-2",
        "ancapw_end_g": "0x0.0p+0",
        "ancapw_f1": "0x1.b333333333333p-1",
    },
}


def _scores_of(n, seed, shift=0.0, scale=1.0):
    # Rounded to two decimals, so that every set carries ties.
    g = torch.Generator().manual_seed(seed)
    x = scale * torch.randn(n, generator=g, dtype=torch.float64) + shift
    return torch.round(100.0 * x) / 100.0


def _small_sets():
    view_1, view_2 = _scores_of(240, 0), _scores_of(410, 1, 0.2)
    views = {"a": view_2, "b": tp.tail_of(view_2 + 0.1), "short": _scores_of(20, 9)}
    return view_1, views, _scores_of(150, 2, 2.0)


def _case(name):
    if name == "small":
        return tp.ancap_w(*_small_sets(), SMALL)
    view_1, view_2 = _scores_of(300, 3), _scores_of(400, 4, 0.1)
    return tp.ancap_w(view_1, {"g": view_2}, _scores_of(120, 5, 2.5), CONFIG)


def _hex(values: dict) -> dict:
    return {key: float(value).hex() for key, value in values.items()}


@pytest.mark.parametrize("name", V2)
def test_the_columns_of_v2_are_unchanged_to_the_bit(name):
    values = _case(name)
    assert _hex({key: values[key] for key in V2[name]}) == V2[name]
    grades = [key[len("ancapw_beta_") :] for key in V2[name] if "_beta_" in key]
    added = {f"ancapw_{p}_{g}" for p in ("se", "r2") for g in grades}
    assert set(values) == set(V2[name]) | added | {"ancapw_f1_se"}


def _pairs(view_1, view_2, reference, config):
    # Stream 0 redrawn row by row: view 1, P, then view 2, each [n_pairs, n].
    rng = np.random.default_rng([config["seed"], 0])
    redrawn = []
    for scores in (view_1, reference, view_2):
        rows = rng.integers(0, len(scores), size=(config["n_pairs"], len(scores)))
        redrawn.append(scores.sort().values[torch.from_numpy(rows)])
    tau_1, tau_2 = (torch.quantile(redrawn[i], 1.0 - ALPHA, dim=1) for i in (0, 2))
    upper, lower = (torch.quantile(redrawn[0], q, dim=1) for q in (0.75, 0.25))
    return tau_1, tau_2, upper - lower, redrawn[1]


@pytest.mark.parametrize("grade", ["a", "b"])
def test_the_errors_are_the_spreads_of_the_pairs_of_stream_zero(grade):
    view_1, views, reference = _small_sets()
    values = tp.ancap_w(view_1, views, reference, SMALL)
    view_2 = views["a"] if grade == "a" else views["a"] + 0.1
    tau_1, tau_2, scale, points = _pairs(view_1, view_2, reference, SMALL)
    f_1 = (points > tau_1[:, None]).double().mean(dim=1)
    assert values["ancapw_f1_se"] == pytest.approx(np.std(f_1.numpy(), ddof=1))
    m_1, m_2 = ((points - t[:, None]) / scale[:, None] for t in (tau_1, tau_2))
    beta = torch.tensor([values[f"ancapw_beta_{grade}"]], dtype=torch.float64)
    sums = tp._agreement_sums(m_1, m_2, torch.ones_like(m_1), beta, ALPHA)
    expected = np.std((sums[:, 0] / points.shape[1]).numpy(), ddof=1)
    assert values[f"ancapw_se_{grade}"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 < values[f"ancapw_se_{grade}"] and 0.0 < values["ancapw_f1_se"]


def test_the_errors_are_seed_reproducible_and_nan_where_their_column_is():
    sets = _small_sets()
    first, again = tp.ancap_w(*sets, SMALL), tp.ancap_w(*sets, SMALL)
    assert _hex(first) == _hex(again)
    other = tp.ancap_w(*sets, SMALL | {"seed": 1})
    assert other["ancapw_f1_se"] != first["ancapw_f1_se"]
    assert other["ancapw_se_a"] != first["ancapw_se_a"]
    assert math.isnan(first["ancapw_short"]) and math.isnan(first["ancapw_se_short"])
    spoiled = torch.cat([sets[2], torch.tensor([math.nan])])
    undefined = tp.ancap_w(sets[0], sets[1], spoiled, SMALL)
    assert math.isnan(undefined["ancapw_f1"]) and math.isnan(undefined["ancapw_f1_se"])


def test_r2_is_the_exact_share_above_the_cut_where_the_tail_reaches_it():
    view_1, view_2 = _scores_of(240, 0), _scores_of(410, 1, 0.2)
    reference = _scores_of(150, 2, 2.0)
    tau_1 = torch.quantile(view_1, 1.0 - ALPHA)
    tail = tp.tail_of(view_2)
    assert tail.top.min() <= tau_1
    expected = (view_2 > tau_1).sum().item() / len(view_2)
    for view in (view_2, tail):
        values = tp.ancap_w(view_1, {"g": view}, reference, SMALL)
        assert values["ancapw_r2_g"] == expected
    assert expected > 0.0


def test_r2_counts_strictly_above_a_cut_at_the_smallest_kept_value():
    view_1, reference = _scores_of(240, 0), _scores_of(150, 2, 2.0)
    tau_1 = torch.quantile(view_1, 1.0 - ALPHA)
    top = tau_1 + torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    values = tp.ancap_w(view_1, {"g": tp.Tail(top, 100)}, reference, SMALL)
    assert values["ancapw_r2_g"] == 4 / 100


def test_r2_is_nan_where_the_cut_lies_below_the_tail():
    view_1, reference = _scores_of(240, 0), _scores_of(150, 2, 2.0)
    view_2 = _scores_of(410, 1, 1.5)
    assert tp.tail_of(view_2).top.min() > torch.quantile(view_1, 1.0 - ALPHA)
    for view in (view_2, tp.tail_of(view_2)):
        values = tp.ancap_w(view_1, {"g": view}, reference, SMALL)
        assert math.isnan(values["ancapw_r2_g"])
        assert math.isfinite(values["ancapw_f1"])


@pytest.mark.parametrize("working_point", [False, True])
def test_the_mirror_records_the_working_point_columns_on_request(working_point):
    loaders = _loaders(["normal", *VIEWS.values(), REFERENCE])
    callback = ThresholdPosterior(
        "ascore/full", "normal", VIEWS, FAST, name="x", working_point=working_point
    )
    records = _evaluate(callback, loaders)
    scores = {name: _scores(loader) for name, loader in loaders.items()}
    views = {grade: scores[name] for grade, name in VIEWS.items()}
    args = (scores["normal"], views, scores[REFERENCE], FAST)
    expected = tp.ancap(*args) | (tp.ancap_w(*args) if working_point else {})
    assert _hex(records) == _hex({f"x/{key}": v for key, v in expected.items()})
    new = {f"x/ancapw_{p}_{g}" for p in ("se", "r2") for g in VIEWS}
    recorded = (new | {"x/ancapw_f1_se"}) & set(records)
    assert recorded == (new | {"x/ancapw_f1_se"} if working_point else set())
