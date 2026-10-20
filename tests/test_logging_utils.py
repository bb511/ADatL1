"""log_hyperparameters must hand the loggers resolved values, not ``${...}`` strings."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.utils.logging_utils import log_hyperparameters, resolve_for_logging


def _pareto_like_cfg():
    return OmegaConf.create(
        {
            "seed": "${pareto_study.candidate.autoencoder_seed}",
            "pareto_study": {
                "candidate": {"autoencoder_seed": 180524, "mi_gamma": 0.1, "encoder_nodes": [64, 32, 8]}
            },
            "algorithm": {
                "mi_gamma": "${pareto_study.candidate.mi_gamma}",
                "encoder": {"nodes": "${pareto_study.candidate.encoder_nodes}"},
            },
            "callbacks": {"losses": {"gamma": "${algorithm.mi_gamma}"}},
            "data": {"seed": "${seed}"},
            "trainer": {"max_epochs": 3},
        }
    )


def test_resolves_interpolations():
    out = resolve_for_logging(_pareto_like_cfg())
    assert out["seed"] == 180524
    assert out["algorithm"]["mi_gamma"] == 0.1
    assert out["algorithm"]["encoder"]["nodes"] == [64, 32, 8]
    assert out["callbacks"]["losses"]["gamma"] == 0.1
    assert out["data"]["seed"] == 180524


def test_unresolvable_values_stay_raw_and_the_rest_resolves(monkeypatch):
    monkeypatch.delenv("ADATL1_TEST_UNSET_VAR", raising=False)
    cfg = OmegaConf.create(
        {
            "a": 1,
            "b": "${a}",
            "env": "${oc.env:ADATL1_TEST_UNSET_VAR}",
            "nested": {"ok": "${a}", "bad": "${does.not.exist}"},
            "lst": ["${a}", "${does.not.exist}"],
            "todo": "???",
        }
    )
    out = resolve_for_logging(cfg)
    assert out["b"] == 1
    assert out["env"] == "${oc.env:ADATL1_TEST_UNSET_VAR}"
    assert out["nested"] == {"ok": 1, "bad": "${does.not.exist}"}
    assert out["lst"] == [1, "${does.not.exist}"]
    assert out["todo"] == "???"


def test_plain_values_pass_through():
    assert resolve_for_logging({"a": "${b}"}) == {"a": "${b}"}
    assert resolve_for_logging(None) is None


class _Logger:
    def __init__(self):
        self.logged = None

    def log_hyperparams(self, params):
        self.logged = params


def test_log_hyperparameters_logs_resolved_values():
    logger = _Logger()
    trainer = SimpleNamespace(logger=logger, loggers=[logger])
    algorithm = SimpleNamespace(parameters=lambda: [])
    log_hyperparameters({"cfg": _pareto_like_cfg(), "algorithm": algorithm, "trainer": trainer})

    hp = logger.logged
    assert hp["seed"] == 180524
    assert hp["algorithm"]["mi_gamma"] == 0.1
    assert hp["callbacks"]["losses"]["gamma"] == 0.1
    assert hp["data"]["seed"] == 180524
    assert "${" not in repr(hp)
