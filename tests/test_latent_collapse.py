import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.evaluation.callbacks.latent_collapse import (
    LatentCollapseDiagnosticsCallback,
)


def make_callback(**overrides) -> LatentCollapseDiagnosticsCallback:
    defaults = {
        "protocol_version": "fet-et-pareto-v1",
        "configuration_id": "candidate-configuration",
        "autoencoder_seed": 123,
        "architecture_id": "h64_32",
        "minimum_joint_code_entropy_bits": 1.0,
        "minimum_fraction_of_paired_gamma_zero_joint_entropy": 0.5,
        "ckpts": {"loss_total": True},
    }
    return LatentCollapseDiagnosticsCallback(**(defaults | overrides))


def test_latent_collapse_metrics_use_hard_code_frequencies() -> None:
    callback = make_callback()
    callback._bit_sums = None
    callback._latent_width = None
    callback._code_counts = Counter()
    callback._n_events = 0

    callback._accumulate_latent_sample(
        torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=torch.float32,
        )
    )

    metrics = callback._metrics()

    assert metrics["latent_width"] == 2
    assert metrics["bits"][0]["activation_probability"] == pytest.approx(0.5)
    assert metrics["bits"][0]["binary_entropy_bits"] == pytest.approx(1.0)
    assert metrics["bits"][1]["activation_probability"] == pytest.approx(0.75)
    assert metrics["summed_marginal_bit_entropy_bits"] == pytest.approx(
        1.811278124459133
    )
    assert metrics["joint_code_entropy_bits"] == pytest.approx(1.5)
    assert metrics["observed_code_count"] == 3
    assert metrics["effective_code_count"] == pytest.approx(2**1.5)


def test_collapsed_code_writes_visible_failed_artifact(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "loss_total.ckpt"
    checkpoint_path.write_bytes(b"checkpoint-contents")
    callback = make_callback()
    callback._bit_sums = None
    callback._latent_width = None
    callback._code_counts = Counter()
    callback._n_events = 0
    callback._event_component_manifests = {
        "cached_data": hashlib.sha256(b"normal-validation-events")
    }
    callback._event_layouts = {}
    callback._accumulate_latent_sample(torch.zeros((4, 2), dtype=torch.float32))
    expected_manifest_hash, _ = callback._event_manifest()

    output_path = tmp_path / "collapse_summary.json"
    callback._write_summary(
        output_path,
        checkpoint=callback._checkpoint_identity(checkpoint_path),
        trainer=SimpleNamespace(split="val"),
        pl_module=SimpleNamespace(
            bernoulli=SimpleNamespace(threshold=torch.tensor(0.5))
        ),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["checkpoint"]["name"] == "loss_total.ckpt"
    assert payload["checkpoint"]["sha256"] == hashlib.sha256(
        b"checkpoint-contents"
    ).hexdigest()
    assert payload["evaluation"]["event_manifest_hash"] == expected_manifest_hash
    assert payload["evaluation"]["event_manifest_components"] == {
        "cached_data": hashlib.sha256(b"normal-validation-events").hexdigest()
    }
    assert payload["metrics"]["joint_code_entropy_bits"] == pytest.approx(0.0)
    assert payload["metrics"]["observed_code_count"] == 1
    assert payload["decision"]["pass"] is False
    assert payload["decision"]["reason"] == "joint_code_entropy_below_minimum"
    assert payload["decision"]["paired_baseline_comparison"] == (
        "deferred_to_phase_2_aggregation"
    )


def test_latent_collapse_rejects_nonbinary_samples() -> None:
    with pytest.raises(RuntimeError, match="hard zero/one"):
        LatentCollapseDiagnosticsCallback._binary_sample(
            {"latent_sample": torch.tensor([[0.5]])},
            expected_batch_size=1,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"autoencoder_seed": 123.0},
        {"minimum_fraction_of_paired_gamma_zero_joint_entropy": 1.01},
    ],
)
def test_latent_collapse_rejects_invalid_policy_values(overrides: dict) -> None:
    with pytest.raises(ValueError):
        make_callback(**overrides)


def test_event_manifest_is_independent_of_validation_batch_boundaries() -> None:
    full_batch = make_callback()
    full_batch._event_component_manifests = {}
    full_batch._event_layouts = {}
    full_batch._n_events = 2
    full_batch._update_event_manifest(
        "cached_data",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    split_batches = make_callback()
    split_batches._event_component_manifests = {}
    split_batches._event_layouts = {}
    split_batches._n_events = 2
    split_batches._update_event_manifest(
        "cached_data",
        torch.tensor([[1.0, 2.0]]),
    )
    split_batches._update_event_manifest(
        "cached_data",
        torch.tensor([[3.0, 4.0]]),
    )

    assert full_batch._event_manifest() == split_batches._event_manifest()


def test_latent_collapse_callback_writes_loss_total_validation_artifact(
    tmp_path: Path,
) -> None:
    class FakeModel:
        training = False
        bernoulli = SimpleNamespace(threshold=torch.tensor(0.5))

        def __init__(self, checkpoint_path: Path) -> None:
            self._ckpt_path = checkpoint_path

        def forward_with_representations(self, x: torch.Tensor):
            return {"latent_sample": (x >= 0).to(dtype=torch.float32)}

    checkpoint_path = tmp_path / "loss_total.ckpt"
    checkpoint_path.write_bytes(b"checkpoint-contents")
    callback = make_callback()
    trainer = SimpleNamespace(
        split="val",
        strat_name="loss_total",
        metric_name=None,
        criterion_name=None,
        test_dataloaders={"normal": object()},
    )
    batch = (
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        torch.ones((2, 2), dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2),
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        torch.ones((2, 2), dtype=torch.bool),
    )

    callback.on_test_epoch_start(trainer, FakeModel(checkpoint_path))
    callback.on_test_batch_end(
        trainer,
        FakeModel(checkpoint_path),
        outputs={},
        batch=batch,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer, FakeModel(checkpoint_path))

    artifact_path = (
        tmp_path
        / "plots"
        / "val"
        / "loss_total"
        / "latent_collapse"
        / "collapse_summary.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["evaluation"]["dataset"] == "normal"
    assert payload["evaluation"]["source_split"] == "valid"
    assert payload["representation"]["name"] == "latent_sample"
    assert payload["representation"]["bernoulli_probability_threshold"] == 0.5
    assert payload["metrics"]["joint_code_entropy_bits"] == pytest.approx(1.0)
    assert payload["decision"]["pass"] is True


def test_latent_collapse_runs_only_for_loss_total_validation() -> None:
    callback = make_callback()

    assert callback._should_run_for_current_ckpt(
        SimpleNamespace(strat_name="loss_total", metric_name=None, criterion_name=None)
    )
    assert not callback._should_run_for_current_ckpt(
        SimpleNamespace(strat_name="last", metric_name=None, criterion_name=None)
    )
