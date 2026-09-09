import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluation.artifact_provenance import (
    EventManifest,
    checkpoint_identity_from_trainer,
    data_cache_identity,
    make_artifact_provenance,
)


def _datamodule(tmp_path: Path):
    return SimpleNamespace(
        main_cache_folder=tmp_path / "mlready-cache",
        control_object_feature_map={"FET": {"Et": [0]}},
    )


def _batch(values: list[list[float]]):
    x = torch.tensor(values)
    return (
        x,
        torch.ones_like(x, dtype=torch.bool),
        torch.zeros(x.shape[0], dtype=torch.bool),
        torch.zeros(x.shape[0]),
        x,
        torch.ones_like(x, dtype=torch.bool),
    )


def test_event_manifest_is_independent_of_evaluation_batch_boundaries() -> None:
    full = EventManifest("valid")
    full.update_batch("normal", _batch([[1.0], [2.0]]))

    split = EventManifest("valid")
    split.update_batch("normal", _batch([[1.0]]))
    split.update_batch("normal", _batch([[2.0]]))

    assert full.payload() == split.payload()


def test_provenance_envelope_uses_shared_checkpoint_cache_and_event_identity(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "loss_total.ckpt"
    checkpoint_path.write_bytes(b"frozen checkpoint")
    trainer = SimpleNamespace()
    checkpoint = checkpoint_identity_from_trainer(trainer, checkpoint_path)
    assert checkpoint == checkpoint_identity_from_trainer(trainer, checkpoint_path)
    assert checkpoint["sha256"] == hashlib.sha256(b"frozen checkpoint").hexdigest()

    manifest = EventManifest("valid")
    manifest.update_batch("normal", _batch([[1.0], [2.0]]))
    payload = make_artifact_provenance(
        {
            "protocol_version": "fet-et-pareto-v1",
            "configuration_id": "configuration-without-seed",
            "autoencoder_seed": 123,
        },
        checkpoint=checkpoint,
        evaluation_mode="validation",
        data_cache=data_cache_identity(_datamodule(tmp_path)),
        event_manifest=manifest.payload(),
    )

    assert payload == {
        "schema_version": 1,
        "protocol_version": "fet-et-pareto-v1",
        "configuration_id": "configuration-without-seed",
        "autoencoder_seed": 123,
        "checkpoint": checkpoint,
        "evaluation_mode": "validation",
        "data": {
            "cache": data_cache_identity(_datamodule(tmp_path)),
            "event_manifest": manifest.payload(),
        },
    }
