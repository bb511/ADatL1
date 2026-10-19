from pathlib import Path

import pytest

from src.callbacks.checkpointing.criterion import Max
from src.callbacks.model_checkpoint import SingleDatasetModelCheckpoint

CALLBACKS = {
    "single": (SingleDatasetModelCheckpoint, {}),
}


@pytest.fixture(params=sorted(CALLBACKS))
def stored_ckpt(request, tmp_path: Path) -> tuple[Path, object]:
    cls, kwargs = CALLBACKS[request.param]
    ckpt = tmp_path / request.param / "eff" / "max" / "ds=normal.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    callback = cls(
        monitor="eff",
        criterion=Max(top_k=1),
        ds=["normal"],
        dirpath=str(tmp_path),
        **kwargs,
    )
    return ckpt, callback


def test_construction_keeps_stored_checkpoints(stored_ckpt):
    ckpt, _ = stored_ckpt
    assert ckpt.read_bytes() == b"weights"


def test_fit_start_empties_checkpoint_dir(stored_ckpt):
    ckpt, callback = stored_ckpt
    callback.on_fit_start(trainer=None, pl_module=None)

    assert callback.dirpath == ckpt.parent
    assert ckpt.parent.is_dir()
    assert not any(ckpt.parent.iterdir())
