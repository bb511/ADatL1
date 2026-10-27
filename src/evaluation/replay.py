"""Serve the stored scores of a score dump to the evaluation callbacks, epoch by epoch.

A run trained in the dump regime keeps no checkpoint: its per-epoch files hold the
scores of the stored sets and the weights (``src/callbacks/score_dump.py`` gives the
layout). ``ReplayModule`` wraps the run's own LightningModule, loads the weights of
one epoch at a time, and answers ``trainer.test`` batch by batch: with the stored
scores of a set the files hold, and with ``model_step`` of the loaded weights for any
other set, so that every evaluation callback runs unchanged over both. A key of
``RESCORED_KEYS`` that a stored set lacks is rescored from the weights as well. An
epoch file without ``state_dict``, as in a scores-only copy of a dump, is replayed
from its stored scores alone: a set or key that would be rescored from its weights
raises. The extra sets of the validation pass (the auxiliary sets, the training set
and the SDS pseudo-sets) are wrapped here in the loaders that pass appends to the
validation loaders.
"""

from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.components.rate import AnomalyRate
from src.callbacks.score_dump import aux_inputs
from src.data.utils import unpack_batch

# The parts of an epoch's record that each pass serves; the loader names of one pass
# are unique across its parts.
PASS_PARTS = {"val": ("val", "aux", "train"), "test": ("test",)}
# The keys rescored from the weights, by part and stored set, where the dump lacks them,
# as dumps written before fit/full joined the dump keys do: the fit callback reads
# validation normal alone.
RESCORED_KEYS = {"val": {"normal": ("fit/full",)}}


def _load_weights(module: LightningModule, state_dict: dict) -> None:
    """Load a stored state strictly, as Evaluator.evaluate_ckpt loads a checkpoint.

    Top-level tensors that training registered on the module (threshold buffers, the
    SVDD center) and that a freshly built module lacks are registered first.
    """
    module_keys = set(module.state_dict())
    for key, value in state_dict.items():
        if key not in module_keys and "." not in key and torch.is_tensor(value):
            module.register_buffer(key, value.clone(), persistent=True)
    module.load_state_dict(state_dict, strict=True)


def _offsets(batch_sizes: list[int]) -> list[int]:
    offsets = [0]
    for size in batch_sizes:
        offsets.append(offsets[-1] + size)
    return offsets


def _missing_keys(record: dict, parts: tuple[str, ...]) -> dict[str, list[str]]:
    """The keys of ``RESCORED_KEYS`` that each stored set of ``parts`` lacks."""
    return {
        name: [k for k in RESCORED_KEYS.get(part, {}).get(name, ()) if k not in scores]
        for part in parts
        for name, scores in record[part].items()
    }


def as_typical(sets: dict, typical: str | None) -> dict:
    """``sets`` with the set ``typical`` renamed ``normal`` and placed first, the
    original ``normal`` left out; ``sets`` itself for ``typical`` None."""
    if typical is None:
        return sets
    rest = {name: s for name, s in sets.items() if name not in ("normal", typical)}
    return {"normal": sets[typical], **rest}


def _typical_meta(meta: dict, typical: str | None) -> dict:
    if typical is None:
        return meta
    sizes = as_typical(meta["val"]["batch_sizes"], typical)
    labels = as_typical(meta["val"]["labels"], typical)
    val = {**meta["val"], "names": list(sizes), "batch_sizes": sizes, "labels": labels}
    return {**meta, "val": val}


def operational_cut(scores: torch.Tensor, meta: dict) -> tuple[torch.Tensor, ...]:
    """Threshold and tie weight at the run's target rate, as the training efficiency
    callback sets them on validation normal."""
    rate = AnomalyRate(meta["target_rate"], meta["base_rate"])
    rate.set_threshold(scores)
    return rate.threshold.clone(), rate.tie_weight.clone()


class ReplayModule(LightningModule):
    """A run's module whose test step serves the stored scores of a dump.

    Call ``load_epoch`` with an epoch's record and ``serve`` with the pass's split
    before each ``trainer.test``. The loaders of a stored set must be those of the
    dump, batch for batch: a batch whose length or position differs from ``meta``
    raises, since it means the rebuilt loader is not the dump's. A record without
    ``state_dict`` leaves the weights unloaded, and a batch that would be rescored from
    them, of a set the record does not store or for a key of ``RESCORED_KEYS`` it
    lacks, raises RuntimeError naming the set and the epoch file. ``thres_operational``
    and ``tie_operational`` are set from the epoch's validation normal ``ascore/full``,
    and ``aux_lower`` and ``aux_upper`` from ``meta``, for the callbacks that read them.

    :param module: The run's LightningModule, freshly built from its configuration.
    :param meta: The dump's ``meta.pt``.
    :param typical: A stored validation set served as ``normal`` in place of the
        dump's, which the validation pass then leaves out; the cut and the rescored
        keys of ``normal`` follow it. None serves the dump's ``normal``.
    """

    def __init__(self, module: LightningModule, meta: dict, typical: str | None = None):
        super().__init__()
        rates = {"target_rate": meta["target_rate"], "base_rate": meta["base_rate"]}
        self.save_hyperparameters(rates, logger=False)
        self.module = module
        self.typical, self.meta = typical, _typical_meta(meta, typical)
        self.aux_lower, self.aux_upper = meta["aux_lower"], meta["aux_upper"]
        # The VAE builds its KL schedule at fit start; its scores do not depend on it.
        if hasattr(module, "_setup_kl_annealing"):
            module._setup_kl_annealing(module.kl_warmup_frac, 1)

    @property
    def target_fpr(self) -> float:
        return self.module.target_fpr

    def compute_operational_ascore(self, ascore: torch.Tensor) -> float:
        return self.module.compute_operational_ascore(ascore)

    def load_epoch(self, record: dict, ckpt_path: Path) -> None:
        """Load an epoch's scores and, where the record holds them, its weights;
        ``ckpt_path`` names the epoch to the callbacks, which take ``initial`` or
        ``epoch_<nnnn>`` from its stem."""
        self._has_weights = "state_dict" in record
        if self._has_weights:
            _load_weights(self.module, record["state_dict"])
        record = {**record, "val": as_typical(record["val"], self.typical)}
        self.record = record
        self._ckpt_path = Path(ckpt_path)
        if "normal" not in record["val"]:
            raise ValueError("The dump does not store validation normal.")
        cut = operational_cut(record["val"]["normal"]["ascore/full"], self.meta)
        self.thres_operational, self.tie_operational = cut

    def serve(self, split: str) -> None:
        """Serve the stored sets of the 'val' or the 'test' pass."""
        parts = PASS_PARTS[split]
        self._served = {
            name: scores for part in parts for name, scores in self.record[part].items()
        }
        self._sizes = {
            name: sizes
            for part in parts
            for name, sizes in self.meta[part]["batch_sizes"].items()
        }
        self._offsets = {name: _offsets(self._sizes[name]) for name in self._served}
        self._missing = _missing_keys(self.record, parts)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        name = list(self.trainer.test_dataloaders)[dataloader_idx]
        if name not in self._served:
            return self._model_step(batch, f"'{name}'")

        n = len(unpack_batch(batch).y)
        self._check_batch(name, batch_idx, n, dataloader_idx)
        start = self._offsets[name][batch_idx]
        out = {key: s[start : start + n] for key, s in self._served[name].items()}
        out.update(self._rescored(name, batch))
        if "ascore/full" in out:
            ascore = self.compute_operational_ascore(out["ascore/full"])
            out["ascore/operational"] = ascore
        return out

    def _model_step(self, batch, rescored: str) -> dict:
        """``model_step`` of the epoch's weights, which raises RuntimeError naming
        ``rescored`` where the record holds none."""
        if not self._has_weights:
            epoch_file = Path(self._ckpt_path.parent.name, self._ckpt_path.name)
            raise RuntimeError(
                f"Rescoring {rescored} needs the weights, which {epoch_file} does not "
                "hold: a scores-only dump serves its stored scores alone."
            )
        return self.module.model_step(batch)

    def _rescored(self, name: str, batch) -> dict:
        keys = self._missing[name]
        if not keys:
            return {}
        outputs = self._model_step(batch, f"{keys} of '{name}'")
        return {key: outputs[key] for key in keys}

    def _check_batch(self, name: str, batch_idx: int, n: int, dataloader_idx: int):
        sizes = self._sizes[name]
        last = batch_idx == self.trainer.num_test_batches[dataloader_idx] - 1
        stored_last = batch_idx == len(sizes) - 1
        if batch_idx >= len(sizes) or sizes[batch_idx] != n or last != stored_last:
            raise RuntimeError(
                f"Batch {batch_idx} of '{name}' holds {n} rows, which the dump's "
                f"layout {sizes} does not: the loader is not the dump's."
            )


def _zero_labelled(x: torch.Tensor, batch_size: int) -> DataLoader:
    labels = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, labels), batch_size=batch_size)


def extra_loaders(meta: dict, data_train, sds: dict | None, aux: bool = True):
    """The loaders the validation pass appends after the validation loaders.

    In order: ``gaussian``, ``uniform`` and ``autouad``, regenerated from ``meta``
    (when ``aux``); ``train``, the training set unshuffled with the dump's batch size
    (when ``data_train`` is not None); ``sds_targets`` and ``sds_outliers`` (when
    ``sds``, a result of ``src.callbacks.sds.sds_inputs``, is not None). Every set
    carries label 0; the dump's batch sizes make the stored scores of the first four
    servable.
    """
    sizes = {**meta["aux"]["batch_sizes"], **meta["train"]["batch_sizes"]}
    loaders = {}
    if aux:
        for name, x in aux_inputs(meta).items():
            loaders[name] = _zero_labelled(x, sizes[name][0])
    if data_train is not None:
        loaders["train"] = DataLoader(data_train, batch_size=sizes["train"][0])
    for name, x in (sds or {}).items():
        loaders[name] = _zero_labelled(x, sizes["gaussian"][0])
    return loaders
