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
raises. The extra sets of the validation pass (the auxiliary sets, the training set,
the SDS pseudo-sets and the points files) are wrapped here in the loaders that pass
appends to the validation loaders; the pseudo-sets and the training set are served
from the dump where it stores their scores and rescored otherwise. The points are
rescored from the weights at every epoch: a dump made with score_dump's ``points``
stores their scores under the split ``points``, which no pass serves (``PASS_PARTS``,
``OPTIONAL_PARTS``). An epoch file without validation normal (the physics dumps store
it at the sanity pass alone, a dump of no stored set never) has it rescored before the
passes, in a pass of its own that ``ScoresCallback`` collects, and handed to the
module by ``add_scores``.
"""

from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule
from torch.utils.data import DataLoader

from src.algorithms.utils.object_feature_map_loader import maybe_get_object_feature_map
from src.callbacks.components.rate import AnomalyRate
from src.callbacks.components.utils import drawn_masks, placeholder_loader
from src.callbacks.score_dump import _Collector, aux_inputs
from src.data.utils import unpack_batch

# The parts of an epoch's record that each pass serves; the loader names of one pass
# are unique across its parts.
PASS_PARTS = {"val": ("val", "aux"), "test": ("test",)}
# The parts served where an epoch's record holds them: dumps written without the
# training pass or the SDS pseudo-sets lack theirs.
OPTIONAL_PARTS = {"val": ("train", "sds"), "test": ()}
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


def _typical_record(val: dict, typical: str | None) -> dict:
    """``as_typical`` of an epoch's stored validation sets; where the dump does not
    store ``typical`` (the wind tunnel's ``sensor_osr``), the stored normal is left out
    instead, so that the set served as normal is rescored from the epoch's weights."""
    if typical is None or typical in val:
        return as_typical(val, typical)
    return {name: s for name, s in val.items() if name != "normal"}


def _typical_meta(meta: dict, typical: str | None) -> dict:
    if typical is None:
        return meta
    sizes = as_typical(meta["val"]["batch_sizes"], typical)
    labels = as_typical(meta["val"]["labels"], typical)
    val = {**meta["val"], "names": list(sizes), "batch_sizes": sizes, "labels": labels}
    return {**meta, "val": val}


def operational_cut(
    scores: torch.Tensor, meta: dict, target_rate: float | None = None
) -> tuple[torch.Tensor, ...]:
    """Threshold and tie weight at ``target_rate``, the run's target rate when None, as
    the training efficiency callback sets them on validation normal."""
    target_rate = meta["target_rate"] if target_rate is None else target_rate
    rate = AnomalyRate(target_rate, meta["base_rate"])
    rate.set_threshold(scores)
    return rate.threshold.clone(), rate.tie_weight.clone()


def cut_rates(meta: dict, rates: list[float]) -> dict[str, float]:
    """Every rate whose cut the replay sets, by the suffix of its ``thres_<suffix>`` and
    ``tie_<suffix>``: ``operational`` for the run's target rate and, for each other
    rate, the suffix of the training efficiency callback's buffers (286.088064 ->
    ``286_088064kHz``), which ``AnomalyEfficiencyCallback._get_thres`` reads."""
    target = float(meta["target_rate"])
    suffixes = {"operational": target}
    for rate in map(float, rates):
        if abs(rate - target) >= 1e-12:
            suffixes[f"{rate}kHz".replace(".", "_")] = rate
    return suffixes


class ScoresCallback(Callback):
    """Collect the per-event outputs under ``keys`` of every batch of a test pass, in
    ``collector``, a ``_Collector`` of the score dump."""

    def __init__(self, keys: list[str]):
        super().__init__()
        self.keys = list(keys)

    def on_test_epoch_start(self, trainer, pl_module):
        self.collector = _Collector(self.keys)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        name = list(trainer.test_dataloaders)[dataloader_idx]
        self.collector.add(name, outputs, batch)


class ReplayModule(LightningModule):
    """A run's module whose test step serves the stored scores of a dump.

    Call ``load_epoch`` with an epoch's record and ``serve`` with the pass's split
    before each ``trainer.test``. The loaders of a stored set must be those of the
    dump, batch for batch: a batch whose length or position differs from ``meta``
    raises, since it means the rebuilt loader is not the dump's. A record without
    ``state_dict`` leaves the weights unloaded, and a batch that would be rescored from
    them, of a set the record does not store or for a key of ``RESCORED_KEYS`` it
    lacks, raises RuntimeError naming the set and the epoch file. ``thres_<suffix>``
    and ``tie_<suffix>`` of every rate of ``cut_rates`` are set from the epoch's
    validation normal ``ascore/full`` (None while the record lacks it), and
    ``aux_lower`` and ``aux_upper`` from ``meta``, for the callbacks that read them.

    :param module: The run's LightningModule, freshly built from its configuration.
    :param meta: The dump's ``meta.pt``.
    :param typical: A stored validation set served as ``normal`` in place of the
        dump's, which the validation pass then leaves out; the cut and the rescored
        keys of ``normal`` follow it. None serves the dump's ``normal``.
    :param rates: Target rates besides the run's own whose cuts are set as well, as the
        training efficiency callback sets those of its ``target_rates``.
    """

    def __init__(
        self,
        module: LightningModule,
        meta: dict,
        typical: str | None = None,
        rates: list[float] = (),
    ):
        super().__init__()
        hparams = {"target_rate": meta["target_rate"], "base_rate": meta["base_rate"]}
        self.save_hyperparameters(hparams, logger=False)
        self.module = module
        self.typical, self.meta = typical, _typical_meta(meta, typical)
        self.cuts = cut_rates(meta, rates)
        self.aux_lower, self.aux_upper = meta["aux_lower"], meta["aux_upper"]
        # The VAE builds its KL schedule at fit start; its scores do not depend on it.
        if hasattr(module, "_setup_kl_annealing"):
            module._setup_kl_annealing(module.kl_warmup_frac, 1)

    @property
    def target_fpr(self) -> float:
        return self.module.target_fpr

    def compute_operational_ascore(self, ascore: torch.Tensor) -> float:
        return self.module.compute_operational_ascore(ascore)

    def on_test_start(self) -> None:
        # Lightning calls the hooks of this module alone. The wrapped module's hook
        # injects the object feature map of the L1AD loaders, which DSAE reads, and
        # raises where the loaders carry none.
        if maybe_get_object_feature_map(self) is not None:
            self.module.on_test_start()

    def load_epoch(self, record: dict, ckpt_path: Path) -> None:
        """Load an epoch's scores and, where the record holds them, its weights;
        ``ckpt_path`` names the epoch to the callbacks, which take ``initial`` or
        ``epoch_<nnnn>`` from its stem.

        A record without validation normal must hold
        the weights, from which normal is rescored for ``add_scores``.
        """
        self._has_weights = "state_dict" in record
        if self._has_weights:
            _load_weights(self.module, record["state_dict"])
        record = {**record, "val": _typical_record(record["val"], self.typical)}
        self.record = record
        self._ckpt_path = Path(ckpt_path)
        if "normal" not in record["val"] and not self._has_weights:
            raise ValueError("The dump does not store validation normal.")
        self._set_cuts()

    def add_scores(self, scores: dict[str, dict[str, torch.Tensor]]) -> None:
        """Add the scores of validation sets rescored from the epoch's weights to its
        record, served from then on as stored ones, and set the cuts again."""
        self.record = {**self.record, "val": {**self.record["val"], **scores}}
        self._set_cuts()

    def _set_cuts(self) -> None:
        normal = self.record["val"].get("normal")
        for suffix, rate in self.cuts.items():
            cut = (None, None)
            if normal is not None:
                cut = operational_cut(normal["ascore/full"], self.meta, rate)
            setattr(self, f"thres_{suffix}", cut[0])
            setattr(self, f"tie_{suffix}", cut[1])

    def serve(self, split: str) -> None:
        """Serve the stored sets of the 'val' or the 'test' pass."""
        optional = [part for part in OPTIONAL_PARTS[split] if part in self.record]
        parts = (*PASS_PARTS[split], *optional)
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


def _point_loaders(points: dict, batch_size: int, taken: dict) -> dict:
    clash = sorted(points.keys() & taken.keys())
    if clash:
        raise ValueError(f"The points {clash} take the names of other extra sets.")
    return {
        name: placeholder_loader(x, batch_size, mask)
        for name, (x, mask) in points.items()
    }


def extra_loaders(
    meta: dict,
    train: DataLoader | None,
    pseudo: dict | None,
    aux: bool = True,
    mask: torch.Tensor | None = None,
    points: dict | None = None,
) -> dict:
    """The loaders the validation pass appends after the validation loaders.

    In order: ``gaussian``, ``uniform`` and ``autouad``, regenerated from ``meta``
    (when ``aux``), each point masked as the dump masked it where ``meta`` has
    ``aux_masked``, with the masks of the training rows that ``drawn_masks(mask,
    aux_size, aux_seed)`` draws from the training masks ``mask``; ``train``, a loader
    of training rows, when not None; the pseudo-sets of ``pseudo``, whose
    ``<set>_mask`` entries, as ``src.callbacks.sds.sds_inputs`` returns them, are the
    data masks of their points; and a loader per entry ``{name: (x, mask)}`` of
    ``points`` (``rebuild.cached_points``), in file order, each point with its own
    mask, or none. The made-up points carry label 0 and the dump's auxiliary batch
    size, which makes the stored scores of every set servable and scores the points
    in the batches of score_dump's ``points``.

    Raises:
        ValueError: If a name of ``points`` is already that of another loader, whose
            stored scores it would be served.
    """
    sizes = meta["aux"]["batch_sizes"]
    loaders = {}
    if aux:
        masks = None
        if meta.get("aux_masked"):
            masks = drawn_masks(mask, meta["aux_size"], meta["aux_seed"])
        for name, x in aux_inputs(meta).items():
            loaders[name] = placeholder_loader(x, sizes[name][0], masks)
    if train is not None:
        loaders["train"] = train
    for name, x in (pseudo or {}).items():
        if not name.endswith("_mask"):
            point_masks = pseudo.get(f"{name}_mask")
            loaders[name] = placeholder_loader(x, sizes["gaussian"][0], point_masks)
    return loaders | _point_loaders(points or {}, sizes["gaussian"][0], loaders)
