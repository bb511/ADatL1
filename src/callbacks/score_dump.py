"""Per-event anomaly scores of every evaluation set, written at every validation epoch.

The files feed ``scripts/cchamber/offline_select.py``, which replays the pipeline's
checkpoint selection on them without retraining. Layout of ``dirpath``:

    meta.pt            written once, after the first validation epoch
        target_rate, base_rate, target_fpr, module
        val, test, aux: {"names": [...], "keys": [...],
                         "batch_sizes": {name: [int]}, "labels": {name: [int]}}
        aux_seed, aux_lower, aux_upper: (d,) bounds of the training inputs
        aux_inputs: {"gaussian", "uniform"}: (aux_size, d), normalised input space
    epoch_<nnnn>.pt    one per validation epoch, the sanity check excluded
        epoch
        val, test, aux: {name: {key: (n,) scores in loader order}}
        state_dict: the module's weights and buffers, on CPU

The validation scores are the ``outputs`` the other callbacks receive in
``on_validation_batch_end``. The test and auxiliary sets are scored by a separate pass
through ``model_step`` at the end of each validation epoch.
"""

import os
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, TensorDataset

from src.data.utils import unpack_batch


def _per_event(score, key: str, n: int) -> torch.Tensor:
    score = torch.as_tensor(score).detach().reshape(-1)
    if score.numel() != n:
        raise ValueError(f"'{key}' holds {score.numel()} values for a batch of {n}.")
    return score.cpu()


class _Collector:
    """Per-batch scores of the datasets of one split, in loader order."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.chunks: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.batch_sizes: dict[str, list[int]] = {}
        self.labels: dict[str, set[int]] = {}

    def add(self, name: str, outputs: dict, batch) -> None:
        y = unpack_batch(batch).y
        self.batch_sizes.setdefault(name, []).append(len(y))
        self.labels.setdefault(name, set()).update(y.unique().tolist())
        chunks = self.chunks.setdefault(name, {})
        for key in self.keys:
            if key in outputs:
                score = _per_event(outputs[key], key, len(y))
                chunks.setdefault(key, []).append(score)

    def present_keys(self) -> list[str]:
        return [k for k in self.keys if any(k in c for c in self.chunks.values())]

    def scores(self) -> dict[str, dict[str, torch.Tensor]]:
        """Concatenate the batches; a key is in every batch of the split or in none."""
        keys = self.present_keys()
        for name, chunks in self.chunks.items():
            for key in keys:
                if len(chunks.get(key, [])) != len(self.batch_sizes[name]):
                    raise ValueError(f"'{key}' is missing from batches of '{name}'.")

        return {
            name: {key: torch.cat(chunks[key]) for key in keys}
            for name, chunks in self.chunks.items()
        }

    def layout(self) -> dict:
        return {
            "names": list(self.batch_sizes),
            "keys": self.present_keys(),
            "batch_sizes": self.batch_sizes,
            "labels": {name: sorted(lab) for name, lab in self.labels.items()},
        }


def _aux_inputs(
    lower: torch.Tensor, upper: torch.Tensor, size: int, seed: int
) -> dict[str, torch.Tensor]:
    # A private generator keeps the sets identical across epochs and across runs
    # that share the seed and the training data.
    gen = torch.Generator().manual_seed(seed)
    shape = (size, *lower.shape)
    gaussian = torch.randn(shape, generator=gen)
    uniform = lower + (upper - lower) * torch.rand(shape, generator=gen)
    return {"gaussian": gaussian, "uniform": uniform}


def _aux_loaders(inputs: dict[str, torch.Tensor], batch_size: int) -> dict:
    # The labels are placeholders: model_step reads only the inputs.
    return {
        name: DataLoader(
            TensorDataset(x, torch.zeros(len(x), dtype=torch.long)),
            batch_size=batch_size,
        )
        for name, x in inputs.items()
    }


def _test_loaders(datamodule) -> dict:
    datamodule.setup("test")
    return datamodule.test_dataloader()


def _score_loaders(trainer, pl_module, loaders: dict, keys: list[str]) -> _Collector:
    collector = _Collector(keys)
    for idx, (name, loader) in enumerate(loaders.items()):
        for batch in loader:
            batch = trainer.strategy.batch_to_device(batch, pl_module.device, idx)
            collector.add(name, pl_module.model_step(batch), batch)

    return collector


def _score_in_eval_mode(trainer, pl_module, loaders: dict, keys: list[str]):
    """Score extra loaders in eval mode, leaving the module's mode as it was.

    Iterating a DataLoader, or a model that samples, draws from the global RNG; the
    forked streams keep the training trajectory identical to a run without the dump.
    """
    was_training = pl_module.training
    devices = [pl_module.device] if pl_module.device.type == "cuda" else []
    pl_module.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            return _score_loaders(trainer, pl_module, loaders, keys)
    finally:
        pl_module.train(was_training)


def _save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class ScoreDumpCallback(Callback):
    """Store the per-event scores of every evaluation set at every validation epoch.

    :param dirpath: Directory for ``meta.pt`` and the ``epoch_<nnnn>.pt`` files. It
        must not already hold epoch files, so that two runs are never mixed.
    :param keys: Per-event output keys to store; keys a module does not emit are
        skipped.
    :param aux_seed: Seed of the two auxiliary input sets.
    :param aux_size: Number of points in each auxiliary set.
    :param aux_batch_size: Batch size of the pass over the auxiliary sets.
    """

    def __init__(
        self,
        dirpath: str,
        keys: list[str],
        aux_seed: int,
        aux_size: int = 10000,
        aux_batch_size: int = 1024,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.keys = list(keys)
        self.aux_seed = aux_seed
        self.aux_size = aux_size
        self.aux_batch_size = aux_batch_size
        self._val = _Collector(self.keys)
        self._meta_written = False

    def on_fit_start(self, trainer, pl_module):
        """Build the test and auxiliary loaders, once per run."""
        if trainer.world_size > 1:
            raise RuntimeError("ScoreDumpCallback sees one rank's batches only.")
        if any(self.dirpath.glob("epoch_*.pt")):
            raise RuntimeError(f"{self.dirpath} already holds a score dump.")

        x_train = trainer.datamodule.data_train.tensors[0]
        lower, upper = x_train.min(dim=0).values, x_train.max(dim=0).values
        inputs = _aux_inputs(lower, upper, self.aux_size, self.aux_seed)
        aux = _aux_loaders(inputs, self.aux_batch_size)
        self._loaders = {"test": _test_loaders(trainer.datamodule), "aux": aux}
        self._meta = self._run_meta(pl_module, inputs, lower, upper)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def _run_meta(self, pl_module, inputs: dict, lower, upper) -> dict:
        module_cls = type(pl_module)
        return {
            "target_rate": pl_module.hparams.target_rate,
            "base_rate": getattr(pl_module.hparams, "base_rate", None),
            "target_fpr": pl_module.compute_target_fpr(),
            "module": f"{module_cls.__module__}.{module_cls.__qualname__}",
            "aux_seed": self.aux_seed,
            "aux_lower": lower,
            "aux_upper": upper,
            "aux_inputs": inputs,
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val = _Collector(self.keys)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking:
            return

        name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self._val.add(name, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the test and auxiliary sets, then write the epoch's file."""
        if trainer.sanity_checking:
            return

        collectors = {"val": self._val}
        for split, loaders in self._loaders.items():
            collectors[split] = _score_in_eval_mode(
                trainer, pl_module, loaders, self.keys
            )

        epoch = trainer.current_epoch
        record = {split: c.scores() for split, c in collectors.items()}
        record["state_dict"] = {
            k: v.detach().cpu() for k, v in pl_module.state_dict().items()
        }
        _save_atomic({"epoch": epoch, **record}, self.dirpath / f"epoch_{epoch:04d}.pt")
        self._write_meta(collectors)

    def _write_meta(self, collectors: dict) -> None:
        if self._meta_written:
            return

        layouts = {split: c.layout() for split, c in collectors.items()}
        _save_atomic({**self._meta, **layouts}, self.dirpath / "meta.pt")
        self._meta_written = True
