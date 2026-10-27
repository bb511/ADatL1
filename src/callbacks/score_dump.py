"""Per-event scores and weights of a training run, written at every validation pass.

The files of a run sit by default in ``lattices/<domain>/<lattice>/<run>/`` under the
project root: ``<domain>`` is the name of the data config, ``<lattice>`` the experiment
name and ``<run>`` the run name. ``src/eval.py`` reads ``lattices/<domain>`` as its
``scores_dir``.

Generate a dump after a run finishes with all the details necessary to replay the
evaluation on top of it:

- ``config.yaml``, ``overrides.yaml``: the run's Hydra files.
- ``meta.pt``: ``target_rate``, ``base_rate``, ``target_fpr`` and ``module`` (the class
  of the LightningModule); for each split (``val``, ``test``, ``aux`` and, with the
  training pass, ``train``) the ``names`` of its loaders, the ``keys`` the files hold,
  and the ``batch_sizes`` and ``labels`` of every set; ``stored``, the val and test
  sets whose scores are kept at every pass; ``initial_sets``, the val sets kept at the
  sanity pass alone; ``aux_seed``, ``aux_size`` and the training moments
  ``aux_lower``, ``aux_upper``, ``aux_mean``, ``aux_std``, each of the shape of one
  input; ``aux_masked``, whether the auxiliary points carry data masks.
- ``initial.pt``: the sanity pass on the untrained weights (``epoch: -1``), which must
  cover the whole validation set (``num_sanity_val_steps=-1``); its ``val`` holds the
  ``initial_sets`` besides the stored sets.
- ``epoch_<nnnn>.pt``: ``epoch``; ``val`` and ``test``, ``{set: {key: (n,)}}`` for the
  stored sets; ``aux``, the scores of ``gaussian``, ``uniform`` and ``autouad``;
  ``train``, with the training pass only, the per-sample ``loss/full`` in the row order
  of ``data_train``; ``state_dict``, every weight and buffer on CPU.

On data with a mask (the L1AD data), the i-th point of each auxiliary set takes the
mask of the i-th training row that ``drawn_masks(mask, aux_size, aux_seed)`` draws and
is +0 wherever that mask is 0, as the points of ``NPDCallback`` are; ``aux_inputs`` and
``drawn_masks`` over the training masks rebuild them.
"""

import os
import shutil
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from src.callbacks.components.utils import drawn_masks, placeholder_loader
from src.data.utils import unpack_batch

# The per-sample loss, the only quantity stored for the training set.
TRAIN_KEYS = ["loss/full"]
HYDRA_FILES = ["config.yaml", "overrides.yaml"]


def _per_event(score, key: str, n: int) -> torch.Tensor:
    score = torch.as_tensor(score).detach().reshape(-1)
    if score.numel() != n:
        raise ValueError(f"'{key}' holds {score.numel()} values for a batch of {n}.")
    return score.cpu()


class _Collector:
    """Per-batch scores of the stored sets of one split, and the layout of every set.

    ``stored`` None stores every set; a set outside it contributes its batch sizes and
    labels only.
    """

    def __init__(self, keys: list[str], stored: list[str] | None = None):
        self.keys, self.stored = keys, stored
        self.chunks: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.batch_sizes: dict[str, list[int]] = {}
        self.labels: dict[str, set[int]] = {}

    def add(self, name: str, outputs: dict, batch) -> None:
        y = unpack_batch(batch).y
        self.batch_sizes.setdefault(name, []).append(len(y))
        self.labels.setdefault(name, set()).update(y.unique().tolist())
        if self.stored is not None and name not in self.stored:
            return
        chunks = self.chunks.setdefault(name, {})
        for key in self.keys:
            if key in outputs:
                chunks.setdefault(key, []).append(_per_event(outputs[key], key, len(y)))

    def present_keys(self) -> list[str]:
        return [k for k in self.keys if any(k in c for c in self.chunks.values())]

    def scores(self) -> dict[str, dict[str, torch.Tensor]]:
        """Concatenate the batches; a key is in every batch of a set or in none."""
        keys = self.present_keys()
        for name, chunks in self.chunks.items():
            n_batches = len(self.batch_sizes[name])
            short = [k for k in keys if len(chunks.get(k, [])) != n_batches]
            if short:
                raise ValueError(f"{short} are missing from batches of '{name}'.")
        return {n: {k: torch.cat(c[k]) for k in keys} for n, c in self.chunks.items()}

    def layout(self) -> dict:
        return {
            "names": list(self.batch_sizes),
            "keys": self.present_keys(),
            "batch_sizes": self.batch_sizes,
            "labels": {name: sorted(lab) for name, lab in self.labels.items()},
        }


def input_moments(x_train: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-feature bounds, mean and standard deviation (ddof 0) over the rows."""
    return {
        "aux_lower": x_train.min(dim=0).values,
        "aux_upper": x_train.max(dim=0).values,
        "aux_mean": x_train.mean(dim=0),
        "aux_std": x_train.std(dim=0, correction=0),
    }


def aux_inputs(meta: dict) -> dict[str, torch.Tensor]:
    """Draw the auxiliary input sets of a dump from its ``meta.pt``.

    Each set has shape (aux_size, *input shape) in the normalised input space:
    ``gaussian`` is N(0, I), ``uniform`` is uniform over the training bounds and
    ``autouad`` is N(aux_mean, diag aux_std^2). The draws reproduce the run's under the
    same PyTorch version only, since the CPU generator stream is not fixed across
    releases.
    """
    # autouad is drawn last, so that gaussian and uniform match older dumps.
    gen = torch.Generator().manual_seed(meta["aux_seed"])
    lower, upper = meta["aux_lower"], meta["aux_upper"]
    shape = (meta["aux_size"], *lower.shape)
    gaussian = torch.randn(shape, generator=gen)
    uniform = lower + (upper - lower) * torch.rand(shape, generator=gen)
    autouad = meta["aux_mean"] + meta["aux_std"] * torch.randn(shape, generator=gen)
    return {"gaussian": gaussian, "uniform": uniform, "autouad": autouad}


def _layout(loaders: dict) -> dict:
    """Layout of every loader, from a pass over the data alone."""
    collector = _Collector([], stored=[])
    # Creating a DataLoader iterator draws from the global RNG.
    with torch.random.fork_rng(devices=[]):
        for name, loader in loaders.items():
            for batch in loader:
                collector.add(name, {}, batch)
    return collector.layout()


def _score_loaders(trainer, pl_module, loaders: dict, keys: list[str]) -> _Collector:
    collector = _Collector(keys)
    for idx, (name, loader) in enumerate(loaders.items()):
        for batch in loader:
            batch = trainer.strategy.batch_to_device(batch, pl_module.device, idx)
            collector.add(name, pl_module.model_step(batch), batch)
    return collector


def score_in_eval_mode(trainer, pl_module, loaders: dict, keys: list[str]):
    """Score extra loaders in eval mode, leaving the module's mode as it was.

    Iterating a DataLoader, or a model that samples, draws from the global RNG; the
    forked streams keep the training trajectory identical to a run without this pass.
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


def _check_names(names: list[str], available: set[str], kind: str) -> None:
    missing = set(names) - available
    if missing:
        raise ValueError(f"No {kind} loader carries {sorted(missing)}.")


class ScoreDumpCallback(Callback):
    """Store the per-event scores and the weights at every validation pass.

    The module docstring gives the layout of the files.

    :param dirpath: Directory of the files; it must not already hold a dump.
    :param run_dir: Hydra output directory of the run, whose ``.hydra`` files are
        copied into ``dirpath``.
    :param keys: Per-event output keys to store; keys a module does not emit are
        skipped.
    :param stored_sets: Validation and test sets whose scores are stored at every
        pass; None stores every set. The auxiliary sets and the weights are always
        stored.
    :param aux_batch_size: Batch size of the passes over the auxiliary sets and the
        training set.
    :param initial_sets: Validation sets whose scores are stored at the sanity pass
        alone, besides the stored sets.
    :param train_pass: Whether every pass stores the per-sample training loss.
    """

    def __init__(
        self,
        dirpath: str,
        run_dir: str,
        keys: list[str],
        aux_seed: int,
        stored_sets: list[str] | None = None,
        aux_size: int = 10000,
        aux_batch_size: int = 1024,
        initial_sets: list[str] | None = None,
        train_pass: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.run_dir = Path(run_dir)
        self.keys = list(keys)
        self.aux_seed = aux_seed
        self.stored_sets = None if stored_sets is None else list(stored_sets)
        self.aux_size = aux_size
        self.aux_batch_size = aux_batch_size
        self.initial_sets = list(initial_sets or [])
        self.train_pass = train_pass
        self._meta_written = False

    def on_fit_start(self, trainer, pl_module):
        """Check the run, copy its Hydra files and build the extra loaders."""
        self._prepare_dir(trainer)
        datamodule = trainer.datamodule
        datamodule.setup("test")
        test = datamodule.test_dataloader()
        self._stored = self._stored_names(list(datamodule.val_dataloader()), list(test))
        self._test_layout = _layout(test)
        train = unpack_batch(datamodule.data_train.tensors)
        mask = drawn_masks(train.mask, self.aux_size, self.aux_seed)
        self._meta = self._run_meta(pl_module, train.x, mask is not None)
        self._loaders = self._build_loaders(datamodule.data_train, test, mask)

    def _prepare_dir(self, trainer) -> None:
        if trainer.world_size > 1:
            raise RuntimeError("ScoreDumpCallback sees one rank's batches only.")
        if trainer.num_sanity_val_steps != float("inf"):  # Lightning stores -1 as inf
            raise RuntimeError("ScoreDumpCallback needs num_sanity_val_steps=-1.")
        if any(self.dirpath.glob("initial.pt")) or any(self.dirpath.glob("epoch_*.pt")):
            raise RuntimeError(f"{self.dirpath} already holds a score dump.")
        self.dirpath.mkdir(parents=True, exist_ok=True)
        for source in (self.run_dir / ".hydra" / name for name in HYDRA_FILES):
            if source.exists():
                shutil.copy(source, self.dirpath / source.name)

    def _stored_names(self, val: list[str], test: list[str]) -> dict[str, list[str]]:
        _check_names(self.stored_sets or [], {*val, *test}, "validation or test")
        _check_names(self.initial_sets, set(val), "validation")
        stored = self.stored_sets
        return {
            "val": [n for n in val if stored is None or n in stored],
            "test": [n for n in test if stored is None or n in stored],
        }

    def _run_meta(self, pl_module, x_train: torch.Tensor, aux_masked: bool) -> dict:
        module_cls = type(pl_module)
        return {
            "target_rate": pl_module.hparams.target_rate,
            "base_rate": getattr(pl_module.hparams, "base_rate", None),
            "target_fpr": pl_module.compute_target_fpr(),
            "module": f"{module_cls.__module__}.{module_cls.__qualname__}",
            "stored": self._stored,
            "initial_sets": self.initial_sets,
            "aux_seed": self.aux_seed,
            "aux_size": self.aux_size,
            "aux_masked": aux_masked,
            **input_moments(x_train),
        }

    def _build_loaders(self, data_train, test: dict, mask: torch.Tensor | None) -> dict:
        batch_size, aux = self.aux_batch_size, aux_inputs(self._meta)
        loaders = {
            "test": {name: test[name] for name in self._stored["test"]},
            "aux": {n: placeholder_loader(x, batch_size, mask) for n, x in aux.items()},
        }
        if self.train_pass:
            # Unshuffled, so that the losses follow the rows of data_train.
            loaders["train"] = {"train": DataLoader(data_train, batch_size=batch_size)}
        return loaders

    def on_validation_epoch_start(self, trainer, pl_module):
        stored = self._stored["val"]
        if trainer.sanity_checking:
            stored = stored + self.initial_sets
        self._val = _Collector(self.keys, stored)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self._val.add(name, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the extra loaders, then write the pass's file."""
        collectors = {"val": self._val}
        for split, loaders in self._loaders.items():
            keys = TRAIN_KEYS if split == "train" else self.keys
            collectors[split] = score_in_eval_mode(trainer, pl_module, loaders, keys)
        epoch = -1 if trainer.sanity_checking else trainer.current_epoch
        record = {split: c.scores() for split, c in collectors.items()}
        state = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
        path = self.dirpath / ("initial.pt" if epoch < 0 else f"epoch_{epoch:04d}.pt")
        _save_atomic({"epoch": epoch, **record, "state_dict": state}, path)
        self._write_meta(collectors)

    def _write_meta(self, collectors: dict) -> None:
        if self._meta_written:
            return
        layouts = {split: c.layout() for split, c in collectors.items()}
        # Only the stored test sets are scored, so the layout is that of the data pass.
        layouts["test"] = {**self._test_layout, "keys": layouts["test"]["keys"]}
        _save_atomic({**self._meta, **layouts}, self.dirpath / "meta.pt")
        self._meta_written = True
