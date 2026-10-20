"""Per-event scores and weights of a training run, written at every validation pass.

The files are the whole record of a run trained in the dump regime, which keeps no
checkpoint: an offline evaluation reads the stored scores and rescores from the stored
weights every set whose scores the files do not hold. Layout of ``dirpath``, by default
``logs/scores/<experiment_name>/<run_name>``::

    config.yaml, overrides.yaml
        copies of the run's Hydra files, when <run_dir>/.hydra exists
    meta.pt            written once, after the first validation pass
        target_rate, base_rate  the module's hparams; base_rate may be None
        target_fpr              float, the module's compute_target_fpr()
        module                  "<module>.<class>" of the LightningModule
        val, test, aux, train   layout of every loader of the split, stored or not
            names               [name], in loader order
            keys                [key] whose scores the files hold for the split
            batch_sizes         {name: [int]}, one entry per batch
            labels              {name: [int]}, the distinct labels, sorted
        stored                  {"val": [name], "test": [name]}, the sets whose
                                scores the files hold, in loader order
        aux_seed, aux_size      int
        aux_lower, aux_upper    (*s,) per-feature min and max of the training inputs
        aux_mean, aux_std       (*s,) per-feature mean and standard deviation (ddof 0)
    initial.pt         the validation pass of the sanity check, on the untrained weights
    epoch_<nnnn>.pt    one per validation epoch, nnnn = trainer.current_epoch
        epoch                   int, -1 in initial.pt
        val, test               {name: {key: (n,)}} for the names in stored
        aux                     {"gaussian", "uniform", "autouad": {key: (aux_size,)}}
        train                   {"train": {"loss/full": (n_train,)}}
        state_dict              the module's weights and buffers, on CPU

Here (*s,) is the shape of one input: (d,) for vectors, (3, 32, 32) for images. A score
vector concatenates the batches of its loader in loader order, and ``batch_sizes``
splits it back into batches. A key the module does not emit is absent. The auxiliary
sets carry the placeholder label 0.

The validation scores are the ``outputs`` the other callbacks receive in
``on_validation_batch_end``; Lightning scores every validation set, and the sets
outside ``stored_sets`` are dropped. The stored test sets, the auxiliary sets and the
training set are scored by a separate pass through ``model_step`` in eval mode at the
end of each validation pass, the training set unshuffled, in the row order of
``trainer.datamodule.data_train``. The test sets outside ``stored_sets`` are iterated
once at fit start for their layout and never scored. Rescoring a set offline takes the
``state_dict`` of a file and the loaders of the run's datamodule; ``aux_inputs(meta)``
regenerates the auxiliary sets. The ``state_dict`` holds the buffers too, so a module
with a residual state carries the one fitted for that pass by ResidualStateCallback.

The sanity check must run the whole validation set (``num_sanity_val_steps=-1``):
``initial.pt`` stores it, with the weights before the first optimisation step.

The operational score is not stored. It is a per-batch statistic of ``ascore/full``
(``compute_operational_ascore``), recomputable from that score, ``batch_sizes`` and
``target_fpr``.

A VAE builds its KL schedule in ``on_fit_start``, so a VAE loaded from a stored
``state_dict`` must call ``module._setup_kl_annealing(module.kl_warmup_frac, 1)``
before ``model_step``. Its ``loss/full`` carries the KL weight of the global step at
which it was scored, which a rescoring outside the trainer (step 0) does not reproduce
under a warm-up; the anomaly scores do not depend on that weight.

The per-sample training loss feeds the loss entropy of EntropyStop (Huang et al., KDD
2024), and the ``autouad`` set the noise of AutoUAD's NPD (Dai and Fan, ICLR 2025).
"""

import os
import shutil
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, TensorDataset

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
    """Layout of the datasets of one split and per-batch scores of the stored ones.

    A dataset outside ``stored`` adds its batch sizes and labels only; None stores
    every dataset.
    """

    def __init__(self, keys: list[str], stored: list[str] | None = None):
        self.keys = keys
        self.stored = stored
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

    Reads ``aux_seed``, ``aux_size`` and the moments ``aux_lower``, ``aux_upper``,
    ``aux_mean`` and ``aux_std``, all of which ``meta.pt`` holds. Each set has shape
    (aux_size, *input shape) and lies in the normalised input space of ``data_train``:
    ``gaussian`` is N(0, I), ``uniform`` is uniform over the bounds and ``autouad`` is
    N(aux_mean, diag aux_std^2). The draws reproduce those of the run under the same
    PyTorch version, since PyTorch does not guarantee its CPU generator stream across
    releases.
    """
    # The autouad set is drawn last, so that gaussian and uniform are those of the
    # dumps written before it existed.
    gen = torch.Generator().manual_seed(meta["aux_seed"])
    lower, upper = meta["aux_lower"], meta["aux_upper"]
    shape = (meta["aux_size"], *lower.shape)
    gaussian = torch.randn(shape, generator=gen)
    uniform = lower + (upper - lower) * torch.rand(shape, generator=gen)
    noise = torch.randn(shape, generator=gen)
    autouad = meta["aux_mean"] + meta["aux_std"] * noise
    return {"gaussian": gaussian, "uniform": uniform, "autouad": autouad}


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


def _layout(loaders: dict) -> dict:
    """Layout of every loader, from a pass over the data alone."""
    collector = _Collector([], stored=[])
    # Creating a DataLoader iterator draws from the global RNG.
    with torch.random.fork_rng(devices=[]):
        for name, loader in loaders.items():
            for batch in loader:
                collector.add(name, {}, batch)
    return collector.layout()


def _select(names: list[str], stored_sets: list[str] | None) -> list[str]:
    if stored_sets is None:
        return list(names)
    return [name for name in names if name in stored_sets]


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


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _make_dump_dir(dirpath: Path, run_dir: Path) -> None:
    """Create dirpath and copy the run's Hydra files into it, those that exist."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for name in HYDRA_FILES:
        source = run_dir / ".hydra" / name
        if source.exists():
            shutil.copy(source, dirpath / name)


class ScoreDumpCallback(Callback):
    """Store the per-event scores and the weights at every validation pass.

    The module docstring gives the layout of the files.

    :param dirpath: Directory of the files. It must not already hold ``initial.pt`` or
        epoch files, so that two runs are never mixed.
    :param run_dir: Hydra output directory of the run, whose ``.hydra/config.yaml``
        and ``.hydra/overrides.yaml`` are copied into ``dirpath``.
    :param keys: Per-event output keys to store; keys a module does not emit are
        skipped.
    :param aux_seed: Seed of the auxiliary input sets.
    :param stored_sets: Names of the validation and test sets whose scores are stored;
        None stores every set. The auxiliary sets, the training loss and the weights
        are always stored.
    :param aux_size: Number of points in each auxiliary set.
    :param aux_batch_size: Batch size of the passes over the auxiliary sets and the
        training set.
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
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.run_dir = Path(run_dir)
        self.keys = list(keys)
        self.aux_seed = aux_seed
        self.stored_sets = None if stored_sets is None else list(stored_sets)
        self.aux_size = aux_size
        self.aux_batch_size = aux_batch_size
        self._meta_written = False

    def on_fit_start(self, trainer, pl_module):
        """Check the run, copy its configuration and build the extra loaders."""
        self._check_run(trainer)
        datamodule = trainer.datamodule
        test = _test_loaders(datamodule)
        self._stored = self._stored_names(list(datamodule.val_dataloader()), list(test))
        self._test_layout = _layout(test)
        aux_meta = {"aux_seed": self.aux_seed, "aux_size": self.aux_size}
        aux_meta.update(input_moments(datamodule.data_train.tensors[0]))
        self._loaders = self._build_loaders(datamodule.data_train, test, aux_meta)
        self._meta = self._run_meta(pl_module, aux_meta)
        _make_dump_dir(self.dirpath, self.run_dir)

    def _check_run(self, trainer) -> None:
        if trainer.world_size > 1:
            raise RuntimeError("ScoreDumpCallback sees one rank's batches only.")
        # Lightning stores num_sanity_val_steps=-1 as inf.
        if trainer.num_sanity_val_steps != float("inf"):
            raise RuntimeError("ScoreDumpCallback needs num_sanity_val_steps=-1.")
        dumped = [*self.dirpath.glob("initial.pt"), *self.dirpath.glob("epoch_*.pt")]
        if dumped:
            raise RuntimeError(f"{self.dirpath} already holds a score dump.")

    def _stored_names(self, val: list[str], test: list[str]) -> dict[str, list[str]]:
        missing = set(self.stored_sets or []) - {*val, *test}
        if missing:
            raise ValueError(f"No validation or test loader carries {sorted(missing)}.")
        return {
            "val": _select(val, self.stored_sets),
            "test": _select(test, self.stored_sets),
        }

    def _build_loaders(self, data_train, test: dict, aux_meta: dict) -> dict:
        # Unshuffled, so that the losses follow the rows of data_train.
        train = DataLoader(data_train, batch_size=self.aux_batch_size)
        return {
            "test": {name: test[name] for name in self._stored["test"]},
            "aux": _aux_loaders(aux_inputs(aux_meta), self.aux_batch_size),
            "train": {"train": train},
        }

    def _run_meta(self, pl_module, aux_meta: dict) -> dict:
        module_cls = type(pl_module)
        return {
            "target_rate": pl_module.hparams.target_rate,
            "base_rate": getattr(pl_module.hparams, "base_rate", None),
            "target_fpr": pl_module.compute_target_fpr(),
            "module": f"{module_cls.__module__}.{module_cls.__qualname__}",
            "stored": self._stored,
            **aux_meta,
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val = _Collector(self.keys, self._stored["val"])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self._val.add(name, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the test, auxiliary and training sets, then write the pass's file."""
        collectors = {"val": self._val}
        for split, loaders in self._loaders.items():
            keys = TRAIN_KEYS if split == "train" else self.keys
            collectors[split] = score_in_eval_mode(trainer, pl_module, loaders, keys)

        epoch = -1 if trainer.sanity_checking else trainer.current_epoch
        record = {split: c.scores() for split, c in collectors.items()}
        record["state_dict"] = _cpu_state_dict(pl_module)
        name = "initial.pt" if epoch < 0 else f"epoch_{epoch:04d}.pt"
        _save_atomic({"epoch": epoch, **record}, self.dirpath / name)
        self._write_meta(collectors)

    def _write_meta(self, collectors: dict) -> None:
        if self._meta_written:
            return

        layouts = {split: c.layout() for split, c in collectors.items()}
        # Only the stored test sets are scored, so the layout is that of the data pass.
        layouts["test"] = {**self._test_layout, "keys": layouts["test"]["keys"]}
        _save_atomic({**self._meta, **layouts}, self.dirpath / "meta.pt")
        self._meta_written = True
