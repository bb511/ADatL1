import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from colorama import Back, Fore
from pyarrow import csv as pa_csv
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.data.components.dataset import CIFARADDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    y: torch.Tensor


def _select(table: pa.Table, where: dict[str, int | float | str]) -> pa.Table:
    """Keep the rows, in file order, where every named column equals its value.

    Each column is compared in the type inferred for it from the CSV, before any cast to
    float32 (the wind tunnel's ``flag`` is written as 0.0 to 31.0 and read as float64).
    """
    for column, value in where.items():
        table = table.filter(pc.equal(table[column], value))

    return table


def _to_tensor(table: pa.Table, readouts: list[str]) -> torch.Tensor:
    """Return the given columns of a table, (nrows, len(readouts)) in float32."""
    columns = [table.column(name).to_numpy() for name in readouts]

    return torch.as_tensor(np.stack(columns, axis=1), dtype=torch.float32)


def _permutation(n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def _check_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"MD5 of {path} is {digest}, expected {expected}. Delete the file to "
            "download it again."
        )


def _download(url: str, archive: Path) -> None:
    """Download to a process-specific name and rename it into place once complete.

    Runs started together on a cold data directory download concurrently, and the rename
    keeps any of them from reading another's partial file.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    log.info(Back.GREEN + f"Downloading {url}...")
    partial = archive.with_name(f"{archive.name}.{os.getpid()}")
    urlretrieve(url, partial)  # nosec B310, URL from the data config
    partial.replace(archive)


def _extract(archive: Path, target: Path) -> None:
    """Extract into a staging directory and move the dataset into place whole.

    The archive holds a single top-level directory named like ``target``. When a
    concurrent run has moved its copy first, the rename fails and that copy is kept.
    """
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        with ZipFile(archive) as zf:
            zf.extractall(staging)
        try:
            (Path(staging) / target.name).rename(target)
        except OSError:
            if not target.exists():
                raise


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pin CPU tensors before an asynchronous CUDA copy."""
    if device.type != "cuda":
        return t.to(device)
    if t.device.type == "cpu" and not t.is_pinned():
        t = t.pin_memory()

    return t.to(device, non_blocking=True)


class CausalChamberDataModule(LightningDataModule):
    """Causal Chamber data (arXiv:2404.11341) for anomaly detection.

    A set is a mapping ``{file: <csv stem>, where: {<column>: <value>, ...}}``: the rows
    of ``<file>.csv`` in which every column named in the optional ``where`` equals its
    value, in file order. The light tunnel has one CSV per experiment and needs no
    ``where``; the regimes of the wind tunnel are blocks of one CSV, selected by its
    ``flag`` column.

    The rows of the ``reference`` set are split into train, validation and test by
    ``train_fraction`` and ``val_fraction``, and each evaluation part is halved into
    ``normal`` and ``reference_normal``, both labeled 0. Every set in ``signals`` is
    split into validation and test by ``signal_val_fraction`` and labeled by its
    1-based position in that mapping. Those in ``nulls``, which carry no intervention,
    are split in the same way and labeled 0. Each permutation comes from a fresh
    generator, seeded with ``seed`` for the reference split and with ``seed + n`` for
    the halving and the set splits (n being the number of rows permuted); this
    reproduces the row sets of the original chamber study. The splits do not depend on
    the training seed; only the batch order of the training loader does.

    The model inputs are the ``readouts``, in the given order. Each is centered on its
    median over the training rows, divided by the spread between the two
    ``robust_quantiles`` of those rows and clipped at ``clip_value``.

    Validation and test loaders are dictionaries ordered ``normal``,
    ``reference_normal``, the sets of ``signals`` in order, then those of ``nulls``.

    :param max_val_batches: Batch cap on every evaluation loader but ``normal``; -1
        means no cap.
    :param train_fold: When set, keep only every ``n_train_folds``-th row of the
        training part, starting at this position, so that the folds are disjoint halves
        (thirds, ...) of the same training permutation and the evaluation splits are
        unchanged; the normalisation is fitted on the fold alone. ``None`` keeps every
        training row.
    :param robust_quantiles: Lower and upper quantile whose difference sets the scale.
    :param train_seed: Seed of the training batch order; ``seed`` when None.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        url: str,
        md5: str,
        readouts: list[str],
        reference: dict,
        signals: dict[str, dict],
        nulls: dict[str, dict] | None = None,
        batch_size: int = 1024,
        max_val_batches: int = -1,
        train_fraction: float = 0.6,
        val_fraction: float = 0.2,
        signal_val_fraction: float = 0.6,
        robust_quantiles: tuple[float, float] = (0.05, 0.95),
        clip_value: float = 10.0,
        seed: int = 314159,
        train_seed: int | None = None,
        num_workers: int = 0,
        train_fold: int | None = None,
        n_train_folds: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dataset_dir = Path(data_dir) / dataset_name
        # inject_object_feature_map looks for the map on datamodule.loader during fit.
        # One object holds every readout, in the {object: {feature: [flat index]}}
        # layout that ADLightningModule._split_by_type_from_flat reads.
        feature_map = {"chamber": {name: [i] for i, name in enumerate(readouts)}}
        self.loader = SimpleNamespace(object_feature_map=feature_map)
        # Advanced once per epoch by the training loader to reorder its batches. The
        # training seed drives it, so that seeds differ in batch order and not only in
        # the initialization, while the split stays with the data seed.
        self.shuffler = torch.Generator().manual_seed(
            seed if train_seed is None else train_seed
        )

        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.data_train: TensorDataset | None = None
        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {}
        # Raw tables by CSV stem, held only while the splits are built.
        self._tables: dict[str, pa.Table] = {}

    def prepare_data(self) -> None:
        """Download, verify and extract the archive unless the CSVs are in place."""
        if self.dataset_dir.exists():
            return
        archive = self.dataset_dir.parent / f"{self.hparams.dataset_name}.zip"
        if not archive.exists():
            _download(self.hparams.url, archive)
        _check_md5(archive, self.hparams.md5)
        _extract(archive, self.dataset_dir)

    def setup(self, stage: str | None = None) -> None:
        """Build every split on the first call, for any stage; later calls reuse it."""
        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented.")
        self._set_batch_size()
        if self.data_train is None:
            self._build_splits()
            self._tables.clear()
            self._data_summary()

    def train_dataloader(self) -> DataLoader:
        x, y = self.data_train.tensors
        return self._to_loader(SplitTensors(x, y), shuffler=self.shuffler)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return self._make_eval_loaders("valid")

    def test_dataloader(self) -> dict[str, DataLoader]:
        return self._make_eval_loaders("test")

    def transfer_batch_to_device(
        self, batch: tuple[torch.Tensor, ...], device: torch.device, dataloader_idx: int
    ) -> tuple[torch.Tensor, ...]:
        return tuple(_to_device(t, device) for t in batch)

    def _read_set(self, spec: dict) -> torch.Tensor:
        stem = spec["file"]
        if stem not in self._tables:
            self._tables[stem] = self._read_table(stem)
        table = _select(self._tables[stem], spec.get("where") or {})

        return _to_tensor(table, self.hparams.readouts)

    def _read_table(self, stem: str) -> pa.Table:
        """Read the readouts of one CSV and every column that a set filters it on."""
        nulls = self.hparams.nulls or {}
        sets = [self.hparams.reference, *self.hparams.signals.values(), *nulls.values()]
        where = [c for s in sets if s["file"] == stem for c in s.get("where") or {}]
        columns = list(dict.fromkeys([*self.hparams.readouts, *where]))
        options = pa_csv.ConvertOptions(include_columns=columns)
        path = self.dataset_dir / f"{stem}.csv"

        return pa_csv.read_csv(path, convert_options=options)

    def _build_splits(self) -> None:
        reference = self._read_set(self.hparams.reference)
        train_idx, valid_idx, test_idx = self._reference_indices(reference.size(0))
        if self.hparams.train_fold is not None:
            train_idx = train_idx[self.hparams.train_fold :: self.hparams.n_train_folds]
        self._fit_normalization(reference[train_idx])
        reference = self._normalize(reference)
        y_train = torch.zeros(train_idx.numel(), dtype=torch.int64)
        self.data_train = TensorDataset(reference[train_idx], y_train)
        for split, idx in (("valid", valid_idx), ("test", test_idx)):
            self._main[split], ref = self._split_main_and_reference(reference[idx])
            self._aux[split] = {"reference_normal": ref}
        self._add_interventions()

    def _reference_indices(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_train = round(self.hparams.train_fraction * n)
        n_valid = round(self.hparams.val_fraction * n)
        perm = _permutation(n, self.hparams.seed)
        train, rest = perm[:n_train], perm[n_train:]

        return train, rest[:n_valid], rest[n_valid:]

    def _fit_normalization(self, x_train: torch.Tensor) -> None:
        q_low, q_high = self.hparams.robust_quantiles
        qs = torch.quantile(x_train, torch.tensor([q_low, 0.5, q_high]), dim=0)
        self.median = qs[1]
        # Floor for a readout that is constant over the bulk of the training rows.
        self.scale = (qs[2] - qs[0]).clamp_min(1e-6)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        clip = self.hparams.clip_value
        return ((x - self.median) / self.scale).clamp(-clip, clip)

    def _split_main_and_reference(
        self, x: torch.Tensor
    ) -> tuple[SplitTensors, SplitTensors]:
        """Halve an evaluation part into the normal and reference_normal sets.

        An odd row is left out, so that both sets are the same size.
        """
        n = x.size(0)
        half = n // 2
        order = _permutation(n, self.hparams.seed + n)
        main, ref = order[:half], order[half : 2 * half]
        y = torch.zeros(half, dtype=torch.int64)

        return SplitTensors(x[main], y), SplitTensors(x[ref], y)

    def _add_interventions(self) -> None:
        signals = enumerate(self.hparams.signals.items(), start=1)
        nulls = [(0, item) for item in (self.hparams.nulls or {}).items()]
        for label, (name, spec) in [*signals, *nulls]:
            x = self._normalize(self._read_set(spec))
            n = x.size(0)
            n_valid = round(self.hparams.signal_val_fraction * n)
            perm = _permutation(n, self.hparams.seed + n)
            y = torch.full((n,), label, dtype=torch.int64)
            self._aux["valid"][name] = SplitTensors(x[perm[:n_valid]], y[:n_valid])
            self._aux["test"][name] = SplitTensors(x[perm[n_valid:]], y[n_valid:])

    def _make_eval_loaders(self, split: str) -> dict[str, DataLoader]:
        """Return the loaders of one split with normal first, as the callbacks need."""
        loaders = {"normal": self._to_loader(self._main[split])}
        for name, part in self._aux[split].items():
            loaders[name] = self._to_loader(part, self.hparams.max_val_batches)

        return loaders

    def _to_loader(
        self,
        split: SplitTensors,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ) -> DataLoader:
        dataset = CIFARADDataset(
            split.x,
            split.y,
            batch_size=self.batch_size_per_device,
            max_batches=max_batches,
            shuffler=shuffler,
        )
        # The evaluator runs without a datamodule and finds the map on the dataset.
        dataset.object_feature_map = self.loader.object_feature_map

        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def _set_batch_size(self) -> None:
        """Set the batch size per device if multiple devices are available."""
        if self.trainer is None:
            self.batch_size_per_device = self.hparams.batch_size
            return

        world_size = self.trainer.world_size
        if self.hparams.batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.hparams.batch_size}) not divisible by the num of "
                f"devices ({world_size})."
            )
        self.batch_size_per_device = self.hparams.batch_size // world_size

    def _data_summary(self) -> None:
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)
        log.info(Fore.GREEN + "Training data:")
        log.info(f"normal: {tuple(self.data_train.tensors[0].shape)}")
        for split, aux in self._aux.items():
            log.info(Fore.GREEN + f"Data of split {split}:")
            log.info(f"normal: {tuple(self._main[split].x.shape)}")
            for name, part in aux.items():
                log.info(f"{name}: {tuple(part.x.shape)}")
