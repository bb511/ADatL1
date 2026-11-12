import tempfile
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from colorama import Fore
from pyarrow import csv as pa_csv
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.data.CausalChamber_datamodule import SplitTensors
from src.data.components.dataset import ChamberRowsDataset
from src.data.utils import check_md5, download, seeded_permutation, to_device
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


def _read_rows(path: Path, inputs: list[str], logged: list[str]) -> np.ndarray:
    """Return the inputs of one file, (nrows, len(inputs)) in float64.

    The columns named in ``logged`` are replaced by their natural log. The files write a
    missing ratio as ``-nan``, which is read as null and returned as NaN.
    """
    types = {name: pa.float64() for name in inputs}
    convert = pa_csv.ConvertOptions(include_columns=inputs, column_types=types)
    parse = pa_csv.ParseOptions(delimiter=" ")
    table = pa_csv.read_csv(path, parse_options=parse, convert_options=convert)
    x = np.stack([pc.fill_null(table[c], np.nan).to_numpy() for c in inputs], axis=1)
    columns = [inputs.index(name) for name in logged]
    with np.errstate(divide="ignore", invalid="ignore"):
        x[:, columns] = np.log(x[:, columns])

    return x


def _extract_folders(archive: Path, dataset_dir: Path, folders: list[str]) -> None:
    """Extract an archive once and move each of its top-level folders into place.

    As ``src.data.utils.extract``, for an archive of several top-level folders: a
    folder that a concurrent run has moved first is kept.
    """
    with tempfile.TemporaryDirectory(dir=dataset_dir) as staging:
        with ZipFile(archive) as zf:
            zf.extractall(staging)
        for folder in folders:
            try:
                (Path(staging) / folder).rename(dataset_dir / folder)
            except OSError:
                if not (dataset_dir / folder).exists():
                    raise


class DijetDataModule(LightningDataModule):
    """The dijet data of Zenodo record 11109786 (arXiv:2311.12924).

    A set is a mapping ``{files: [<path under the dataset directory>, ...]}``, the rows
    of its text files concatenated in the given order. The model inputs are the columns
    ``inputs``, in the given order, those of ``log_inputs`` replaced by their natural
    log. A row with a non-finite input after the log is dropped from every set, the
    signals included, and counted. Each input is then centered on its median over the
    training rows, divided by the spread between the two ``robust_quantiles`` of those
    rows and clipped at ``clip_value``.

    Every draw is a permutation seeded from ``seed``, so that every training seed sees
    the same rows:

    - ``reference``, the training source: one permutation seeded ``seed`` gives
      ``n_train`` training rows, then ``n_valid`` validation ``normal`` and ``n_valid``
      validation ``reference_normal``;
    - ``test_reference``, held out from the same source: a permutation seeded
      ``seed + n`` (n its finite rows) gives ``n_test`` test ``normal``, then ``n_test``
      test ``reference_normal``;
    - a set of ``signals`` (labels 1, 2, ... in order) or of ``nulls`` (label 0): a
      permutation seeded ``seed + n`` gives its validation rows, then its test rows;
      ``n_valid`` and ``n_test`` when the set names them, otherwise the first
      ``signal_val_fraction`` of the rows and the remainder.

    The rows a set reads and leaves unused are logged with the drop counts.

    Validation and test loaders are dictionaries ordered ``normal``,
    ``reference_normal``, the sets of ``signals`` in order, then those of ``nulls``.

    :param archives: ``{<archive stem>: {md5, folders}}``, the archives of ``url`` and
        the top-level folders each extracts into the dataset directory.
    :param url: The download URL, with ``{name}`` for the archive's file name.
    :param eval_batch_size: Batch size of the validation and test loaders, divided
        across devices as ``batch_size`` is; ``batch_size`` when None.
    :param max_val_batches: Batch cap on every evaluation loader but ``normal``; -1
        means no cap.
    :param robust_quantiles: Lower and upper quantile whose difference sets the scale.
    :param train_seed: Seed of the training batches; ``seed`` when None.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        url: str,
        archives: dict[str, dict],
        inputs: list[str],
        log_inputs: list[str],
        reference: dict,
        test_reference: dict,
        signals: dict[str, dict],
        nulls: dict[str, dict] | None = None,
        batch_size: int = 1024,
        eval_batch_size: int | None = None,
        max_val_batches: int = -1,
        signal_val_fraction: float = 0.6,
        robust_quantiles: tuple[float, float] = (0.05, 0.95),
        clip_value: float = 10.0,
        seed: int = 314159,
        train_seed: int | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dataset_dir = Path(data_dir) / dataset_name
        # inject_object_feature_map looks for the map on datamodule.loader during fit,
        # in the {object: {feature: [flat index]}} layout of the chamber.
        feature_map = {"dijet": {name: [i] for i, name in enumerate(inputs)}}
        self.loader = SimpleNamespace(object_feature_map=feature_map)
        self.shuffler = torch.Generator().manual_seed(
            seed if train_seed is None else train_seed
        )

        self.median: torch.Tensor | None = None
        self.spread: torch.Tensor | None = None
        self.data_train: TensorDataset | None = None
        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {}
        # Per set: rows read, rows dropped as non-finite, finite rows left unused.
        self.counts: dict[str, dict[str, int]] = {}

    def prepare_data(self) -> None:
        """Download, verify and extract every archive whose folders are missing."""
        for stem, spec in self.hparams.archives.items():
            if all((self.dataset_dir / f).exists() for f in spec["folders"]):
                continue
            archive = self.dataset_dir / f"{stem}.zip"
            if not archive.exists():
                download(self.hparams.url.format(name=archive.name), archive)
            check_md5(archive, spec["md5"])
            _extract_folders(archive, self.dataset_dir, spec["folders"])

    def setup(self, stage: str | None = None) -> None:
        """Build every split on the first call, for any stage; later calls reuse it."""
        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented.")
        self._set_batch_size()
        if self.data_train is None:
            self._build_splits()
            self._data_summary()

    def train_dataloader(self) -> DataLoader:
        x, y = self.data_train.tensors
        return self._to_loader(
            SplitTensors(x, y), self.batch_size_per_device, shuffler=self.shuffler
        )

    def val_dataloader(self) -> dict[str, DataLoader]:
        return self._make_eval_loaders("valid")

    def test_dataloader(self) -> dict[str, DataLoader]:
        return self._make_eval_loaders("test")

    def transfer_batch_to_device(
        self, batch: tuple[torch.Tensor, ...], device: torch.device, dataloader_idx: int
    ) -> tuple[torch.Tensor, ...]:
        return tuple(to_device(t, device) for t in batch)

    def _read_set(self, name: str, spec: dict) -> torch.Tensor:
        """Return the finite rows of a set, (nrows, len(inputs)) in float32."""
        inputs, logged = list(self.hparams.inputs), list(self.hparams.log_inputs)
        paths = [self.dataset_dir / f for f in spec["files"]]
        x = np.concatenate([_read_rows(p, inputs, logged) for p in paths])
        finite = np.isfinite(x).all(axis=1)
        self.counts[name] = {"read": len(x), "dropped": int((~finite).sum())}

        return torch.as_tensor(x[finite], dtype=torch.float32)

    def _build_splits(self) -> None:
        spec = self.hparams.reference
        reference = self._read_set("reference", spec)
        sizes = (spec["n_train"], spec["n_valid"], spec["n_valid"])
        train, valid, ref = self._draw("reference", reference, sizes, self.hparams.seed)
        self._fit_normalization(train)
        train = self._labelled(self._robust_scale(train), 0)
        self.data_train = TensorDataset(train.x, train.y)
        self._main["valid"] = self._labelled(self._robust_scale(valid), 0)
        self._aux["valid"] = {
            "reference_normal": self._labelled(self._robust_scale(ref), 0)
        }
        self._add_test_reference()
        self._add_sets()

    def _draw(
        self, name: str, x: torch.Tensor, sizes: tuple[int, ...], seed: int
    ) -> list[torch.Tensor]:
        """Cut one seeded permutation of x into consecutive parts of the given sizes."""
        n = x.size(0)
        if sum(sizes) > n:
            raise ValueError(f"{name} has {n} finite rows, {sum(sizes)} are drawn.")
        self.counts[name]["unused"] = n - sum(sizes)
        bounds = np.cumsum([0, *sizes])
        perm = seeded_permutation(n, seed)

        return [x[perm[lo:hi]] for lo, hi in zip(bounds[:-1], bounds[1:])]

    def _add_test_reference(self) -> None:
        spec = self.hparams.test_reference
        x = self._robust_scale(self._read_set("test_reference", spec))
        sizes = (spec["n_test"], spec["n_test"])
        main, ref = self._draw("test_reference", x, sizes, self.hparams.seed + len(x))
        self._main["test"] = self._labelled(main, 0)
        self._aux["test"] = {"reference_normal": self._labelled(ref, 0)}

    def _add_sets(self) -> None:
        signals = enumerate(self.hparams.signals.items(), start=1)
        nulls = [(0, item) for item in (self.hparams.nulls or {}).items()]
        for label, (name, spec) in [*signals, *nulls]:
            x = self._robust_scale(self._read_set(name, spec))
            n_valid = spec.get(
                "n_valid", round(self.hparams.signal_val_fraction * len(x))
            )
            n_test = spec.get("n_test", len(x) - n_valid)
            parts = self._draw(name, x, (n_valid, n_test), self.hparams.seed + len(x))
            for split, part in zip(("valid", "test"), parts):
                self._aux[split][name] = self._labelled(part, label)

    def _labelled(self, x: torch.Tensor, label: int) -> SplitTensors:
        return SplitTensors(x, torch.full((x.size(0),), label, dtype=torch.int64))

    def _fit_normalization(self, x_train: torch.Tensor) -> None:
        """Set each input's training median and its inter-quantile spread."""
        low, high = self.hparams.robust_quantiles
        levels = torch.tensor([low, 0.5, high])
        lower, self.median, upper = torch.quantile(x_train, levels, dim=0)
        # An input constant over the bulk of the training rows has zero spread.
        self.spread = (upper - lower).clamp_min(1e-6)

    def _robust_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Return x less the median, in units of the spread, within ±clip_value."""
        bound = self.hparams.clip_value
        return ((x - self.median) / self.spread).clamp(-bound, bound)

    def _make_eval_loaders(self, split: str) -> dict[str, DataLoader]:
        """Return the loaders of one split with normal first, as the callbacks need."""
        batch_size = self.eval_batch_size_per_device
        max_batches = self.hparams.max_val_batches
        loaders = {"normal": self._to_loader(self._main[split], batch_size)}
        for name, part in self._aux[split].items():
            loaders[name] = self._to_loader(part, batch_size, max_batches)

        return loaders

    def _to_loader(
        self,
        split: SplitTensors,
        batch_size: int,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ) -> DataLoader:
        dataset = ChamberRowsDataset(
            split.x,
            split.y,
            batch_size=batch_size,
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
        """Set the training and evaluation batch sizes per device."""
        eval_batch_size = self.hparams.eval_batch_size
        if eval_batch_size is None:
            eval_batch_size = self.hparams.batch_size
        self.batch_size_per_device = self._per_device(self.hparams.batch_size)
        self.eval_batch_size_per_device = self._per_device(eval_batch_size)

    def _per_device(self, batch_size: int) -> int:
        if self.trainer is None:
            return batch_size

        world_size = self.trainer.world_size
        if batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size ({batch_size}) not divisible by the num of "
                f"devices ({world_size})."
            )
        return batch_size // world_size

    def _data_summary(self) -> None:
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)
        log.info(f"Training data: normal: {tuple(self.data_train.tensors[0].shape)}")
        for split, aux in self._aux.items():
            log.info(Fore.GREEN + f"Data of split {split}:")
            log.info(f"normal: {tuple(self._main[split].x.shape)}")
            for name, part in aux.items():
                log.info(f"{name}: {tuple(part.x.shape)}")
        for name, counts in self.counts.items():
            log.info(f"Rows of {name}: {counts}")
