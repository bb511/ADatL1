import os
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
from src.data.utils import check_md5, download, extract, seeded_permutation, to_device
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

# A case is one patient's admission at one center; its counts share these columns.
CASE = ["Center", "Id"]
# Columns fixed per case by the data's construction, checked rather than assumed.
CASE_LABELS = ["Set", "Diagnosis"]
TIME = "Time"
# The member of the Zenodo archive that holds the zipped CSV, and the CSV inside it.
INNER_ZIP = "inst/extdata/sbcdata.csv.zip"
CSV = Path("extdata") / "sbcdata.csv"


def _extract_member(archive: Path, suffix: str, target: Path) -> None:
    """Write the one member of ``archive`` whose name ends in ``suffix`` to ``target``.

    Written to a process-specific name and renamed into place, as ``download`` does.
    """
    with ZipFile(archive) as zf:
        (name,) = (n for n in zf.namelist() if n.endswith(suffix))
        partial = target.with_name(f"{target.name}.{os.getpid()}")
        partial.write_bytes(zf.read(name))
    partial.replace(target)


def _float64(table: pa.Table, name: str) -> np.ndarray:
    """Return a column in float64, a missing value as NaN."""
    return table.column(name).cast(pa.float64()).to_numpy(zero_copy_only=False)


def _inputs(table: pa.Table, inputs: list[str], logged: list[str]) -> np.ndarray:
    """Return the inputs, (nrows, len(inputs)) in float64, the ``logged`` ones as logs.

    The log of a count at or below zero is -inf or NaN, so that the row is incomplete.
    """
    columns = [_float64(table, name) for name in inputs]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = [np.log(c) if n in logged else c for n, c in zip(inputs, columns)]

    return np.stack(columns, axis=1)


def _case_codes(table: pa.Table) -> np.ndarray:
    """Return one int64 code per row, equal for the rows of a case."""
    center = table.column(CASE[0]).combine_chunks().dictionary_encode().indices
    ids = _float64(table, CASE[1]).astype(np.int64)

    return center.to_numpy().astype(np.int64) * (ids.max() + 1) + ids


def _first_complete(case: np.ndarray, time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return, in file order, the row of each case's earliest complete count.

    Complete means every input finite. Ties in time go to the earlier row in the file.
    """
    rows = np.flatnonzero(np.isfinite(x).all(axis=1))
    rows = rows[np.lexsort((rows, time[rows], case[rows]))]
    first = np.ones(rows.size, dtype=bool)
    first[1:] = case[rows[1:]] != case[rows[:-1]]

    return np.sort(rows[first])


def _check_case_labels(table: pa.Table) -> None:
    """Raise if a case carries more than one value of a column of CASE_LABELS."""
    aggregates = [(c, "count_distinct") for c in CASE_LABELS]
    counts = table.group_by(CASE).aggregate(aggregates)
    mixed = {
        c: pc.sum(pc.greater(counts[f"{c}_count_distinct"], 1)).as_py() or 0
        for c in CASE_LABELS
    }
    if any(mixed.values()):
        raise ValueError(f"Cases with more than one value per column: {mixed}.")


def _where(table: pa.Table, where: dict[str, str]) -> np.ndarray:
    """Return the boolean mask of the rows where every named column equals its value."""
    mask = np.ones(table.num_rows, dtype=bool)
    for column, value in where.items():
        equal = pc.equal(table.column(column), value).to_numpy(zero_copy_only=False)
        mask &= equal

    return mask


class SBCDataModule(LightningDataModule):
    """Small blood counts of the AMPEL sbcdata release (Zenodo 10781419) for anomaly
    detection.

    The CSV holds blood counts of hospital cases from Leipzig (sets Training and
    Validation) and Greifswald (Validation), each case labelled Control, Sepsis or SIRS.
    A set is a mapping ``{<column>: <value>, ...}``: the cases whose rows equal every
    value. Every set has one row per case (``Center``, ``Id``), the case's earliest count
    by ``Time`` whose inputs are all finite after the log, ties in file order; cases
    without one are dropped, and their number is logged per set. Each case must carry
    one ``Set`` and one ``Diagnosis``, or the setup raises.

    The model inputs are the ``inputs``, in the given order, those named in
    ``log_inputs`` replaced by their natural log in float64. Each is then centered on its
    median over the training rows, divided by the spread between the two
    ``robust_quantiles`` of those rows and clipped at ``clip_value``, as the Causal
    Chamber's readouts are.

    The draws, all fixed by ``seed`` alone: the reference set's rows, in the order of
    ``seeded_permutation(n, seed)``, give ``train_size`` training rows, then
    ``eval_size`` rows to each of validation ``normal``, validation
    ``reference_normal``, test ``normal`` and test ``reference_normal``, in this order;
    the remaining rows are unused. A null set of ``n`` rows gives ``null_size``
    validation rows, then ``null_size`` test rows, in the order of
    ``seeded_permutation(n, seed + n)``; a signal set of ``n`` rows its first
    ``round(signal_val_fraction * n)`` rows of the same permutation to validation and
    the rest to test. The training batches follow ``train_seed``.

    Validation and test loaders are dictionaries ordered ``normal``,
    ``reference_normal``, the sets of ``signals`` in order (labels 1, 2, ...), then
    those of ``nulls`` (label 0).

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
        url: str,
        md5: str,
        inputs: list[str],
        log_inputs: list[str],
        reference: dict[str, str],
        signals: dict[str, dict],
        nulls: dict[str, dict],
        train_size: int = 300000,
        eval_size: int = 50000,
        null_size: int = 50000,
        signal_val_fraction: float = 0.6,
        batch_size: int = 1024,
        eval_batch_size: int | None = None,
        max_val_batches: int = -1,
        robust_quantiles: tuple[float, float] = (0.05, 0.95),
        clip_value: float = 10.0,
        seed: int = 314159,
        train_seed: int | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if not set(log_inputs) <= set(inputs):
            raise ValueError(f"log_inputs names a non-input: {log_inputs}.")
        self.data_dir = Path(data_dir)
        # inject_object_feature_map looks for the map on datamodule.loader during fit,
        # in the {object: {feature: [flat index]}} layout of the Causal Chamber's.
        feature_map = {"sbc": {name: [i] for i, name in enumerate(inputs)}}
        self.loader = SimpleNamespace(object_feature_map=feature_map)
        self.shuffler = torch.Generator().manual_seed(
            seed if train_seed is None else train_seed
        )

        self.median: torch.Tensor | None = None
        self.spread: torch.Tensor | None = None
        self.data_train: TensorDataset | None = None
        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {}
        # Per set, (cases, cases kept); "unused" the reference rows no split takes.
        self.case_counts: dict[str, tuple[int, int]] = {}
        self.n_unused: int | None = None

    def prepare_data(self) -> None:
        """Download, verify and extract both archive levels unless the CSV is there."""
        if (self.data_dir / CSV).exists():
            return
        # The URL ends in <archive name>/content.
        archive = self.data_dir / Path(self.hparams.url).parent.name
        if not archive.exists():
            download(self.hparams.url, archive)
        check_md5(archive, self.hparams.md5)
        inner = self.data_dir / Path(INNER_ZIP).name
        _extract_member(archive, INNER_ZIP, inner)
        # The inner archive names its member ../extdata/sbcdata.csv; ZipFile drops the
        # "..", which leaves the single directory extdata that extract expects.
        extract(inner, self.data_dir / CSV.parent)

    def setup(self, stage: str | None = None) -> None:
        """Build every split on the first call, for any stage; later calls reuse it."""
        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented.")
        self._set_batch_size()
        if self.data_train is None:
            self._build_splits(self._read_table())
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

    def _read_table(self) -> pa.Table:
        sets = [self.hparams.reference, *self.hparams.signals.values()]
        where = [c for s in [*sets, *self.hparams.nulls.values()] for c in s]
        columns = [*CASE, *CASE_LABELS, TIME, *self.hparams.inputs, *where]
        options = pa_csv.ConvertOptions(include_columns=list(dict.fromkeys(columns)))

        return pa_csv.read_csv(self.data_dir / CSV, convert_options=options)

    def _build_splits(self, table: pa.Table) -> None:
        _check_case_labels(table)
        x = _inputs(table, self.hparams.inputs, self.hparams.log_inputs)
        case = _case_codes(table)
        kept = np.zeros(table.num_rows, dtype=bool)
        kept[_first_complete(case, _float64(table, TIME), x)] = True
        sets = {"reference": self.hparams.reference, **self.hparams.signals}
        rows = {
            name: self._set_rows(name, _where(table, where), case, kept, x)
            for name, where in {**sets, **self.hparams.nulls}.items()
        }
        self._split_reference(rows.pop("reference"))
        self._add_sets(rows)

    def _set_rows(
        self,
        name: str,
        in_set: np.ndarray,
        case: np.ndarray,
        kept: np.ndarray,
        x: np.ndarray,
    ) -> torch.Tensor:
        """Return the kept rows of a set in float32 and record its case counts."""
        self.case_counts[name] = (np.unique(case[in_set]).size, kept[in_set].sum())

        return torch.as_tensor(x[in_set & kept], dtype=torch.float32)

    def _reference_order(self, n: int) -> torch.Tensor:
        n_used = self.hparams.train_size + 4 * self.hparams.eval_size
        if n < n_used:
            raise ValueError(f"{n} reference rows, {n_used} needed.")
        self.n_unused = n - n_used

        return seeded_permutation(n, self.hparams.seed)

    def _split_reference(self, reference: torch.Tensor) -> None:
        n_train, n_eval = self.hparams.train_size, self.hparams.eval_size
        perm = self._reference_order(reference.size(0))
        self._fit_normalization(reference[perm[:n_train]])
        reference = self._robust_scale(reference)
        y = torch.zeros(n_train, dtype=torch.int64)
        self.data_train = TensorDataset(reference[perm[:n_train]], y)
        parts = perm[n_train:].split(n_eval)
        y = torch.zeros(n_eval, dtype=torch.int64)
        for split, (main, ref) in zip(("valid", "test"), (parts[:2], parts[2:4])):
            self._main[split] = SplitTensors(reference[main], y)
            self._aux[split] = {"reference_normal": SplitTensors(reference[ref], y)}

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

    def _add_sets(self, rows: dict[str, torch.Tensor]) -> None:
        signals = enumerate(self.hparams.signals, start=1)
        nulls = [(0, name) for name in self.hparams.nulls]
        for label, name in [*signals, *nulls]:
            x = self._robust_scale(rows[name])
            parts = self._null_parts(x) if label == 0 else self._signal_parts(x)
            for split, part in zip(("valid", "test"), parts):
                y = torch.full((part.size(0),), label, dtype=torch.int64)
                self._aux[split][name] = SplitTensors(part, y)

    def _signal_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        n_valid = round(self.hparams.signal_val_fraction * n)
        perm = seeded_permutation(n, self.hparams.seed + n)

        return x[perm[:n_valid]], x[perm[n_valid:]]

    def _null_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, n_null = x.size(0), self.hparams.null_size
        if n < 2 * n_null:
            raise ValueError(f"A null set of {n} rows, {2 * n_null} needed.")
        perm = seeded_permutation(n, self.hparams.seed + n)

        return x[perm[:n_null]], x[perm[n_null : 2 * n_null]]

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
        for name, (cases, kept) in self.case_counts.items():
            log.info(f"{name}: {kept} of {cases} cases kept, {cases - kept} dropped")
        log.info(f"reference rows unused: {self.n_unused}")
        log.info(Fore.GREEN + "Training data:")
        log.info(f"normal: {tuple(self.data_train.tensors[0].shape)}")
        for split, aux in self._aux.items():
            log.info(Fore.GREEN + f"Data of split {split}:")
            log.info(f"normal: {tuple(self._main[split].x.shape)}")
            for name, part in aux.items():
                log.info(f"{name}: {tuple(part.x.shape)}")
