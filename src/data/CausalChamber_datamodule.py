import gc
import os
import time
from contextlib import contextmanager
from hashlib import md5
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from colorama import Fore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.causal_chamber import (
    META_COLUMNS,
    READOUT_FEATURES,
    CausalChamberDataBuilder,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class _CausalChamberLoader:
    def __init__(self, object_feature_map: dict[str, dict[str, list[int]]]):
        self.object_feature_map = object_feature_map


class CausalChamberDataModule(LightningDataModule):
    """Causal Chamber light-tunnel intervention benchmark.

    The datamodule only handles download/extraction and Lightning dataloader
    orchestration. Feature selection, metadata retention, paired validation view
    construction, and the experiment contract live in
    ``src.data.components.causal_chamber``. All samples come from the public
    Causal Chamber CSV files.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "lt_interventions_standard_v1",
        url: str = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_interventions_standard_v1.zip",
        md5sum: str = "476664d024f88e8b7640998bb5e9ee33",
        feature_set: str = "readouts",
        feature_columns: list[str] | None = None,
        signal_experiments: list[str] | None = None,
        pairing_columns: list[str] | None = None,
        pairing_strategy: str = "nearest",
        batch_size: int = 512,
        max_val_batches: int | None = -1,
        train_fraction: float = 0.6,
        val_fraction: float = 0.2,
        reference_fraction: float = 0.5,
        signal_val_fraction: float = 0.6,
        normalize: bool = True,
        robust_quantiles: list[float] | tuple[float, float] = (0.05, 0.95),
        clip_value: float | None = 10.0,
        seed: int = 123,
        train_seed: int | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / dataset_name
        self.archive_path = self.data_dir / f"{dataset_name}.zip"

        self.builder: CausalChamberDataBuilder | None = None
        self.loader: _CausalChamberLoader | None = None
        self.feature_names: list[str] | None = None
        self.contract: dict | None = None
        # Keep the split seed and optimization-order seed independent. This is
        # essential for fine-tuning from one common pretrained initialization.
        self.train_seed = int(seed if train_seed is None else train_seed)
        self.shuffler = torch.Generator().manual_seed(self.train_seed)

        self._validate_config()

    def prepare_data(self) -> None:
        """Download and extract the public Causal Chamber archive if needed."""
        if self._dataset_ready() and not self._lock_path.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)

        with self._prepare_lock():
            if self._dataset_ready():
                return

            if not self.archive_path.exists() or not self._md5_matches(
                self.archive_path, self.hparams.md5sum
            ):
                if self.archive_path.exists():
                    log.warning(f"Removing invalid partial archive: {self.archive_path}")
                    self.archive_path.unlink()

                log.info(f"Downloading Causal Chamber dataset from {self.hparams.url}")
                tmp_archive = self.archive_path.with_name(
                    f"{self.archive_path.name}.{os.getpid()}.tmp"
                )
                urlretrieve(self.hparams.url, tmp_archive)
                self._check_md5(tmp_archive, self.hparams.md5sum)
                tmp_archive.replace(self.archive_path)

            self._check_md5(self.archive_path, self.hparams.md5sum)
            log.info(f"Extracting {self.archive_path}")
            with ZipFile(self.archive_path) as archive:
                archive.extractall(self.data_dir)

            if not self._dataset_ready():
                raise RuntimeError(
                    f"Expected extracted dataset at {self.dataset_dir}, but it was not found."
                )

    def setup(self, stage: str | None = None) -> None:
        self._set_batch_size()
        self.prepare_data()

        self.builder = CausalChamberDataBuilder(
            dataset_dir=self.dataset_dir,
            dataset_name=self.hparams.dataset_name,
            feature_set=self.hparams.feature_set,
            feature_columns=self.hparams.feature_columns,
            signal_experiments=self.hparams.signal_experiments,
            pairing_columns=self.hparams.pairing_columns,
            pairing_strategy=self.hparams.pairing_strategy,
            train_fraction=self.hparams.train_fraction,
            val_fraction=self.hparams.val_fraction,
            reference_fraction=self.hparams.reference_fraction,
            signal_val_fraction=self.hparams.signal_val_fraction,
            normalize=self.hparams.normalize,
            robust_quantiles=self.hparams.robust_quantiles,
            clip_value=self.hparams.clip_value,
            seed=self.hparams.seed,
        )
        self.builder.setup(
            stage=stage,
            batch_size=self.batch_size_per_device,
            max_val_batches=self.hparams.max_val_batches,
            train_shuffler=self.shuffler,
        )

        self.feature_names = self.builder.feature_names
        self.contract = None if self.builder.contract is None else self.builder.contract.to_dict()
        if self.builder.object_feature_map is not None:
            self.loader = _CausalChamberLoader(self.builder.object_feature_map)

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def train_dataloader(self):
        self._require_builder()
        return self._to_loader(self.builder.main["train"])

    def val_dataloader(self):
        self._require_builder()
        return self._make_eval_loaders("valid")

    def test_dataloader(self):
        self._require_builder()
        return self._make_eval_loaders("test")

    def teardown(self, stage: str | None = None) -> None:
        if self.builder is not None:
            if stage in ("fit", None):
                self.builder.main.pop("train", None)
                self.builder.main.pop("valid", None)
                self.builder.aux.get("valid", {}).clear()

            if stage in ("test", None):
                self.builder.main.pop("test", None)
                self.builder.aux.get("test", {}).clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            out = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    if (
                        device.type == "cuda"
                        and value.device.type == "cpu"
                        and not value.is_pinned()
                    ):
                        value = value.pin_memory()
                    value = value.to(device, non_blocking=device.type == "cuda")
                out[key] = value
            return out

        if device.type != "cuda":
            return tuple(t.to(device) for t in batch)

        out = []
        for tensor in batch:
            if tensor.device.type == "cpu" and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            out.append(tensor.to(device, non_blocking=True))
        return tuple(out)

    def _make_eval_loaders(self, split_name: str) -> dict[str, DataLoader]:
        loaders = {"normal": self._to_loader(self.builder.main[split_name])}
        for name, dataset in self.builder.aux.get(split_name, {}).items():
            loaders[name] = self._to_loader(dataset)
        return loaders

    def _to_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
        )

    def _require_builder(self) -> None:
        if self.builder is None:
            raise RuntimeError("CausalChamberDataModule.setup() must be called first.")

    @property
    def _lock_path(self) -> Path:
        return self.data_dir / f".{self.hparams.dataset_name}.lock"

    def _dataset_ready(self) -> bool:
        if not self.dataset_dir.exists():
            return False

        required = ["uniform_reference"]
        if self.hparams.signal_experiments:
            required.extend(self.hparams.signal_experiments)

        return all((self.dataset_dir / f"{name}.csv").exists() for name in required)

    @contextmanager
    def _prepare_lock(self):
        start = time.monotonic()
        lock_timeout = 600.0

        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                break
            except FileExistsError:
                if time.monotonic() - start > lock_timeout:
                    raise TimeoutError(
                        f"Timed out waiting for Causal Chamber data lock: {self._lock_path}"
                    )
                time.sleep(1.0)

        try:
            yield
        finally:
            self._lock_path.unlink(missing_ok=True)

    def _set_batch_size(self) -> None:
        if self.trainer is None:
            self.batch_size_per_device = self.hparams.batch_size
            return

        world_size = self.trainer.world_size
        if self.hparams.batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.hparams.batch_size}) not divisible by "
                f"the number of devices ({world_size})."
            )
        self.batch_size_per_device = self.hparams.batch_size // world_size

    def _data_summary(self, stage: str | None) -> None:
        log.info(Fore.MAGENTA + "-" * 5 + " Causal Chamber Data Summary " + "-" * 5)
        if self.feature_names is not None:
            log.info(f"Features ({len(self.feature_names)}): {self.feature_names}")
        if self.contract is not None:
            log.info(f"Pairing: {self.contract['pairing']}")

        def show_split(title: str, split_name: str) -> None:
            log.info(Fore.GREEN + title)
            if self.builder is None:
                return
            if split_name in self.builder.main:
                log.info(f"normal: {tuple(self.builder.main[split_name].x.shape)}")
            for name, dataset in self.builder.aux.get(split_name, {}).items():
                log.info(f"{name}: {tuple(dataset.x.shape)}")

        if stage in (None, "fit"):
            if self.builder and "train" in self.builder.main:
                log.info(Fore.GREEN + "Training data:")
                log.info(f"train: {tuple(self.builder.main['train'].x.shape)}")
            show_split("Validation data:", "valid")
        elif stage == "validate":
            show_split("Validation data:", "valid")
        elif stage == "test":
            show_split("Test data:", "test")

    def _validate_config(self) -> None:
        if not (0.0 < self.hparams.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1).")
        if not (0.0 < self.hparams.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1).")
        if self.hparams.train_fraction + self.hparams.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1.")
        if not (0.0 < self.hparams.reference_fraction <= 1.0):
            raise ValueError("reference_fraction must be in (0, 1].")
        if not (0.0 < self.hparams.signal_val_fraction < 1.0):
            raise ValueError("signal_val_fraction must be in (0, 1).")
        if self.hparams.pairing_strategy not in {"nearest", "metadata_nearest", "random"}:
            raise ValueError("pairing_strategy must be one of: nearest, metadata_nearest, random.")

    @staticmethod
    def _check_md5(path: Path, expected: str | None) -> None:
        if not expected:
            return
        digest = md5(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, got {digest}.")

    @staticmethod
    def _md5_matches(path: Path, expected: str | None) -> bool:
        if not expected:
            return True
        if not path.exists():
            return False
        return md5(path.read_bytes()).hexdigest() == expected
