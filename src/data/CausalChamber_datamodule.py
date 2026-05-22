import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
import torch
from colorama import Fore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.dataset import L1ADDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


META_COLUMNS = ("timestamp", "config", "counter", "flag", "intervention")
READOUT_FEATURES = (
    "current",
    "angle_1",
    "angle_2",
    "ir_1",
    "vis_1",
    "ir_2",
    "vis_2",
    "ir_3",
    "vis_3",
    "v_board",
    "v_reg",
)


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    mask: torch.Tensor
    l1bit: torch.Tensor
    y: torch.Tensor


class _CausalChamberLoader:
    def __init__(self, feature_names: list[str]):
        self.object_feature_map = {
            "chamber": {feature: [idx] for idx, feature in enumerate(feature_names)}
        }


class CausalChamberDataModule(LightningDataModule):
    """Causal Chamber light-tunnel intervention benchmark.

    The module casts ``lt_interventions_standard_v1`` as a vector anomaly-detection
    task. Training uses only ``uniform_reference``. The two signal-agnostic normal
    domains used by CAP/W1 are disjoint splits of ``uniform_reference`` named
    ``normal`` and ``reference_normal``. All other ``uniform_*`` CSV files are
    exposed as held-out intervention/anomaly datasets.
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
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / dataset_name
        self.archive_path = self.data_dir / f"{dataset_name}.zip"

        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)

        self.feature_names: list[str] | None = None
        self.loader: _CausalChamberLoader | None = None
        self.center: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

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

        raw_reference = self._load_features("uniform_reference")
        train_raw, valid_raw, test_raw = self._split_reference(raw_reference)
        self._fit_normalizer(train_raw)

        if stage in (None, "fit"):
            self._main["train"] = self._make_split(self._normalize(train_raw), label=0)
            self._setup_eval_split("valid", valid_raw)

        if stage in (None, "validate"):
            self._setup_eval_split("valid", valid_raw)

        if stage in (None, "test"):
            self._setup_eval_split("test", test_raw)

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def train_dataloader(self):
        return self._to_loader(self._main["train"], shuffler=self.shuffler)

    def val_dataloader(self):
        return self._make_eval_loaders("valid", "valid")

    def test_dataloader(self):
        return self._make_eval_loaders("test", "test")

    def teardown(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self._main.pop("train", None)
            self._main.pop("valid", None)
            self._aux.get("valid", {}).clear()

        if stage in ("test", None):
            self._main.pop("test", None)
            self._aux.get("test", {}).clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if device.type != "cuda":
            return tuple(t.to(device) for t in batch)

        out = []
        for tensor in batch:
            if tensor.device.type == "cpu" and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            out.append(tensor.to(device, non_blocking=True))
        return tuple(out)

    def _setup_eval_split(self, split_name: str, normal_raw: torch.Tensor) -> None:
        main, reference = self._split_main_and_reference(self._normalize(normal_raw))
        self._main[split_name] = main

        aux: dict[str, SplitTensors] = {"reference_normal": reference}
        for label, name in enumerate(self._signal_experiments(), start=1):
            signal_raw = self._load_features(name)
            signal_part = self._split_signal(signal_raw, split_name)
            aux[name] = self._make_split(self._normalize(signal_part), label=label)

        self._aux[split_name] = aux

    def _split_reference(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_total = data.size(0)
        n_train = int(round(self.hparams.train_fraction * n_total))
        n_valid = int(round(self.hparams.val_fraction * n_total))
        n_test = n_total - n_train - n_valid
        if min(n_train, n_valid, n_test) <= 1:
            raise RuntimeError("Reference split is too small. Adjust train_fraction/val_fraction.")

        gen = torch.Generator().manual_seed(self.hparams.seed)
        perm = torch.randperm(n_total, generator=gen)
        train_idx = perm[:n_train]
        valid_idx = perm[n_train : n_train + n_valid]
        test_idx = perm[n_train + n_valid :]
        return data[train_idx], data[valid_idx], data[test_idx]

    def _split_main_and_reference(self, data: torch.Tensor) -> tuple[SplitTensors, SplitTensors]:
        n_total = data.size(0)
        n_ref = max(1, int(round(self.hparams.reference_fraction * n_total)))
        n_ref = min(n_ref, n_total - 1)

        gen = torch.Generator().manual_seed(self.hparams.seed + n_total)
        perm = torch.randperm(n_total, generator=gen)
        ref_idx = perm[:n_ref]
        main_idx = perm[n_ref:]
        return self._make_split(data[main_idx], 0), self._make_split(data[ref_idx], -1)

    def _split_signal(self, data: torch.Tensor, split_name: str) -> torch.Tensor:
        n_valid = int(round(self.hparams.signal_val_fraction * data.size(0)))
        n_valid = min(max(1, n_valid), data.size(0) - 1)
        gen = torch.Generator().manual_seed(self.hparams.seed + data.size(0))
        perm = torch.randperm(data.size(0), generator=gen)
        if split_name == "valid":
            return data[perm[:n_valid]]
        if split_name == "test":
            return data[perm[n_valid:]]
        raise ValueError(f"Unsupported split '{split_name}'.")

    def _load_features(self, experiment: str) -> torch.Tensor:
        path = self.dataset_dir / f"{experiment}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Causal Chamber experiment not found: {path}")

        df = pd.read_csv(path)
        feature_names = self._resolve_feature_names(df)
        if self.feature_names is None:
            self.feature_names = feature_names
            self.loader = _CausalChamberLoader(feature_names)
        elif feature_names != self.feature_names:
            raise ValueError(
                f"Feature columns for {experiment} differ from the reference dataset."
            )

        x = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        if x.isna().any().any():
            bad = list(x.columns[x.isna().any()])
            raise ValueError(f"NaN/non-numeric values found in columns: {bad}")

        return torch.as_tensor(x.to_numpy(), dtype=torch.float32)

    def _resolve_feature_names(self, df: pd.DataFrame) -> list[str]:
        feature_set = self.hparams.feature_set
        if feature_set == "readouts":
            names = list(READOUT_FEATURES)
        elif feature_set == "all_numeric_no_meta":
            names = [
                c
                for c in df.columns
                if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
            ]
        elif feature_set == "custom":
            names = list(self.hparams.feature_columns or [])
        else:
            raise ValueError("feature_set must be one of: readouts, all_numeric_no_meta, custom.")

        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing requested Causal Chamber columns: {missing}")
        if not names:
            raise ValueError("No Causal Chamber feature columns were selected.")
        return names

    def _fit_normalizer(self, train_raw: torch.Tensor) -> None:
        if not self.hparams.normalize:
            self.center = None
            self.scale = None
            return

        q_low, q_high = [float(q) for q in self.hparams.robust_quantiles]
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("robust_quantiles must satisfy 0 <= low < high <= 1.")

        qs = torch.quantile(train_raw, torch.tensor([q_low, 0.5, q_high]), dim=0)
        self.center = qs[1]
        self.scale = (qs[2] - qs[0]).clamp_min(1.0e-6)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hparams.normalize:
            return x.float().contiguous()
        if self.center is None or self.scale is None:
            raise RuntimeError("Normalizer requested but not fitted.")
        out = (x - self.center) / self.scale
        if self.hparams.clip_value is not None:
            out = out.clamp(-float(self.hparams.clip_value), float(self.hparams.clip_value))
        return out.float().contiguous()

    def _make_split(self, x: torch.Tensor, label: int) -> SplitTensors:
        n = x.size(0)
        return SplitTensors(
            x=x.contiguous(),
            mask=torch.ones_like(x, dtype=torch.bool),
            l1bit=torch.zeros(n, dtype=torch.bool),
            y=torch.full((n,), label, dtype=torch.int64),
        )

    def _make_eval_loaders(self, main_key: str, aux_key: str) -> dict[str, DataLoader]:
        loaders = {"normal": self._to_loader(self._main[main_key])}
        for name, split in self._aux.get(aux_key, {}).items():
            loaders[name] = self._to_loader(split, max_b=self.hparams.max_val_batches)
        return loaders

    def _to_loader(
        self,
        split: SplitTensors,
        max_b: int | None = None,
        shuffler: torch.Generator | None = None,
    ) -> DataLoader:
        if max_b is not None and int(max_b) < 0:
            max_b = None

        ds = L1ADDataset(
            split.x,
            split.mask,
            split.l1bit,
            split.y,
            batch_size=self.batch_size_per_device,
            max_batches=max_b,
            shuffler=shuffler,
        )
        if self.loader is not None:
            ds.object_feature_map = self.loader.object_feature_map
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
        )

    def _signal_experiments(self) -> list[str]:
        if self.hparams.signal_experiments:
            return list(self.hparams.signal_experiments)
        return sorted(
            p.stem for p in self.dataset_dir.glob("*.csv") if p.stem != "uniform_reference"
        )

    @property
    def _lock_path(self) -> Path:
        return self.data_dir / f".{self.hparams.dataset_name}.lock"

    def _dataset_ready(self) -> bool:
        if not self.dataset_dir.exists():
            return False

        required = ["uniform_reference"]
        if self.hparams.signal_experiments:
            required.extend(self.hparams.signal_experiments)

        if required:
            return all((self.dataset_dir / f"{name}.csv").exists() for name in required)

        return any(self.dataset_dir.glob("*.csv"))

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

        def show_split(title: str, key: str, aux_key: str | None = None) -> None:
            log.info(Fore.GREEN + title)
            if key in self._main:
                log.info(f"normal: {tuple(self._main[key].x.shape)}")
            if aux_key:
                for name, split in self._aux.get(aux_key, {}).items():
                    log.info(f"{name}: {tuple(split.x.shape)}")

        if stage in (None, "fit"):
            show_split("Training data:", "train")
            show_split("Validation data:", "valid", "valid")
        elif stage == "validate":
            show_split("Validation data:", "valid", "valid")
        elif stage == "test":
            show_split("Test data:", "test", "test")

    def _validate_config(self) -> None:
        if not (0.0 < self.hparams.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1).")
        if not (0.0 < self.hparams.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1).")
        if self.hparams.train_fraction + self.hparams.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1.")
        if not (0.0 < self.hparams.reference_fraction < 1.0):
            raise ValueError("reference_fraction must be in (0, 1).")
        if not (0.0 < self.hparams.signal_val_fraction < 1.0):
            raise ValueError("signal_val_fraction must be in (0, 1).")

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
