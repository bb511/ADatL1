from dataclasses import dataclass
from pathlib import Path
import gc

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import CIFAR10

from src.utils import pylogger
from colorama import Fore, Back
from src.data.components.dataset import CIFARADDataset

log = pylogger.RankedLogger(__name__)


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    y: torch.Tensor


class CIFAR10DataModule(LightningDataModule):
    """Lightning DataModule for CIFAR-10 anomaly detection experiments.

    This module mirrors the structure of the physics datamodules used in the project:
    the main normal split is stored separately, while all auxiliary evaluation splits
    are exposed through the auxiliary dataset dictionaries. In particular,
    ``reference_normal`` is treated as an auxiliary dataset, just like anomaly or
    background datasets.

    Labeling convention:
      - normal classes are assigned label 0,
      - signal/anomalous classes are assigned labels > 0,
      - optional background classes are assigned labels < 0.

    Validation and test dataloaders are returned as dictionaries where:
      - ``"normal"`` is always the first/main dataset,
      - ``"reference_normal"`` is part of the auxiliary datasets,
      - the remaining auxiliary datasets are keyed by their original CIFAR class id
        converted to string, e.g. ``"1"``, ``"7"``, etc.

    Standard image normalization is computed from the training split of the normal
    classes only, then applied to all splits.

    :param data_dir: Path where CIFAR-10 will be downloaded/stored.
    :param normal_classes: List of CIFAR-10 class indices to be treated as normal.
    :param signal_classes: List of class indices treated as anomalies.
    :param background_classes: Optional list of class indices treated as auxiliary
        background datasets.
    :param batch_size: Global batch size. It is divided across devices if using DDP.
    :param max_val_batches: Maximum number of batches for auxiliary validation/test
        loaders. ``-1`` means full dataset.
    :param val_fraction: Fraction of the normal training split reserved for validation.
    :param reference_fraction: Fraction of normal validation/test data reserved for the
        auxiliary ``reference_normal`` split. The remainder becomes the main
        ``normal`` split.
    :param seed: Random seed used for splitting and shuffling.
    :param num_workers: Number of DataLoader workers.
    :param normalize: Whether to apply standard normalization.
    :param stats_file: Filename used to cache normalization statistics.
    """

    def __init__(
        self,
        data_dir: str,
        normal_classes: list[int],
        signal_classes: list[int],
        background_classes: list[int] | None = None,
        batch_size: int = 512,
        max_val_batches: int = -1,
        val_fraction: float = 0.1,
        reference_fraction: float = 0.5,
        seed: int = 42,
        num_workers: int = 0,
        normalize: bool = True,
        stats_file: str = "cifar10_normal_stats.pt",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)
        self.max_val_batches = max_val_batches

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        if not (0.0 < self.hparams.val_fraction < 1.0):
            raise ValueError("val_fraction must be strictly between 0 and 1.")
        if not (0.0 < self.hparams.reference_fraction < 1.0):
            raise ValueError("reference_fraction must be strictly between 0 and 1.")

    def prepare_data(self) -> None:
        """Download CIFAR-10 if needed."""
        log.info(Back.GREEN + "Preparing CIFAR10 data...")
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        """Load CIFAR-10 into memory and prepare train/val/test splits."""
        self._set_batch_size()

        if self.hparams.normalize and (self.mean is None or self.std is None):
            self._load_or_compute_stats()

        if stage in (None, "fit", "validate"):
            train_ds = CIFAR10(
                root=self.hparams.data_dir,
                train=True,
                download=False,
            )
            self._setup_fit_validate(train_ds)

        if stage in (None, "test"):
            test_ds = CIFAR10(
                root=self.hparams.data_dir,
                train=False,
                download=False,
            )
            self._setup_test(test_ds)

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def train_dataloader(self):
        """Create and return the training dataloader."""
        split = self._main["train"]
        dataset = CIFARADDataset(
            split.x,
            split.y,
            batch_size=self.batch_size_per_device,
            shuffler=self.shuffler,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """Return validation dataloaders with normal first and aux datasets after."""
        return self._make_eval_loaders(
            main_key="valid",
            aux_key="valid",
            main_name="normal",
        )

    def test_dataloader(self):
        """Return test dataloaders with normal first and aux datasets after."""
        return self._make_eval_loaders(
            main_key="test",
            aux_key="test",
            main_name="normal",
        )

    def teardown(self, stage: str | None = None) -> None:
        """Release cached tensors."""
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
        """Transfer 2-tuple CIFAR batches to device."""
        x, y = batch

        if device.type != "cuda":
            return x.to(device), y.to(device)

        if x.device.type == "cpu" and not x.is_pinned():
            x = x.pin_memory()
        if y.device.type == "cpu" and not y.is_pinned():
            y = y.pin_memory()

        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def _load_or_compute_stats(self) -> None:
        """Load cached normalization stats or compute them from normal train data."""
        stats_path = Path(self.hparams.data_dir) / self.hparams.stats_file

        if stats_path.exists():
            stats = torch.load(stats_path, map_location="cpu")
            self.mean = stats["mean"].float()
            self.std = stats["std"].float()
            log.info(f"Loaded CIFAR10 normalization stats from {stats_path}")
            return

        train_ds = CIFAR10(
            root=self.hparams.data_dir,
            train=True,
            download=False,
        )
        self._compute_normalization(train_ds)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, stats_path)
        log.info(f"Saved CIFAR10 normalization stats to {stats_path}")

    def _compute_normalization(self, dataset: CIFAR10) -> None:
        """Compute mean/std using only the normal-class training samples."""
        normal_indices = self._get_class_indices(
            dataset.targets, self.hparams.normal_classes
        )
        xs = []

        for idx in normal_indices:
            x = (
                torch.as_tensor(dataset.data[idx], dtype=torch.float32)
                .permute(2, 0, 1)
                / 255.0
            )
            xs.append(x)

        x = torch.stack(xs, dim=0)
        self.mean = x.mean(dim=(0, 2, 3))
        self.std = x.std(dim=(0, 2, 3)).clamp_min(1e-8)

        log.info(f"CIFAR10 normalization mean: {self.mean.tolist()}")
        log.info(f"CIFAR10 normalization std: {self.std.tolist()}")

    def _setup_fit_validate(self, train_ds: CIFAR10) -> None:
        """Prepare train and validation splits from the CIFAR train set."""
        normal_indices = self._get_class_indices(
            train_ds.targets, self.hparams.normal_classes
        )
        normal_data, normal_labels = self._extract_subset(train_ds, normal_indices, label=0)

        n_total = normal_data.size(0)
        n_valid = max(2, int(round(self.hparams.val_fraction * n_total)))
        n_train = n_total - n_valid
        if n_train <= 0:
            raise RuntimeError("Validation split too large; no training data remains.")

        full_normal = TensorDataset(normal_data, normal_labels)
        train_subset, valid_subset = random_split(
            full_normal,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        if "train" not in self._main:
            x_train, y_train = self._tensor_dataset_to_tensors(train_subset)
            self._main["train"] = SplitTensors(
                x=x_train.contiguous(),
                y=y_train.contiguous(),
            )

        if "valid" not in self._main or not self._aux["valid"]:
            x_valid, y_valid = self._tensor_dataset_to_tensors(valid_subset)
            main_valid, ref_valid = self._split_main_and_reference(x_valid, y_valid)
            self._main["valid"] = main_valid

            aux_valid = self._build_aux_split(train_ds)
            aux_valid["reference_normal"] = ref_valid
            self._aux["valid"] = aux_valid

    def _setup_test(self, test_ds: CIFAR10) -> None:
        """Prepare test splits from the CIFAR test set."""
        normal_indices = self._get_class_indices(
            test_ds.targets, self.hparams.normal_classes
        )
        x_test, y_test = self._extract_subset(test_ds, normal_indices, label=0)

        if "test" not in self._main or not self._aux["test"]:
            main_test, ref_test = self._split_main_and_reference(x_test, y_test)
            self._main["test"] = SplitTensors(
                x=main_test.x.contiguous(),
                y=main_test.y.contiguous(),
            )

            aux_test = self._build_aux_split(test_ds)
            aux_test["reference_normal"] = SplitTensors(
                x=ref_test.x.contiguous(),
                y=ref_test.y.contiguous(),
            )
            self._aux["test"] = aux_test

    def _split_main_and_reference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[SplitTensors, SplitTensors]:
        """Split a normal dataset into main normal and reference normal subsets."""
        n = x.size(0)
        if n < 2:
            raise RuntimeError(
                f"Need at least 2 normal samples to create reference_normal, got {n}."
            )

        n_ref = max(1, int(round(self.hparams.reference_fraction * n)))
        n_ref = min(n_ref, n - 1)

        gen = torch.Generator().manual_seed(self.hparams.seed)
        perm = torch.randperm(n, generator=gen)

        ref_idx = perm[:n_ref]
        main_idx = perm[n_ref:]

        main = SplitTensors(
            x=x[main_idx].contiguous(),
            y=y[main_idx].contiguous(),
        )
        ref = SplitTensors(
            x=x[ref_idx].contiguous(),
            y=y[ref_idx].contiguous(),
        )
        return main, ref

    def _build_aux_split(self, dataset: CIFAR10) -> dict[str, SplitTensors]:
        """Build auxiliary validation/test datasets.

        The returned dict does not contain the main normal split; that is stored in
        ``self._main`` and exposed separately under the ``"normal"`` key by
        ``_make_eval_loaders``. The reference normal split is included here under
        ``"reference_normal"``.
        """
        out: dict[str, SplitTensors] = {}

        signal_label = 0
        for cls in self.hparams.signal_classes:
            signal_label += 1
            indices = self._get_class_indices(dataset.targets, [cls])
            x, y = self._extract_subset(dataset, indices, label=signal_label)
            out[str(cls)] = SplitTensors(
                x=x.contiguous(),
                y=y.contiguous(),
            )

        background_label = 0
        for cls in (self.hparams.background_classes or []):
            background_label -= 1
            indices = self._get_class_indices(dataset.targets, [cls])
            x, y = self._extract_subset(dataset, indices, label=background_label)
            out[str(cls)] = SplitTensors(
                x=x.contiguous(),
                y=y.contiguous(),
            )

        return out

    def _make_eval_loaders(
        self,
        main_key: str,
        aux_key: str,
        main_name: str,
    ) -> dict[str, DataLoader]:
        """Construct evaluation dataloaders with normal first, then auxiliary sets."""
        main = self._main[main_key]
        loaders: dict[str, DataLoader] = {}

        loaders[main_name] = self._to_loader(
            main,
            batch_size=self.batch_size_per_device,
        )

        for name, split in self._aux.get(aux_key, {}).items():
            loaders[name] = self._to_loader(
                split,
                batch_size=self.batch_size_per_device,
                max_b=self.max_val_batches,
            )

        return loaders

    def _to_loader(
        self,
        split: SplitTensors,
        batch_size: int,
        max_b: int = None,
    ) -> DataLoader:
        """Transform a split into an iterable DataLoader."""
        ds = CIFARADDataset(
            split.x,
            split.y,
            batch_size=batch_size,
            max_batches=max_b,
        )
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def _extract_subset(
        self,
        dataset: CIFAR10,
        indices: list[int],
        label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and optionally normalize a class subset."""
        xs = []
        for idx in indices:
            x = (
                torch.as_tensor(dataset.data[idx], dtype=torch.float32)
                .permute(2, 0, 1)
                / 255.0
            )
            if self.hparams.normalize:
                if self.mean is None or self.std is None:
                    raise RuntimeError(
                        "Normalization requested but mean/std were not set."
                    )
                x = (x - self.mean[:, None, None]) / self.std[:, None, None]
            xs.append(x)

        x = torch.stack(xs, dim=0)
        y = torch.full((x.size(0),), label, dtype=torch.int64)
        return x, y

    def _tensor_dataset_to_tensors(self, subset) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a TensorDataset subset into two stacked tensors."""
        xs = []
        ys = []
        for x, y in subset:
            xs.append(x)
            ys.append(y)

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    @staticmethod
    def _get_class_indices(targets, classes: list[int]) -> list[int]:
        """Return dataset indices belonging to the requested classes."""
        class_set = set(classes)
        return [i for i, t in enumerate(targets) if int(t) in class_set]

    def _set_batch_size(self):
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

    def _data_summary(self, stage: str | None) -> None:
        """Print a summary of the currently loaded data."""
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)

        def show_split(title: str, key: str, aux_key: str | None = None):
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
