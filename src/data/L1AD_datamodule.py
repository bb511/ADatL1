# Lightning data module for loading parquet data produced with:
# https://github.com/bb511/adl1t_datamaker
from dataclasses import dataclass
from pathlib import Path
import gc
import warnings

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.utils import pylogger
from colorama import Fore, Back
from src.data.components.dataset import L1ADDataset
from src.data.components.normalization import L1DataNormalizer

log = pylogger.RankedLogger(__name__)


# Ignore warnings inherent to the custom data loader.
# They do not make a difference anyways.
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    mask: torch.Tensor
    l1bit: torch.Tensor
    y: torch.Tensor


class L1ADDataModule(LightningDataModule):
    def __init__(
        self,
        zerobias: dict,
        signal: dict,
        background: dict,
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
        data_normalizer: "L1DataNormalizer",
        data_mlready: "L1DataMLReady",
        data_awkward2torch: "L1DataAwkward2Torch",
        train_features: dict,
        l1_scales: dict,
        batch_size: int = 16384,
        max_val_batches: int = -1,
        seed: int = 42,
    ) -> None:
        """Prepare the L1 data for using it to train and validate ML models.

        The five data_* components form the pipeline, run in that order by
        prepare_data(): extract -> process -> normalize + split -> awkward to torch.

        :param zerobias: {name: path} of real zero bias data, the training set.
        :param signal: {name: path} of simulated anomalies, validation only.
        :param background: {name: path} of simulation guaranteed anomaly-free.
        :param train_features: {object: [features]} to train on, e.g. muons: [Et, eta].
        :param l1_scales: Hardware-to-physical unit factors, kept for the pure-rate
            calculation. That is not implemented yet, so nothing applies them today and
            the tensors stay in integer hardware units.
        :param max_val_batches: Batches to keep per auxiliary val/test set. -1 keeps all.
        :param seed: Seeds the training batch order only. The train/valid/test split is
            seeded separately by data_mlready.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.l1_scales = l1_scales
        self.normalizer: L1DataNormalizer | None = None
        self.main_cache_folder: Path | None = None

        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {"valid": {}, "test": {}}
        # seed=None means 'do not seed', matching train.py's `if cfg.get("seed")`.
        self.shuffler = torch.Generator()
        if seed is not None:
            self.shuffler.manual_seed(seed)
        self.max_val_batches = max_val_batches

    def prepare_data(self) -> None:
        """Get zero bias data and the simulated MC signal data."""
        log.info(Back.GREEN + "Extracting Data...")
        self.hparams.data_extractor.extract(self.hparams.zerobias, "zerobias")
        self.hparams.data_extractor.extract(self.hparams.background, "background")
        self.hparams.data_extractor.extract(self.hparams.signal, "signal")

        log.info(Back.GREEN + "Processing Data...")
        self.hparams.data_processor.process("zerobias")
        self.hparams.data_processor.process("background")
        self.hparams.data_processor.process("signal")

        log.info(Back.GREEN + "Splitting data into train, val, test and normalizing...")
        self.normalizer = self.hparams.data_normalizer
        self.hparams.data_mlready.prepare(self.normalizer, self.hparams.train_features)
        self.main_cache_folder = self.hparams.data_mlready.cache_folder

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        Label the zerobias data with 0.
        Label the signal simulation with labels > 0.
        Label the background simulation with labels < 0.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()
        if self.main_cache_folder is None:
            raise RuntimeError("Cache folder not set. Did prepare_data() run?")

        log.info(Back.GREEN + "Loading data in memory...")
        self.loader = self.hparams.data_awkward2torch
        data_dir = self.main_cache_folder

        if stage in (None, "fit"):
            self._main.setdefault(
                "train", self._load_main_split(data_dir, "train", label=0)
            )
            self._main.setdefault(
                "valid", self._load_main_split(data_dir, "valid", label=0)
            )
            self._aux["valid"] = self._aux["valid"] or self._load_aux_split(
                data_dir, "valid"
            )

        if stage in (None, "validate"):
            self._main.setdefault(
                "valid", self._load_main_split(data_dir, "valid", label=0)
            )
            self._aux["valid"] = self._aux["valid"] or self._load_aux_split(
                data_dir, "valid"
            )

        if stage in (None, "test"):
            self._main.setdefault(
                "test", self._load_main_split(data_dir, "test", label=0)
            )
            self._aux["test"] = self._aux["test"] or self._load_aux_split(
                data_dir, "test"
            )

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def train_dataloader(self) -> Dataset:
        """Create and return the training dataloader.

        This dataloader is based on a custom dataset class from components/dataset.py,
        which basically makes the loading of numpy arrays that are already in memory
        a bit faster.
        """
        split = self._main["train"]
        dataset = L1ADDataset(
            split.x.float(),
            split.mask,
            split.l1bit,
            split.y.float(),
            batch_size=self.batch_size_per_device,
            shuffler=self.shuffler,
        )
        dataset = self._attach_object_feature_map(dataset)
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return self._make_eval_loaders(
            main_key="valid", aux_key="valid", main_name="normal"
        )

    def test_dataloader(self):
        return self._make_eval_loaders(
            main_key="test", aux_key="test", main_name="normal"
        )

    def teardown(self, stage: str | None = None) -> None:
        # Drop references to large tensors so they become collectible
        if stage in ("fit", None):
            # free train/valid (+ aux valid)
            self._main.pop("train", None)
            self._main.pop("valid", None)
            self._aux.get("valid", {}).clear()

        if stage in ("test", None):
            # free test (+ aux test)
            self._main.pop("test", None)
            self._aux.get("test", {}).clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom dataset to gpu faster."""
        if device.type != "cuda":
            return tuple(t.to(device) for t in batch)

        out = []
        for t in batch:
            # Only pin CPU tensors; skip if already pinned
            if (
                isinstance(t, torch.Tensor)
                and t.device.type == "cpu"
                and not t.is_pinned()
            ):
                t = t.pin_memory()
            out.append(t.to(device, non_blocking=True))

        return tuple(out)

    def _load_main_split(self, data_dir: Path, split: str, label: int) -> SplitTensors:
        """Load main data splits: train, val, and test of ZB data."""
        x, mask, l1bit = self.loader.load_folder(data_dir / split)
        y = torch.full((x.size(0),), label, dtype=torch.int64)
        x = x.contiguous()
        mask = mask.contiguous()
        l1bit = l1bit.contiguous()
        y = y.contiguous()

        return SplitTensors(x=x, mask=mask, l1bit=l1bit, y=y)

    def _load_aux_split(self, data_dir: Path, split: str) -> dict[str, SplitTensors]:
        """Load a split of auxiliary data, either val or test.

        The auxiliary data is not used at training time, since it consists of
        simulations for the background of the signal.
        """
        aux_dir = data_dir / "aux"
        out: dict[str, SplitTensors] = {}

        label_signal = 0
        label_background = 0

        for dataset_path in sorted(
            p for p in aux_dir.iterdir() if p.is_dir() and not p.name.startswith("._")
        ):
            name = dataset_path.stem
            if "SingleNeutrino" in name:
                label_background -= 1
                label = label_background
            else:
                label_signal += 1
                label = label_signal

            x, mask, l1bit = self.loader.load_folder(dataset_path / split)
            y = torch.full((x.size(0),), label, dtype=torch.int64)
            x = x.contiguous()
            mask = mask.contiguous()
            l1bit = l1bit.contiguous()
            y = y.contiguous()
            out[name] = SplitTensors(x=x, mask=mask, l1bit=l1bit, y=y)

        return out

    def _make_eval_loaders(
        self, main_key: str, aux_key: str, main_name: str
    ) -> dict[str, DataLoader]:
        """Make an evaluation loader out of the main data and aux data."""
        main = self._main[main_key]
        loaders: dict[str, DataLoader] = {}

        loaders[main_name] = self._to_loader(
            main, batch_size=self.batch_size_per_device
        )

        for name, split in self._aux.get(aux_key, {}).items():
            loaders[name] = self._to_loader(
                split, batch_size=self.batch_size_per_device, max_b=self.max_val_batches
            )

        return loaders

    def _to_loader(
        self, split: SplitTensors, batch_size: int, max_b: int = None
    ) -> DataLoader:
        """Transform a SplitTensor to a proper pytorch DataLoader."""
        ds = L1ADDataset(
            split.x,
            split.mask,
            split.l1bit,
            split.y,
            batch_size=batch_size,
            max_batches=max_b,
        )
        ds = self._attach_object_feature_map(ds)
        return DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=0, persistent_workers=False
        )

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
        """Make a neat little summary of data to be used."""
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)

        def show_split(title: str, key: str, aux_key: str | None = None):
            log.info(Fore.GREEN + title)
            if key in self._main:
                log.info(f"Zero bias: {tuple(self._main[key].x.shape)}")
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

    def _attach_object_feature_map(self, ds: Dataset) -> Dataset:
        if hasattr(self, "loader") and hasattr(self.loader, "object_feature_map"):
            ds.object_feature_map = self.loader.object_feature_map
        return ds
