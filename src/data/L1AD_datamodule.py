# Lightning data module for loading parquet data produced with:
# https://github.com/bb511/adl1t_datamaker
from dataclasses import dataclass
from pathlib import Path
import gc
import warnings

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.utils import pylogger
from colorama import Fore, Back, Style
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

        The files processed by this class originate from samples produced with
        https://gitlab.cern.ch/cms-l1-ad/l1tntuple-maker/-/tree/new_h5generation

        :param zerobias: Dictionary of paths to the zerobias data.
        :param signals: Dictionary of paths to simulation data of possible anomalies.
        :param background: Dictionary of paths to simulation of data that is guaranteed
            to not contain any anomalies.
        :param data_extractor: Class that extracts the data from the given h5 files.
        :param data_processor: Class that processes the extracted data.
        :param data_normalizer: Class that normalizes the processed data.
        :param data_mlready: Class formats the data to be ready for ML pipeline.
        :param data_awkward2np: Class that converts the data from jagged awkward arrays
            to fixed size numpy arrays to give to the torch dataloader.
        :param train_features: Dictionary where keys are strings of the objects that
            point to list of features to be used during
        :param l1_scales: Dictionary of the scales that the l1 trigger applies to
            all features that could be in the data set.
        :param batch_size: Integer specifying the batch size of the data.
        :param max_val_batches: Integer specifying how many batches to use for the val
            data sets.
        :param seed: Integer specifying the seed with which to shuffle the training
            data when constructing the data set.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.l1_scales = l1_scales
        self.normalizer: L1DataNormalizer | None = None
        self.main_cache_folder: Path | None = None

        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)
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
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return self._make_eval_loaders(
            main_key="valid", aux_key="valid", main_name="main_val"
        )

    def test_dataloader(self):
        return self._make_eval_loaders(
            main_key="test", aux_key="test", main_name="main_test"
        )

    def teardown(self, stage: str | None = None) -> None:
        # Drop references to large tensors/dicts so they become collectible
        gc.collect()

        # If CUDA is ever used in other experiments, this helps allocator reuse
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom dataset to gpu faster."""
        tensors = [t.contiguous() for t in batch]
        if device.type == "cuda":
            tensors = [t.pin_memory().to(device, non_blocking=True) for t in tensors]
        else:
            tensors = [t.to(device) for t in tensors]

        return tuple(tensors)

    def _load_main_split(self, data_dir: Path, split: str, label: int) -> SplitTensors:
        """Load main data splits: train, val, and test of ZB data."""
        x, mask, l1bit = self.loader.load_folder(data_dir / split)
        y = torch.full((x.size(0),), label, dtype=torch.int64)
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

    def get_extra(
        self, normalizer: L1DataNormalizer, extra_feats: dict, stage: str, flag: str
    ):
        """Hook for callbacks to get additional data.

        The data provided through this hook should not be already included in the
        training data. Otherwise, no point in calling this hook.

        :param normalizer: Normalizer object for the additional data.
        :param extra_feats: Dictionary containing the object and the features to be
            extracted from that object.
        :param flag: String specifying subdirectory to put the extra feature parquet
            files in so they don't get mixed up at training time.
        """
        log.info(Back.GREEN + f"Extracting additional features: {extra_feats}...")
        self.hparams.data_mlready.prepare(normalizer, extra_feats, flag)
        data_dir: Path = self.hparams.data_mlready.cache_folder

        if stage == "train":
            split = self._load_main_split(data_dir, "train", label=0, flag=flag)
            return L1ADDataset(
                split.x,
                split.mask,
                split.l1bit,
                split.y,
                batch_size=self.batch_size_per_device,
                shuffler=self.shuffler,
            )

        if stage not in {"val", "test"}:
            raise ValueError(
                f"Unknown stage '{stage}'. Expected one of: 'train', 'val', 'test'."
            )

        split_name = "valid" if stage == "val" else "test"
        main_key = "main_val" if stage == "val" else "main_test"

        # Main split (ensure it's first in the returned dict)
        main = self._load_main_split(data_dir, split_name, label=0, flag=flag)

        out: dict[str, L1ADDataset] = {
            main_key: L1ADDataset(
                main.x,
                main.mask,
                main.l1bit,
                main.y,
                batch_size=self.batch_size_per_device,
            )
        }

        # Aux splits
        aux = self._load_aux_split(data_dir, split_name, flag=flag)
        for name, split in aux.items():
            out[name] = L1ADDataset(
                split.x,
                split.mask,
                split.l1bit,
                split.y,
                batch_size=self.batch_size_per_device,
            )

        return out
