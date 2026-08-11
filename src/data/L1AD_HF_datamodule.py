# Lightning data module for the record published at:
# https://huggingface.co/datasets/podagiu/anomaly_detection_cmsl1t
import gc
import sys
import warnings
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.utils import pylogger
from colorama import Fore, Back
from src.data.components.dataset import L1ADDataset

log = pylogger.RankedLogger(__name__)

# The record keeps the same normalisation schemes as this project, under config names of
# its own, so an experiment that swaps the normalizer can be carried over to it.
RECORD_NORMALIZERS = {
    "robust": "robust",
    "standard": "standard",
    "unnormalized": "default",
    "robust_axov4": "axov4",
}


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


def _record_normalizer(name: str) -> str:
    """The record's configuration name for one of this project's normalisation schemes."""
    if name not in RECORD_NORMALIZERS:
        raise ValueError(
            f"The record has no normalisation called {name}. It carries "
            f"{sorted(RECORD_NORMALIZERS)}."
        )

    return RECORD_NORMALIZERS[name]


def compose_record_config(root: str, overrides: list) -> DictConfig:
    """Compose the record's own configuration tree, leaving the caller's Hydra intact.

    initialize_config_dir refuses to run while a global Hydra is live, which it is
    inside any hydra.main application, and it restores only the state it saw at its own
    entry, which is the state after the clear below. The running application's Hydra is
    therefore set aside here and put back by hand. Composing outside an application,
    as the comparison script does, takes the same path with nothing to restore.
    """
    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(Path(root) / "configs"), version_base=None):
            return compose("config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
        if outer is not None:
            GlobalHydra.instance().initialize(outer)


class L1ADHFDataModule(LightningDataModule):
    """Train on the published record, through the loader that ships inside it.

    The record carries its own pipeline under loader/ and the configuration tree that
    drives it under configs/, so no stage of this project runs here: the record is
    downloaded, its own configuration is composed, and this class wraps the tensors it
    hands back in the dataloaders the rest of the project expects. L1ADDataModule is
    the parallel route, reading the raw files with this project's own stages.

    :param repo_id: The data set repository on the HuggingFace hub.
    :param revision: A commit of that repository to pin, or None to follow its head.
    :param cache_dir: Where the record's derived caches go, its paths.base_data_dir.
        None leaves the record's own default, which honours ADL1T_CACHE.
    :param data_normalizer: Read for its name alone, which picks the matching scheme in
        the record. It is here so that an experiment overriding /data/data_normalizer,
        as the dte ones do, composes against this config as well; the record runs its
        own normalizer and this object is never used to normalize anything.
    :param overrides: Anything else to override when composing the record's
        configuration, e.g. ['data/data_awkward2torch=minimal'].
    :param max_val_batches: Batches to keep per auxiliary val/test set. -1 keeps all.
    :param seed: Seeds the training batch order only. The record is published already
        split, so nothing here draws one.
    """

    def __init__(
        self,
        repo_id: str = "podagiu/anomaly_detection_cmsl1t",
        revision: str | None = None,
        cache_dir: str | None = None,
        data_normalizer: "L1DataNormalizer" = None,
        overrides: list | None = None,
        batch_size: int = 16384,
        max_val_batches: int = -1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.record = None
        # The record's awkward2torch stage, which carries the object feature map.
        self.loader = None

        self._main: dict = {}
        self._aux: dict[str, dict] = {"valid": {}, "test": {}}
        # seed=None means 'do not seed', matching train.py's `if cfg.get("seed")`.
        self.shuffler = torch.Generator()
        if seed is not None:
            self.shuffler.manual_seed(seed)
        self.max_val_batches = max_val_batches

    def prepare_data(self) -> None:
        """Download the record and run its pipeline. Cached, so reruns are cheap."""
        log.info(Back.GREEN + f"Preparing {self.hparams.repo_id}...")
        self._ensure_record().prepare()

    def setup(self, stage: str = None) -> None:
        """Load the record's tensors in memory.

        The record labels its zero bias 0, its simulated backgrounds negative and its
        signals positive, which is the convention the callbacks read the sign of.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()
        record = self._ensure_record()

        log.info(Back.GREEN + "Loading data in memory...")
        if stage in (None, "fit"):
            self._main.setdefault("train", record.load("train"))

        if stage in (None, "fit", "validate"):
            self._main.setdefault("valid", record.load("valid"))
            self._aux["valid"] = self._aux["valid"] or record.load_aux("valid")

        if stage in (None, "test"):
            self._main.setdefault("test", record.load("test"))
            self._aux["test"] = self._aux["test"] or record.load_aux("test")

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def _ensure_record(self):
        """Download the record and instantiate the pipeline it ships, once."""
        if self.record is not None:
            return self.record

        root = snapshot_download(
            self.hparams.repo_id,
            repo_type="dataset",
            revision=self.hparams.revision,
        )
        # The record's _target_s name loader.*, so its root has to be importable.
        if root not in sys.path:
            sys.path.insert(0, root)

        cfg = compose_record_config(root, self._record_overrides(root))
        self.record = instantiate(cfg.data)
        self.loader = self.record.data_awkward2torch

        return self.record

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        overrides = [f"paths.root_dir={root}"]
        if self.hparams.cache_dir is not None:
            overrides.append(f"paths.base_data_dir={self.hparams.cache_dir}")
        if self.hparams.data_normalizer is not None:
            scheme = _record_normalizer(self.hparams.data_normalizer.name)
            overrides.append(f"data/data_normalizer={scheme}")

        return overrides + [str(o) for o in (self.hparams.overrides or [])]

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

    def _to_loader(self, split, batch_size: int, max_b: int = None) -> DataLoader:
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
        if getattr(self.loader, "object_feature_map", None) is not None:
            ds.object_feature_map = self.loader.object_feature_map
        return ds
