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
from src.utils.instrumentation import log_memory
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
    control_x: torch.Tensor | None = None
    control_mask: torch.Tensor | None = None


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
        model_input_exclude_features: list[str] | None = None,
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
        self.model_input_exclude_features = list(model_input_exclude_features or [])

        # `control_object_feature_map` describes the full tensor loaded from the
        # mlready cache. `object_feature_map` describes the model-input tensor after
        # any configured exclusions have been removed and indices re-numbered.
        self.object_feature_map: dict[str, dict[str, list[int]]] | None = None
        self.control_object_feature_map: dict[str, dict[str, list[int]]] | None = None
        self._model_keep_indices: list[int] | None = None
        self._model_excluded_indices: set[int] = set()
        self._probe_split_name: str | None = None
        self._probe_split: SplitTensors | None = None

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
        self._log_tensor_footprint(stage)

    def _log_tensor_footprint(self, stage: str | None) -> None:
        """Report how much RAM the loaded splits actually occupy.

        Two numbers, because they differ and only one of them is the memory
        the process pays for:

        * logical - element_size * nelement per tensor. This is what a naive
          sum reports, and it DOUBLE-COUNTS views. The model input is a view of
          the control tensor (see _build_model_input_view), so a logical sum
          shows x and control_x at ~5.5 GiB each when between them they own one
          5.5 GiB allocation.
        * resident - the unique underlying storages, deduplicated by data_ptr.
          This is the real footprint and the number to size request_memory
          against.

        Failures here are swallowed: instrumentation must never break a run.
        """
        try:
            mib = 1024.0 * 1024.0
            logical_total = 0.0
            rows: list[tuple[str, str, float, bool]] = []
            seen_storages: dict[int, float] = {}

            def measure(group: str, name: str, container) -> None:
                nonlocal logical_total

                if container is None:
                    return

                for field in ("x", "mask", "control_x", "control_mask"):
                    tensor = getattr(container, field, None)

                    if tensor is None or not hasattr(tensor, "element_size"):
                        continue

                    logical = tensor.element_size() * tensor.nelement() / mib
                    logical_total += logical

                    try:
                        storage = tensor.untyped_storage()
                        pointer = storage.data_ptr()
                        storage_mib = storage.nbytes() / mib
                    except Exception:  # noqa: BLE001 - older torch, or a fake tensor
                        pointer = id(tensor)
                        storage_mib = logical

                    shared = pointer in seen_storages
                    seen_storages.setdefault(pointer, storage_mib)

                    rows.append(
                        (
                            f"{group}/{name}",
                            f"{field}{tuple(tensor.shape)}",
                            logical,
                            shared,
                        )
                    )

            for name, split in sorted(self._main.items()):
                measure("main", name, split)

            for name, split in sorted(getattr(self, "_aux", {}).items()):
                if isinstance(split, dict):
                    for sub_name, sub_split in sorted(split.items()):
                        measure("aux", f"{name}/{sub_name}", sub_split)
                else:
                    measure("aux", name, split)

            resident_total = sum(seen_storages.values())

            log.info(f"[data] resident tensors after setup({stage}):")
            for label, shape, size, shared in rows:
                marker = "  (view, shares storage)" if shared else ""
                log.info(f"[data]   {label:<42} {shape:<34} {size:9.1f} MiB{marker}")

            log.info(
                f"[data]   {'LOGICAL TOTAL':<42} {'(double-counts views)':<34} "
                f"{logical_total:9.1f} MiB ({logical_total / 1024.0:.1f} GiB) "
                f"across {len(rows)} tensor(s)"
            )
            log.info(
                f"[data]   {'RESIDENT TOTAL':<42} {'(unique storages)':<34} "
                f"{resident_total:9.1f} MiB ({resident_total / 1024.0:.1f} GiB) "
                f"across {len(seen_storages)} allocation(s)"
            )
            saved = logical_total - resident_total
            if saved > 1.0:
                log.info(
                    f"[data]   {'SAVED BY VIEWS':<42} {'':<34} "
                    f"{saved:9.1f} MiB ({saved / 1024.0:.1f} GiB)"
                )
            log_memory(f"datamodule.setup({stage})")
        except Exception as error:  # noqa: BLE001 - diagnostics must not break runs
            log.warning(f"Could not compute tensor footprint: {error}")

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
            control_data=split.control_x,
            control_mask=split.control_mask,
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

    def setup_probe_split(
        self,
        split: str,
        *,
        max_samples: int | None = None,
        sample_seed: int = 12345,
    ) -> None:
        """Load one unshuffled main-background split for probe evaluation.

        Only one probe split may be resident at a time. Auxiliary signal and
        simulated-background datasets are deliberately not loaded.
        """
        allowed = {"train", "valid", "test"}
        if split not in allowed:
            raise ValueError(
                f"Unknown probe split {split!r}. "
                f"Expected one of: {sorted(allowed)}."
            )

        if self._probe_split is not None:
            raise RuntimeError(
                "A probe split is already loaded: "
                f"{self._probe_split_name!r}. "
                "Call release_probe_split() before loading another split."
            )

        if self.main_cache_folder is None:
            raise RuntimeError(
                "Cannot load a probe split because main_cache_folder is not set. "
                "Did prepare_data() run?"
            )

        self._set_batch_size()
        self.loader = self.hparams.data_awkward2torch

        load_kwargs = {}
        if max_samples is not None:
            load_kwargs = {
                "max_samples": max_samples,
                "sample_seed": sample_seed,
            }

        loaded = self._load_main_split(
            self.main_cache_folder,
            split,
            label=0,
            **load_kwargs,
        )

        self._probe_split = loaded
        self._probe_split_name = split

    def probe_dataloader(self) -> DataLoader:
        """Return an unshuffled loader for the currently loaded probe split."""
        if self._probe_split is None or self._probe_split_name is None:
            raise RuntimeError(
                "No probe split is loaded. Call setup_probe_split() first."
            )

        return self._to_loader(
            self._probe_split,
            batch_size=self.batch_size_per_device,
            max_b=None,
        )

    def release_probe_split(self) -> None:
        """Release the currently resident probe split."""
        self._probe_split = None
        self._probe_split_name = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def teardown(self, stage: str | None = None) -> None:
        # Drop references to large tensors so they become collectible
        if stage in ("fit", None):
            self._train_loader = None
            self._main.pop("train", None)

        if stage in ("fit", "validate", None):
            # Validation is also run by a separate evaluator with explicit loaders.
            # It must be releasable before the held-out test tensors are loaded.
            self._val_loaders = None
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
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

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

    def _load_main_split(
        self,
        data_dir: Path,
        split: str,
        label: int,
        flag: str | None = None,
        *,
        max_samples: int | None = None,
        sample_seed: int = 12345,
    ) -> SplitTensors:
        """Load main data splits: train, val, and test of ZB data."""
        load_kwargs = {}
        if max_samples is not None:
            load_kwargs = {
                "max_events": max_samples,
                "sample_seed": sample_seed,
            }
        control_x, control_mask, l1bit = self.loader.load_folder(
            data_dir / split,
            **load_kwargs,
        )
        self._configure_feature_views(self.loader.object_feature_map)

        x, mask = self._build_model_input_view(control_x, control_mask)

        y = torch.full((x.size(0),), label, dtype=torch.int64)
        # Not .contiguous() when the model input is a view of the control
        # tensor: materialising it here is exactly the ~14 GiB duplication
        # this avoids. See _build_model_input_view.
        if not getattr(self, '_model_input_is_view', False):
            x = x.contiguous()
            mask = mask.contiguous()
        l1bit = l1bit.contiguous()
        y = y.contiguous()

        if self._model_excluded_indices:
            control_x = control_x.contiguous()
            control_mask = control_mask.contiguous()
        else:
            control_x = None
            control_mask = None

        return SplitTensors(
            x=x,
            mask=mask,
            l1bit=l1bit,
            y=y,
            control_x=control_x,
            control_mask=control_mask,
        )

    def _load_aux_split(
        self, data_dir: Path, split: str, flag: str | None = None
    ) -> dict[str, SplitTensors]:
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

            control_x, control_mask, l1bit = self.loader.load_folder(dataset_path / split)
            self._configure_feature_views(self.loader.object_feature_map)

            x, mask = self._build_model_input_view(control_x, control_mask)

            y = torch.full((x.size(0),), label, dtype=torch.int64)
            # Not .contiguous() when the model input is a view of the control
            # tensor: materialising it here is exactly the ~14 GiB duplication
            # this avoids. See _build_model_input_view.
            if not getattr(self, '_model_input_is_view', False):
                x = x.contiguous()
                mask = mask.contiguous()
            l1bit = l1bit.contiguous()
            y = y.contiguous()

            if self._model_excluded_indices:
                control_x = control_x.contiguous()
                control_mask = control_mask.contiguous()
            else:
                control_x = None
                control_mask = None

            out[name] = SplitTensors(
                x=x,
                mask=mask,
                l1bit=l1bit,
                y=y,
                control_x=control_x,
                control_mask=control_mask,
            )

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
            control_data=split.control_x,
            control_mask=split.control_mask,
        )
        ds = self._attach_object_feature_map(ds)
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
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
                split = self._main[key]
                log.info(f"Zero bias model input: {tuple(split.x.shape)}")
                if split.control_x is not None:
                    log.info(f"Zero bias control tensor: {tuple(split.control_x.shape)}")

            if aux_key:
                for name, split in self._aux.get(aux_key, {}).items():
                    log.info(f"{name} model input: {tuple(split.x.shape)}")
                    if split.control_x is not None:
                        log.info(f"{name} control tensor: {tuple(split.control_x.shape)}")

        if stage in (None, "fit"):
            show_split("Training data:", "train")
            show_split("Validation data:", "valid", "valid")
        elif stage == "validate":
            show_split("Validation data:", "valid", "valid")
        elif stage == "test":
            show_split("Test data:", "test", "test")

    def _configure_feature_views(self, raw_object_feature_map: dict) -> None:
        """Create model-input and control feature maps from the raw cached layout."""
        raw_map = self._normalise_feature_map(raw_object_feature_map)

        if self.control_object_feature_map is not None:
            if raw_map != self.control_object_feature_map:
                raise RuntimeError(
                    "Loaded splits do not share the same object_feature_map layout. "
                    "Cannot safely apply model-input feature exclusions."
                )
            return

        self.control_object_feature_map = raw_map
        n_features = self._num_flat_features(raw_map)

        excluded = self._resolve_feature_indices(
            raw_map,
            self.model_input_exclude_features,
        )

        self._model_excluded_indices = set(excluded)
        self._model_keep_indices = [
            idx for idx in range(n_features) if idx not in self._model_excluded_indices
        ]

        self.object_feature_map = self._reindex_feature_map(raw_map, excluded)
        self._assert_excluded_features_absent()

        if excluded:
            log.info(
                Back.GREEN
                + "Model-input feature exclusion active: "
                + f"removed {self.model_input_exclude_features} at raw indices {excluded}. "
                + f"Model input has {len(self._model_keep_indices)} features; "
                + f"control tensor keeps {n_features} features."
            )


    def _keep_indices_are_contiguous_run(self) -> bool:
        """True when the kept features form one unbroken ascending range.

        This is the condition under which the model input can be a *view* of
        the control tensor instead of a second copy of it. With the current
        configuration - excluding only FET.Et, which sits at raw index 0 - the
        kept indices are 1..116, so the view path applies.
        """
        keep = self._model_keep_indices

        if not keep:
            return False

        return keep == list(range(keep[0], keep[-1] + 1))

    def _build_model_input_view(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the anomaly-detector input view while keeping control data intact.

        The model input is the control tensor minus the excluded (sensitive)
        features, so materialising both costs almost exactly twice the memory
        of the data itself. Measured 2026-09-18: setup("fit") held 27.5 GiB, of
        which ~14 GiB was this duplication - for the train split alone, x was
        5545.6 MiB against control_x at 5593.4 MiB for the same numbers.

        When the kept features form one contiguous run, basic slicing returns a
        *view* of the control tensor rather than a copy, so the model input
        costs nothing. The view is strided rather than contiguous. Batching
        slices rows (L1ADDataset does data[s:e]), which preserves the view, and
        any per-batch copy PyTorch makes inside the first Linear is ~7.6 MB at
        batch_size 16384 - irrelevant beside gigabytes of resident tensor.

        Two invariants follow, both true today: the caller must NOT call
        .contiguous() on the result, which would reinstate the copy, and the
        control tensor must outlive the model input, since it now owns the
        storage.

        index_select remains the fallback for a non-contiguous keep set,
        because it is the only correct option there. That path still copies.
        """
        if self._model_keep_indices is None:
            raise RuntimeError("Feature views are not configured yet.")

        if not self._model_excluded_indices:
            self._model_input_is_view = False
            return x, mask

        x_flat = torch.flatten(x, start_dim=1)
        mask_flat = torch.flatten(mask, start_dim=1)

        if self._keep_indices_are_contiguous_run():
            first = self._model_keep_indices[0]
            last = self._model_keep_indices[-1] + 1
            x_view = x_flat[:, first:last]

            # Cheap equivalence guard on a handful of rows. The slice and the
            # index_select must agree exactly; if a future exclusion set makes
            # the arithmetic wrong, this fails loudly at load time instead of
            # silently training on the wrong columns - including, potentially,
            # on the sensitive variable the protocol forbids in the input.
            probe_rows = min(8, x_view.shape[0])
            if probe_rows:
                keep_probe = torch.as_tensor(
                    self._model_keep_indices, device=x.device, dtype=torch.long
                )
                if not torch.equal(
                    x_view[:probe_rows],
                    x_flat[:probe_rows].index_select(dim=1, index=keep_probe),
                ):
                    raise RuntimeError(
                        "Contiguous-run model-input view does not match the "
                        "index_select result. Refusing to continue: this would "
                        "feed the wrong feature columns to the model."
                    )

            self._model_input_is_view = True
            log.info(
                f"Model input is a view of the control tensor "
                f"(columns {first}:{last}); no second copy is materialised."
            )
            return x_view, mask_flat[:, first:last]

        keep = torch.as_tensor(
            self._model_keep_indices,
            device=x.device,
            dtype=torch.long,
        )
        self._model_input_is_view = False

        return (
            x_flat.index_select(dim=1, index=keep),
            mask_flat.index_select(dim=1, index=keep),
        )


    def _normalise_feature_map(
        self,
        object_feature_map: dict,
    ) -> dict[str, dict[str, list[int]]]:
        return {
            str(obj): {
                str(feat): [int(idx) for idx in indices]
                for feat, indices in feature_map.items()
            }
            for obj, feature_map in object_feature_map.items()
        }


    def _num_flat_features(self, object_feature_map: dict) -> int:
        all_indices = [
            int(idx)
            for feature_map in object_feature_map.values()
            for indices in feature_map.values()
            for idx in indices
        ]

        if not all_indices:
            raise RuntimeError("object_feature_map does not contain any feature indices.")

        return max(all_indices) + 1


    def _resolve_feature_indices(
        self,
        object_feature_map: dict,
        feature_refs: list[str],
    ) -> list[int]:
        excluded: list[int] = []

        for feature_ref in feature_refs:
            if "." not in feature_ref:
                raise ValueError(
                    "model_input_exclude_features entries must have format "
                    f"'<object>.<feature>', got {feature_ref!r}."
                )

            object_name, feature_name = feature_ref.split(".", maxsplit=1)

            object_key = self._find_case_insensitive_key(
                object_feature_map,
                object_name,
                "object",
            )

            feature_map = object_feature_map[object_key]

            feature_key = self._find_case_insensitive_key(
                feature_map,
                feature_name,
                f"feature for object {object_key!r}",
            )

            excluded.extend(int(idx) for idx in feature_map[feature_key])

        return sorted(set(excluded))


    def _reindex_feature_map(
        self,
        object_feature_map: dict,
        excluded_indices: list[int],
    ) -> dict[str, dict[str, list[int]]]:
        excluded = set(excluded_indices)
        n_features = self._num_flat_features(object_feature_map)

        old_to_new = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(
                idx for idx in range(n_features) if idx not in excluded
            )
        }

        reindexed: dict[str, dict[str, list[int]]] = {}

        for object_name, feature_map in object_feature_map.items():
            kept_features: dict[str, list[int]] = {}

            for feature_name, indices in feature_map.items():
                kept = [
                    old_to_new[int(idx)]
                    for idx in indices
                    if int(idx) in old_to_new
                ]

                if kept:
                    kept_features[feature_name] = kept

            if kept_features:
                reindexed[object_name] = kept_features

        return reindexed


    def _assert_excluded_features_absent(self) -> None:
        if not self.model_input_exclude_features:
            return

        try:
            leaked = self._resolve_feature_indices(
                self.object_feature_map,
                self.model_input_exclude_features,
            )
        except KeyError:
            leaked = []

        if leaked:
            raise RuntimeError(
                "Configured control-only features are still present in the model input: "
                f"{self.model_input_exclude_features}. Reindexed positions: {leaked}."
            )


    def _find_case_insensitive_key(
        self,
        mapping: dict,
        requested_key: str,
        kind: str,
    ) -> str:
        for key in mapping.keys():
            if str(key).lower() == requested_key.lower():
                return key

        raise KeyError(
            f"Could not find {kind} {requested_key!r}. "
            f"Available keys: {list(mapping.keys())}"
        )

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

            dataset = L1ADDataset(
                split.x,
                split.mask,
                split.l1bit,
                split.y,
                batch_size=self.batch_size_per_device,
                shuffler=self.shuffler,
                control_data=split.control_x,
                control_mask=split.control_mask,
            )

            return self._attach_object_feature_map(dataset)

        if stage not in {"val", "test"}:
            raise ValueError(
                f"Unknown stage '{stage}'. Expected one of: 'train', 'val', 'test'."
            )

        split_name = "valid" if stage == "val" else "test"
        main_key = "normal"

        # Main split. Keep this first in the returned dict.
        main = self._load_main_split(data_dir, split_name, label=0, flag=flag)

        main_dataset = L1ADDataset(
            main.x,
            main.mask,
            main.l1bit,
            main.y,
            batch_size=self.batch_size_per_device,
            control_data=main.control_x,
            control_mask=main.control_mask,
        )

        out: dict[str, L1ADDataset] = {
            main_key: self._attach_object_feature_map(main_dataset)
        }

        # Auxiliary signal/background splits.
        aux = self._load_aux_split(data_dir, split_name, flag=flag)

        for name, split in aux.items():
            dataset = L1ADDataset(
                split.x,
                split.mask,
                split.l1bit,
                split.y,
                batch_size=self.batch_size_per_device,
                control_data=split.control_x,
                control_mask=split.control_mask,
            )

            out[name] = self._attach_object_feature_map(dataset)

        return out

    def _attach_object_feature_map(self, ds: Dataset) -> Dataset:
        if self.object_feature_map is not None:
            ds.object_feature_map = self.object_feature_map
        elif hasattr(self, "loader") and hasattr(self.loader, "object_feature_map"):
            ds.object_feature_map = self.loader.object_feature_map

        if self.control_object_feature_map is not None:
            ds.control_object_feature_map = self.control_object_feature_map

        return ds
