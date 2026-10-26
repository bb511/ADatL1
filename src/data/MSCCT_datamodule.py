import gc
import hashlib
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from colorama import Fore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.data.components import mscct
from src.data.components.dataset import CIFARADDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

PARTS = ("train", "valid", "test")


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    y: torch.Tensor


def _set_rng(seed: int, part: str, name: str) -> np.random.Generator:
    """Return the generator of one set's draw, the same in every process and independent
    of the other sets'.

    The names enter through sha256, since Python's own string hash is salted per
    process.
    """
    digest = hashlib.sha256(f"{part}/{name}".encode()).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "big")])


def _read_cache(path: Path, key: dict) -> dict | None:
    """Return a scan's cached patches and cells, or None if the file is absent or holds
    another key."""
    if not path.exists():
        return None
    cached = torch.load(path, map_location="cpu")
    if cached["key"] != key:
        return None
    log.info(f"Loaded MS-CCT patches of {key['file']} from {path}")
    return cached


def _write_cache(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Saved under a name specific to the host and the process, then renamed into place,
    # so that runs started together on a cold data directory, shared by several hosts
    # (the NGT pods' /shared) or not, never read or write each other's partial file.
    partial = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}")
    torch.save(content, partial)
    partial.replace(path)


class MSCCTDataModule(LightningDataModule):
    """Lightning DataModule for MS-CCT, the multi-scanner canine cutaneous SCC set.

    X1 is the reference scanner (CS2). Its normal-class patches of the training slides
    train every model; those of the validation and of the test slides are halved by
    patch into ``normal`` and ``reference_normal``. The signal and background classes
    on the reference scanner follow, then one set per shift scanner holding that
    scanner's normal-class patches of the same slides, as delivered (no resampling).
    Patches, tissue filter and slide split are those of
    ``src/data/components/mscct.py``; every set holds every patch of its slides.

    Labels, as CIFAR-10's: normal 0; signal classes 1, 2, ... and background classes
    -1, -2, ... in config order; the shift scanners' sets 0. Validation and test
    loaders are ``normal``, the signal classes, the background classes,
    ``reference_normal`` and the shift scanners, each set named by its class or
    scanner.

    Each scan's patches of every configured class are cached in
    ``<data_dir>/patches/`` with their grid cells, under a key of the file name, its
    md5 and that of ``scc.json`` as the record's ``record.json`` lists them, the level,
    the patch size, the classes and the filter. A cold build checks the scan and the
    annotations against those md5s before extracting; a warm run reads ``record.json``
    and the caches only, so the scans are needed for a cold build alone. The key
    carries no code version: a change to the patch rule needs the caches deleted or a
    fresh data directory.

    The normalisation is the per-channel mean and standard deviation of the training
    patches, recomputed exactly at every setup. The training patches arrive grouped by
    slide, so they are permuted once with the split seed; ``CIFARADDataset`` then fixes
    batch membership and shuffles the batch order with ``train_seed``, as on CIFAR-10.

    ``provenance[part][set]`` holds every row's (slide, cell row, cell col) in loader
    order, for the coupling oracle of ``src/data/components/mscct_pairing.py``.

    Args:
        data_dir: Directory of the record's files and of ``patches/``.
        reference_scanner: The scanner of X1, the training set and the anomaly sets.
        shift_scanners: The scanners served as sets of their own, label 0.
        normal_classes: Tissue classes of label 0, keys of ``mscct.CLASS_IDS``.
        signal_classes: Anomaly classes of positive label.
        background_classes: Classes of negative label, reported per set.
        split_sizes: Slides of the training, validation and test parts.
        level: The pyramid page the patches are cut from.
        patch_size: Side of a patch in pixels of that page.
        white_share: The filter's share of exactly white pixels.
        flat_sd: The filter's standard deviation per channel, in grey levels.
        batch_size: Batch size of every loader, divided across devices.
        max_val_batches: Batches served of every evaluation set but ``normal``; -1
            serves all.
        reference_fraction: Share of a part's normal patches in ``reference_normal``.
        seed: Seed of the slide split and of every set's draw, fixed apart from the
            training seed so that every run sees the same patches.
        train_seed: Seed of the training batch order; ``seed`` when None.
        num_workers: Workers of every loader.
        normalize: Whether to standardise each channel.
    """

    def __init__(
        self,
        data_dir: str,
        reference_scanner: str = "cs2",
        shift_scanners: tuple[str, ...] = ("nz20", "nz210", "p1000", "gt450"),
        normal_classes: tuple[str, ...] = ("dermis",),
        signal_classes: tuple[str, ...] = ("tumour",),
        background_classes: tuple[str, ...] = ("inflammation_necrosis",),
        split_sizes: tuple[int, ...] = (24, 10, 10),
        level: int = 1,
        patch_size: int = 32,
        white_share: float = 0.5,
        flat_sd: float = 2.0,
        batch_size: int = 256,
        max_val_batches: int = -1,
        reference_fraction: float = 0.5,
        seed: int = 314159,
        train_seed: int | None = None,
        num_workers: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if not 0.0 < reference_fraction < 1.0:
            raise ValueError("reference_fraction must be strictly between 0 and 1.")
        self.data_dir = Path(data_dir)
        self.shuffler = torch.Generator()
        self.shuffler.manual_seed(seed if train_seed is None else train_seed)
        self.max_val_batches = max_val_batches
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        # {part: {set: tensors}} in loader order; the training part holds normal alone.
        self._sets: dict[str, dict[str, SplitTensors]] = {}
        self.provenance: dict[str, dict[str, np.ndarray]] = {}
        self._md5s: dict[str, str] | None = None
        self._annotations: dict | None = None

    @property
    def footprint(self) -> int:
        """Page-0 side of a patch, the grid unit of the coupling oracle."""
        return self.hparams.patch_size << self.hparams.level

    @property
    def data_train(self) -> TensorDataset | None:
        """The training split as (x, y), where the dump and residual callbacks read it.

        None before setup.
        """
        train = self._sets.get("train")
        normal = None if train is None else train["normal"]
        return None if normal is None else TensorDataset(normal.x, normal.y)

    def prepare_data(self) -> None:
        """Check that the record's file list is present; setup reads the scans."""
        self._record()

    def setup(self, stage: str | None = None) -> None:
        """Assemble the parts the stage needs: train and validation, test, or all."""
        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented.")
        self._set_batch_size()
        parts = self._parts()
        if self.hparams.normalize and self.mean is None:
            self._set_normalization(parts["train"])
        wanted = {"fit": PARTS[:2], "validate": PARTS[:2], "test": PARTS[2:]}
        for part in wanted.get(stage, PARTS):
            if part not in self._sets:
                self._assemble(part, parts[part])
        self._data_summary()

    def train_dataloader(self) -> DataLoader:
        """Return the training loader, its batch order drawn anew every epoch."""
        split = self._sets["train"]["normal"]
        return self._loader(
            CIFARADDataset(
                split.x,
                split.y,
                batch_size=self.batch_size_per_device,
                shuffler=self.shuffler,
            )
        )

    def val_dataloader(self) -> dict[str, DataLoader]:
        """Return the validation loaders by set name, normal first."""
        return self._evaluation_loaders("valid")

    def test_dataloader(self) -> dict[str, DataLoader]:
        """Return the test loaders by set name, normal first."""
        return self._evaluation_loaders("test")

    def teardown(self, stage: str | None = None) -> None:
        """Release the tensors of the stage's parts, as CIFAR10DataModule does."""
        released = {"fit": PARTS[:2], "test": PARTS[2:], None: PARTS}.get(stage, ())
        for part in released:
            self._sets.pop(part, None)
            self.provenance.pop(part, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer 2-tuple batches to device."""
        x, y = batch

        if device.type != "cuda":
            return x.to(device), y.to(device)

        if x.device.type == "cpu" and not x.is_pinned():
            x = x.pin_memory()
        if y.device.type == "cpu" and not y.is_pinned():
            y = y.pin_memory()

        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def _record(self) -> dict[str, str]:
        """Return the record's md5s by file name, read once."""
        if self._md5s is None:
            path = self.data_dir / mscct.RECORD
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing: MS-CCT is Zenodo record {mscct.DOI}, kept "
                    "with the API's record.json beside its files."
                )
            self._md5s = mscct.record_md5s(path)
        return self._md5s

    def _parts(self) -> dict[str, list[int]]:
        """Return the slides of the training, validation and test parts."""
        suffix = f"_{self.hparams.reference_scanner}.tif"
        slides = [int(n.split("_")[1]) for n in self._record() if n.endswith(suffix)]
        sizes = list(self.hparams.split_sizes)
        return dict(zip(PARTS, mscct.slide_split(slides, sizes, self.hparams.seed)))

    def _set_normalization(self, slides: list[int]) -> None:
        """Set the per-channel moments of the training patches, on the 0-1 scale."""
        hp = self.hparams
        patches, _ = self._raw(hp.reference_scanner, hp.normal_classes, slides)
        mean, std = mscct.channel_moments(patches)
        self.mean = torch.from_numpy(mean / 255.0).float()
        self.std = torch.from_numpy(std / 255.0).float().clamp_min(1e-8)
        log.info(f"MS-CCT normalization mean: {self.mean.tolist()}")
        log.info(f"MS-CCT normalization std: {self.std.tolist()}")

    def _assemble(self, part: str, slides: list[int]) -> None:
        """Build a part's sets and their provenance."""
        hp = self.hparams
        normal, provenance = self._raw(hp.reference_scanner, hp.normal_classes, slides)
        if part == "train":
            order = _set_rng(hp.seed, part, "normal").permutation(len(normal))
            sets = {"normal": (normal[order], provenance[order], 0)}
        else:
            sets = self._evaluation_sets(part, slides, normal, provenance)
        self._sets[part] = {n: self._tensors(x, y) for n, (x, _, y) in sets.items()}
        self.provenance[part] = {n: p for n, (_, p, _) in sets.items()}

    def _evaluation_sets(
        self,
        part: str,
        slides: list[int],
        normal: np.ndarray,
        provenance: np.ndarray,
    ) -> dict[str, tuple]:
        """Return a validation or test part's sets as ``{name: (patches, provenance,
        label)}`` in loader order."""
        hp = self.hparams
        order = _set_rng(hp.seed, part, "reference_normal").permutation(len(normal))
        n_ref = round(hp.reference_fraction * len(normal))
        n_ref = min(max(1, n_ref), len(normal) - 1)
        main, ref = order[n_ref:], order[:n_ref]
        sets = {"normal": (normal[main], provenance[main], 0)}
        labelled = [(c, i + 1) for i, c in enumerate(hp.signal_classes)]
        labelled += [(c, -(i + 1)) for i, c in enumerate(hp.background_classes or [])]
        for name, label in labelled:
            sets[name] = (*self._raw(hp.reference_scanner, [name], slides), label)
        sets["reference_normal"] = (normal[ref], provenance[ref], 0)
        for scanner in hp.shift_scanners or []:
            sets[scanner] = (*self._raw(scanner, hp.normal_classes, slides), 0)
        return sets

    def _raw(
        self, scanner: str, classes: list[str], slides: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the patches [n, p, p, 3] uint8 of ``classes`` on one scanner's scans
        of ``slides``, slide by slide and class by class, and their provenance [n, 3]
        int32 (slide, cell row, cell col)."""
        patches, provenance = [], []
        for slide in slides:
            scan = self._scan(slide, scanner)
            for name in classes:
                cells = scan["cells"][name].numpy()
                patches.append(scan["patches"][name].numpy())
                slide_column = np.full(len(cells), slide, dtype=np.int32)
                provenance.append(np.column_stack([slide_column, cells]))
        return np.concatenate(patches), np.concatenate(provenance)

    def _tensors(self, patches: np.ndarray, label: int) -> SplitTensors:
        """Return patches [n, p, p, 3] uint8 as float32 [n, 3, p, p] with their
        labels."""
        x = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
        if self.hparams.normalize:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        y = torch.full((len(x),), label, dtype=torch.int64)
        return SplitTensors(x=x.contiguous(), y=y)

    def _scan(self, slide: int, scanner: str) -> dict:
        """Return one scan's patches and cells of every configured class, from its cache
        when the key matches, otherwise extracted and cached."""
        name = mscct.file_name(slide, scanner)
        key = self._cache_key(name)
        stem = f"{Path(name).stem}_level{key['level']}_p{key['patch']}.pt"
        path = self.data_dir / "patches" / stem
        scan = _read_cache(path, key)
        if scan is None:
            scan = self._extract(name, key)
            _write_cache(path, scan)
        return scan

    def _cache_key(self, name: str) -> dict:
        """Return the key a scan's cache must hold, everything that fixes its content.

        The md5s are the record's; the cold build checks the scan and the annotations
        against them, so a warm run reads neither.
        """
        hp, md5s = self.hparams, self._record()
        classes = {*hp.normal_classes, *hp.signal_classes}
        classes |= set(hp.background_classes or [])
        return dict(
            file=name,
            md5=md5s[name],
            annotations=md5s[mscct.ANNOTATIONS],
            level=int(hp.level),
            patch=int(hp.patch_size),
            classes=sorted(classes),
            white_share=float(hp.white_share),
            flat_sd=float(hp.flat_sd),
            painting="decreasing_area",
        )

    def _extract(self, name: str, key: dict) -> dict:
        """Return a scan's patches and cells in the cache's layout, read from the
        scan."""
        start = time.perf_counter()
        hp = self.hparams
        image = self._annotations_by_file()[name]
        extracted = mscct.extract(
            self._checked(name),
            image,
            key["classes"],
            hp.level,
            hp.patch_size,
            hp.white_share,
            hp.flat_sd,
        )
        elapsed = time.perf_counter() - start
        log.info(f"Extracted MS-CCT patches of {name} in {elapsed:.1f} s")
        patches = {c: torch.from_numpy(p) for c, (p, _) in extracted.items()}
        cells = {c: torch.from_numpy(rc) for c, (_, rc) in extracted.items()}
        return dict(key=key, patches=patches, cells=cells)

    def _annotations_by_file(self) -> dict:
        """Return the polygons of every scan, read once after checking scc.json."""
        if self._annotations is None:
            path = self._checked(mscct.ANNOTATIONS)
            self._annotations = mscct.annotations_by_file(path)
        return self._annotations

    def _checked(self, file: str) -> Path:
        """Return the path of one of the record's files after checking its md5."""
        path = self.data_dir / file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: MS-CCT is Zenodo record {mscct.DOI}."
            )
        if mscct.file_md5(path) != self._record()[file]:
            raise ValueError(f"{path} differs from the record's md5.")
        return path

    def _evaluation_loaders(self, part: str) -> dict[str, DataLoader]:
        """Return a part's loaders, normal first; ``max_val_batches`` caps the rest."""
        loaders = {}
        for name, split in self._sets[part].items():
            max_batches = None if name == "normal" else self.max_val_batches
            dataset = CIFARADDataset(
                split.x,
                split.y,
                batch_size=self.batch_size_per_device,
                max_batches=max_batches,
            )
            loaders[name] = self._loader(dataset)
        return loaders

    def _loader(self, dataset: CIFARADDataset) -> DataLoader:
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
        log.info(Fore.MAGENTA + "-" * 5 + " MS-CCT Data Summary " + "-" * 5)
        for part, sets in self._sets.items():
            log.info(Fore.GREEN + f"{part}:")
            for name, split in sets.items():
                log.info(f"{name}: {tuple(split.x.shape)}")
