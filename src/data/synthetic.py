from dataclasses import dataclass
from typing import Literal

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.dataset import CIFARADDataset, L1ADDataset


PHYSICS_SIGNAL_NAMES = (
    "GluGluHTo2B_Par-MH-125",
    "GluGluHto2G_Par-MH-125",
    "GluGluHto2G_Par-MH-90",
    "GluGluHto2Tau_Par-MH-125",
    "GluGlutoHHto2B2WtoLNu2Q_Par-c2-0-kl-1-kt-1",
    "HHHto4B2Tau_Par-c3-0-d4-0",
    "HHHto6B_Par-c3-0-d4-0",
    "SUSYGluGluToBBHTo2B_Par-M-1200",
    "SUSYGluGluToBBHToBB_Par-M-120",
    "SUSYGluGluToBBHToBB_Par-M-350",
    "SUSYGluGluToBBHToBB_Par-M-600",
    "TTHTo2C_Par-MH-125",
    "TTHto2B_Par-MH-125",
    "VBFHTo2C_Par-MH-125",
    "VBFHto2B_Par-MH-125",
    "VBFHto2Tau_Par-MH-125",
    "WtoTauto3Mu",
    "ggH-suep-decay",
    "smj-case-A",
    "haa-4b-ma15",
)


@dataclass(frozen=True)
class _Split:
    x: torch.Tensor
    mask: torch.Tensor
    l1bit: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class _ImageSplit:
    x: torch.Tensor
    y: torch.Tensor


class _IdentityNormalizer:
    def __init__(self, n_features: int):
        self.scale_tensor = torch.ones(n_features)

    def setup_1d_denorm(self, object_feature_map):  # noqa: D401
        return None

    def denorm_1d_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def norm_1d_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SyntheticLoader:
    def __init__(self, n_features: int):
        self.object_feature_map = self._feature_map(n_features)

    @staticmethod
    def _feature_map(n_features: int):
        if n_features >= 117:
            return _SyntheticLoader._map_from_layout(
                n_features,
                (
                    ("FET", ("Et", "eta", "phi"), 1),
                    ("egammas", ("Et", "eta", "phi"), 12),
                    ("jets", ("Et", "eta", "phi"), 10),
                    ("muons", ("Et", "eta", "phi"), 4),
                    ("taus", ("Et", "eta", "phi"), 12),
                ),
            )

        return _SyntheticLoader._map_from_layout(
            n_features,
            (
                ("MET", ("Et", "eta", "phi"), 1),
                ("muons", ("Et", "eta", "phi"), 5),
                ("jets", ("Et", "eta", "phi"), 5),
                ("egammas", ("Et", "eta", "phi"), 5),
                ("taus", ("Et", "eta", "phi"), 5),
            ),
        )

    @staticmethod
    def _map_from_layout(n_features: int, layout):
        mapping = {}
        idx = 0
        for obj_name, feat_names, n_objects in layout:
            feats = {feat_name: [] for feat_name in feat_names}
            width = len(feat_names)
            for _ in range(n_objects):
                if idx + width > n_features:
                    break
                for offset, feat_name in enumerate(feat_names):
                    feats[feat_name].append(idx + offset)
                idx += width
            if feats[feat_names[0]]:
                mapping[obj_name] = feats
        return mapping


class SyntheticL1ADDataModule(LightningDataModule):
    """In-memory L1-like datamodule for smoke tests and controlled studies.

    The default ``shifted`` generator preserves the historical smoke-test behavior:
    each dataset is a Gaussian cloud with a label-dependent global mean shift.

    The ``gaussian_subspace`` generator is intended for paper-grade synthetic
    anomaly-detection studies. Normal and reference samples are Gaussian typical
    domains, while anomalies are a controlled mean shift in one identifiable
    feature. This yields closed-form score distributions for linear/projection
    anomaly scores while keeping the same dataloader contract as the L1 data.
    """

    def __init__(
        self,
        n_features: int = 57,
        n_train: int = 128,
        n_val: int = 64,
        n_test: int = 64,
        batch_size: int = 32,
        max_val_batches: int | None = None,
        seed: int = 123,
        paper_aliases: bool = False,
        generator: Literal["shifted", "gaussian_subspace"] = "shifted",
        noise_std: float = 1.0,
        reference_shift: float = 0.0,
        reference_shift_dim: int = 1,
        anomaly_shift: float = 4.0,
        anomaly_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._validate_generator_config()
        self.loader = _SyntheticLoader(n_features)
        self.normalizer = _IdentityNormalizer(n_features)
        self.l1_scales = {
            obj: {"phi": 1.0} for obj in self.loader.object_feature_map.keys()
        }
        self.shuffler = torch.Generator().manual_seed(seed)

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        n_features = self.hparams.n_features

        if stage in (None, "fit"):
            self.train_split = self._make_split(
                self.hparams.n_train, n_features, 0, self._split_generator(0)
            )
            self.val_normal = self._make_split(
                self.hparams.n_val, n_features, 0, self._split_generator(1)
            )
            self.val_reference = self._make_split(
                self.hparams.n_val, n_features, -1, self._split_generator(2)
            )
            self.val_signal = self._make_split(
                self.hparams.n_val, n_features, 1, self._split_generator(3)
            )

        if stage in (None, "validate"):
            self.val_normal = self._make_split(
                self.hparams.n_val, n_features, 0, self._split_generator(1)
            )
            self.val_reference = self._make_split(
                self.hparams.n_val, n_features, -1, self._split_generator(2)
            )
            self.val_signal = self._make_split(
                self.hparams.n_val, n_features, 1, self._split_generator(3)
            )

        if stage in (None, "test"):
            self.test_normal = self._make_split(
                self.hparams.n_test, n_features, 0, self._split_generator(4)
            )
            self.test_reference = self._make_split(
                self.hparams.n_test, n_features, -1, self._split_generator(5)
            )
            self.test_signal = self._make_split(
                self.hparams.n_test, n_features, 1, self._split_generator(6)
            )

    def _split_generator(self, offset: int) -> torch.Generator:
        return torch.Generator().manual_seed(int(self.hparams.seed) + int(offset))

    def train_dataloader(self):
        return self._loader(self.train_split, shuffler=self.shuffler)

    def val_dataloader(self):
        return self._eval_loaders(
            normal=self.val_normal,
            reference=self.val_reference,
            signal=self.val_signal,
        )

    def test_dataloader(self):
        return self._eval_loaders(
            normal=self.test_normal,
            reference=self.test_reference,
            signal=self.test_signal,
        )

    def _eval_loaders(self, normal: _Split, reference: _Split, signal: _Split):
        loaders = {
            "normal": self._loader(normal),
            "reference_normal": self._loader(reference),
            "synthetic_signal": self._loader(signal),
        }

        if self.hparams.paper_aliases:
            loaders["SingleNeutrino_E-10-gun"] = self._loader(reference)
            for name in PHYSICS_SIGNAL_NAMES:
                loaders[name] = self._loader(signal)

        return loaders

    def _make_split(
        self, n_samples: int, n_features: int, label: int, gen: torch.Generator
    ) -> _Split:
        if self.hparams.generator == "gaussian_subspace":
            return self._make_gaussian_subspace_split(
                n_samples=n_samples,
                n_features=n_features,
                label=label,
                gen=gen,
            )

        if label == 0:
            shift = 0.0
        elif label < 0:
            shift = 0.15
        else:
            shift = 1.5
        x = torch.randn(n_samples, n_features, generator=gen) + shift
        mask = torch.ones(n_samples, n_features, dtype=torch.bool)
        l1bit = torch.zeros(n_samples, dtype=torch.bool)
        y = torch.full((n_samples,), label, dtype=torch.long)
        return _Split(x=x.float(), mask=mask, l1bit=l1bit, y=y)

    def _make_gaussian_subspace_split(
        self, n_samples: int, n_features: int, label: int, gen: torch.Generator
    ) -> _Split:
        x = torch.randn(n_samples, n_features, generator=gen) * float(
            self.hparams.noise_std
        )

        if label < 0:
            shift = float(self.hparams.reference_shift)
            if shift != 0.0:
                x[:, int(self.hparams.reference_shift_dim)] += shift
        elif label > 0:
            x[:, int(self.hparams.anomaly_dim)] += float(self.hparams.anomaly_shift)

        mask = torch.ones(n_samples, n_features, dtype=torch.bool)
        l1bit = torch.zeros(n_samples, dtype=torch.bool)
        y = torch.full((n_samples,), label, dtype=torch.long)
        return _Split(x=x.float(), mask=mask, l1bit=l1bit, y=y)

    def _validate_generator_config(self) -> None:
        generator = self.hparams.generator
        if generator not in {"shifted", "gaussian_subspace"}:
            raise ValueError(
                "SyntheticL1ADDataModule generator must be one of "
                f"'shifted' or 'gaussian_subspace', got {generator!r}."
            )

        if float(self.hparams.noise_std) <= 0.0:
            raise ValueError("noise_std must be positive.")

        n_features = int(self.hparams.n_features)
        for name in ("reference_shift_dim", "anomaly_dim"):
            dim = int(getattr(self.hparams, name))
            if dim < 0 or dim >= n_features:
                raise ValueError(
                    f"{name}={dim} is outside the feature range [0, {n_features})."
                )

    def _loader(self, split: _Split, shuffler: torch.Generator | None = None):
        ds = L1ADDataset(
            split.x,
            split.mask,
            split.l1bit,
            split.y,
            batch_size=self.hparams.batch_size,
            max_batches=self.hparams.max_val_batches,
            shuffler=shuffler,
        )
        ds.object_feature_map = self.loader.object_feature_map
        return DataLoader(ds, batch_size=None, shuffle=False, num_workers=0)


class SyntheticImageADDataModule(LightningDataModule):
    """In-memory image anomaly datamodule for CIFAR/RobustAD smoke tests."""

    def __init__(
        self,
        channels: int = 3,
        image_size: list[int] | tuple[int, int] = (32, 32),
        n_train: int = 128,
        n_val: int = 64,
        n_test: int = 64,
        batch_size: int = 32,
        max_val_batches: int = -1,
        shifted_domains: int = 6,
        n_cifar_signals: int = 9,
        seed: int = 123,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.shuffler = torch.Generator().manual_seed(seed)

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        gen = torch.Generator().manual_seed(self.hparams.seed)

        if stage in (None, "fit"):
            self.train_split = self._make_split(self.hparams.n_train, 0, 0.0, gen)
            self._setup_validation(gen)

        if stage in (None, "validate"):
            self._setup_validation(gen)

        if stage in (None, "test"):
            self._setup_test(gen)

    def train_dataloader(self):
        return self._loader(self.train_split, shuffler=self.shuffler)

    def val_dataloader(self):
        return self._eval_loaders(
            normal=self.val_normal,
            reference=self.val_reference,
            cifar_signals=self.val_cifar_signals,
            shifted_normals=self.val_shifted_normals,
            shifted_normal_all=self.val_shifted_normal_all,
            shifted_anomalies=self.val_shifted_anomalies,
        )

    def test_dataloader(self):
        return self._eval_loaders(
            normal=self.test_normal,
            reference=self.test_reference,
            cifar_signals=self.test_cifar_signals,
            shifted_normals=self.test_shifted_normals,
            shifted_normal_all=self.test_shifted_normal_all,
            shifted_anomalies=self.test_shifted_anomalies,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        x, y = batch
        return x.to(device), y.to(device)

    def _setup_validation(self, gen: torch.Generator):
        self.val_normal = self._make_split(self.hparams.n_val, 0, 0.0, gen)
        self.val_reference = self._make_split(self.hparams.n_val, 0, 0.10, gen)
        self.val_cifar_signals = self._make_cifar_signals(self.hparams.n_val, gen)
        self.val_shifted_normals = self._make_shifted_normals(self.hparams.n_val, gen)
        self.val_shifted_anomalies = self._make_shifted_anomalies(
            self.hparams.n_val, gen
        )
        self.val_shifted_normal_all = self._concat(self.val_shifted_normals)

    def _setup_test(self, gen: torch.Generator):
        self.test_normal = self._make_split(self.hparams.n_test, 0, 0.0, gen)
        self.test_reference = self._make_split(self.hparams.n_test, 0, 0.10, gen)
        self.test_cifar_signals = self._make_cifar_signals(self.hparams.n_test, gen)
        self.test_shifted_normals = self._make_shifted_normals(self.hparams.n_test, gen)
        self.test_shifted_anomalies = self._make_shifted_anomalies(
            self.hparams.n_test, gen
        )
        self.test_shifted_normal_all = self._concat(self.test_shifted_normals)

    def _make_cifar_signals(self, n_samples: int, gen: torch.Generator):
        return {
            str(idx): self._make_split(n_samples, idx, 0.55 + 0.03 * idx, gen)
            for idx in range(1, self.hparams.n_cifar_signals + 1)
        }

    def _make_shifted_normals(self, n_samples: int, gen: torch.Generator):
        return [
            self._make_split(n_samples, -(idx + 1), 0.08 * (idx + 1), gen)
            for idx in range(self.hparams.shifted_domains)
        ]

    def _make_shifted_anomalies(self, n_samples: int, gen: torch.Generator):
        return [
            self._make_split(n_samples, idx + 1, 0.65 + 0.05 * idx, gen)
            for idx in range(self.hparams.shifted_domains)
        ]

    def _make_split(
        self, n_samples: int, label: int, shift: float, gen: torch.Generator
    ) -> _ImageSplit:
        h, w = [int(v) for v in self.hparams.image_size]
        x = 0.25 * torch.randn(
            n_samples, self.hparams.channels, h, w, generator=gen
        )
        x = x + shift

        if label > 0:
            h0, h1 = h // 4, h // 2
            w0, w1 = w // 4, w // 2
            x[:, :, h0:h1, w0:w1] += 0.75

        y = torch.full((n_samples,), label, dtype=torch.long)
        return _ImageSplit(x=x.float(), y=y)

    def _concat(self, splits: list[_ImageSplit]) -> _ImageSplit:
        return _ImageSplit(
            x=torch.cat([split.x for split in splits], dim=0).contiguous(),
            y=torch.cat([split.y for split in splits], dim=0).contiguous(),
        )

    def _eval_loaders(
        self,
        normal: _ImageSplit,
        reference: _ImageSplit,
        cifar_signals: dict[str, _ImageSplit],
        shifted_normals: list[_ImageSplit],
        shifted_normal_all: _ImageSplit,
        shifted_anomalies: list[_ImageSplit],
    ):
        loaders = {
            "normal": self._loader(normal),
            "reference_normal": self._loader(reference),
        }
        for name, split in cifar_signals.items():
            loaders[name] = self._loader(split)
        for idx, split in enumerate(shifted_normals):
            loaders[f"shifted_normal_{idx}"] = self._loader(split)
        loaders["shifted_normal_all"] = self._loader(shifted_normal_all)
        for idx, split in enumerate(shifted_anomalies):
            loaders[f"shifted_anomaly_{idx}"] = self._loader(split)
        return loaders

    def _loader(self, split: _ImageSplit, shuffler: torch.Generator | None = None):
        ds = CIFARADDataset(
            split.x,
            split.y,
            batch_size=self.hparams.batch_size,
            max_batches=self.hparams.max_val_batches,
            shuffler=shuffler,
        )
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
        )
