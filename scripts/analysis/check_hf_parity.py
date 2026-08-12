# Checks that the two physics pipelines end at the same tensors.
#
# One reads the raw parquet files with this project's own stages, the other runs the
# loader that ships inside the published record. Each side is surveyed in turn and kept
# only as digests, so the two sets of tensors never sit in memory together: an equal
# digest over shape, dtype and bytes means the tensors are identical.
import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Sets PROJECT_ROOT, which configs/paths/default.yaml reads and both pipelines resolve
# on their way to a cache directory. src/train.py bootstraps itself the same way.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def report(label: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}", flush=True)

    return ok


def announce(message: str) -> None:
    print(f"== {message}", flush=True)


def digest(tensor: torch.Tensor) -> str:
    """One tensor's shape, dtype and bytes, hashed without copying it."""
    array = tensor.detach().cpu().contiguous().numpy()
    hashed = hashlib.sha256(f"{array.shape}|{array.dtype}|".encode())
    hashed.update(memoryview(array.reshape(-1)).cast("B"))

    return hashed.hexdigest()


def loader_digests(loader) -> dict:
    """What one dataloader hands a model, as digests plus the label it carries."""
    dataset = loader.dataset

    return {
        "x": digest(dataset.data),
        "mask": digest(dataset.mask),
        "l1bit": digest(dataset.l1bit),
        "y": digest(dataset.labels),
        "events": int(dataset.n),
        "label": int(dataset.labels.flatten()[0]),
    }


def mlready_folder(datamodule) -> Path:
    """Where a pipeline wrote the normalised splits it loads from."""
    if getattr(datamodule, "main_cache_folder", None) is not None:
        return Path(datamodule.main_cache_folder)

    return Path(datamodule.record.data_mlready.cache_folder)


def norm_params(datamodule) -> dict:
    """The normalisation parameters each pipeline fitted, keyed by object."""
    folder = mlready_folder(datamodule)
    params = {}
    for path in sorted(folder.glob("*_norm_params.pkl")):
        with path.open("rb") as handle:
            params[path.stem] = pickle.load(handle)

    return params


def survey(datamodule) -> dict:
    """Everything one pipeline ends with: its tensors, its keys and its feature map."""
    datamodule.prepare_data()

    datamodule.setup("fit")
    surveyed = {"train": loader_digests(datamodule.train_dataloader())}
    surveyed["valid"] = _split_digests(datamodule.val_dataloader())
    surveyed["feature_map"] = datamodule.loader.object_feature_map
    surveyed["norm_params"] = norm_params(datamodule)
    datamodule.teardown("fit")

    datamodule.setup("test")
    surveyed["test"] = _split_digests(datamodule.test_dataloader())
    datamodule.teardown("test")

    return surveyed


def _split_digests(loaders: dict) -> dict:
    return {name: loader_digests(loader) for name, loader in loaders.items()}


def _same_value(a, b) -> bool:
    """Equality over the nested dictionaries of arrays the normalisers pickle."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)

    return np.array_equal(np.asarray(a), np.asarray(b))


def compare(raw: dict, mirrored: dict) -> bool:
    """Report every difference between the two pipelines' final outputs."""
    ok = report("train", raw["train"] == mirrored["train"])
    for split in ("valid", "test"):
        ok &= report(f"{split} data set order", list(raw[split]) == list(mirrored[split]))
        ok &= report(f"{split} normal first", next(iter(mirrored[split]), None) == "normal")
        for name in raw[split]:
            ok &= report(f"{split}/{name}", raw[split][name] == mirrored[split].get(name))

    ok &= report("object feature map", raw["feature_map"] == mirrored["feature_map"])
    ok &= report("normalisation parameters", _same_value(raw["norm_params"], mirrored["norm_params"]))

    return ok


def datamodule_from(overrides: list) -> object:
    """Build one pipeline's datamodule from this project's configuration tree."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    return instantiate(cfg.data)


def check_physics(raw_data_dir: str, raw_cache: str | None, hf_cache: str | None) -> bool:
    """Survey each pipeline in turn, comparing what a model would be handed."""
    announce("surveying the on-disk pipeline")
    raw_overrides = ["data=basis", f"paths.raw_data_dir={raw_data_dir}"]
    if raw_cache:
        raw_overrides.append(f"paths.data_dir={raw_cache}")
    raw = survey(datamodule_from(raw_overrides))

    announce("surveying the published record's pipeline")
    overrides = ["data=basis_hf"]
    if hf_cache:
        overrides.append(f"data.cache_dir={hf_cache}")
    mirrored = survey(datamodule_from(overrides))

    announce("comparing")

    return compare(raw, mirrored)


def same_labelled_images(first: tuple, second: tuple) -> bool:
    """Whether two sets hold the same pictures under the same labels, in whatever order.

    Each side is (images, labels), compared as pairs rather than as two independent
    sets: CIFAR-10 is class balanced, so a label column matches on its own whatever
    picture it is attached to, and a misalignment would go unnoticed.
    """
    pairs = [
        sorted(zip(map(bytes, images.reshape(len(images), -1)), labels))
        for images, labels in (first, second)
    ]

    return pairs[0] == pairs[1]


def check_cifar(data_dir: str) -> bool:
    """The hub and torchvision must hold the same images.

    Their test splits agree row by row. The hub publishes the training images in the
    order of the archive it was built from, which is not the order torchvision
    concatenates the batches in, so that split is compared as a set: the seeded normal
    train and validation partition drawn over positions differs between the two.
    """
    from datasets import load_dataset
    from torchvision.datasets import CIFAR10

    ok = True
    for split in ("train", "test"):
        original = CIFAR10(root=data_dir, train=split == "train", download=False)
        hub = load_dataset("uoft-cs/cifar10", split=split).with_format("numpy")
        images = np.stack(hub["img"])
        labels = [int(label) for label in hub["label"]]
        targets = [int(label) for label in original.targets]
        ok &= report(
            f"cifar {split} labelled images",
            same_labelled_images((images, labels), (original.data, targets)),
        )
        if split == "test":
            ok &= report("cifar test row order", np.array_equal(images, original.data))
            ok &= report("cifar test label order", labels == targets)

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["physics", "cifar"])
    parser.add_argument("--raw-data-dir", help="the raw parquet files the on-disk pipeline reads")
    parser.add_argument("--raw-cache", help="where the on-disk pipeline's own caches are built")
    parser.add_argument("--hf-cache", help="where the record's own caches are built")
    parser.add_argument("--cifar-dir", default="data/cifar10")
    args = parser.parse_args()

    register_resolvers()
    if args.check == "cifar":
        ok = check_cifar(args.cifar_dir)
    elif not args.raw_data_dir:
        raise SystemExit("physics needs --raw-data-dir, the raw parquet files.")
    else:
        ok = check_physics(args.raw_data_dir, args.raw_cache, args.hf_cache)

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
