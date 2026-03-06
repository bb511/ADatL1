# Load object_feature_map if required/available.

from typing import Any


def _first_dataloader(dls):
    """Determine what kind of object the first dataloader is."""
    if dls is None:
        return None
    if isinstance(dls, dict):
        return next(iter(dls.values()), None)
    if isinstance(dls, (list, tuple)):
        return dls[0] if len(dls) > 0 else None
    return dls


def maybe_get_object_feature_map(pl_module) -> Any | None:
    """Get the object_feature_map attach to the datamodule.

    If datamodule is not provided, as is the case for the evaluation stage in this
    setup, then look at the dataloders specifically and extract object_feature_map from
    there. The object_feature_map is a dictionary that returns the objects and the
    feature indices in a flattened data array.
    """
    trainer = getattr(pl_module, "trainer", None)
    if trainer is None:
        return None

    # 1) datamodule path
    dm = getattr(trainer, "datamodule", None)
    if dm is not None:
        loader = getattr(dm, "loader", None)
        if loader is not None and hasattr(loader, "object_feature_map"):
            return loader.object_feature_map

    # 2) fallback: any attached dataloaders (val/test/sanity)
    for attr in ("test_dataloaders", "val_dataloaders", "train_dataloader"):
        dls = getattr(trainer, attr, None)
        dl0 = _first_dataloader(dls)
        if dl0 is None:
            continue

        ds = getattr(dl0, "dataset", None)
        if ds is not None and hasattr(ds, "object_feature_map"):
            return ds.object_feature_map

        loader = getattr(dl0, "loader", None)
        if loader is not None and hasattr(loader, "object_feature_map"):
            return loader.object_feature_map

    return None


def inject_object_feature_map(pl_module) -> None:
    """Inject the object_feature_map into the lightning module."""
    ofm = maybe_get_object_feature_map(pl_module)
    if ofm is None:
        raise RuntimeError("Could not find object_feature_map.")

    reco = getattr(getattr(pl_module, "loss", None), "reco_loss", None)
    if reco is not None and hasattr(reco, "set_object_feature_map"):
        reco.set_object_feature_map(ofm)
