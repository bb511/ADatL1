# Load object_feature_map if required/available.

from typing import Any
from pathlib import Path
import json


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
        if getattr(dm, "object_feature_map", None) is not None:
            return dm.object_feature_map
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

def maybe_get_control_object_feature_map(pl_module) -> Any | None:
    """Get the full/control feature map, if a datamodule/dataset exposes one."""
    trainer = getattr(pl_module, "trainer", None)
    if trainer is None:
        return None

    dm = getattr(trainer, "datamodule", None)
    if dm is not None:
        if getattr(dm, "control_object_feature_map", None) is not None:
            return dm.control_object_feature_map

        loader = getattr(dm, "loader", None)
        if loader is not None and hasattr(loader, "control_object_feature_map"):
            return loader.control_object_feature_map

    for attr in ("test_dataloaders", "val_dataloaders", "train_dataloader"):
        dls = getattr(trainer, attr, None)
        dl0 = _first_dataloader(dls)

        if dl0 is None:
            continue

        ds = getattr(dl0, "dataset", None)
        if ds is not None and hasattr(ds, "control_object_feature_map"):
            return ds.control_object_feature_map

        loader = getattr(dl0, "loader", None)
        if loader is not None and hasattr(loader, "control_object_feature_map"):
            return loader.control_object_feature_map

    return None

def inject_object_feature_map(pl_module) -> None:
    """Inject the object_feature_map into the lightning module.

    If no mapping is attached to the trainer/datamodule/dataloaders, try to
    locate a local `object_feature_map.json` in the workspace (preferably under
    an `mlready`/`data` folder) and load it as a fallback.
    """
    ofm = maybe_get_object_feature_map(pl_module)

    # Fallback: search workspace for a cached object_feature_map.json
    if ofm is None:
        try:
            cwd = Path.cwd()
            candidates = list(cwd.rglob("object_feature_map.json"))
            found = None
            # prefer files under mlready or data directories
            for p in candidates:
                sp = str(p)
                if "mlready" in sp or ("/data/" in sp or "\\data\\" in sp):
                    found = p
                    break
            if found is None and candidates:
                found = candidates[0]

            if found is not None:
                with open(found, "r") as f:
                    ofm = json.load(f)
        except Exception:
            ofm = None

    if ofm is None:
        raise RuntimeError("Could not find object_feature_map.")

    # Save directly on the module for anything else that needs it.
    pl_module.object_feature_map = ofm
    pl_module.control_object_feature_map = maybe_get_control_object_feature_map(
        pl_module
    ) or ofm

    reco = getattr(getattr(pl_module, "loss", None), "reco_loss", None)
    if reco is not None and hasattr(reco, "set_object_feature_map"):
        reco.set_object_feature_map(ofm)
