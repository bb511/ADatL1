"""Turn the published L1 anomaly-detection data into model input.

The files ship raw: original L1Ntuple field names, every event, every object. This
module applies the same steps the paper's pipeline did, each of them optional, and
produces the (N, 39, 3) float32 array its models consume.

    import adl1t_l1ad as l1

    train = l1.read_splits(glob("adl1t-l1ad-v1/zerobias/*/train"))
    norms = l1.fit_norm_params(train)
    x, mask = l1.to_model_tensor(train, norms)       # (N, 39, 3), flatten for (N, 117)

Pass ``apply_cuts=False`` to keep the saturated events and objects the paper dropped,
or call ``apply_saturation_cuts`` yourself with different thresholds.
"""

from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

# Constituents kept per object. FET carries one entry per event; the four object
# collections are clipped to these counts, so the tensor is 1+12+10+4+12 = 39 wide.
NCONST = {"FET": 1, "egammas": 12, "jets": 10, "muons": 4, "taus": 12}

# Features the models see, and the union schema every object is padded up to.
TRAIN_FEATURES = {
    "muons": ["Et", "eta", "phi"],
    "jets": ["Et", "eta", "phi"],
    "egammas": ["Et", "eta", "phi"],
    "taus": ["Et", "eta", "phi"],
    "FET": ["Et", "phi"],
}
SCHEMA = ["Et", "eta", "phi"]

# Raw L1Ntuple names -> the short names used from here on.
FEAT_NAME_MAP = {
    "muons": {"muonIEt": "Et", "muonIPhi": "phi", "muonIEta": "eta"},
    "jets": {"jetIEt": "Et", "jetIPhi": "phi", "jetIEta": "eta"},
    "egammas": {"egIEt": "Et", "egIPhi": "phi", "egIEta": "eta"},
    "taus": {"tauIEt": "Et", "tauIPhi": "phi", "tauIEta": "eta"},
}

# Saturation, not physics: the hardware counters top out here.
EVENT_CUT = ("ET", "Et", 4095)
OBJECT_CUTS = {"muons": 511, "egammas": 511, "jets": 511, "taus": 511, "FET": 4095}


def read_split(split_dir) -> dict:
    """Read one published split directory into {object: awkward array}."""
    split_dir = Path(split_dir)

    return {
        obj.name: ak.from_arrow(pq.read_table(sorted(obj.glob("*.parquet"))))
        for obj in sorted(split_dir.iterdir())
        if obj.is_dir()
    }


def read_splits(split_dirs) -> dict:
    """Read several directories that make up one split, in the study's row order.

    The two zero-bias runs were permuted together, so their training rows interleave.
    `order` is the position across the whole split, which makes the pieces recombinable:
    concatenate, then sort by it. Rows the event cut removed carry -1 and go last.

    Use this whenever a split spans more than one directory; for a single directory it
    is equivalent to :func:`read_split`.
    """
    parts = [read_split(d) for d in split_dirs]
    objects = {
        name: ak.concatenate([part[name] for part in parts]) for name in parts[0]
    }

    order = ak.to_numpy(objects["event_info"]["order"])
    seen = np.flatnonzero(order >= 0)
    seen = seen[np.argsort(order[seen], kind="stable")]
    row_order = np.concatenate([seen, np.flatnonzero(order < 0)])

    return {name: array[row_order] for name, array in objects.items()}


def fit_norm_params(objects: dict) -> dict:
    """Fit the robust normalisation the study used. Fit on the TRAINING split only.

    Median and 5-95 interquantile range per object and feature, over real constituents
    only. To normalise valid or test, fit here on train and pass the result through --
    refitting on the split you are evaluating would leak.
    """
    objects = apply_saturation_cuts(rename_fields(objects))
    params = {}
    for name, features in TRAIN_FEATURES.items():
        if name not in objects:
            continue
        params[name] = {}
        for feature in features:
            flat = ak.to_numpy(ak.flatten(objects[name][feature]))
            low, high = np.quantile(flat, [0.05, 0.95])
            params[name][feature] = {
                "shift": float(np.median(flat)),
                "scale": float(high - low) or 1e-12,
            }

    return params


def rename_fields(objects: dict) -> dict:
    """Map raw L1Ntuple field names onto Et/eta/phi."""
    out = {}
    for name, array in objects.items():
        mapping = FEAT_NAME_MAP.get(name, {})
        out[name] = ak.Array({mapping.get(f, f): array[f] for f in array.fields})

    return out


def apply_saturation_cuts(objects: dict, event_cut: bool = True, object_cut: bool = True) -> dict:
    """Drop ET-saturated events and mask out saturated objects.

    Expects short field names, i.e. call after :func:`rename_fields`. The event cut
    removes whole events; the object cuts only remove the offending object.
    """
    out = dict(objects)
    if event_cut:
        obj, feat, limit = EVENT_CUT
        if obj not in out:
            raise KeyError(f"The {obj} collection is needed for the event cut.")
        keep = ak.all(out[obj][feat] < limit, axis=1)
        out = {name: array[keep] for name, array in out.items()}

    if object_cut:
        for name, limit in OBJECT_CUTS.items():
            if name in out:
                out[name] = out[name][out[name]["Et"] < limit]

    return out


def normalise(objects: dict, norm_params: dict) -> dict:
    """Apply (x - shift) / scale per object and feature."""
    out = {}
    for name, array in objects.items():
        params = norm_params.get(name)
        if params is None:
            out[name] = array
            continue
        out[name] = ak.Array(
            {
                f: (array[f] - params[f]["shift"]) / params[f]["scale"] if f in params else array[f]
                for f in array.fields
            }
        )

    return out


def _pad_object(array: ak.Array, nconst: int) -> tuple:
    """Clip or pad one object to nconst constituents and SCHEMA features."""
    array = ak.Array({f: array[f] for f in array.fields if f in SCHEMA})
    empty = ak.Array([[]] * len(array))
    for feature in SCHEMA:
        if feature not in array.fields:
            array = ak.with_field(array, empty, feature)

    padded = ak.pad_none(array, nconst, axis=-1, clip=True)
    mask = ak.Array({f: ~ak.is_none(padded[f], axis=-1) for f in padded.fields})
    padded = ak.values_astype(ak.fill_none(padded, 0.0), np.float32)

    values = np.empty((len(array), nconst, len(SCHEMA)), dtype=np.float32)
    flags = np.empty((len(array), nconst, len(SCHEMA)), dtype=bool)
    for index, feature in enumerate(sorted(SCHEMA)):
        values[..., index] = ak.to_numpy(padded[feature], allow_missing=False)
        flags[..., index] = ak.to_numpy(mask[feature], allow_missing=False)

    return values, flags


def to_model_tensor(
    objects: dict, norm_params: dict, apply_cuts: bool = True, nconst: dict | None = None
) -> tuple:
    """Build the (N, 39, 3) float32 input and its padding mask.

    :param apply_cuts: Apply the documented saturation cuts, as the paper did.
    :param nconst: Override the per-object constituent counts.
    """
    nconst = nconst or NCONST
    objects = rename_fields(objects)
    if apply_cuts:
        objects = apply_saturation_cuts(objects)
    objects = {k: v for k, v in objects.items() if k in TRAIN_FEATURES}
    objects = {
        k: ak.Array({f: v[f] for f in TRAIN_FEATURES[k]}) for k, v in objects.items()
    }
    objects = normalise(objects, norm_params)

    # Object order follows the pipeline, which sorts by filename: FET before the
    # lowercase collections.
    parts = [_pad_object(objects[name], nconst[name]) for name in sorted(objects)]

    return (
        np.concatenate([v for v, _ in parts], axis=1),
        np.concatenate([m for _, m in parts], axis=1),
    )
