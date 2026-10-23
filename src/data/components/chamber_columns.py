"""Raw CSV columns of the Causal Chamber evaluation sets, in the order they are served.

``CausalChamberDataModule`` serves only its standardised readouts. ``read_columns``
returns other columns of the same CSVs, such as the actuator settings of the light
tunnel, for the rows of one evaluation set, in the order in which the set's loader
serves their inputs. The row indices are derived again from the datamodule's own split
methods, seeds and fractions.
"""

import numpy as np
import torch
from pyarrow import csv as pa_csv

from src.data.CausalChamber_datamodule import CausalChamberDataModule, _select
from src.data.utils import seeded_permutation


def _read_plain(
    dm: CausalChamberDataModule, spec: dict, columns: list[str]
) -> torch.Tensor:
    """Return the columns of a plain set in file order, (nrows, len(columns))."""
    where = spec.get("where") or {}
    include = list(dict.fromkeys([*columns, *where]))
    options = pa_csv.ConvertOptions(include_columns=include)
    path = dm.dataset_dir / f"{spec['file']}.csv"
    table = _select(pa_csv.read_csv(path, convert_options=options), where)
    values = [table.column(name).to_numpy().astype(np.float64) for name in columns]

    return torch.from_numpy(np.stack(values, axis=1))


def _reference_rows(
    dm: CausalChamberDataModule, name: str, split: str, columns: list[str]
) -> torch.Tensor:
    raw = _read_plain(dm, dm.hparams.reference, columns)
    _, valid, test = dm._reference_indices(raw.size(0))
    # The halving only permutes the rows it is given, so it halves the indices exactly
    # as it halves the inputs.
    main, reference = dm._split_main_and_reference(valid if split == "valid" else test)

    return raw[(main if name == "normal" else reference).x]


def _member_rows(
    dm: CausalChamberDataModule, spec: dict, split: str, columns: list[str]
) -> torch.Tensor:
    """Return the rows of a plain set in a split, in the order of ``_split_set``."""
    raw = _read_plain(dm, spec, columns)
    n = raw.size(0)
    n_valid = round(dm.hparams.signal_val_fraction * n)
    perm = seeded_permutation(n, dm.hparams.seed + n)

    return raw[perm[:n_valid] if split == "valid" else perm[n_valid:]]


def read_columns(
    datamodule: CausalChamberDataModule, name: str, split: str, columns: list[str]
) -> torch.Tensor:
    """Return CSV columns of an evaluation set, row-aligned to the set's loader.

    Row i holds the values, in float64, of the CSV row behind input i of the ``split``
    (``valid`` or ``test``) loader of ``name``: ``normal``, ``reference_normal``, or a
    set of ``signals`` or ``nulls``, a pool being its members' rows concatenated in
    member order. The values are raw, with neither ``relative_to`` nor the
    standardisation applied, and the shape is (nrows, len(columns)). The datamodule
    needs its CSVs in place but need not be set up. Under a ``max_val_batches`` cap,
    every loader but that of ``normal`` serves only the leading rows.
    """
    if split not in ("valid", "test"):
        raise ValueError(f"split must be 'valid' or 'test', not {split!r}.")
    if name in ("normal", "reference_normal"):
        return _reference_rows(datamodule, name, split, columns)
    spec = {**datamodule.hparams.signals, **(datamodule.hparams.nulls or {})}[name]
    members = spec.get("members") or [spec]

    return torch.cat([_member_rows(datamodule, m, split, columns) for m in members])
