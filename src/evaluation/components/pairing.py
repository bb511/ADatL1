"""The hyperpoint of a dumped run, and the pairing of the runs of two lattices by it
that UDR reads."""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.components.rebuild import ROOT
from src.evaluation.components.tables import normal_scores

log = logging.getLogger(__name__)

SEARCH_DIR = ROOT / "configs" / "hparams_search"


def _override(text: str) -> tuple[str, str]:
    key, _, value = text.partition("=")
    return key.lstrip("+~"), value


def hyperpoint(run_dir: Path, search_dir: Path = SEARCH_DIR) -> dict[str, str]:
    """The sampled hyperparameters of a run, as its ``overrides.yaml`` spells them.

    These are the overrides whose key is a parameter of the sweep that the run's
    ``hparams_search=<name>`` override names, read from ``<search_dir>/<name>.yaml``
    (``hydra.sweeper.params``); a run outside a sweep, or without ``overrides.yaml``,
    has none. The values stay strings as Hydra wrote them, so that the runs of two
    lattices drawn with the same sampler seed pair by plain equality.
    """
    path = Path(run_dir) / "overrides.yaml"
    if not path.exists():
        return {}
    overrides = dict(_override(text) for text in yaml.safe_load(path.read_text()))
    search = overrides.get("hparams_search")
    if search is None:
        return {}
    config = yaml.safe_load((Path(search_dir) / f"{search}.yaml").read_text())
    params = config["hydra"]["sweeper"]["params"]
    return {key: overrides[key] for key in params if key in overrides}


def dump_hyperpoint(run_dir: Path, keys) -> dict[str, str]:
    """The entries of a dump's ``overrides.yaml`` whose key is in ``keys``, as the value
    strings of the command line."""
    overrides = yaml.safe_load((Path(run_dir) / "overrides.yaml").read_text())
    entries = (override.split("=", 1) for override in overrides)
    return {key: value for key, value in entries if key in keys}


def _frozen(hyperpoint: dict) -> str:
    return json.dumps(hyperpoint, sort_keys=True)


def pair_runs(
    hyperpoints: dict[str, dict], partner: dict[str, dict], drop_unpaired: bool = False
) -> dict[str, str]:
    """Map every run to the partner lattice's run of equal hyperpoint.

    Both arguments map run names to hyperpoints. Raise ValueError when two runs of a
    lattice share a hyperpoint or, unless ``drop_unpaired``, when a run has no partner;
    with it, such a run is left out of the map.
    """
    by_point = {_frozen(point): run for run, point in partner.items()}
    own = {_frozen(point) for point in hyperpoints.values()}
    if len(by_point) != len(partner) or len(own) != len(hyperpoints):
        raise ValueError("Two runs of a lattice share a hyperpoint.")
    alone = sorted(r for r, p in hyperpoints.items() if _frozen(p) not in by_point)
    if alone and not drop_unpaired:
        raise ValueError(f"No partner run has the hyperpoint of {alone}.")
    return {r: by_point[_frozen(p)] for r, p in hyperpoints.items() if r not in alone}


def pair_lattice(
    scores_dir: Path,
    lattice: str,
    partner: str,
    names: list[str],
    keys: set[str],
    drop_unpaired: bool = False,
) -> dict[str, str]:
    """``pair_runs`` on the dumps, logging every run left without a partner.

    The runs are ``names`` of the lattice and every run of the partner lattice, each by
    the overrides of its ``overrides.yaml`` whose key is in ``keys``.
    """
    own = {n: dump_hyperpoint(scores_dir / lattice / n, keys) for n in names}
    dirs = sorted(path.parent for path in (scores_dir / partner).glob("*/meta.pt"))
    points = {path.name: dump_hyperpoint(path, keys) for path in dirs}
    pairs = pair_runs(own, points, drop_unpaired)
    alone = sorted(set(own) - set(pairs))
    if alone:
        log.warning(f"No run of {partner} pairs with {alone}, left out of UDR.")
    return pairs


def check_pairing(
    scores_dir: Path, lattice: str, partner: str, drop_unpaired: bool = False
) -> dict[str, str]:
    """``pair_lattice`` before the evaluation, from the dumps alone.

    The runs are the dumps that hold a ``meta.pt``, those the evaluation replays, and
    the keys those of their sweeps, which their run.json will record.
    """
    lattice_dir = Path(scores_dir) / lattice
    names = sorted(path.parent.name for path in lattice_dir.glob("*/meta.pt"))
    keys = set().union(*(hyperpoint(lattice_dir / name) for name in names))
    return pair_lattice(Path(scores_dir), lattice, partner, names, keys, drop_unpaired)


def partner_scores(
    scores_dir: Path,
    lattice: str,
    partner: str | None,
    names: list[str],
    info: list,
    drop_unpaired: bool = False,
) -> list[np.ndarray | None] | None:
    """Stored validation normal scores of each run's partner, aligned with the runs,
    None for a run left without one; None without a partner lattice."""
    if partner is None:
        return None
    keys = set().union(*(run["hyperpoint"] for run in info))
    pairs = pair_lattice(scores_dir, lattice, partner, names, keys, drop_unpaired)
    return [
        normal_scores(scores_dir / partner / pairs[n]) if n in pairs else None
        for n in names
    ]
