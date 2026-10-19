#!/usr/bin/env python3
"""Score the registered CAP design space on the offline selector's outputs and freeze
one configuration.

Reads ``scripts/cchamber/registration.yaml`` and the ``selection.csv`` and
``metrics.csv`` files written by ``scripts/cchamber/offline_select.py``. Three
commands:

    configs  one YAML file per configuration of ``design_space``, in the format that
             ``offline_select.py --cap-config`` reads; the stem names normalisation,
             mode, energy (with the adaptive scale) and the upper bound B of the beta
             sweep, e.g. ``sigmoid-joint-adaptive0.5-B10``
    table    design_table.csv and design_table.md in ``--out``, from the cap strategy
             over every configuration, the drift and w1 strategies, and the run of
             ``reference_row.py``, all with the declared score ``ascore/full``; the
             frozen configuration goes to ``--frozen`` (default
             ``configs/cap/frozen.yaml``) as a flat ``cap`` config group like
             ``configs/cap/provisional.yaml``, with its SHA-256 beside it in
             ``<stem>.sha256``
    native   native_table.csv: per family with a ``comparison_score``, the declared
             and the native score (``ascore/mse``, ``ascore/kl``) side by side, from
             cap rows of a single configuration, the frozen one

The criterion of a row is the validation efficiency of the best-Q' run of each family
and seed (the ``best`` flag), averaged over the interventions, then the seeds, then the
families with equal weight; AUPRC and the test split are averaged in the same way. A
configuration is ineligible when, for some family and seed, the relative spread of Q'
at the selected checkpoints, (max - min) / max(abs(Q')) over the candidates, is below
1e-6. The standard error is the registered crossed bootstrap, with the same draws for
every row: per replicate, one ``default_rng(seed)`` draws the seeds, then the target
clusters, with replacement; the interventions enter with the multiplicity of their
cluster, the seeds with their own, and every best run stays as selected. It is the
standard deviation (ddof 1) of the replicated criterion, overall and per family. The
frozen configuration is the first eligible one in ``simplicity_order`` whose criterion
is at least the best eligible criterion minus the standard error of that best
configuration.

``--held-out`` marks the inputs of the wind-tunnel test (``held_out_test``): the
clusters file need not hold the registered counts of interventions and target
clusters, and ``table`` scores the CAP configurations that the selection files hold,
by name and without a simplicity rank, in place of the design space; no configuration
is frozen and no frozen file is written.

Usage
-----
    python scripts/cchamber/design_study.py configs --out design/configs
    python scripts/cchamber/design_study.py table \\
        --selection design_out/selection.csv baselines_out/selection.csv \\
            reference_out/selection.csv \\
        --metrics design_out/metrics.csv baselines_out/metrics.csv \\
            reference_out/metrics.csv \\
        --clusters intervention_target_clusters.csv --out design \\
        --frozen configs/cap/frozen.yaml
    python scripts/cchamber/design_study.py table --held-out \\
        --selection wt_frozen/selection.csv wt_baselines/selection.csv \\
            wt_reference/selection.csv \\
        --metrics wt_frozen/metrics.csv wt_baselines/metrics.csv \\
            wt_reference/metrics.csv \\
        --clusters scripts/cchamber/wt_target_clusters.csv --out wt_table
    python scripts/cchamber/design_study.py native \\
        --selection native_*/selection.csv reference_out/selection.csv \\
        --metrics native_*/metrics.csv reference_out/metrics.csv \\
        --clusters intervention_target_clusters.csv --out design
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.callbacks.metrics.cap.metric import ApproximationCapacity  # noqa: E402

HERE = Path(__file__).parent
FROZEN = HERE.parents[1] / "configs" / "cap" / "frozen.yaml"
DECLARED_KEY = "ascore/full"
# Family of the run that reference_row.py dumps, reference_000_s0.
REFERENCE = "reference"
BASELINES = ("drift", "w1")
# The registration's eligibility threshold on the relative spread of Q'.
MIN_SPREAD = 1e-6
# Axes of the design grid, in the registration's order.
GRID = ("normalization_type", "normalization_mode", "energy", "beta_max")
# The keys of configs/cap/provisional.yaml, in its order.
CAP_KEYS = (
    "beta_min",
    "beta_max",
    "n_beta",
    "normalization_type",
    "normalization_params",
    "energy_type",
    "energy_params",
    "regularization_type",
    "regularization_params",
    "binary",
    "process_group",
    "dist_sync_fn",
)
QUANTITIES = ("criterion", "auprc", "efficiency_test", "auprc_test")
COLUMNS = [
    "config",
    "strategy",
    "normalization_type",
    "normalization_mode",
    "energy_type",
    "adaptive_scale",
    "beta_max",
    "simplicity_rank",
    "criterion",
    "se",
    "auprc",
    "efficiency_test",
    "auprc_test",
    "eligible",
    "min_spread",
    "within_one_se",
    "frozen",
]
PER_FAMILY = ("criterion", "se", "auprc")
MARKDOWN = {
    "config": "configuration",
    "criterion": "criterion",
    "se": "SE",
    "auprc": "AUPRC",
    "eligible": "eligible",
    "within_one_se": "within 1 SE",
    "frozen": "frozen",
}
TOP = 20
FROZEN_HEADER = """\
# @package cap

# Frozen CAP configuration of the Causal Chamber design study (subplan 1, E.13).
# registration: {name}, sha256 {sha256}
# configuration: {frozen[config]}, criterion {frozen[criterion]:.6f}
# best eligible: {best[config]}, criterion {best[criterion]:.6f}
# standard error of the best: {best[se]:.6f}
# eligible configurations within one standard error of the best: {n_within}
"""


def _name(c: dict) -> str:
    scale = c["adaptive_scale"]
    energy = c["energy_type"] + ("" if scale is None else str(scale))
    norm = f"{c['normalization_type']}-{c['normalization_mode']}"
    return f"{norm}-{energy}-B{c['beta_max']}"


def _cap_block(fixed: dict, components: dict) -> dict:
    scale = components["adaptive_scale"]
    varying = {
        "beta_max": components["beta_max"],
        "normalization_type": components["normalization_type"],
        "normalization_params": {"mode": components["normalization_mode"]},
        "energy_type": components["energy_type"],
        "energy_params": None if scale is None else {"scale": scale},
    }
    metric = {key: varying.get(key, fixed.get(key)) for key in CAP_KEYS}
    return {"pairing_type": fixed["pairing_type"], "cap_metric_config": metric}


def _entry(fixed: dict, norm: str, mode: str, energy: dict, beta_max: float) -> dict:
    components = {
        "normalization_type": norm,
        "normalization_mode": mode,
        "energy_type": energy["energy_type"],
        "adaptive_scale": energy.get("scale"),
        "beta_max": beta_max,
    }
    cap = _cap_block(fixed, components)
    return {"config": _name(components), **components, "cap": cap}


def _design(space: dict) -> list[dict]:
    """Every configuration of the design space: name, components and CAP block."""
    grid = itertools.product(*(space[axis] for axis in GRID))
    entries = [_entry(space["fixed"], *point) for point in grid]
    if len(entries) != space["n_configurations"]:
        raise ValueError(f"The design space holds {len(entries)} configurations.")
    return entries


def _simplicity(entry: dict, order: dict) -> tuple[int, ...]:
    # Lexicographic in the registered key order. A non-adaptive energy has no scale,
    # so the last key does not rank it.
    return tuple(-1 if entry[k] is None else order[k].index(entry[k]) for k in order)


def write_configs(registration: dict, out: Path) -> None:
    """Write every configuration and check that the metric builds from each file."""
    for entry in _design(registration["design_space"]):
        path = out / f"{entry['config']}.yaml"
        path.write_text(yaml.safe_dump(entry["cap"], sort_keys=False))
        ApproximationCapacity(**yaml.safe_load(path.read_text())["cap_metric_config"])


def _read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as file:
            rows += csv.DictReader(file)
    return rows


def _selection(paths: list[Path]) -> list[dict]:
    rows = _read_rows(paths)
    keys = {(r["config"], r["strategy"], r["score_key"], r["run"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("A run appears twice under one configuration and strategy.")
    return rows


def _metric_index(paths: list[Path]) -> dict[tuple, dict[str, dict]]:
    """Metrics rows by (run, score key, epoch, split), then by dataset.

    Invocations that share a run and epoch repeat its rows with equal values, so the
    last one read is kept.
    """
    index = defaultdict(dict)
    for row in _read_rows(paths):
        head = (row["run"], row["score_key"], int(row["epoch"]), row["split"])
        index[head][row["dataset"]] = row
    return dict(index)


def _clusters(
    path: Path, registration: dict, held_out: bool
) -> tuple[list[str], np.ndarray]:
    """Interventions in file order and the index of each one's target cluster.

    Unless the file belongs to the held-out test, its counts must be the registered ones
    of the light tunnel.
    """
    rows = _read_rows([path])
    names = sorted({row["target_cluster"] for row in rows})
    units = registration["criterion"]["standard_error"]["units"]
    expected = (registration["system"]["interventions"], units["target_clusters"])
    if not held_out and (len(rows), len(names)) != expected:
        raise SystemExit(f"{path}: {len(rows)} interventions in {len(names)} clusters.")
    cluster_of = [names.index(row["target_cluster"]) for row in rows]
    return [row["intervention"] for row in rows], np.array(cluster_of)


def _groups(selection: list[dict], families: list[str]) -> dict[tuple, list[dict]]:
    """Rows of the declared score and the registered families by (config, strategy)."""
    groups = defaultdict(list)
    for row in selection:
        if row["score_key"] == DECLARED_KEY and row["family"] in families:
            groups[(row["config"], row["strategy"])].append(row)
    return groups


def _values(index: dict, unit: dict, interventions: list[str]) -> list[list[float]]:
    """Validation and test efficiency and AUPRC per intervention at a run's selected
    epoch."""
    head = (unit["run"], unit["score_key"], int(unit["selected_epoch"]))
    values = []
    for split in ("val", "test"):
        rows = index[(*head, split)]
        if set(rows) != {"normal", *interventions}:
            raise ValueError(f"{head}: the {split} sets are not the interventions.")
        for quantity in ("efficiency", "auprc"):
            values.append([float(rows[name][quantity]) for name in interventions])
    return values


def _cells(units: list[dict], index: dict, families: list[str], interventions):
    """(quantity, family, seed, intervention) at the best run of each family and seed,
    the quantities in the order of _values."""
    best = {(u["family"], int(u["seed"])): u for u in units if u["best"] == "True"}
    if not best:
        raise SystemExit(f"No best run of {families} among {len(units)} rows.")
    seeds = sorted({seed for _, seed in best})
    cells = [
        [_values(index, best[(f, s)], interventions) for s in seeds] for f in families
    ]
    return np.moveaxis(np.array(cells), 2, 0)


def _summary(cells: np.ndarray, families: list[str]) -> dict:
    """The four quantities over families with equal weight, and the first two per
    family."""
    per_family = cells.mean(axis=(2, 3))
    row = dict(zip(QUANTITIES, per_family.mean(axis=1)))
    for f, family in enumerate(families):
        row[f"criterion_{family}"], row[f"auprc_{family}"] = per_family[:2, f]
    return row


def _relative_spread(q: np.ndarray) -> float:
    scale = np.abs(q).max()
    return 0.0 if scale == 0.0 else (q.max() - q.min()) / scale


def _min_spread(units: list[dict]) -> float:
    """Smallest relative spread of Q' over the candidates of a family and seed."""
    q = defaultdict(list)
    for unit in units:
        q[(unit["family"], unit["seed"])].append(float(unit["q_main"]))
    return float(np.min([_relative_spread(np.array(v)) for v in q.values()]))


def _cap_row(entry: dict, rank: int, data: SimpleNamespace) -> tuple[dict, np.ndarray]:
    units = data.groups.get((entry["config"], "cap"))
    if not units:
        raise SystemExit(f"No cap rows of {entry['config']} in the selection files.")
    cells = _cells(units, data.index, data.families, data.interventions)
    row = {k: v for k, v in entry.items() if k != "cap"}
    row.update(strategy="cap", simplicity_rank=rank, min_spread=_min_spread(units))
    # A NaN spread is not below the threshold, so it leaves the row eligible.
    row.update(eligible=not row["min_spread"] < MIN_SPREAD, frozen=False)
    return {**row, **_summary(cells, data.families)}, cells


def _baseline_row(strategy: str, data: SimpleNamespace) -> tuple[dict, np.ndarray]:
    # Drift and w1 do not depend on the CAP configuration: the rows of any one serve.
    units = next((u for (_, s), u in data.groups.items() if s == strategy), None)
    if units is None:
        raise SystemExit(f"No {strategy} rows in the selection files.")
    cells = _cells(units, data.index, data.families, data.interventions)
    row = {"config": strategy, "strategy": strategy}
    return {**row, **_summary(cells, data.families)}, cells


def _reference_row(data: SimpleNamespace) -> tuple[dict, np.ndarray]:
    units = [u for u in data.selection if u["family"] == REFERENCE]
    units = [u for u in units if u["score_key"] == DECLARED_KEY]
    cells = _cells(units[:1], data.index, [REFERENCE], data.interventions)
    # The one reference run stands for every family.
    cells = np.repeat(cells, len(data.families), axis=1)
    row = {"config": REFERENCE, "strategy": REFERENCE}
    return {**row, **_summary(cells, data.families)}, cells


def _counts(draws: np.ndarray, n: int) -> np.ndarray:
    return (draws[:, :, None] == np.arange(n)).sum(axis=1)


def _draws(se: dict, n_seeds: int, cluster_of: np.ndarray):
    """Multiplicities per replicate of every seed, (replicates, seeds), and of every
    intervention through its cluster, (replicates, interventions)."""
    rng = np.random.default_rng(se["seed"])
    n_clusters = int(cluster_of.max()) + 1
    seeds = rng.integers(n_seeds, size=(se["replicates"], n_seeds))
    clusters = rng.integers(n_clusters, size=(se["replicates"], n_clusters))
    return _counts(seeds, n_seeds), _counts(clusters, n_clusters)[:, cluster_of]


def _replicates(eff: np.ndarray, seed_counts: np.ndarray, weights: np.ndarray):
    """Replicated per-family criterion, (replicates, rows, families), of eff (rows,
    families, seeds, interventions).

    Within a family and seed the interventions are weighted by their multiplicity, and
    within a family the seeds by theirs.
    """
    per_seed = np.einsum("cfsi,ri->rcfs", eff, weights, optimize=True)
    per_seed /= weights.sum(axis=1)[:, None, None, None]
    per_family = np.einsum("rcfs,rs->rcf", per_seed, seed_counts, optimize=True)
    return per_family / seed_counts.sum(axis=1)[:, None, None]


def _attach_errors(pairs: list[tuple], se: dict, data: SimpleNamespace) -> None:
    """Set the standard errors of every row, all from the same draws."""
    shape = pairs[0][1].shape[1:]
    if shape[1] != se["units"]["seeds"]:
        raise SystemExit(f"The best runs hold {shape[1]} seeds.")
    # The reference has one seed; its replicates vary with the clusters only.
    eff = np.stack([np.broadcast_to(cells[0], shape) for _, cells in pairs])
    replicates = _replicates(eff, *_draws(se, shape[1], data.cluster_of))
    errors = replicates.mean(axis=2).std(axis=0, ddof=1)
    by_family = replicates.std(axis=0, ddof=1)
    for (row, _), error, family_errors in zip(pairs, errors, by_family):
        row["se"] = error
        row.update({f"se_{f}": e for f, e in zip(data.families, family_errors)})


def _ranked_design(registration: dict) -> list[tuple[dict, int]]:
    """Every configuration of the design space, in the registration's order, with its
    simplicity rank."""
    design = _design(registration["design_space"])
    order = registration["criterion"]["simplicity_order"]
    ranked = sorted(design, key=lambda entry: _simplicity(entry, order))
    rank = {entry["config"]: i for i, entry in enumerate(ranked)}
    return [(entry, rank[entry["config"]]) for entry in design]


def _held_out_configs(data: SimpleNamespace) -> list[tuple[dict, None]]:
    """The CAP configurations of the selection files by name, which carry no simplicity
    rank."""
    names = sorted({config for config, strategy in data.groups if strategy == "cap"})
    if not names:
        raise SystemExit("No cap rows in the selection files.")
    return [({"config": name}, None) for name in names]


def _scored_rows(
    registration: dict, data: SimpleNamespace, held_out: bool
) -> list[dict]:
    """CAP rows, then drift, w1 and the reference.

    The CAP rows are those of the registered design or, in the held-out test, those of
    the configurations the selection files hold.
    """
    configs = _held_out_configs(data) if held_out else _ranked_design(registration)
    pairs = [_cap_row(entry, rank, data) for entry, rank in configs]
    pairs += [_baseline_row(strategy, data) for strategy in BASELINES]
    pairs.append(_reference_row(data))
    _attach_errors(pairs, registration["criterion"]["standard_error"], data)
    return [row for row, _ in pairs]


def _freeze(rows: list[dict]) -> tuple[dict, dict, int]:
    """Apply the one-standard-error rule to the CAP rows; return the best eligible row,
    the frozen row and the number of eligible rows within one standard error."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise SystemExit("No configuration is eligible.")
    best = max(eligible, key=lambda row: (row["criterion"], -row["simplicity_rank"]))
    for row in rows:
        row["within_one_se"] = bool(row["criterion"] >= best["criterion"] - best["se"])
    within = [row for row in eligible if row["within_one_se"]]
    frozen = min(within, key=lambda row: row["simplicity_rank"])
    frozen["frozen"] = True
    return best, frozen, len(within)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _cell(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _markdown(rows: list[dict]) -> str:
    """The top configurations by criterion, the frozen one, drift, w1 and the
    reference."""
    cap = [row for row in rows if row["strategy"] == "cap"]
    shown = [(i + 1, row) for i, row in enumerate(cap) if i < TOP or row["frozen"]]
    shown += [("", row) for row in rows if row["strategy"] != "cap"]
    lines = ["| rank | " + " | ".join(MARKDOWN.values()) + " |"]
    lines.append("|" + "---|" * (len(MARKDOWN) + 1))
    for rank, row in shown:
        cells = [str(rank), *(_cell(row.get(key)) for key in MARKDOWN)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_frozen(registration: dict, args, summary: dict) -> None:
    entries = _design(registration["design_space"])
    cap = next(e["cap"] for e in entries if e["config"] == summary["frozen"]["config"])
    sha256 = hashlib.sha256(args.registration.read_bytes()).hexdigest()
    header = FROZEN_HEADER.format(name=args.registration.name, sha256=sha256, **summary)
    metric = yaml.safe_dump(cap["cap_metric_config"], sort_keys=False)
    args.frozen.write_text(header + metric)
    digest = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    args.frozen.with_suffix(".sha256").write_text(f"{digest}  {args.frozen.name}\n")


def _by_criterion(row: dict) -> tuple[float, int]:
    # Held-out configurations have no simplicity rank, so a tie keeps their order.
    rank = row["simplicity_rank"]
    return -row["criterion"], -1 if rank is None else rank


def table(registration: dict, data: SimpleNamespace, args) -> None:
    rows = _scored_rows(registration, data, args.held_out)
    cap = sorted((row for row in rows if row["strategy"] == "cap"), key=_by_criterion)
    if not args.held_out:
        best, frozen, n_within = _freeze(cap)
        summary = {"best": best, "frozen": frozen, "n_within": n_within}
        _write_frozen(registration, args, summary)
    rows = cap + [row for row in rows if row["strategy"] != "cap"]
    per_family = [f"{q}_{f}" for f in data.families for q in PER_FAMILY]
    _write_csv(args.out / "design_table.csv", rows, COLUMNS + per_family)
    (args.out / "design_table.md").write_text(_markdown(rows))


def _native_row(family: str, spec: dict, cap: list[dict], data: SimpleNamespace):
    """Criterion and AUPRC of one family under its declared and its native score."""
    row = {"family": family, "declared_score": spec["score"]}
    row["native_score"] = spec["comparison_score"]
    keys = {"declared": DECLARED_KEY, "native": f"ascore/{spec['comparison_score']}"}
    for label, key in keys.items():
        units = [u for u in cap if (u["family"], u["score_key"]) == (family, key)]
        summary = _summary(_cells(units, data.index, [family], data.interventions), [])
        row[f"criterion_{label}"] = summary["criterion"]
        row[f"auprc_{label}"] = summary["auprc"]
    return row


def native(registration: dict, data: SimpleNamespace, out: Path) -> None:
    cap = [u for u in data.selection if u["strategy"] == "cap"]
    configs = sorted({u["config"] for u in cap if u["family"] in data.families})
    if len(configs) != 1:
        raise SystemExit(f"The cap rows hold {len(configs)} configurations: {configs}.")
    specs = registration["families"].items()
    rows = [_native_row(f, s, cap, data) for f, s in specs if "comparison_score" in s]
    reference = _reference_row(data)[0]
    rows.append({"family": REFERENCE, "criterion_declared": reference["criterion"]})
    rows[-1]["auprc_declared"] = reference["auprc"]
    _write_csv(out / "native_table.csv", rows, list(rows[0]))


def _load(args, registration: dict) -> SimpleNamespace:
    """The inputs of table and native."""
    interventions, cluster_of = _clusters(args.clusters, registration, args.held_out)
    families, selection = list(registration["families"]), _selection(args.selection)
    return SimpleNamespace(
        selection=selection,
        index=_metric_index(args.metrics),
        groups=_groups(selection, families),
        families=families,
        interventions=interventions,
        cluster_of=cluster_of,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registration", type=Path, default=HERE / "registration.yaml")
    common.add_argument("--out", type=Path, required=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("configs", parents=[common])
    for name in ("table", "native"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("--selection", nargs="+", type=Path, required=True)
        command.add_argument("--metrics", nargs="+", type=Path, required=True)
        command.add_argument("--clusters", type=Path, required=True)
    commands.choices["native"].add_argument("--held-out", action="store_true")
    output = commands.choices["table"].add_mutually_exclusive_group()
    output.add_argument("--frozen", type=Path, default=FROZEN)
    output.add_argument("--held-out", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    registration = yaml.safe_load(args.registration.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "configs":
        write_configs(registration, args.out)
    elif args.command == "table":
        table(registration, _load(args, registration), args)
    else:
        native(registration, _load(args, registration), args.out)


if __name__ == "__main__":
    main()
