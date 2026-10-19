#!/usr/bin/env python3
"""Run the three limit experiments of the frozen CAP configuration.

Each subcommand runs one experiment of ``registration_limits.yaml`` as registered:

    blind-direction  The 22 signed normalised readouts of the data module, +r and -r,
                     are the candidate scores. CAP between validation normal and
                     reference_normal selects the largest (ties: readout order, then +
                     before -). Per intervention, the selected readout is set against
                     the signed comparator of every affected readout of the readout
                     map: +r for increase, -r for decrease, the inferred direction
                     where the documented one is unknown (block ``inferred``), and none
                     for variance_decrease, saturation and constant_reading (block
                     ``none``). Outcomes are the efficiency at the 1 % threshold of
                     validation normal and the AUPRC against the split's normal, on
                     validation and, descriptively, on test.
    sensitivity      For the CAP-selected model of every family and seed (the ``best``
                     cap rows of ``--selection``) and every osr and t intervention: the
                     response of CAP, threshold drift and Wasserstein, i.e. the
                     criterion with the intervention's validation scores in place of
                     reference_normal minus the criterion with reference_normal, each
                     second set subsampled to 600 rows with seed 4242. A response is
                     signed so that a larger value is a larger departure: CAP is
                     maximised on normal data, the other two are minimised. The shift
                     size of an intervention is the energy distance of its validation
                     inputs from those of reference_normal; the statistic is Spearman's
                     rho of response and shift size over the interventions, per model.
    pairing          Per pairing and family, from ``offline_select.py`` outputs run
                     under each pairing: the registered criterion (mean validation
                     efficiency at 1 % of the CAP-selected models), the mean AUPRC and
                     their standard errors: the registered crossed bootstrap, seeds and
                     the target clusters of ``--clusters`` drawn independently with
                     replacement (2,000 replicates, seed 271828, ddof 1).

CAP and Wasserstein come from their training callbacks, driven as ``offline_select.py``
drives them, with ``torch.manual_seed(--torch-seed)`` before each CAP. Threshold drift
follows ``ThresholdDriftCallback``: the threshold and the weight of the scores tied
with it come from the calibration half of normal, and the exceedances, ties weighted,
are counted on the second set instead of the callback's evaluation half of normal,
which reference_normal or an intervention could not otherwise enter. The frozen
configuration may be in the nested or the flat format that ``offline_select.py``
reads.

Outputs, in ``--out``:

    blind_direction.csv             one row per intervention and affected readout
    blind_direction_summary.csv     per block (documented, inferred, signed = both,
                                    none): interventions whose selected readout is a
                                    comparator, mean gaps (comparator - selected)
    blind_direction_candidates.csv  the CAP of every candidate
    sensitivity_responses.csv       one row per model, intervention and criterion
    sensitivity_<set>.csv           rho per model and criterion, sets primary and
                                    secondary
    sensitivity_<set>_summary.csv   mean rho per family and over all models
    pairing.csv                     one row per pairing and family, and over families

Usage
-----
    python scripts/cchamber/limit_experiments.py blind-direction \\
        --frozen-cap configs/cap/frozen.yaml --data-dir data/causal_chamber \\
        --torch-seed 20261007 --out limits
    python scripts/cchamber/limit_experiments.py sensitivity \\
        --selection frozen/selection.csv --frozen-cap configs/cap/frozen.yaml \\
        --data-dir data/causal_chamber --torch-seed 20261007 --out limits
    python scripts/cchamber/limit_experiments.py pairing frozen nn random \\
        --clusters intervention_target_clusters.csv --out limits
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cchamber import offline_select as offline  # noqa: E402
from scripts.cchamber.make_pairs import load_inputs  # noqa: E402
from scripts.cchamber.ranking_validity import _spearman, pairing_of  # noqa: E402
from src.callbacks.metrics.rate import (  # noqa: E402
    AnomalyRate,
    compute_tie_weight,
    count_exceedances,
)
from src.callbacks.thres_drift import ThresholdDriftCallback  # noqa: E402
from src.data.CausalChamber_datamodule import READOUTS  # noqa: E402

REGISTRATION = ROOT / "scripts" / "cchamber" / "registration_limits.yaml"
# The registered 1 % false-positive rate, the target_rate of the chamber experiments.
META = {"target_rate": 0.01, "base_rate": None}
KEY = "score"
SIGNS = {"increase": "+", "decrease": "-"}
BLOCKS = {
    "documented": ("documented",),
    "inferred": ("inferred",),
    "signed": ("documented", "inferred"),
    "none": ("none",),
}
OUTCOMES = ("efficiency_val", "auprc_val", "efficiency_test", "auprc_test")
SUBSAMPLE, SUBSAMPLE_SEED = 600, 4242
# A departure from normal lowers a maximised criterion and raises a minimised one.
DEPARTURE = {
    s: -1.0 if mode == "max" else 1.0 for s, (_, mode) in offline.MONITORS.items()
}
REPLICATES, BOOTSTRAP_SEED = 2000, 271828


def _replayed(strategy: str, normal, other, cap_cfg: dict, seed: int) -> float:
    """CAP or Wasserstein of normal against other, from its training callback."""
    callback = offline._callbacks([strategy], KEY, cap_cfg)[strategy]
    val = {"normal": normal, "reference_normal": other}
    module, monitor = offline._Module(META), offline.MONITORS[strategy][0]
    return offline._validation_epoch(callback, module, val, KEY, seed, monitor)[0]


def _drift(normal: torch.Tensor, other: torch.Tensor) -> float:
    callback = ThresholdDriftCallback(output_name=KEY, **offline.DRIFT_KWARGS)
    calibration, _ = callback._split_scores(normal)
    fpr = META["target_rate"]
    threshold = callback._compute_threshold(calibration, exceedance_prob=fpr)
    tie_weight = compute_tie_weight(calibration, threshold, fpr)
    p_hat = count_exceedances(other, threshold, tie_weight).item() / len(other)
    eps = 0.5 / len(other)
    return abs(math.log((p_hat + eps) / (fpr + eps)))


def _criteria(normal, other, cap_cfg: dict, seed: int) -> dict[str, float]:
    return {
        "cap": _replayed("cap", normal, other, cap_cfg, seed),
        "drift": _drift(normal, other),
        "w1": _replayed("w1", normal, other, cap_cfg, seed),
    }


def _signed_readouts(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The 22 candidate scores of (n, 11) inputs, in readout order and + before -."""
    return {
        f"{sign}{name}": factor * x[:, j]
        for j, name in enumerate(READOUTS)
        for sign, factor in (("+", 1.0), ("-", -1.0))
    }


def _comparator(entry: dict, readout: str) -> dict:
    """The signed comparator of one affected readout and the block it is reported in."""
    direction = entry["direction"][readout]
    row = {"readout": readout, "direction": direction, "block": "documented"}
    if direction == "unknown":
        direction = entry.get("direction_inferred", {}).get(readout)
        row["block"] = "inferred"
    sign = SIGNS.get(direction)
    if sign is None:
        return {**row, "block": "none", "comparator": ""}
    return {**row, "comparator": f"{sign}{readout}"}


def _outcomes(scores: dict, candidate: str, name: str) -> dict[str, float]:
    """Efficiency at the threshold of validation normal and AUPRC against the split's
    normal, of one candidate score on one intervention."""
    rate = AnomalyRate(META["target_rate"], META["base_rate"])
    rate.set_threshold(scores["val"]["normal"][candidate])
    outcomes = {}
    for split in ("val", "test"):
        normal = scores[split]["normal"][candidate]
        signal = scores[split][name][candidate]
        efficiency = offline._efficiency(rate.threshold, rate.tie_weight, signal, META)
        outcomes[f"efficiency_{split}"] = efficiency
        outcomes[f"auprc_{split}"] = offline._auprc(normal, signal)
    return outcomes


def _blind_row(scores: dict, name: str, selected: str, comparator: dict) -> dict:
    signed = comparator["comparator"]
    chosen = _outcomes(scores, selected, name)
    other = _outcomes(scores, signed, name) if signed else dict.fromkeys(chosen)
    row = {"intervention": name, **comparator, "selected": selected}
    row["selected_is_comparator"] = selected == signed
    row.update({f"{k}_selected": v for k, v in chosen.items()})
    return {**row, **{f"{k}_comparator": v for k, v in other.items()}}


def _mean_gap(rows: list[dict], outcome: str) -> float | None:
    gaps = [r[f"{outcome}_comparator"] - r[f"{outcome}_selected"] for r in rows]
    return float(np.mean(gaps)) if gaps else None


def _block_summary(block: str, rows: list[dict]) -> dict:
    part = [r for r in rows if r["block"] in BLOCKS[block]]
    names = {r["intervention"] for r in part}
    hits = {r["intervention"] for r in part if r["selected_is_comparator"]}
    summary = {"block": block, "rows": len(part), "interventions": len(names)}
    summary["selected_is_comparator"] = len(hits)
    signed = [r for r in part if r["comparator"]]
    return {**summary, **{f"{o}_gap": _mean_gap(signed, o) for o in OUTCOMES}}


def _readout_map(interventions: set[str]) -> dict:
    registration = yaml.safe_load(REGISTRATION.read_text())
    readout_map = yaml.safe_load((ROOT / registration["readout_map"]).read_text())
    if set(readout_map) != interventions:
        raise SystemExit("The readout map and the data module's interventions differ.")
    return readout_map


def _candidate_caps(val: dict, cap_cfg: dict, seed: int) -> dict[str, float]:
    normal, reference = val["normal"], val["reference_normal"]
    return {c: _replayed("cap", normal[c], reference[c], cap_cfg, seed) for c in normal}


def _blind_rows(scores: dict, selected: str, readout_map: dict) -> list[dict]:
    return [
        _blind_row(scores, name, selected, _comparator(entry, readout))
        for name, entry in readout_map.items()
        for readout in entry["affected_readouts"]
    ]


def _blind_direction(args: argparse.Namespace) -> None:
    inputs = load_inputs(args.data_dir)
    scores = {
        s: {n: _signed_readouts(x) for n, x in v.items()} for s, v in inputs.items()
    }
    caps = _candidate_caps(scores["val"], _frozen_cap(args.frozen_cap), args.torch_seed)
    # min keeps the first of equal values, and the candidates are in the tie order.
    selected = min(caps, key=lambda c: offline._badness(caps[c], True))
    readout_map = _readout_map(set(inputs["val"]) - {"normal", "reference_normal"})
    rows = _blind_rows(scores, selected, readout_map)
    _write(args.out, "blind_direction", rows, [_block_summary(b, rows) for b in BLOCKS])
    candidates = [
        dict(candidate=c, cap=v, selected=c == selected) for c, v in caps.items()
    ]
    offline._write_csv(args.out / "blind_direction_candidates.csv", candidates)


def _subsample(scores: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SUBSAMPLE_SEED)
    return scores[torch.randperm(len(scores), generator=generator)[:SUBSAMPLE]]


def _mean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The matrix-product shortcut of cdist leaves rounding residue on the diagonal.
    mode = "donot_use_mm_for_euclid_dist"
    return torch.cdist(x.double(), y.double(), compute_mode=mode).mean()


def _energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """2 E|X - Y| - E|X - X'| - E|Y - Y'| for (n, d) and (m, d) samples.

    The means run over all pairs, diagonals included (the V-statistic), so identical
    sets give exactly zero.
    """
    distances = 2 * _mean_distance(x, y) - _mean_distance(x, x) - _mean_distance(y, y)
    return distances.item()


def _intervention_sets(names: list[str]) -> dict[str, list[str]]:
    """The registered primary set and the secondary set of every osr and t one."""
    registration = yaml.safe_load(REGISTRATION.read_text())
    primary = registration["sensitivity"]["interventions"]["primary"]
    secondary = [n for n in names if n.startswith(("uniform_osr_", "uniform_t_"))]
    if len(secondary) != 27 or not set(primary) <= set(secondary):
        raise SystemExit("The osr and t interventions are not the registered 27.")
    return {"primary": primary, "secondary": secondary}


def _selected_models(path: Path) -> list[dict]:
    """The best rows of strategy cap: the CAP-selected run of every family and seed."""
    rows = [
        r for r in _read_csv(path) if r["strategy"] == "cap" and r["best"] == "True"
    ]
    if len({(r["family"], r["seed"]) for r in rows}) != len(rows):
        raise SystemExit(f"{path} holds more than one CAP configuration.")
    return rows


def _response_rows(head: dict, values: dict, name: str, shift: float) -> list[dict]:
    head = dict(head, intervention=name, shift=shift)
    reference, rows = values["reference_normal"], []
    for c, value in values[name].items():
        row = dict(head, criterion=c, value=value, reference=reference[c])
        rows.append({**row, "response": DEPARTURE[c] * (value - reference[c])})
    return rows


def _model_responses(row: dict, shifts: dict, cap_cfg: dict, seed: int) -> list[dict]:
    """Responses of the three criteria of one selected model to every intervention."""
    epoch = int(row["selected_epoch"])
    path = Path(row["run"]) / f"epoch_{epoch:04d}.pt"
    val = offline._epoch_scores(path, row["score_key"], epoch)["val"]
    values = {
        name: _criteria(val["normal"], _subsample(val[name]), cap_cfg, seed)
        for name in ["reference_normal", *shifts]
    }
    head = {k: row[k] for k in ("family", "candidate", "seed", "selected_epoch")}
    return [r for n, s in shifts.items() for r in _response_rows(head, values, n, s)]


def _columns(rows: list[dict], *names: str) -> list[np.ndarray]:
    return [np.array([row[name] for row in rows]) for name in names]


def _rho_rows(responses: list[dict], names: list[str]) -> list[dict]:
    """Spearman's rho of response and shift size over the named interventions, per model
    and criterion."""
    groups = defaultdict(list)
    for r in responses:
        if r["intervention"] in names:
            groups[r["family"], r["candidate"], r["seed"], r["criterion"]].append(r)
    keys = ("family", "candidate", "seed", "criterion")
    return [
        dict(zip(keys, key), rho=_spearman(*_columns(g, "response", "shift")), n=len(g))
        for key, g in groups.items()
    ]


def _rho_summary(table: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in table:
        for family in (row["family"], "all"):
            groups[family, row["criterion"]].append(row["rho"])
    rows = [
        dict(family=f, criterion=c, mean_rho=float(np.mean(rhos)), models=len(rhos))
        for (f, c), rhos in groups.items()
    ]
    return sorted(
        rows, key=lambda r: (r["family"] == "all", r["family"], r["criterion"])
    )


def _sensitivity(args: argparse.Namespace) -> None:
    val = load_inputs(args.data_dir)["val"]
    sets = _intervention_sets(list(val))
    reference = val["reference_normal"]
    shifts = {n: _energy_distance(val[n], reference) for n in sets["secondary"]}
    cap_cfg, seed = _frozen_cap(args.frozen_cap), args.torch_seed
    models = _selected_models(args.selection)
    responses = [r for m in models for r in _model_responses(m, shifts, cap_cfg, seed)]
    offline._write_csv(args.out / "sensitivity_responses.csv", responses)
    for name, names in sets.items():
        table = _rho_rows(responses, names)
        _write(args.out, f"sensitivity_{name}", table, _rho_summary(table))


def _clusters(path: Path) -> dict[str, int]:
    """Intervention -> index of its target cluster."""
    rows = _read_csv(path)
    names = sorted({r["target_cluster"] for r in rows})
    return {r["intervention"]: names.index(r["target_cluster"]) for r in rows}


def _selected_metrics(out_dir: Path) -> tuple[str, dict]:
    """The pairing of one offline_select.py output and the validation (efficiency,
    AUPRC) of every intervention at each CAP-selected checkpoint."""
    models = _selected_models(out_dir / "selection.csv")
    selected = {(m["run"], m["selected_epoch"]): m for m in models}
    (pairing,) = {pairing_of(m["config"]) for m in models}
    metrics = defaultdict(dict)
    for row in _read_csv(out_dir / "metrics.csv"):
        model = selected.get((row["run"], row["epoch"]))
        if model is not None and row["split"] == "val" and row["dataset"] != "normal":
            pair = (float(row["efficiency"]), float(row["auprc"]))
            metrics[model["family"], int(model["seed"])][row["dataset"]] = pair
    return pairing, metrics


def _matrices(metrics: dict, interventions: list[str]) -> dict[str, np.ndarray]:
    """(seeds, interventions, 2) efficiency and AUPRC per family, seeds ascending."""
    seeds = defaultdict(list)
    for family, seed in sorted(metrics):
        seeds[family].append([metrics[family, seed][n] for n in interventions])
    return {family: np.array(values) for family, values in seeds.items()}


def _counts(draws: np.ndarray, n: int) -> np.ndarray:
    """(replicates, n) multiplicities of (replicates, k) draws from range(n)."""
    return (draws[..., None] == np.arange(n)).sum(axis=1)


def _bootstrap_weights(n_seeds: int, clusters: np.ndarray) -> tuple:
    """Crossed bootstrap multiplicities of seeds and interventions.

    Seeds and target clusters are drawn with replacement, independently; an intervention
    enters with its cluster's multiplicity. Returns (replicates, seeds) and (replicates,
    interventions).
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_clusters = int(clusters.max()) + 1
    seed_draws = rng.integers(0, n_seeds, (REPLICATES, n_seeds))
    cluster_draws = rng.integers(0, n_clusters, (REPLICATES, n_clusters))
    return _counts(seed_draws, n_seeds), _counts(cluster_draws, n_clusters)[:, clusters]


def _replicated(
    matrix: np.ndarray, seed_w: np.ndarray, item_w: np.ndarray
) -> np.ndarray:
    """(replicates, 2) weighted means of a (seeds, interventions, 2) matrix."""
    total = np.einsum("bs,sik,bi->bk", seed_w, matrix, item_w)
    return total / (seed_w.sum(axis=1) * item_w.sum(axis=1))[:, None]


def _pairing_row(pairing: str, family: str, means, replicates) -> dict:
    se = replicates.std(axis=0, ddof=1)
    row = {"pairing": pairing, "family": family}
    row.update(efficiency=means[0], auprc=means[1])
    return {**row, "se_efficiency": se[0], "se_auprc": se[1]}


def _pairing_rows(pairing: str, matrices: dict, clusters: np.ndarray) -> list[dict]:
    """Per family and over families with equal weight; the replicates are shared."""
    n_seeds = len(next(iter(matrices.values())))
    weights = _bootstrap_weights(n_seeds, clusters)
    means = {f: m.mean(axis=(0, 1)) for f, m in matrices.items()}
    replicates = {f: _replicated(m, *weights) for f, m in matrices.items()}
    rows = [_pairing_row(pairing, f, means[f], replicates[f]) for f in matrices]
    overall = [np.mean(list(d.values()), axis=0) for d in (means, replicates)]
    return rows + [_pairing_row(pairing, "all", *overall)]


def _pairing(args: argparse.Namespace) -> None:
    clusters = _clusters(args.clusters)
    indices, rows = np.array(list(clusters.values())), []
    for out_dir in args.outputs:
        pairing, metrics = _selected_metrics(out_dir)
        rows += _pairing_rows(pairing, _matrices(metrics, list(clusters)), indices)
    offline._write_csv(args.out / "pairing.csv", rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _frozen_cap(path: Path) -> dict:
    return offline._cap_config(yaml.safe_load(path.read_text()))


def _write(out: Path, stem: str, rows: list[dict], summary: list[dict]) -> None:
    offline._write_csv(out / f"{stem}.csv", rows)
    offline._write_csv(out / f"{stem}_summary.csv", summary)


def _subparser(commands, name: str, frozen: bool) -> argparse.ArgumentParser:
    parser = commands.add_parser(name)
    parser.add_argument("--out", type=Path, required=True)
    if frozen:
        parser.add_argument("--frozen-cap", type=Path, required=True)
        parser.add_argument("--data-dir", required=True)
        parser.add_argument("--torch-seed", type=int, required=True)
    return parser


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    _subparser(commands, "blind-direction", frozen=True)
    sensitivity = _subparser(commands, "sensitivity", frozen=True)
    sensitivity.add_argument("--selection", type=Path, required=True)
    pairing = _subparser(commands, "pairing", frozen=False)
    pairing.add_argument("outputs", nargs="+", type=Path)
    pairing.add_argument("--clusters", type=Path, required=True)
    return parser.parse_args(argv)


COMMANDS = {
    "blind-direction": _blind_direction,
    "sensitivity": _sensitivity,
    "pairing": _pairing,
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    COMMANDS[args.command](args)
    print(f"{args.command}: {time.monotonic() - start:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
