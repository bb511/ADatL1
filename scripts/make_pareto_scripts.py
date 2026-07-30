#!/usr/bin/env python3
"""Generate the run<model>_pareto.sh training scripts from the Pareto CSVs.

Reads the per-study CSVs produced by ``scripts/fetch_optuna_pareto.py`` (in
``notebooks/paretos/<domain>/``) and writes, next to the existing
``run<model>_search.sh`` scripts, one ``run<model>_pareto.sh`` per model (plus
``run<model>_q99_pareto.sh`` for the physics q99 studies) containing a full
training command for every point on each strategy's Pareto front.

The command templates mirror the original hand-written run scripts exactly:
same experiment configs, same strategy-specific callback overrides, same
hyperparameter override keys (the Optuna ``algorithm.loss.*`` param names are
remapped to the current config paths, validated against the original scripts'
worked examples). Runs are named ``<strategy>_t<trial>`` and grouped into a
dedicated MLflow experiment ``<domain>_<model>[_q99]_pareto`` so they cannot
collide with the original submission's checkpoints.

Annotations: each command carries its Pareto quantities, and the point that was
picked in the original scripts (``ORIGINAL PICK``), the best primary-objective
point (``BEST <objective>``) and the knee point (``KNEE``, max distance from
the chord between the front's endpoints in normalised objective space) are
marked in the comments.

Fronts up to ``TRIM_THRESHOLD`` (30) points are emitted in full ("retrain all
Pareto trials"). The few near-degenerate fronts above that (some hold 100+
points) are trimmed to a ``CAP_PER_FRONT``-wide knee-centered window, with both
front endpoints (rule-(b) chord), the ``BEST`` point and the ``ORIGINAL PICK``
always retained; the section header records the full front size and the
subsampling, and every point remains in the CSVs.

Layout follows the original run scripts: every command is written out in full
and commented (``taskset`` pinning, GPU devices cycling 0-3); uncomment the
points you want to run locally. Each file additionally ends with a single
commented submit command that sends every point of the file to slurm on
clariden -- one job per point -- via ``scripts/submit_pareto.sh`` (submitit
launcher; it strips the taskset/device pinning and prepends ``-m
hydra/launcher=submitit_slurm_clariden``).

Usage (from anywhere; paths are anchored to the repo root)::

    python scripts/make_pareto_scripts.py            # write all scripts
    python scripts/make_pareto_scripts.py --dry-run  # print summary only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# --------------------------------------------------------------------------- #
# Static knowledge extracted from the original run scripts
# --------------------------------------------------------------------------- #

# Optuna search-space param names -> current config override paths. Validated
# against every worked example in the original run scripts (60/82 exact
# matches; the rest differ because those picks predate the current study dbs).
PARAM_REMAP = {
    "algorithm.loss.delta": "algorithm.delta",
    "algorithm.loss.kl_scale": "algorithm.kl_scale",
    "algorithm.loss.nu": "algorithm.nu",
    "algorithm.loss.soft_boundary": "algorithm.soft_boundary",
    "algorithm.loss.weight_decay": "algorithm.weight_decay",
}

DOMAINS = {
    "physics": {
        "models": ["ae", "vae", "dsae", "dsvae", "svdd", "realnvp", "dte"],
        "ref_ds": "SingleNeutrino_E-10-gun",
        "fixed": ["paths.raw_data_dir=/path/to/adl1t_data/parquet_files"],
        "fixed_per_model": {},
    },
    "cifar10": {
        "models": ["ae", "vae", "svdd", "realnvp", "dte"],
        "ref_ds": "reference_normal",
        "fixed": [],
        "fixed_per_model": {},
    },
    "robustad": {
        "models": ["ae", "vae", "svdd", "realnvp", "dte"],
        "ref_ds": "shifted_normal_all",
        "fixed": [],
        "fixed_per_model": {"ae": ["data.image_size=[128,128]"]},
    },
}

# Strategy templates keyed by the study's first objective. ``nulls`` are the
# training callbacks disabled for that agnostic strategy; ``evals`` are the
# evaluator checkpoint groups deleted (``<REF>`` -> domain reference dataset).
STRATEGIES = {
    "cvar25eff": {"title": "CVAR25", "prefix": "cvar25", "agnostic": False, "extras": []},
    "cvar10eff": {
        "title": "CVAR10",
        "prefix": "cvar10",
        "agnostic": False,
        "extras": ["evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10"],
    },
    "cap": {
        "title": "CAP",
        "prefix": "cap",
        "agnostic": True,
        "nulls": [
            "wasserstein_dist",
            "thres_drift",
            "consistency_sn_zb",
            "wasserstein_dist_ema_ckpt",
            "thres_drift_ema_ckpt",
            "consistency_sn_zb_ema_ckpt",
        ],
        "evals": [
            "operational_drift_ema",
            "w1dist_ema_normal_vs_<REF>",
            "consistency_ema_normal_vs_<REF>",
        ],
    },
    "consistency": {
        "title": "CONSISTENCY",
        "prefix": "consistency",
        "agnostic": True,
        "nulls": [
            "wasserstein_dist",
            "thres_drift",
            "cap_sn_zb",
            "wasserstein_dist_ema_ckpt",
            "thres_drift_ema_ckpt",
            "cap_sn_zb_ema_ckpt",
        ],
        "evals": [
            "operational_drift_ema",
            "w1dist_ema_normal_vs_<REF>",
            "cap_ema_normal_vs_<REF>",
        ],
    },
    "drift": {
        "title": "STABILITY",
        "prefix": "stability",
        "agnostic": True,
        "nulls": [
            "wasserstein_dist",
            "cap_sn_zb",
            "consistency_sn_zb",
            "wasserstein_dist_ema_ckpt",
            "cap_sn_zb_ema_ckpt",
            "consistency_sn_zb_ema_ckpt",
        ],
        "evals": [
            "w1dist_ema_normal_vs_<REF>",
            "cap_ema_normal_vs_<REF>",
            "consistency_ema_normal_vs_<REF>",
        ],
    },
    "wasserstein": {
        "title": "WASSERSTEIN",
        "prefix": "wasserstein",
        "agnostic": True,
        "nulls": [
            "thres_drift",
            "cap_sn_zb",
            "consistency_sn_zb",
            "thres_drift_ema_ckpt",
            "cap_sn_zb_ema_ckpt",
            "consistency_sn_zb_ema_ckpt",
        ],
        "evals": [
            "operational_drift_ema",
            "cap_ema_normal_vs_<REF>",
            "consistency_ema_normal_vs_<REF>",
        ],
    },
}
STRATEGY_ORDER = ["cvar25eff", "cvar10eff", "cap", "consistency", "drift", "wasserstein"]

# The first objective is maximised for these strategies (minimised otherwise);
# the second objective (reconstruction/kl/logp/dist) is always minimised.
MAXIMIZE_OBJ0 = {"cvar25eff", "cvar10eff", "cap", "consistency"}

# Trial picked in the original run scripts per (domain, model, tier, strategy),
# with how it relates to the current study dbs:
#   ok     -- params match the db trial and it is on the current front
#   ok_off -- params match but the trial is no longer Pareto-optimal
# Extracted programmatically from the original scripts before their training
# sections were removed (see runae.sh & co. in git history). Several trial
# numbers in the old scripts were mistyped; they were re-identified by matching
# hyperparameters against the study dbs (see the inline comments). The old
# cvar10 commands reused cvar25-search configurations, so no cvar10-study trial
# is an original pick; physics/ae wasserstein t584 predates the current db and
# is carried verbatim via VERBATIM_FALLBACKS instead.
ORIGINAL_PICKS = {
    ("cifar10", "ae", "250", "cap"): (211, "ok"),
    ("cifar10", "ae", "250", "cvar25eff"): (592, "ok"),
    ("cifar10", "ae", "250", "drift"): (241, "ok"),
    ("cifar10", "ae", "250", "wasserstein"): (279, "ok"),
    ("cifar10", "realnvp", "250", "cap"): (211, "ok"),
    ("cifar10", "realnvp", "250", "cvar25eff"): (535, "ok"),  # was mistyped as t339
    ("cifar10", "realnvp", "250", "drift"): (104, "ok_off"),
    ("cifar10", "realnvp", "250", "wasserstein"): (475, "ok"),
    ("cifar10", "svdd", "250", "cap"): (599, "ok"),
    ("cifar10", "svdd", "250", "cvar25eff"): (284, "ok_off"),
    ("cifar10", "svdd", "250", "drift"): (191, "ok_off"),
    ("cifar10", "svdd", "250", "wasserstein"): (266, "ok"),
    ("cifar10", "vae", "250", "cap"): (520, "ok"),
    ("cifar10", "vae", "250", "cvar25eff"): (568, "ok"),
    ("cifar10", "vae", "250", "drift"): (587, "ok"),
    ("cifar10", "vae", "250", "wasserstein"): (596, "ok"),  # was mistyped as t390
    ("physics", "ae", "250", "cap"): (175, "ok"),
    ("physics", "ae", "250", "cvar25eff"): (169, "ok"),
    ("physics", "ae", "250", "drift"): (564, "ok"),
    ("physics", "ae", "q99", "cap"): (520, "ok"),
    ("physics", "ae", "q99", "cvar25eff"): (335, "ok"),
    ("physics", "ae", "q99", "drift"): (560, "ok"),
    ("physics", "ae", "q99", "wasserstein"): (585, "ok"),
    ("physics", "dsae", "250", "cap"): (570, "ok"),
    ("physics", "dsae", "250", "cvar25eff"): (599, "ok"),
    ("physics", "dsae", "250", "drift"): (565, "ok"),
    ("physics", "dsae", "250", "wasserstein"): (383, "ok"),  # was mistyped as t565
    ("physics", "dsae", "q99", "cap"): (794, "ok"),
    ("physics", "dsae", "q99", "cvar25eff"): (417, "ok"),
    ("physics", "dsae", "q99", "drift"): (362, "ok"),
    ("physics", "dsae", "q99", "wasserstein"): (551, "ok"),
    ("physics", "dsvae", "250", "cap"): (324, "ok"),
    ("physics", "dsvae", "250", "cvar25eff"): (372, "ok"),
    ("physics", "dsvae", "250", "drift"): (445, "ok"),
    ("physics", "dsvae", "250", "wasserstein"): (503, "ok"),
    ("physics", "dsvae", "q99", "cap"): (595, "ok"),
    ("physics", "dsvae", "q99", "cvar25eff"): (192, "ok"),  # see PICK_CAVEATS
    ("physics", "dsvae", "q99", "drift"): (565, "ok"),
    ("physics", "dsvae", "q99", "wasserstein"): (357, "ok"),
    ("physics", "realnvp", "250", "cap"): (376, "ok"),
    ("physics", "realnvp", "250", "cvar25eff"): (523, "ok"),
    ("physics", "realnvp", "250", "drift"): (505, "ok"),
    ("physics", "realnvp", "250", "wasserstein"): (383, "ok"),
    ("physics", "realnvp", "q99", "cap"): (200, "ok"),
    ("physics", "realnvp", "q99", "cvar25eff"): (466, "ok"),
    ("physics", "realnvp", "q99", "drift"): (595, "ok"),
    ("physics", "realnvp", "q99", "wasserstein"): (580, "ok"),
    ("physics", "svdd", "250", "cap"): (536, "ok"),
    ("physics", "svdd", "250", "cvar25eff"): (739, "ok_off"),
    ("physics", "svdd", "250", "drift"): (419, "ok"),
    ("physics", "svdd", "250", "wasserstein"): (545, "ok"),
    ("physics", "svdd", "q99", "cap"): (598, "ok"),
    ("physics", "svdd", "q99", "cvar25eff"): (465, "ok_off"),
    ("physics", "svdd", "q99", "drift"): (587, "ok"),
    ("physics", "svdd", "q99", "wasserstein"): (464, "ok"),
    ("physics", "vae", "250", "cap"): (179, "ok"),
    ("physics", "vae", "250", "cvar25eff"): (345, "ok"),  # see PICK_CAVEATS
    ("physics", "vae", "250", "drift"): (529, "ok"),
    ("physics", "vae", "250", "wasserstein"): (539, "ok"),
    ("physics", "vae", "q99", "cap"): (577, "ok"),
    ("physics", "vae", "q99", "cvar25eff"): (275, "ok"),  # see PICK_CAVEATS
    ("physics", "vae", "q99", "drift"): (567, "ok"),
    ("physics", "vae", "q99", "wasserstein"): (504, "ok"),
    ("robustad", "ae", "250", "cap"): (350, "ok"),
    ("robustad", "ae", "250", "cvar25eff"): (526, "ok"),
    ("robustad", "ae", "250", "drift"): (568, "ok"),
    ("robustad", "ae", "250", "wasserstein"): (299, "ok"),
    ("robustad", "realnvp", "250", "cap"): (174, "ok"),  # was mistyped as t240
    ("robustad", "realnvp", "250", "cvar25eff"): (333, "ok"),
    ("robustad", "realnvp", "250", "drift"): (478, "ok"),  # was mistyped as t390
    ("robustad", "realnvp", "250", "wasserstein"): (591, "ok"),  # was mistyped as t390
    ("robustad", "svdd", "250", "cap"): (546, "ok"),  # was mistyped as t240
    ("robustad", "svdd", "250", "cvar25eff"): (518, "ok"),  # was mistyped as t339
    ("robustad", "svdd", "250", "drift"): (525, "ok"),
    ("robustad", "svdd", "250", "wasserstein"): (581, "ok"),
    ("robustad", "vae", "250", "cap"): (402, "ok"),
    ("robustad", "vae", "250", "cvar25eff"): (62, "ok"),
    ("robustad", "vae", "250", "drift"): (389, "ok"),
    ("robustad", "vae", "250", "wasserstein"): (587, "ok"),
}

# Original picks whose db trial matches on the trial number but differs in
# exactly one hyperparameter (transcription slip or manual tweak in the old
# script); marked as ORIGINAL PICK with this caveat attached.
PICK_CAVEATS = {
    ("physics", "dsvae", "q99", "cvar25eff"):
        "old script had algorithm.encoder.activation=relu; db trial 192 has gelu",
    ("physics", "vae", "250", "cvar25eff"):
        "old script had algorithm.optimizer.lr=5.236233832409967e-05; "
        "db trial 345 has 5.008239492340467e-05",
    ("physics", "vae", "q99", "cvar25eff"):
        "old script had algorithm.encoder.nodes=[64,32,32]; db trial 275 has [64,32,24]",
}

# Handpicked commands with no matching trial in the current study dbs, carried
# verbatim from the original scripts (params exactly as written there).
VERBATIM_FALLBACKS = {
    ("physics", "ae", "250", "wasserstein"): {
        "number": 584,
        "note": (
            "handpicked; predates the current study db -- no matching trial, "
            "command copied verbatim from the original runae.sh"
        ),
        "params": {
            "algorithm.optimizer.lr": "0.00047124714609726086",
            "algorithm.delta": "5.0",
            "trainer.gradient_clip_val": "0.5",
            "algorithm.optimizer.betas": "[0.9,0.99]",
            "algorithm.optimizer.weight_decay": "0.0",
            "algorithm.encoder.nodes": "[64,32,32]",
            "algorithm.input_noise_std": "0.001",
        },
    },
}

N_GPUS = 4
# Fronts up to TRIM_THRESHOLD points are retrained in full; larger
# (near-degenerate) fronts are trimmed to a CAP_PER_FRONT-wide knee-centered
# window with both endpoints, BEST and ORIGINAL PICK always retained.
TRIM_THRESHOLD = 30
CAP_PER_FRONT = 10
BANNER = "# " + 72 * "="
SEP = "# " + 72 * "-"


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def load_front(csv_path: Path) -> list[dict]:
    """Pareto rows of a study CSV, sorted by trial number."""
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_pareto"] == "True"]
    return sorted(rows, key=lambda r: int(r["number"]))


def load_trial(csv_path: Path, number: int) -> dict | None:
    """A single trial's row from a study CSV (Pareto or not)."""
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if int(r["number"]) == number:
                return r
    return None


def quote(value: str) -> str:
    """Single-quote values that contain characters the shell would mangle."""
    return f"'{value}'" if any(c in value for c in "[]{},") else value


def knee_index(points: list[tuple[float, float]], maximize0: bool) -> int | None:
    """Index of the knee: max distance from the chord between the front's
    endpoints, in [0,1]-normalised space with both axes oriented as goodness.

    Returns None for degenerate fronts (<3 distinct points or zero extent).
    """
    if len({p for p in points}) < 3:
        return None
    g = []
    for v0, v1 in points:
        g.append((v0 if maximize0 else -v0, -v1))  # goodness: higher = better
    lo0, hi0 = min(p[0] for p in g), max(p[0] for p in g)
    lo1, hi1 = min(p[1] for p in g), max(p[1] for p in g)
    if hi0 == lo0 or hi1 == lo1:
        return None
    norm = [((p0 - lo0) / (hi0 - lo0), (p1 - lo1) / (hi1 - lo1)) for p0, p1 in g]
    order = sorted(range(len(norm)), key=lambda i: norm[i])
    (x1, y1), (x2, y2) = norm[order[0]], norm[order[-1]]
    chord = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if chord == 0:
        return None
    best_i, best_d = None, -1.0
    for i in order[1:-1]:
        x0, y0 = norm[i]
        d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / chord
        if d > best_d:
            best_i, best_d = i, d
    return best_i


def select_around_knee(
    values: list[tuple[float, float]],
    maximize0: bool,
    forced: set[int],
    cap: int,
    threshold: int,
) -> list[int]:
    """Indices to retrain: the full front, or a knee window if it is too big.

    Fronts up to ``threshold`` points are returned whole. Larger ones are
    trimmed to a ``cap``-wide window centred on the knee (points ordered along
    the front by goodness of the first objective), always keeping both front
    endpoints (they define the rule-(b) chord) and any ``forced`` indices
    (best point, original pick) even when outside the window.
    """
    n = len(values)
    if n <= threshold:
        return list(range(n))
    g0 = [v0 if maximize0 else -v0 for v0, _ in values]
    order = sorted(range(n), key=lambda i: (g0[i], -values[i][1]))
    knee = knee_index(values, maximize0)
    center = order.index(knee) if knee is not None else n // 2
    start = max(0, min(center - cap // 2, n - cap))
    window = set(order[start : start + cap])
    window |= {order[0], order[-1]}
    window |= {i for i in forced if i is not None}
    return sorted(window)


def _command_body(
    domain: str,
    model: str,
    strategy: dict,
    tier: str,
    row: dict,
) -> list[str]:
    """The override tokens shared by both launch variants of one command."""
    dom = DOMAINS[domain]
    exp = f"{domain}/{model}_agnostic" if strategy["agnostic"] else f"{domain}/{model}"
    exp_name = f"{domain}_{model}_pareto" if tier == "250" else f"{domain}_{model}_q99_pareto"
    run_name = f"{strategy['prefix']}_t{row['number']}"

    body = list(dom["fixed"])
    body += [f"experiment={exp}", f"experiment_name={exp_name}", f"run_name={run_name}"]
    if tier == "q99":
        body += ["algorithm.target_rate=0.01", "algorithm.base_rate=null"]
    body += strategy.get("extras", [])
    if strategy["agnostic"]:
        body += [f"callbacks.{cb}=null" for cb in strategy["nulls"]]
        body += [
            f"~evaluation.evaluator.ckpts.summary.{ev.replace('<REF>', dom['ref_ds'])}"
            for ev in strategy["evals"]
        ]
    body += dom["fixed_per_model"].get(model, [])
    for col in sorted(row):
        if col.startswith("params_") and row[col] != "":
            key = PARAM_REMAP.get(col[len("params_"):], col[len("params_"):])
            body.append(f"{key}={quote(row[col])}")
    return body


def _join(lines: list[str], body: list[str]) -> list[str]:
    return lines + [f"    {tok} \\" for tok in body[:-1]] + [f"    {body[-1]}"]


def command_lines(
    domain: str,
    model: str,
    strategy: dict,
    tier: str,
    row: dict,
    device: int,
) -> list[str]:
    """Local run variant (olqti): taskset pinning + cycling GPU devices.

    For slurm submission on clariden, the submit command at the bottom of each
    generated file feeds these same blocks through scripts/submit_pareto.sh,
    which swaps the taskset/device pinning for the submitit launcher.
    """
    body = _command_body(domain, model, strategy, tier, row)
    body += ["trainer=gpu", f"trainer.devices=[{device}]"]
    return _join(
        [f"taskset -c {3 * device}-{3 * device + 2} \\", "python3 src/train.py \\"],
        body,
    )


def pick_note(trial: int, status: str) -> str | None:
    if status in ("ok", "ok_off"):  # ok_off picks are appended to the section
        return None
    on = "on the current front, but" if status == "differ" else "neither on the front, and"
    return (
        f"# NOTE: original pick t{trial} predates the current study db -- trial "
        f"{trial} there is {on} its stored hyperparameters do not match the "
        "original command; not marked below."
    )


def generate_file(
    domain: str, model: str, tier: str, studies: list[tuple[str, Path]], out_path: Path
) -> tuple[int, int]:
    """Write one run<model>[_q99]_pareto.sh; returns (#commands, #studies)."""
    title = f"{model.upper()} PARETO-FRONT TRAINING COMMANDS"
    subtitle = (
        "These are the training commands for every point on the Pareto front of"
        if tier == "250"
        else "q99 background-rate study: training commands for every point on the Pareto front of"
    )
    out = [
        BANNER,
        f"# {title}",
        BANNER,
        f"# {subtitle}",
        "# each validation strategy. Generated by scripts/make_pareto_scripts.py from",
        f"# notebooks/paretos/{domain}/ -- regenerate rather than edit by hand.",
        "#",
        "# Run from the repository root. All commands are commented out -- uncomment",
        "# the points you want to run locally (taskset pinning, GPUs cycling 0-3).",
        "# To run the WHOLE file on clariden instead, use the single submit command",
        "# at the bottom: it sends every point above to slurm, one job each, via",
        "# scripts/submit_pareto.sh (submitit launcher).",
        "",
    ]
    n_cmds = 0
    for study, path in studies:
        obj0, obj1 = study.split("_vs_")
        strategy = STRATEGIES[obj0]
        front = load_front(path)
        values = [(float(r["values_0"]), float(r["values_1"])) for r in front]
        maximize0 = obj0 in MAXIMIZE_OBJ0
        best_v0 = max(v[0] for v in values) if maximize0 else min(v[0] for v in values)
        best_i = next(i for i, v in enumerate(values) if v[0] == best_v0)
        knee_i = knee_index(values, maximize0)
        pick = ORIGINAL_PICKS.get((domain, model, tier, obj0))
        obj1_label = obj1.replace("_b16k", "")

        pick_i = next(
            (
                j
                for j, r in enumerate(front)
                if pick and pick[1] == "ok" and int(r["number"]) == pick[0]
            ),
            None,
        )
        selected = select_around_knee(
            values, maximize0, {best_i, pick_i}, CAP_PER_FRONT, TRIM_THRESHOLD
        )
        trimmed = (
            f", trimmed to {len(selected)} around the knee (endpoints kept)"
            if len(selected) < len(front)
            else ""
        )

        out += [
            BANNER,
            f"# {strategy['title']} TRAINING  "
            f"(study: {study}, {len(front)} Pareto points{trimmed})",
            BANNER,
        ]
        if pick:
            note = pick_note(*pick)
            if note:
                out.append(note)

        for pos, i in enumerate(selected):
            row = front[i]
            marks = []
            if pick and pick[1] == "ok" and int(row["number"]) == pick[0]:
                marks.append("ORIGINAL PICK")
            if i == best_i:
                marks.append(f"BEST {obj0}")
            if knee_i is not None and i == knee_i:
                marks.append("KNEE")
            mark = ("  << " + " | ".join(marks)) if marks else ""
            v0, v1 = values[i]
            hdr = [
                SEP,
                f"# trial {row['number']}: {obj0}={v0:.5g}, {obj1_label}={v1:.5g}{mark}",
            ]
            if "ORIGINAL PICK" in marks:
                caveat = PICK_CAVEATS.get((domain, model, tier, obj0))
                if caveat:
                    hdr.append(f"#   caveat: {caveat}")
            out += hdr + [SEP]
            cmd = command_lines(domain, model, strategy, tier, row, pos % N_GPUS)
            out += [f"# {l}" for l in cmd]
            out.append("")
            n_cmds += 1

        # The handpicked original trial, when it has dropped off the current
        # front, is kept as a fallback in case the front holds bad points.
        if pick and pick[1] == "ok_off":
            row = load_trial(path, pick[0])
            if row is not None:
                v0, v1 = float(row["values_0"]), float(row["values_1"])
                out += [
                    SEP,
                    f"# trial {row['number']}: {obj0}={v0:.5g}, {obj1_label}={v1:.5g}"
                    "  << ORIGINAL PICK (handpicked; no longer on the current front)",
                    SEP,
                ]
                cmd = command_lines(
                    domain, model, strategy, tier, row, len(selected) % N_GPUS
                )
                out += [f"# {l}" for l in cmd]
                out.append("")
                n_cmds += 1

        # Handpicked commands not reproducible from the current dbs, verbatim.
        vb = VERBATIM_FALLBACKS.get((domain, model, tier, obj0))
        if vb is not None:
            row = {"number": str(vb["number"])}
            row.update({f"params_{k}": v for k, v in vb["params"].items()})
            out += [
                SEP,
                f"# trial {vb['number']}: objective values unknown"
                f"  << ORIGINAL PICK ({vb['note']})",
                SEP,
            ]
            cmd = command_lines(domain, model, strategy, tier, row, len(selected) % N_GPUS)
            out += [f"# {l}" for l in cmd]
            out.append("")
            n_cmds += 1

    rel = f"scripts/{domain}/{out_path.name}"
    out += [
        BANNER,
        "# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)",
        BANNER,
        "# Set paths.raw_data_dir to the data location on clariden; any extra",
        "# hydra overrides appended here are added to every job.",
        f"# bash scripts/submit_pareto.sh {rel} \\",
        "#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files",
        "",
    ]

    out_path.write_text("\n".join(out).rstrip("\n") + "\n")
    return n_cmds, len(studies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Print summary only.")
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    paretos = repo / "notebooks" / "paretos"

    total_cmds = 0
    for domain, dom in DOMAINS.items():
        for model in dom["models"]:
            for tier in ("250", "q99"):
                studies = []
                for path in sorted((paretos / domain).glob(f"{model}_*.csv")):
                    if path.name.startswith("ae_cap_exploration_"):
                        continue
                    study = path.stem[len(model) + 1 :]
                    if study.startswith("cap_exploration"):
                        continue
                    obj0, _, obj1 = study.partition("_vs_")
                    if obj0 not in STRATEGIES:
                        continue
                    if ("q99" in obj1) != (tier == "q99"):
                        continue
                    studies.append((study, path))
                if not studies:
                    continue
                studies.sort(key=lambda s: STRATEGY_ORDER.index(s[0].split("_vs_")[0]))
                suffix = "_pareto.sh" if tier == "250" else "_q99_pareto.sh"
                out_path = repo / "scripts" / domain / f"run{model}{suffix}"
                if args.dry_run:
                    n = sum(len(load_front(p)) for _, p in studies)
                    print(f"{out_path.relative_to(repo)}: {len(studies)} studies, {n} commands")
                    total_cmds += n
                else:
                    n, ns = generate_file(domain, model, tier, studies, out_path)
                    print(f"wrote {out_path.relative_to(repo)}: {ns} studies, {n} commands")
                    total_cmds += n
    print(f"\ntotal training commands: {total_cmds}")


if __name__ == "__main__":
    main()
