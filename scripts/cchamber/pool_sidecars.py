#!/usr/bin/env python3
"""Write side-car epoch files that ``offline_select.py --sidecar`` merges into the
records of score dumps.

The dumps of ``src/callbacks/score_dump.py`` hold the scores of the sets that training
scored. Some label-free criteria need scores of other input sets, or values that
compare runs, and this script writes them per run beside the pool, under ``--out``:

    <run_name>/meta.pt          what the epoch files hold: sets and sizes, seeds,
                                moments of the training inputs, SDS parameters
    <run_name>/epoch_<nnnn>.pt  one per epoch of the dump
        epoch
        aux: {name: {"ascore/full": (n,)}}, scores in the order of the set
        values: {name: float}

``run_name`` is the name of the run directory, the parent of ``scores``. Two commands:

    rescore    rebuild each run's module from ``<run>/.hydra/config.yaml``, load the
               weights of every epoch from the dump, and score with the declared
               ``ascore/full`` the sets
                   autouad       N(mu, diag sigma^2) with the moments (ddof 0) of the
                                 training inputs and as many points as validation
                                 normal: AutoUAD's X_gen (Dai and Fan, ICLR 2025,
                                 Def. 6)
                   uniform_100k  100,000 points uniform over the box of the dump's
                                 auxiliary sets, the number of Goix's em_bench.py
                   sds_targets,  the pseudo-sets of self-adaptive data shifting
                   sds_outliers  (label_free_criteria.sds_pseudo_sets) of the
                                 training inputs, the data the detector is trained on
               Each set is drawn once, with seeds fixed here, and shared by every
               run; a set whose name the dump's ``meta.pt`` already lists among its
               auxiliary sets (the autouad set of newer dumps) is not rescored. A
               run is checked first: the bounds of its dump must be those of the
               training inputs, and rescoring the dump's own gaussian set with the
               weights of epoch 0 must reproduce the stored scores within
               ``RTOL``, ``MAX_FRACTION`` and ``RTOL_ANY``, which allow for the
               float32 differences between training on GPU and rescoring on CPU.
               A run that fails is skipped and listed; ``rescore_report.csv`` in
               ``--out`` gains one row per run and call, with the check's largest
               absolute difference, the fraction of points beyond rtol 1e-4, the
               largest relative difference and the time the run took. A run whose
               directory under ``--out`` already holds epoch files is skipped too.
    consensus  read the validation normal scores of every run once and write per
               run and epoch the values udr (against the other seeds of the same
               candidate) and mc (against every other run of the family) of
               ``scripts/analysis/label_free_criteria.py``

Usage
-----
    python scripts/cchamber/pool_sidecars.py rescore pool/*/scores \\
        --data-dir data/causal_chamber --out sidecars/rescore --workers 16
    python scripts/cchamber/pool_sidecars.py consensus pool/*/scores \\
        --out sidecars/consensus --workers 16
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis import label_free_criteria as lfc  # noqa: E402
from scripts.cchamber import offline_select as offline  # noqa: E402
from scripts.cchamber import reference_row  # noqa: E402
from src.callbacks.score_dump import _moments, _save_atomic  # noqa: E402
from src.utils.omegaconf import register_resolvers  # noqa: E402

KEY = "ascore/full"
AUTOUAD_SEED, UNIFORM_SEED = 20261020, 20261021
UNIFORM_SIZE = 100_000
# The box of the dump's uniform set, which must be that of the training inputs.
BOUNDS = ("aux_lower", "aux_upper")
# The self-check. The pool was trained on GPU and is rescored on CPU, in float32 both
# times; the flows score far-out points at 1e8 to 1e18, where a handful of points
# differ by up to 2e-3 relative. A run fails when more than MAX_FRACTION of the points
# lie beyond (RTOL, ATOL), or when any point differs by more than RTOL_ANY relative to
# its stored value: a wrong rebuild moves most points, or some by far more.
RTOL, ATOL, MAX_FRACTION, RTOL_ANY = 1e-4, 1e-5, 0.01, 1e-2
CHECK = ("max_gaussian_difference", "fraction_beyond_rtol", "max_relative_difference")


def _gaussian(moments: dict, size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn((size, *moments["aux_mean"].shape), generator=gen)
    return moments["aux_mean"] + moments["aux_std"] * noise


def _uniform(moments: dict, size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    lower, upper = moments["aux_lower"], moments["aux_upper"]
    return lower + (upper - lower) * torch.rand((size, *lower.shape), generator=gen)


def _description(sets: dict, moments: dict, n_inputs: int) -> dict:
    sds = {"source": "training inputs", "n_inputs": n_inputs}
    sds.update(k=math.ceil(5 * math.log10(n_inputs)), threshold=lfc.SDS_THRESHOLD)
    seeds = {"autouad": AUTOUAD_SEED, "uniform_100k": UNIFORM_SEED}
    sizes = {name: len(x) for name, x in sets.items()}
    return {"key": KEY, "sizes": sizes, "seeds": seeds, **moments, "sds": sds}


def rescore_inputs(x_train: torch.Tensor, n_normal: int) -> tuple[dict, dict]:
    """The sets every run is scored on, (n, d) each, and their description."""
    moments = _moments(x_train)
    targets, outliers = lfc.sds_pseudo_sets(x_train.double().numpy())
    sets = {
        "autouad": _gaussian(moments, n_normal, AUTOUAD_SEED),
        "uniform_100k": _uniform(moments, UNIFORM_SIZE, UNIFORM_SEED),
        "sds_targets": torch.from_numpy(targets).float(),
        "sds_outliers": torch.from_numpy(outliers).float(),
    }
    return sets, _description(sets, moments, len(x_train))


def build_model(run_dir: Path):
    """The run's module in eval mode, built from its Hydra config."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model = instantiate(cfg.algorithm)
    # The VAE builds its KL warm-up schedule at fit start; model_step reads it, while
    # the declared score does not depend on it.
    if hasattr(model, "_setup_kl_annealing") and not hasattr(model, "kl_scale"):
        model._setup_kl_annealing(model.kl_warmup_frac, 1)
    return model.eval()


def score(model, state_dict: dict, x: torch.Tensor) -> torch.Tensor:
    """The declared score of every row of x under the given weights, (n,)."""
    model.load_state_dict(state_dict)
    with torch.no_grad():
        out = model.model_step((x, torch.zeros(len(x), dtype=torch.long)))
    return out[KEY].detach().reshape(-1).float().cpu()


def _gaussian_check(model, meta: dict, run_dir: Path) -> dict:
    """How far the rescored gaussian scores of epoch 0 lie from the stored ones."""
    record = torch.load(run_dir / "epoch_0000.pt", weights_only=True)
    stored = record["aux"]["gaussian"][KEY]
    rescored = score(model, record["state_dict"], meta["aux_inputs"]["gaussian"])
    difference = (rescored - stored).abs()
    relative = torch.where(difference == 0, 0.0, difference / stored.abs())
    beyond = ~torch.isclose(rescored, stored, rtol=RTOL, atol=ATOL)
    check = {"max_gaussian_difference": difference.max().item()}
    check["fraction_beyond_rtol"] = beyond.double().mean().item()
    return {**check, "max_relative_difference": relative.max().item()}


def _problem(model, meta: dict, run_dir: Path, description: dict) -> tuple:
    """Why a run fails its check (None when it passes), and the check's numbers."""
    if not all(torch.equal(meta[name], description[name]) for name in BOUNDS):
        problem = "the dump's input bounds are not those of the training inputs"
        return problem, dict.fromkeys(CHECK, math.nan)
    check = _gaussian_check(model, meta, run_dir)
    failed = check["fraction_beyond_rtol"] > MAX_FRACTION
    if failed or check["max_relative_difference"] > RTOL_ANY:
        return "rescoring does not reproduce the stored gaussian scores", check
    return None, check


def _occupied(out: Path) -> str | None:
    # Two passes are never mixed in one directory, as in the dump itself.
    return f"{out} already holds side-car files" if any(out.glob("epoch_*")) else None


def _new_sets(sets: dict, meta: dict) -> dict:
    # A set the dump scored itself, such as the autouad set of newer dumps, is not
    # rescored: the replay would refuse the repeated name.
    held = set(meta.get("aux", {}).get("names", []))
    return {name: x for name, x in sets.items() if name not in held}


def _write_epochs(model, paths: list[Path], sets: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for epoch, path in enumerate(paths):
        record = torch.load(path, weights_only=True)
        if record["epoch"] != epoch:
            raise RuntimeError(f"{path.parent}: epoch {epoch} is missing.")
        state = record["state_dict"]
        aux = {name: {KEY: score(model, state, x)} for name, x in sets.items()}
        sidecar = {"epoch": epoch, "aux": aux, "values": {}}
        _save_atomic(sidecar, out / f"epoch_{epoch:04d}.pt")


def _write_sidecar(model, paths: list, sets: dict, meta: dict, out: Path) -> None:
    _write_epochs(model, paths, sets, out)
    sizes = {name: len(x) for name, x in sets.items()}
    torch.save({**meta, "sizes": sizes}, out / "meta.pt")


def rescore_run(run_dir: Path, sets: dict, description: dict, root: Path) -> dict:
    """Check one run and write its side-car files; return its report row."""
    start, out = time.monotonic(), root / run_dir.parent.name
    model, paths = build_model(run_dir.parent), sorted(run_dir.glob("epoch_*.pt"))
    meta = torch.load(run_dir / "meta.pt", weights_only=True)
    problem, check = _problem(model, meta, run_dir, description)
    problem = problem or _occupied(out)
    if problem is None:
        new = _new_sets(sets, meta)
        _write_sidecar(model, paths, new, {**description, **check}, out)
    row = {"run": str(run_dir), "problem": problem or "", "epochs": len(paths)}
    return {**row, **check, "seconds": time.monotonic() - start}


def _rescore_task(task: tuple) -> dict:
    return rescore_run(*task)


def _init_worker() -> None:
    torch.set_num_threads(1)
    # The pool's .hydra/config.yaml calls ${smaller_decoder:...} and ${reverse:...}.
    # Registered here, once per process: src/train.py registers the same resolvers
    # on import, and a second registration raises.
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()


def _map(function, tasks: list, workers: int):
    if workers <= 1:
        _init_worker()
        yield from map(function, tasks)
        return

    # Spawned, not forked, as in offline_select.py; in task order, which consensus
    # relies on.
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        yield from pool.imap(function, tasks)


def _append_report(path: Path, rows: list[dict]) -> None:
    # Appended, so that a later pass over skipped runs keeps the earlier rows.
    new = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["run"]))


def rescore(args: argparse.Namespace) -> None:
    datamodule = reference_row._datamodule(args.data_config, args.data_dir)
    n_normal = sum(len(x) for x, _ in datamodule.val_dataloader()["normal"])
    sets, description = rescore_inputs(datamodule.data_train.tensors[0], n_normal)
    tasks = [(Path(run), sets, description, args.out) for run in args.runs]
    rows = list(_map(_rescore_task, tasks, args.workers))
    _append_report(args.out / "rescore_report.csv", rows)
    for row in rows:
        if row["problem"]:
            print(f"skipped {row['run']}: {row['problem']}", file=sys.stderr)


def _normal_scores(run_dir: str) -> torch.Tensor:
    """Validation normal scores of KEY at every epoch of a run, (epochs, n)."""
    paths = sorted(Path(run_dir).glob("epoch_*.pt"))
    epochs = [offline._epoch_scores(p, KEY, e) for e, p in enumerate(paths)]
    return torch.stack([scores["val"]["normal"] for scores in epochs])


def _peers(identity: dict, runs: list[dict]) -> tuple[list[str], list[str]]:
    """The other seeds of a run's candidate, and every other run of its family."""
    family = [r for r in runs if r["family"] == identity["family"]]
    family = [r for r in family if r["run"] != identity["run"]]
    seeds = [r["run"] for r in family if r["candidate"] == identity["candidate"]]
    if not seeds:
        raise SystemExit(f"{identity['run']}: no other seed of its candidate.")
    return seeds, [r["run"] for r in family]


def _epoch_values(scores: dict, run: str, peers: tuple, epoch: int) -> dict:
    own = scores[run][epoch]
    seeds, family = ([scores[r][epoch] for r in group] for group in peers)
    udr = lfc.udr(own, torch.stack(seeds))
    return {"udr": udr, "mc": lfc.model_centrality(own, torch.stack(family))}


def _write_consensus(identity: dict, runs: list[dict], scores: dict, root: Path):
    peers = _peers(identity, runs)
    out = root / Path(identity["run"]).parent.name
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(len(scores[identity["run"]])):
        values = _epoch_values(scores, identity["run"], peers, epoch)
        sidecar = {"epoch": epoch, "aux": {}, "values": values}
        _save_atomic(sidecar, out / f"epoch_{epoch:04d}.pt")
    names = [[Path(run).parent.name for run in group] for group in peers]
    torch.save({"key": KEY, "udr_runs": names[0], "mc_runs": names[1]}, out / "meta.pt")


def consensus(args: argparse.Namespace) -> None:
    runs = [offline._run_identity(run, args.run_pattern) for run in args.runs]
    loaded = _map(_normal_scores, [r["run"] for r in runs], args.workers)
    scores = dict(zip([r["run"] for r in runs], loaded))
    if len({len(s) for s in scores.values()}) != 1:
        raise SystemExit("The runs hold different numbers of epochs.")
    for identity in runs:
        _write_consensus(identity, runs, scores, args.out)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("rescore", "consensus"):
        command = commands.add_parser(name)
        command.add_argument("runs", nargs="+", help="score directories, one per run")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--workers", type=int, default=1)
    commands.choices["rescore"].add_argument("--data-dir", required=True)
    commands.choices["rescore"].add_argument(
        "--data-config", type=Path, default=reference_row.DATA_CONFIG
    )
    commands.choices["consensus"].add_argument(
        "--run-pattern", default=offline.DEFAULT_PATTERN
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "rescore":
        rescore(args)
    else:
        consensus(args)


if __name__ == "__main__":
    main()
