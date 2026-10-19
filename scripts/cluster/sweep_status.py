#!/usr/bin/env python3
"""Status of every Optuna sweep on this machine, and the commands to finish it.

For each study: how many COMPLETE trials it has against the n_trials its
run*_search.sh drivers ask for between them, and which of those drivers are
still alive to produce the rest. Prints a table, then a submit_search.sh
command per study that needs one.

    python3 scripts/cluster/sweep_status.py            # table + commands
    python3 scripts/cluster/sweep_status.py --commands # commands only

Driver liveness is read from the driver log, not from ps: clariden rotates
login nodes, so a driver started on another node is invisible to ps here. It is
also not read from the trial counts -- a dead driver leaves its last batch
marked RUNNING forever, so a stalled study looks busy.

A log is classified by scanning its tail:
  * an error signature anywhere in the tail  -> DEAD (the driver exited)
  * Hydra's closing best-value report        -> FINISHED (reached n_trials)
  * anything else                            -> alive
Checking only the final line is not enough: a Hydra config error ends with
'Set the environment variable HYDRA_FULL_ERROR=1 ...', several lines below the
actual exception.

A study below target with no live driver is topped up by relaunching its first
driver for the shortfall. The random sampler replays its draws from the seed,
so the relaunch takes a fresh one: 1000 times the driver's seed plus the
COMPLETE count, which differs from the seed of every earlier top-up that
completed a trial.
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

TAIL = 15
STEM_RE = re.compile(r"run([a-z0-9]+?)_search")
ERR = re.compile(
    r"Traceback \(most recent call last\)"
    r"|HYDRA_FULL_ERROR"
    r"|mismatched input"
    r"|LexerNoViableAltException"
    r"|optuna\.exceptions\."
    r"|^[A-Za-z_.]*(Error|Exception):"
    r"|^\S+\.py: error:",
    re.M,
)
FINISHED = re.compile(r"\[HYDRA\]\s+(Best|Value)", re.M)


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.project-root``."""
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def override(block: str, key: str) -> str | None:
    m = re.search(r"%s=(\S+)" % re.escape(key), block)
    return m.group(1) if m else None


def config_value(hparams: str | None, key: str) -> str | None:
    """A sweeper key of configs/hparams_search/<hparams>.yaml, read as text."""
    path = Path("configs/hparams_search/%s.yaml" % hparams)
    if not hparams or not path.exists():
        return None
    m = re.search(r"^\s*%s:\s*(\S+)" % key, path.read_text(), re.M)
    return m.group(1) if m else None


def setting(block: str, key: str) -> str | None:
    """A sweeper key from the block's override, else from its hparams_search."""
    value = override(block, "hydra.sweeper." + key)
    if value:
        return value.strip("'")
    return config_value(override(block, "hparams_search"), key)


def drivers(script: Path) -> list[dict]:
    """Index, study, storage, n_trials and sampler seed of each driver block.

    The pool blocks leave the study and the storage to their hparams_search config.
    """
    text = script.read_text()
    out = []
    for i, block in enumerate(re.split(r"^# python3 src/train\.py", text, flags=re.M)):
        b = block.replace("\\\n", " ").replace("#", " ")
        study, storage = setting(b, "study_name"), setting(b, "storage")
        n, seed = setting(b, "n_trials"), setting(b, "sampler.seed") or 0
        if i and study and storage and n:
            d = dict(index=i, study=study, storage=storage)
            out.append(d | dict(target=int(n), seed=int(seed)))
    return out


def db_path(storage: str, db_root: str | None) -> str:
    """The sqlite file behind a storage URL, moved under --db-root if given."""
    path = storage.removeprefix("sqlite:///").split("?")[0]
    if db_root:
        path = os.path.join(db_root, os.path.relpath(path, "logs/optuna"))
    return path


def classify(log):
    if not os.path.exists(log):
        return "no-log", None
    with open(log, errors="replace") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    tail = "\n".join(lines[-TAIL:])
    age = (
        datetime.now() - datetime.fromtimestamp(os.path.getmtime(log))
    ).total_seconds() / 60
    if ERR.search(tail):
        return "DEAD", age
    if FINISHED.search(tail):
        return "FINISHED", age
    return "alive", age


def complete_of(db, study, writable=False):
    """COMPLETE trials in a study.

    A driver that crashed mid-transaction leaves a hot '<db>-journal' behind.
    Rolling it back needs a writable connection, so a read-only open of such a
    file fails outright with 'database is locked'. Point --db-root at copies
    (copied together with their -journal siblings) to read those safely without
    touching the live files -- ``writable`` is set for exactly that case.
    """
    if not os.path.exists(db):
        return 0
    mode = "" if writable else "?mode=ro"
    con = sqlite3.connect(f"file:{db}{mode}", uri=True, timeout=60)
    con.execute("pragma busy_timeout=60000")
    row = con.execute(
        "select study_id from studies where study_name=?", (study,)
    ).fetchone()
    n = 0
    if row:
        n = con.execute(
            "select count(*) from trials where study_id=? and state='COMPLETE'",
            (row[0],),
        ).fetchone()[0]
    con.close()
    return n


def grouped_drivers(path: Path) -> dict[str, list[dict]]:
    """The catalogue's drivers grouped by study, each with its log's state."""
    domain, stem = path.parent.name, path.stem
    studies = {}
    for d in drivers(path):
        log = "logs/searches/%s_%s/%d_%s.log" % (domain, stem, d["index"], d["study"])
        d["state"], d["age"] = classify(log)
        studies.setdefault(d["study"], []).append(d)
    return studies


def study_rows(path: Path, db_root: str | None) -> list[dict]:
    """One row per study of a catalogue, with its drivers' n_trials summed."""
    domain, model = path.parent.name, STEM_RE.match(path.stem).group(1)
    rows = []
    for study, ds in grouped_drivers(path).items():
        db = db_path(ds[0]["storage"], db_root)
        row = dict(domain=domain, model=model, study=study, drivers=ds)
        row["script"] = f"scripts/{domain}/{path.stem}.sh"
        row["target"] = sum(d["target"] for d in ds)
        row["done"] = complete_of(db, study, writable=bool(db_root))
        rows.append(row)
    return rows


def status(row: dict) -> str:
    states = [d["state"] for d in row["drivers"]]
    if row["done"] >= row["target"]:
        return "DONE"
    if "alive" in states:
        return "running"
    if all(s == "no-log" for s in states):
        return "new"  # no driver of this study was launched on this machine
    return "short"  # every driver exited, cleanly or not, below target


def launch_command(row: dict) -> str:
    """The submit_search.sh invocation that tops this study up to its target."""
    first = row["drivers"][0]
    return (
        "bash scripts/cluster/submit_search.sh --only %d %s "
        "hydra.sweeper.n_trials=%d hydra.sweeper.sampler.seed=%d"
        % (
            first["index"],
            row["script"],
            row["target"] - row["done"],
            1000 * first["seed"] + row["done"],
        )
    )


def collect(repo: Path, db_root: str | None) -> list[dict]:
    rows = []
    for path in sorted(repo.glob("scripts/*/run*_search.sh")):
        if not STEM_RE.match(path.stem):
            print("skipping unrecognised file name: %s" % path, file=sys.stderr)
            continue
        rows += study_rows(path, db_root)
    for r in rows:
        r["status"] = status(r)
    return sorted(rows, key=lambda r: (r["domain"], r["model"], r["study"]))


def driver_states(row: dict) -> str:
    """'alive(12m) DEAD(3m) ...', one entry per driver in block order."""
    return " ".join(
        d["state"] + ("" if d["age"] is None else "(%dm)" % d["age"])
        for d in row["drivers"]
    )


def print_table(rows: list[dict]) -> None:
    hdr = "%-9s %-8s %-10s %6s %8s %6s  %-8s %s"
    print(hdr % tuple("domain model study target complete todo status drivers".split()))
    print("-" * 104)
    for r in rows:
        if r["status"] != "DONE":
            todo = max(r["target"] - r["done"], 0) or ""
            cols = (r["domain"], r["model"], r["study"], r["target"], r["done"], todo)
            print(hdr % (*cols, r["status"], driver_states(r)))
    tally = Counter(r["status"] for r in rows)
    print("\n" + "  ".join("%s=%d" % kv for kv in sorted(tally.items())) + "\n")


def print_new(rows: list[dict]) -> None:
    # A study absent here may be running on another machine, so its launch is
    # printed commented out rather than offered as a command to paste.
    new = [r for r in rows if r["status"] == "new"]
    if new:
        print("# %d study(ies) never launched on this machine:" % len(new))
        for r in new:
            print("#   bash scripts/cluster/submit_search.sh %s" % r["script"])
        print()


def print_topups(rows: list[dict]) -> None:
    need = [r for r in rows if r["status"] == "short"]
    if not need:
        print("# nothing to top up - every study is DONE, running or new")
        return
    print("# %d study(ies) to top up  (run from the repository root)" % len(need))
    for r in need:
        print(
            "# %-9s %-8s %-10s %d/%d complete"
            % (r["domain"], r["model"], r["study"], r["done"], r["target"])
        )
        print(launch_command(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--commands", action="store_true", help="print only the launch commands"
    )
    ap.add_argument(
        "--db-root",
        help="read DBs from here instead of logs/optuna (use copies "
        "when a crashed driver left a hot -journal)",
    )
    args = ap.parse_args()

    # Anchored on .project-root rather than the CWD: globbing 'scripts/*' from
    # the wrong directory silently yields an empty table instead of an error.
    repo = find_repo_root(Path(__file__).resolve().parent)
    os.chdir(repo)
    rows = collect(repo, args.db_root)
    if not args.commands:
        print_table(rows)
    print_new(rows)
    print_topups(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
