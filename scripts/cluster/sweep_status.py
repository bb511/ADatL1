#!/usr/bin/env python3
"""Status of every Optuna sweep on this machine, and the commands to finish it.

For each study: how many COMPLETE trials it has against the n_trials its
run*_search.sh block asks for, and whether a driver is still alive to produce
the rest. Prints a table, then a submit_search.sh command per study that needs
one.

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
"""

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

TAIL = 15
STEM_RE = re.compile(r"run([a-z0-9]+?)(?:_q99)?_search")
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


def blocks(script):
    """study -> (index, n_trials) for each driver block in a run*_search.sh."""
    out = {}
    if not os.path.exists(script):
        return out
    with open(script) as fh:
        text = fh.read()
    i = 0
    for block in re.split(r"^# python3 src/train\.py", text, flags=re.M)[1:]:
        i += 1
        b = block.replace("\\\n", " ").replace("#", " ")
        s = re.search(r"hydra\.sweeper\.study_name=(\S+)", b)
        n = re.search(r"hydra\.sweeper\.n_trials=(\d+)", b)
        if s and n:
            out[s.group(1)] = (i, int(n.group(1)))
    return out


def classify(log):
    if not os.path.exists(log):
        return "no-log", None
    with open(log, errors="replace") as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    tail = "\n".join(lines[-TAIL:])
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(log))).total_seconds() / 60
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
    con = sqlite3.connect("file:%s%s" % (db, mode), uri=True, timeout=60)
    con.execute("pragma busy_timeout=60000")
    row = con.execute("select study_id from studies where study_name=?", (study,)).fetchone()
    n = 0
    if row:
        n = con.execute(
            "select count(*) from trials where study_id=? and state='COMPLETE'", (row[0],)
        ).fetchone()[0]
    con.close()
    return n


def launch_command(row):
    """The submit_search.sh invocation that tops this study up to its target."""
    return ("bash scripts/cluster/submit_search.sh --only %s %s "
            "hydra.sweeper.n_trials=%d"
            % (row["study"], row["script"], row["target"] - row["done"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", action="store_true", help="print only the launch commands")
    ap.add_argument("--db-root", help="read DBs from here instead of logs/optuna (use copies "
                                      "when a crashed driver left a hot -journal)")
    args = ap.parse_args()

    # Anchored on .project-root rather than the CWD: globbing 'scripts/*' from
    # the wrong directory silently yields an empty table instead of an error.
    repo = find_repo_root(Path(__file__).resolve().parent)
    os.chdir(repo)

    rows = []
    for path in sorted(repo.glob("scripts/*/run*_search.sh")):
        script = str(path.relative_to(repo))
        domain = path.parent.name
        stem = path.stem                                          # runsvdd_q99_search
        m = STEM_RE.match(stem)
        if not m:
            print("skipping unrecognised file name: %s" % script, file=sys.stderr)
            continue
        model = m.group(1)
        root = args.db_root or "logs/optuna"
        db = "%s/%s/%s.db" % (root, domain, model)
        # No database on this machine means the sweep was never run here at all -
        # the physics ae/dsae/dsvae/realnvp/vae studies live on olqti and were
        # finished long ago. Reporting them as "launch" would burn a lot of GPU
        # redoing completed work, so they are called out and never given a command.
        has_db = os.path.exists(db)
        for study, (idx, target) in blocks(script).items():
            done = complete_of(db, study, writable=bool(args.db_root)) if has_db else 0
            state, age = classify("logs/searches/%s_%s/%d_%s.log" % (domain, stem, idx, study))
            if not has_db:
                status = "elsewhere"
            elif done >= target:
                status = "DONE"
            elif state == "alive":
                status = "running"
            elif state == "no-log":
                status = "migrated"   # data came from olqti; no driver ever ran here
            elif state == "FINISHED":
                status = "short"      # driver exited cleanly but below target
            else:
                status = "LAUNCH"     # driver crashed
            rows.append(dict(domain=domain, model=model, study=study, script=script,
                             target=target, done=done, status=status, state=state, age=age))

    rows.sort(key=lambda r: (r["domain"], r["model"], r["study"]))

    if not args.commands:
        hdr = "%-9s %-8s %-24s %6s %8s %6s  %-8s %s"
        print(hdr % ("domain", "model", "study", "target", "complete", "todo", "status", "driver"))
        print("-" * 104)
        for r in rows:
            if r["status"] in ("DONE", "elsewhere"):
                continue
            todo = max(r["target"] - r["done"], 0)
            drv = r["state"]
            if r["age"] is not None:
                drv += " (%dm)" % r["age"]
            print(hdr % (r["domain"], r["model"], r["study"], r["target"], r["done"],
                         todo or "", r["status"], drv))
        tally = {}
        for r in rows:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        print("\n" + "  ".join("%s=%d" % kv for kv in sorted(tally.items())))
        print()

    need = [r for r in rows if r["status"] in ("LAUNCH", "short")]
    migrated = [r for r in rows if r["status"] == "migrated"]
    if not need:
        print("# nothing to launch - every study is DONE or has a live driver")
        return 0
    if migrated:
        print("# %d study(ies) carried over from olqti sit below target; "
              "top up only if you want" % len(migrated))
        for r in migrated:
            print("#   %-9s %-8s %-24s %d/%d   (%s)"
                  % (r["domain"], r["model"], r["study"], r["done"], r["target"],
                     launch_command(r)))
        print()
    print("# %d study(ies) to launch  (run from the repository root)" % len(need))
    for r in need:
        print("# %-9s %-8s %-24s %d/%d complete"
              % (r["domain"], r["model"], r["study"], r["done"], r["target"]))
        print(launch_command(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
