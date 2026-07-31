"""Emit the submit_search.sh command for every driver that died.

A driver is dead if the last line of its log is a traceback/error rather than a
Hydra job line or Hydra's closing best-value report. n_trials is set to the
remaining COMPLETE trials needed, so a relaunch tops the study up instead of
adding another full target on top (the sweeper does not subtract existing
trials: n_trials_to_go = self.n_trials).
"""
import glob
import os
import re
import sqlite3

DEAD = re.compile(r"^(Traceback|[A-Za-z_.]*(Error|Exception)\b|.*\berror:)")


def target_of(script, study):
    if not os.path.exists(script):
        return None
    for block in re.split(r"^# python3 src/train\.py", open(script).read(), flags=re.M)[1:]:
        b = block.replace("\\\n", " ").replace("#", " ")
        s = re.search(r"hydra\.sweeper\.study_name=(\S+)", b)
        n = re.search(r"hydra\.sweeper\.n_trials=(\d+)", b)
        if s and n and s.group(1) == study:
            return int(n.group(1))
    return None


def complete_of(db, study):
    if not os.path.exists(db):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=45)
    con.execute("pragma busy_timeout=45000")
    row = con.execute("select study_id from studies where study_name=?", (study,)).fetchone()
    if row is None:
        con.close()
        return 0
    n = con.execute(
        "select count(*) from trials where study_id=? and state='COMPLETE'", (row[0],)
    ).fetchone()[0]
    con.close()
    return n


rows = []
for log in sorted(glob.glob("logs/searches/*/[0-9]_*.log")):
    with open(log, errors="replace") as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    if not lines or not DEAD.match(lines[-1]):
        continue
    stem = os.path.basename(os.path.dirname(log))        # e.g. physics_runsvdd_q99_search
    domain, script_stem = stem.split("_", 1)             # physics, runsvdd_q99_search
    study = os.path.basename(log)[:-4].split("_", 1)[1]  # 5_consistency_vs_distq99.log
    model = re.match(r"run([a-z0-9]+?)(?:_q99)?_search", script_stem).group(1)
    script = "scripts/%s/%s.sh" % (domain, script_stem)
    db = "logs/optuna/%s/%s.db" % (domain, model)
    tgt, done = target_of(script, study), complete_of(db, study)
    if tgt is None or done is None:
        print("# SKIP %s (target=%s complete=%s)" % (log, tgt, done))
        continue
    rows.append((domain, model, study, script, max(tgt - done, 0), done, tgt, lines[-1][:60]))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
print("# %d dead driver(s)\n" % len(rows))
for domain, model, study, script, todo, done, tgt, err in rows:
    if todo == 0:
        print("# %s/%s %s already at target (%d/%d) - no relaunch needed\n" % (domain, model, study, done, tgt))
        continue
    print("# %-9s %-8s %-24s %3d/%d complete   died: %s" % (domain, model, study, done, tgt, err))
    print("bash scripts/submit_search.sh --only %s %s hydra.sweeper.n_trials=%d" % (study, script, todo))
    print()
