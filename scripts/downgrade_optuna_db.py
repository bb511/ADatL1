#!/usr/bin/env python3
"""Downgrade an Optuna sqlite database from the 3.2 schema back to 2.10.1's.

Why this exists
---------------
This repository pins ``optuna==2.10.1`` because ``hydra-optuna-sweeper==1.2.0``
requires ``optuna>=2.10,<3``. Reading those databases with a modern Optuna (the
``optuna-ui`` env, 4.7.0) requires ``optuna storage upgrade``, which rewrites the
schema in place from ``v2.6.0.a`` to ``v3.2.0.a``. That upgrade is one-way as far
as Optuna's own tooling is concerned -- there is no ``storage downgrade`` -- so
an upgraded database can no longer be written by the sweeper, and a study can no
longer be added to it.

Alembic can still walk the graph backwards, though: every revision between the
two heads ships a real ``downgrade()``, and ``v3.0.0.a``'s calls
``restore_old_distribution()``, which rewrites ``FloatDistribution`` /
``IntDistribution`` back into the ``Uniform`` / ``LogUniform`` / ``IntUniform``
JSON that 2.10.1 parses. So the migration is fully reversible in practice.

    v3.2.0.a -> v3.0.0.d -> v3.0.0.c -> v3.0.0.b -> v3.0.0.a -> v2.6.0.a

Run this with a **modern** Optuna interpreter (>= 3.2): it is the one that ships
the migration scripts that know how to go backwards. The source file is copied
first and is never opened for writing.

Usage
-----
    # on olqti, with the optuna-ui env (4.7.0)
    python3 scripts/downgrade_optuna_db.py <src.db> <dst.db>

Then verify with the pinned env, which should report identical digests::

    python3 scripts/downgrade_optuna_db.py --digest <dst.db>   # optuna 2.10.1
    python3 scripts/downgrade_optuna_db.py --digest <src.db>   # optuna 4.7.0

``--digest`` normalises the distribution classes across Optuna versions, so the
hashes are directly comparable; matching hashes mean no trial, parameter, value,
or direction changed.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys

import optuna

TARGET = "v2.6.0.a"  # optuna 2.10.1's alembic head


# --------------------------------------------------------------------------- #
# digest: version-independent fingerprint of a database's contents
# --------------------------------------------------------------------------- #
def _norm_dist(d):
    """Normalise a distribution to a version-independent tuple."""
    n = type(d).__name__
    if n == "CategoricalDistribution":
        return ["cat", [repr(c) for c in d.choices]]
    if n == "FloatDistribution":
        return ["float", d.low, d.high, bool(d.log), d.step]
    if n == "IntDistribution":
        return ["int", d.low, d.high, bool(d.log), d.step]
    # optuna 2.x spellings
    if n == "UniformDistribution":
        return ["float", d.low, d.high, False, None]
    if n == "LogUniformDistribution":
        return ["float", d.low, d.high, True, None]
    if n == "DiscreteUniformDistribution":
        return ["float", d.low, d.high, False, d.q]
    if n == "IntUniformDistribution":
        return ["int", d.low, d.high, False, d.step]
    if n == "IntLogUniformDistribution":
        return ["int", d.low, d.high, True, d.step]
    raise SystemExit("unhandled distribution: " + n)


def digest(path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    url = "sqlite:///" + path
    out = []
    for summary in optuna.study.get_all_study_summaries(storage=url):
        study = optuna.load_study(study_name=summary.study_name, storage=url)
        try:
            directions = [d.name for d in study.directions]
        except AttributeError:  # single-objective study
            directions = [study.direction.name]
        rows = []
        for t in study.get_trials(deepcopy=False):
            rows.append(
                [
                    t.number,
                    t.state.name,
                    t.values,
                    sorted((k, repr(v)) for k, v in t.params.items()),
                    sorted((k, _norm_dist(v)) for k, v in t.distributions.items()),
                ]
            )
        rows.sort(key=lambda r: r[0])
        blob = json.dumps(rows, sort_keys=True, default=str).encode()
        out.append(
            {
                "study": summary.study_name,
                "directions": directions,
                "n_trials": len(rows),
                "sha256": hashlib.sha256(blob).hexdigest()[:16],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# downgrade
# --------------------------------------------------------------------------- #
def _patch_stale_enum_names(script):
    """Work around a broken ``downgrade()`` in Optuna's v3.0.0.c migration.

    Its final statement is::

        sa.Enum(IntermediateValueModel.FloatTypeEnum).drop(bind, checkfirst=True)

    but that migration's own ``IntermediateValueModel`` names the enum
    ``TrialIntermediateValueType``. As shipped it therefore raises
    ``AttributeError`` *after* already dropping the column, leaving the database
    stranded between revisions. The class is local to the migration module, so
    the alias has to be attached to the module Alembic actually loaded.

    Dropping a ``sa.Enum`` is a no-op on SQLite -- it has no native ENUM type, so
    SQLAlchemy renders VARCHAR + CHECK -- which means this only lets the step run
    to completion. The resulting schema and data are unaffected.
    """
    patched = []
    for rev in script.walk_revisions():
        model = getattr(getattr(rev, "module", None), "IntermediateValueModel", None)
        enum_cls = getattr(model, "TrialIntermediateValueType", None)
        if model is not None and enum_cls is not None and not hasattr(model, "FloatTypeEnum"):
            model.FloatTypeEnum = enum_cls
            patched.append(rev.revision)
    return patched


def _counts(path):
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    version = con.execute("select version_num from alembic_version").fetchone()[0]
    studies = con.execute("select study_id, study_name from studies").fetchall()
    per_study = {
        name: con.execute(
            "select count(*) from trials where study_id=?", (sid,)
        ).fetchone()[0]
        for sid, name in studies
    }
    params = con.execute("select count(*) from trial_params").fetchone()[0]
    values = con.execute("select count(*) from trial_values").fetchone()[0]
    con.close()
    return version, per_study, params, values


def downgrade(src, dst):
    from alembic.runtime.environment import EnvironmentContext
    from alembic.script import ScriptDirectory

    if os.path.exists(dst):
        sys.exit("refusing to overwrite existing %s" % dst)
    if os.path.dirname(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    print("copying %s -> %s" % (src, dst))
    shutil.copyfile(src, dst)

    before = _counts(dst)
    if before[0] == TARGET:
        print("already at %s, nothing to do" % TARGET)
        return 0
    print("before: alembic=%s  trial_params=%d  trial_values=%d" % (before[0], before[2], before[3]))
    for name, n in before[1].items():
        print("        %-32s %d trials" % (name, n))

    storage = optuna.storages.RDBStorage("sqlite:///" + os.path.abspath(dst))
    config = storage._version_manager._create_alembic_config()

    # Drive Alembic through a ScriptDirectory we own; alembic.command.downgrade()
    # builds its own and re-imports the version modules, discarding the patch.
    script = ScriptDirectory.from_config(config)
    patched = _patch_stale_enum_names(script)
    print("\npatched stale enum name in: %s" % (", ".join(patched) or "nothing"))
    print("downgrading %s -> %s ..." % (before[0], TARGET))
    with EnvironmentContext(
        config,
        script,
        fn=lambda rev, _ctx: script._downgrade_revs(TARGET, rev),
        as_sql=False,
        starting_rev=None,
        destination_rev=TARGET,
        tag=None,
    ):
        script.run_env()
    storage.engine.dispose()

    after = _counts(dst)
    print("\nafter:  alembic=%s  trial_params=%d  trial_values=%d" % (after[0], after[2], after[3]))
    for name, n in after[1].items():
        print("        %-32s %d trials" % (name, n))

    ok = after[0] == TARGET and after[1] == before[1] and after[2:] == before[2:]
    print("\n%s" % ("OK: schema at %s, no rows lost" % TARGET if ok else "MISMATCH - do not use"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source .db (never modified), or the db to digest")
    ap.add_argument("dst", nargs="?", help="destination .db (must not exist)")
    ap.add_argument("--digest", action="store_true", help="print a digest of src and exit")
    args = ap.parse_args()

    if args.digest:
        for row in digest(args.src):
            print(json.dumps(row))
        return 0
    if not args.dst:
        ap.error("dst is required unless --digest is given")
    return downgrade(args.src, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())
