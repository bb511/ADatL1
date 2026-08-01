#!/usr/bin/env python3
"""Insert one strategy's section into an existing run<model>_pareto.sh.

Why this exists
---------------
``scripts/make_pareto_scripts.py`` writes whole files (``out_path.write_text``),
so regenerating in place to add a strategy also rewrites every existing block.
That is not wanted here: the cap/cvar/drift/wasserstein points have already been
retrained and their blocks document exactly what produced the current results
table. Commit ``02024aa`` also changed those strategies' callback lists without
the scripts ever being regenerated, so a plain rerun would silently rewrite all
786 of them.

So instead: generate into a scratch tree from a directory holding only the new
strategy's CSVs (``make_pareto_scripts.py --paretos-dir ... --out-root ...``),
which yields files with exactly one section, and splice that section into the
real file at the position ``STRATEGY_ORDER`` dictates.

    cvar25eff -> cvar10eff -> cap -> consistency -> drift -> wasserstein

A section runs from its ``BANNER`` header to the line before the next banner
title, and is inserted immediately before the section that follows it in the
order (for ``consistency``, that is ``STABILITY``; ``drift``'s section title).

Safety
------
The splice is verified to be purely additive: after writing, the inserted lines
are removed again and the result must equal the original file byte for byte.
Anything else aborts before the file is touched. Files that already contain the
section are skipped, so re-running is harmless.

Usage
-----
    python scripts/splice_pareto_section.py --generated-root /tmp/pareto_gen
    python scripts/splice_pareto_section.py --generated-root /tmp/pareto_gen \\
        --exclude '*_q99_pareto.sh' --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

BANNER = "# " + 72 * "="

# Section titles in STRATEGY_ORDER. Used to find where a new section belongs:
# it goes immediately before the first title that follows it here and is present
# in the target file.
TITLE_ORDER = ["CVAR25", "CVAR10", "CAP", "CONSISTENCY", "STABILITY", "WASSERSTEIN"]


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def title_at(lines: list[str], i: int) -> str | None:
    """Section title if lines[i] opens a section banner, else None.

    A section header is three lines: BANNER, '# <TITLE> TRAINING  (study: ...)',
    BANNER. Matching on the middle line alone would also hit the file's own
    title banner and the SUBMIT footer.
    """
    if i + 2 >= len(lines):
        return None
    if lines[i] != BANNER or lines[i + 2] != BANNER:
        return None
    mid = lines[i + 1]
    if not mid.startswith("# ") or " TRAINING  (study:" not in mid:
        return None
    return mid[2:].split(" TRAINING")[0].strip()


def sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """(title, start, end) for each section; end is exclusive."""
    starts = [(t, i) for i in range(len(lines)) if (t := title_at(lines, i))]
    out = []
    for n, (title, start) in enumerate(starts):
        if n + 1 < len(starts):
            end = starts[n + 1][1]
        else:
            # Last section runs to the SUBMIT footer banner, or to EOF.
            end = len(lines)
            for j in range(start + 3, len(lines) - 1):
                if lines[j] == BANNER and lines[j + 1].startswith("# SUBMIT "):
                    end = j
                    break
        out.append((title, start, end))
    return out


def insertion_point(lines: list[str], title: str) -> int:
    """Line index at which `title`'s section should be inserted."""
    present = sections(lines)
    by_title = {t: (s, e) for t, s, e in present}
    order = TITLE_ORDER.index(title)
    # Before the first later section that exists.
    for later in TITLE_ORDER[order + 1 :]:
        if later in by_title:
            return by_title[later][0]
    # No later section: after the last existing one.
    if present:
        return present[-1][2]
    raise SystemExit("no sections found in target")


def splice(generated: Path, target: Path, title: str, dry: bool) -> tuple[bool, str]:
    gen_lines = generated.read_text().splitlines()
    tgt_lines = target.read_text().splitlines()

    gen_secs = [s for s in sections(gen_lines) if s[0] == title]
    if not gen_secs:
        return False, "no %s section in generated file" % title
    if len(gen_secs) > 1:
        return False, "generated file has %d %s sections" % (len(gen_secs), title)
    _, gs, ge = gen_secs[0]
    block = gen_lines[gs:ge]

    if any(t == title for t, _, _ in sections(tgt_lines)):
        return False, "target already has a %s section - skipped" % title

    at = insertion_point(tgt_lines, title)
    new_lines = tgt_lines[:at] + block + tgt_lines[at:]

    # Purely-additive check: strip the inserted span back out and require an
    # exact match with what was there before.
    if new_lines[:at] + new_lines[at + len(block) :] != tgt_lines:
        return False, "ABORT: splice would alter existing content"

    n_cmds = sum(1 for l in block if l.startswith("# python3 src/train.py"))
    before = sum(1 for l in tgt_lines if l.startswith("# python3 src/train.py"))
    after = sum(1 for l in new_lines if l.startswith("# python3 src/train.py"))
    if after != before + n_cmds:
        return False, "ABORT: command count %d != %d + %d" % (after, before, n_cmds)

    if not dry:
        target.write_text("\n".join(new_lines).rstrip("\n") + "\n")
    return True, "%d commands inserted at line %d (was %d, now %d)" % (
        n_cmds,
        at + 1,
        before,
        after,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generated-root", required=True,
                    help="tree written by make_pareto_scripts.py --out-root")
    ap.add_argument("--strategy-title", default="CONSISTENCY",
                    help="section title to move (default: CONSISTENCY)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob of generated filenames to skip (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.strategy_title not in TITLE_ORDER:
        ap.error("unknown section title %r; known: %s"
                 % (args.strategy_title, ", ".join(TITLE_ORDER)))

    repo = find_repo_root(Path(__file__).resolve().parent)
    root = Path(args.generated_root).expanduser()
    gens = sorted(root.glob("*/run*_pareto.sh"))
    if not gens:
        raise SystemExit("no generated files under %s" % root)

    ok = skipped = failed = 0
    for gen in gens:
        rel = "%s/%s" % (gen.parent.name, gen.name)
        if any(fnmatch.fnmatch(gen.name, pat) for pat in args.exclude):
            print("%-42s SKIP (excluded)" % rel)
            skipped += 1
            continue
        target = repo / "scripts" / gen.parent.name / gen.name
        if not target.exists():
            print("%-42s SKIP (no target: %s)" % (rel, target))
            skipped += 1
            continue
        good, msg = splice(gen, target, args.strategy_title, args.dry_run)
        print("%-42s %s %s" % (rel, "OK  " if good else "FAIL", msg))
        if good:
            ok += 1
        elif "skipped" in msg:
            skipped += 1
        else:
            failed += 1

    print("\n%d spliced, %d skipped, %d failed%s"
          % (ok, skipped, failed, "  (dry run)" if args.dry_run else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
