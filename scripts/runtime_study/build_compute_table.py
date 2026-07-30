#!/usr/bin/env python3
"""Build the paper's computational-resources table from measured timings.

Combines the two harvests of this directory:

* ``harvest_run_times.py`` -- per-epoch cost of every Pareto retraining,
  measured on clariden GH200 with one training process per GPU. This is the
  clean, uncontended number, and it is what the table's cells report.
* ``harvest_optuna_times.py`` -- wall-clock of every hyperparameter search
  trial, from the Optuna databases. The sweeps ran with ``logger=none``, so
  this is the only record of what they cost, and it is what the per-domain
  totals are built from.

The two cannot be interchanged: sweeps ran 50 epochs on the mixed hardware and
concurrency listed in ``HARDWARE`` below, Pareto retrainings ran 200 epochs on
a dedicated GH200. Reporting per-epoch cost sidesteps the epoch difference, and
each stage's total is summed from its own measurements.

GPU-hours divide the summed trial wall-clock by the concurrency: three trials
sharing one GPU for an hour cost one GPU-hour, not three.

Only the 250 Hz tier is used, so that the physics rows rest on the same amount
of data as CIFAR-10 and RobustAD, which have no q99 counterpart. The diagnostics
print what the excluded tier would have said.

Writes ``compute_summary.csv`` (one row per table cell), the table itself as
both ``compute_resources_table.tex`` and ``compute_resources_table.md``, and
prints diagnostics for judging how far each number can be trusted.

Usage::

    python scripts/runtime_study/build_compute_table.py
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

# The four strategies of the paper's table, in column order. Optuna study names
# and Pareto run names use different prefixes for the same strategy.
COLUMNS = [
    ("cvar25eff", "cvar25", "Semi"),
    ("drift", "stability", "Stable"),
    ("wasserstein", "wasserstein", "W1"),
    ("cap", "cap", "CAP"),
]

# GPU and concurrent training processes per GPU during the sweeps, exactly as
# reported in the paper's existing table.
HARDWARE = {
    ("physics", "ae"): ("L40S", 3),
    ("physics", "vae"): ("L40S", 3),
    ("physics", "dsae"): ("L40S", 3),
    ("physics", "dsvae"): ("H100", 2),
    ("physics", "realnvp"): ("H100", 2),
    ("physics", "svdd"): ("L40S", 3),
    ("cifar10", "ae"): ("gh200", 1),
    ("cifar10", "vae"): ("gh200", 1),
    ("cifar10", "realnvp"): ("gh200", 1),
    ("cifar10", "svdd"): ("gh200", 1),
    ("robustad", "ae"): ("L40S", 3),
    ("robustad", "vae"): ("L40S", 3),
    ("robustad", "realnvp"): ("L40S", 3),
    ("robustad", "svdd"): ("L40S", 3),
}

DOMAINS = [
    ("physics", "LHC L1 AD", ["ae", "vae", "dsae", "dsvae", "realnvp", "svdd"]),
    ("cifar10", "CIFAR-10", ["ae", "vae", "realnvp", "svdd"]),
    ("robustad", "RobustAD", ["ae", "vae", "realnvp", "svdd"]),
]

MODEL_LABEL = {"ae": "AE", "vae": "VAE", "dsae": "DSAE", "dsvae": "DSVAE",
               "realnvp": "RealNVP", "svdd": "SVDD"}

TIER = "250"
SEARCH_EPOCHS = 50  # trainer.max_epochs in every run*_search.sh
OUTLIER_IQR_FACTOR = 5.0
THIN_CELL_RUNS = 3  # below this, a cell is one hyperparameter draw, not a mean
THIN_CELL_TOLERANCE = 0.25  # deviation from sibling strategies still called ok


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in [0, 1]."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def split_outliers(values: list[float]) -> tuple[list[float], list[float]]:
    """Split into (bulk, outliers) at median + factor x IQR."""
    if len(values) < 4:
        return values, []
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    cutoff = statistics.median(values) + OUTLIER_IQR_FACTOR * (q3 - q1)
    bulk = [v for v in values if v <= cutoff]
    out = [v for v in values if v > cutoff]
    return (bulk, out) if bulk else (values, [])


def load_sweeps(indir: Path) -> dict:
    """(domain, model, strategy) -> trial minutes, 250 Hz tier."""
    sweeps: dict = {}
    for path in sorted(indir.glob("optuna_*.csv")):
        for row in read_csv(path):
            if row["tier"] != TIER:
                continue
            key = (row["domain"], row["model"], row["strategy"])
            sweeps.setdefault(key, []).append(float(row["minutes"]))
    return sweeps


def load_retrains(indir: Path) -> tuple[dict, dict]:
    """Retraining times, and per-epoch medians split by tier for the diagnostic.

    The first mapping holds (per-epoch seconds, total seconds) per run of the
    250 Hz tier; the second holds per-epoch seconds for every tier, used only
    to show that the excluded q99 runs would have given the same answer.
    """
    retrains: dict = {}
    by_tier: dict = {}
    for path in sorted(indir.glob("*_pareto.csv")):
        for row in read_csv(path):
            key = (row["domain"], row["model"], row["strategy"])
            sec_per_epoch = float(row["sec_per_epoch"])
            by_tier.setdefault((*key, row["tier"]), []).append(sec_per_epoch)
            if row["tier"] == TIER:
                retrains.setdefault(key, []).append(
                    (sec_per_epoch, float(row["total_s"]))
                )
    return retrains, by_tier


def build_summary(sweeps: dict, retrains: dict) -> list[dict]:
    """One row per table cell, holding every number the outputs draw on."""
    rows = []
    for domain, _, models in DOMAINS:
        for model in models:
            gpu, conc = HARDWARE[(domain, model)]
            for study_strategy, run_strategy, label in COLUMNS:
                trials = sweeps.get((domain, model, study_strategy), [])
                runs = retrains.get((domain, model, run_strategy), [])
                bulk, outliers = split_outliers(trials)

                row = {
                    "domain": domain, "model": model, "strategy": study_strategy,
                    "column": label, "sweep_gpu": gpu, "sweep_conc": conc,
                    "n_trials": len(trials),
                    "trial_median_min":
                        round(statistics.median(bulk), 3) if bulk else "",
                    "trial_p10_min":
                        round(percentile(bulk, 0.10), 3) if bulk else "",
                    "trial_sum_h": round(sum(trials) / 60, 2) if trials else 0,
                    "n_outliers": len(outliers),
                    "outlier_h": round(sum(outliers) / 60, 2) if outliers else 0,
                    "n_runs": len(runs),
                }
                if runs:
                    eps = [sec for sec, _ in runs]
                    row["sec_per_epoch"] = round(statistics.median(eps), 3)
                    row["sec_per_epoch_iqr"] = round(
                        percentile(eps, 0.75) - percentile(eps, 0.25), 3
                    )
                    row["retrain_sum_h"] = round(
                        sum(total for _, total in runs) / 3600, 3
                    )
                else:
                    row["sec_per_epoch"] = ""
                    row["sec_per_epoch_iqr"] = ""
                    row["retrain_sum_h"] = 0
                rows.append(row)
    return rows


def domain_totals(cell: dict) -> dict:
    """Per-domain GPU-hours, split into sweep and retraining."""
    totals = {}
    for domain, _, models in DOMAINS:
        sweep_h = retrain_h = equiv_h = 0.0
        n_trials = n_runs = 0
        for model in models:
            _, conc = HARDWARE[(domain, model)]
            for study_strategy, _, _ in COLUMNS:
                r = cell[(domain, model, study_strategy)]
                n_trials += r["n_trials"]
                n_runs += r["n_runs"]
                sweep_h += r["trial_sum_h"] / conc
                retrain_h += r["retrain_sum_h"]
                if r["sec_per_epoch"] != "":
                    equiv_h += (
                        r["n_trials"] * SEARCH_EPOCHS * r["sec_per_epoch"] / 3600
                    )
        totals[domain] = {
            "sweep_h": sweep_h, "retrain_h": retrain_h,
            "total_h": sweep_h + retrain_h, "equiv_h": equiv_h,
            "n_trials": n_trials, "n_runs": n_runs,
        }
    return totals


def cell_values(cell: dict, domain: str, model: str) -> list[str]:
    """The four per-epoch numbers of one table row, '--' where unmeasured."""
    out = []
    for study_strategy, _, _ in COLUMNS:
        v = cell[(domain, model, study_strategy)]["sec_per_epoch"]
        out.append(f"{v:.1f}" if v != "" else "--")
    return out


def latex_table(cell: dict, totals: dict) -> str:
    def num(value):
        return f"{value:,.0f}".replace(",", "{,}")

    grand = sum(t["total_h"] for t in totals.values())
    lines = [
        r"\begin{table}[t]",
        r"  \caption{",
        r"  Computational resources for the study.",
        r"  Each entry is the median wall-clock time per training epoch, in",
        r"  seconds, measured on a single NVIDIA GH200 running one training",
        r"  process per GPU.",
        r"  Epoch cost is quoted rather than time per trial because the",
        f"  hyperparameter sweeps ran for {SEARCH_EPOCHS} epochs while the models",
        r"  retrained from the Pareto fronts ran for 200.",
        r"  Each domain heading gives the total cost of that study: the",
        r"  hyperparameter sweep plus the Pareto retrainings, in GPU-hours, taken",
        r"  from the recorded duration of every trial and every retraining and",
        r"  divided by the number of training processes sharing each GPU.",
        f"  The three studies together account for {num(grand)} GPU-hours.",
        r"  }",
        r"  \label{tab:compute_resources}",
        r"  \centering",
        r"  \scriptsize",
        r"  \renewcommand{\arraystretch}{1.25}",
        r"  \setlength{\tabcolsep}{7.5pt}",
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        "  Model & " + " & ".join(c[2] for c in COLUMNS) + r" \\",
        r"  \midrule",
    ]
    for i, (domain, title, models) in enumerate(DOMAINS):
        t = totals[domain]
        header = (f"\\textbf{{{title}}} --- {num(t['n_trials'])} sweep trials + "
                  f"{num(t['n_runs'])} retrainings, {num(t['total_h'])} GPU-h")
        lines.append(f"  \\multicolumn{{5}}{{l}}{{{header}}} \\\\")
        lines.append(r"  \specialrule{0.08em}{0.15em}{0.15em}")
        for model in models:
            vals = " & ".join(cell_values(cell, domain, model))
            lines.append(f"  {MODEL_LABEL[model]:<8}& {vals} " + r"\\")
        if i < len(DOMAINS) - 1:
            lines += [r"  \specialrule{0.08em}{0.15em}{0.15em}", ""]
    lines += [r"  \bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def markdown_table(cell: dict, totals: dict) -> str:
    grand = sum(t["total_h"] for t in totals.values())
    lines = [
        "# Computational resources",
        "",
        "Median wall-clock **seconds per training epoch**, measured on a single",
        "NVIDIA GH200 running one training process per GPU. Epoch cost is quoted",
        f"rather than time per trial because the sweeps ran {SEARCH_EPOCHS} epochs",
        "while the Pareto retrainings ran 200.",
        "",
        "Each domain heading gives that study's total: the hyperparameter sweep",
        "plus the Pareto retrainings, in GPU-hours, from the recorded duration of",
        "every trial and retraining divided by the training processes sharing each",
        f"GPU. All three together account for **{grand:,.0f} GPU-hours**.",
        "",
        "| Model | " + " | ".join(c[2] for c in COLUMNS) + " |",
        "|:--|" + "--:|" * len(COLUMNS),
    ]
    for domain, title, models in DOMAINS:
        t = totals[domain]
        lines.append(
            f"| **{title}** — {t['n_trials']:,} sweep trials + "
            f"{t['n_runs']:,} retrainings, {t['total_h']:,.0f} GPU-h |"
            + " |" * len(COLUMNS)
        )
        for model in models:
            vals = " | ".join(cell_values(cell, domain, model))
            lines.append(f"| {MODEL_LABEL[model]} | {vals} |")
    return "\n".join(lines)


def print_totals(totals: dict) -> None:
    print("\n" + "=" * 78)
    print("TOTALS  (sweep GPU-h = summed trial wall-clock / concurrency)")
    print("=" * 78)
    print(f"{'domain':<10}{'trials':>8}{'sweep h':>10}{'runs':>7}"
          f"{'retrain h':>11}{'TOTAL h':>10}{'GH200-equiv':>13}")
    for domain, _, _ in DOMAINS:
        t = totals[domain]
        print(f"{domain:<10}{t['n_trials']:>8,}{t['sweep_h']:>10,.1f}"
              f"{t['n_runs']:>7,}{t['retrain_h']:>11,.1f}"
              f"{t['total_h']:>10,.1f}{t['equiv_h']:>13,.1f}")
    grand = sum(t["total_h"] for t in totals.values())
    print(f"{'ALL':<10}{'':>8}{'':>10}{'':>7}{'':>11}{grand:>10,.1f}")
    print(f"\nGH200-equiv = n_trials x {SEARCH_EPOCHS} epochs x measured GH200")
    print("per-epoch cost: what the sweeps would have cost re-run on clariden,")
    print("one job per GPU. It exceeds 'sweep h' mainly because of packing, not")
    print("raw speed -- the sweeps put 2-3 trials on each GPU and they barely")
    print("slowed each other (these models are small enough to leave a GPU")
    print("underused). Quote 'sweep h' as what was actually spent.")


def print_spread(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SPREAD  trial minutes, and where the spread comes from")
    print("=" * 78)
    print("A cell's trial times vary for two reasons: GPU contention (one-sided --")
    print("a busy GPU only ever slows a trial) and the search space itself (trials")
    print("sample different architectures, so some are genuinely cheaper). The two")
    print("cannot be separated from durations alone, but the last column can tell")
    print("them apart: the Pareto retrainings span an equally diverse set of")
    print("hyperparameters on a dedicated GPU, so if their per-epoch spread is just")
    print("as wide, the variation is the search space, not contention.")
    print(f"\n{'domain':<9}{'model':<9}{'strat':<12}{'n':>6}{'p10':>8}{'med':>8}"
          f"{'med/p10':>9}{'outl':>6}{'out h':>8}{'GH200 iqr/med':>15}")
    for r in rows:
        if not r["n_trials"]:
            continue
        ratio = r["trial_median_min"] / r["trial_p10_min"] if r["trial_p10_min"] else 0
        if r["sec_per_epoch"] not in ("", 0):
            rel = f"{r['sec_per_epoch_iqr'] / r['sec_per_epoch']:>15.2f}"
        else:
            rel = f"{'--':>15}"
        flag = "  <-- wide" if ratio > 1.5 else ""
        print(f"{r['domain']:<9}{r['model']:<9}{r['strategy']:<12}{r['n_trials']:>6}"
              f"{r['trial_p10_min']:>8.2f}{r['trial_median_min']:>8.2f}{ratio:>9.2f}"
              f"{r['n_outliers']:>6}{r['outlier_h']:>8.1f}{rel}{flag}")


def print_cross_check(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"CROSS-CHECK  original-hardware trial vs GH200 fit, same {SEARCH_EPOCHS} epochs")
    print("=" * 78)
    print("Two independent measurements of the same 50 epochs. They are not")
    print("expected to be equal -- the trial also carries setup and its own")
    print("post-fit evaluation, and the agnostic sweeps ran with anomaly_eff")
    print("disabled while the Pareto runs kept it -- so read the ratio for")
    print("CONSISTENCY, not for closeness to 1. Roughly constant across the three")
    print("cheap strategies within a model means both sources agree; a cell far")
    print("off its neighbours is the one to distrust.")
    print(f"\n{'domain':<9}{'model':<9}{'strat':<12}{'p10 min':>9}"
          f"{'GH200 min':>11}{'ratio':>8}{'n_runs':>8}")
    for r in rows:
        if not r["n_trials"] or r["sec_per_epoch"] == "":
            continue
        gh200_min = SEARCH_EPOCHS * r["sec_per_epoch"] / 60
        flag = "  <-- thin" if r["n_runs"] < THIN_CELL_RUNS else ""
        print(f"{r['domain']:<9}{r['model']:<9}{r['strategy']:<12}"
              f"{r['trial_p10_min']:>9.2f}{gh200_min:>11.2f}"
              f"{r['trial_p10_min'] / gh200_min:>8.2f}{r['n_runs']:>8}{flag}")


def print_tier_check(by_tier: dict) -> None:
    """What the excluded q99 tier would have said, per cell."""
    pairs = []
    for (domain, model, strategy, tier), eps in by_tier.items():
        other = by_tier.get((domain, model, strategy, "q99"))
        if tier == TIER and other:
            pairs.append((domain, model, strategy,
                          statistics.median(eps), statistics.median(other)))
    if not pairs:
        return
    print("\n" + "=" * 78)
    print(f"TIER CHECK  per-epoch seconds, {TIER} Hz vs the excluded q99 tier")
    print("=" * 78)
    print("q99 is excluded so the physics rows rest on the same amount of data as")
    print("the other domains. Agreement here means that exclusion costs precision")
    print("rather than changing the answer.")
    print(f"\n{'domain':<9}{'model':<9}{'strat':<12}{TIER:>9}{'q99':>9}{'ratio':>8}")
    for domain, model, strategy, a, b in sorted(pairs):
        flag = "  <-- differs" if a and abs(b / a - 1) > 0.15 else ""
        print(f"{domain:<9}{model:<9}{strategy:<12}{a:>9.2f}{b:>9.2f}"
              f"{b / a if a else 0:>8.2f}{flag}")


def print_thin_cells(rows: list[dict], cell: dict) -> None:
    thin = [r for r in rows if 0 < r["n_runs"] < THIN_CELL_RUNS]
    if not thin:
        return
    print("\n" + "=" * 78)
    print(f"THIN CELLS  per-epoch cost rests on fewer than {THIN_CELL_RUNS} runs")
    print("=" * 78)
    print("Per-epoch cost is driven mostly by the model, so a thin cell can be")
    print("sanity-checked against the same model's other cheap strategies (CAP is")
    print("excluded from the reference -- it is the one that genuinely costs more).")
    print(f"'ok' means it sits within {THIN_CELL_TOLERANCE:.0%} of them.")
    print(f"\n{'domain':<9}{'model':<9}{'strat':<12}{'n':>3}{'value':>8}"
          f"{'siblings':>10}{'dev':>8}   verdict")
    for r in thin:
        peers = [
            c["sec_per_epoch"]
            for (d, m, s), c in cell.items()
            if d == r["domain"] and m == r["model"]
            and s not in (r["strategy"], "cap") and c["sec_per_epoch"] != ""
        ]
        if not peers or r["strategy"] == "cap":
            print(f"  {r['domain']}/{r['model']} {r['strategy']}: "
                  f"n={r['n_runs']}, {r['sec_per_epoch']} s (no reference)")
            continue
        ref = statistics.median(peers)
        dev = r["sec_per_epoch"] / ref - 1
        verdict = ("ok" if abs(dev) <= THIN_CELL_TOLERANCE
                   else "SUSPECT -- check before quoting")
        print(f"{r['domain']:<9}{r['model']:<9}{r['strategy']:<12}"
              f"{r['n_runs']:>3}{r['sec_per_epoch']:>8.2f}{ref:>10.2f}"
              f"{dev * 100:>+7.0f}%   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--indir", default=None,
                        help="Dir holding the harvested CSVs "
                             "(default: <repo>/notebooks/run_times).")
    parser.add_argument("--outdir", default=None,
                        help="Where to write the summary and tables "
                             "(default: same as --indir).")
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    indir = Path(args.indir) if args.indir else repo / "notebooks" / "run_times"
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    sweeps = load_sweeps(indir)
    retrains, by_tier = load_retrains(indir)
    if not sweeps or not retrains:
        raise SystemExit(f"no harvested CSVs found in {indir}")

    rows = build_summary(sweeps, retrains)
    cell = {(r["domain"], r["model"], r["strategy"]): r for r in rows}
    totals = domain_totals(cell)

    csv_path = outdir / "compute_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex, md = latex_table(cell, totals), markdown_table(cell, totals)
    (outdir / "compute_resources_table.tex").write_text(tex + "\n")
    (outdir / "compute_resources_table.md").write_text(md + "\n")
    for name in ("compute_summary.csv", "compute_resources_table.tex",
                 "compute_resources_table.md"):
        print(f"wrote {outdir / name}")

    print()
    print(md)
    print_totals(totals)
    print_spread(rows)
    print_cross_check(rows)
    print_tier_check(by_tier)
    print_thin_cells(rows, cell)


if __name__ == "__main__":
    main()
