"""Render the final physics background-pairing confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)
from src.evaluation.physics_background_pairing_paper_comparison import (
    build_paper_comparison,
    load_authenticated_paper_reference,
)

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SELECTORS = (
    ("cap", "flat_physical", "Flat CAP"),
    ("cap", "physics_summary", "Summary CAP"),
    ("cap", "typed_sliced_wasserstein", "Typed CAP"),
    ("cap", "jetclr", "JetCLR CAP"),
    ("cap", "cdf", "CDF CAP"),
    ("wasserstein", "not_applicable", "W1"),
    ("drift", "not_applicable", "Drift"),
)


def _sha256(path: Path) -> str:
    """Hash one artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one nonempty rectangular CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid report input header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty report input: {path}")
    return rows


def _load_analysis(analysis_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Authenticate the analysis manifest and every table it declares."""
    manifest_path = analysis_dir / "confirmation_endpoint_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("content_sha256", None)
    if digest != _content_hash(manifest):
        raise ValueError("Confirmation analysis manifest content hash differs.")
    manifest["content_sha256"] = digest
    tables = {}
    for name, expected_hash in manifest.get("artifacts", {}).items():
        path = (analysis_dir / name).resolve(strict=True)
        if not path.is_relative_to(analysis_dir.resolve()) or _sha256(path) != expected_hash:
            raise ValueError(f"Confirmation analysis artifact changed: {path}")
        tables[name] = _read_csv(path)
    required = {
        "endpoints_across_seeds.csv",
        "comparison_summary.csv",
        "oas_vs_native_summary.csv",
        "reserve_seed_decisions.csv",
    }
    if not required <= set(tables):
        raise ValueError("Confirmation analysis lacks final-report tables.")
    return manifest, tables


def _number(row: dict[str, str], field: str) -> float:
    """Read one finite numeric result."""
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite report value for {field}.")
    return value


def _fmt(value: float, *, percent: bool = False, bold: bool = False) -> str:
    """Format an endpoint value and optional row winner."""
    text = f"{100.0 * value:.2f}%" if percent else f"{value:.3f}"
    return f"**{text}**" if bold else text


def _cell_table(rows: list[dict[str, str]]) -> tuple[list[str], dict[tuple[str, str, str], float]]:
    """Render all 42 cells with independent efficiency/AUPRC winners."""
    index = {
        (row["model"], row["selector"], row["pairing"], row["endpoint"]): _number(row, "mean")
        for row in rows
    }
    expected = {
        (model, selector, pairing, endpoint)
        for model in MODELS
        for selector, pairing, _ in SELECTORS
        for endpoint in ("efficiency_250hz", "auprc")
    }
    if set(index) != expected:
        raise ValueError("Across-seed summaries do not contain the exact 42-cell matrix.")
    lines = [
        "| Model | " + " | ".join(label for _, _, label in SELECTORS) + " |",
        "|---|" + "---:|" * len(SELECTORS),
    ]
    compact = {}
    for model in MODELS:
        efficiencies = [
            index[(model, selector, pairing, "efficiency_250hz")]
            for selector, pairing, _ in SELECTORS
        ]
        auprcs = [index[(model, selector, pairing, "auprc")] for selector, pairing, _ in SELECTORS]
        best_efficiency, best_auprc = max(efficiencies), max(auprcs)
        cells = []
        for (selector, pairing, _), efficiency, auprc in zip(SELECTORS, efficiencies, auprcs):
            cells.append(
                f"{_fmt(efficiency, percent=True, bold=efficiency == best_efficiency)} / "
                f"{_fmt(auprc, bold=auprc == best_auprc)}"
            )
            compact[(model, selector, pairing, "efficiency_250hz")] = efficiency
            compact[(model, selector, pairing, "auprc")] = auprc
        lines.append(f"| {model.upper()} | " + " | ".join(cells) + " |")
    return lines, compact


def _paper_rows(
    repository: Path, cell_values: dict[tuple[str, str, str, str], float]
) -> tuple[list[dict[str, Any]], str]:
    """Build semantically matched paper comparisons from confirmation means."""
    reference, reference_hash = load_authenticated_paper_reference(repository)
    winners = []
    for model in MODELS:
        for metric, pairing in (
            ("cap", "cdf"),
            ("drift", "not_applicable"),
            ("wasserstein", "not_applicable"),
        ):
            winners.append(
                {
                    "model": model,
                    "score": "native",
                    "metric": metric,
                    "strategy": "cdf" if metric == "cap" else "",
                    "mean_efficiency": cell_values[(model, metric, pairing, "efficiency_250hz")],
                }
            )
    return build_paper_comparison(reference, winners), reference_hash


def build_confirmation_report(
    analysis_dir: Path,
    repository: Path,
    *,
    random_aggregate: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build an information-dense report and machine-readable result summary."""
    manifest, tables = _load_analysis(analysis_dir.resolve(strict=True))
    cell_lines, cell_values = _cell_table(tables["endpoints_across_seeds.csv"])
    paper_rows, paper_reference_hash = _paper_rows(repository.resolve(strict=True), cell_values)
    comparisons = tables["comparison_summary.csv"]
    controlled = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "controlled"
    ]
    pairing = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "pairing"
    ]
    if len(controlled) != 6 * 2 * 2 or len(pairing) != 6 * 4 * 2:
        raise ValueError("Model-level comparison summary cardinality differs.")

    comparable_paper = [row for row in paper_rows if row["comparison_status"] == "comparable"]
    paper_better = sum(float(row["delta_percentage_points"]) > 0 for row in comparable_paper)
    cap_better = sum(_number(row, "mean_delta") > 0 for row in controlled)
    pairing_better = sum(_number(row, "mean_delta") > 0 for row in pairing)
    pairing_meaningful = sum(row["classification"] == "meaningful_improvement" for row in pairing)

    lines = [
        "# Physics background-pairing confirmation results",
        "",
        "## Direct answers",
        "",
        f"1. **Background0/background1 versus the paper:** {paper_better}/{len(comparable_paper)} semantically matched model/selector efficiency means exceed the paper values. This comparison is available only for efficiency; the paper did not report matching AUPRC.",
        f"2. **Does pairing improve CAP?** {pairing_better}/{len(pairing)} learned-pairing model/endpoint contrasts are positive relative to CDF CAP; {pairing_meaningful} are classified as practically meaningful after multiplicity adjustment.",
        f"3. **Does CAP beat W1/drift?** CDF CAP is positive in {cap_better}/{len(controlled)} model/endpoint comparisons. Learned pairing effects are reported separately, avoiding a best-of-four selection advantage.",
        "",
        "All values below are means over the fixed seeds and the same 20 downstream anomalies. Efficiency is at 250 Hz. Each cell is `efficiency / AUPRC`; bold marks the independent winner within that model and endpoint.",
        "",
        "## Complete confirmation matrix",
        "",
        *cell_lines,
        "",
        "## CDF CAP versus W1 and drift",
        "",
        "| Model | Endpoint | Contrast | Mean delta | Adjusted CI | Classification |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in controlled:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['model'].upper()} | {row['endpoint']} | {row['contrast']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, {scale * _number(row, 'adjusted_ci_high'):+.3f}] | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Learned pairing versus CDF CAP",
            "",
            "| Model | Pairing contrast | Endpoint | Mean delta | Adjusted CI | Classification |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in pairing:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['model'].upper()} | {row['contrast']} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, {scale * _number(row, 'adjusted_ci_high'):+.3f}] | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Matched comparison with the paper",
            "",
            "| Model | Method | Paper efficiency | Confirmation efficiency | Delta (pp) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparable_paper:
        lines.append(
            f"| {row['model'].upper()} | {row['paper_method']} | "
            f"{float(row['paper_mean_efficiency_percent']):.2f}% | "
            f"{float(row['campaign_mean_efficiency_percent']):.2f}% | "
            f"{float(row['delta_percentage_points']):+.2f} |"
        )
    lines.extend(
        [
            "",
            "These are semantically matched native-score comparisons: CDF CAP to paper CAP, forward drift to paper Stable, and W1 to paper W1. They test the full changed validation/selection system, not dataset choice in isolation.",
            "",
            "## AE/VAE masked-OAS diagnostic",
            "",
            "OAS is evaluated on the same native-selected checkpoints; it did not select configurations or checkpoints.",
            "",
            "| Model | Selector/pairing | Endpoint | OAS − native | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in tables["oas_vs_native_summary.csv"]:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        label = row["pairing"] if row["selector"] == "cap" else row["selector"]
        lines.append(
            f"| {row['model'].upper()} | {label} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'ci95_low'):+.3f}, {scale * _number(row, 'ci95_high'):+.3f}] |"
        )

    random_status = "not_supplied"
    if random_aggregate is not None:
        payload = json.loads(random_aggregate.read_text(encoding="utf-8"))
        if payload.get("seed_split_values_sha256") != _content_hash(
            payload.get("seed_split_values")
        ) or payload.get("summary_sha256") != _content_hash(payload.get("summary")):
            raise ValueError("Random-CAP aggregate content hash differs.")
        random_status = f"authenticated: {len(payload['seed_split_values'])} seed-level rows, {len(payload['summary'])} summaries"
    lines.extend(
        [
            "",
            "## Random-pairing negative control",
            "",
            f"Status: {random_status}. Random CAP is diagnostic only and never selected a hyperparameter or checkpoint.",
            "",
            "## Provenance and limitations",
            "",
            f"- Confirmation analysis SHA-256: `{manifest['content_sha256']}`.",
            f"- Paper-reference SHA-256: `{paper_reference_hash}`.",
            f"- Seeds: {', '.join(map(str, manifest['method']['seeds']))}; anomalies: {manifest['method']['anomalies']}.",
            "- Downstream anomalies were already inspected during discovery. These results establish 200-epoch and seed stability, not a fully untouched external holdout.",
            "- Paper AUPRC is unavailable and is not inferred.",
            "",
        ]
    )
    summary = {
        "schema_version": 1,
        "analysis_content_sha256": manifest["content_sha256"],
        "paper_reference_sha256": paper_reference_hash,
        "paper_matched_positive": paper_better,
        "paper_matched_total": len(comparable_paper),
        "cdf_cap_positive": cap_better,
        "cdf_cap_total": len(controlled),
        "learned_pairing_positive": pairing_better,
        "learned_pairing_total": len(pairing),
        "learned_pairing_meaningful": pairing_meaningful,
        "random_cap_status": random_status,
    }
    summary["content_sha256"] = _content_hash(summary)
    return "\n".join(lines), summary


def write_confirmation_report(
    analysis_dir: Path,
    repository: Path,
    output_dir: Path,
    *,
    random_aggregate: Path | None = None,
) -> dict[str, Any]:
    """Persist final Markdown and JSON atomically without overwriting different results."""
    report, summary = build_confirmation_report(
        analysis_dir, repository, random_aggregate=random_aggregate
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "REPORT.md": report + "\n",
        "report_summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
    }
    for name, content in artifacts.items():
        path = output_dir / name
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary = Path(temporary_name)
            if path.exists():
                if path.read_bytes() != temporary.read_bytes():
                    raise FileExistsError(f"Refusing to replace different final report: {path}")
            else:
                os.replace(temporary, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    return summary


__all__ = ["build_confirmation_report", "write_confirmation_report"]
