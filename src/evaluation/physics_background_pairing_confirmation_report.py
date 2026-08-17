"""Render the final physics background-pairing confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
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
from src.evaluation.physics_background_pairing_random_cap_analysis import (
    DISCOVERY_MANIFEST_CONTENT_SHA256,
    DISCOVERY_MANIFEST_SHA256,
    DISCOVERY_MEMBERSHIP_SCORES,
    DISCOVERY_MEMBERSHIPS,
    DISCOVERY_NAMESPACE,
    DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256,
    DISCOVERY_TASKS,
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


def _load_random_analysis(
    analysis_dir: Path,
    expected_mode: str,
    *,
    confirmation_provenance: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Authenticate one portable random-CAP analysis package and its tables."""
    analysis_dir = analysis_dir.resolve(strict=True)
    manifest_path = analysis_dir / "random_cap_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("content_sha256", None)
    if digest != _content_hash(manifest):
        raise ValueError("Random-CAP analysis receipt content hash differs.")
    manifest["content_sha256"] = digest
    if (
        manifest.get("schema_version") != 1
        or manifest.get("method", {}).get("mode") != expected_mode
        or manifest.get("method", {}).get("purpose") != "negative_control_diagnostic_only"
        or not manifest.get("method", {}).get(
            "random_cap_did_not_select_hyperparameters_or_checkpoints"
        )
    ):
        raise ValueError(f"Random-CAP {expected_mode} analysis identity differs.")
    inputs = manifest.get("inputs", {})
    if expected_mode == "discovery":
        expected_identity = {
            "manifest_namespace": DISCOVERY_NAMESPACE,
            "manifest_sha256": DISCOVERY_MANIFEST_SHA256,
            "manifest_content_sha256": DISCOVERY_MANIFEST_CONTENT_SHA256,
            "random_pairing_content_sha256": DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256,
        }
        expected_counts = {
            "tasks": DISCOVERY_TASKS,
            "memberships": DISCOVERY_MEMBERSHIPS,
            "membership_scores": DISCOVERY_MEMBERSHIP_SCORES,
            "seed_rows": DISCOVERY_MEMBERSHIP_SCORES * 10,
            "variability_rows": DISCOVERY_MEMBERSHIP_SCORES * 2,
        }
        if any(inputs.get(key) != value for key, value in expected_identity.items()) or any(
            int(manifest["method"].get(key, -1)) != value for key, value in expected_counts.items()
        ):
            raise ValueError(
                "Discovery random-CAP authoritative provenance or cardinality differs."
            )
    elif confirmation_provenance is None or any(
        inputs.get(key) != value for key, value in confirmation_provenance.items()
    ):
        raise ValueError(
            "Confirmation random-CAP analysis is not bound to the endpoint provenance."
        )
    tables = {}
    for name, expected_hash in manifest.get("artifacts", {}).items():
        path = (analysis_dir / name).resolve(strict=True)
        if not path.is_relative_to(analysis_dir) or _sha256(path) != expected_hash:
            raise ValueError(f"Random-CAP analysis artifact changed: {path}")
        tables[name] = _read_csv(path)
    if "random_cap_variability.csv" not in tables or "random_cap_values.csv" not in tables:
        raise ValueError("Random-CAP analysis lacks values or variability evidence.")
    if expected_mode == "discovery" and "structured_random_rank_associations.csv" not in tables:
        raise ValueError("Discovery random-CAP analysis lacks rank associations.")
    values = tables["random_cap_values.csv"]
    variability = tables["random_cap_variability.csv"]
    required_value_fields = {
        "task_index",
        "membership_index",
        "evaluation_score",
        "split",
        "replicate",
        "pairing_seed",
        "checkpoint_sha256",
        "pair_table_sha256",
        "cap",
    }
    required_variability_fields = {
        "task_index",
        "membership_index",
        "evaluation_score",
        "split",
        "n_pairings",
        "cap_mean",
        "cap_sample_variance",
        "cap_sample_std",
        "cap_range",
    }
    if (
        any(not required_value_fields <= set(row) for row in values)
        or any(not required_variability_fields <= set(row) for row in variability)
        or len(values) != int(manifest["method"].get("seed_rows", -1))
        or len(variability) != int(manifest["method"].get("variability_rows", -1))
    ):
        raise ValueError("Random-CAP analysis table schema or cardinality differs.")
    value_keys = {
        (
            row["task_index"],
            row["membership_index"],
            row["evaluation_score"],
            row["split"],
            row["replicate"],
        )
        for row in values
    }
    variability_keys = {
        (row["task_index"], row["membership_index"], row["evaluation_score"], row["split"])
        for row in variability
    }
    if len(value_keys) != len(values) or len(variability_keys) != len(variability):
        raise ValueError("Random-CAP analysis contains duplicate value or variability keys.")
    for row in values:
        _number(row, "cap")
    for row in variability:
        if int(row["n_pairings"]) != 5:
            raise ValueError("Random-CAP variability row does not contain five pairings.")
        for field in ("cap_mean", "cap_sample_variance", "cap_sample_std", "cap_range"):
            _number(row, field)
    if expected_mode == "discovery":
        associations = tables["structured_random_rank_associations.csv"]
        required_association_fields = {
            "model",
            "logical_objective",
            "split",
            "random_source",
            "n_checkpoints",
            "spearman_rho",
            "pairing_seed_rho_finite_count",
            "pairing_seed_rho_undefined_count",
            "pairing_seed_rho_min",
            "pairing_seed_rho_max",
            "pairing_seed_rho_range",
        }
        if len(associations) != int(manifest["method"].get("rank_association_rows", -1)) or any(
            not required_association_fields <= set(row) for row in associations
        ):
            raise ValueError("Random-CAP rank-association schema or cardinality differs.")
        panels: dict[tuple[str, str, str], set[str]] = {}
        for row in associations:
            key = (row["model"], row["logical_objective"], row["split"])
            sources = panels.setdefault(key, set())
            if row["random_source"] in sources:
                raise ValueError(f"Duplicate random-CAP association source for {key}.")
            sources.add(row["random_source"])
            if row["spearman_rho"]:
                rho = _number(row, "spearman_rho")
                if not -1.0 <= rho <= 1.0:
                    raise ValueError("Random-CAP Spearman rho is outside [-1, 1].")
            finite_count = int(row["pairing_seed_rho_finite_count"])
            undefined_count = int(row["pairing_seed_rho_undefined_count"])
            if finite_count + undefined_count != 5:
                raise ValueError("Random-CAP pairing-seed rho coverage is not five.")
            if finite_count:
                low = _number(row, "pairing_seed_rho_min")
                high = _number(row, "pairing_seed_rho_max")
                span = _number(row, "pairing_seed_rho_range")
                if low > high or not math.isclose(span, high - low, abs_tol=1e-12):
                    raise ValueError("Random-CAP pairing-seed rho range differs.")
        expected_sources = {"five_pairing_mean", *[f"pairing_replicate_{i}" for i in range(5)]}
        if len(panels) != int(manifest["method"].get("rank_panels", -1)) or any(
            sources != expected_sources for sources in panels.values()
        ):
            raise ValueError("Random-CAP rank-panel coverage differs.")
    return manifest, tables


def _confirmation_provenance(
    analysis_manifest: dict[str, Any],
) -> dict[str, str | None]:
    """Authenticate and expose the endpoint analysis's upstream identities."""
    inputs = analysis_manifest.get("inputs", {})
    matrix_path = Path(inputs.get("matrix", "")).resolve(strict=True)
    if _sha256(matrix_path) != inputs.get("matrix_sha256"):
        raise ValueError("Confirmation endpoint matrix file hash differs.")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    digest = matrix.get("content_sha256")
    if digest != _content_hash(
        {key: value for key, value in matrix.items() if key != "content_sha256"}
    ):
        raise ValueError("Confirmation endpoint matrix content hash differs.")
    endpoint_path = Path(inputs.get("endpoint_receipt", "")).resolve(strict=True)
    endpoint_sha256 = inputs.get("endpoint_receipt_sha256")
    if _sha256(endpoint_path) != endpoint_sha256:
        raise ValueError("Confirmation endpoint receipt file hash differs.")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    if endpoint.get("matrix_content_sha256") != digest:
        raise ValueError("Confirmation endpoint receipt matrix identity differs.")
    return {
        "confirmation_matrix_content_sha256": str(digest),
        "runner_manifest_content_sha256": endpoint.get("effective_runner_content_sha256"),
        "reserve_extension_content_sha256": endpoint.get("reserve_extension_content_sha256"),
        "checkpoint_selections_sha256": endpoint.get("artifacts", {}).get(
            "checkpoint_selections.csv"
        ),
        "endpoint_receipt_sha256": endpoint_sha256,
    }


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


def _validated_reserve_decisions(
    rows: list[dict[str, str]], seeds_by_model: dict[str, list[int]]
) -> dict[str, dict[str, str]]:
    """Require one terminal reserve-seed decision for every model."""
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        if model not in MODELS or model in decisions:
            raise ValueError("Reserve-seed decisions must name each model exactly once.")
        decision = row.get("decision")
        if decision not in {"no_extension", "extension_complete"}:
            raise RuntimeError(
                f"Final report blocked: reserve-seed extension is incomplete for {model}: "
                f"{decision!r}."
            )
        expected = 3 if decision == "no_extension" else 5
        observed = int(row.get("observed_seed_count", expected))
        if observed != expected or observed != len(seeds_by_model.get(model, [])):
            raise ValueError(
                f"Reserve decision/count mismatch for {model}: {decision} with {observed} seeds."
            )
        decisions[model] = row
    if set(decisions) != set(MODELS):
        raise ValueError("Reserve-seed decisions must cover all six models.")
    return decisions


def _contrast_counts(rows: list[dict[str, str]], endpoint: str) -> tuple[int, int, int]:
    """Return directional-positive, adjusted-improvement, and total counts."""
    selected = [row for row in rows if row["endpoint"] == endpoint]
    return (
        sum(_number(row, "mean_delta") > 0.0 for row in selected),
        sum(row["classification"] == "meaningful_improvement" for row in selected),
        len(selected),
    )


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
    discovery_random_analysis: Path | None = None,
    confirmation_random_analysis: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build an information-dense report and machine-readable result summary."""
    manifest, tables = _load_analysis(analysis_dir.resolve(strict=True))
    seeds_by_model = manifest["method"].get("training_seeds_by_model") or {
        model: manifest["method"]["seeds"] for model in MODELS
    }
    reserve_decisions = _validated_reserve_decisions(
        tables["reserve_seed_decisions.csv"], seeds_by_model
    )
    cell_lines, cell_values = _cell_table(tables["endpoints_across_seeds.csv"])
    paper_rows, paper_reference_hash = _paper_rows(repository.resolve(strict=True), cell_values)
    comparisons = tables["comparison_summary.csv"]
    controlled = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "controlled"
    ]
    pairing = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "pairing"
    ]
    macro_controlled = [
        row for row in comparisons if row["scope"] == "macro" and row["family"] == "controlled"
    ]
    macro_pairing = [
        row for row in comparisons if row["scope"] == "macro" and row["family"] == "pairing"
    ]
    if len(controlled) != 6 * 2 * 2 or len(pairing) != 6 * 4 * 2:
        raise ValueError("Model-level comparison summary cardinality differs.")
    if len(macro_controlled) != 2 * 2 or len(macro_pairing) != 4 * 2:
        raise ValueError("Macro comparison summary cardinality differs.")

    comparable_paper = [row for row in paper_rows if row["comparison_status"] == "comparable"]
    paper_better = sum(float(row["delta_percentage_points"]) > 0 for row in comparable_paper)
    cap_eff = _contrast_counts(controlled, "efficiency_250hz")
    cap_auprc = _contrast_counts(controlled, "auprc")
    pairing_eff = _contrast_counts(pairing, "efficiency_250hz")
    pairing_auprc = _contrast_counts(pairing, "auprc")

    lines = [
        "# Physics background-pairing confirmation results",
        "",
        "## Direct answers",
        "",
        f"1. **Background0/background1 system versus the paper:** {paper_better}/{len(comparable_paper)} method-matched efficiency means are directionally higher. This is a cross-system benchmark, not a causal isolation of the changed validation datasets; the paper has no matching AUPRC.",
        f"2. **Does pairing improve CAP?** Efficiency: {pairing_eff[0]}/{pairing_eff[2]} directional gains and {pairing_eff[1]} multiplicity-adjusted meaningful improvements. AUPRC: {pairing_auprc[0]}/{pairing_auprc[2]} directional gains and {pairing_auprc[1]} adjusted meaningful improvements.",
        f"3. **Does CDF CAP beat W1/drift?** Efficiency: {cap_eff[0]}/{cap_eff[2]} directional gains and {cap_eff[1]} adjusted meaningful improvements. AUPRC: {cap_auprc[0]}/{cap_auprc[2]} directional gains and {cap_auprc[1]} adjusted meaningful improvements. W1 and drift remain separate contrasts below.",
        "",
        "All values below are means over the fixed seeds and the same 20 downstream anomalies. Efficiency is at 250 Hz. Each cell is `efficiency / AUPRC`; bold marks the independent winner within that model and endpoint.",
        "",
        "## Complete confirmation matrix",
        "",
        *cell_lines,
        "",
        "## CDF CAP versus W1 and drift",
        "",
        "| Model | Endpoint | Contrast | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
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
            "| Model | Pairing contrast | Endpoint | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
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
            "## Equal-model macro contrasts",
            "",
            "| Family | Contrast | Endpoint | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in (*macro_controlled, *macro_pairing):
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['family']} | {row['contrast']} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, "
            f"{scale * _number(row, 'adjusted_ci_high'):+.3f}] | "
            f"{row['classification']} |"
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
            "These are method-matched native-score cross-system comparisons: CDF CAP to paper CAP, forward drift to paper Stable, and W1 to paper W1. They test the full changed validation/selection system, not dataset choice in isolation.",
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
    oas_rows = tables["oas_vs_native_summary.csv"]
    oas_eff = [row for row in oas_rows if row["endpoint"] == "efficiency_250hz"]
    oas_auprc = [row for row in oas_rows if row["endpoint"] == "auprc"]
    lines.extend(
        [
            "",
            f"Masked OAS is directionally higher than native scoring in "
            f"{sum(_number(row, 'mean_delta') > 0 for row in oas_eff)}/{len(oas_eff)} "
            f"efficiency cells and {sum(_number(row, 'mean_delta') > 0 for row in oas_auprc)}/"
            f"{len(oas_auprc)} AUPRC cells. Interpret the confidence intervals above; "
            "these are same-checkpoint score effects, not independently optimized selectors.",
            "",
            "## Reserve-seed decisions",
            "",
            "| Model | Observed seeds | Decision |",
            "|---|---:|---|",
            *[
                f"| {model.upper()} | "
                f"{reserve_decisions[model].get('observed_seed_count', '3')} | "
                f"{reserve_decisions[model]['decision']} |"
                for model in MODELS
            ],
        ]
    )

    random_status = "not_supplied"
    random_receipts: dict[str, str] = {}
    lines.extend(["", "## Random-pairing negative control", ""])
    if discovery_random_analysis is not None:
        discovery_receipt, discovery_tables = _load_random_analysis(
            discovery_random_analysis, "discovery"
        )
        random_receipts["discovery"] = discovery_receipt["content_sha256"]
        mean_associations = [
            row
            for row in discovery_tables["structured_random_rank_associations.csv"]
            if row["random_source"] == "five_pairing_mean"
        ]
        association_index: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
        for row in mean_associations:
            key = (row["model"], row["logical_objective"])
            if row["split"] in association_index.setdefault(key, {}):
                raise ValueError(f"Duplicate random-CAP rank association: {key}.")
            association_index[key][row["split"]] = row
        if any(
            set(split_rows) != {"validate", "test"} for split_rows in association_index.values()
        ):
            raise ValueError("Random-CAP rank association lacks one analysis split.")
        lines.extend(
            [
                "Structured-CAP versus mean-random-CAP checkpoint-rank association on the frozen discovery universe:",
                "",
                "| Model | Structured objective | N | Validation rho | Validation seed-rho range | Test rho | Test seed-rho range |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for (model, objective), split_rows in sorted(association_index.items()):
            validate, test = split_rows["validate"], split_rows["test"]
            if validate["n_checkpoints"] != test["n_checkpoints"]:
                raise ValueError("Random-CAP rank-association split cardinalities differ.")

            def seed_range(row: dict[str, str]) -> str:
                if int(row["pairing_seed_rho_finite_count"]) == 0:
                    return f"undefined ({row['pairing_seed_rho_undefined_count']}/5)"
                return (
                    f"[{float(row['pairing_seed_rho_min']):.3f}, "
                    f"{float(row['pairing_seed_rho_max']):.3f}] "
                    f"({row['pairing_seed_rho_undefined_count']}/5 undefined)"
                )

            lines.append(
                f"| {model.upper()} | {objective} | {validate['n_checkpoints']} | "
                f"{validate['spearman_rho'] or 'undefined'} | "
                f"{seed_range(validate)} | {test['spearman_rho'] or 'undefined'} | "
                f"{seed_range(test)} |"
            )
        random_status = (
            f"discovery authenticated ({discovery_receipt['method']['seed_rows']} seed rows)"
        )
    if confirmation_random_analysis is not None:
        confirmation_receipt, confirmation_tables = _load_random_analysis(
            confirmation_random_analysis,
            "confirmation",
            confirmation_provenance=_confirmation_provenance(manifest),
        )
        random_receipts["confirmation"] = confirmation_receipt["content_sha256"]
        variability_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in confirmation_tables["random_cap_variability.csv"]:
            variability_groups.setdefault(
                (row["model"], row["evaluation_score"], row["split"]), []
            ).append(row)
        lines.extend(
            [
                "",
                "Five-random-bijection variability on sealed confirmation checkpoints:",
                "",
                "| Model | Score | Split | Checkpoint memberships | Median random-CAP SD | Maximum range |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for (model, score, split), rows in sorted(variability_groups.items()):
            standard_deviations = [_number(row, "cap_sample_std") for row in rows]
            ranges = [_number(row, "cap_range") for row in rows]
            lines.append(
                f"| {model.upper()} | {score} | {split} | {len(rows)} | "
                f"{statistics.median(standard_deviations):.4g} | {max(ranges):.4g} |"
            )
        random_status = (
            f"{random_status}; confirmation authenticated "
            f"({confirmation_receipt['method']['seed_rows']} seed rows)"
        )
    lines.extend(
        [
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
        "cdf_cap_efficiency_directional_positive": cap_eff[0],
        "cdf_cap_efficiency_meaningful": cap_eff[1],
        "cdf_cap_efficiency_total": cap_eff[2],
        "cdf_cap_auprc_directional_positive": cap_auprc[0],
        "cdf_cap_auprc_meaningful": cap_auprc[1],
        "cdf_cap_auprc_total": cap_auprc[2],
        "learned_pairing_efficiency_directional_positive": pairing_eff[0],
        "learned_pairing_efficiency_meaningful": pairing_eff[1],
        "learned_pairing_efficiency_total": pairing_eff[2],
        "learned_pairing_auprc_directional_positive": pairing_auprc[0],
        "learned_pairing_auprc_meaningful": pairing_auprc[1],
        "learned_pairing_auprc_total": pairing_auprc[2],
        "reserve_seed_decisions": {
            model: reserve_decisions[model]["decision"] for model in MODELS
        },
        "random_cap_status": random_status,
        "random_cap_analysis_receipts": random_receipts,
    }
    summary["content_sha256"] = _content_hash(summary)
    return "\n".join(lines), summary


def write_confirmation_report(
    analysis_dir: Path,
    repository: Path,
    output_dir: Path,
    *,
    discovery_random_analysis: Path | None = None,
    confirmation_random_analysis: Path | None = None,
) -> dict[str, Any]:
    """Persist final Markdown and JSON atomically without overwriting different results."""
    report, summary = build_confirmation_report(
        analysis_dir,
        repository,
        discovery_random_analysis=discovery_random_analysis,
        confirmation_random_analysis=confirmation_random_analysis,
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
