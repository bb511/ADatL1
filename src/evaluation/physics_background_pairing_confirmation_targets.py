"""Build deterministic Stage-2 confirmation-target previews without freezing state."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.evaluation.physics_background_pairing_paper_comparison import MODELS

MAPPING_ORDER = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
)


def _canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible payload deterministically."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _objective_config(score: str, strategy: str) -> dict[str, Any]:
    """Return the exact two-objective evaluator configuration for one CAP target."""
    metric = f"cap_{score}_{strategy}"
    return {
        "primary_checkpoint_metric": (f"{metric}_ema_ZB_run396102_vs_ZB_run398183"),
        "main_metric": {
            "callback": {"name": metric, "params": {}},
            "direction": "maximize",
        },
        "sec_metric": {
            "callback": {
                "name": "ascore_operational",
                "params": {"ckpt_name": None, "test_ds": "normal"},
            },
            "direction": "minimize",
        },
    }


def _families(scope: str) -> tuple[tuple[str, str], ...]:
    """Return the fixed family universe for a native-only or all-score preview."""
    if scope not in {"native", "all"}:
        raise ValueError("Confirmation preview scope must be 'native' or 'all'.")
    return tuple(
        (model, score)
        for model in MODELS
        for score in (
            ("native", "residual_oas")
            if scope == "all" and model in {"ae", "vae"}
            else ("native",)
        )
    )


def build_confirmation_target_preview(analysis: dict[str, Any], *, scope: str) -> dict[str, Any]:
    """Select best learned CAP mapping and CDF control in every requested family.

    This function deliberately returns a preview. Freezing or launching the payload is a separate
    authorized transition because the 12-versus-16 study choice changes scientific coverage and
    substantial compute cost.
    """
    method = analysis.get("method", {})
    if (
        method.get("namespace") != "provisional_search50"
        or method.get("reported_choices_use_downstream_anomalies") is not True
    ):
        raise ValueError("Expected the authenticated downstream-oracle search50 analysis.")
    rows = analysis.get("front_winners", [])
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for family in _families(scope):
        family_rows = [
            row
            for row in rows
            if (row.get("model"), row.get("score")) == family and row.get("metric") == "cap"
        ]
        strategies = {str(row.get("strategy")) for row in family_rows}
        if strategies != {*MAPPING_ORDER, "cdf"}:
            raise ValueError(f"CAP strategy coverage differs for family {family}.")
        indexed[family] = family_rows

    targets = []
    for family in _families(scope):
        family_rows = indexed[family]
        mappings = [row for row in family_rows if row["strategy"] in MAPPING_ORDER]
        mapping = min(
            mappings,
            key=lambda row: (
                -float(row["mean_efficiency"]),
                -float(row["bottom_quartile_cvar"]),
                -float(row["worst_efficiency"]),
                MAPPING_ORDER.index(row["strategy"]),
                int(row["membership_index"]),
            ),
        )
        cdf = next(row for row in family_rows if row["strategy"] == "cdf")
        for role, selected in (("best_mapping", mapping), ("cdf_control", cdf)):
            objective = f"{selected['score']}__cap__{selected['strategy']}"
            target = {
                "target_index": len(targets),
                "role": role,
                "model": selected["model"],
                "score": selected["score"],
                "metric": "cap",
                "strategy": selected["strategy"],
                "logical_objective": objective,
                "screening_membership_index": int(selected["membership_index"]),
                "screening_trial_number": int(selected["trial_number"]),
                "screening_checkpoint": selected["checkpoint"],
                "screening_checkpoint_sha256": selected["checkpoint_sha256"],
                "screening_mean_efficiency": float(selected["mean_efficiency"]),
                "screening_bottom_quartile_cvar": float(selected["bottom_quartile_cvar"]),
                "optimized_metric_config": _objective_config(
                    selected["score"], selected["strategy"]
                ),
                "directions": ["maximize", "minimize"],
                "hparams_search": f"{selected['model']}_optuna",
                "sampler": "motpe_paper_ae" if selected["model"] == "ae" else "motpe_paper",
                "search_trials": 600,
                "search_epochs": 50,
            }
            targets.append(target)

    expected = 12 if scope == "native" else 16
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} confirmation targets, found {len(targets)}.")
    payload = {
        "schema_version": 1,
        "kind": "unfrozen_confirmation_target_preview",
        "scope": scope,
        "target_rule": "best downstream-mean mapping CAP plus CDF CAP control per family",
        "uses_downstream_for_target_selection": True,
        "target_selection_bias": (
            "Targets use the same 20 anomalies later used for reporting; Stage 2 is exploratory."
        ),
        "strict_motpe_requires_n_jobs": 1,
        "targets": targets,
    }
    payload["content_sha256"] = _canonical_hash(payload)
    return payload


__all__ = ["build_confirmation_target_preview"]
