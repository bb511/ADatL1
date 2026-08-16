from src.evaluation.physics_background_pairing_confirmation_targets import (
    MAPPING_ORDER,
    build_confirmation_target_preview,
)
from src.evaluation.physics_background_pairing_paper_comparison import MODELS


def _analysis() -> dict:
    """Return a complete synthetic eight-family CAP result universe."""
    rows = []
    index = 0
    for model in MODELS:
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        for score in scores:
            for strategy_index, strategy in enumerate((*MAPPING_ORDER, "cdf")):
                rows.append(
                    {
                        "model": model,
                        "score": score,
                        "metric": "cap",
                        "strategy": strategy,
                        "logical_objective": f"{score}__cap__{strategy}",
                        "membership_index": index,
                        "trial_number": 100 + index,
                        "checkpoint": f"/{index}.ckpt",
                        "checkpoint_sha256": f"{index:064x}",
                        "mean_efficiency": 0.2 if strategy == "jetclr" else 0.1,
                        "bottom_quartile_cvar": 0.02 + strategy_index / 1000,
                        "worst_efficiency": 0.0,
                    }
                )
                index += 1
    return {
        "method": {
            "namespace": "provisional_search50",
            "reported_choices_use_downstream_anomalies": True,
        },
        "front_winners": rows,
    }


def test_native_preview_has_six_mapping_control_pairs() -> None:
    """Native scope must create one learned/control pair for every model."""
    preview = build_confirmation_target_preview(_analysis(), scope="native")
    assert len(preview["targets"]) == 12
    assert {row["score"] for row in preview["targets"]} == {"native"}
    assert [row["role"] for row in preview["targets"]].count("best_mapping") == 6
    assert all(row["directions"] == ["maximize", "minimize"] for row in preview["targets"])
    assert all(len(row["optimized_metric_config"]) == 3 for row in preview["targets"])


def test_all_score_preview_has_eight_mapping_control_pairs() -> None:
    """All-score scope must add the two AE/VAE OAS family pairs."""
    preview = build_confirmation_target_preview(_analysis(), scope="all")
    assert len(preview["targets"]) == 16
    assert sum(row["score"] == "residual_oas" for row in preview["targets"]) == 4
    mapping = [row for row in preview["targets"] if row["role"] == "best_mapping"]
    assert {row["strategy"] for row in mapping} == {"jetclr"}
    assert preview["strict_motpe_requires_n_jobs"] == 1
    assert preview["uses_downstream_for_target_selection"] is True


def test_preview_fails_closed_on_missing_control() -> None:
    """A missing CDF control must prevent target preview construction."""
    analysis = _analysis()
    analysis["front_winners"] = [
        row
        for row in analysis["front_winners"]
        if not (row["model"] == "ae" and row["score"] == "native" and row["strategy"] == "cdf")
    ]
    try:
        build_confirmation_target_preview(analysis, scope="native")
    except ValueError as error:
        assert "coverage differs" in str(error)
    else:
        raise AssertionError("Missing CDF control was accepted.")
