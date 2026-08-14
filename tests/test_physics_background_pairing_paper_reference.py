"""Contracts for the machine-readable physics paper reference artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "research" / "physics_background_pairing" / "paper_reference.json"
MODELS = {"ae", "vae", "dsae", "dsvae", "svdd", "realnvp"}
METHODS = {"semi", "cap", "stable", "wasserstein"}


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 identity of a local source artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_physics_paper_reference_is_complete_and_source_authenticated() -> None:
    """The reference must be complete, finite, and tied to its notebook sources."""
    artifact = json.loads(REFERENCE.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["units"]["efficiency_means"] == "percent"
    assert set(artifact["method_semantics"]) == METHODS

    sources = {row["path"]: row for row in artifact["source_artifacts"]}
    assert set(sources) == {"notebooks/effs.nb", "notebooks/corrs.nb"}
    for relative_path, source in sources.items():
        assert _sha256(ROOT / relative_path) == source["sha256"]
        assert len(source["git_blob"]) == 40

    efficiencies = artifact["efficiency_means_percent"]
    assert set(efficiencies) == {"250hz", "q99"}
    for operating_point in efficiencies.values():
        assert set(operating_point) == MODELS
        for values in operating_point.values():
            assert set(values) == METHODS
            assert all(
                value is None or (math.isfinite(value) and 0.0 <= value <= 100.0)
                for value in values.values()
            )

    assert efficiencies["250hz"]["dsae"]["cap"] == 8.55
    assert efficiencies["q99"]["vae"]["cap"] == 73.52
    assert efficiencies["q99"]["svdd"]["semi"] is None
    assert efficiencies["q99"]["svdd"]["wasserstein"] is None
    assert {row["field"] for row in artifact["missing_values"]} == {
        "efficiency_means_percent.q99.svdd.semi",
        "efficiency_means_percent.q99.svdd.wasserstein",
    }

    correlations = artifact["epoch_metric_downstream_correlations"]["values"]
    assert set(correlations) == MODELS
    for values in correlations.values():
        assert set(values) == METHODS
        assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values.values())
    assert correlations["dsvae"]["cap"] < 0.0
    assert correlations["vae"]["wasserstein"] < 0.0

    trials = artifact["paper_selected_trial_numbers"]
    assert set(trials) == MODELS
    assert all(
        set(values) == METHODS
        and all(isinstance(value, int) and value >= 0 for value in values.values())
        for values in trials.values()
    )
    assert artifact["comparison_caveats"]
