"""Public API for the leakage-probe evaluation package."""

from .baselines import (
    evaluate_dummy_baseline_representation,
    evaluate_primary_dummy_baselines,
    fit_dummy_baseline,
)
from .constants import (
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    MLP_PROBE_CONFIG,
    PRIMARY_PROBE_REPRESENTATIONS,
    PROBE_INITIALIZATION_SEEDS,
    PROBE_INNER_SPLIT_SEED,
    PROBE_INNER_VALIDATION_FRACTION,
    PROBE_REPRESENTATION_METRIC_NAMES,
    PROBE_TARGET_SHUFFLE_SEED,
)
from .diagnostics import (
    evaluate_shuffled_target_mlp_controls,
    make_shuffled_training_target,
)
from .errors import ProbeExtractionError, ProbeFitError, ProbePartitionError
from .evaluation import evaluate_four_leakage_probes
from .extraction import extract_probe_split
from .linear import (
    evaluate_linear_probe_representation,
    evaluate_primary_linear_probes,
    fit_linear_probe,
)
from .mlp import (
    AllMLPProbeCandidatesFailed,
    evaluate_mlp_probe_representation,
    evaluate_primary_mlp_probes,
    fit_mlp_probe_candidate,
    refit_selected_mlp_probe,
    select_mlp_probe_seed,
)
from .partition import make_probe_inner_partition
from .persistence import (
    evaluate_and_record_loss_total_leakage_probes,
    evaluate_and_write_loss_total_leakage_probes,
    leakage_probe_output_path,
    log_leakage_probe_outcome_metadata,
    write_invalid_leakage_probe_result,
    write_leakage_probe_results,
)
from .serialization import (
    four_probe_metric_values,
    four_probe_result_payload,
    log_four_probe_metrics,
    log_shuffled_target_metrics,
    shuffled_target_metric_values,
)

from .types import (
    DummyBaselineOuterResult,
    FourProbeEvaluationResult,
    LeakageProbeRunOutcome,
    LinearProbeOuterResult,
    MLPProbeCandidateFailure,
    MLPProbeCandidateResult,
    MLPProbeOuterResult,
    MLPProbeSeedSelection,
    NamedDummyBaselineResult,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
    PrimaryDummyBaselineResult,
    PrimaryLinearProbeResult,
    PrimaryMLPLeakageResult,
    ProbeInnerPartition,
    ProbeRepresentationSet,
    ShuffledTargetMLPResult,
    ShuffledTrainingTarget,
)

__all__ = [
    "AllMLPProbeCandidatesFailed",
    "DummyBaselineOuterResult",
    "FourProbeEvaluationResult",
    "LEAKAGE_PROBE_PROTOCOL_VERSION",
    "LeakageProbeRunOutcome",
    "LinearProbeOuterResult",
    "log_shuffled_target_metrics",
    "MLPProbeCandidateFailure",
    "MLPProbeCandidateResult",
    "MLPProbeOuterResult",
    "MLPProbeSeedSelection",
    "MLP_PROBE_CONFIG",
    "NamedDummyBaselineResult",
    "NamedLinearProbeResult",
    "NamedMLPProbeResult",
    "PRIMARY_PROBE_REPRESENTATIONS",
    "PROBE_INITIALIZATION_SEEDS",
    "PROBE_INNER_SPLIT_SEED",
    "PROBE_INNER_VALIDATION_FRACTION",
    "PROBE_REPRESENTATION_METRIC_NAMES",
    "PROBE_TARGET_SHUFFLE_SEED",
    "PrimaryDummyBaselineResult",
    "PrimaryLinearProbeResult",
    "PrimaryMLPLeakageResult",
    "ProbeExtractionError",
    "ProbeFitError",
    "ProbeInnerPartition",
    "ProbePartitionError",
    "ProbeRepresentationSet",
    "shuffled_target_metric_values",
    "ShuffledTargetMLPResult",
    "ShuffledTrainingTarget",
    "evaluate_and_record_loss_total_leakage_probes",
    "evaluate_and_write_loss_total_leakage_probes",
    "evaluate_dummy_baseline_representation",
    "evaluate_four_leakage_probes",
    "evaluate_linear_probe_representation",
    "evaluate_mlp_probe_representation",
    "evaluate_primary_dummy_baselines",
    "evaluate_primary_linear_probes",
    "evaluate_primary_mlp_probes",
    "evaluate_shuffled_target_mlp_controls",
    "extract_probe_split",
    "fit_dummy_baseline",
    "fit_linear_probe",
    "fit_mlp_probe_candidate",
    "four_probe_metric_values",
    "four_probe_result_payload",
    "leakage_probe_output_path",
    "log_four_probe_metrics",
    "log_leakage_probe_outcome_metadata",
    "make_probe_inner_partition",
    "make_shuffled_training_target",
    "refit_selected_mlp_probe",
    "select_mlp_probe_seed",
    "write_invalid_leakage_probe_result",
    "write_leakage_probe_results",
]
