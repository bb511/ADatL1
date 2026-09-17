"""Public API for the leakage-probe evaluation package."""

from .aggregation import (
    ProbeAggregationError,
    aggregate_paired_seed_leakage,
    write_paired_seed_leakage_aggregate,
)

from .constants import (
    LEAKAGE_PROBE_EVALUATION_MODES,
    LEAKAGE_PROBE_INVALID_RUN_POLICY,
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    MLP_PROBE_CONFIG,
    PRIMARY_PROBE_REPRESENTATIONS,
    PROBE_INITIALIZATION_SEED,
    PROBE_EVENT_SAMPLE_SEED,
    PROBE_REPRESENTATION_METRIC_NAMES,
    PROBE_TARGET_SHUFFLE_SEED,
    SHUFFLED_TARGET_R2_CLIPPED_MAX,
)
from .diagnostics import (
    enforce_shuffled_target_guardrail,
    evaluate_shuffled_target_mlp_controls,
    make_shuffled_training_target,
    shuffled_target_guardrail_failures,
)
from .errors import (
    ProbeExtractionError,
    ProbeFitError,
    ShuffledTargetGuardrailError,
)
from .evaluation import evaluate_four_leakage_probes
from .extraction import extract_probe_split
from .linear import (
    evaluate_linear_probe_representation,
    evaluate_primary_linear_probes,
    fit_linear_probe,
)
from .mlp import (
    fit_mlp_probe,
    evaluate_mlp_probe_representation,
    evaluate_primary_mlp_probes,
)
from .provenance import (
    concatenate_probe_representation_sets,
    leakage_probe_configuration_id,
    make_leakage_probe_run_metadata,
    make_probe_evaluation_context,
    probe_split_provenance,
)
from .persistence import (
    evaluate_and_record_loss_total_leakage_probes,
    evaluate_and_write_loss_total_leakage_probes,
    leakage_probe_output_path,
    leakage_probe_summary_output_path,
    log_leakage_probe_outcome_metadata,
    write_invalid_leakage_probe_result,
    write_leakage_probe_results,
)
from .serialization import (
    four_probe_metric_values,
    four_probe_result_payload,
    four_probe_summary_payload,
    log_four_probe_metrics,
    log_shuffled_target_metrics,
    shuffled_target_metric_values,
)

from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunOutcome,
    LeakageProbeRunMetadata,
    LinearProbeOuterResult,
    MLPProbeOuterResult,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
    PrimaryLinearProbeResult,
    PrimaryMLPLeakageResult,
    ProbeEvaluationContext,
    ProbeRepresentationSet,
    ProbeSplitProvenance,
    ShuffledTargetMLPResult,
    ShuffledTrainingTarget,
)

__all__ = [
    "ProbeAggregationError",
    "aggregate_paired_seed_leakage",
    "FourProbeEvaluationResult",
    "LEAKAGE_PROBE_PROTOCOL_VERSION",
    "LEAKAGE_PROBE_EVALUATION_MODES",
    "LEAKAGE_PROBE_INVALID_RUN_POLICY",
    "LeakageProbeRunOutcome",
    "LeakageProbeRunMetadata",
    "LinearProbeOuterResult",
    "log_shuffled_target_metrics",
    "MLPProbeOuterResult",
    "MLP_PROBE_CONFIG",
    "NamedLinearProbeResult",
    "NamedMLPProbeResult",
    "PRIMARY_PROBE_REPRESENTATIONS",
    "PROBE_INITIALIZATION_SEED",
    "PROBE_EVENT_SAMPLE_SEED",
    "PROBE_REPRESENTATION_METRIC_NAMES",
    "PROBE_TARGET_SHUFFLE_SEED",
    "PrimaryLinearProbeResult",
    "PrimaryMLPLeakageResult",
    "ProbeExtractionError",
    "ProbeEvaluationContext",
    "ProbeFitError",
    "ProbeRepresentationSet",
    "ProbeSplitProvenance",
    "shuffled_target_metric_values",
    "ShuffledTargetMLPResult",
    "ShuffledTrainingTarget",
    "evaluate_and_record_loss_total_leakage_probes",
    "evaluate_and_write_loss_total_leakage_probes",
    "evaluate_four_leakage_probes",
    "evaluate_linear_probe_representation",
    "evaluate_mlp_probe_representation",
    "evaluate_primary_linear_probes",
    "evaluate_primary_mlp_probes",
    "fit_mlp_probe",
    "evaluate_shuffled_target_mlp_controls",
    "extract_probe_split",
    "concatenate_probe_representation_sets",
    "fit_linear_probe",
    "four_probe_metric_values",
    "four_probe_result_payload",
    "four_probe_summary_payload",
    "leakage_probe_output_path",
    "leakage_probe_summary_output_path",
    "log_four_probe_metrics",
    "log_leakage_probe_outcome_metadata",
    "leakage_probe_configuration_id",
    "make_leakage_probe_run_metadata",
    "make_probe_evaluation_context",
    "probe_split_provenance",
    "make_shuffled_training_target",
    "write_invalid_leakage_probe_result",
    "write_paired_seed_leakage_aggregate",
    "write_leakage_probe_results",
    "SHUFFLED_TARGET_R2_CLIPPED_MAX",
    "ShuffledTargetGuardrailError",
    "enforce_shuffled_target_guardrail",
    "shuffled_target_guardrail_failures",
]
