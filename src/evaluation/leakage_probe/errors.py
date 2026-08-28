"""Domain errors raised by leakage-probe evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import SHUFFLED_TARGET_R2_CLIPPED_MAX

if TYPE_CHECKING:
    from .types import FourProbeEvaluationResult


class ProbeExtractionError(RuntimeError):
    """Failure to construct a scientifically valid probe dataset."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ProbePartitionError(ValueError):
    """Failure to construct the fixed inner probe partition."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ProbeFitError(RuntimeError):
    """Failure while fitting or evaluating one probe candidate."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ShuffledTargetGuardrailError(ProbeFitError):
    """A completed evaluation whose shuffled control is too strong."""

    def __init__(
        self,
        result: FourProbeEvaluationResult,
        failed_controls: tuple[str, ...],
    ) -> None:
        self.result = result
        self.failed_controls = failed_controls

        controls = result.shuffled_target_controls
        scores = {
            "z_logits": float(
                controls.latent_logits
                .outer_result.outer_r2_clipped
            ),
            "reconstruction": float(
                controls.reconstructed_data
                .outer_result.outer_r2_clipped
            ),
        }

        self.failed_scores = tuple(
            (name, scores[name])
            for name in failed_controls
        )

        formatted_scores = ", ".join(
            f"{name}={score:.6f}"
            for name, score in self.failed_scores
        )

        super().__init__(
            "shuffled_target_guardrail_failed",
            "Shuffled-target clipped R2 exceeded "
            f"{SHUFFLED_TARGET_R2_CLIPPED_MAX:.6f}: "
            f"{formatted_scores}.",
        )