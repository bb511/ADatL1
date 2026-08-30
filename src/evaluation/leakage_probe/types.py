"""Typed data exchanged between leakage-probe stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class ProbeRepresentationSet:
    """Representations and physical sensitive targets for one AE split."""

    split: str
    latent_logits: np.ndarray
    latent_sample: np.ndarray
    reconstructed_data: np.ndarray
    sensitive_target: np.ndarray
    n_events: int
    sample_seed: int
    max_samples: int | None
    manifest_hash: str


@dataclass(frozen=True)
class ProbeInnerPartition:
    """Indices dividing AE training events into probe fit and validation."""

    fit_indices: np.ndarray
    validation_indices: np.ndarray
    seed: int
    validation_fraction: float
    manifest_hash: str


@dataclass(frozen=True)
class ShuffledTrainingTarget:
    """One deterministic training-target permutation."""

    values: np.ndarray
    permutation_indices: np.ndarray
    seed: int
    manifest_hash: str


@dataclass(frozen=True)
class MLPProbeCandidateResult:
    """Fitted MLP and its inner-validation diagnostics."""

    seed: int
    inner_r2_raw: float
    inner_mae_gev: float
    convergence_warnings: tuple[str, ...]
    n_iter: int
    final_loss: float
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    estimator: MLPRegressor


@dataclass(frozen=True)
class MLPProbeCandidateFailure:
    """Recorded failure of one probe initialization."""

    seed: int
    reason: str
    message: str


@dataclass(frozen=True)
class MLPProbeSeedSelection:
    """Result of selecting one MLP initialization using inner R2."""

    selected_seed: int
    selected_candidate: MLPProbeCandidateResult
    successful_candidates: tuple[MLPProbeCandidateResult, ...]
    failed_candidates: tuple[MLPProbeCandidateFailure, ...]


@dataclass(frozen=True)
class MLPProbeOuterResult:
    """Fresh selected-seed MLP evaluated on outer validation."""

    selected_seed: int
    outer_r2_raw: float
    outer_r2_clipped: float
    outer_mae_gev: float
    convergence_warnings: tuple[str, ...]
    n_iter: int
    final_loss: float
    n_train: int
    n_validation: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    estimator: MLPRegressor


@dataclass(frozen=True)
class NamedMLPProbeResult:
    """Complete MLP procedure for one named representation."""

    representation_name: str
    metric_name: str
    feature_dimension: int
    seed_selection: MLPProbeSeedSelection
    outer_result: MLPProbeOuterResult


@dataclass(frozen=True)
class PrimaryMLPLeakageResult:
    """Primary latent and reconstruction leakage measurement."""

    latent_logits: NamedMLPProbeResult
    reconstructed_data: NamedMLPProbeResult
    inner_partition: ProbeInnerPartition
    leakage_worst: float

@dataclass(frozen=True)
class ShuffledTargetMLPResult:
    """MLP negative controls fitted to shuffled training targets."""

    latent_logits: NamedMLPProbeResult
    reconstructed_data: NamedMLPProbeResult
    inner_partition: ProbeInnerPartition
    shuffle_seed: int
    permutation_manifest_hash: str


@dataclass(frozen=True)
class LinearProbeOuterResult:
    """Linear regression evaluated on outer validation."""

    outer_r2_raw: float
    outer_r2_clipped: float
    outer_mae_gev: float
    n_train: int
    n_validation: int
    feature_scaler: StandardScaler
    estimator: LinearRegression


@dataclass(frozen=True)
class NamedLinearProbeResult:
    """Linear probe for one named AE representation."""

    representation_name: str
    metric_name: str
    feature_dimension: int
    outer_result: LinearProbeOuterResult


@dataclass(frozen=True)
class PrimaryLinearProbeResult:
    """Independent latent and reconstruction linear probes."""

    latent_logits: NamedLinearProbeResult
    reconstructed_data: NamedLinearProbeResult

@dataclass(frozen=True)
class FourProbeEvaluationResult:
    """Four primary probes, diagnostics, and global leakage."""

    mlp_latent_logits: NamedMLPProbeResult
    mlp_reconstructed_data: NamedMLPProbeResult
    linear_latent_logits: NamedLinearProbeResult
    linear_reconstructed_data: NamedLinearProbeResult
    shuffled_target_controls: ShuffledTargetMLPResult | None
    inner_partition: ProbeInnerPartition
    worst_probe: str
    leakage_worst: float


@dataclass(frozen=True)
class LeakageProbeRunOutcome:
    """Recorded outcome of one complete leakage evaluation."""

    probe_valid: bool
    result: FourProbeEvaluationResult | None
    output_path: Path
    rejection_reason: str | None
    rejection_message: str | None
    diagnostic_result: FourProbeEvaluationResult | None = None
