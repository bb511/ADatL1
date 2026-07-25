"""Analytical experiments for Section 3 of ``theory.tex``.

The experiments instantiate the Gaussian linear anomaly model with a stable
nuisance direction:

    X^(d) = sqrt(lambda_Z) Z e0 + sqrt(lambda_U) U e1 + xi^(d).

Future anomalies shift the latent anomaly coordinate Z, so the observed anomaly
mean is delta * sqrt(lambda_Z) * e0. Candidate anomaly scores are normalized
linear projections

    S_w(x) = w^T x,     ||w||_2 = 1,     w^T e0 >= 0.

Section 3 separates anomaly-aligned reliability rho_Z from total reproducible
reliability rho_R. CAP is monotone in rho_R; TPR is monotone in rho_Z. CAP
therefore recovers the anomaly direction when the maximizer of rho_R is also a
maximizer of rho_Z, for example when lambda_Z > ||Gamma_U||_op.

This script generates the corresponding finite-sample evidence and plots.

Example:
    uv run python src/analytical.py --output-dir figures/section3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.spatial import cKDTree
from scipy.stats import chi2, ncx2, norm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.evaluation.callbacks.metrics.cap.kernel import ApproximationCapacityKernel


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    output_dir: Path
    n_features: int
    n_pairs: int
    n_test: int
    n_channel_pairs: int
    seed: int
    fpr: float
    anomaly_shift: float
    lambda_z: float
    lambda_u: float
    benign_shift: float
    n_match_features: int
    match_noise: float
    ratio_sweep_lambda_z: float
    ratio_min: float
    ratio_max: float
    n_ratios: int
    n_angles: int
    n_rhos: int
    beta_max: float
    n_betas: int
    quadrature_order: int


@dataclass(frozen=True)
class ScoreCandidate:
    name: str
    family: str
    description: str
    theorem_scope: str
    weights: tuple[float, ...] = ()
    dims: tuple[int, ...] = ()


PAPER_DEFAULTS = {
    "n_features": 6,
    "n_pairs": 60_000,
    "n_test": 150_000,
    "n_channel_pairs": 60_000,
    "seed": 123,
    "fpr": 1e-3,
    "anomaly_shift": 3.5,
    "lambda_z": 0.90,
    "lambda_u": 0.12,
    "benign_shift": 1.0,
    "n_match_features": 8,
    "match_noise": 0.05,
    "ratio_sweep_lambda_z": 0.45,
    "ratio_min": 0.05,
    "ratio_max": 2.0,
    "n_ratios": 81,
    "n_angles": 81,
    "n_rhos": 31,
    "beta_max": 8.0,
    "n_betas": 81,
    "quadrature_order": 45,
}

SMOKE_DEFAULTS = {
    **PAPER_DEFAULTS,
    "n_pairs": 800,
    "n_test": 4_000,
    "n_channel_pairs": 1_200,
    "n_match_features": 4,
    "n_ratios": 9,
    "n_angles": 9,
    "n_rhos": 7,
    "n_betas": 17,
    "quadrature_order": 21,
}

ARTIFACT_FILENAMES = (
    "channel_reliability.csv",
    "linear_direction_sweep.csv",
    "marginal_shift_trap.csv",
    "marginal_shift_selector_sweep.csv",
    "score_family_summary.csv",
    "alignment_assumption_check.csv",
    "alignment_ratio_sweep.csv",
    "score_distribution_summary.csv",
    "theory_cap_tpr_vs_reliability.png",
    "score_pair_scatter.png",
    "feature_space_geometry.png",
    "population_metric_landscape.png",
    "linear_direction_sweep.png",
    "cap_vs_tpr.png",
    "marginal_shift_trap.png",
    "marginal_shift_selector_sweep.png",
    "score_distributions.png",
    "score_family_comparison.png",
    "score_family_distributions.png",
    "alignment_assumption_check.png",
    "alignment_ratio_sweep.png",
)


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    profile_args, _ = profile_parser.parse_known_args(argv)
    defaults = PAPER_DEFAULTS if profile_args.profile == "paper" else SMOKE_DEFAULTS

    parser = argparse.ArgumentParser(
        description="Run Section 3 Gaussian linear CAP synthetic experiments."
    )
    parser.add_argument(
        "--profile",
        choices=("paper", "smoke"),
        default=profile_args.profile,
        help="Use publication-scale or fast CI/smoke defaults.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/section3"))
    parser.add_argument("--n-features", type=int, default=defaults["n_features"])
    parser.add_argument("--n-pairs", type=int, default=defaults["n_pairs"])
    parser.add_argument("--n-test", type=int, default=defaults["n_test"])
    parser.add_argument("--n-channel-pairs", type=int, default=defaults["n_channel_pairs"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--fpr", type=float, default=defaults["fpr"])
    parser.add_argument("--anomaly-shift", type=float, default=defaults["anomaly_shift"])
    parser.add_argument("--lambda-z", dest="lambda_z", type=float, default=defaults["lambda_z"])
    parser.add_argument("--lambda-u", dest="lambda_u", type=float, default=defaults["lambda_u"])
    parser.add_argument("--benign-shift", type=float, default=defaults["benign_shift"])
    parser.add_argument("--n-match-features", type=int, default=defaults["n_match_features"])
    parser.add_argument("--match-noise", type=float, default=defaults["match_noise"])
    parser.add_argument(
        "--ratio-sweep-lambda-z",
        type=float,
        default=defaults["ratio_sweep_lambda_z"],
    )
    parser.add_argument("--ratio-min", type=float, default=defaults["ratio_min"])
    parser.add_argument("--ratio-max", type=float, default=defaults["ratio_max"])
    parser.add_argument("--n-ratios", type=int, default=defaults["n_ratios"])
    parser.add_argument("--n-angles", type=int, default=defaults["n_angles"])
    parser.add_argument("--n-rhos", type=int, default=defaults["n_rhos"])
    parser.add_argument("--beta-max", type=float, default=defaults["beta_max"])
    parser.add_argument("--n-betas", type=int, default=defaults["n_betas"])
    parser.add_argument("--quadrature-order", type=int, default=defaults["quadrature_order"])
    args = parser.parse_args(argv)

    if args.n_features < 2:
        raise ValueError("n_features must be at least 2.")
    if not 0.0 < args.fpr < 1.0:
        raise ValueError("fpr must lie in (0, 1).")
    if not 0.0 < args.lambda_z <= 0.999:
        raise ValueError("Require 0 < lambda_z <= 0.999.")
    if not 0.0 <= args.lambda_u <= 0.999:
        raise ValueError("Require 0 <= lambda_u <= 0.999.")
    if args.n_angles < 3:
        raise ValueError("n_angles must be at least 3.")
    if args.n_rhos < 3:
        raise ValueError("n_rhos must be at least 3.")
    if args.beta_max <= 0.0 or args.n_betas < 2:
        raise ValueError("beta grid must be nontrivial.")
    if args.quadrature_order < 10:
        raise ValueError("quadrature_order should be at least 10.")
    if args.n_match_features < 1:
        raise ValueError("n_match_features must be positive.")
    if args.match_noise < 0.0:
        raise ValueError("match_noise must be non-negative.")
    if not 0.0 < args.ratio_sweep_lambda_z < 1.0:
        raise ValueError("ratio_sweep_lambda_z must lie in (0, 1).")
    if args.ratio_min <= 0.0 or args.ratio_max <= args.ratio_min:
        raise ValueError("Require 0 < ratio_min < ratio_max.")
    if args.ratio_sweep_lambda_z * args.ratio_max >= 1.0:
        raise ValueError("ratio_sweep_lambda_z * ratio_max must be less than 1.")
    if args.n_ratios < 3:
        raise ValueError("n_ratios must be at least 3.")

    return ExperimentConfig(**vars(args))


def beta_grid(config: ExperimentConfig) -> np.ndarray:
    return np.linspace(0.0, config.beta_max, config.n_betas, dtype=np.float64)


def validation_correlations(config: ExperimentConfig) -> np.ndarray:
    correlations = np.full(config.n_features, config.lambda_u, dtype=np.float64)
    correlations[0] = config.lambda_z
    return correlations


def projection_directions(config: ExperimentConfig) -> pd.DataFrame:
    angles = np.linspace(0.0, 0.5 * np.pi, config.n_angles, dtype=np.float64)
    c = np.cos(angles)
    s = np.sin(angles)
    rho_z = config.lambda_z * c**2
    rho_u = config.lambda_u * s**2
    rho_r = rho_z + rho_u
    pi = np.divide(rho_z, rho_r, out=np.zeros_like(rho_z), where=rho_r > 0.0)
    return pd.DataFrame(
        {
            "angle_rad": angles,
            "angle_deg": np.degrees(angles),
            "cos_to_anomaly": c,
            "sin_to_noise": s,
            "rho_z": rho_z,
            "rho_u": rho_u,
            "rho_r": rho_r,
            "alignment_pi": pi,
            "rho": rho_r,
        }
    )


def make_direction_matrix(curve: pd.DataFrame, n_features: int) -> np.ndarray:
    weights = np.zeros((len(curve), n_features), dtype=np.float64)
    weights[:, 0] = curve["cos_to_anomaly"].to_numpy()
    weights[:, 1] = curve["sin_to_noise"].to_numpy()
    return weights


def build_score_candidates(config: ExperimentConfig) -> list[ScoreCandidate]:
    d = config.n_features
    inv_sqrt2 = float(1.0 / np.sqrt(2.0))
    return [
        ScoreCandidate(
            name="linear_oracle_w_star",
            family="linear",
            description="Likelihood-ratio direction S=x0.",
            theorem_scope="inside_section3",
            weights=(1.0,),
        ),
        ScoreCandidate(
            name="linear_mixed_45deg",
            family="linear",
            description="Part anomaly coordinate and part nuisance coordinate.",
            theorem_scope="inside_section3",
            weights=(inv_sqrt2, inv_sqrt2),
        ),
        ScoreCandidate(
            name="linear_noise_dim",
            family="linear",
            description="Projection on a coordinate unrelated to the anomaly.",
            theorem_scope="inside_section3_boundary",
            weights=(0.0, 1.0),
        ),
        ScoreCandidate(
            name="linear_negative_oracle",
            family="linear",
            description="Correct coordinate with the wrong anomaly orientation.",
            theorem_scope="outside_orientation_assumption",
            weights=(-1.0,),
        ),
        ScoreCandidate(
            name="residual_oracle_r1",
            family="residual",
            description="One-dimensional squared residual on the anomaly coordinate.",
            theorem_scope="nonlinear_analytic_control",
            dims=(0,),
        ),
        ScoreCandidate(
            name="residual_mixed_r2",
            family="residual",
            description="Two-dimensional residual containing one anomaly coordinate.",
            theorem_scope="nonlinear_analytic_control",
            dims=(0, 1),
        ),
        ScoreCandidate(
            name="radial_all",
            family="residual",
            description="Full radial energy over all coordinates.",
            theorem_scope="nonlinear_analytic_control",
            dims=tuple(range(d)),
        ),
        ScoreCandidate(
            name="residual_noise_r1",
            family="residual",
            description="Squared residual on an unrelated coordinate.",
            theorem_scope="nonlinear_null_control",
            dims=(1,),
        ),
        ScoreCandidate(
            name="residual_without_anomaly",
            family="residual",
            description="Residual subspace that excludes the anomaly coordinate.",
            theorem_scope="nonlinear_null_control",
            dims=tuple(range(1, d)),
        ),
        ScoreCandidate(
            name="constant_collapse",
            family="constant",
            description="Degenerate collapsed score.",
            theorem_scope="degenerate_control",
        ),
    ]


def project_many(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Project rows of ``x`` on many candidate directions.

    ``np.matmul`` can emit false floating-point warnings with Apple's BLAS on
    some NumPy builds even when the result is finite. ``einsum`` avoids that
    noisy path while computing the same matrix of scores.
    """
    return np.einsum("nd,kd->nk", x, weights, optimize=True)


def sample_correlated_typical_views(
    rng: np.random.Generator, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample typical views and an independent descriptor for NN pairing.

    The score features follow the Gaussian stable-nuisance model. The matching descriptors are
    fixed, model-independent observed covariates: both views share a latent descriptor with small
    measurement noise. This lets empirical CAP use nearest-neighbor pairing without using the
    candidate score itself.
    """
    correlations = validation_correlations(config)
    shared = rng.normal(size=(config.n_pairs, config.n_features)) * np.sqrt(correlations)
    noise_scale = np.sqrt(1.0 - correlations)
    x1 = shared + rng.normal(size=(config.n_pairs, config.n_features)) * noise_scale
    x2 = shared + rng.normal(size=(config.n_pairs, config.n_features)) * noise_scale

    match_latent = rng.normal(size=(config.n_pairs, config.n_match_features))
    match_1 = match_latent + config.match_noise * rng.normal(
        size=(config.n_pairs, config.n_match_features)
    )
    match_2 = match_latent + config.match_noise * rng.normal(
        size=(config.n_pairs, config.n_match_features)
    )
    return x1, x2, match_1, match_2


def nearest_neighbor_pair_indices(
    match_1: np.ndarray, match_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each object in view 1 to its nearest neighbor in view 2."""
    distances, indices = cKDTree(match_2).query(match_1, k=1, workers=-1)
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def sample_test_sets(
    rng: np.random.Generator, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    normal = rng.normal(size=(config.n_test, config.n_features))
    anomaly = rng.normal(size=(config.n_test, config.n_features))
    anomaly[:, 0] += config.anomaly_shift * np.sqrt(config.lambda_z)
    return normal, anomaly


def shift_vector(config: ExperimentConfig, angle_rad: float) -> np.ndarray:
    shift = np.zeros(config.n_features, dtype=np.float64)
    shift[0] = config.benign_shift * np.cos(angle_rad)
    shift[1] = config.benign_shift * np.sin(angle_rad)
    return shift


def score_shift_for_angles(
    score_angles: np.ndarray, shift_angle_rad: float, config: ExperimentConfig
) -> np.ndarray:
    return config.benign_shift * np.cos(score_angles - shift_angle_rad)


def sample_shifted_validation_domains(
    rng: np.random.Generator,
    config: ExperimentConfig,
    shift_angle_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    correlations = validation_correlations(config)
    shared = rng.normal(size=(config.n_pairs, config.n_features)) * np.sqrt(correlations)
    noise_scale = np.sqrt(1.0 - correlations)
    view_1 = shared + rng.normal(size=(config.n_pairs, config.n_features)) * noise_scale
    view_2 = shared + rng.normal(size=(config.n_pairs, config.n_features)) * noise_scale
    view_2 += shift_vector(config, shift_angle_rad)[None, :]
    return view_1, view_2


def candidate_weights(candidate: ScoreCandidate, config: ExperimentConfig) -> np.ndarray:
    weights = np.zeros(config.n_features, dtype=np.float64)
    for idx, value in enumerate(candidate.weights):
        weights[idx] = value
    return weights


def score_candidate(
    x: np.ndarray, candidate: ScoreCandidate, config: ExperimentConfig
) -> np.ndarray:
    if candidate.family == "linear":
        return np.einsum("nd,d->n", x, candidate_weights(candidate, config), optimize=True)
    if candidate.family == "residual":
        if not candidate.dims:
            return np.zeros(x.shape[0], dtype=np.float64)
        return np.sum(x[:, candidate.dims] ** 2, axis=1)
    if candidate.family == "constant":
        return np.zeros(x.shape[0], dtype=np.float64)
    raise ValueError(f"Unknown score family {candidate.family!r}.")


def candidate_null_moments(
    candidate: ScoreCandidate, config: ExperimentConfig
) -> tuple[float, float]:
    if candidate.family == "linear":
        return 0.0, 1.0
    if candidate.family == "residual":
        dof = len(candidate.dims)
        return float(dof), float(np.sqrt(2.0 * dof))
    if candidate.family == "constant":
        return 0.0, 0.0
    raise ValueError(f"Unknown score family {candidate.family!r}.")


def standardize_candidate_scores(
    scores: np.ndarray, candidate: ScoreCandidate, config: ExperimentConfig
) -> np.ndarray:
    mean, std = candidate_null_moments(candidate, config)
    if std == 0.0:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - mean) / std


def candidate_population_threshold_tpr(
    candidate: ScoreCandidate, config: ExperimentConfig
) -> tuple[float, float]:
    alpha = config.fpr
    delta = config.anomaly_shift

    if candidate.family == "linear":
        weights = candidate_weights(candidate, config)
        mean_shift = delta * np.sqrt(config.lambda_z) * weights[0]
        threshold = float(norm.isf(alpha))
        return threshold, float(norm.sf(threshold - mean_shift))

    if candidate.family == "residual":
        dof = len(candidate.dims)
        threshold = float(chi2.isf(alpha, df=dof))
        noncentrality = float(delta**2 * config.lambda_z if 0 in candidate.dims else 0.0)
        tpr = float(ncx2.sf(threshold, df=dof, nc=noncentrality))
        return threshold, tpr

    return float("nan"), float("nan")


def candidate_shifted_marginal_theory(
    candidate: ScoreCandidate,
    config: ExperimentConfig,
    n_quantiles: int = 20_001,
) -> tuple[float, float]:
    eta = config.benign_shift
    alpha = config.fpr

    if candidate.family == "linear":
        weights = candidate_weights(candidate, config)
        shift = eta * weights[0]
        threshold = norm.isf(alpha)
        drift = abs(norm.sf(threshold - shift) - alpha)
        return float(abs(shift)), float(drift)

    if candidate.family == "residual":
        dof = len(candidate.dims)
        if 0 not in candidate.dims:
            return 0.0, 0.0

        noncentrality = eta**2
        q = np.linspace(0.0, 1.0, n_quantiles + 2, dtype=np.float64)[1:-1]
        central_q = chi2.ppf(q, df=dof)
        shifted_q = ncx2.ppf(q, df=dof, nc=noncentrality)
        wasserstein = float(np.mean(np.abs(central_q - shifted_q)) / np.sqrt(2.0 * dof))

        threshold = chi2.isf(alpha, df=dof)
        drift = abs(float(ncx2.sf(threshold, df=dof, nc=noncentrality)) - alpha)
        return wasserstein, drift

    if candidate.family == "constant":
        return 0.0, float("nan")

    raise ValueError(f"Unknown score family {candidate.family!r}.")


def candidate_linear_reliability(candidate: ScoreCandidate, config: ExperimentConfig) -> float:
    if candidate.family != "linear":
        return float("nan")
    weights = candidate_weights(candidate, config)
    correlations = validation_correlations(config)
    return float(np.sum(correlations * weights**2))


def candidate_linear_reliability_components(
    candidate: ScoreCandidate, config: ExperimentConfig
) -> tuple[float, float, float, float]:
    if candidate.family != "linear":
        return float("nan"), float("nan"), float("nan"), float("nan")
    weights = candidate_weights(candidate, config)
    rho_z = float(config.lambda_z * weights[0] ** 2)
    rho_u = float(np.sum(validation_correlations(config)[1:] * weights[1:] ** 2))
    rho_r = rho_z + rho_u
    alignment_pi = rho_z / rho_r if rho_r > 0.0 else 0.0
    return rho_z, rho_u, rho_r, alignment_pi


def tpr_with_empirical_threshold(
    normal_scores: np.ndarray, anomaly_scores: np.ndarray, fpr: float
) -> tuple[float, float]:
    if np.allclose(normal_scores, normal_scores[0]):
        return float("nan"), float("nan")
    threshold = float(np.quantile(normal_scores, 1.0 - fpr, method="higher"))
    return threshold, float(np.mean(anomaly_scores >= threshold))


def cap_lift_for_score_matrix(
    scores_1: np.ndarray, scores_2: np.ndarray, betas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical CAP lift using the repository ApproximationCapacity kernel.

    The theory uses q_beta(s)=sigmoid(beta*s). The binary baseline energy in the CAP kernel gives
    q_beta(p)=sigmoid(beta*(2p-1)), so we pass p=(s+1)/2. The kernel returns log((1 + m1*m2)/2);
    adding log(2) gives the lift plotted in the theory figures.
    """
    best = np.full(scores_1.shape[1], -np.inf, dtype=np.float64)
    best_raw = np.full(scores_1.shape[1], -np.inf, dtype=np.float64)
    best_beta = np.zeros(scores_1.shape[1], dtype=np.float64)

    prob_1 = torch.as_tensor((scores_1 + 1.0) / 2.0, dtype=torch.float64)
    prob_2 = torch.as_tensor((scores_2 + 1.0) / 2.0, dtype=torch.float64)

    def theorem_energy(prob: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        view_shape = (len(y),) + (1,) * (prob.ndim - 1)
        yy = y.to(dtype=prob.dtype).view(view_shape)
        return yy * (1.0 - prob) + (1.0 - yy) * prob

    kernel = ApproximationCapacityKernel(
        beta0=0.0,
        energy_fn=theorem_energy,
        normalize_gradients=False,
    )

    for beta in betas:
        with torch.no_grad():
            raw = kernel.compute_mutual_information(
                prob_1,
                prob_2,
                beta=float(beta),
            )
        raw_values = raw.detach().cpu().numpy() / scores_1.shape[0]
        values = raw_values + np.log(2.0)

        improved = values > best
        best[improved] = values[improved]
        best_raw[improved] = raw_values[improved]
        best_beta[improved] = beta

    return best, best_raw, best_beta


def cap_lift_single(
    scores_1: np.ndarray, scores_2: np.ndarray, betas: np.ndarray
) -> tuple[float, float, float]:
    matrix_1 = np.asarray(scores_1, dtype=np.float64)[:, None]
    matrix_2 = np.asarray(scores_2, dtype=np.float64)[:, None]
    cap, raw, beta = cap_lift_for_score_matrix(matrix_1, matrix_2, betas)
    return float(cap[0]), float(raw[0]), float(beta[0])


def cap_lift_quadrature(
    rho: float,
    betas: np.ndarray,
    quadrature_order: int,
    mean_1: float = 0.0,
    mean_2: float = 0.0,
) -> tuple[float, float]:
    """Deterministic Gaussian expectation for a bivariate normal pair."""
    nodes, weights = hermgauss(quadrature_order)
    u = mean_1 + np.sqrt(2.0) * nodes[:, None]
    e = np.sqrt(2.0) * nodes[None, :]
    normal_weights = (weights[:, None] * weights[None, :]) / np.pi

    rho = float(np.clip(rho, 0.0, 0.999999))
    centered_u = u - mean_1
    v = mean_2 + rho * centered_u + np.sqrt(max(0.0, 1.0 - rho**2)) * e

    best = -np.inf
    best_beta = 0.0
    for beta in betas:
        if beta == 0.0:
            value = 0.0
        else:
            mprod = np.tanh(0.5 * beta * u) * np.tanh(0.5 * beta * v)
            value = float(np.sum(normal_weights * np.log1p(np.clip(mprod, -1.0 + 1e-14, None))))
        if value > best:
            best = value
            best_beta = float(beta)
    return float(best), best_beta


def cap_lift_shift_grid(
    rhos: np.ndarray,
    means_2: np.ndarray,
    betas: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Gaussian CAP lift for a grid of second-view mean shifts."""
    nodes, weights = hermgauss(quadrature_order)
    u = np.sqrt(2.0) * nodes[None, None, :, None]
    e = np.sqrt(2.0) * nodes[None, None, None, :]
    normal_weights = ((weights[:, None] * weights[None, :]) / np.pi)[None, None, :, :]

    rho = np.clip(rhos, 0.0, 0.999999)[None, :, None, None]
    mean = means_2[:, :, None, None]
    v = mean + rho * u + np.sqrt(np.maximum(0.0, 1.0 - rho**2)) * e

    best = np.zeros(means_2.shape, dtype=np.float64)
    best_beta = np.zeros(means_2.shape, dtype=np.float64)
    for beta in betas:
        if beta == 0.0:
            values = np.zeros_like(best)
        else:
            mprod = np.tanh(0.5 * beta * u) * np.tanh(0.5 * beta * v)
            values = np.sum(
                normal_weights * np.log1p(np.clip(mprod, -1.0 + 1e-14, None)),
                axis=(-2, -1),
            )
        improved = values > best
        best[improved] = values[improved]
        best_beta[improved] = beta

    return best, best_beta


def wasserstein_1d_columns(scores_1: np.ndarray, scores_2: np.ndarray) -> np.ndarray:
    sorted_1 = np.sort(scores_1, axis=0)
    sorted_2 = np.sort(scores_2, axis=0)
    return np.mean(np.abs(sorted_1 - sorted_2), axis=0)


def threshold_drift_columns(scores_1: np.ndarray, scores_2: np.ndarray, fpr: float) -> np.ndarray:
    thresholds = np.quantile(scores_1, 1.0 - fpr, axis=0, method="higher")
    observed = np.mean(scores_2 >= thresholds[None, :], axis=0)
    return np.abs(observed - fpr)


def tpr_at_fpr_columns(
    normal_scores: np.ndarray, anomaly_scores: np.ndarray, fpr: float
) -> tuple[np.ndarray, np.ndarray]:
    # In this controlled model every normalized linear null score is exactly
    # N(0, 1), so the theorem's fixed-FPR threshold is known without quantile
    # noise. ``normal_scores`` is kept in the signature to document the
    # evaluated null distribution and to mirror real experiments.
    thresholds = np.full(normal_scores.shape[1], norm.isf(fpr), dtype=np.float64)
    tpr = np.mean(anomaly_scores >= thresholds[None, :], axis=0)
    return thresholds, tpr


def run_channel_reliability_experiment(
    rng: np.random.Generator, config: ExperimentConfig, betas: np.ndarray
) -> pd.DataFrame:
    rows = []
    tau = norm.isf(config.fpr)
    rhos = np.linspace(0.0, 0.97, config.n_rhos, dtype=np.float64)

    for rho in rhos:
        z = rng.normal(size=config.n_channel_pairs)
        eps_1 = rng.normal(size=config.n_channel_pairs)
        eps_2 = rng.normal(size=config.n_channel_pairs)
        scores_1 = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps_1
        scores_2 = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps_2

        normal_scores = rng.normal(size=config.n_test)
        anomaly_scores = rng.normal(
            loc=config.anomaly_shift * np.sqrt(rho), scale=1.0, size=config.n_test
        )
        threshold = float(np.quantile(normal_scores, 1.0 - config.fpr, method="higher"))
        tpr = float(np.mean(anomaly_scores >= threshold))
        cap_emp, cap_raw_emp, beta_emp = cap_lift_single(scores_1, scores_2, betas)
        cap_theory, beta_theory = cap_lift_quadrature(rho, betas, config.quadrature_order)

        score_threshold = np.quantile(scores_1, 1.0 - config.fpr, method="higher")
        score_drift = abs(np.mean(scores_2 >= score_threshold) - config.fpr)

        rows.append(
            {
                "rho": rho,
                "cap_empirical": cap_emp,
                "cap_raw_empirical": cap_raw_emp,
                "cap_theory": cap_theory,
                "best_beta_empirical": beta_emp,
                "best_beta_theory": beta_theory,
                "wasserstein_empirical": float(
                    np.mean(np.abs(np.sort(scores_1) - np.sort(scores_2)))
                ),
                "threshold_drift_empirical": float(score_drift),
                "threshold_empirical": threshold,
                "tpr_empirical": tpr,
                "tpr_theory": float(norm.sf(tau - config.anomaly_shift * np.sqrt(rho))),
            }
        )

    return pd.DataFrame(rows)


def run_linear_direction_experiment(
    rng: np.random.Generator, config: ExperimentConfig, betas: np.ndarray
) -> pd.DataFrame:
    curve = projection_directions(config)
    weights = make_direction_matrix(curve, config.n_features)
    tau = norm.isf(config.fpr)

    x1, x2, match_1, match_2 = sample_correlated_typical_views(rng, config)
    pair_idx, pair_distances = nearest_neighbor_pair_indices(match_1, match_2)
    pair_accuracy = float(np.mean(pair_idx == np.arange(config.n_pairs)))
    mean_pair_distance = float(np.mean(pair_distances))
    normal, anomaly = sample_test_sets(rng, config)

    scores_1 = project_many(x1, weights)
    scores_2 = project_many(x2, weights)
    cap_scores_2 = scores_2[pair_idx]
    normal_scores = project_many(normal, weights)
    anomaly_scores = project_many(anomaly, weights)

    cap_emp, cap_raw_emp, beta_emp = cap_lift_for_score_matrix(scores_1, cap_scores_2, betas)
    thresholds, tpr_emp = tpr_at_fpr_columns(normal_scores, anomaly_scores, config.fpr)
    empirical_thresholds = np.quantile(normal_scores, 1.0 - config.fpr, axis=0, method="higher")
    empirical_threshold_tpr = np.mean(anomaly_scores >= empirical_thresholds[None, :], axis=0)
    w1_emp = wasserstein_1d_columns(scores_1, scores_2)
    drift_emp = threshold_drift_columns(scores_1, scores_2, config.fpr)

    cap_theory = []
    beta_theory = []
    for rho in curve["rho_r"].to_numpy():
        cap, beta = cap_lift_quadrature(rho, betas, config.quadrature_order)
        cap_theory.append(cap)
        beta_theory.append(beta)

    out = curve.copy()
    out["cap_empirical"] = cap_emp
    out["cap_raw_empirical"] = cap_raw_emp
    out["cap_theory"] = np.asarray(cap_theory)
    out["best_beta_empirical"] = beta_emp
    out["best_beta_theory"] = np.asarray(beta_theory)
    out["nn_pair_accuracy"] = pair_accuracy
    out["nn_pair_mean_distance"] = mean_pair_distance
    out["wasserstein_empirical"] = w1_emp
    out["wasserstein_theory"] = 0.0
    out["threshold_drift_empirical"] = drift_emp
    out["threshold_drift_theory"] = 0.0
    out["threshold_empirical"] = thresholds
    out["tpr_empirical"] = tpr_emp
    out["threshold_quantile_empirical"] = empirical_thresholds
    out["tpr_quantile_empirical"] = empirical_threshold_tpr
    out["tpr_theory"] = norm.sf(tau - config.anomaly_shift * np.sqrt(out["rho_z"].to_numpy()))
    return out


def run_marginal_shift_experiment(
    rng: np.random.Generator, config: ExperimentConfig, betas: np.ndarray
) -> pd.DataFrame:
    curve = projection_directions(config)
    weights = make_direction_matrix(curve, config.n_features)
    tau = norm.isf(config.fpr)

    view_1, view_2 = sample_shifted_validation_domains(rng, config)
    match_latent = rng.normal(size=(config.n_pairs, config.n_match_features))
    match_1 = match_latent + config.match_noise * rng.normal(
        size=(config.n_pairs, config.n_match_features)
    )
    match_2 = match_latent + config.match_noise * rng.normal(
        size=(config.n_pairs, config.n_match_features)
    )
    pair_idx, pair_distances = nearest_neighbor_pair_indices(match_1, match_2)
    pair_accuracy = float(np.mean(pair_idx == np.arange(config.n_pairs)))
    mean_pair_distance = float(np.mean(pair_distances))
    normal, anomaly = sample_test_sets(rng, config)

    scores_1 = project_many(view_1, weights)
    scores_2 = project_many(view_2, weights)
    cap_scores_2 = scores_2[pair_idx]
    normal_scores = project_many(normal, weights)
    anomaly_scores = project_many(anomaly, weights)

    cap_emp, cap_raw_emp, beta_emp = cap_lift_for_score_matrix(scores_1, cap_scores_2, betas)
    thresholds, tpr_emp = tpr_at_fpr_columns(normal_scores, anomaly_scores, config.fpr)
    empirical_thresholds = np.quantile(normal_scores, 1.0 - config.fpr, axis=0, method="higher")
    empirical_threshold_tpr = np.mean(anomaly_scores >= empirical_thresholds[None, :], axis=0)
    w1_emp = wasserstein_1d_columns(scores_1, scores_2)
    drift_emp = threshold_drift_columns(scores_1, scores_2, config.fpr)

    score_angles = curve["angle_rad"].to_numpy()
    score_shift = score_shift_for_angles(score_angles, 0.0, config)
    cap_theory = []
    beta_theory = []
    for rho, mu_2 in zip(curve["rho_r"].to_numpy(), score_shift, strict=True):
        cap, beta = cap_lift_quadrature(
            rho,
            betas,
            config.quadrature_order,
            mean_1=0.0,
            mean_2=float(mu_2),
        )
        cap_theory.append(cap)
        beta_theory.append(beta)

    out = curve.copy()
    out["cap_empirical"] = cap_emp
    out["cap_raw_empirical"] = cap_raw_emp
    out["cap_theory"] = np.asarray(cap_theory)
    out["best_beta_empirical"] = beta_emp
    out["best_beta_theory"] = np.asarray(beta_theory)
    out["nn_pair_accuracy"] = pair_accuracy
    out["nn_pair_mean_distance"] = mean_pair_distance
    out["wasserstein_empirical"] = w1_emp
    out["wasserstein_theory"] = np.abs(score_shift)
    out["threshold_drift_empirical"] = drift_emp
    out["threshold_drift_theory"] = np.abs(norm.sf(tau - score_shift) - config.fpr)
    out["threshold_empirical"] = thresholds
    out["tpr_empirical"] = tpr_emp
    out["threshold_quantile_empirical"] = empirical_thresholds
    out["tpr_quantile_empirical"] = empirical_threshold_tpr
    out["tpr_theory"] = norm.sf(
        tau - config.anomaly_shift * np.sqrt(config.lambda_z) * curve["cos_to_anomaly"].to_numpy()
    )
    return out


def selected_angle_range(
    values: np.ndarray,
    angles_deg: np.ndarray,
    *,
    maximize: bool,
    tolerance: float = 1e-10,
) -> tuple[float, float, float]:
    best = float(np.max(values) if maximize else np.min(values))
    if maximize:
        selected = np.flatnonzero(values >= best - tolerance)
    else:
        selected = np.flatnonzero(values <= best + tolerance)
    selected_angles = angles_deg[selected]
    return float(np.min(selected_angles)), float(np.max(selected_angles)), best


def run_marginal_shift_selector_sweep(config: ExperimentConfig, betas: np.ndarray) -> pd.DataFrame:
    """Population selectors as the benign validation shift rotates.

    CAP selects by maximizing the population paired CAP objective. Marginal stability baselines
    select by minimizing their population drift.
    """
    curve = projection_directions(config)
    score_angles = curve["angle_rad"].to_numpy()
    score_angles_deg = curve["angle_deg"].to_numpy()
    shift_angles = score_angles
    tau = norm.isf(config.fpr)

    shift_means = config.benign_shift * np.cos(shift_angles[:, None] - score_angles[None, :])
    cap_grid, beta_grid = cap_lift_shift_grid(
        curve["rho_r"].to_numpy(),
        shift_means,
        betas,
        config.quadrature_order,
    )
    tpr_values = norm.sf(tau - config.anomaly_shift * np.sqrt(curve["rho_z"].to_numpy()))

    rows = []
    for shift_idx, shift_angle in enumerate(shift_angles):
        score_shift = score_shift_for_angles(score_angles, float(shift_angle), config)
        w1_values = np.abs(score_shift)
        drift_values = np.abs(norm.sf(tau - score_shift) - config.fpr)

        cap_min, cap_max, cap_best = selected_angle_range(
            cap_grid[shift_idx], score_angles_deg, maximize=True
        )
        w1_min, w1_max, w1_best = selected_angle_range(w1_values, score_angles_deg, maximize=False)
        drift_min, drift_max, drift_best = selected_angle_range(
            drift_values, score_angles_deg, maximize=False
        )

        cap_selected = (score_angles_deg >= cap_min) & (score_angles_deg <= cap_max)
        w1_selected = (score_angles_deg >= w1_min) & (score_angles_deg <= w1_max)
        drift_selected = (score_angles_deg >= drift_min) & (score_angles_deg <= drift_max)

        rows.append(
            {
                "shift_angle_rad": float(shift_angle),
                "shift_angle_deg": float(np.degrees(shift_angle)),
                "shift_z_weight": float(np.cos(shift_angle)),
                "shift_u_weight": float(np.sin(shift_angle)),
                "cap_selected_angle_min_deg": cap_min,
                "cap_selected_angle_max_deg": cap_max,
                "cap_selected_value": cap_best,
                "cap_selected_beta": float(
                    beta_grid[shift_idx, int(np.argmax(cap_grid[shift_idx]))]
                ),
                "cap_selected_tpr_min": float(np.min(tpr_values[cap_selected])),
                "cap_selected_tpr_max": float(np.max(tpr_values[cap_selected])),
                "w1_selected_angle_min_deg": w1_min,
                "w1_selected_angle_max_deg": w1_max,
                "w1_selected_value": w1_best,
                "w1_selected_tpr_min": float(np.min(tpr_values[w1_selected])),
                "w1_selected_tpr_max": float(np.max(tpr_values[w1_selected])),
                "threshold_selected_angle_min_deg": drift_min,
                "threshold_selected_angle_max_deg": drift_max,
                "threshold_selected_value": drift_best,
                "threshold_selected_tpr_min": float(np.min(tpr_values[drift_selected])),
                "threshold_selected_tpr_max": float(np.max(tpr_values[drift_selected])),
                "tpr_optimal_angle_deg": 0.0,
                "tpr_optimal": float(tpr_values[0]),
            }
        )

    return pd.DataFrame(rows)


def run_score_family_experiment(
    rng: np.random.Generator, config: ExperimentConfig, betas: np.ndarray
) -> pd.DataFrame:
    x1, x2, match_1, match_2 = sample_correlated_typical_views(rng, config)
    pair_idx, pair_distances = nearest_neighbor_pair_indices(match_1, match_2)
    pair_accuracy = float(np.mean(pair_idx == np.arange(config.n_pairs)))
    mean_pair_distance = float(np.mean(pair_distances))
    shift_1, shift_2 = sample_shifted_validation_domains(rng, config)
    normal, anomaly = sample_test_sets(rng, config)

    rows = []
    for candidate in build_score_candidates(config):
        rho_z, rho_u, rho_r, alignment_pi = candidate_linear_reliability_components(
            candidate, config
        )
        view_1_raw = score_candidate(x1, candidate, config)
        view_2_raw = score_candidate(x2, candidate, config)
        view_2_cap_raw = view_2_raw[pair_idx]
        shift_1_raw = score_candidate(shift_1, candidate, config)
        shift_2_raw = score_candidate(shift_2, candidate, config)
        normal_raw = score_candidate(normal, candidate, config)
        anomaly_raw = score_candidate(anomaly, candidate, config)

        view_1 = standardize_candidate_scores(view_1_raw, candidate, config)
        view_2 = standardize_candidate_scores(view_2_raw, candidate, config)
        view_2_cap = standardize_candidate_scores(view_2_cap_raw, candidate, config)
        shift_1_scores = standardize_candidate_scores(shift_1_raw, candidate, config)
        shift_2_scores = standardize_candidate_scores(shift_2_raw, candidate, config)

        cap_emp, cap_raw_emp, beta_emp = cap_lift_single(view_1, view_2_cap, betas)
        pop_threshold, pop_tpr = candidate_population_threshold_tpr(candidate, config)
        emp_threshold, emp_tpr = tpr_with_empirical_threshold(normal_raw, anomaly_raw, config.fpr)

        if np.isnan(pop_threshold):
            pop_threshold_tpr = float("nan")
        else:
            pop_threshold_tpr = float(np.mean(anomaly_raw >= pop_threshold))
        shifted_w1_theory, shifted_drift_theory = candidate_shifted_marginal_theory(
            candidate, config
        )

        rows.append(
            {
                "score": candidate.name,
                "family": candidate.family,
                "theorem_scope": candidate.theorem_scope,
                "description": candidate.description,
                "linear_rho_z": rho_z,
                "linear_rho_u": rho_u,
                "linear_rho_r": rho_r,
                "linear_alignment_pi": alignment_pi,
                "linear_reliability_rho": candidate_linear_reliability(candidate, config),
                "cap_empirical": cap_emp,
                "cap_raw_empirical": cap_raw_emp,
                "best_beta_empirical": beta_emp,
                "nn_pair_accuracy": pair_accuracy,
                "nn_pair_mean_distance": mean_pair_distance,
                "unshifted_wasserstein_empirical": float(
                    np.mean(np.abs(np.sort(view_1) - np.sort(view_2)))
                ),
                "unshifted_wasserstein_theory": 0.0,
                "unshifted_threshold_drift_empirical": float(
                    threshold_drift_columns(view_1[:, None], view_2[:, None], config.fpr)[0]
                ),
                "unshifted_threshold_drift_theory": 0.0,
                "shifted_wasserstein_empirical": float(
                    np.mean(np.abs(np.sort(shift_1_scores) - np.sort(shift_2_scores)))
                ),
                "shifted_wasserstein_theory": shifted_w1_theory,
                "shifted_threshold_drift_empirical": float(
                    threshold_drift_columns(
                        shift_1_scores[:, None],
                        shift_2_scores[:, None],
                        config.fpr,
                    )[0]
                ),
                "shifted_threshold_drift_theory": shifted_drift_theory,
                "population_threshold": pop_threshold,
                "empirical_threshold": emp_threshold,
                "tpr_empirical_population_threshold": pop_threshold_tpr,
                "tpr_empirical_quantile_threshold": emp_tpr,
                "tpr_population": pop_tpr,
            }
        )

    return pd.DataFrame(rows)


def run_alignment_assumption_experiment(
    config: ExperimentConfig, betas: np.ndarray
) -> pd.DataFrame:
    tau = norm.isf(config.fpr)
    angles = np.linspace(0.0, 0.5 * np.pi, config.n_angles, dtype=np.float64)
    rows = []

    cases = {
        "aligned": (config.lambda_z, config.lambda_u),
        "nuisance-dominated": (config.lambda_u, config.lambda_z),
    }
    for case, (lambda_z_case, lambda_u_case) in cases.items():
        for angle in angles:
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            rho_z = lambda_z_case * c**2
            rho_u = lambda_u_case * s**2
            rho_r = rho_z + rho_u
            alignment_pi = rho_z / rho_r if rho_r > 0.0 else 0.0
            cap, beta = cap_lift_quadrature(rho_r, betas, config.quadrature_order)
            rows.append(
                {
                    "case": case,
                    "angle_rad": angle,
                    "angle_deg": float(np.degrees(angle)),
                    "cos_to_anomaly": c,
                    "rho_z": rho_z,
                    "rho_u": rho_u,
                    "rho_r": rho_r,
                    "alignment_pi": alignment_pi,
                    "rho": rho_r,
                    "cap_theory": cap,
                    "best_beta_theory": beta,
                    "tpr_theory": float(norm.sf(tau - config.anomaly_shift * np.sqrt(rho_z))),
                }
            )

    return pd.DataFrame(rows)


def run_alignment_ratio_sweep(config: ExperimentConfig, betas: np.ndarray) -> pd.DataFrame:
    """Sweep the nuisance-to-signal reproducibility ratio.

    The signal reliability is fixed and the nuisance reliability varies. The transition occurs at
    lambda_U / lambda_Z = 1, where every direction in the two-dimensional span has the same total
    reproducible reliability.
    """
    tau = norm.isf(config.fpr)
    angles = np.linspace(0.0, 0.5 * np.pi, config.n_angles, dtype=np.float64)
    n_low = config.n_ratios // 2
    n_high = config.n_ratios - n_low - 1
    ratios = np.concatenate(
        [
            np.geomspace(config.ratio_min, 1.0, n_low + 1)[:-1],
            np.array([1.0], dtype=np.float64),
            np.geomspace(1.0, config.ratio_max, n_high + 1)[1:],
        ]
    )
    lambda_z = config.ratio_sweep_lambda_z
    rows = []

    for ratio in ratios:
        lambda_u = lambda_z * ratio
        c = np.cos(angles)
        s = np.sin(angles)
        rho_z = lambda_z * c**2
        rho_u = lambda_u * s**2
        rho_r = rho_z + rho_u
        cap_values = []
        beta_values = []
        for rho in rho_r:
            cap, beta = cap_lift_quadrature(rho, betas, config.quadrature_order)
            cap_values.append(cap)
            beta_values.append(beta)
        cap_values = np.asarray(cap_values)
        beta_values = np.asarray(beta_values)
        tpr_values = norm.sf(tau - config.anomaly_shift * np.sqrt(rho_z))

        cap_max = float(np.max(cap_values))
        tie_tol = 1e-10
        selected = np.flatnonzero(cap_values >= cap_max - tie_tol)
        selected_angles = np.degrees(angles[selected])
        selected_tpr = tpr_values[selected]
        selected_rho_z = rho_z[selected]

        if np.isclose(ratio, 1.0, rtol=1e-6, atol=1e-8):
            selected_type = "tie"
        elif ratio < 1.0:
            selected_type = "anomaly"
        else:
            selected_type = "nuisance"

        rows.append(
            {
                "lambda_ratio_u_over_z": float(ratio),
                "lambda_z": float(lambda_z),
                "lambda_u": float(lambda_u),
                "selected_type": selected_type,
                "cap_max": cap_max,
                "cap_at_anomaly": float(cap_values[0]),
                "cap_at_nuisance": float(cap_values[-1]),
                "best_beta_at_cap_max": float(beta_values[selected[0]]),
                "cap_selected_angle_min_deg": float(np.min(selected_angles)),
                "cap_selected_angle_max_deg": float(np.max(selected_angles)),
                "cap_selected_rho_z_min": float(np.min(selected_rho_z)),
                "cap_selected_rho_z_max": float(np.max(selected_rho_z)),
                "cap_selected_tpr_min": float(np.min(selected_tpr)),
                "cap_selected_tpr_max": float(np.max(selected_tpr)),
                "tpr_optimal_angle_deg": 0.0,
                "tpr_optimal": float(tpr_values[0]),
                "rho_r_anomaly": float(rho_r[0]),
                "rho_r_nuisance": float(rho_r[-1]),
            }
        )

    return pd.DataFrame(rows)


def save_score_distribution_data(
    rng: np.random.Generator, config: ExperimentConfig, output_dir: Path
) -> pd.DataFrame:
    n = min(config.n_test, 150_000)
    normal = rng.normal(size=(n, config.n_features))
    anomaly = rng.normal(size=(n, config.n_features))
    anomaly[:, 0] += config.anomaly_shift * np.sqrt(config.lambda_z)

    rows = []
    for direction_name, weight in {
        "oracle_w_star": np.array([1.0, 0.0]),
        "orthogonal": np.array([0.0, 1.0]),
    }.items():
        w = np.zeros(config.n_features)
        w[:2] = weight
        s0 = normal[:, 0] * weight[0] + normal[:, 1] * weight[1]
        s1 = anomaly[:, 0] * weight[0] + anomaly[:, 1] * weight[1]
        rows.append(
            {
                "direction": direction_name,
                "normal_mean": float(np.mean(s0)),
                "normal_std": float(np.std(s0)),
                "anomaly_mean": float(np.mean(s1)),
                "anomaly_std": float(np.std(s1)),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "score_distribution_summary.csv", index=False)
    plot_score_distributions(normal, anomaly, config, output_dir)
    return summary


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def plot_channel_reliability(df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    axes[0].plot(df["rho"], df["cap_theory"], color="#1f77b4", lw=2.2, label="population")
    axes[0].scatter(df["rho"], df["cap_empirical"], color="#ff7f0e", s=22, label="finite sample")
    axes[0].set_xlabel("paired reliability rho")
    axes[0].set_ylabel("optimized CAP lift L*")
    axes[0].set_title("CAP rises with reproducible score signal")
    axes[0].legend()

    axes[1].plot(df["rho"], df["tpr_theory"], color="#2ca02c", lw=2.2, label="population")
    axes[1].scatter(df["rho"], df["tpr_empirical"], color="#d62728", s=22, label="finite sample")
    axes[1].axhline(config.fpr, color="0.35", lw=1.0, ls=":", label="FPR target")
    axes[1].set_xlabel("paired reliability rho")
    axes[1].set_ylabel(f"TPR at FPR={config.fpr:g}")
    axes[1].set_title("No-nuisance case: rho = rho_Z = rho_R")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "theory_cap_tpr_vs_reliability.png", dpi=220)
    plt.close(fig)


def plot_score_pair_scatter(config: ExperimentConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(config.seed + 101)
    n = min(5_000, config.n_pairs)
    local_config = replace(config, n_pairs=n)
    x1, x2, match_1, match_2 = sample_correlated_typical_views(rng, local_config)
    pair_idx, _ = nearest_neighbor_pair_indices(match_1, match_2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, theta, title in [
        (axes[0], 0.5 * np.pi, "orthogonal low-reliability direction"),
        (axes[1], 0.0, "anomaly direction w*"),
    ]:
        w = np.zeros(config.n_features, dtype=np.float64)
        w[0] = np.cos(theta)
        w[1] = np.sin(theta)
        score_1 = np.einsum("nd,d->n", x1, w, optimize=True)
        score_2 = np.einsum("nd,d->n", x2, w, optimize=True)[pair_idx]
        ax.scatter(score_1, score_2, s=5, alpha=0.22, color="#4c78a8", linewidths=0)
        ax.set_title(f"{title}\nNN-paired corr={np.corrcoef(score_1, score_2)[0, 1]:.2f}")
        ax.set_xlabel("validation view 1 score")
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.axvline(0.0, color="0.75", lw=0.8)
    axes[0].set_ylabel("validation view 2 score")
    fig.tight_layout()
    fig.savefig(output_dir / "score_pair_scatter.png", dpi=220)
    plt.close(fig)


def plot_linear_direction_sweep(
    df: pd.DataFrame, config: ExperimentConfig, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    x = df["angle_deg"]

    axes[0, 0].plot(x, df["tpr_theory"], color="#2ca02c", lw=2.2, label="population")
    axes[0, 0].scatter(x, df["tpr_empirical"], color="#d62728", s=16, label="finite sample")
    axes[0, 0].axhline(config.fpr, color="0.35", lw=1.0, ls=":", label="FPR target")
    axes[0, 0].set_ylabel(f"TPR at FPR={config.fpr:g}")
    axes[0, 0].set_title("Downstream anomaly power")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend()

    axes[0, 1].plot(x, df["cap_theory"], color="#1f77b4", lw=2.2, label="population")
    axes[0, 1].scatter(x, df["cap_empirical"], color="#ff7f0e", s=16, label="finite sample")
    axes[0, 1].set_ylabel("optimized CAP lift L*")
    axes[0, 1].set_title("CAP follows two-view reliability")
    axes[0, 1].legend()

    axes[1, 0].plot(x, df["rho_z"], color="#2ca02c", lw=2.2, label="rho_Z(w)")
    axes[1, 0].plot(x, df["rho_u"], color="#9467bd", lw=2.2, label="rho_U(w)")
    axes[1, 0].plot(x, df["rho_r"], color="#1f77b4", lw=2.2, label="rho_R(w)")
    axes[1, 0].set_xlabel("angle from anomaly direction w* (degrees)")
    axes[1, 0].set_ylabel("reliability")
    axes[1, 0].set_title("CAP follows total reliability rho_R")
    axes[1, 0].legend()

    axes[1, 1].axhline(
        0.0,
        color="#8c564b",
        lw=1.6,
        ls="--",
        label="W1 population",
    )
    axes[1, 1].scatter(
        x,
        df["wasserstein_empirical"],
        color="#8c564b",
        s=18,
        alpha=0.68,
        label="W1 empirical",
    )
    drift_axis = axes[1, 1].twinx()
    drift_axis.axhline(
        0.0,
        color="#e377c2",
        lw=1.6,
        ls="--",
        label="drift population",
    )
    drift_axis.scatter(
        x,
        df["threshold_drift_empirical"],
        color="#e377c2",
        marker="D",
        s=16,
        alpha=0.68,
        label="drift empirical",
    )
    axes[1, 1].set_xlabel("angle from anomaly direction w* (degrees)")
    axes[1, 1].set_ylabel("W1")
    drift_axis.set_ylabel("threshold drift")
    axes[1, 1].set_title("Marginal estimators: population is zero")
    handles_1, labels_1 = axes[1, 1].get_legend_handles_labels()
    handles_2, labels_2 = drift_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_1 + handles_2, labels_1 + labels_2)

    for ax in axes.flat:
        ax.axvline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.65)
        ax.text(
            0.01,
            0.04,
            "w*",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="0.2",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "linear_direction_sweep.png", dpi=220)
    plt.close(fig)


def plot_cap_vs_tpr(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sc = ax.scatter(
        df["cap_empirical"],
        df["tpr_empirical"],
        c=df["angle_deg"],
        cmap="viridis_r",
        s=34,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("empirical CAP lift L*")
    ax.set_ylabel("empirical TPR at target FPR")
    ax.set_title("The CAP-selected direction is also the high-power direction")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("angle from w* (degrees)")
    fig.tight_layout()
    fig.savefig(output_dir / "cap_vs_tpr.png", dpi=220)
    plt.close(fig)


def plot_marginal_shift_trap(df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> None:
    x = df["angle_deg"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)

    axes[0, 0].plot(
        x,
        df["cap_theory"],
        color="#1f77b4",
        lw=2.2,
        label="CAP population",
    )
    axes[0, 0].scatter(
        x,
        df["cap_empirical"],
        color="#ff7f0e",
        s=16,
        alpha=0.70,
        label="CAP finite sample",
    )
    axes[0, 0].set_ylabel("optimized CAP lift L*")
    axes[0, 0].set_title("CAP selects the paired anomaly direction")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        x,
        df["tpr_theory"],
        color="#2ca02c",
        lw=2.2,
        label="TPR population",
    )
    axes[0, 1].scatter(
        x,
        df["tpr_empirical"],
        color="#d62728",
        s=16,
        alpha=0.70,
        label="TPR finite sample",
    )
    axes[0, 1].axhline(config.fpr, color="0.35", lw=1.0, ls=":", label="FPR target")
    axes[0, 1].set_ylabel(f"TPR at FPR={config.fpr:g}")
    axes[0, 1].set_title("Anomaly power is also maximized at w*")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        x,
        df["wasserstein_theory"],
        color="#8c564b",
        lw=2.2,
        label="W1 population",
    )
    axes[1, 0].scatter(
        x,
        df["wasserstein_empirical"],
        color="#8c564b",
        s=16,
        alpha=0.60,
        label="W1 finite sample",
    )
    axes[1, 0].set_xlabel("angle from anomaly direction w* (degrees)")
    axes[1, 0].set_ylabel("W1 validation shift")
    axes[1, 0].set_title("W1 minimization selects the orthogonal direction")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        x,
        df["threshold_drift_theory"],
        color="#e377c2",
        lw=2.2,
        label="threshold drift population",
    )
    axes[1, 1].scatter(
        x,
        df["threshold_drift_empirical"],
        color="#e377c2",
        s=16,
        alpha=0.60,
        label="threshold drift finite sample",
    )
    axes[1, 1].set_xlabel("angle from anomaly direction w* (degrees)")
    axes[1, 1].set_ylabel("threshold drift")
    axes[1, 1].set_title("Threshold-drift minimization also selects orthogonal")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.axvline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.65)
        ax.axvline(90.0, color="0.2", lw=1.0, ls=":", alpha=0.55)
        ax.text(
            0.01,
            0.05,
            "w*",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.2",
        )
        ax.text(
            0.99,
            0.05,
            "orthogonal",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.2",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "marginal_shift_trap.png", dpi=220)
    plt.close(fig)


def plot_marginal_shift_selector_sweep(
    df: pd.DataFrame, config: ExperimentConfig, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = df["shift_angle_deg"]

    selectors = [
        ("cap", "CAP max", "#1f77b4", "-"),
        ("w1", "min W1", "#8c564b", "--"),
        ("threshold", "min threshold drift", "#e377c2", ":"),
    ]
    for prefix, label, color, linestyle in selectors:
        y_min = df[f"{prefix}_selected_angle_min_deg"]
        y_max = df[f"{prefix}_selected_angle_max_deg"]
        ax.fill_between(x, y_min, y_max, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y_min, color=color, lw=2.2, ls=linestyle, label=label)
        if not np.allclose(y_min, y_max):
            ax.plot(x, y_max, color=color, lw=2.2, ls=linestyle)

    ax.axhline(0.0, color="0.25", lw=1.0, ls="-.", alpha=0.70, label="anomaly")
    ax.axhline(90.0, color="0.25", lw=1.0, ls=":", alpha=0.70, label="nuisance")
    ax.axvline(45.0, color="0.35", lw=1.0, ls="--", alpha=0.65)
    ax.text(
        45.8,
        4.0,
        "marginal switch",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.set_xlabel("benign shift angle from anomaly direction (degrees)")
    ax.set_ylabel("selected score angle from anomaly direction (degrees)")
    ax.set_title("Selector induced by the benign validation-domain shift")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(-4.0, 94.0)
    ax.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "marginal_shift_selector_sweep.png", dpi=220)
    plt.close(fig)


def plot_score_distributions(
    normal: np.ndarray, anomaly: np.ndarray, config: ExperimentConfig, output_dir: Path
) -> None:
    tau = norm.isf(config.fpr)
    anomaly_mean = config.anomaly_shift * np.sqrt(config.lambda_z)
    bins = np.linspace(-4.5, anomaly_mean + 4.5, 150)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for ax, col, title in [
        (axes[0], 0, "oracle direction w*"),
        (axes[1], 1, "orthogonal direction"),
    ]:
        s0 = normal[:, col]
        s1 = anomaly[:, col]
        ax.hist(s0, bins=bins, density=True, alpha=0.42, color="#4c78a8", label="normal")
        ax.hist(
            s1,
            bins=bins,
            density=True,
            alpha=0.42,
            color="#f58518",
            label="anomaly",
        )
        grid = np.linspace(bins.min(), bins.max(), 700)
        ax.plot(grid, norm.pdf(grid, 0.0, 1.0), color="#1f77b4", lw=1.8)
        mean_signal = anomaly_mean if col == 0 else 0.0
        ax.plot(grid, norm.pdf(grid, mean_signal, 1.0), color="#ff7f0e", lw=1.8)
        ax.axvline(tau, color="0.2", lw=1.2, ls="--", label="FPR threshold")
        ax.set_title(title)
        ax.set_xlabel("score")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=220)
    plt.close(fig)


def plot_feature_space_geometry(config: ExperimentConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(config.seed + 202)
    n = 1_800
    normal = rng.normal(size=(n, 2))
    anomaly = rng.normal(size=(n, 2))
    anomaly[:, 0] += config.anomaly_shift * np.sqrt(config.lambda_z)
    shifted = rng.normal(size=(n, 2))
    shifted[:, 0] += config.benign_shift
    shared = rng.normal(size=(n, 2)) * np.sqrt([config.lambda_z, config.lambda_u])
    noise_scale = np.sqrt(1.0 - np.array([config.lambda_z, config.lambda_u]))
    x1 = shared + rng.normal(size=(n, 2)) * noise_scale
    x2 = shared + rng.normal(size=(n, 2)) * noise_scale
    angles = np.linspace(0.0, 0.5 * np.pi, 300)
    c = np.cos(angles)
    rho_z = config.lambda_z * c**2
    rho_u = config.lambda_u * np.sin(angles) ** 2
    rho_r = rho_z + rho_u

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    ax.scatter(
        normal[:, 0],
        normal[:, 1],
        s=8,
        alpha=0.22,
        color="#4c78a8",
        label="typical P0",
    )
    ax.scatter(
        anomaly[:, 0],
        anomaly[:, 1],
        s=8,
        alpha=0.22,
        color="#f58518",
        label="anomaly P1",
    )
    ax.scatter(
        shifted[:, 0],
        shifted[:, 1],
        s=8,
        alpha=0.16,
        color="#54a24b",
        label="shifted validation view",
    )
    ax.arrow(
        0.0,
        0.0,
        1.7,
        0.0,
        width=0.025,
        head_width=0.18,
        color="black",
        length_includes_head=True,
    )
    ax.arrow(
        0.0,
        0.0,
        0.0,
        1.35,
        width=0.025,
        head_width=0.18,
        color="0.35",
        length_includes_head=True,
    )
    ax.text(1.78, 0.05, "w* / anomaly shift", ha="left", va="bottom")
    ax.text(0.08, 1.42, "orthogonal control", ha="left", va="bottom")
    ax.set_xlabel("feature x0")
    ax.set_ylabel("feature x1")
    ax.set_title("Feature-space distributions")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.scatter(
        x1[:, 0],
        x2[:, 0],
        s=7,
        alpha=0.18,
        color="#1f77b4",
        label="score x0 = w*",
    )
    ax.scatter(
        x1[:, 1],
        x2[:, 1],
        s=7,
        alpha=0.18,
        color="#9467bd",
        label="score x1",
    )
    lim = 3.8
    ax.plot([-lim, lim], [-lim, lim], color="0.25", lw=1.0, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("validation view 1 score")
    ax.set_ylabel("validation view 2 score")
    ax.set_title("Paired validation reliability")
    ax.text(
        0.03,
        0.97,
        f"corr(x0 views) = {config.lambda_z:.2f}\n" f"corr(x1 views) = {config.lambda_u:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88},
    )
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    ax.plot(np.degrees(angles), c, color="#2ca02c", lw=2.2, label="alignment c(w)")
    ax.plot(np.degrees(angles), rho_z, color="#54a24b", lw=2.0, label="rho_Z(w)")
    ax.plot(np.degrees(angles), rho_r, color="#1f77b4", lw=2.2, label="rho_R(w)")
    ax.axvline(0.0, color="0.2", lw=1.0, ls="--")
    ax.axvline(90.0, color="0.2", lw=1.0, ls=":")
    ax.scatter([0.0], [1.0], color="#2ca02c", s=35, zorder=3)
    ax.scatter([0.0], [config.lambda_z], color="#54a24b", s=35, zorder=3)
    ax.scatter([0.0], [config.lambda_z], color="#1f77b4", s=25, zorder=4)
    ax.set_xlabel("angle from w* (degrees)")
    ax.set_ylabel("value")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("rho_R peaks at w* when lambda_Z > lambda_U")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "feature_space_geometry.png", dpi=220)
    plt.close(fig)


def plot_population_metric_landscape(
    direction: pd.DataFrame,
    marginal: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    x = direction["angle_deg"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    axes[0].plot(x, direction["rho_z"], lw=2.2, label="rho_Z(w)")
    axes[0].plot(x, direction["rho_u"], lw=2.0, label="rho_U(w)")
    axes[0].plot(x, direction["rho_r"], lw=2.2, label="rho_R(w)")
    axes[0].set_xlabel("angle from w* (degrees)")
    axes[0].set_title("Geometry of candidate directions")
    axes[0].legend()

    axes[1].plot(x, marginal["cap_theory"], color="#1f77b4", lw=2.2)
    ax_tpr = axes[1].twinx()
    ax_tpr.plot(x, marginal["tpr_theory"], color="#2ca02c", lw=2.2)
    axes[1].set_xlabel("angle from w* (degrees)")
    axes[1].set_ylabel("shifted-domain CAP lift", color="#1f77b4")
    ax_tpr.set_ylabel("population TPR", color="#2ca02c")
    axes[1].set_title("CAP and TPR still share the maximizer")

    axes[2].plot(
        x,
        marginal["wasserstein_theory"],
        color="#8c564b",
        lw=2.2,
        label="W1 under shift",
    )
    axes[2].plot(
        x,
        marginal["threshold_drift_theory"],
        color="#e377c2",
        lw=2.2,
        label="threshold drift under shift",
    )
    axes[2].set_xlabel("angle from w* (degrees)")
    axes[2].set_title("Shifted-domain marginal trap")
    axes[2].legend()

    for ax in axes:
        ax.axvline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.65)
        ax.axvline(90.0, color="0.2", lw=1.0, ls=":", alpha=0.55)

    fig.tight_layout()
    fig.savefig(output_dir / "population_metric_landscape.png", dpi=220)
    plt.close(fig)


def plot_score_family_comparison(
    df: pd.DataFrame, config: ExperimentConfig, output_dir: Path
) -> None:
    plot_df = df[df["score"] != "constant_collapse"].copy()
    plot_df = plot_df.sort_values("tpr_population", ascending=False)
    labels = plot_df["score"].to_list()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)

    axes[0].bar(x, plot_df["tpr_population"], color="#2ca02c", alpha=0.75)
    axes[0].scatter(
        x,
        plot_df["tpr_empirical_quantile_threshold"],
        color="#d62728",
        s=30,
        zorder=3,
        label="empirical threshold",
    )
    axes[0].axhline(config.fpr, color="0.35", lw=1.0, ls=":", label="FPR target")
    axes[0].set_ylabel("TPR at target FPR")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Supervised anomaly power of diverse score families")
    axes[0].legend()

    axes[1].bar(x, plot_df["cap_empirical"], color="#1f77b4", alpha=0.78)
    axes[1].set_ylabel("CAP lift")
    axes[1].set_title("CAP on paired typical validation views")

    width = 0.36
    axes[2].bar(
        x - width / 2.0,
        plot_df["shifted_wasserstein_empirical"],
        width=width,
        color="#8c564b",
        alpha=0.72,
        label="W1 empirical",
    )
    axes[2].bar(
        x + width / 2.0,
        plot_df["shifted_wasserstein_theory"],
        width=width,
        facecolor="none",
        edgecolor="#8c564b",
        linewidth=1.5,
        label="W1 population",
    )
    ax_drift = axes[2].twinx()
    ax_drift.plot(
        x,
        plot_df["shifted_threshold_drift_empirical"],
        color="#e377c2",
        marker="D",
        lw=1.6,
        label="drift empirical",
    )
    ax_drift.plot(
        x,
        plot_df["shifted_threshold_drift_theory"],
        color="#e377c2",
        marker="o",
        mfc="white",
        ls="--",
        lw=1.4,
        label="drift population",
    )
    axes[2].set_ylabel("shifted-domain W1")
    ax_drift.set_ylabel("shifted-domain threshold drift")
    axes[2].set_title("Marginal stability under benign shift")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")

    handles_1, labels_1 = axes[2].get_legend_handles_labels()
    handles_2, labels_2 = ax_drift.get_legend_handles_labels()
    axes[2].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "score_family_comparison.png", dpi=220)
    plt.close(fig)


def plot_score_family_distributions(config: ExperimentConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(config.seed + 303)
    n = min(config.n_test, 150_000)
    normal, anomaly = sample_test_sets(rng, config)

    panels = [
        (
            "linear_oracle_w_star",
            "linear oracle",
            (-4.5, config.anomaly_shift * np.sqrt(config.lambda_z) + 4.5),
        ),
        ("linear_noise_dim", "linear noise coordinate", (-4.5, 4.5)),
        ("residual_oracle_r1", "residual oracle r=1", (0.0, 35.0)),
        ("radial_all", f"radial all r={config.n_features}", (0.0, 55.0)),
    ]
    candidate_map = {c.name: c for c in build_score_candidates(config)}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    for ax, (name, title, xlim) in zip(axes.flat, panels, strict=True):
        candidate = candidate_map[name]
        s0 = score_candidate(normal, candidate, config)
        s1 = score_candidate(anomaly, candidate, config)
        bins = np.linspace(xlim[0], xlim[1], 140)
        ax.hist(s0, bins=bins, density=True, alpha=0.38, color="#4c78a8")
        ax.hist(s1, bins=bins, density=True, alpha=0.38, color="#f58518")
        grid = np.linspace(xlim[0], xlim[1], 700)

        if candidate.family == "linear":
            weights = candidate_weights(candidate, config)
            mean_shift = config.anomaly_shift * np.sqrt(config.lambda_z) * weights[0]
            ax.plot(grid, norm.pdf(grid), color="#1f77b4", lw=1.7)
            ax.plot(grid, norm.pdf(grid, mean_shift), color="#ff7f0e", lw=1.7)
        else:
            dof = len(candidate.dims)
            noncentrality = (
                config.anomaly_shift**2 * config.lambda_z if 0 in candidate.dims else 0.0
            )
            ax.plot(grid, chi2.pdf(grid, df=dof), color="#1f77b4", lw=1.7)
            ax.plot(
                grid,
                ncx2.pdf(grid, df=dof, nc=noncentrality),
                color="#ff7f0e",
                lw=1.7,
            )

        threshold, _ = candidate_population_threshold_tpr(candidate, config)
        ax.axvline(threshold, color="0.2", lw=1.0, ls="--")
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.set_xlabel("raw score")
        ax.set_ylabel("density")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4c78a8", alpha=0.38),
        plt.Rectangle((0, 0), 1, 1, color="#f58518", alpha=0.38),
    ]
    fig.legend(handles, ["normal", "anomaly"], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "score_family_distributions.png", dpi=220)
    plt.close(fig)


def plot_alignment_assumption_check(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False)

    for ax, case in zip(axes, ["aligned", "nuisance-dominated"], strict=True):
        group = df[df["case"] == case]
        x = group["angle_deg"]
        ax.plot(x, group["cap_theory"], color="#1f77b4", lw=2.2, label="CAP")
        twin = ax.twinx()
        twin.plot(x, group["tpr_theory"], color="#2ca02c", lw=2.2, label="TPR")
        cap_argmax = group.loc[group["cap_theory"].idxmax(), "angle_deg"]
        tpr_argmax = group.loc[group["tpr_theory"].idxmax(), "angle_deg"]
        ax.axvline(cap_argmax, color="#1f77b4", lw=1.0, ls="--")
        ax.axvline(tpr_argmax, color="#2ca02c", lw=1.0, ls=":")
        ax.set_xlabel("angle from w* (degrees)")
        ax.set_ylabel("population CAP lift", color="#1f77b4")
        twin.set_ylabel("population TPR", color="#2ca02c")
        ax.set_title(f"{case}: CAP max={cap_argmax:.0f} deg, TPR max={tpr_argmax:.0f} deg")

    fig.tight_layout()
    fig.savefig(output_dir / "alignment_assumption_check.png", dpi=220)
    plt.close(fig)


def plot_alignment_ratio_sweep(
    df: pd.DataFrame, config: ExperimentConfig, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    x = df["lambda_ratio_u_over_z"]

    anomaly_selected = df["selected_type"] == "anomaly"
    nuisance_selected = df["selected_type"] == "nuisance"
    tie_selected = df["selected_type"] == "tie"
    axes[0].scatter(
        x[anomaly_selected],
        df.loc[anomaly_selected, "cap_selected_angle_min_deg"],
        color="#2ca02c",
        s=18,
        label="anomaly selected",
    )
    axes[0].scatter(
        x[nuisance_selected],
        df.loc[nuisance_selected, "cap_selected_angle_min_deg"],
        color="#9467bd",
        s=18,
        label="nuisance selected",
    )
    axes[0].vlines(
        x[tie_selected],
        df.loc[tie_selected, "cap_selected_angle_min_deg"],
        df.loc[tie_selected, "cap_selected_angle_max_deg"],
        color="#1f77b4",
        lw=3.0,
        label="all angles tied",
    )
    axes[0].axhline(0.0, color="#2ca02c", lw=1.2, ls=":", label="anomaly direction")
    axes[0].axhline(90.0, color="#9467bd", lw=1.2, ls=":", label="nuisance direction")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"nuisance/signal reliability ratio $\lambda_U/\lambda_Z$")
    axes[0].set_ylabel("selected angle from w* (degrees)")
    axes[0].set_ylim(-5.0, 95.0)
    axes[0].set_title("CAP selection switches at equal reliabilities")
    axes[0].legend(loc="center left", fontsize=8)

    axes[1].plot(x, df["cap_max"], color="#1f77b4", lw=2.2, label="max CAP")
    axes[1].plot(
        x,
        df["cap_at_anomaly"],
        color="#2ca02c",
        lw=1.7,
        ls="--",
        label="CAP at w*",
    )
    axes[1].plot(
        x,
        df["cap_at_nuisance"],
        color="#9467bd",
        lw=1.7,
        ls="--",
        label="CAP at nuisance",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"nuisance/signal reliability ratio $\lambda_U/\lambda_Z$")
    axes[1].set_ylabel("population CAP lift")
    axes[1].set_title("The larger reproducible eigenvalue sets CAP")
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        x,
        df["cap_selected_tpr_min"],
        color="#d62728",
        s=18,
        label="CAP-selected TPR",
    )
    axes[2].vlines(
        x[tie_selected],
        df.loc[tie_selected, "cap_selected_tpr_min"],
        df.loc[tie_selected, "cap_selected_tpr_max"],
        color="#d62728",
        lw=3.0,
        alpha=0.65,
        label="TPR range at tie",
    )
    axes[2].plot(
        x,
        df["tpr_optimal"],
        color="#2ca02c",
        lw=2.2,
        ls="--",
        label="optimal TPR at w*",
    )
    axes[2].axhline(config.fpr, color="0.35", lw=1.0, ls=":", label="FPR target")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"nuisance/signal reliability ratio $\lambda_U/\lambda_Z$")
    axes[2].set_ylabel("TPR at target FPR")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Power collapses after nuisance dominates")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.axvline(1.0, color="0.15", lw=1.1, ls="--")
        ax.text(
            1.03,
            0.96,
            r"$\lambda_U=\lambda_Z$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="0.15",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "alignment_ratio_sweep.png", dpi=220)
    plt.close(fig)


def write_metadata(config: ExperimentConfig, output_dir: Path) -> None:
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    metadata["artifacts"] = list(ARTIFACT_FILENAMES)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def print_key_results(
    channel: pd.DataFrame,
    direction: pd.DataFrame,
    marginal: pd.DataFrame,
    marginal_selector: pd.DataFrame,
    score_family: pd.DataFrame,
    alignment: pd.DataFrame,
    ratio_sweep: pd.DataFrame,
) -> None:
    w_star = direction.iloc[0]
    orthogonal = direction.iloc[-1]
    marginal_w_star = marginal.iloc[0]
    marginal_orthogonal = marginal.iloc[-1]

    print("Section 3 synthetic study complete.")
    print("")
    print("Reliability channel:")
    print(
        channel.loc[
            channel["rho"].isin([channel["rho"].min(), channel["rho"].max()]),
            ["rho", "cap_empirical", "cap_theory", "tpr_empirical", "tpr_theory"],
        ].to_string(index=False)
    )
    print("")
    print("Linear direction sweep:")
    print(
        pd.DataFrame(
            [
                {
                    "candidate": "w_star",
                    "angle_deg": w_star["angle_deg"],
                    "rho_z": w_star["rho_z"],
                    "rho_r": w_star["rho_r"],
                    "nn_pair_accuracy": w_star["nn_pair_accuracy"],
                    "cap_empirical": w_star["cap_empirical"],
                    "tpr_empirical": w_star["tpr_empirical"],
                    "w1_empirical": w_star["wasserstein_empirical"],
                    "threshold_drift_empirical": w_star["threshold_drift_empirical"],
                },
                {
                    "candidate": "orthogonal",
                    "angle_deg": orthogonal["angle_deg"],
                    "rho_z": orthogonal["rho_z"],
                    "rho_r": orthogonal["rho_r"],
                    "nn_pair_accuracy": orthogonal["nn_pair_accuracy"],
                    "cap_empirical": orthogonal["cap_empirical"],
                    "tpr_empirical": orthogonal["tpr_empirical"],
                    "w1_empirical": orthogonal["wasserstein_empirical"],
                    "threshold_drift_empirical": orthogonal["threshold_drift_empirical"],
                },
            ]
        ).to_string(index=False)
    )
    print("")
    print("Score-family controls:")
    control_cols = [
        "score",
        "theorem_scope",
        "cap_empirical",
        "tpr_population",
        "shifted_wasserstein_empirical",
    ]
    control_rows = [
        "linear_oracle_w_star",
        "linear_noise_dim",
        "linear_negative_oracle",
        "residual_oracle_r1",
        "residual_noise_r1",
        "radial_all",
    ]
    controls = score_family[score_family["score"].isin(control_rows)]
    print(controls[control_cols].to_string(index=False))
    print("")
    print("Alignment assumption check:")
    rows = []
    for case, group in alignment.groupby("case"):
        cap_row = group.loc[group["cap_theory"].idxmax()]
        tpr_row = group.loc[group["tpr_theory"].idxmax()]
        rows.append(
            {
                "case": case,
                "cap_argmax_deg": cap_row["angle_deg"],
                "tpr_argmax_deg": tpr_row["angle_deg"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("")
    print("Alignment ratio sweep:")
    ratio_points = np.asarray([0.25, 0.75, 1.0, 1.25, 2.0])
    ratio_rows = []
    for point in ratio_points:
        idx = int(np.argmin(np.abs(ratio_sweep["lambda_ratio_u_over_z"].to_numpy() - point)))
        row = ratio_sweep.iloc[idx]
        ratio_rows.append(
            {
                "lambda_u/lambda_z": row["lambda_ratio_u_over_z"],
                "selected": row["selected_type"],
                "selected_angle_min": row["cap_selected_angle_min_deg"],
                "selected_angle_max": row["cap_selected_angle_max_deg"],
                "cap_max": row["cap_max"],
                "cap_selected_tpr_min": row["cap_selected_tpr_min"],
                "tpr_optimal": row["tpr_optimal"],
            }
        )
    print(pd.DataFrame(ratio_rows).to_string(index=False))
    print("")
    print("CAP versus marginal-stability trap:")
    print(
        pd.DataFrame(
            [
                {
                    "candidate": "w_star",
                    "alignment": marginal_w_star["cos_to_anomaly"],
                    "cap_empirical": marginal_w_star["cap_empirical"],
                    "tpr_empirical": marginal_w_star["tpr_empirical"],
                    "w1_empirical": marginal_w_star["wasserstein_empirical"],
                    "threshold_drift_empirical": marginal_w_star["threshold_drift_empirical"],
                },
                {
                    "candidate": "orthogonal",
                    "alignment": marginal_orthogonal["cos_to_anomaly"],
                    "cap_empirical": marginal_orthogonal["cap_empirical"],
                    "tpr_empirical": marginal_orthogonal["tpr_empirical"],
                    "w1_empirical": marginal_orthogonal["wasserstein_empirical"],
                    "threshold_drift_empirical": marginal_orthogonal["threshold_drift_empirical"],
                },
            ]
        ).to_string(index=False)
    )
    print("")
    print("Shift-angle selector sweep:")
    selector_points = np.asarray([0.0, 30.0, 45.0, 60.0, 90.0])
    selector_rows = []
    for point in selector_points:
        idx = int(np.argmin(np.abs(marginal_selector["shift_angle_deg"].to_numpy() - point)))
        row = marginal_selector.iloc[idx]
        selector_rows.append(
            {
                "shift_angle": row["shift_angle_deg"],
                "cap_angle": row["cap_selected_angle_min_deg"],
                "w1_angle_min": row["w1_selected_angle_min_deg"],
                "w1_angle_max": row["w1_selected_angle_max_deg"],
                "threshold_angle_min": row["threshold_selected_angle_min_deg"],
                "threshold_angle_max": row["threshold_selected_angle_max_deg"],
            }
        )
    print(pd.DataFrame(selector_rows).to_string(index=False))


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    rng = np.random.default_rng(config.seed)
    betas = beta_grid(config)

    channel = run_channel_reliability_experiment(rng, config, betas)
    direction = run_linear_direction_experiment(rng, config, betas)
    marginal = run_marginal_shift_experiment(rng, config, betas)
    marginal_selector = run_marginal_shift_selector_sweep(config, betas)
    score_family = run_score_family_experiment(rng, config, betas)
    alignment = run_alignment_assumption_experiment(config, betas)
    ratio_sweep = run_alignment_ratio_sweep(config, betas)
    save_score_distribution_data(rng, config, config.output_dir)

    channel.to_csv(config.output_dir / "channel_reliability.csv", index=False)
    direction.to_csv(config.output_dir / "linear_direction_sweep.csv", index=False)
    marginal.to_csv(config.output_dir / "marginal_shift_trap.csv", index=False)
    marginal_selector.to_csv(config.output_dir / "marginal_shift_selector_sweep.csv", index=False)
    score_family.to_csv(config.output_dir / "score_family_summary.csv", index=False)
    alignment.to_csv(config.output_dir / "alignment_assumption_check.csv", index=False)
    ratio_sweep.to_csv(config.output_dir / "alignment_ratio_sweep.csv", index=False)

    plot_channel_reliability(channel, config, config.output_dir)
    plot_score_pair_scatter(config, config.output_dir)
    plot_feature_space_geometry(config, config.output_dir)
    plot_population_metric_landscape(direction, marginal, config, config.output_dir)
    plot_linear_direction_sweep(direction, config, config.output_dir)
    plot_cap_vs_tpr(direction, config.output_dir)
    plot_marginal_shift_trap(marginal, config, config.output_dir)
    plot_marginal_shift_selector_sweep(marginal_selector, config, config.output_dir)
    plot_score_family_comparison(score_family, config, config.output_dir)
    plot_score_family_distributions(config, config.output_dir)
    plot_alignment_assumption_check(alignment, config.output_dir)
    plot_alignment_ratio_sweep(ratio_sweep, config, config.output_dir)
    write_metadata(config, config.output_dir)

    print_key_results(
        channel,
        direction,
        marginal,
        marginal_selector,
        score_family,
        alignment,
        ratio_sweep,
    )
    print("")
    print(f"Wrote artifacts to {config.output_dir}")


if __name__ == "__main__":
    main()
