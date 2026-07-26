"""Run an exploratory linear-Gaussian bridge on real Causal Chamber readouts.

The confirmatory Causal Chamber campaign is not modified.  This post-confirmatory
mechanistic study first audits the full readout distribution, then constructs a
restricted two-dimensional approximation from the six optical readouts:

1. fit marginal rank-Gaussian transforms on the frozen normal training split;
2. fit a whitened PCA subspace on the same split;
3. compute CAP over linear directions using metadata-matched held-out normal
   views; and
4. compare the CAP-selected direction with matched-strength red and blue real
   interventions.

The two interventions instantiate the analytical alignment stress test: red is
approximately aligned with the reproducible direction, while blue is nearly
orthogonal.  Exact Gaussianity is tested and is not assumed merely because the
restricted approximation is useful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analytical import cap_lift_for_score_matrix, cap_lift_quadrature  # noqa: E402
from src.data.components.causal_chamber import (  # noqa: E402
    READOUT_FEATURES,
    CausalChamberDataBuilder,
)

OPTICAL_FEATURES = ("ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3")
INTERVENTIONS = ("uniform_red_mid", "uniform_blue_mid")
SEED = 314159


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(path: Path) -> Path:
    """Create an empty output directory without overwriting prior artifacts."""
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the production 60/20/20 normal split."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_rows, generator=generator).numpy()
    n_train = int(round(0.6 * n_rows))
    n_validation = int(round(0.2 * n_rows))
    return (
        order[:n_train],
        order[n_train : n_train + n_validation],
        order[n_train + n_validation :],
    )


def _signal_test_indices(n_rows: int, seed: int) -> np.ndarray:
    """Return the production 40% held-out signal indices."""
    n_validation = min(max(1, int(round(0.6 * n_rows))), n_rows - 1)
    generator = torch.Generator().manual_seed(seed + n_rows)
    return torch.randperm(n_rows, generator=generator).numpy()[n_validation:]


def _metadata_test_pair_indices(dataset_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the production metadata-nearest held-out test pairing."""
    builder = CausalChamberDataBuilder(
        dataset_dir=dataset_dir,
        dataset_name="lt_interventions_standard_v1",
        feature_set="readouts",
        feature_columns=None,
        signal_experiments=[],
        pairing_columns=None,
        pairing_strategy="metadata_nearest",
        train_fraction=0.6,
        val_fraction=0.2,
        reference_fraction=0.5,
        signal_val_fraction=0.6,
        normalize=True,
        robust_quantiles=(0.05, 0.95),
        clip_value=10.0,
        seed=seed,
    )
    builder.setup(
        stage="test",
        batch_size=512,
        max_val_batches=-1,
        train_shuffler=None,
    )
    normal = builder.main["test"].sample_id.numpy()
    reference = builder.aux["test"]["reference_normal"].sample_id.numpy()
    if (
        len(normal) != 1_000
        or len(reference) != 1_000
        or not np.equal(normal, np.floor(normal)).all()
        or not np.equal(reference, np.floor(reference)).all()
    ):
        raise ValueError("Unexpected metadata-paired test contract.")
    return normal.astype(np.int64), reference.astype(np.int64)


def _robust_normalize(
    train: np.ndarray,
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Apply the production median/q95-q05 normalization to several arrays."""
    center = np.median(train, axis=0)
    scale = np.maximum(
        np.quantile(train, 0.95, axis=0) - np.quantile(train, 0.05, axis=0),
        1.0e-6,
    )
    return tuple(np.clip((array - center) / scale, -10.0, 10.0) for array in arrays)


def full_readout_audit(
    reference: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit marginal and multivariate Gaussian adequacy of all 11 readouts."""
    raw = reference.loc[:, list(READOUT_FEATURES)].to_numpy(dtype=float)
    train, validation, test = _robust_normalize(
        raw[train_indices],
        (raw[train_indices], raw[validation_indices], raw[test_indices]),
    )
    rows = []
    for index, feature in enumerate(READOUT_FEATURES):
        values = train[:, index]
        normal_test = stats.normaltest(values)
        rows.append(
            {
                "feature": feature,
                "n_unique_full_reference": int(np.unique(raw[:, index]).size),
                "train_zero_fraction": float(np.mean(values == 0)),
                "train_skew": float(stats.skew(values)),
                "train_excess_kurtosis": float(stats.kurtosis(values)),
                "dagostino_k2": float(normal_test.statistic),
                "dagostino_p": float(normal_test.pvalue),
            }
        )

    gaussian = LedoitWolf().fit(train)
    delta = test - gaussian.location_
    mahalanobis = np.einsum("ni,ij,nj->n", delta, gaussian.precision_, delta)
    mahalanobis_test = stats.kstest(
        mahalanobis,
        stats.chi2(df=len(READOUT_FEATURES)).cdf,
    )
    mixture_scores = {}
    mixture_bic = {}
    for components in (1, 2, 4):
        mixture = GaussianMixture(
            components,
            covariance_type="full",
            reg_covar=1.0e-5,
            random_state=seed,
            max_iter=100,
            n_init=1,
        ).fit(train)
        mixture_scores[str(components)] = {
            "validation": float(mixture.score(validation)),
            "test": float(mixture.score(test)),
        }
        mixture_bic[str(components)] = float(mixture.bic(train))

    summary = {
        "n_dimensions": len(READOUT_FEATURES),
        "mahalanobis_chi_square_ks_d": float(mahalanobis_test.statistic),
        "mahalanobis_chi_square_ks_p": float(mahalanobis_test.pvalue),
        "mahalanobis_coverage": {
            str(quantile): float(
                np.mean(mahalanobis <= stats.chi2.ppf(quantile, df=len(READOUT_FEATURES)))
            )
            for quantile in (0.5, 0.9, 0.95, 0.99)
        },
        "covariance_condition_number": float(np.linalg.cond(gaussian.covariance_)),
        "gaussian_mixture_mean_log_likelihood": mixture_scores,
        "gaussian_mixture_bic": mixture_bic,
        "two_minus_one_component_test_log_likelihood": (
            mixture_scores["2"]["test"] - mixture_scores["1"]["test"]
        ),
        "exact_single_gaussian_supported": False,
    }
    return pd.DataFrame(rows), summary


def fit_optical_bridge(
    optical: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> tuple[QuantileTransformer, PCA, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Fit and audit the rank-Gaussian two-dimensional optical subspace."""
    transformer = QuantileTransformer(
        n_quantiles=1_000,
        output_distribution="normal",
        subsample=None,
        random_state=seed,
    ).fit(optical[train_indices])
    gaussianized = transformer.transform(optical)
    pca = PCA(n_components=2, whiten=True, random_state=seed).fit(gaussianized[train_indices])
    latent = pca.transform(gaussianized)
    train = latent[train_indices]
    validation = latent[validation_indices]
    test = latent[test_indices]

    gaussian = LedoitWolf().fit(train)
    delta = test - gaussian.location_
    mahalanobis = np.einsum("ni,ij,nj->n", delta, gaussian.precision_, delta)
    mahalanobis_test = stats.kstest(mahalanobis, stats.chi2(df=2).cdf)
    mixture_scores = {}
    for components in (1, 2):
        mixture = GaussianMixture(
            components,
            covariance_type="full",
            reg_covar=1.0e-5,
            random_state=seed,
            max_iter=100,
            n_init=1,
        ).fit(train)
        mixture_scores[str(components)] = {
            "validation": float(mixture.score(validation)),
            "test": float(mixture.score(test)),
        }

    conditional_rows = []
    transformed_train = gaussianized[train_indices]
    transformed_test = gaussianized[test_indices]
    for index, feature in enumerate(OPTICAL_FEATURES):
        others = [column for column in range(len(OPTICAL_FEATURES)) if column != index]
        linear = Ridge(alpha=1.0e-3).fit(
            transformed_train[:, others],
            transformed_train[:, index],
        )
        nonlinear = ExtraTreesRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=8,
            random_state=seed,
        ).fit(
            transformed_train[:, others],
            transformed_train[:, index],
        )
        linear_r2 = r2_score(
            transformed_test[:, index],
            linear.predict(transformed_test[:, others]),
        )
        nonlinear_r2 = r2_score(
            transformed_test[:, index],
            nonlinear.predict(transformed_test[:, others]),
        )
        conditional_rows.append(
            {
                "feature": feature,
                "linear_test_r2": float(linear_r2),
                "nonlinear_test_r2": float(nonlinear_r2),
                "nonlinear_minus_linear_test_r2": float(nonlinear_r2 - linear_r2),
            }
        )

    summary = {
        "features": list(OPTICAL_FEATURES),
        "n_dimensions": 2,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "mahalanobis_chi_square_ks_d": float(mahalanobis_test.statistic),
        "mahalanobis_chi_square_ks_p": float(mahalanobis_test.pvalue),
        "mahalanobis_coverage_95": float(np.mean(mahalanobis <= stats.chi2.ppf(0.95, df=2))),
        "gaussian_mixture_mean_log_likelihood": mixture_scores,
        "two_minus_one_component_test_log_likelihood": (
            mixture_scores["2"]["test"] - mixture_scores["1"]["test"]
        ),
        "median_linear_conditional_test_r2": float(
            np.median([row["linear_test_r2"] for row in conditional_rows])
        ),
        "median_nonlinear_minus_linear_test_r2": float(
            np.median([row["nonlinear_minus_linear_test_r2"] for row in conditional_rows])
        ),
        "interpretation": (
            "Useful approximate Gaussian-copula linear subspace; exact "
            "multivariate Gaussianity remains rejected."
        ),
    }
    return transformer, pca, latent, pd.DataFrame(conditional_rows), summary


def _directional_signal_metrics(
    normal: np.ndarray,
    signal: np.ndarray,
    weights: np.ndarray,
    fpr: float,
) -> dict[str, np.ndarray]:
    """Compute oriented one-sided detection metrics over linear directions."""
    normal_scores = normal @ weights.T
    signal_scores = signal @ weights.T
    orientation = np.where(
        signal_scores.mean(axis=0) - normal_scores.mean(axis=0) >= 0,
        1.0,
        -1.0,
    )
    normal_scores *= orientation
    signal_scores *= orientation
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(signal_scores))])
    auprc = np.asarray(
        [
            average_precision_score(
                labels,
                np.concatenate([normal_scores[:, column], signal_scores[:, column]]),
            )
            for column in range(weights.shape[0])
        ],
        dtype=float,
    )
    threshold = np.quantile(normal_scores, 1.0 - fpr, axis=0, method="higher")
    efficiency = np.mean(signal_scores >= threshold, axis=0)
    normal_std = np.maximum(normal_scores.std(axis=0, ddof=1), 1.0e-12)
    standardized_shift = (signal_scores.mean(axis=0) - normal_scores.mean(axis=0)) / normal_std
    gaussian_tpr = stats.norm.sf(stats.norm.isf(fpr) - standardized_shift)
    return {
        "auprc": auprc,
        "efficiency": efficiency,
        "standardized_mean_shift": standardized_shift,
        "gaussian_tpr": gaussian_tpr,
    }


def bridge_direction_sweep(
    dataset_dir: Path,
    reference: pd.DataFrame,
    transformer: QuantileTransformer,
    pca: PCA,
    latent_reference: np.ndarray,
    normal_indices: np.ndarray,
    reference_indices: np.ndarray,
    *,
    seed: int,
    n_angles: int,
    fpr: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Compute empirical/theoretical CAP and real-intervention direction sweeps."""
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    weights = np.column_stack([np.cos(angles), np.sin(angles)])
    view_1 = latent_reference[normal_indices]
    view_2 = latent_reference[reference_indices]
    scores_1 = view_1 @ weights.T
    scores_2 = view_2 @ weights.T
    betas = np.linspace(0.0, 8.0, 81)
    cap, _, best_beta = cap_lift_for_score_matrix(scores_1, scores_2, betas)
    reliability = np.asarray(
        [
            np.corrcoef(scores_1[:, column], scores_2[:, column])[0, 1]
            for column in range(n_angles)
        ],
        dtype=float,
    )
    theory_cap = np.asarray(
        [
            cap_lift_quadrature(
                float(np.clip(rho, -0.999999, 0.999999)),
                betas,
                quadrature_order=45,
            )[0]
            for rho in reliability
        ],
        dtype=float,
    )

    frame = pd.DataFrame(
        {
            "angle_index": np.arange(n_angles),
            "angle_degrees": np.degrees(angles),
            "weight_pc1": weights[:, 0],
            "weight_pc2": weights[:, 1],
            "cap_empirical": cap,
            "cap_gaussian_theory": theory_cap,
            "cap_best_beta": best_beta,
            "paired_score_correlation": reliability,
        }
    )
    signals: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    cap_index = int(np.argmax(cap))
    for intervention in INTERVENTIONS:
        path = dataset_dir / f"{intervention}.csv"
        raw = pd.read_csv(path, usecols=list(OPTICAL_FEATURES)).loc[:, list(OPTICAL_FEATURES)]
        values = raw.to_numpy(dtype=float)
        signal_indices = _signal_test_indices(len(values), seed)
        signal = pca.transform(transformer.transform(values[signal_indices]))
        signals[intervention] = signal
        metrics = _directional_signal_metrics(view_1, signal, weights, fpr)
        for name, values_by_angle in metrics.items():
            frame[f"{intervention}_{name}"] = values_by_angle

        shift = signal.mean(axis=0) - view_1.mean(axis=0)
        shift_norm = float(np.linalg.norm(shift))
        cap_alignment = float(abs(np.dot(shift, weights[cap_index])) / max(shift_norm, 1.0e-15))
        oracle_index = int(np.argmax(metrics["auprc"]))
        efficiency_index = int(np.argmax(metrics["efficiency"]))
        summaries[intervention] = {
            "n_test_signal": len(signal),
            "latent_mean_shift": shift.tolist(),
            "latent_mean_shift_norm": shift_norm,
            "latent_shift_angle_degrees_modulo_180": float(
                np.degrees(np.arctan2(shift[1], shift[0])) % 180.0
            ),
            "absolute_alignment_with_cap_direction": cap_alignment,
            "cap_direction_auprc": float(metrics["auprc"][cap_index]),
            "oracle_auprc": float(metrics["auprc"][oracle_index]),
            "oracle_auprc_angle_degrees": float(np.degrees(angles[oracle_index])),
            "cap_direction_efficiency": float(metrics["efficiency"][cap_index]),
            "oracle_efficiency": float(metrics["efficiency"][efficiency_index]),
            "oracle_efficiency_angle_degrees": float(np.degrees(angles[efficiency_index])),
            "cap_direction_gaussian_tpr": float(metrics["gaussian_tpr"][cap_index]),
            "cap_direction_standardized_mean_shift": float(
                metrics["standardized_mean_shift"][cap_index]
            ),
        }

    summary = {
        "n_angles": n_angles,
        "fpr": fpr,
        "n_normal_pairs": len(view_1),
        "cap_optimal_angle_degrees": float(np.degrees(angles[cap_index])),
        "cap_empirical_max": float(cap[cap_index]),
        "cap_empirical_min": float(cap.min()),
        "paired_score_correlation_at_cap_optimum": float(reliability[cap_index]),
        "paired_score_correlation_max": float(reliability.max()),
        "paired_score_correlation_max_angle_degrees": float(
            np.degrees(angles[int(np.argmax(reliability))])
        ),
        "empirical_theory_cap_spearman": float(stats.spearmanr(cap, theory_cap).statistic),
        "empirical_theory_cap_rmse": float(np.sqrt(np.mean((cap - theory_cap) ** 2))),
        "interventions": summaries,
        "selection_boundary": (
            "Matched-strength red/blue interventions are a post-confirmatory "
            "mechanistic alignment contrast, not a confirmatory hypothesis test."
        ),
    }
    return frame, signals, summary


def _plot_bridge(
    direction: pd.DataFrame,
    normal: np.ndarray,
    signals: dict[str, np.ndarray],
    summary: dict[str, Any],
    output: Path,
) -> None:
    """Plot latent geometry, CAP, AUPRC, and fixed-FPR efficiency."""
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    geometry, cap_axis, auprc_axis, efficiency_axis = axes.ravel()
    colors = {
        "uniform_red_mid": "#D55E00",
        "uniform_blue_mid": "#0072B2",
    }
    labels = {
        "uniform_red_mid": "red mid (aligned)",
        "uniform_blue_mid": "blue mid (near-orthogonal)",
    }
    geometry.scatter(
        normal[:, 0],
        normal[:, 1],
        s=8,
        c="#888888",
        alpha=0.16,
        label="normal",
    )
    for name, signal in signals.items():
        geometry.scatter(
            signal[:, 0],
            signal[:, 1],
            s=11,
            color=colors[name],
            alpha=0.28,
            label=labels[name],
        )
        shift = np.asarray(summary["interventions"][name]["latent_mean_shift"])
        geometry.arrow(
            0,
            0,
            shift[0],
            shift[1],
            color=colors[name],
            width=0.025,
            length_includes_head=True,
        )
    angle = math.radians(summary["cap_optimal_angle_degrees"])
    cap_weight = np.asarray([math.cos(angle), math.sin(angle)])
    geometry.arrow(
        0,
        0,
        2.5 * cap_weight[0],
        2.5 * cap_weight[1],
        color="black",
        width=0.025,
        length_includes_head=True,
    )
    geometry.plot([], [], color="black", linestyle="--", label="CAP direction")
    geometry.axhline(0, color="black", linewidth=0.5)
    geometry.axvline(0, color="black", linewidth=0.5)
    geometry.set_xlabel("Whitened optical PC1")
    geometry.set_ylabel("Whitened optical PC2")
    geometry.set_title("Real intervention geometry")
    geometry.legend(fontsize=8, loc="best")

    degrees = direction["angle_degrees"]
    cap_axis.plot(
        degrees,
        direction["cap_empirical"],
        color="black",
        label="empirical CAP",
    )
    cap_axis.plot(
        degrees,
        direction["cap_gaussian_theory"],
        color="#777777",
        linestyle="--",
        label="Gaussian theory from paired ρ",
    )
    cap_axis.set_ylabel("CAP lift")
    cap_axis.set_title("Normal-pair CAP direction sweep")
    cap_axis.legend(fontsize=8)

    for name in INTERVENTIONS:
        auprc_axis.plot(
            degrees,
            direction[f"{name}_auprc"],
            color=colors[name],
            label=labels[name],
        )
        efficiency_axis.plot(
            degrees,
            direction[f"{name}_efficiency"],
            color=colors[name],
            label=f"{labels[name]} empirical",
        )
        efficiency_axis.plot(
            degrees,
            direction[f"{name}_gaussian_tpr"],
            color=colors[name],
            linestyle="--",
            alpha=0.75,
            label=f"{labels[name]} Gaussian",
        )
    auprc_axis.set_ylabel("AUPRC")
    auprc_axis.set_title("Real-intervention directional detection")
    auprc_axis.legend(fontsize=8)
    efficiency_axis.set_ylabel("Efficiency")
    efficiency_axis.set_title("Frozen 1% FPR: empirical vs Gaussian shift model")
    efficiency_axis.legend(fontsize=7, ncol=2)

    for axis in (cap_axis, auprc_axis, efficiency_axis):
        axis.axvline(
            summary["cap_optimal_angle_degrees"],
            color="black",
            linestyle=":",
            linewidth=1,
            label="CAP optimum",
        )
        axis.set_xlim(0, 179)
        axis.set_xlabel("Linear direction (degrees, modulo sign)")
        axis.grid(alpha=0.2)
    figure.suptitle("Exploratory Causal Chamber linear-Gaussian bridge: alignment is the boundary")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(
    dataset_dir: Path,
    physical_shift: Path,
    output_dir: Path,
    *,
    seed: int = SEED,
    n_angles: int = 180,
    fpr: float = 0.01,
) -> list[Path]:
    """Run the assumption audit and exploratory real-intervention bridge."""
    dataset_dir = dataset_dir.expanduser().resolve()
    physical_shift = physical_shift.expanduser().resolve()
    output_dir = _prepare_output(output_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not physical_shift.is_file():
        raise FileNotFoundError(physical_shift)
    if n_angles < 18 or not 0 < fpr < 0.5:
        raise ValueError("n_angles must be >=18 and fpr must be in (0, 0.5).")

    input_paths = {
        "uniform_reference": dataset_dir / "uniform_reference.csv",
        **{intervention: dataset_dir / f"{intervention}.csv" for intervention in INTERVENTIONS},
        "physical_shift": physical_shift,
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = pd.read_csv(input_paths["uniform_reference"])
    train_indices, validation_indices, test_indices = _split_indices(len(reference), seed)
    normal_indices, paired_reference_indices = _metadata_test_pair_indices(dataset_dir, seed)
    if not set(normal_indices).issubset(set(test_indices)) or not set(
        paired_reference_indices
    ).issubset(set(test_indices)):
        raise ValueError("Metadata pairs do not belong to the frozen test split.")
    if set(normal_indices) & set(paired_reference_indices):
        raise ValueError("Normal and paired-reference rows overlap.")

    marginal, full_summary = full_readout_audit(
        reference,
        train_indices,
        validation_indices,
        test_indices,
        seed,
    )
    optical = reference.loc[:, list(OPTICAL_FEATURES)].to_numpy(dtype=float)
    transformer, pca, latent, conditional, optical_summary = fit_optical_bridge(
        optical,
        train_indices,
        validation_indices,
        test_indices,
        seed,
    )
    direction, signals, direction_summary = bridge_direction_sweep(
        dataset_dir,
        reference,
        transformer,
        pca,
        latent,
        normal_indices,
        paired_reference_indices,
        seed=seed,
        n_angles=n_angles,
        fpr=fpr,
    )

    physical = pd.read_csv(physical_shift).set_index("intervention")
    for intervention in INTERVENTIONS:
        if intervention not in physical.index:
            raise ValueError(f"Physical-shift table misses {intervention}.")
        direction_summary["interventions"][intervention]["physical_energy_distance"] = float(
            physical.loc[intervention, "biased_energy_distance"]
        )
        direction_summary["interventions"][intervention]["physical_class"] = str(
            physical.loc[intervention, "physical_class"]
        )

    outputs: list[Path] = []
    for name, table in (
        ("full_readout_marginal_audit.csv", marginal),
        ("optical_conditional_linearity.csv", conditional),
        ("bridge_direction_sweep.csv", direction),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    summary_path = output_dir / "bridge_summary.json"
    summary = {
        "schema_version": 1,
        "classification": "post_confirmatory_exploratory_mechanistic_bridge",
        "seed": seed,
        "full_readout_assumption_audit": full_summary,
        "restricted_optical_subspace_audit": optical_summary,
        "direction_sweep": direction_summary,
        "scientific_decision": (
            "Reject a full 11-readout single-Gaussian model. Retain only the "
            "rank-Gaussian two-PC optical subspace as an approximate bridge, "
            "with exact-Gaussian caveats and no confirmatory status."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(summary_path)

    figure = output_dir / "gaussian_bridge_alignment.png"
    _plot_bridge(
        direction,
        latent[normal_indices],
        signals,
        direction_summary,
        figure,
    )
    outputs.append(figure)

    provenance_path = output_dir / "bridge_provenance.json"
    provenance = {
        "schema_version": 1,
        "confirmatory_campaign_modified": False,
        "classification": "post_confirmatory_exploratory",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(provenance_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--physical-shift", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-angles", type=int, default=180)
    parser.add_argument("--fpr", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge and print each created artifact."""
    args = parse_args(argv)
    for path in run(
        args.dataset_dir,
        args.physical_shift,
        args.output_dir,
        seed=args.seed,
        n_angles=args.n_angles,
        fpr=args.fpr,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
