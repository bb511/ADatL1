"""Test CAP's alignment intuition on un-Gaussianized Causal Chamber data.

This post-confirmatory bridge deliberately avoids rank-Gaussianization, PCA
truncation, and intervention-driven representation choices.  It:

1. applies only training-normal affine scaling and full-rank whitening;
2. removes only dimensions that are degenerate under a frozen normal-only rule;
3. freezes one linear CAP direction using metadata-matched validation-normal
   pairs, with an internal fit/selection split;
4. tests fixed-magnitude angular shifts on held-out real normal backgrounds; and
5. evaluates alignment against a held-out supervised linear reference for every
   real intervention.

The semi-synthetic component tests whether the analytical alignment mechanism
survives the real non-Gaussian background.  The all-intervention component asks
whether the same intuition describes real physical shifts.  Neither component
alters the confirmatory architecture study.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analytical import cap_lift_for_score_matrix  # noqa: E402
from src.data.components.causal_chamber import (  # noqa: E402
    READOUT_FEATURES,
    CausalChamberDataBuilder,
    parse_intervention_name,
)

SEED = 314159
BETAS = np.linspace(0.0, 8.0, 81)
DEGENERATE_MIN_UNIQUE = 10
DEGENERATE_MIN_ROBUST_SCALE = 1.0e-5
CAP_FIT_PAIRS = 600
CAP_RESTARTS = 64
CAP_STEPS = 500
CAP_LEARNING_RATE = 0.03
SYNTHETIC_SHIFT_NORM = 3.0
SYNTHETIC_ANGLES = np.linspace(0.0, 90.0, 19)
FPR = 0.01
N_RESAMPLES = 10_000


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
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n_rows, generator=generator).numpy()
    n_train = int(round(0.6 * n_rows))
    n_validation = int(round(0.2 * n_rows))
    return (
        order[:n_train],
        order[n_train : n_train + n_validation],
        order[n_train + n_validation :],
    )


def _interventions(dataset_dir: Path) -> list[str]:
    """Return every non-reference intervention name in stable order."""
    names = sorted(path.stem for path in dataset_dir.glob("*.csv"))
    names = [name for name in names if name != "uniform_reference"]
    if len(names) != 58:
        raise ValueError(f"Expected 58 interventions, found {len(names)}.")
    return names


def _builder(dataset_dir: Path, interventions: Sequence[str]) -> CausalChamberDataBuilder:
    """Build the frozen real-data splits with affine normalization and no clipping."""
    builder = CausalChamberDataBuilder(
        dataset_dir=dataset_dir,
        dataset_name="lt_interventions_standard_v1",
        feature_set="readouts",
        feature_columns=None,
        signal_experiments=list(interventions),
        pairing_columns=None,
        pairing_strategy="metadata_nearest",
        train_fraction=0.6,
        val_fraction=0.2,
        reference_fraction=0.5,
        signal_val_fraction=0.6,
        normalize=True,
        robust_quantiles=(0.05, 0.95),
        clip_value=None,
        seed=SEED,
    )
    builder.setup(
        stage=None,
        batch_size=512,
        max_val_batches=-1,
        train_shuffler=None,
    )
    return builder


def feature_audit(
    reference: pd.DataFrame,
    normalizer_contract: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Freeze the normal-only degeneracy rule and return retained dimensions."""
    train_indices, _, _ = _split_indices(len(reference), seed)
    centers = np.asarray(normalizer_contract["center"], dtype=float)
    scales = np.asarray(normalizer_contract["scale"], dtype=float)
    if len(centers) != len(READOUT_FEATURES) or len(scales) != len(READOUT_FEATURES):
        raise ValueError("Normalizer does not cover the exact readout feature contract.")
    rows = []
    keep = []
    for index, feature in enumerate(READOUT_FEATURES):
        values = reference.loc[train_indices, feature].to_numpy(dtype=float)
        unique = int(np.unique(values).size)
        retained = unique >= DEGENERATE_MIN_UNIQUE and scales[index] > DEGENERATE_MIN_ROBUST_SCALE
        reasons = []
        if unique < DEGENERATE_MIN_UNIQUE:
            reasons.append(f"train_unique<{DEGENERATE_MIN_UNIQUE}")
        if scales[index] <= DEGENERATE_MIN_ROBUST_SCALE:
            reasons.append(f"q95_minus_q05<={DEGENERATE_MIN_ROBUST_SCALE:g}")
        rows.append(
            {
                "feature": feature,
                "train_unique": unique,
                "affine_center": float(centers[index]),
                "affine_scale": float(scales[index]),
                "retained": bool(retained),
                "exclusion_reason": ";".join(reasons),
            }
        )
        keep.append(retained)
    keep_array = np.asarray(keep, dtype=bool)
    if keep_array.sum() < 2:
        raise ValueError("Normal-only degeneracy rule retained fewer than two readouts.")
    return pd.DataFrame(rows), keep_array


def fit_whitening(
    normalized_train: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a full-rank affine Ledoit-Wolf whitening transform."""
    retained = np.asarray(normalized_train, dtype=float)[:, keep]
    covariance = LedoitWolf().fit(retained)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
        raise ValueError("Shrinkage covariance is not finite positive definite.")
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    summary = {
        "n_input_readouts": int(len(keep)),
        "n_retained_readouts": int(keep.sum()),
        "n_discarded_readouts": int((~keep).sum()),
        "covariance_condition_number": float(np.linalg.cond(covariance.covariance_)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "transformation": (
            "training-normal robust affine scaling followed by full-rank "
            "Ledoit-Wolf whitening; no clipping, Gaussianization, or PCA truncation"
        ),
    }
    return covariance.location_, whitening, summary


def _transform(
    values: np.ndarray,
    keep: np.ndarray,
    location: np.ndarray,
    whitening: np.ndarray,
) -> np.ndarray:
    """Apply the frozen retained-feature affine whitening transform."""
    transformed = (np.asarray(values, dtype=float)[:, keep] - location) @ whitening
    if not np.isfinite(transformed).all():
        raise ValueError("Affine transformation produced non-finite values.")
    return transformed


def _cap_objective(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weights: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Return empirical CAP lift for several unit linear directions."""
    scores_1 = x1 @ weights.T
    scores_2 = x2 @ weights.T
    product = torch.tanh(0.5 * beta * scores_1) * torch.tanh(0.5 * beta * scores_2)
    return torch.log1p(product.clamp_min(-1.0 + 1.0e-12)).mean(dim=0)


def optimize_cap_direction(
    validation_1: np.ndarray,
    validation_2: np.ndarray,
    test_1: np.ndarray,
    test_2: np.ndarray,
    *,
    seed: int = SEED,
    n_fit_pairs: int = CAP_FIT_PAIRS,
    n_restarts: int = CAP_RESTARTS,
    n_steps: int = CAP_STEPS,
    learning_rate: float = CAP_LEARNING_RATE,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Fit candidate CAP directions and select one on held-out normal pairs."""
    validation_1 = np.asarray(validation_1, dtype=float)
    validation_2 = np.asarray(validation_2, dtype=float)
    test_1 = np.asarray(test_1, dtype=float)
    test_2 = np.asarray(test_2, dtype=float)
    if (
        validation_1.shape != validation_2.shape
        or test_1.shape != test_2.shape
        or validation_1.shape[1] != test_1.shape[1]
    ):
        raise ValueError("CAP direction arrays have incompatible shapes.")
    if not 1 <= n_fit_pairs < len(validation_1):
        raise ValueError("CAP fit pair count must leave a nonempty selection split.")
    if n_restarts < 2 or n_steps < 1 or learning_rate <= 0:
        raise ValueError("CAP optimizer settings are invalid.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(validation_1))
    fit_indices = order[:n_fit_pairs]
    selection_indices = order[n_fit_pairs:]
    torch_generator = torch.Generator().manual_seed(int(seed))
    parameters = torch.nn.Parameter(
        torch.randn(
            n_restarts,
            validation_1.shape[1],
            generator=torch_generator,
            dtype=torch.float64,
        )
    )
    beta_logit = torch.nn.Parameter(
        torch.full(
            (n_restarts,),
            math.log((1.0 / 8.0) / (1.0 - 1.0 / 8.0)),
            dtype=torch.float64,
        )
    )
    optimizer = torch.optim.Adam(
        [parameters, beta_logit],
        lr=float(learning_rate),
    )
    fit_1 = torch.as_tensor(validation_1[fit_indices], dtype=torch.float64)
    fit_2 = torch.as_tensor(validation_2[fit_indices], dtype=torch.float64)
    for _ in range(int(n_steps)):
        optimizer.zero_grad(set_to_none=True)
        weights = parameters / torch.linalg.vector_norm(parameters, dim=1, keepdim=True).clamp_min(
            1.0e-12
        )
        beta = 8.0 * torch.sigmoid(beta_logit)
        objective = _cap_objective(fit_1, fit_2, weights, beta)
        (-objective.mean()).backward()
        optimizer.step()

    weights = (
        (parameters / torch.linalg.vector_norm(parameters, dim=1, keepdim=True).clamp_min(1.0e-12))
        .detach()
        .cpu()
        .numpy()
    )
    fit_beta = (8.0 * torch.sigmoid(beta_logit)).detach().cpu().numpy()
    with torch.no_grad():
        fit_cap = (
            _cap_objective(
                fit_1,
                fit_2,
                torch.as_tensor(weights, dtype=torch.float64),
                torch.as_tensor(fit_beta, dtype=torch.float64),
            )
            .cpu()
            .numpy()
        )

    selection_scores_1 = validation_1[selection_indices] @ weights.T
    selection_scores_2 = validation_2[selection_indices] @ weights.T
    selection_cap, _, selection_beta = cap_lift_for_score_matrix(
        selection_scores_1,
        selection_scores_2,
        BETAS,
    )
    best = int(np.argmax(selection_cap))
    direction = weights[best]
    direction /= np.linalg.norm(direction)
    largest = int(np.argmax(np.abs(direction)))
    if direction[largest] < 0:
        direction = -direction

    test_cap, _, test_beta = cap_lift_for_score_matrix(
        (test_1 @ direction)[:, None],
        (test_2 @ direction)[:, None],
        BETAS,
    )
    random_order = np.random.default_rng(seed + 1).permutation(len(test_2))
    random_cap, _, _ = cap_lift_for_score_matrix(
        (test_1 @ direction)[:, None],
        (test_2[random_order] @ direction)[:, None],
        BETAS,
    )
    test_rho = float(stats.spearmanr(test_1 @ direction, test_2 @ direction).statistic)
    top_indices = np.argsort(selection_cap)[-min(10, n_restarts) :]
    top_cosines = np.abs(weights[top_indices] @ direction)
    candidates = pd.DataFrame(
        {
            "restart": np.arange(n_restarts),
            "fit_cap_at_optimized_beta": fit_cap,
            "fit_beta": fit_beta,
            "selection_cap": selection_cap,
            "selection_beta": selection_beta,
            "absolute_cosine_to_selected": np.abs(weights @ direction),
            "selected": np.arange(n_restarts) == best,
        }
    )
    summary = {
        "n_dimensions": int(validation_1.shape[1]),
        "n_validation_pairs": int(len(validation_1)),
        "n_fit_pairs": int(len(fit_indices)),
        "n_selection_pairs": int(len(selection_indices)),
        "n_test_pairs": int(len(test_1)),
        "n_restarts": int(n_restarts),
        "n_steps": int(n_steps),
        "learning_rate": float(learning_rate),
        "selected_restart": best,
        "selection_cap": float(selection_cap[best]),
        "selection_beta": float(selection_beta[best]),
        "test_cap": float(test_cap[0]),
        "test_beta": float(test_beta[0]),
        "test_random_pair_cap": float(random_cap[0]),
        "test_paired_spearman": test_rho,
        "top_restart_absolute_cosine_min": float(top_cosines.min()),
        "top_restart_absolute_cosine_median": float(np.median(top_cosines)),
        "intervention_labels_used": False,
    }
    return direction, candidates, summary


def _orthogonal_basis(direction: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis perpendicular to one direction."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    _, _, right = np.linalg.svd(direction[None, :], full_matrices=True)
    basis = right[1:]
    for row in basis:
        largest = int(np.argmax(np.abs(row)))
        if row[largest] < 0:
            row *= -1
    if not np.allclose(basis @ direction, 0.0, atol=1.0e-10):
        raise ValueError("Failed to construct a perpendicular basis.")
    return basis


def _score_metrics(
    normal: np.ndarray,
    signal: np.ndarray,
    validation_normal: np.ndarray,
    direction: np.ndarray,
    *,
    two_sided: bool,
    fpr: float,
) -> dict[str, float]:
    """Evaluate one frozen linear score with a validation-only threshold."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    center = float(np.median(validation_normal @ direction))

    def score(values: np.ndarray) -> np.ndarray:
        projected = np.asarray(values, dtype=float) @ direction - center
        return np.abs(projected) if two_sided else projected

    validation_scores = score(validation_normal)
    normal_scores = score(normal)
    signal_scores = score(signal)
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(signal_scores))])
    auprc = float(
        average_precision_score(
            labels,
            np.concatenate([normal_scores, signal_scores]),
        )
    )
    threshold = float(np.quantile(validation_scores, 1.0 - fpr, method="higher"))
    return {
        "auprc": auprc,
        "efficiency": float(np.mean(signal_scores >= threshold)),
        "threshold": threshold,
    }


def synthetic_alignment_sweep(
    cap_direction: np.ndarray,
    validation_normal: np.ndarray,
    test_normal: np.ndarray,
    signal_background: np.ndarray,
    *,
    shift_norm: float = SYNTHETIC_SHIFT_NORM,
    angles: np.ndarray = SYNTHETIC_ANGLES,
    fpr: float = FPR,
) -> pd.DataFrame:
    """Inject fixed-norm angular shifts into held-out real normal backgrounds."""
    if shift_norm <= 0 or len(signal_background) == 0:
        raise ValueError("Synthetic shift design is invalid.")
    cap_direction = np.asarray(cap_direction, dtype=float)
    cap_direction /= np.linalg.norm(cap_direction)
    basis = _orthogonal_basis(cap_direction)
    rows = []
    for basis_index, orthogonal in enumerate(basis):
        for angle in np.asarray(angles, dtype=float):
            radians = math.radians(float(angle))
            shift_direction = math.cos(radians) * cap_direction + math.sin(radians) * orthogonal
            shift_direction /= np.linalg.norm(shift_direction)
            signal = signal_background + float(shift_norm) * shift_direction
            cap = _score_metrics(
                test_normal,
                signal,
                validation_normal,
                cap_direction,
                two_sided=True,
                fpr=fpr,
            )
            oracle = _score_metrics(
                test_normal,
                signal,
                validation_normal,
                shift_direction,
                two_sided=True,
                fpr=fpr,
            )
            rows.append(
                {
                    "orthogonal_basis_index": basis_index,
                    "angle_degrees": float(angle),
                    "absolute_alignment": float(abs(np.dot(cap_direction, shift_direction))),
                    "shift_norm": float(shift_norm),
                    "cap_auprc": cap["auprc"],
                    "oracle_auprc": oracle["auprc"],
                    "cap_efficiency": cap["efficiency"],
                    "oracle_efficiency": oracle["efficiency"],
                }
            )
    return pd.DataFrame(rows)


def real_intervention_alignment(
    builder: CausalChamberDataBuilder,
    intervention_names: Sequence[str],
    keep: np.ndarray,
    location: np.ndarray,
    whitening: np.ndarray,
    cap_direction: np.ndarray,
    *,
    fpr: float = FPR,
) -> pd.DataFrame:
    """Evaluate the frozen CAP direction and a held-out supervised linear reference."""
    validation_normal = _transform(
        builder.main["valid"].x.numpy(),
        keep,
        location,
        whitening,
    )
    oracle_normal_train = _transform(
        builder.aux["valid"]["reference_normal"].x.numpy(),
        keep,
        location,
        whitening,
    )
    test_normal = _transform(
        builder.main["test"].x.numpy(),
        keep,
        location,
        whitening,
    )
    baseline_auprc = 400.0 / 1_400.0
    cap_direction = np.asarray(cap_direction, dtype=float)
    cap_direction /= np.linalg.norm(cap_direction)
    rows = []
    for name in intervention_names:
        signal_validation = _transform(
            builder.aux["valid"][name].x.numpy(),
            keep,
            location,
            whitening,
        )
        signal_test = _transform(
            builder.aux["test"][name].x.numpy(),
            keep,
            location,
            whitening,
        )
        train_x = np.concatenate([oracle_normal_train, signal_validation], axis=0)
        train_y = np.concatenate(
            [
                np.zeros(len(oracle_normal_train), dtype=int),
                np.ones(len(signal_validation), dtype=int),
            ]
        )
        classifier = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2_000,
            random_state=SEED,
        ).fit(train_x, train_y)
        oracle_direction = classifier.coef_.reshape(-1).astype(float)
        oracle_direction /= np.linalg.norm(oracle_direction)
        cap = _score_metrics(
            test_normal,
            signal_test,
            validation_normal,
            cap_direction,
            two_sided=True,
            fpr=fpr,
        )
        oracle = _score_metrics(
            test_normal,
            signal_test,
            validation_normal,
            oracle_direction,
            two_sided=False,
            fpr=fpr,
        )
        validation_shift = signal_validation.mean(axis=0) - oracle_normal_train.mean(axis=0)
        validation_shift_norm = float(np.linalg.norm(validation_shift))
        test_shift = signal_test.mean(axis=0) - test_normal.mean(axis=0)
        test_shift_norm = float(np.linalg.norm(test_shift))
        info = parse_intervention_name(name)
        oracle_lift = oracle["auprc"] - baseline_auprc
        recovered_lift = (
            (cap["auprc"] - baseline_auprc) / oracle_lift if oracle_lift > 0.05 else math.nan
        )
        efficiency_fraction = (
            cap["efficiency"] / oracle["efficiency"] if oracle["efficiency"] >= 0.10 else math.nan
        )
        rows.append(
            {
                "intervention": name,
                "target": info["target"],
                "family": info["family"],
                "strength": info["strength"],
                "n_validation_signal": int(len(signal_validation)),
                "n_test_signal": int(len(signal_test)),
                "validation_mean_shift_norm": validation_shift_norm,
                "test_mean_shift_norm": test_shift_norm,
                "cap_validation_mean_shift_absolute_alignment": float(
                    abs(np.dot(cap_direction, validation_shift))
                    / max(validation_shift_norm, 1.0e-15)
                ),
                "cap_linear_reference_absolute_alignment": float(
                    abs(np.dot(cap_direction, oracle_direction))
                ),
                "cap_auprc": cap["auprc"],
                "linear_reference_auprc": oracle["auprc"],
                "auprc_baseline": baseline_auprc,
                "auprc_recovered_lift_fraction": recovered_lift,
                "cap_efficiency": cap["efficiency"],
                "linear_reference_efficiency": oracle["efficiency"],
                "efficiency_recovered_fraction": efficiency_fraction,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 58 or frame["intervention"].duplicated().any():
        raise ValueError("Real-intervention alignment table is incomplete.")
    return frame


def association_summary(
    frame: pd.DataFrame,
    alignment_column: str,
    outcome: str,
    *,
    seed: int,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, Any]:
    """Compute one-sided permutation and target-cluster bootstrap association."""
    selected = frame[np.isfinite(frame[alignment_column]) & np.isfinite(frame[outcome])].copy()
    if len(selected) < 10:
        raise ValueError(f"Too few eligible interventions for {outcome}.")
    alignment = selected[alignment_column].to_numpy(dtype=float)
    values = selected[outcome].to_numpy(dtype=float)
    observed = float(stats.spearmanr(alignment, values).statistic)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(int(n_resamples)):
        permuted = rng.permutation(values)
        rho = float(stats.spearmanr(alignment, permuted).statistic)
        exceedances += int(rho >= observed)
    permutation_p = (exceedances + 1.0) / (int(n_resamples) + 1.0)

    targets = np.asarray(sorted(selected["target"].astype(str).unique()))
    grouped = {target: selected[selected["target"].astype(str) == target] for target in targets}
    bootstrap = []
    for _ in range(int(n_resamples)):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        sample = pd.concat(
            [grouped[target] for target in sampled_targets],
            ignore_index=True,
        )
        rho = float(
            stats.spearmanr(
                sample[alignment_column],
                sample[outcome],
            ).statistic
        )
        if math.isfinite(rho):
            bootstrap.append(rho)
    if not bootstrap:
        raise ValueError(f"All target-cluster bootstrap draws failed for {outcome}.")
    draws = np.asarray(bootstrap, dtype=float)
    return {
        "alignment": alignment_column,
        "outcome": outcome,
        "n_interventions": int(len(selected)),
        "n_targets": int(len(targets)),
        "spearman_rho": observed,
        "one_sided_permutation_p": float(permutation_p),
        "cluster_bootstrap_median": float(np.median(draws)),
        "cluster_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "cluster_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "n_permutations": int(n_resamples),
        "n_cluster_bootstrap_effective": int(len(draws)),
    }


def _plot_bridge(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    associations: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    """Plot synthetic angular stress tests and all-intervention alignment."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
        layout="constrained",
    )
    primary_synthetic, secondary_synthetic, primary_real, secondary_real = axes.ravel()
    grouped = synthetic.groupby("angle_degrees", sort=True)
    angles = np.asarray(sorted(synthetic["angle_degrees"].unique()), dtype=float)
    for axis, cap_column, oracle_column, ylabel in (
        (primary_synthetic, "cap_auprc", "oracle_auprc", "AUPRC"),
        (
            secondary_synthetic,
            "cap_efficiency",
            "oracle_efficiency",
            "Efficiency at 1% FPR",
        ),
    ):
        cap_mean = grouped[cap_column].mean().reindex(angles).to_numpy()
        cap_low = grouped[cap_column].quantile(0.025).reindex(angles).to_numpy()
        cap_high = grouped[cap_column].quantile(0.975).reindex(angles).to_numpy()
        oracle_mean = grouped[oracle_column].mean().reindex(angles).to_numpy()
        oracle_low = grouped[oracle_column].quantile(0.025).reindex(angles).to_numpy()
        oracle_high = grouped[oracle_column].quantile(0.975).reindex(angles).to_numpy()
        axis.plot(angles, cap_mean, color="#0072B2", label="frozen CAP direction")
        axis.fill_between(angles, cap_low, cap_high, color="#0072B2", alpha=0.18)
        axis.plot(angles, oracle_mean, color="#D55E00", label="known shift direction")
        axis.fill_between(angles, oracle_low, oracle_high, color="#D55E00", alpha=0.15)
        axis.set_xlabel("Shift angle from CAP direction (degrees)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 90)
        axis.grid(alpha=0.2)
    primary_synthetic.set_title("Fixed-norm shifts on held-out real backgrounds")
    secondary_synthetic.set_title("Operating-point alignment stress test")
    primary_synthetic.legend(fontsize=8)

    scatter_specs = (
        (
            primary_real,
            "auprc_recovered_lift_fraction",
            "Recovered attainable AUPRC lift",
        ),
        (
            secondary_real,
            "efficiency_recovered_fraction",
            "Recovered oracle efficiency",
        ),
    )
    for axis, outcome, ylabel in scatter_specs:
        selected = real[np.isfinite(real[outcome])].copy()
        axis.scatter(
            selected["cap_validation_mean_shift_absolute_alignment"],
            selected[outcome],
            color="#0072B2",
            s=34,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
        summary = associations[outcome]
        axis.text(
            0.03,
            0.97,
            (
                f"ρ={summary['spearman_rho']:+.2f}, "
                f"p={summary['one_sided_permutation_p']:.3g}\n"
                f"target-bootstrap 95% CI "
                f"[{summary['cluster_bootstrap_ci_low']:+.2f}, "
                f"{summary['cluster_bootstrap_ci_high']:+.2f}]"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Absolute alignment: CAP vs validation mean shift")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    primary_real.set_title("All real interventions: primary endpoint")
    secondary_real.set_title("All real interventions: secondary endpoint")
    figure.suptitle(
        "Causal Chamber empirical alignment bridge: affine full-readout geometry",
        fontsize=16,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: int = SEED,
    n_resamples: int = N_RESAMPLES,
) -> list[Path]:
    """Run the redesigned empirical alignment bridge."""
    if int(seed) != SEED:
        raise ValueError(f"The frozen bridge requires seed {SEED}.")
    if n_resamples < 100:
        raise ValueError("At least 100 resamples are required.")
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = _prepare_output(output_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    names = _interventions(dataset_dir)
    input_paths = {
        "uniform_reference": dataset_dir / "uniform_reference.csv",
        **{name: dataset_dir / f"{name}.csv" for name in names},
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    builder = _builder(dataset_dir, names)
    if builder.normalizer is None:
        raise RuntimeError("Causal Chamber normalizer was not fitted.")
    reference = pd.read_csv(input_paths["uniform_reference"])
    audit, keep = feature_audit(
        reference,
        builder.normalizer.to_contract(),
        seed,
    )
    normalized_train = builder.main["train"].x.numpy()
    location, whitening, geometry = fit_whitening(normalized_train, keep)

    def transform_dataset(dataset: Any) -> np.ndarray:
        return _transform(dataset.x.numpy(), keep, location, whitening)

    validation_1 = transform_dataset(builder.main["valid"])
    validation_2 = transform_dataset(builder.aux["valid"]["reference_normal"])
    test_1 = transform_dataset(builder.main["test"])
    test_2 = transform_dataset(builder.aux["test"]["reference_normal"])
    cap_direction, candidates, cap_summary = optimize_cap_direction(
        validation_1,
        validation_2,
        test_1,
        test_2,
        seed=seed,
    )
    retained_features = audit.loc[audit["retained"], "feature"].tolist()
    weights = pd.DataFrame(
        {
            "feature": retained_features,
            "whitened_direction_weight": cap_direction,
        }
    )

    signal_background = test_2[:400]
    synthetic = synthetic_alignment_sweep(
        cap_direction,
        validation_1,
        test_1,
        signal_background,
    )
    real = real_intervention_alignment(
        builder,
        names,
        keep,
        location,
        whitening,
        cap_direction,
    )
    primary_associations = {
        outcome: association_summary(
            real,
            "cap_validation_mean_shift_absolute_alignment",
            outcome,
            seed=seed + index + 100,
            n_resamples=n_resamples,
        )
        for index, outcome in enumerate(
            ("auprc_recovered_lift_fraction", "efficiency_recovered_fraction")
        )
    }
    linear_reference_sensitivity = {
        outcome: association_summary(
            real,
            "cap_linear_reference_absolute_alignment",
            outcome,
            seed=seed + index + 200,
            n_resamples=n_resamples,
        )
        for index, outcome in enumerate(
            ("auprc_recovered_lift_fraction", "efficiency_recovered_fraction")
        )
    }

    outputs: list[Path] = []
    for filename, table in (
        ("feature_audit.csv", audit),
        ("cap_direction_candidates.csv", candidates),
        ("cap_direction_weights.csv", weights),
        ("synthetic_alignment_sweep.csv", synthetic),
        ("real_intervention_alignment.csv", real),
    ):
        path = output_dir / filename
        table.to_csv(path, index=False)
        outputs.append(path)

    synthetic_grouped = synthetic.groupby("angle_degrees", sort=True)
    aligned = synthetic_grouped.get_group(float(SYNTHETIC_ANGLES[0]))
    orthogonal = synthetic_grouped.get_group(float(SYNTHETIC_ANGLES[-1]))
    summary = {
        "schema_version": 1,
        "classification": "post_confirmatory_empirical_alignment_bridge",
        "scientific_question": (
            "Does CAP's alignment intuition survive a real non-Gaussian "
            "background and describe all real chamber interventions?"
        ),
        "seed": seed,
        "feature_geometry": {
            **geometry,
            "retained_features": retained_features,
            "discarded_features": audit.loc[~audit["retained"], "feature"].tolist(),
        },
        "cap_direction": cap_summary,
        "semi_synthetic": {
            "shift_norm": SYNTHETIC_SHIFT_NORM,
            "n_orthogonal_directions": int(synthetic["orthogonal_basis_index"].nunique()),
            "n_angles": int(synthetic["angle_degrees"].nunique()),
            "aligned_mean_cap_auprc": float(aligned["cap_auprc"].mean()),
            "orthogonal_mean_cap_auprc": float(orthogonal["cap_auprc"].mean()),
            "aligned_mean_cap_efficiency": float(aligned["cap_efficiency"].mean()),
            "orthogonal_mean_cap_efficiency": float(orthogonal["cap_efficiency"].mean()),
            "aligned_mean_oracle_auprc": float(aligned["oracle_auprc"].mean()),
            "orthogonal_mean_oracle_auprc": float(orthogonal["oracle_auprc"].mean()),
        },
        "all_real_interventions": {
            "n_interventions": int(len(real)),
            "primary_validation_mean_shift_associations": primary_associations,
            "linear_reference_alignment_sensitivity": linear_reference_sensitivity,
        },
        "claim_boundary": (
            "The semi-synthetic sweep is a causal stress test on real backgrounds; "
            "the all-intervention association is exploratory external validation. "
            "Neither establishes the exact Gaussian theorem."
        ),
    }
    summary_path = output_dir / "bridge_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(summary_path)

    figure = output_dir / "empirical_alignment_bridge.png"
    _plot_bridge(synthetic, real, primary_associations, figure)
    outputs.append(figure)

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
    provenance_path = output_dir / "bridge_provenance.json"
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge and print each created artifact."""
    args = parse_args(argv)
    for path in run(
        args.dataset_dir,
        args.output_dir,
        seed=args.seed,
        n_resamples=args.n_resamples,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
