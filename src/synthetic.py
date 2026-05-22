"""Controlled synthetic experiments for the CAP anomaly-detection paper.

The script evaluates a Gaussian subspace benchmark with closed-form anomaly-score
distributions. It intentionally separates:

* agnostic validation quantities computed only on two typical domains;
* supervised downstream quantities computed only for reporting against anomalies.

Example:
    uv run python src/synthetic.py --output-dir results/synthetic_gaussian
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, ncx2, norm
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.synthetic import SyntheticL1ADDataModule
from src.evaluation.callbacks.metrics.cap.binary import get_pairing_fn
from src.evaluation.callbacks.metrics.cap.metric import ApproximationCapacity


@dataclass(frozen=True)
class ScoreAlgorithm:
    name: str
    description: str
    fn: Callable[[np.ndarray], np.ndarray]
    analytic_family: str
    dof: int | None = None
    noncentral_scale: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled Gaussian-subspace CAP experiments."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--anomaly-dim", type=int, default=0)
    parser.add_argument("--wrong-dim", type=int, default=1)
    parser.add_argument("--anomaly-shift", type=float, default=4.0)
    parser.add_argument("--reference-shift", type=float, default=0.0)
    parser.add_argument("--reference-shift-dim", type=int, default=1)
    parser.add_argument("--fpr", type=float, default=1e-3)
    parser.add_argument("--cap-samples", type=int, default=8192)
    parser.add_argument("--cap-epochs", type=int, default=15)
    parser.add_argument("--cap-lr", type=float, default=0.03)
    parser.add_argument("--cap-beta0", type=float, default=1.0)
    parser.add_argument("--cap-batch-size", type=int, default=4096)
    parser.add_argument(
        "--cap-normalization",
        choices=["sigmoid", "rank_mid", "rank", "log_sigmoid", "minmax"],
        default="sigmoid",
    )
    parser.add_argument(
        "--cap-energy",
        choices=["baseline", "adaptive", "focal", "margin", "contrastive"],
        default="baseline",
    )
    parser.add_argument(
        "--shift-grid",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        help="Anomaly mean shifts used for power curves.",
    )
    return parser.parse_args()


def make_datamodule(args: argparse.Namespace, anomaly_shift: float):
    return SyntheticL1ADDataModule(
        n_features=args.n_features,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        batch_size=args.batch_size,
        max_val_batches=None,
        seed=args.seed,
        paper_aliases=False,
        generator="gaussian_subspace",
        noise_std=1.0,
        reference_shift=args.reference_shift,
        reference_shift_dim=args.reference_shift_dim,
        anomaly_shift=anomaly_shift,
        anomaly_dim=args.anomaly_dim,
    )


def collect_loader_tensors(loader) -> np.ndarray:
    xs = []
    for batch in loader:
        x, _, _, _ = batch
        xs.append(x)
    return torch.cat(xs, dim=0).cpu().numpy()


def load_splits(args: argparse.Namespace, anomaly_shift: float) -> dict[str, np.ndarray]:
    dm = make_datamodule(args, anomaly_shift=anomaly_shift)
    dm.setup("test")
    loaders = dm.test_dataloader()
    return {
        "normal": collect_loader_tensors(loaders["normal"]),
        "reference_normal": collect_loader_tensors(loaders["reference_normal"]),
        "synthetic_signal": collect_loader_tensors(loaders["synthetic_signal"]),
    }


def build_score_algorithms(args: argparse.Namespace) -> list[ScoreAlgorithm]:
    a = int(args.anomaly_dim)
    w = int(args.wrong_dim)
    if a == w:
        raise ValueError("wrong_dim must differ from anomaly_dim.")

    return [
        ScoreAlgorithm(
            name="constant",
            description="Collapsed score, no event information.",
            fn=lambda x: np.zeros(x.shape[0], dtype=np.float64),
            analytic_family="constant",
        ),
        ScoreAlgorithm(
            name="linear_oracle",
            description="Neyman-Pearson score for a positive Gaussian mean shift.",
            fn=lambda x, a=a: x[:, a],
            analytic_family="linear_oracle",
        ),
        ScoreAlgorithm(
            name="linear_wrong",
            description="Linear score on a feature unrelated to the anomaly.",
            fn=lambda x, w=w: x[:, w],
            analytic_family="null",
        ),
        ScoreAlgorithm(
            name="negative_oracle",
            description="Correct feature with the wrong score orientation.",
            fn=lambda x, a=a: -x[:, a],
            analytic_family="linear_negative",
        ),
        ScoreAlgorithm(
            name="residual_oracle_r1",
            description="One-dimensional AE residual containing the anomaly feature.",
            fn=lambda x, a=a: x[:, a] ** 2,
            analytic_family="noncentral_chi2",
            dof=1,
            noncentral_scale=1.0,
        ),
        ScoreAlgorithm(
            name="residual_wrong_r1",
            description="One-dimensional AE residual on an unrelated feature.",
            fn=lambda x, w=w: x[:, w] ** 2,
            analytic_family="null_chi2",
            dof=1,
        ),
        ScoreAlgorithm(
            name="radial_all",
            description="Full radial energy over all low-dimensional features.",
            fn=lambda x: np.sum(x**2, axis=1),
            analytic_family="noncentral_chi2",
            dof=int(args.n_features),
            noncentral_scale=1.0,
        ),
    ]


def threshold_power(
    normal_scores: np.ndarray, anomaly_scores: np.ndarray, fpr: float
) -> tuple[float, float]:
    if np.allclose(normal_scores, normal_scores[0]):
        return float(normal_scores[0]), float("nan")

    threshold = float(np.quantile(normal_scores, 1.0 - fpr, method="higher"))
    tpr = float(np.mean(anomaly_scores >= threshold))
    return threshold, tpr


def threshold_drift(scores: np.ndarray, fpr: float, seed: int) -> float:
    if np.allclose(scores, scores[0]):
        return float("nan")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scores))
    n_cal = len(scores) // 2
    cal = scores[perm[:n_cal]]
    eva = scores[perm[n_cal:]]
    threshold = float(np.quantile(cal, 1.0 - fpr, method="higher"))
    p_hat = float(np.mean(eva >= threshold))
    eps = 0.5 / max(len(eva), 1)
    return abs(math.log((p_hat + eps) / (fpr + eps)))


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    xq = np.sort(np.asarray(x, dtype=np.float64))
    yq = np.sort(np.asarray(y, dtype=np.float64))
    if len(xq) == len(yq):
        return float(np.mean(np.abs(xq - yq)))

    n = min(len(xq), len(yq))
    q = np.linspace(0.0, 1.0, n)
    return float(
        np.mean(
            np.abs(
                np.quantile(xq, q, method="linear")
                - np.quantile(yq, q, method="linear")
            )
        )
    )


def paired_cap(
    normal_scores: np.ndarray,
    reference_scores: np.ndarray,
    args: argparse.Namespace,
) -> float:
    rng = np.random.default_rng(args.seed)
    n = min(len(normal_scores), len(reference_scores), args.cap_samples)
    idx_a = rng.choice(len(normal_scores), size=n, replace=False)
    idx_b = rng.choice(len(reference_scores), size=n, replace=False)

    s1 = torch.as_tensor(normal_scores[idx_a], dtype=torch.float32)
    s2 = torch.as_tensor(reference_scores[idx_b], dtype=torch.float32)
    pair_1, pair_2 = get_pairing_fn("cdf")(s1, s2)
    s1 = s1[pair_1]
    s2 = s2[pair_2]

    energy_params = {"scale": 0.5} if args.cap_energy == "adaptive" else {}
    metric = ApproximationCapacity(
        beta0=args.cap_beta0,
        normalization_type=args.cap_normalization,
        normalization_params={},
        energy_type=args.cap_energy,
        energy_params=energy_params,
        regularization_type="none",
        regularization_params={},
        binary=True,
        lr=args.cap_lr,
        n_epochs=args.cap_epochs,
        batch_size=args.cap_batch_size,
        device="cpu",
        normalize_gradients=True,
    )
    with torch.inference_mode(False), torch.enable_grad():
        metric.update(s1.requires_grad_(True), s2.requires_grad_(True))
    return float(metric.compute())


def analytic_tpr(algorithm: ScoreAlgorithm, delta: float, fpr: float) -> float | None:
    if algorithm.analytic_family == "linear_oracle":
        return float(norm.sf(norm.isf(fpr) - delta))
    if algorithm.analytic_family == "linear_negative":
        return float(norm.sf(norm.isf(fpr) + delta))
    if algorithm.analytic_family == "null":
        return float(fpr)
    if algorithm.analytic_family == "noncentral_chi2":
        assert algorithm.dof is not None
        threshold = chi2.isf(fpr, df=algorithm.dof)
        noncentrality = algorithm.noncentral_scale * delta**2
        return float(ncx2.sf(threshold, df=algorithm.dof, nc=noncentrality))
    if algorithm.analytic_family == "null_chi2":
        return float(fpr)
    return None


def evaluate_algorithms(
    splits: dict[str, np.ndarray], algorithms: list[ScoreAlgorithm], args: argparse.Namespace
) -> pd.DataFrame:
    rows = []
    y_true = np.concatenate(
        [
            np.zeros(len(splits["normal"]), dtype=np.int64),
            np.ones(len(splits["synthetic_signal"]), dtype=np.int64),
        ]
    )

    for algorithm in algorithms:
        s_normal = algorithm.fn(splits["normal"])
        s_reference = algorithm.fn(splits["reference_normal"])
        s_signal = algorithm.fn(splits["synthetic_signal"])
        threshold, tpr = threshold_power(s_normal, s_signal, args.fpr)
        supervised_scores = np.concatenate([s_normal, s_signal])

        try:
            auroc = float(roc_auc_score(y_true, supervised_scores))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(y_true, supervised_scores))
        except ValueError:
            auprc = float("nan")

        cap = paired_cap(s_normal, s_reference, args)
        rows.append(
            {
                "algorithm": algorithm.name,
                "description": algorithm.description,
                "cap": cap,
                "cap_lift_over_collapse": cap + math.log(2.0),
                "wasserstein_normal_reference": wasserstein_1d(s_normal, s_reference),
                "threshold_drift": threshold_drift(s_normal, args.fpr, args.seed),
                "threshold": threshold,
                "tpr_at_fpr": tpr,
                "analytic_tpr_at_fpr": analytic_tpr(
                    algorithm, args.anomaly_shift, args.fpr
                ),
                "auroc": auroc,
                "auprc": auprc,
            }
        )

    return pd.DataFrame(rows)


def evaluate_power_grid(
    args: argparse.Namespace, algorithms: list[ScoreAlgorithm]
) -> pd.DataFrame:
    rows = []
    selected = {
        "linear_oracle",
        "linear_wrong",
        "negative_oracle",
        "residual_oracle_r1",
        "residual_wrong_r1",
        "radial_all",
    }
    selected_algorithms = [a for a in algorithms if a.name in selected]

    for delta in args.shift_grid:
        splits = load_splits(args, anomaly_shift=float(delta))
        for algorithm in selected_algorithms:
            s_normal = algorithm.fn(splits["normal"])
            s_signal = algorithm.fn(splits["synthetic_signal"])
            _, tpr = threshold_power(s_normal, s_signal, args.fpr)
            rows.append(
                {
                    "algorithm": algorithm.name,
                    "anomaly_shift": float(delta),
                    "tpr_at_fpr": tpr,
                    "analytic_tpr_at_fpr": analytic_tpr(
                        algorithm, float(delta), args.fpr
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_score_distributions(
    splits: dict[str, np.ndarray], args: argparse.Namespace, output_dir: Path
) -> None:
    normal_scores = splits["normal"][:, args.anomaly_dim] ** 2
    signal_scores = splits["synthetic_signal"][:, args.anomaly_dim] ** 2

    xmax = float(np.quantile(signal_scores, 0.995))
    grid = np.linspace(0.0, xmax, 600)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(
        normal_scores,
        bins=120,
        range=(0.0, xmax),
        density=True,
        alpha=0.35,
        label="normal empirical",
    )
    ax.hist(
        signal_scores,
        bins=120,
        range=(0.0, xmax),
        density=True,
        alpha=0.35,
        label="anomaly empirical",
    )
    ax.plot(grid, chi2.pdf(grid, df=1), lw=2.0, label="normal chi2(1)")
    ax.plot(
        grid,
        ncx2.pdf(grid, df=1, nc=args.anomaly_shift**2),
        lw=2.0,
        label=f"anomaly nc-chi2(1, {args.anomaly_shift**2:.1f})",
    )
    ax.set_xlabel("residual_oracle_r1 score")
    ax.set_ylabel("density")
    ax.set_title("Closed-form score distributions")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=200)
    plt.close(fig)


def plot_agnostic_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    ordered = df.sort_values("cap_lift_over_collapse", ascending=False)
    labels = ordered["algorithm"].to_list()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].bar(x, ordered["cap_lift_over_collapse"])
    axes[0].set_ylabel("CAP + log 2")
    axes[0].set_title("Agnostic metrics on two typical domains")

    axes[1].bar(x, ordered["wasserstein_normal_reference"])
    axes[1].set_ylabel("W1 normal/reference")

    axes[2].bar(x, ordered["threshold_drift"])
    axes[2].set_ylabel("threshold drift")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "agnostic_metrics.png", dpi=200)
    plt.close(fig)


def plot_power_grid(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, group in df.groupby("algorithm"):
        group = group.sort_values("anomaly_shift")
        ax.plot(group["anomaly_shift"], group["tpr_at_fpr"], marker="o", label=name)
        if group["analytic_tpr_at_fpr"].notna().all():
            ax.plot(
                group["anomaly_shift"],
                group["analytic_tpr_at_fpr"],
                linestyle="--",
                alpha=0.65,
            )
    ax.set_xlabel("anomaly mean shift delta")
    ax.set_ylabel("TPR at target FPR")
    ax.set_title("Empirical power against analytic prediction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "power_vs_shift.png", dpi=200)
    plt.close(fig)


def plot_cap_vs_power(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(df["cap_lift_over_collapse"], df["tpr_at_fpr"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            row["algorithm"],
            (row["cap_lift_over_collapse"], row["tpr_at_fpr"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("CAP + log 2")
    ax.set_ylabel("TPR at target FPR")
    ax.set_title("CAP measures assignment capacity, not labels")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "cap_vs_power.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_splits(args, anomaly_shift=args.anomaly_shift)
    algorithms = build_score_algorithms(args)
    summary = evaluate_algorithms(splits, algorithms, args)
    power = evaluate_power_grid(args, algorithms)

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    power.to_csv(args.output_dir / "power_grid.csv", index=False)

    plot_score_distributions(splits, args, args.output_dir)
    plot_agnostic_metrics(summary, args.output_dir)
    plot_power_grid(power, args.output_dir)
    plot_cap_vs_power(summary, args.output_dir)

    metadata = {
        "n_features": args.n_features,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "seed": args.seed,
        "anomaly_dim": args.anomaly_dim,
        "wrong_dim": args.wrong_dim,
        "anomaly_shift": args.anomaly_shift,
        "reference_shift": args.reference_shift,
        "fpr": args.fpr,
        "cap": {
            "samples": args.cap_samples,
            "epochs": args.cap_epochs,
            "lr": args.cap_lr,
            "beta0": args.cap_beta0,
            "normalization": args.cap_normalization,
            "energy": args.cap_energy,
        },
        "artifacts": [
            "summary.csv",
            "power_grid.csv",
            "score_distributions.png",
            "agnostic_metrics.png",
            "power_vs_shift.png",
            "cap_vs_power.png",
        ],
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote synthetic study artifacts to {args.output_dir}")
    print(summary[["algorithm", "cap_lift_over_collapse", "tpr_at_fpr", "auroc"]])


if __name__ == "__main__":
    main()
