# ROC curve plots.
from pathlib import Path
from pathvalidate import sanitize_filename

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


def plot(roc: dict, auroc: dict, metric: str, save_dir: Path):
    """Plots the ROC curve in the data.

    Expects two dictionaries: one with the roc data, i.e., tpr, fpr and thresholds;
    the other with the auroc data. The keys in these dictionaries correspond to the
    data set names. 'metric' is the name of the metric that is used as the
    anomaly score.
    """
    plt.style.use(hep.style.CMS)

    for ds_name in roc.keys():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fpr, tpr, thresh = roc[ds_name]
        auc = auroc[ds_name]
        ax.plot(fpr, tpr, label=f"{metric}\n AUC = {auc:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{ds_name}")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        # hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

        filename = sanitize_filename(f"{ds_name}_{metric}")
        filename = filename.replace(" ", "_")
        fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
        fig.clear()
        plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "roc_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_fpr = np.array([0.0, 0.02, 0.08, 0.2, 0.45, 1.0])
    sample_tpr = np.array([0.0, 0.35, 0.62, 0.81, 0.94, 1.0])
    sample_thresholds = np.linspace(1.0, 0.0, len(sample_fpr))
    plot(
        roc={"signal": (sample_fpr, sample_tpr, sample_thresholds)},
        auroc={"signal": 0.87},
        metric="loss_reco",
        save_dir=output_dir,
    )

    output_path = output_dir / "signal_loss_reco.jpg"
    assert output_path.is_file(), f"Expected ROC plot at {output_path}"
    print(f"Manual ROC test passed. Plot saved to {output_path}")
