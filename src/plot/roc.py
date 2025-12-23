# ROC curve plots.
from pathlib import Path
from pathvalidate import sanitize_filename

import matplotlib.pyplot as plt
import mplhep as hep


def plot(roc: dict, auroc: dict, metric: str, save_dir: Path):
    """Plots the ROC curve in the data.

    Expects two dictionaries: one with the roc data, i.e., tpr, fpr and thresholds;
    the other with the auroc data. The keys in these dictionaries correspond to the
    data set names. The metric is the name of the metric that is used as the
    anomaly score.
    """
    plt.style.use(hep.style.CMS)

    for ds_name in roc.keys():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=60)
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
