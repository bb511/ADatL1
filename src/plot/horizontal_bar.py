# Horizontal bar plots.
import math
from pathlib import Path
from pathvalidate import sanitize_filename

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

#: Colours of plot_comparison: the reference model (gamma = 0), the compared run and
#: the change between them.
REFERENCE_COLOR = "#eb6834"
RUN_COLOR = "#2a78d6"
CHANGE_COLOR = "#8a8984"


def filename_from_label(xlabel: str) -> str:
    """File stem of a plot with this x label (the rule every plot here uses)."""
    filename = xlabel.replace("\n", "__")
    filename = sanitize_filename(filename)
    return filename.replace(" ", "_")


def plot(data: dict, xlabel: str, save_dir: Path):
    """Plots the data in a horizontal bar plot.

    Expects a dictionary with data set names corresponding to number entries. The y-axis
    will be the data set names while the x-axis are the corresponding numbers.
    """
    plt.style.use(hep.style.CMS)

    dataset_names = list(data.keys())
    dataset_values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 16), dpi=120)

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    filename = filename_from_label(xlabel)
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def plot_yright(
    data: dict,
    ydata: dict,
    xlabel: str,
    ylabel: str,
    save_dir: Path,
    percent: bool = False,
):
    """Plots the data in a horizontal bar plot.

    Expects a dictionary with data set names corresponding to number entries. The y-axis
    will be the data set names while the x-axis are the corresponding numbers.

    This also puts values on the right of the y-axis, contained in ydata.
    """
    plt.style.use(hep.style.CMS)

    dataset_names = list(data.keys())
    dataset_values = list(data.values())
    dataset_yvals = list(ydata.values())
    if percent:
        dataset_yvals = [f"{yval * 100:.2f}%" for yval in dataset_yvals]
    else:
        dataset_yvals = [round(yval, 4) for yval in dataset_yvals]

    fig, ax = plt.subplots(figsize=(8, 16), dpi=120)

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(dataset_values) + 0.01)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    ax2 = ax.twinx()

    # Match the ticks using the same index positions
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_yticklabels(dataset_yvals)
    ax2.set_ylabel(ylabel)

    filename = filename_from_label(xlabel)
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def _value_text(value: float, percent: bool) -> str:
    """A bar value as plot_yright prints it."""
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%" if percent else f"{round(value, 4)}"


def relative_change(reference: float, run: float) -> float | None:
    """``100 * (run - reference) / reference`` in percent; None if undefined."""
    if not (math.isfinite(reference) and math.isfinite(run)) or reference == 0:
        return None
    return 100.0 * (run - reference) / reference


def plot_comparison(
    reference: dict,
    run: dict,
    xlabel: str,
    save_dir: Path,
    *,
    reference_label: str,
    run_label: str,
    title: str | None = None,
    percent: bool = False,
    filename: str | None = None,
) -> Path:
    """Two models side by side in the style of :func:`plot_yright`, plus their change.

    Left: per key (data set, checkpoint, ...) an orange bar for ``reference`` (e.g.
    the gamma = 0 run) above a blue bar for ``run``, each with its value printed at
    the bar end (in percent with ``percent``). Right, in gray and titled "Change":
    ``100 * (run - reference) / reference`` per key, "n/a" where the reference is 0.
    Only keys present in both dictionaries are drawn, in the order of ``run``; as in
    :func:`plot_yright` the first key is at the bottom.

    Saved as ``save_dir/<filename>.jpg``; ``filename`` defaults to the stem
    :func:`plot_yright` would use for ``xlabel``. Returns the path.
    """
    keys = [key for key in run if key in reference]
    ref_values = np.array([float(reference[key]) for key in keys], dtype=float)
    run_values = np.array([float(run[key]) for key in keys], dtype=float)
    changes = [relative_change(r, v) for r, v in zip(ref_values, run_values)]
    n = len(keys)

    height = max(7.0, 1.0 * n + 4.5)
    with plt.style.context(hep.style.CMS):
        fig, (ax, ax_change) = plt.subplots(
            1, 2, sharey=True, figsize=(16, height), dpi=120,
            gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.04},
        )
        fig.subplots_adjust(top=1 - 2.1 / height)

        y = np.arange(n)
        bar = 0.4
        ax.barh(y + bar / 2, ref_values, height=bar, color=REFERENCE_COLOR, label=reference_label)
        ax.barh(y - bar / 2, run_values, height=bar, color=RUN_COLOR, label=run_label)
        ax.set_yticks(y)
        ax.set_yticklabels([str(key) for key in keys])
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel(xlabel)

        finite = np.concatenate([ref_values, run_values])
        finite = finite[np.isfinite(finite)]
        low = min(0.0, float(finite.min())) if finite.size else 0.0
        high = max(0.0, float(finite.max())) if finite.size else 0.0
        span = (high - low) or 1.0
        ax.set_xlim(low - (0.05 * span if low < 0 else 0.0), high + 0.32 * span)
        for offset, values in ((bar / 2, ref_values), (-bar / 2, run_values)):
            for yi, value in zip(y, values):
                position = value if math.isfinite(value) else 0.0
                ax.text(position + 0.01 * span, yi + offset, _value_text(value, percent),
                        ha="left", va="center", fontsize=14)
        hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

        shown = [c if c is not None else 0.0 for c in changes]
        ax_change.barh(y, shown, height=0.6, color=CHANGE_COLOR)
        ax_change.axvline(0, color="black", lw=1)
        reach = max([abs(c) for c in shown] + [1.0])
        ax_change.set_xlim(-2.0 * reach, 2.0 * reach)
        for yi, change in zip(y, changes):
            text = "n/a" if change is None else f"{change:+.1f}%"
            right = change is None or change >= 0
            ax_change.text((change or 0.0) + (0.04 if right else -0.04) * reach, yi, text,
                           ha="left" if right else "right", va="center", fontsize=14)
        ax_change.set_title("Change", fontsize=22, pad=12)
        ax_change.set_xlabel("change [%]")
        ax_change.tick_params(axis="y", which="both", left=False, right=False,
                              labelleft=False)

        fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=18,
                   bbox_to_anchor=(0.5, 1 - 0.85 / height))
        if title:
            fig.suptitle(title, y=1 - 0.05 / height, va="top", fontsize=18, fontweight="bold")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{filename or filename_from_label(xlabel)}.jpg"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "horizontal_bar_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_counts = {
        "normal": 0.84,
        "signal_A": 0.61,
        "signal_B": 0.43,
    }
    sample_efficiencies = {
        "normal": 0.98,
        "signal_A": 0.76,
        "signal_B": 0.52,
    }

    plot(sample_counts, xlabel="Event count", save_dir=output_dir)
    plot_yright(
        sample_counts,
        sample_efficiencies,
        xlabel="Selected event count",
        ylabel="Efficiency",
        save_dir=output_dir,
        percent=True,
    )

    expected_paths = [
        output_dir / "Event_count.jpg",
        output_dir / "Selected_event_count.jpg",
    ]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected horizontal bar plot at {output_path}"
    print(f"Manual horizontal-bar tests passed. Plots saved to {output_dir}")
