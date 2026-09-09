# Horizontal bar plots.
from pathlib import Path
from pathvalidate import sanitize_filename

import matplotlib.pyplot as plt
import mplhep as hep


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

    filename = xlabel.replace("\n", "__")
    filename = sanitize_filename(filename)
    filename = filename.replace(" ", "_")
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

    filename = xlabel.replace("\n", "__")
    filename = sanitize_filename(filename)
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


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
