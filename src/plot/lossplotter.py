import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from src.analysis.dataloader import DataLoader
from datetime import datetime

LOSS_COLORS = {
    "Loss_reco": "firebrick",
    "Loss_mi": "royalblue",
    "Loss_total": "darkgray",
}


@dataclass(frozen=True)
class MetricSpecs:
    path: str | Path
    metric: str
    label: str | None = None


class Plotter():
    def __init__(self, metrics: list[MetricSpecs] | None = None) -> None:
        self.metrics: list[MetricSpecs] = list(metrics) if metrics is not None else []

        self.minimum: float = float("inf")
        self.maximum: float = float("-inf")
        self.epochs: int = 0
        print(f"Initialized Plotter with {len(self.metrics)} metric(s).")

    def add_metric(self, path: str | Path, metric: str, label: str | None = None) -> None:
        self.metrics.append(MetricSpecs(path=path, metric=metric, label=label))
        print(f"Added metric '{metric}' from {path}.")

    def set_epoch(self, epochs: int) -> None:
        self.epochs = epochs
        print(f"Set expected epochs to {epochs}.")

    def plot(self, title: str, ylable: str) -> None:
        if not self.metrics:
            raise ValueError("Cannot plot without metrics.")
        
        print(f"Creating plot '{title}' with {len(self.metrics)} metric(s).")
        self._find_max()
        self._find_min()


        runs_normalized = []

        for metric in self.metrics:
            values = self._load_metric_values(path=metric.path, metric=metric.metric)
            normalized = (values - values.min()) / (values.max() - values.min())
            runs_normalized.append((metric.label, normalized))

        plt.figure(figsize=(10,4))

        epochs = range(1, self.epochs + 1)

        for name, values in runs_normalized:
            if len(values) != self.epochs:
                raise ValueError(
                    f"Expected {self.epochs} epochs for '{name}', "
                    f"but found {len(values)} values."
                )

            color = next(
                (
                    loss_color
                    for loss_name, loss_color in LOSS_COLORS.items()
                    if loss_name in name
                ),
                None,
            )
            plt.plot(epochs, values, color=color, label=name)


        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylable)
        plt.ylim(0, 1)
        plt.legend()
        repo_root = Path(__file__).resolve().parents[2]
        save_dir = repo_root / "logs" / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{title}_{timestamp}.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}.")



    def _find_max(self) -> None:
        if not self.metrics:
            raise ValueError("Cannot find maximum without metrics.")

        for metric in self.metrics:

            x = self._load_metric_values(path=metric.path, metric=metric.metric)
            metric_max = x.max()
            
            if  metric_max > self.maximum:
                self.maximum = metric_max
    
    def _find_min(self) -> float:
        if not self.metrics:
            raise ValueError("Cannot find maximum without metrics.")

        for metric in self.metrics:
        
            x = self._load_metric_values(path=metric.path, metric=metric.metric)
            metric_min = x.min()
            
            if  metric_min < self.minimum:
                self.minimum = metric_min

    def _load_metric_values(self, path: str | Path, metric: str) -> pd.Series:
        print(f"Loading metric '{metric}' from {path}.")
        data = DataLoader(path=path, metric=metric).load()
        return pd.to_numeric(data.squeeze(), errors="raise")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    fixture_dir = repo_root / "logs" / "plots" / "lossplotter_manual_test"

    sample_runs = {
        "run_a": [1.0, 0.82, 0.69, 0.61, 0.56],
        "run_b": [1.1, 0.91, 0.77, 0.66, 0.60],
    }
    for run_name, values in sample_runs.items():
        metric_path = fixture_dir / run_name / "metrics" / "train" / "loss_reco"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_lines = [f"{index} {value} {index}" for index, value in enumerate(values)]
        metric_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    plotter = Plotter()
    plotter.add_metric(fixture_dir / "run_a", "loss_reco", "Run A")
    plotter.add_metric(fixture_dir / "run_b", "loss_reco", "Run B")
    plotter.set_epoch(5)

    title = "Manual normalized loss"
    plotter.plot(title, "Normalized loss")
    plt.close("all")

    output_dir = repo_root / "logs" / "plots"
    output_paths = list(output_dir.glob(f"{title}_*.png"))
    assert output_paths, f"Expected normalized-loss plot in {output_dir}"
    assert plotter.minimum == min(min(values) for values in sample_runs.values())
    assert plotter.maximum == max(max(values) for values in sample_runs.values())
    print(f"Manual loss-plotter test passed. Plot saved to {max(output_paths)}")
