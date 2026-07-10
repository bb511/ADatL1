import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from dataloader import DataLoader
from datetime import datetime

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
        print(f"Global normalization range: min={self.minimum}, max={self.maximum}.")


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
            print(f"Plotting '{name}' with {len(values)} values.")

            if "Loss_reco: Gamma = 0.0 Run 2" in name:
                plt.plot(epochs, values, label=name, marker="*")
            else:
                plt.plot(epochs, values, label=name)


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
    data_source = Path("logs/mlflow/mlruns/573861611623376687")

    data_1 = Path(repo_root / data_source / "9f28cb695d2b4c438960dd62603d2209")
    metric1 = MetricSpecs(data_1, "loss_mi", "Loss_mi: Gamma = 0.1 Bins = 64 Run 1")

    data_2 = Path(repo_root / data_source / "926e737ce2794955b018a86b3c7614f6")
    metric2 = MetricSpecs(data_2, "loss_mi", "Loss_mi: Gamma = 0.1 Bins = 50 Run 2")
    # metric3 = MetricSpecs(data_2, "loss_mi", "Loss_mi: Gamma = 0.1")

    # data_4 = Path(repo_root / data_source / "5f291c162300412083fda956d0e3d359")
    # metric4 = MetricSpecs(data_1, "loss_reco", "Loss_reco: Gamma = 0.0 Run 1")



    plotter = Plotter([metric1, metric2])
    # plotter.add_metric("data.csv", "metric", "test")
    plotter.set_epoch(50)
    plotter.plot("Loss_mi for bins = 50 vs bins = 64", "loss")