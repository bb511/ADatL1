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

    def add_metric(self, path: str | Path, metric: str, label: str | None = None) -> None:
        self.metrics.append(MetricSpecs(path=path, metric=metric, label=label))

    def set_epoch(self, epochs: int) -> None:
        self.epochs = epochs

    def plot(self, title: str, ylable: str) -> None:
        if not self.metrics:
            raise ValueError("Cannot plot without metrics.")
        
        self._find_max()
        self._find_min()


        runs_normalized = [
            (metric.label, ((self._load_metric_values(path=metric.path, metric=metric.metric) - self.minimum) / (self.maximum - self.minimum)))
            for metric in self.metrics
        ]

        plt.figure(figsize=(10,4))

        epochs = range(1, self.epochs + 1)

        for name, values in runs_normalized:
            if len(values) != self.epochs:
                raise ValueError(
                    f"Expected {self.epochs} epochs for '{name}', "
                    f"but found {len(values)} values."
                )
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
        plt.savefig(save_dir / f"{title}_{timestamp}.png")



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
        data = DataLoader(path=path, metric=metric).load()
        return pd.to_numeric(data.squeeze(), errors="raise")
    
