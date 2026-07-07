from plotter import Plotter, MetricSpecs
from pathlib import Path


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data.csv"
    print(f"data path: {data_path}")
    plotter = Plotter([MetricSpecs(data_path, "loss", "test")])
    # plotter.add_metric("data.csv", "metric", "test")
    plotter.set_epoch(10)
    plotter.plot("TestPlot", "TestData")