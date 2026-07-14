import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import DataLoader
from src.plot.twodplotter import TwoDPlotter

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[2]
    data_source = Path("logs/mlflow/mlruns/573861611623376687")
    data = Path(repo_root / data_source / "926e737ce2794955b018a86b3c7614f6")
    # metric3 = MetricSpecs(data_2, "loss_mi", "Loss_mi: Gamma = 0.1")

    plot = TwoDPlotter(
        DataLoader(data, "loss_reco").load(), 
        DataLoader(data, "loss_mi").load(), 
        range(1, 50 + 1)
        ).plot("2D: Loss_reco vs loss_mi for Gamma = 0.1")
