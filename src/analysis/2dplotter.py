import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import DataLoader
from datetime import datetime

class TwoDPlotter():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def plot(self, title: str):

        fig, ax = plt.subplots()

        scatter = ax.scatter(self.x, self.y, c=self.z, cmap="viridis")

        ax.set_xlabel("Loss_reco")
        ax.set_ylabel("Loss_mi")
        ax.set_title("Loss_reco vs loss_mi for Gamma 0.1")

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Epochs")
        repo_root = Path(__file__).resolve().parents[2]
        save_dir = repo_root / "logs" / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{title}_{timestamp}.png"
        plt.savefig(save_path)

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[2]
    data_source = Path("logs/mlflow/mlruns/573861611623376687")
    data = Path(repo_root / data_source / "926e737ce2794955b018a86b3c7614f6")
    # metric3 = MetricSpecs(data_2, "loss_mi", "Loss_mi: Gamma = 0.1")

    plot = TwoDPlotter(DataLoader(data, "loss_reco").load(), DataLoader(data, "loss_mi").load(), range(1, 50 + 1)).plot("2D: Loss_reco vs loss_mi for Gamma = 0.1")
