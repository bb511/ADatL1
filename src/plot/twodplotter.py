from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
        ax.set_title(title)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Epochs")
        repo_root = Path(__file__).resolve().parents[2]
        save_dir = repo_root / "logs" / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{title}_{timestamp}.png"
        plt.savefig(save_path)