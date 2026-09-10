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


if __name__ == "__main__":
    sample_x = np.array([0.95, 0.81, 0.70, 0.62, 0.57])
    sample_y = np.array([0.12, 0.18, 0.27, 0.39, 0.51])
    sample_epochs = np.arange(1, len(sample_x) + 1)

    title = "Manual 2D loss plot"
    TwoDPlotter(sample_x, sample_y, sample_epochs).plot(title)
    plt.close("all")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots"
    output_paths = list(output_dir.glob(f"{title}_*.png"))
    assert output_paths, f"Expected 2D plot in {output_dir}"
    print(f"Manual 2D-plot test passed. Plot saved to {max(output_paths)}")
