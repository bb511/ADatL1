from plotter import Plotter, MetricSpecs
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_source = Path("logs/mlflow/mlruns/573861611623376687")

    data_1 = Path(repo_root / data_source / "ee1871645c54491c8c862897bde4c3bc")
    metric1 = MetricSpecs(data_1, "loss_reco", "Loss_reco: Gamma = 0.0 Run 2")

    data_2 = Path(repo_root / data_source / "926e737ce2794955b018a86b3c7614f6")
    metric2 = MetricSpecs(data_2, "loss_reco", "Loss_reco: Gamma = 0.1 Run 2")
    # metric3 = MetricSpecs(data_2, "loss_mi", "Loss_mi: Gamma = 0.1")

    data_4 = Path(repo_root / data_source / "5f291c162300412083fda956d0e3d359")
    metric4 = MetricSpecs(data_1, "loss_reco", "Loss_reco: Gamma = 0.0 Run 1")



    plotter = Plotter([metric1, metric2, metric4])
    # plotter.add_metric("data.csv", "metric", "test")
    plotter.set_epoch(50)
    plotter.plot("Loss_reco for Gamma = 0.0 and Gamma = 0.1", "loss")