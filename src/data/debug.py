from typing import Dict
import torch
import numpy as np
import json
from pathlib import Path

from src.utils import pylogger
from src.data.components.dataset import L1ADDataset
from src.data.L1AD_datamodule import L1ADDataModule

log = pylogger.RankedLogger(__name__)


class DebugL1ADDataModule(L1ADDataModule):
    """Small debugging datamodule that generates gaussian data of different means for every signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.bsim = ["SingleNeutrino_E-10-gun", "SingleNeutrino_Pt-2To20-gun"]
        self.ssim = [
            "GluGluHToBB_M-125",
            "GluGluHToGG_M-125",
            "GluGluHToGG_M-90",
            "GluGluHToTauTau_M-125",
        ]
        self.batch_size = 100
        self.ndata = {"train": 100, "val": 100, "test": 100}
        self.shufflers = {
            "train": None,
            "val": np.random.default_rng(seed=self.hparams.seed),
            "test": np.random.default_rng(seed=self.hparams.seed + 1),
        }

        # For compatibility with the real datamodule:
        self.normalizer = self.hparams.data_normalizer
        self.loader = self.hparams.data_awkward2torch

        mlready_path = Path("./data/data_2024E/mlready/eminimal_pdefault_default/robust_axov4")  
        with open(mlready_path / "object_feature_map.json", "r") as f:
            obj_feat_map = json.load(f)
        self.loader.object_feature_map = obj_feat_map

        for obj in obj_feat_map.keys():
            self.normalizer.import_norm_params(mlready_path / f"{obj}_norm_params.pkl", obj)
            
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        pass

    def train_dataloader(self) -> L1ADDataset:
        ntrain = self.ndata.get("train")
        return L1ADDataset(
            data=torch.randn(ntrain, 57, dtype=torch.float32),
            mask=torch.ones(ntrain, 57, dtype=torch.bool),
            labels=torch.zeros(ntrain, dtype=torch.long),
            batch_size=self.batch_size,
            shuffler=None,
        )

    def _dataset_dictionary(self, stage: str) -> Dict[str, L1ADDataset]:
        ndata = self.ndata.get(stage)
        return {
            f"main_{stage}": L1ADDataset(
                data=torch.randn(ndata, 57, dtype=torch.float32),
                mask=torch.ones(ndata, 57, dtype=torch.bool),
                labels=torch.zeros(ndata, dtype=torch.long),
                batch_size=self.batch_size,
                shuffler=self.shuffler,
            ),
            **{
                signal_name: L1ADDataset(
                    data=sign
                    * float(1 + isignal)
                    * torch.randn(ndata, 57, dtype=torch.float32),
                    mask=torch.ones(ndata, 57, dtype=torch.bool),
                    labels=torch.full((ndata,), sign * (1 + isignal), dtype=torch.long),
                    batch_size=self.batch_size,
                    shuffler=self.shuffler,
                )
                for sign, sims in zip([-1, 1], [self.bsim, self.ssim])
                for isignal, signal_name in enumerate(sims)
            },
        }

    def val_dataloader(self) -> Dict[str, L1ADDataset]:
        return self._dataset_dictionary("val")

    def test_dataloader(self) -> Dict[str, L1ADDataset]:
        return self._dataset_dictionary("test")
