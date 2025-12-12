from typing import Dict
import torch

from src.utils import pylogger
from src.data.components.dataset import L1ADDataset
from src.data.L1AD_datamodule import L1ADDataModule

log = pylogger.RankedLogger(__name__)


class DebugL1ADDataModule(L1ADDataModule):
    """Small debugging datamodule that generates gaussian data of different means for every signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.sims = ["GluGluHToBB_M-125", "GluGluHToGG_M-125", "SingleNeutrino_E-10-gun", "SingleNeutrino_Pt-2To20-gun"]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        pass

    def train_dataloader(self) -> L1ADDataset:
        return L1ADDataset(
            data=torch.randn(1000, 57, dtype=torch.float32),
            labels=torch.cat([
                torch.zeros(500, dtype=torch.long),
                torch.ones(500, dtype=torch.long),
            ]),
            batch_size=100,
            shuffle=True,
        )
    
    def val_dataloader(self) -> Dict[str, L1ADDataset]:
        return self._main_signals_datasets("val")
    
    def test_dataloader(self) -> Dict[str, L1ADDataset]:
        return self._main_signals_datasets("test")
    
    def _main_signals_datasets(self, stage: str) -> Dict[str, L1ADDataset]:
        return {
            f"main_{stage}": L1ADDataset(
                data=torch.randn(200, 57, dtype=torch.float32),
                labels=torch.cat([
                    torch.zeros(100, dtype=torch.long),
                    torch.ones(100, dtype=torch.long),
                ]),
                batch_size=100,
                shuffle=False,
            ),
            **{
                signal_name: L1ADDataset(
                    data=float(isignal) * torch.randn(100, 57, dtype=torch.float32),
                    labels=torch.ones(100, dtype=torch.long),
                    batch_size=100,
                    shuffle=False,
                )
                for isignal, signal_name in enumerate(self.sims)
            }
        }