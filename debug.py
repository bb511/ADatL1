#!/usr/bin/env python3
"""
Minimal debug script to test ApproximationCapacityCallback.
Assumes you can import your own callback and metric.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


from capmetric import ApproximationCapacity
from src.callbacks.cap import ValidationCAP


class SimpleDataModule(pl.LightningDataModule):
    """DataModule that returns the same dataloader for all stages"""
    
    def __init__(self, batch_size=16, num_samples=100, num_features=10):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_features = num_features
        
    def setup(self, stage=None):
        self.X = torch.randn(self.num_samples, self.num_features)
        self.Xp = torch.randn(self.num_samples, self.num_features)
        self.y = torch.randn(self.num_samples, 1)

    def _dataloader(self, prime: bool = False, shuffle: bool = True):
        return DataLoader(
            TensorDataset(self.X if not prime else self.Xp, self.y),
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def train_dataloader(self):
        return self._dataloader(False, True)
        
    def val_dataloader(self):
        return {"main_val": self._dataloader(True, True), "val_aux": self._dataloader(False, False)}
    
    def test_dataloader(self):
        return {"test_val": self._dataloader(True, True), "test_aux": self._dataloader(False, False)}


class IdentityModel(pl.LightningModule):
    """Identity model with a learnable parameter for optimization"""
    
    def __init__(self, num_features=10):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        return x
        
    def model_step(self, batch, batch_idx):
        """Common step function called by all other step methods"""
        x, y = batch
        loss = self.weight.sum() + torch.randn(len(y))
        output = self.forward(x)
        return {'loss': loss.mean(), 'loss_full': loss, 'output': output, 'y': y}
        
    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model_step(batch, batch_idx)
        
    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    """Main debug function"""
    
    # Set seed
    pl.seed_everything(42)
    
    # Create datamodule and model
    datamodule = SimpleDataModule(batch_size=16, num_samples=80, num_features=10)
    model = IdentityModel(num_features=10)
    
    # Create your callback here:
    cap_metric = ApproximationCapacity(
        beta0=1.0,
        normalization_type="minmax",
        normalization_params=None,
        energy_type="baseline",
        energy_params=None,
        regularization_type="none",
        regularization_params=None,
        binary=True,
        lr=0.01,
        n_epochs=100,
        batch_size=128,
        process_group=None,
        dist_sync_fn=None
    )
    callback = ValidationCAP(
        capmetric=cap_metric,
        pairing_type="cdf",
        output_name="loss_full",
        log_every_n_epochs=1,
        data_pairs={
            "background": ["main_val", "val_aux"],
            "simulation": ["test_val", "test_aux"]
        }
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[callback],
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule)
    print("Training completed!")


if __name__ == "__main__":
    main()