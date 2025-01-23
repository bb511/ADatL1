import pytorch_lightning as pl
import torch
import numpy as np

class SyntheticQVAEDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        input_features: int = 20,  # Match the input features of your encoder
        num_samples: int = 10000,
        batch_size: int = 64,
        train_split: float = 0.8,
        seed: int = 42
    ):
        super().__init__()
        
        # Save parameters
        self.input_features = input_features
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.train_split = train_split
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate synthetic dataset
        self.generate_dataset()
        
    def generate_dataset(self):
        """
        Generate synthetic data that mimics real-world data
        - Creates a dataset with multiple underlying patterns
        - Adds some noise to make it more realistic
        """
        # Generate base data with multiple underlying patterns
        base_data = torch.zeros(self.num_samples, self.input_features)
        
        # Create multiple independent patterns
        for i in range(self.input_features):
            # Different pattern types: sine waves, exponentials, linear trends
            if i % 4 == 0:
                # Sine wave
                base_data[:, i] = torch.sin(torch.linspace(0, 10, self.num_samples) * (i+1))
            elif i % 4 == 1:
                # Exponential decay
                base_data[:, i] = torch.exp(-torch.linspace(0, 5, self.num_samples) * (i+1))
            elif i % 4 == 2:
                # Linear trend with noise
                base_data[:, i] = torch.linspace(0, 5, self.num_samples) + torch.randn(self.num_samples) * 0.5
            else:
                # Random gaussian distribution
                base_data[:, i] = torch.randn(self.num_samples)
        
        # Add some noise
        base_data += torch.randn_like(base_data) * 0.1
        
        # Normalize data
        self.data = (base_data - base_data.mean()) / base_data.std()
        
        # Split into train and validation
        train_size = int(self.num_samples * self.train_split)
        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:]
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def teardown(self, stage: str = None):
        """Clean up after fitting or testing"""
        pass