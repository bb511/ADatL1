from typing import Any, Callable, Optional
from tqdm import tqdm

import torch
from torch import Tensor, nn
import torch.optim as optim

from torchmetrics.metric import Metric
from torch.utils.data import DataLoader, TensorDataset


class ApproximationCapacityKernel(nn.Module):
    """Computes the approximation capacity of a VAE."""
    def __init__(
            self,
            beta0: Optional[float] = None,
        ):
        super().__init__()
        beta0 = beta0 if beta0 else 1.0
        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        
        self.beta = torch.nn.Parameter(torch.tensor([beta0], dtype=torch.float), requires_grad=True)
        self.cap = torch.tensor([0.0], dtype=torch.float)
           
    def compute_energy(self, Lreco: Tensor, anomaly: int):
        """Assume anomaly in {0,1}^n and Lreco 0-1 normalized."""
        return anomaly * (1 - Lreco) + (1 - anomaly) * Lreco
    
    def compute_mutual_information(self, loss1: Tensor, loss2: Tensor, beta: Optional[float] = None):
        beta = self.beta if beta is None else beta

        # Energy:
        e1_0 = self.compute_energy(loss1, 0)
        e1_1 = self.compute_energy(loss1, 1)
        e2_0 = self.compute_energy(loss2, 0)
        e2_1 = self.compute_energy(loss2, 1)

        num = torch.log(torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1)))
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1)) * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
        mi = torch.sum(num - den, dim=0)
        return mi

    def forward(self, loss1: Tensor, loss2: Tensor):
        self.beta.data.clamp_(min=0.0)
        with torch.set_grad_enabled(True):
            cap = self.compute_mutual_information(loss1, loss2)
            self.cap = self.cap + cap.detach().to(self.cap.device)
            return cap

    def evaluate(self, loss1: Tensor, loss2: Tensor, beta: float):
        with torch.set_grad_enabled(False):
            cap = self.compute_mutual_information(loss1, loss2, beta=beta)
            self.cap = self.cap + cap
            return cap
    
    def reset(self):
        """Reset accumulated state if needed for a new computation sequence."""
        self.cap = torch.tensor([0.0], dtype=torch.float)
    
    @property
    def module(self):
        """Returns the kernel itself. It helps the kernel be accessed in both DDP and non-DDP mode."""
        return self


class CAP(Metric):
    """
    Approximation Capacity (CAP) Metric implemented with torchmetrics.
    
    Args:
        kernel: The ApproximationCapacityKernel instance
        dataset: The ReconstructionLoss dataset
        batch_size: Batch size for processing the dataset
        n_epochs: Number of optimization epochs
        learning_rate: Learning rate for beta optimization
        compute_on_step: Whether to compute the metric on each step
        dist_sync_on_step: Synchronize metric state across processes at each step
        process_group: DDP process group
        dist_sync_fn: Function to synchronize metrics across processes
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        beta0: Optional[float] = 1.0,
        lr: Optional[float] = 0.01,
        n_epochs: Optional[int] = 5,
        batch_size: Optional[int] = 64,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        self.kernel = ApproximationCapacityKernel(beta0=beta0)
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.batch_size = batch_size
        
        # Initialize CAP:
        self.add_state("cap", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, dataset_loss: TensorDataset, **kwargs):
        """Optimize beta parameter over multiple epochs."""

        self.reset()
        dev = self.kernel.beta.device
        optimizer = optim.Adam([self.kernel.beta], lr=self.learning_rate)
        
        dataloader = DataLoader(
            dataset_loss,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=0
        )
        # Optimization loop:
        best_beta = 0.0
        best_cap = - float('inf')
        for epoch in tqdm(range(self.n_epochs), desc="CAP Optimization Progress", unit="epoch"):
            self.kernel.reset() 

            with torch.set_grad_enabled(True):
                for batch in dataloader:
                    loss1, loss2 = batch[0].to(dev), batch[1].to(dev)
                    
                    # Maximize capacity
                    optimizer.zero_grad()
                    loss = - self.kernel(loss1, loss2)
                    loss.backward()
                    optimizer.step()
            
            # Update best beta if this epoch is better
            if self.kernel.cap.item() > best_cap:
                best_cap = self.kernel.cap.item()
                best_beta = self.kernel.beta.clone().detach()
            
        # Perform a final evaluation with the best beta
        self.is_optimized = torch.tensor(1, dtype=torch.bool)
        with torch.no_grad():
            self.kernel.reset()
            for batch in dataloader:
                loss1, loss2 = batch[0].to(dev), batch[1].to(dev)
                _ = self.kernel.evaluate(loss1, loss2, beta=best_beta) 

            # Set a value to the metric state
            self.cap.fill_(self.kernel.cap.item())
        return
    
    def compute(self) -> Tensor:
        """Return the computed CAP value."""
        return self.cap.item()