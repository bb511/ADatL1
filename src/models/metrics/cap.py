from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
import torch.optim as optim

from torchmetrics.metric import Metric
from torch.utils.data import DataLoader, Dataset


class ApproximationCapacityKernel(nn.Module):
    """Computes the approximation capacity of a VAE."""
    def __init__(
            self,
            beta0: Optional[float] = None,
            device: str = "cpu"
        ):
        super().__init__()
        beta0 = beta0 if beta0 else 1.0
        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        
        self.dev = device
        self.beta = torch.nn.Parameter(torch.tensor([beta0], dtype=torch.float), requires_grad=True).to(self.dev)
        self.cap = torch.tensor([0.0], dtype=torch.float).to(self.dev)
           
    def compute_energy(self, Lreco: Tensor, anomaly: int):
        """Assume anomaly in {0,1}^n and Lreco 0-1 normalized."""
        return anomaly * (1 - Lreco) + (1 - anomaly) * Lreco
    
    def compute_mutual_information(
            self,
            Lreco1: Tensor,
            Lreco2: Tensor,
            beta: Optional[float] = None
        ):
        beta = self.beta if beta is None else beta

        # Energy:
        e1_0 = self.compute_energy(Lreco1, 0)
        e1_1 = self.compute_energy(Lreco1, 1)
        e2_0 = self.compute_energy(Lreco2, 0)
        e2_1 = self.compute_energy(Lreco2, 1)

        num = torch.log(torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1)))
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1)) * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
        return torch.sum(num - den, dim=0)

    def forward(self, Lreco1: Tensor, Lreco2: Tensor):
        self.dev = Lreco1.device
        
        self.beta = self.beta.to(self.dev)
        self.beta.requires_grad_(True)
        self.beta.data.clamp_(min=0.0)
        with torch.set_grad_enabled(True):
            cap = self.compute_mutual_information(Lreco1, Lreco2)
            self.cap = self.cap.to(self.dev) + cap.detach()
            return cap
        
    def evaluate(self, Lreco1: Tensor, Lreco2: Tensor, beta: float):
        self.dev = Lreco1.device
        with torch.set_grad_enabled(False):
            cap = self.compute_mutual_information(Lreco1, Lreco2, beta=beta)
            self.cap = self.cap.to(self.dev) + cap
            return cap
    
    def reset(self):
        """Reset accumulated state if needed for a new computation sequence."""
        self.cap = torch.tensor([0.0], dtype=torch.float).to(self.dev)
    
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
        dataset: Dataset,
        beta0: Optional[float] = 1.0,
        lr: Optional[float] = 0.01,
        n_epochs: Optional[int] = 5,
        batch_size: Optional[int] = 64,
        compute_on_step: Optional[bool] = True,
        dist_sync_on_step: Optional[bool] = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        self.kernel = ApproximationCapacityKernel(beta0=beta0)
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.batch_size = batch_size
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize state
        self.add_state("cap", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("is_optimized", default=torch.tensor(0, dtype=torch.bool), dist_reduce_fx="sum")
    
    def optimize(self):
        """Optimize beta parameter over multiple epochs."""
        dev = self.kernel.beta.device
        optimizer = optim.Adam([self.kernel.beta], lr=self.learning_rate)
        
        # Optimization loop:
        best_beta = 0.0
        best_cap = - float('inf')
        for _ in range(self.n_epochs):
            epoch_cap = torch.tensor(0.0).to(dev)
            batch_count = 0
            
            # Process dataset in batches
            for batch in self.dataloader:
                # Extract losses from batch (assuming dataset returns a dict or tuple)
                if isinstance(batch, dict):
                    loss1, loss2 = batch['loss1'], batch['loss2']
                else:
                    loss1, loss2 = batch
                
                # Move to same device as kernel
                loss1 = loss1.to(dev)
                loss2 = loss2.to(dev)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute capacity (we want to maximize it, so negate for minimization)
                capacity = self.kernel(loss1, loss2)
                loss = -capacity  # Negative because we want to maximize capacity
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Enforce non-negativity constraint
                with torch.no_grad():
                    self.kernel.beta.data.clamp_(min=0.0)
                
                # Track epoch statistics
                epoch_cap += capacity.detach()
                batch_count += 1
            
            # Compute average capacity for this epoch
            if batch_count > 0:
                avg_epoch_cap = epoch_cap / batch_count
                
                # Update best beta if this epoch is better
                if avg_epoch_cap > best_cap:
                    best_cap = avg_epoch_cap
                    best_beta = self.kernel.beta.clone().detach()
        
        # Set the kernel's beta to the best found
        with torch.no_grad():
            self.kernel.beta.copy_(best_beta)
    
        self.is_optimized = torch.tensor(1, dtype=torch.bool)
        return best_cap
    
    def evaluate(self):
        """Evaluate CAP with the optimized beta parameter."""
        self.kernel.reset()
        dev = self.kernel.beta.device
                
        # Create a fresh dataloader with no shuffling to ensure consistent evaluation
        eval_dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluation loop
        with torch.no_grad():
            for batch in eval_dataloader:
                Lreco1, Lreco2 = batch
                
                # Compute capacity using the epoch-wise beta:
                capacity = self.kernel.evaluate(Lreco1.to(dev), Lreco2.to(dev), self.kernel.beta)
                
                # Track total capacity
                self.cap += capacity
        
        return self.cap
    
    def update(self, *args, **kwargs):
        """
        Triggers the optimize and evaluate processes.
        """
        if not self.is_optimized:
            self.optimize()
            self.evaluate()
    
    def compute(self) -> Tensor:
        """Return the computed CAP value."""
        if not self.is_optimized:
            self.update()
        
        if self.num_batches > 0:
            return self.cap_value
        else:
            return torch.tensor(0.0, device=self.kernel.beta.device)
    
    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.kernel.reset()