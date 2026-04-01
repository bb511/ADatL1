# Approximation Capacity implementation. Based on the work of Victor Jimenez.
from typing import Optional, Any, Callable
from tqdm import tqdm
import gc

import torch
from torch import Tensor, optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric

from src.callbacks.metrics.cap.kernel import ApproximationCapacityKernel
from src.callbacks.metrics.cap.binary import (
    get_normalizer_fn as get_normalizer_fn_binary,
    get_energy_fn as get_energy_fn_binary,
    get_regularizer_fn as get_regularizer_fn_binary
)


class ApproximationCapacity(Metric):
    """
    Approximation Capacity (CAP) Metric for classification tasks.

    The metric maximizes the mutual information between two sets of classification
    scores (e.g. log-probabilities, reconstruction losses, ...) parametrizing their
    distribution over the hypothesis space (i.e. the set of possible clustering
    assignments) using Gibbs distributions of inverse temperature `beta`.

        CAP ∝ log(Z_12 / (Z_1 * Z_2))

    Where Z_i are partition functions over the set of clustering assignments, computed
    using different energy functions and constraints.

    Functions that are imported:
    - Energy: Determines the probability of a specific clustering assignment.
    - Normalization: Normalizes model outputs (logits, loss...) to [0,1] range.
    - Regularization: Enforces constraints on the hypothesis space to implicitly
        delimit a feasible clustering assignments. For instance, by limiting
        the number of observations assigned to a specific cluster.

    Args:
        beta0: Initial inverse temperature parameter for Gibbs distribution.
        normalization_type: Method to normalize reconstruction losses to [0,1].
            Options: "minmax", "quantile", "sigmoid" (default: "minmax")
        normalization_params: Parameters for normalization method.
        energy_type: Energy function for Gibbs distribution over anomaly assignments.
            Options: "baseline", "focal", "exponential", "margin", "contrastive", "adaptive"
        energy_params: Parameters for energy function.
        regularization_type: Method to enforce anomaly count constraints.
            Options: "none", "threshold_max", "threshold_mean", "threshold_zero",
        regularization_params: Parameters for regularization method.
        binary: Whether to use binary (anomaly/normal) or multi-class classification.
            Currently only binary=True is supported (default: True)
        lr: Learning rate for beta parameter optimization during metric computation (default: 0.01)
        n_epochs: Number of optimization epochs for beta parameter learning (default: 5)
        batch_size: Batch size for processing reconstruction loss pairs (default: 64)
        process_group: Process group for distributed training (DDP support)
        dist_sync_fn: Function to synchronize metric state across processes

    Torchmetrics Arguments:
        is_differentiable: False (metric computation is not differentiable)
        higher_is_better: True (higher CAP indicates better anomaly detection)
        full_state_update: False (uses incremental state updates)

    Usage Example:
        ```python
        # Basic usage with default baseline energy
        cap_metric = ApproximationCapacityMetric()

        # Advanced usage with contrastive energy and constraints
        cap_metric = ApproximationCapacityMetric(
            energy_type="contrastive",
            energy_params={"center": 0.75},
            regularization_type="threshold_max",
            regularization_params={"maxcount": 50}
        )

        # Compute CAP score
        cap_score = cap_metric(loss_batch1, loss_batch2)
        ```
    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        beta0: Optional[float] = 1.0,
        normalization_type: Optional[str] = "minmax",
        normalization_params: Optional[dict] = None,
        energy_type: Optional[str] = "baseline",
        energy_params: Optional[dict] = None,
        regularization_type: Optional[str] = "none",
        regularization_params: Optional[dict] = None,
        binary: Optional[bool] = True,
        lr: Optional[float] = 0.01,
        n_epochs: Optional[int] = 5,
        batch_size: Optional[int] = 64,
        device: Optional[str] = "cpu",
        normalize_gradients: Optional[bool] = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        self.beta0 = beta0
        if self.beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")

        super().__init__(
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        # Parameters for energy and regularization functions:
        self.binary = binary
        self.normalization_type = normalization_type
        self.normalization_params = normalization_params or {}
        self.energy_type = energy_type
        self.energy_params = energy_params or {}
        self.regularization_type = regularization_type
        self.regularization_params = regularization_params or {}

        # Learning parameters:
        self.dev = device
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.batch_size = batch_size
        self.normalize_gradients = normalize_gradients

        # Initialize CAP:
        self.add_state("cap", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Normalizer of the loss:
        self.normalizer_fn = self._get_normalizer_fn()

    def _get_normalizer_fn(self):
        if self.binary:
            normalizer_fn = get_normalizer_fn_binary(
                self.normalization_type,
                self.normalization_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")

        return normalizer_fn

    def _get_energy_fn(self):
        if self.binary:
            energy_fn = get_energy_fn_binary(
                self.energy_type,
                self.energy_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")

        return energy_fn

    def _get_regularizer_fn(self):
        if self.binary:
            regularizer_fn = get_regularizer_fn_binary(
                self.regularization_type,
                self.regularization_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")

        return regularizer_fn

    def update(self, logits1: Tensor, logits2: Tensor, **kwargs):
        """Optimize beta parameter over multiple epochs."""
        self.reset()

        # Normalize the logits
        logits1, logits2 = self.normalizer_fn(logits1, logits2)

        # Add mean and std to the energy parameters
        combined = torch.cat([logits1, logits2], dim=0)
        self.energy_params.update({
            "mean": combined.mean().item(),
            "std": combined.std().item(),
        })
        del combined

        # Initialize kernel with updated energy function
        kernel = ApproximationCapacityKernel(
            beta0=self.beta0,
            energy_fn=self._get_energy_fn(),
            normalize_gradients=self.normalize_gradients,
        ).to(self.dev)

        optimizer = optim.Adam([kernel.beta], lr=self.learning_rate)

        use_cuda = str(self.dev).startswith("cuda")
        inputs_on_cpu = logits1.device.type == "cpu" and logits2.device.type == "cpu"
        use_pin_memory = use_cuda and inputs_on_cpu

        dataloader = DataLoader(
            TensorDataset(logits1, logits2),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin_memory,
            drop_last=False,
        )

        best_beta = kernel.beta.clone().detach()
        best_cap = -float("inf")
        self.epoch_logs = []

        for _ in range(self.n_epochs):
            kernel.reset()

            with torch.set_grad_enabled(True):
                for batch in dataloader:
                    batch_logits1 = batch[0].to(self.dev, non_blocking=use_pin_memory)
                    batch_logits2 = batch[1].to(self.dev, non_blocking=use_pin_memory)

                    optimizer.zero_grad(set_to_none=True)
                    loss = -kernel(batch_logits1, batch_logits2)
                    loss.backward()
                    optimizer.step()

            current_cap = kernel.cap.item()
            current_beta = kernel.beta.item()
            self.epoch_logs.append({
                "cap": current_cap,
                "beta": current_beta,
            })

            if current_cap > best_cap:
                best_cap = current_cap
                best_beta = kernel.beta.clone().detach()

        # Final evaluation with the best beta
        with torch.no_grad():
            kernel.reset()
            for batch in dataloader:
                batch_logits1 = batch[0].to(self.dev, non_blocking=use_pin_memory)
                batch_logits2 = batch[1].to(self.dev, non_blocking=use_pin_memory)
                _ = kernel.evaluate(batch_logits1, batch_logits2, beta=best_beta)

            self.cap.fill_(kernel.capacity.item())

        kernel.to("cpu")
        del kernel

    def compute(self) -> Tensor:
        """Return the computed CAP value."""
        return self.cap.item()
