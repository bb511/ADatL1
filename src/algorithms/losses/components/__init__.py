from typing import Literal
import inspect
import torch
import torch.nn as nn


class ADLoss(nn.Module):
    """Base class for all loss functions.

    :param scale: Scaling factor for the loss.
    :param reduction: Reduction method to apply to the loss.
        Options are 'none', 'mean', 'sum'.
    """

    name: str = "total"  # name for the logs

    def __init__(
        self,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
    ):
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(self) -> torch.Tensor:
        """Forward method to compute the loss."""
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean" and loss.dim() > 0:
            return loss.mean()
        if self.reduction == "sum" and loss.dim() > 0:
            return loss.sum()
        if self.reduction == "none" and loss.dim() > 0:
            return loss
        if self.reduction in {"none", "mean", "sum", None} and loss.dim() == 0:
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")


L1ADLoss = ADLoss


class MultiLoss(ADLoss):
    """Compose named loss components from Hydra configs.

    Each child loss receives the subset of keyword arguments accepted by its
    ``forward`` method. Outputs are returned under ``loss/<name>`` plus
    ``loss/total``.
    """

    def __init__(
        self,
        list_losses: list[str],
        reduction: Literal["none", "mean", "sum"] = "none",
        scale: float = 1.0,
        **losses: nn.Module,
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.list_losses = list(list_losses)
        self.losses = nn.ModuleDict(
            {
                name: module
                for name, module in losses.items()
                if isinstance(module, nn.Module)
            }
        )

        missing = [name for name in self.list_losses if name not in self.losses]
        if missing:
            raise ValueError(f"Missing configured losses: {missing}")

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        total_terms: list[torch.Tensor] = []

        for name in self.list_losses:
            module = self.losses[name]
            result = module(**self._accepted_kwargs(module, kwargs))

            if isinstance(result, tuple):
                if name == "kl" and len(result) >= 2:
                    raw, scaled = result[0], result[1]
                    out["loss/kl_raw"] = raw
                    out["loss/kl"] = scaled
                    total_terms.append(scaled)
                else:
                    value = result[-1]
                    out[f"loss/{name}"] = value
                    total_terms.append(value)
            else:
                out[f"loss/{name}"] = result
                total_terms.append(result)

        if not total_terms:
            raise RuntimeError("MultiLoss has no active loss terms.")

        total = total_terms[0]
        for term in total_terms[1:]:
            total = total + term

        out["loss/total"] = self.scale * total
        return out

    @staticmethod
    def _accepted_kwargs(module: nn.Module, kwargs: dict) -> dict:
        signature = inspect.signature(module.forward)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return kwargs
        return {key: val for key, val in kwargs.items() if key in signature.parameters}
