from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.svdd import SVDDLoss
from src.data.utils import unpack_batch


class DeepSVDD(ADLightningModule):
    """Deep SVDD with a fixed nominal-data center."""

    def __init__(
        self,
        encoder: nn.Module,
        features: Optional[nn.Module] = None,
        center_init_method: str = "mean",
        center_eps: float = 0.1,
        center_max_batches: int | None = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        objective: str | None = None,
        soft_boundary: bool = False,
        nu: float = 0.1,
        network_weight_decay: float = 1e-6,
        radius_warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder"])

        if center_init_method != "mean":
            raise ValueError(
                "Deep SVDD requires center_init_method='mean'; a zero center permits "
                "the trivial all-zero solution."
            )
        if center_eps < 0.0:
            raise ValueError("center_eps must be non-negative.")
        if center_max_batches is not None and center_max_batches <= 0:
            raise ValueError("center_max_batches must be positive or None.")
        if network_weight_decay < 0.0:
            raise ValueError("network_weight_decay must be non-negative.")
        if radius_warmup_epochs < 0:
            raise ValueError("radius_warmup_epochs must be non-negative.")

        inferred_objective = "soft_boundary" if soft_boundary else "one_class"
        if objective is None:
            objective = inferred_objective
        elif soft_boundary and objective != "soft_boundary":
            raise ValueError(
                "soft_boundary=True conflicts with objective="
                f"{objective!r}; use objective='soft_boundary'."
            )

        self.encoder = encoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.center_eps = float(center_eps)
        self.center_max_batches = center_max_batches
        self.ckpt_path = ckpt
        self.network_weight_decay = float(network_weight_decay)
        self.radius_warmup_epochs = int(radius_warmup_epochs)

        self.loss = SVDDLoss(objective=objective, nu=nu)
        self.register_buffer("center", torch.empty(0))
        self.register_buffer("radius", torch.tensor(0.0, dtype=torch.float32))
        self._training_distances: list[torch.Tensor] = []

    @property
    def objective(self) -> str:
        return self.loss.objective

    @property
    def center_initialized(self) -> bool:
        return self.center.numel() > 0

    def on_fit_start(self) -> None:
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

        if not self.center_initialized:
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is None:
                raise RuntimeError(
                    "Deep SVDD center initialization requires a trainer datamodule."
                )
            self.initialize_center(datamodule.train_dataloader())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.encoder(x)

    def _prepare_input(self, batch) -> torch.Tensor:
        return torch.flatten(unpack_batch(batch).x, start_dim=1)

    @torch.no_grad()
    def initialize_center(self, train_dataloader) -> None:
        """Compute a fixed center from the complete nominal training loader."""
        encoder_was_training = self.encoder.training
        features_were_training = self.features.training
        self.encoder.eval()
        self.features.eval()

        representation_sum = None
        count = 0
        try:
            for batch_idx, batch in enumerate(train_dataloader):
                if self.center_max_batches is not None and batch_idx >= self.center_max_batches:
                    break
                x = self._prepare_input(batch).to(self.device)
                z = self.forward(x)
                if z.ndim != 2:
                    raise ValueError(
                        "SVDD encoder must return [batch, latent_dim], got " f"{tuple(z.shape)}."
                    )
                batch_sum = z.detach().sum(dim=0)
                representation_sum = (
                    batch_sum if representation_sum is None else representation_sum + batch_sum
                )
                count += z.shape[0]
        finally:
            self.encoder.train(encoder_was_training)
            self.features.train(features_were_training)

        if representation_sum is None or count == 0:
            raise RuntimeError("Cannot initialize the SVDD center from an empty loader.")

        count_tensor = representation_sum.new_tensor(float(count))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(representation_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        center = representation_sum / count_tensor
        if self.center_eps > 0.0:
            small = center.abs() < self.center_eps
            center = center.clone()
            center[small & (center >= 0)] = self.center_eps
            center[small & (center < 0)] = -self.center_eps
        self.center = center.detach()

    def _compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        if not self.center_initialized:
            raise RuntimeError("SVDD center has not been initialized.")
        return torch.sum((z - self.center) ** 2, dim=1)

    def _network_regularization(self) -> torch.Tensor:
        weights = [
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]
        if not weights or self.network_weight_decay == 0.0:
            return self.radius.new_zeros(())
        squared_norm = torch.stack([weight.square().sum() for weight in weights]).sum()
        return 0.5 * self.network_weight_decay * squared_norm

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        x = self._prepare_input(batch)
        z = self.forward(x)
        ascore = self._compute_distance(z)
        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        data_loss = self.loss(
            distances=ascore,
            radius=self.radius if self.objective == "soft_boundary" else None,
        )
        regularization = self._network_regularization()
        loss_mean = self._add_hgq_loss(data_loss.mean() + regularization)

        if self.training and self.objective == "soft_boundary":
            self._training_distances.append(ascore.detach().cpu())

        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(ascore)
            q50, q99 = self.compute_score_quantiles(ascore)
            z_squared = torch.square(z).sum(dim=1).mean().item()
            center_norm = torch.square(self.center).sum().item()

        return {
            "loss": loss_mean,
            "loss/mean": loss_mean,
            "loss/distance_raw/mean": ascore.mean(),
            "loss/reg_scaled/mean": regularization,
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_squared": z_squared,
            "center_norm": center_norm,
            "radius": self.radius.detach(),
            "loss/full": data_loss.detach(),
            "ascore/full": ascore.detach(),
        }

    def on_train_epoch_end(self) -> None:
        if (
            self.objective == "soft_boundary"
            and self.current_epoch >= self.radius_warmup_epochs
            and self._training_distances
        ):
            local_distances = torch.cat(self._training_distances)
            gathered = [local_distances]
            distributed = dist.is_available() and dist.is_initialized()
            if distributed:
                gathered_objects = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_objects, local_distances)
                gathered = gathered_objects
            if not distributed or dist.get_rank() == 0:
                all_distances = torch.cat(gathered)
                new_radius = torch.quantile(torch.sqrt(all_distances), 1.0 - self.loss.nu).to(
                    self.radius.device
                )
                self.radius.copy_(new_radius)
            if distributed:
                dist.broadcast(self.radius, src=0)

        self._training_distances.clear()
        super().on_train_epoch_end()

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_distance_raw": outdict.get("loss/distance_raw/mean"),
            "loss_reg_scaled": outdict.get("loss/reg_scaled/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
            "z_squared": outdict.get("z_squared"),
            "center_norm": outdict.get("center_norm"),
            "radius": outdict.get("radius"),
        }

    def _load_checkpoint(self) -> None:
        """Load matching weights from an SVDD or encoder-compatible checkpoint."""
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        if "center" in state_dict:
            self.center = state_dict["center"].detach().clone()
        self.load_state_dict(state_dict, strict=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        center_key = prefix + "center"
        if center_key in state_dict and self.center.shape != state_dict[center_key].shape:
            self.center = torch.empty_like(state_dict[center_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _add_hgq_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            extra = torch.stack(
                [torch.as_tensor(value, device=loss.device) for value in self.encoder.losses]
            ).sum()
            return loss + extra
        return loss
