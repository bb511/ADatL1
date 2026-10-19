# Vanilla auto-encoder model implementations
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


def _unfitted_residual_state(dim: int) -> dict[str, torch.Tensor]:
    return {
        "residual_oas_location": torch.zeros(dim),
        "residual_oas_precision": torch.eye(dim),
        "residual_oas_ready": torch.tensor(False),
    }


def _whitened_energy(
    residual: torch.Tensor,
    location: torch.Tensor,
    precision: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mahalanobis energy of the centred residual per observed feature.

    The residual and mask have shape (batch, features) and the result (batch,).
    Unobserved features are zeroed before the quadratic form.
    """
    mask = torch.ones_like(residual) if mask is None else mask.to(residual.dtype)
    centred = (residual - location) * mask
    n_observed = mask.sum(dim=1).clamp_min(1.0)
    return torch.einsum("bi,ij,bj->b", centred, precision, centred) / n_observed


class AE(ADLightningModule):
    """Autoencoder module.

    :param encoder: PyTorch module of the encoder.
    :param decoder: PyTorch module of the decoder.
    :param features: Optional PyTorch module to apply to the data before feeding it
        to the autoencoder.
    :param input_noise_std: Float specifying how much noise to add to the feature
        distributions before feeding them to the AE.
    :param delta: Float defining how close the loss is to L1 or L2, i.e., how much
        importance is given to tail examples. Parameter in the HuberLoss.
    :param anomaly_score: Score emitted as 'ascore/full' in validation and test:
        'mse', or 'residual_oas', the whitened residual, which needs the state set by
        src.callbacks.residual_state.ResidualStateCallback.
    :param target_rate: Float of the target background rate or FPR for the AE.
    :param base_rate: Float of the base rate, used to compute FPR given a target rate.
        If this is 'None', then target_rate is taken as the FPR directly.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_noise_std: float = 0.0,
        delta: float = 3.0,
        features: nn.Module = None,
        anomaly_score: str = "mse",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder, self.decoder = encoder, decoder
        self.input_noise_std = input_noise_std
        self.loss = HuberAELoss(delta=delta, reduction="none")
        self.ascore = MSEReconstructionLoss(reduction="none")

        if anomaly_score not in ("mse", "residual_oas"):
            raise ValueError(f"Unknown AE anomaly_score {anomaly_score!r}.")
        self.anomaly_score = anomaly_score
        # Buffers, so that every checkpoint carries the state fitted at its epoch.
        for name, value in _unfitted_residual_state(decoder.out_dim).items():
            self.register_buffer(name, value)

    def on_fit_start(self):
        inject_object_feature_map(self)

    def on_test_start(self):
        inject_object_feature_map(self)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Checkpoints written before the residual state existed load as unfitted.
        if "residual_oas_ready" not in state_dict:
            dim = self.residual_oas_location.numel()
            state_dict = {**state_dict, **_unfitted_residual_state(dim)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Residual x - x̂ of a flattened input, without input noise."""
        _, reconstruction = self.forward(x)
        return x - reconstruction

    def set_residual_state(self, location: torch.Tensor, precision: torch.Tensor):
        self.residual_oas_location.copy_(location)
        self.residual_oas_precision.copy_(precision)
        self.residual_oas_ready.fill_(True)

    def residual_oas_score(
        self, residual: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not bool(self.residual_oas_ready):
            raise RuntimeError(
                "The residual_oas score needs a fitted residual state: add the "
                "residual_state callback, or load a checkpoint written with it."
            )
        location, precision = self.residual_oas_location, self.residual_oas_precision
        return _whitened_energy(residual, location, precision, mask)

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        m = b.mask
        if m is not None:
            m = torch.flatten(m, start_dim=1).float()

        x_noisy = x
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x) * self.input_noise_std
            if m is not None:
                noise = noise * m

            x_noisy = x + noise

        z, reconstruction = self.forward(x_noisy)
        loss = self.loss(reco=reconstruction, target=x, mask=m)
        # The anomaly score is expected to be a distribution over events.
        scores = self._anomaly_scores(x, reconstruction, m)
        ascore = scores["mse"] if self.training else scores[self.anomaly_score]
        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        del x, z

        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(ascore)

        return {
            # Used for backpropagation:
            "loss": loss.mean(),
            # Used for logging:
            "loss/mean": loss.mean(),
            "ascore/operational": operational_ascore,
            # Used for callbacks:
            "loss/full": loss.detach(),
            "ascore/full": ascore.detach(),
            **{f"ascore/{name}": score.detach() for name, score in scores.items()},
            "reconstructed_data": reconstruction.detach(),
        }

    def _anomaly_scores(
        self, x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Per-event scores by name, the residual one outside training only.

        The residual state is refitted at the start of every validation epoch, so during
        training it is stale, or unfitted in the first epoch.
        """
        scores = {"mse": self.ascore(x, reconstruction, mask)}
        fitted = bool(self.residual_oas_ready)
        if not self.training and (fitted or self.anomaly_score == "residual_oas"):
            scores["residual_oas"] = self.residual_oas_score(x - reconstruction, mask)
        return scores

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
        }
