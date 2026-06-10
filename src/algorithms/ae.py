# Vanilla auto-encoder model implementations
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss, PileupMIAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


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
        target_rate: float = 0.25,
        base_rate: float | None = None,
        mi_temperature: float = 6.0,
        mi_gamma: float = 0.1,
        mi_reduction: str = "sum",
        **kwargs,
    ):
        # Only forward keys expected by ADLightningModule.__init__.
        # Extract parent kwargs and avoid accidentally passing algorithm-specific keys
        # (e.g., legacy `gamma`, `mi_reduction`, etc.). This prevents TypeErrors
        # during Hydra instantiation when the full algorithm config is provided.
        parent_keys = {"optimizer", "scheduler", "save_hyperparameters"}
        parent_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in parent_keys}
        super().__init__(model=None, **parent_kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder, self.decoder = encoder, decoder
        self.input_noise_std = input_noise_std

        self.reco_loss = HuberAELoss(delta=delta, scale=1.0, reduction="none")
        self.ascore_loss = MSEReconstructionLoss(scale=1.0, reduction="none")
        self.mi_loss = PileupMIAELoss(mi_temperature=mi_temperature, input_is_logits=True)
        self.mi_gamma = float(mi_gamma)

        self._warned_no_denorm_for_mi = False
        self._energy_feature_indices: list[int] | None = None

    def on_fit_start(self):
        inject_object_feature_map(self)

    def on_test_start(self):
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def ascore(self, x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute per-event anomaly score: reconstruction error per observation."""
        # Rely on the configured reconstruction loss to produce a per-sample score.
        return self.ascore_loss(target=x, reco=reconstruction, mask=mask)

    def _energy_feature_names(self) -> tuple[str, ...]:
        # Feature names seen in L1 object maps usually use Et-like naming.
        return ("et", "pt", "energy")

    def _is_energy_feature(self, feature_name: str) -> bool:
        name = feature_name.lower()
        if "eta" in name or "phi" in name:
            return False
        return any(key in name for key in self._energy_feature_names())

    def _get_energy_feature_indices(self) -> list[int]:
        if self._energy_feature_indices is not None:
            return self._energy_feature_indices

        ofm = getattr(self, "object_feature_map", None)
        if not isinstance(ofm, dict):
            self._energy_feature_indices = []
            return self._energy_feature_indices

        idxs: list[int] = []
        for _, feature_map in ofm.items():
            for feature_name, feature_idxs in feature_map.items():
                if self._is_energy_feature(feature_name):
                    idxs.extend(int(i) for i in feature_idxs)

        # Keep deterministic ordering and remove duplicates.
        self._energy_feature_indices = sorted(set(idxs))
        return self._energy_feature_indices

    def _get_denormalized_x_for_mi(self, x: torch.Tensor) -> torch.Tensor:
        """Return a physical-scale copy of x when the datamodule normalizer is available.

        The autoencoder receives normalized inputs. Computing a physics control
        variable such as total event energy on normalized features is usually wrong,
        especially for padded entries, because a denormalized padded value can become
        non-zero after applying the shift. Therefore this method tries to use the
        datamodule's normalizer and falls back to the normalized tensor only when no
        normalizer is attached.
        """
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        normalizer = getattr(datamodule, "normalizer", None) if datamodule is not None else None
        ofm = getattr(self, "object_feature_map", None)

        if normalizer is None or ofm is None or not hasattr(normalizer, "denorm_1d_tensor"):
            if not self._warned_no_denorm_for_mi:
                print(
                    "Warning: computing MI energy control variable from normalized x "
                    "because no datamodule normalizer was available."
                )
                self._warned_no_denorm_for_mi = True
            return x.detach()

        x_phys = x.detach().clone()
        if getattr(normalizer, "scale_tensor", None) is None or getattr(normalizer, "shift_tensor", None) is None:
            normalizer.setup_1d_denorm(ofm)
        return normalizer.denorm_1d_tensor(x_phys)

    def _compute_total_energy(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        idxs = self._get_energy_feature_indices()
        x_for_energy = self._get_denormalized_x_for_mi(x)

        if idxs:
            idx_tensor = torch.as_tensor(idxs, device=x.device, dtype=torch.long)
            energy_terms = x_for_energy.index_select(dim=1, index=idx_tensor)
            if mask is not None:
                energy_mask = mask.index_select(dim=1, index=idx_tensor).to(dtype=energy_terms.dtype)
                energy_terms = energy_terms * energy_mask
            return energy_terms.sum(dim=1)

        # Conservative fallback: keep the old behavior, but only if no object map is available.
        if x_for_energy.size(1) >= 3:
            return x_for_energy[:, :3].sum(dim=1)
        return x_for_energy[:, 0]

    def _bin_sensitive_by_batch_quantiles(self, sensitive_value: torch.Tensor, num_bins: int = 5) -> torch.Tensor:
        """Discretize a continuous control variable for the Bernoulli MI estimator.

        The original MI loss expects a discrete S. Batch quantiles are a pragmatic
        local discretization; for production studies, prefer fixed thresholds computed
        once on the training set so the meaning of each bin is stable across batches.
        """
        sensitive_value = sensitive_value.detach().flatten()
        if sensitive_value.numel() < num_bins:
            return torch.zeros_like(sensitive_value, dtype=torch.long).unsqueeze(1)

        q = torch.linspace(
            0.0,
            1.0,
            steps=num_bins + 1,
            device=sensitive_value.device,
            dtype=sensitive_value.dtype,
        )[1:-1]
        thresholds = torch.quantile(sensitive_value, q).contiguous()
        return torch.bucketize(sensitive_value, thresholds).unsqueeze(1).long()

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
        reco_loss = self.reco_loss(target=x, reco=reconstruction, mask=m)

        total_energy = self._compute_total_energy(x=x, mask=m)
        sensitive = self._bin_sensitive_by_batch_quantiles(total_energy, num_bins=5)

        mi_loss = self.mi_loss(latent=z, sensitive=sensitive)

        total_loss = reco_loss.mean() + self.mi_gamma * mi_loss

        # The anomaly score is expected to be a distribution over events.
        # Allow subclasses to override `ascore`; otherwise fall back to
        # the reconstruction loss per observation for robustness.
        ascore_fn = getattr(self, "ascore", None)
        if callable(ascore_fn):
            ascore = ascore_fn(x, reconstruction, m)
        else:
            ascore = reco_loss

        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        del x, z

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            # If the operational tail is too small, use a top-k average for stability.
            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(ascore, 1.0 - self.target_fpr).item()

        return {
            # Used for backpropagation:
            "loss": total_loss,

            # Used for logging:
            "loss/mean": total_loss.detach(),
            "loss/reco": reco_loss.mean().detach(),
            "loss/mi": mi_loss.detach(),
            "ascore/operational": operational_ascore,

            # Used for callbacks:
            # Keep this event-level and reconstruction-only.
            "loss/full": reco_loss.detach(),
            "ascore/full": ascore.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_reco": outdict.get("loss/reco"),
            "loss_mi": outdict.get("loss/mi"),
            "ascore_operational": outdict.get("ascore/operational"),
        }
