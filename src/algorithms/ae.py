# Vanilla auto-encoder model implementations
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss, PileupMIAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch
from src.data.sensitive_binning import FixedQuantileSensitiveBinner


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
        mi_sensitive_variable: str = "FET.Et",
        mi_sensitive_num_bins: int = 10,
        mi_sensitive_reduction: str = "First",
        mi_sensitive_use_denormalization: bool = False,
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
        self.sensitive_binner = FixedQuantileSensitiveBinner(variable=mi_sensitive_variable, num_bins=mi_sensitive_num_bins, 
                                                             reduction=mi_sensitive_reduction, use_denormalized=mi_sensitive_use_denormalization)

    def on_fit_start(self):
        inject_object_feature_map(self)
        self._fit_sensitive_binner()

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

        return any(key in name for key in self._energy_feature_names())


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

        sensitive = self._compute_sensitive_bins(x=x, mask=m)

        if self.training and self.global_step < 3:
            unique, counts = torch.unique(sensitive.detach().cpu(), return_counts=True)
            print(
                f"[MI] step={self.global_step} sensitive bins: "
                f"{dict(zip(unique.tolist(), counts.tolist()))}"
            )

        mi_loss = self.mi_loss(latent=z, sensitive=sensitive)
        gamma_mi_loss = self.mi_gamma * mi_loss

        total_loss = reco_loss.mean() + gamma_mi_loss

        with torch.no_grad():
            z_detached = z.detach()
            probs = torch.sigmoid(self.mi_loss.mi_loss.temperature * z_detached)

            reco_loss_mean = reco_loss.mean().detach()
            mi_loss_detached = mi_loss.detach()
            gamma_mi_loss_detached = gamma_mi_loss.detach()

            mi_to_reco_ratio = gamma_mi_loss_detached / reco_loss_mean.clamp_min(1e-12)

            latent_mean = z_detached.mean()
            latent_std = z_detached.std(unbiased=False)

            prob_mean = probs.mean()
            prob_std = probs.std(unbiased=False)
            prob_min = probs.min()
            prob_max = probs.max()

            prob_saturation_low = (probs < 0.01).float().mean()
            prob_saturation_high = (probs > 0.99).float().mean()

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

                # Existing project-style scalar losses:
                "loss/mean": total_loss.detach(),
                "loss/reco": reco_loss.mean().detach(),
                "loss/mi": mi_loss.detach(),
                "loss/gamma_mi": gamma_mi_loss.detach(),

                # Binner diagnostics:
                "sensitive/bin_min": sensitive.min().float().detach(),
                "sensitive/bin_max": sensitive.max().float().detach(),
                "sensitive/bin_mean": sensitive.float().mean().detach(),

                # Anomaly score logging:
                "ascore/operational": operational_ascore,

                # Used for callbacks:
                # Keep this event-level and reconstruction-only.
                "loss/full": reco_loss.detach(),
                "ascore/full": ascore.detach(),
                "reconstructed_data": reconstruction.detach(),
            }

    def outlog(self, outdict: dict) -> dict:
        """Values logged by ADLightningModule at epoch end."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_reco": outdict.get("loss/reco"),
            "loss_mi": outdict.get("loss/mi"),
            "loss_gamma_mi": outdict.get("loss/gamma_mi"),

            # Sensitive-bin diagnostics:
            "sensitive_bin_min": outdict.get("sensitive/bin_min"),
            "sensitive_bin_max": outdict.get("sensitive/bin_max"),
            "sensitive_bin_mean": outdict.get("sensitive/bin_mean"),

            # Existing anomaly-score logging:
            "ascore_operational": outdict.get("ascore/operational"),
        }
    
    def _fit_sensitive_binner(self) -> None:
        """Compute fixed sensitive-variable bin edges from the full training split."""
        if self.sensitive_binner.is_fitted:
            return

        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None

        if datamodule is None:
            raise RuntimeError(
                "Cannot fit sensitive binner because trainer.datamodule is missing."
            )

        train_splits = getattr(datamodule, "_main", None)
        if not isinstance(train_splits, dict) or "train" not in train_splits:
            raise RuntimeError(
                "Cannot fit sensitive binner because datamodule._main['train'] is missing. "
                "The binner must be fitted after datamodule.setup('fit')."
            )

        train_split = train_splits["train"]

        normalizer = getattr(datamodule, "normalizer", None)

        edges = self.sensitive_binner.fit(
            x=train_split.x,
            mask=train_split.mask,
            object_feature_map=self.object_feature_map,
            normalizer=normalizer,
        )

        stats = self.sensitive_binner.fit_stats

        print(f"[MI] Fixed sensitive variable: {self.sensitive_binner.variable}")
        print(f"[MI] Requested bins: {stats['num_bins_requested']}")
        print(f"[MI] Effective bins: {stats['num_bins_effective']}")
        print(f"[MI] Values used: {stats['num_values']}")
        print(
            "[MI] Value stats: "
            f"min={stats['min']:.6g}, "
            f"max={stats['max']:.6g}, "
            f"mean={stats['mean']:.6g}, "
            f"std={stats['std']:.6g}"
        )
        print(f"[MI] Bin edges: {stats['edges']}")
        print(f"[MI] Bin counts: {stats['counts']}")

    def _compute_sensitive_bins(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Compute batch sensitive labels from fixed precomputed bin edges."""
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        normalizer = getattr(datamodule, "normalizer", None) if datamodule is not None else None

        return self.sensitive_binner.transform(x=x, mask=mask, object_feature_map=self.object_feature_map, normalizer=normalizer)
