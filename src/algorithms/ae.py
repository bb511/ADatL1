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
from src.algorithms.components.bernoulli import BernoulliSampling


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
        mi_bernoulli_num_samples: int = 10,
        mi_bernoulli_std: float = 1.0,
        mi_bernoulli_threshold: float = 0.5,
        mi_use_quantized_sigmoid: bool = False,
        mi_bits_bernoulli_sigmoid: int = 8,
        mi_use_float64_entropy: bool = True,
        mi_sensitive_variable: str = "FET.Et",
        mi_sensitive_num_bins: int = 10,
        mi_sensitive_reduction: str = "First",
        mi_sensitive_use_denormalization: bool = False,
        forbid_sensitive_variable_in_input: bool = True,
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

        # HepInfo Bernoulli latent bottleneck. The decoder consumes the
        # straight-through Bernoulli sample, while the MI estimator below is
        # computed on the pre-sampling logits z.
        self.bernoulli = BernoulliSampling(
            num_samples=mi_bernoulli_num_samples,
            std=mi_bernoulli_std,
            threshold=mi_bernoulli_threshold,
            temperature=mi_temperature,
            use_quantized=mi_use_quantized_sigmoid,
            bits_bernoulli_sigmoid=mi_bits_bernoulli_sigmoid,
        )

        self.mi_loss = PileupMIAELoss(
            mi_temperature=mi_temperature,
            input_is_logits=True,
            use_float64=mi_use_float64_entropy,
            use_quantized_sigmoid=mi_use_quantized_sigmoid,
            bits_bernoulli_sigmoid=mi_bits_bernoulli_sigmoid,
        )

        self.mi_gamma = float(mi_gamma)
        self.forbid_sensitive_variable_in_input = forbid_sensitive_variable_in_input
        self.sensitive_binner = FixedQuantileSensitiveBinner(variable=mi_sensitive_variable, num_bins=mi_sensitive_num_bins, 
                                                             reduction=mi_sensitive_reduction, use_denormalized=mi_sensitive_use_denormalization)

    def on_fit_start(self):
        inject_object_feature_map(self)
        self._assert_sensitive_not_in_model_input()
        self._fit_sensitive_binner()


    def on_test_start(self):
        inject_object_feature_map(self)
        self._assert_sensitive_not_in_model_input()

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)

        # Pre-Bernoulli latent logits / activations.
        # This is the tensor used for the MI estimator, matching hepinfo.
        z = self.encoder(x)

        # Straight-through Bernoulli bottleneck.
        # This is the tensor consumed by the decoder, matching hepinfo MiVAE.
        z_sample = self.bernoulli(z)

        reconstruction = self.decoder(z_sample)

        return z, reconstruction

    def ascore(self, x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute per-event anomaly score: reconstruction error per observation."""
        # Rely on the configured reconstruction loss to produce a per-sample score.
        return self.ascore_loss(target=x, reco=reconstruction, mask=mask)

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:

        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        m = b.mask
        if m is not None:
            m = torch.flatten(m, start_dim=1).float()

        control_x, control_mask = self._get_sensitive_inputs(b)


        x_noisy = x
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x) * self.input_noise_std
            if m is not None:
                noise = noise * m

            x_noisy = x + noise

        z, reconstruction = self.forward(x_noisy)
        reco_loss = self.reco_loss(target=x, reco=reconstruction, mask=m)

        sensitive = self._compute_sensitive_bins(x=control_x, mask=control_mask)

        mi_loss = self.mi_loss(latent=z, sensitive=sensitive)
        with torch.no_grad():
            perm = torch.randperm(sensitive.shape[0], device=sensitive.device)
            sensitive_perm = sensitive[perm]
        gamma_mi_loss = self.mi_gamma * mi_loss

        total_loss = reco_loss.mean() + gamma_mi_loss

        with torch.no_grad():

            reco_loss_mean = reco_loss.mean().detach()

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
            x=train_split.control_x if train_split.control_x is not None else train_split.x,
            mask=(
                train_split.control_mask
                if train_split.control_mask is not None
                else train_split.mask
            ),
            object_feature_map=self.control_object_feature_map,
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

    def _compute_sensitive_bins(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute batch sensitive labels from fixed precomputed bin edges."""
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        normalizer = (
            getattr(datamodule, "normalizer", None)
            if datamodule is not None
            else None
        )

        return self.sensitive_binner.transform(
            x=x,
            mask=mask,
            object_feature_map=self.control_object_feature_map,
            normalizer=normalizer,
        )

    def extract_sensitive_values(self, batch) -> torch.Tensor:
        """Extract the MI-sensitive values from a training batch before binning."""
        batch_view = unpack_batch(batch)
        control_x, control_mask = self._get_sensitive_inputs(batch_view)
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        normalizer = (
            getattr(datamodule, "normalizer", None)
            if datamodule is not None
            else None
        )

        return self.sensitive_binner.extract_values(
            x=control_x,
            mask=control_mask,
            object_feature_map=self.control_object_feature_map,
            normalizer=normalizer,
        )

    def _get_sensitive_inputs(self, batch_view):
        """Select and validate the same control tensor used by the MI loss."""
        control_required = (
            getattr(self, "control_object_feature_map", None) is not None
            and getattr(self, "object_feature_map", None) is not None
            and self.control_object_feature_map != self.object_feature_map
        )

        if batch_view.control_x is None:
            if control_required:
                raise RuntimeError(
                    "Batch does not contain control_x, but control_object_feature_map "
                    "differs from object_feature_map. The MI sensitive variable cannot be "
                    "extracted safely. Check that L1ADDataset.__iter__ yields "
                    "(x, mask, l1bit, y, control_x, control_mask) when control_data is set."
                )

            control_x = batch_view.x
            control_mask = batch_view.mask
        else:
            control_x = batch_view.control_x
            control_mask = batch_view.control_mask

        expected_control_dim = self._num_flat_features_from_map(
            getattr(self, "control_object_feature_map", None)
        )

        if expected_control_dim is not None:
            actual_control_dim = torch.flatten(control_x, start_dim=1).shape[1]

            if actual_control_dim != expected_control_dim:
                raise RuntimeError(
                    "control_x does not match control_object_feature_map. "
                    f"Expected {expected_control_dim} flattened features from the control map, "
                    f"but got {actual_control_dim}. This usually means the MI target is being "
                    "read from the model-input tensor instead of the full/control tensor."
                )

        return control_x, control_mask

    def _assert_sensitive_not_in_model_input(self) -> None:
        """Fail fast if the MI target leaks into the AE input feature map."""
        if not self.forbid_sensitive_variable_in_input:
            return

        object_feature_map = getattr(self, "object_feature_map", None)

        if object_feature_map is None:
            return

        object_name, feature_name = self.sensitive_binner.variable.split(".", maxsplit=1)

        for obj_key, feature_map in object_feature_map.items():
            if str(obj_key).lower() != object_name.lower():
                continue

            for feat_key in feature_map.keys():
                if str(feat_key).lower() == feature_name.lower():
                    raise RuntimeError(
                        f"Sensitive MI variable {self.sensitive_binner.variable!r} is "
                        "still present in pl_module.object_feature_map, which is the "
                        "anomaly-detector input map. Configure "
                        "data.model_input_exclude_features to remove it from the model "
                        "input while keeping it in control_object_feature_map."
                    )
                
    @staticmethod
    def _num_flat_features_from_map(object_feature_map: dict | None) -> int | None:
        if object_feature_map is None:
            return None

        indices = [
            int(idx)
            for feature_map in object_feature_map.values()
            for idxs in feature_map.values()
            for idx in idxs
        ]

        if not indices:
            return None

        return max(indices) + 1
