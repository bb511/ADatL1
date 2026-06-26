# Vanilla auto-encoder model implementations
from typing import Optional
import os
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss, PileupMIAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch
from src.data.sensitive_binning import FixedQuantileSensitiveBinner

_HEPINFO_TF = None
_HEPINFO_MI_LOSS = None


def _get_hepinfo_tf_mi_loss():
    """Import and cache the original hepinfo TensorFlow/Keras MI loss.

    This is intentionally fail-fast because this branch is for verification.
    """
    global _HEPINFO_TF, _HEPINFO_MI_LOSS

    if _HEPINFO_TF is not None and _HEPINFO_MI_LOSS is not None:
        return _HEPINFO_TF, _HEPINFO_MI_LOSS

    # Keras 3 selects its backend at import time. ADatL1/src/train.py currently
    # sets KERAS_BACKEND="torch", which is wrong for the original hepinfo loss.
    # Therefore, force TensorFlow before importing tensorflow/keras/hepinfo.
    if "keras" in sys.modules:
        import keras

        backend = None
        if hasattr(keras, "config") and hasattr(keras.config, "backend"):
            backend = keras.config.backend()

        if backend != "tensorflow":
            raise RuntimeError(
                "Keras is already imported with backend "
                f"{backend!r}. The hepinfo verification loss requires "
                "KERAS_BACKEND='tensorflow' before the first keras import."
            )
    else:
        os.environ["KERAS_BACKEND"] = "tensorflow"

    repo = os.environ.get("HEPINFO_REPO")
    if repo is None:
        raise RuntimeError(
            "HEPINFO_REPO is not set. Set it to the hepinfo repository root, "
            "i.e. the directory that contains the inner 'hepinfo/' package."
        )

    repo_path = Path(repo).expanduser().resolve()
    if not repo_path.is_dir():
        raise RuntimeError(f"HEPINFO_REPO does not exist or is not a directory: {repo_path}")

    if not (repo_path / "hepinfo").is_dir():
        raise RuntimeError(
            f"HEPINFO_REPO must point to the repo root containing 'hepinfo/'. "
            f"Got: {repo_path}"
        )

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("Failed to import TensorFlow for hepinfo MI verification.") from exc

    # Keep TensorFlow on CPU if a TF GPU backend is visible. This does not move
    # the PyTorch model; it only affects TensorFlow.
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except Exception as exc:
        raise RuntimeError("Failed to force TensorFlow CPU-only execution.") from exc

    try:
        from hepinfo.util import MILoss
    except Exception as exc:
        raise RuntimeError(
            "Failed to import hepinfo.util.MILoss. "
            "For this verification branch, BinaryMI.py is intentionally not imported."
        ) from exc

    try:
        mi_loss = MILoss(use_quantized_sigmoid=False, bits_bernoulli_sigmoid=8)
    except Exception as exc:
        raise RuntimeError("Failed to construct hepinfo.util.MILoss.") from exc

    _HEPINFO_TF = tf
    _HEPINFO_MI_LOSS = mi_loss
    return _HEPINFO_TF, _HEPINFO_MI_LOSS

def _hepinfo_tf_mi_loss_value(
    latent: torch.Tensor,
    sensitive: torch.Tensor,
) -> torch.Tensor:
    """Compute original hepinfo MI loss value inside ADatL1.

    This is verification-only. It deliberately detaches from PyTorch autograd.
    """
    if not torch.is_tensor(latent):
        raise TypeError(f"latent must be a torch.Tensor, got {type(latent)}")

    if not torch.is_tensor(sensitive):
        raise TypeError(f"sensitive must be a torch.Tensor, got {type(sensitive)}")

    if not torch.is_floating_point(latent):
        raise TypeError(f"latent must be floating point, got dtype={latent.dtype}")

    if latent.ndim < 2:
        raise ValueError(f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}")

    batch_size = latent.shape[0]

    if sensitive.ndim == 2:
        if sensitive.shape[1] != 1:
            raise ValueError(
                "For hepinfo MI verification, sensitive must have shape [batch] "
                f"or [batch, 1]. Got {tuple(sensitive.shape)}"
            )
        sensitive_1d = sensitive[:, 0]
    elif sensitive.ndim == 1:
        sensitive_1d = sensitive
    else:
        raise ValueError(
            "For hepinfo MI verification, sensitive must have shape [batch] "
            f"or [batch, 1]. Got {tuple(sensitive.shape)}"
        )

    if sensitive_1d.shape[0] != batch_size:
        raise ValueError(
            f"Sensitive batch size {sensitive_1d.shape[0]} does not match "
            f"latent batch size {batch_size}."
        )

    allowed_sensitive_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if sensitive_1d.dtype not in allowed_sensitive_dtypes:
        raise TypeError(
            "sensitive must contain integer bin labels before calling hepinfo MI. "
            f"Got dtype={sensitive_1d.dtype}. Do not pass raw continuous FET.Et here; "
            "pass the FixedQuantileSensitiveBinner output."
        )

    tf, hepinfo_mi_loss = _get_hepinfo_tf_mi_loss()

    # Verification-only bridge:
    # - detach: remove PyTorch autograd history
    # - cpu: TensorFlow CPU execution
    # - numpy: cross-framework transfer
    # - float32/int32: stable boundary dtypes
    latent_np = (
        torch.flatten(latent.detach(), start_dim=1)
        .to(dtype=torch.float32)
        .cpu()
        .contiguous()
        .numpy()
    )

    sensitive_np = (
        sensitive_1d.detach()
        .to(dtype=torch.int32)
        .cpu()
        .contiguous()
        .numpy()
    )

    if latent_np.ndim != 2:
        raise RuntimeError(f"Expected latent_np to be rank 2, got shape={latent_np.shape}")

    if sensitive_np.ndim != 1:
        raise RuntimeError(f"Expected sensitive_np to be rank 1, got shape={sensitive_np.shape}")

    if latent_np.shape[0] != sensitive_np.shape[0]:
        raise RuntimeError(
            f"latent_np batch {latent_np.shape[0]} != sensitive_np batch {sensitive_np.shape[0]}"
        )

    if latent_np.dtype != np.float32:
        raise RuntimeError(f"latent_np must be float32, got {latent_np.dtype}")

    if sensitive_np.dtype != np.int32:
        raise RuntimeError(f"sensitive_np must be int32, got {sensitive_np.dtype}")

    try:
        latent_tf = tf.convert_to_tensor(latent_np, dtype=tf.float32)
        sensitive_tf = tf.convert_to_tensor(sensitive_np, dtype=tf.int32)
    except Exception as exc:
        raise RuntimeError("Failed to convert PyTorch tensors to TensorFlow tensors.") from exc

    try:
        # Use .call(...) to directly execute hepinfo's original MILoss.call body.
        # Avoid relying on Keras Loss.__call__ reduction semantics.
        mi_tf = hepinfo_mi_loss.call(sensitive_tf, latent_tf)
        mi_tf = tf.convert_to_tensor(mi_tf)
    except Exception as exc:
        raise RuntimeError("Original hepinfo MI loss call failed.") from exc

    try:
        mi_np = np.asarray(mi_tf.numpy(), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError("Failed to convert TensorFlow MI result to NumPy.") from exc

    if mi_np.size != 1:
        raise RuntimeError(f"Expected scalar MI result, got shape={mi_np.shape}, size={mi_np.size}")

    mi_value = float(mi_np.reshape(-1)[0])

    if not np.isfinite(mi_value):
        raise RuntimeError(f"hepinfo MI returned NaN/Inf: {mi_value}")

    try:
        return latent.new_tensor(mi_value, dtype=latent.dtype)
    except Exception as exc:
        raise RuntimeError("Failed to convert TensorFlow MI scalar back to PyTorch tensor.") from exc


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
        forbid_sensitive_variable_in_input: bool =  True,
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
        z = self.encoder(x)
        reconstruction = self.decoder(z)
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

        control_required = (
            getattr(self, "control_object_feature_map", None) is not None
            and getattr(self, "object_feature_map", None) is not None
            and self.control_object_feature_map != self.object_feature_map
        )

        if b.control_x is None:
            if control_required:
                raise RuntimeError(
                    "Batch does not contain control_x, but control_object_feature_map "
                    "differs from object_feature_map. The MI sensitive variable cannot be "
                    "extracted safely. Check that L1ADDataset.__iter__ yields "
                    "(x, mask, l1bit, y, control_x, control_mask) when control_data is set."
                )

            control_x = b.x
            control_mask = b.mask
        else:
            control_x = b.control_x
            control_mask = b.control_mask

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

        x_noisy = x
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x) * self.input_noise_std
            if m is not None:
                noise = noise * m

            x_noisy = x + noise

        z, reconstruction = self.forward(x_noisy)
        reco_loss = self.reco_loss(target=x, reco=reconstruction, mask=m)

        # Sanity check / debug trace, only used once
        #TODO: Delete
        if self.training and self.global_step < 3:
            x_flat = torch.flatten(b.x, start_dim=1)
            control_flat = torch.flatten(control_x, start_dim=1)

            print("[DEBUG][AE] model input shape:", tuple(x_flat.shape))
            print("[DEBUG][AE] control_x shape:", tuple(control_flat.shape))
            print("[DEBUG][AE] b.control_x is None:", b.control_x is None)

            print(
                "[DEBUG][AE] FET.Et in model map:",
                "FET" in self.object_feature_map
                and "Et" in self.object_feature_map.get("FET", {}),
            )
            print(
                "[DEBUG][AE] FET.Et in control map:",
                "FET" in self.control_object_feature_map
                and "Et" in self.control_object_feature_map.get("FET", {}),
            )

            fet_values = self.sensitive_binner.extract_values(
                x=control_x,
                mask=control_mask,
                object_feature_map=self.control_object_feature_map,
                normalizer=None,
            )

            finite = torch.isfinite(fet_values)
            print("[DEBUG][AE] FET.Et values shape:", tuple(fet_values.shape))
            print("[DEBUG][AE] FET.Et finite:", int(finite.sum()), "/", fet_values.numel())
            print("[DEBUG][AE] FET.Et NaNs:", int(torch.isnan(fet_values).sum()))
            print("[DEBUG][AE] FET.Et infs:", int(torch.isinf(fet_values).sum()))

            if finite.any():
                vals = fet_values[finite]
                print("[DEBUG][AE] FET.Et min:", float(vals.min()))
                print("[DEBUG][AE] FET.Et max:", float(vals.max()))
                print("[DEBUG][AE] FET.Et mean:", float(vals.mean()))
                print("[DEBUG][AE] FET.Et std:", float(vals.std(unbiased=False)))
                print("[DEBUG][AE] FET.Et unique first 4096:", int(torch.unique(vals[:4096]).numel()))

        sensitive = self._compute_sensitive_bins(x=control_x, mask=control_mask)

        # Sanity check / debug trace, only used once
        #TODO: Delete
        if self.training and self.global_step < 3:
            unique, counts = torch.unique(sensitive.detach().cpu(), return_counts=True)
            print("[DEBUG][AE] sensitive bin counts:", dict(zip(unique.tolist(), counts.tolist())))

        if self.training and self.global_step < 3:
            unique, counts = torch.unique(sensitive.detach().cpu(), return_counts=True)
            print(
                f"[MI] step={self.global_step} sensitive bins: "
                f"{dict(zip(unique.tolist(), counts.tolist()))}"
            )


        mi_loss = _hepinfo_tf_mi_loss_value(latent=z, sensitive=sensitive)

        with torch.no_grad():
            perm = torch.randperm(sensitive.shape[0], device=sensitive.device)
            sensitive_perm = sensitive[perm]
            mi_loss_permuted = self.mi_loss(latent=z.detach(), sensitive=sensitive_perm)

        gamma_mi_loss = self.mi_gamma * mi_loss
        total_loss = reco_loss.mean() + gamma_mi_loss

        with torch.no_grad():
            z_detached = z.detach()
            probs = torch.sigmoid(self.mi_loss.mi_loss.temperature * z_detached)

            reco_loss_mean = reco_loss.mean().detach()
            gamma_mi_loss_detached = gamma_mi_loss.detach()

            mi_to_reco_ratio = gamma_mi_loss_detached / reco_loss_mean.clamp_min(1e-12)

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
                "loss": total_loss,

                "loss/mean": total_loss.detach(),
                "loss/reco": reco_loss.mean().detach(),

                # Existing MI log should remain the actual training MI.
                "loss/mi": mi_loss.detach(),
                "loss/gamma_mi": gamma_mi_loss.detach(),

                "ascore/operational": operational_ascore,
                "loss/full": reco_loss.detach(),
                "ascore/full": ascore.detach(),
                "reconstructed_data": reconstruction.detach(),
                "loss/mi_to_reco_ratio": mi_to_reco_ratio,
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
            # "sensitive_bin_min": outdict.get("sensitive/bin_min"),
            # "sensitive_bin_max": outdict.get("sensitive/bin_max"),
            # "sensitive_bin_mean": outdict.get("sensitive/bin_mean"),

            # Existing anomaly-score logging:
            "ascore_operational": outdict.get("ascore/operational"),

            # "latent_mean": outdict.get("latent/mean"),
            # "latent_std": outdict.get("latent/std"),

            # "bernoulli_prob_mean": outdict.get("bernoulli_prob/mean"),
            # "bernoulli_prob_std": outdict.get("bernoulli_prob/std"),
            # "bernoulli_prob_min": outdict.get("bernoulli_prob/min"),
            # "bernoulli_prob_max": outdict.get("bernoulli_prob/max"),
            # "bernoulli_prob_saturation_low": outdict.get("bernoulli_prob/saturation_low"),
            # "bernoulli_prob_saturation_high": outdict.get("bernoulli_prob/saturation_high"),
            # "loss_mi_permuted": outdict.get("loss/mi_permuted"),
            # "loss_mi_minus_permuted": outdict.get("loss/mi_minus_permuted"),
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

    def _compute_sensitive_bins(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
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

