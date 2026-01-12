from typing import Tuple, Dict, Union
import os
import json
import pickle
import copy
import numpy as np

import torch
import torch.nn as nn
import keras

from src.algorithms import L1ADLightningModule
from src.algorithms.components.mlp import MLP
from src.algorithms.components.augmentation import FastFeatureBlur
from src.algorithms.components.augmentation import FastObjectMask
from src.algorithms.components.augmentation import FastLorentzRotation


class VICReg(L1ADLightningModule):
    """VICreg deep learning algorithm.

    See https://arxiv.org/abs/2105.04906 for more details.
    Used to generate an intermediate representation on which one can run the anomaly
    detection algorithm of choice on.

    :param projector: Projection head module.
    :param feature_blur: Feature blur augmentation module.
    :param object_mask: Object mask augmentation module.
    :param lorentz_rotation: Lorentz rotation augmentation module.
    :param seed: Int to seed the augmentations with.
    :param diagnosis_metrics: Bool whether to run all the diagnostics or not.
    :param ckpt: String with path pointing to a pytorch lightning checkpoint.
    """

    def __init__(
        self,
        projector: MLP,
        feature_blur: FastFeatureBlur = None,
        object_mask: FastObjectMask = None,
        lorentz_rotation: FastLorentzRotation = None,
        seed: int = 42,
        diagnosis_metrics: bool = False,
        ckpt: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Ignore saving hyperparams to avoid memory bloating.
        self.save_hyperparameters(
            ignore=[
                "loss",
                "projector",
                "feature_blur",
                "object_mask",
                "lorentz_rotation",
            ]
        )
        self.seed = seed
        self.diagnosis_metrics = diagnosis_metrics
        self.ckpt_path = ckpt

        self.projector = projector

        # Instantiate augmentation modules with different rngs
        self.feat_blurs = {}
        feature_blur_1 = copy.deepcopy(feature_blur)
        feature_blur_2 = copy.deepcopy(feature_blur)
        feature_blur_1.rng.set_seed(self.seed)
        feature_blur_2.rng.set_seed(self.seed + 1)
        self.feat_blurs.update({'1': feature_blur_1})
        self.feat_blurs.update({'2': feature_blur_2})


        self.obj_masks = {}
        object_mask_1 = copy.deepcopy(object_mask)
        object_mask_2 = copy.deepcopy(object_mask)
        object_mask_1.rng.set_seed(self.seed)
        object_mask_2.rng.set_seed(self.seed + 1)
        self.obj_masks.update({'1': object_mask_1})
        self.obj_masks.update({'2': object_mask_2})

        self.lorentz_rotation = lorentz_rotation

    def on_fit_start(self):
        """Set up the Lorentz augmentations and load weights if given ckpt.

        Save them as dictionaries so pytorch lightning does not instantiate them when
        loading from checkpoints, since they are not needed for inference.
        """
        self.lorentz_rot = {}
        lorentz_rotation = self._setup_phi_lorentz_rotation()
        lorentz_rotation_1 = copy.deepcopy(lorentz_rotation)
        lorentz_rotation_2 = copy.deepcopy(lorentz_rotation)
        lorentz_rotation_1.rng.set_seed(self.seed)
        lorentz_rotation_2.rng.set_seed(self.seed + 1)
        self.lorentz_rot.update({"1": lorentz_rotation_1})
        self.lorentz_rot.update({"2": lorentz_rotation_2})

        # Load weights from checkpoint.
        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"]
            self._load_weights_from_lightning_ckpt(self.model, state_dict, 'model', False)
            self._load_weights_from_lightning_ckpt(self.projector, state_dict, 'projector', False)

    def _setup_phi_lorentz_rotation(self) -> FastLorentzRotation:
        """Sets up the lorents rotation on the phi feature of each object.

        First, get the indices of the phi values in the batch torch tensor.
        Then, get the l1 scales for the phi values, to turn them into real physical phi
        values.
        """
        object_feature_map = self.trainer.datamodule.loader.object_feature_map
        normalizer = self.trainer.datamodule.normalizer
        normalizer.setup_1d_denorm(object_feature_map)
        l1_scales = self.trainer.datamodule.l1_scales

        phi_idxs = []
        l1_scale_phi = []
        for obj_name, feature_map in object_feature_map.items():
            for feat_name, idxs in feature_map.items():
                if feat_name != "phi":
                    continue

                phi_idxs.extend(idxs)
                nconst = len(idxs)
                l1_scale_phi.extend(nconst * [l1_scales[obj_name]["phi"]])

        phi_mask = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        phi_mask[phi_idxs] = True
        l1_scale_phi = torch.tensor(l1_scale_phi, dtype=torch.float32)

        return self.lorentz_rotation(
            normalizer=normalizer, phi_mask=phi_mask, l1_scale_phi=l1_scale_phi
        )

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, m, _ = batch
        x = torch.flatten(x, start_dim=1)

        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        x1 = self.lorentz_rot['1'](self.obj_masks['1'](self.feat_blurs['1'](x1)))
        x2 = self.lorentz_rot['2'](self.obj_masks['2'](self.feat_blurs['2'](x2)))

        # Get projections
        z1 = self.projector(self.model(x1))
        z2 = self.projector(self.model(x2))
        del x1, x2

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)
        with torch.no_grad():
            inv_ratio = (loss_inv / loss_total).item()
            var_ratio = (loss_var / loss_total).item()
            cov_ratio = (loss_cov / loss_total).item()
            z1_std = torch.sqrt(z1.var(dim=0) + 1e-4).mean()
            z2_std = torch.sqrt(z2.var(dim=0) + 1e-4).mean()
            outdict_diag = self._compute_diagnosis_metrics(z1, z2)

        # Add additional losses if using HGQ.
        add_loss = 0.0
        if hasattr(self.model, "losses") and len(self.model.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.model.losses]).sum()

        loss_total += add_loss

        return {
            "loss": loss_total,
            # Used for logging:
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
            "loss/inv/ratio": inv_ratio,
            "loss/var/ratio": var_ratio,
            "loss/cov/ratio": cov_ratio,
            "z1/std": z1_std,
            "z2/std": z2_std,
            "vicreg_rep_data": self.model(x).detach(),
            **outdict_diag
        }

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        x, m, _ = batch
        x = torch.flatten(x, start_dim=1)
        z = self.model(x)

        loss_total = self.model_step(batch)['loss']
        return {
            "loss": loss_total.detach(),
            "vicreg_rep_data": z.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        diag_entries = {k: v for k, v in outdict.items() if k.startswith("diag_")}
        return {
            "loss": outdict.get("loss"),
            "loss_inv": outdict.get("loss/inv"),
            "loss_var": outdict.get("loss/var"),
            "loss_cov": outdict.get("loss/cov"),
            "loss_inv_ratio": outdict.get("loss/inv/ratio"),
            "loss_var_ratio": outdict.get("loss/var/ratio"),
            "loss_cov_ratio": outdict.get("loss/cov/ratio"),
            "z1_std": outdict.get("z1/std"),
            "z2_std": outdict.get("z2/std"),
            **diag_entries
        }

    def _compute_diagnosis_metrics(self, z1: torch.Tensor, z2: torch.Tensor):
        """Compute metrics to diagnose what the vicreg has learnt."""
        diag_outdict = {}
        if not self.diagnosis_metrics:
            return diag_outdict

        z1_efr, z1_max_sv = self._effective_rank_svd(z1)
        z2_efr, z2_max_sv = self._effective_rank_svd(z2)
        rep_eff_rank = min(z1_efr, z2_efr)
        max_sv = max(z1_max_sv, z2_max_sv)
        diag_outdict.update({"diag_eff_rank": rep_eff_rank, "diag_max_sv": max_sv})

        z1_cos_similarity = self._average_pairwise_cosine(z1)
        z2_cos_similarity = self._average_pairwise_cosine(z2)
        avg_cos_sim = max(z1_cos_similarity, z2_cos_similarity)
        diag_outdict.update({"diag_avg_cos_sim": avg_cos_sim})

        pair_cos = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean().item()
        diag_outdict.update({"diag_avg_pair_cos_sim": pair_cos})

        return diag_outdict

    @torch.no_grad()
    def _effective_rank_svd(self, z, eps=1e-12):
        zc = z - z.mean(dim=0, keepdim=True)
        batch_size = zc.shape[0]
        # singular values of [B, D] matrix
        s = torch.linalg.svdvals(zc)  # length = min(B, D)
        max_sv = (s[0] ** 2 / (batch_size - 1)).item()

        eigvals = (s ** 2).clamp_min(eps)

        trace_sum = eigvals.sum()
        trace_squared_sum = (eigvals ** 2).sum()

        return (trace_sum ** 2 / trace_squared_sum).item(), max_sv

    @torch.no_grad()
    def _average_pairwise_cosine(self, z):
        z = z - z.mean(dim=0, keepdim=True)
        z = torch.nn.functional.normalize(z, dim=1)  # L2 norm
        sim = z @ z.T                                # [B, B]
        B = z.shape[0]

        # remove diagonal (self-similarity = 1)
        avg_cos = (sim.sum() - B) / (B * (B - 1))
        return avg_cos.item()

    def _load_weights_from_lightning_ckpt(
        self,
        model,
        state_dict: dict,
        module_key: str,
        has_bn: bool = False,
    ):
        """Load weights from a Lightning .ckpt into either:
          - a PyTorch nn.Module (normal state_dict load), or
          - a Keras HGQ MLP (manual weight transfer).

        model: The torch.nn.Module OR keras.Model to load weights into
        module_key: String of prefix in Lightning state_dict ("model" or "projector").
        has_bn: Whether the MLP uses BatchNorm.
        """
        sub_sd = {
            k[len(module_key) + 1 :]: v
            for k, v in state_dict.items()
            if k.startswith(module_key + ".")
        }

        if isinstance(model, torch.nn.Module) and not isinstance(model, keras.Model):
            model.load_state_dict(sub_sd, strict=True)
            return model

        if not isinstance(model, keras.Model):
            raise TypeError("model must be torch.nn.Module or keras.Model")

        # force variable creation
        in_dim = model.input_shape[-1]
        _ = model(np.zeros((1, in_dim), dtype=np.float32))

        nodes = self._infer_nodes_hgq(model)

        # extract Linear / BN params in order
        linear_keys = sorted(
            k for k in sub_sd
            if k.endswith("weight")
            and not any(x in k.lower() for x in ["bn", "batchnorm"])
        )

        bn_keys = sorted(
            k for k in sub_sd
            if k.endswith("weight")
            and any(x in k.lower() for x in ["bn", "batchnorm"])
        )

        # hidden layers
        for i in range(len(nodes)):
            W = sub_sd[linear_keys[i]].numpy()
            b = sub_sd[linear_keys[i].replace("weight", "bias")].numpy()
            self._assign_dense_like_weights(model.get_layer(f"qdense_{i}"), W, b)

            if has_bn:
                prefix = bn_keys[i].replace("weight", "")
                gamma = sub_sd[prefix + "weight"].numpy()
                beta  = sub_sd[prefix + "bias"].numpy()
                mean  = sub_sd[prefix + "running_mean"].numpy()
                var   = sub_sd[prefix + "running_var"].numpy()

                model.get_layer(f"bn_{i}").set_weights([gamma, beta, mean, var])

        # output layer
        W = sub_sd[linear_keys[len(nodes)]].numpy()
        b = sub_sd[linear_keys[len(nodes)].replace("weight", "bias")].numpy()
        self._assign_dense_like_weights(model.get_layer("qdense_out"), W, b)

        return model

    def _infer_nodes_hgq(self, model):
        """Returns hidden-layer sizes (excluding output layer) of hgq MLP."""
        qdense_layers = sorted(
            [
                l for l in model.layers
                if l.name.startswith("qdense_") and l.name != "qdense_out"
            ],
            key=lambda l: int(l.name.split("_")[-1]),
        )

        if not qdense_layers:
            raise ValueError("No qdense_* layers found")

        return [l.units for l in qdense_layers]

    def _assign_dense_like_weights(self, layer, W_torch, b_torch=None):
        """Assign weights and biases to QKeras dense layer."""
        W = np.asarray(W_torch, dtype=np.float32).T
        b = None if b_torch is None else np.asarray(b_torch, dtype=np.float32)

        k = next(w for w in layer.weights if len(w.shape) == 2 and tuple(w.shape) == W.shape)
        k.assign(W)

        # bias: first 1D weight that matches b.shape (if provided)
        if b is not None:
            try:
                bb = next(w for w in layer.weights if len(w.shape) == 1 and tuple(w.shape) == b.shape)
                bb.assign(b)
            except StopIteration:
                pass  # bias may be disabled
