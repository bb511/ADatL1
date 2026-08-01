from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn


class ObjectTransformerEncoder(nn.Module):
    """Transformer encoder over flattened L1 object features.

    The flattened feature tensor is converted back into object tokens using the
    datamodule's ``object_feature_map``. Each token receives a shared feature
    projection plus an object-type embedding, then a Transformer encoder maps the
    event to a fixed latent vector.
    """

    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...] = ("Et", "eta", "phi"),
        object_types: list[str]
        | tuple[str, ...] = (
            "FET",
            "egammas",
            "jets",
            "muons",
            "taus",
        ),
        feature_dim: int | None = None,
        d_model: int = 64,
        out_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be one of 'cls' or 'mean'.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.feature_names = list(feature_names)
        self.object_types = list(object_types)
        self.feature_dim = int(feature_dim or len(self.feature_names))
        self.d_model = int(d_model)
        self.out_dim = int(out_dim)
        self.pooling = pooling
        self.object_feature_map: dict | None = None

        self.feature_proj = nn.Linear(self.feature_dim, self.d_model)
        self.type_embedding = nn.Embedding(len(self.object_types), self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.output = (
            nn.Identity()
            if self.out_dim == self.d_model
            else nn.Linear(self.d_model, self.out_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_object_feature_map(self, object_feature_map: dict) -> None:
        """Set flattened feature indices for each object type."""
        self.object_feature_map = object_feature_map

    def forward(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of flattened L1 events into latent vectors."""
        if x_flat.ndim != 2:
            x_flat = torch.flatten(x_flat, start_dim=1)
        if m_flat is not None and m_flat.ndim != 2:
            m_flat = torch.flatten(m_flat, start_dim=1).float()

        token_x, token_mask, token_type = self._tokens_from_flat(x_flat, m_flat)
        h = self.feature_proj(token_x) + self.type_embedding(token_type)

        bsz = x_flat.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1)

        cls_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=x_flat.device)
        padding_mask = torch.cat([cls_mask, ~token_mask], dim=1)

        h = self.encoder(h, src_key_padding_mask=padding_mask)
        if self.pooling == "cls":
            pooled = h[:, 0]
        else:
            valid = token_mask.unsqueeze(-1).float()
            pooled = (h[:, 1:] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        return self.output(self.norm(pooled))

    def _tokens_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build object tokens, presence masks, and type indices."""
        if self.object_feature_map is None:
            return self._fallback_tokens(x_flat, m_flat)

        token_values = []
        token_masks = []
        token_types = []

        for type_idx, obj_name in enumerate(self.object_types):
            feature_map = self.object_feature_map.get(obj_name)
            if not feature_map:
                continue

            n_obj = self._num_objects(feature_map.values())
            for obj_idx in range(n_obj):
                values = []
                masks = []
                energy_mask = None
                for feat_name in self.feature_names[: self.feature_dim]:
                    idxs = feature_map.get(feat_name, [])
                    if obj_idx >= len(idxs):
                        values.append(torch.zeros(x_flat.shape[0], device=x_flat.device))
                        continue

                    idx = int(idxs[obj_idx])
                    values.append(x_flat[:, idx])
                    if m_flat is not None:
                        feature_mask = m_flat[:, idx].bool()
                        masks.append(feature_mask)
                        if feat_name == "Et":
                            energy_mask = feature_mask

                while len(values) < self.feature_dim:
                    values.append(torch.zeros(x_flat.shape[0], device=x_flat.device))

                token_values.append(torch.stack(values, dim=1))
                if energy_mask is not None:
                    # Some object types do not physically define every feature in the
                    # unified schema.  In particular, FET has (Et, phi) but receives a
                    # structurally empty eta column during preprocessing.  Et is the
                    # authoritative object-presence mask, so that structural columns
                    # neither remove a real token nor admit an otherwise empty token.
                    token_masks.append(energy_mask)
                elif masks:
                    # Non-kinematic schemas may not have Et; for those, preserve the
                    # fallback convention that any observed feature makes a token.
                    token_masks.append(torch.stack(masks, dim=1).any(dim=1))
                else:
                    token_masks.append(
                        torch.ones(x_flat.shape[0], dtype=torch.bool, device=x_flat.device)
                    )
                token_types.append(type_idx)

        if not token_values:
            return self._fallback_tokens(x_flat, m_flat)

        token_x = torch.stack(token_values, dim=1)
        token_mask = torch.stack(token_masks, dim=1)
        token_type = torch.tensor(token_types, dtype=torch.long, device=x_flat.device)
        token_type = token_type.unsqueeze(0).expand(x_flat.shape[0], -1)
        return token_x, token_mask, token_type

    def _fallback_tokens(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fixed-width tokens when no object feature map is available."""
        bsz, n_features = x_flat.shape
        pad = (-n_features) % self.feature_dim
        if pad:
            x_flat = torch.cat(
                [x_flat, torch.zeros(bsz, pad, dtype=x_flat.dtype, device=x_flat.device)],
                dim=1,
            )
            if m_flat is not None:
                m_flat = torch.cat(
                    [
                        m_flat,
                        torch.zeros(bsz, pad, dtype=m_flat.dtype, device=m_flat.device),
                    ],
                    dim=1,
                )

        token_x = x_flat.reshape(bsz, -1, self.feature_dim)
        if m_flat is None:
            token_mask = torch.ones(bsz, token_x.shape[1], dtype=torch.bool, device=x_flat.device)
        else:
            token_mask = m_flat.reshape(bsz, -1, self.feature_dim).bool().any(dim=2)

        token_type = torch.zeros(bsz, token_x.shape[1], dtype=torch.long, device=x_flat.device)
        return token_x, token_mask, token_type

    @staticmethod
    def _num_objects(feature_indices: Iterable) -> int:
        """Return the largest object multiplicity represented in a feature map."""
        return max((len(v) for v in feature_indices), default=0)


def safe_module_name(name: str) -> str:
    """Replace characters that are unsafe in PyTorch module names."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)
