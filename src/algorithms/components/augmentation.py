import torch
from torch import nn

from src.algorithms.components.utils import RandomNumberGenerator
from src.data.components.normalization import L1DataNormalizer


class FastFeatureBlur(nn.Module):
    """Selectively blurs features by mixing with random values.

    :param p: Probability of applying the noise transformation (0 to 1).
    :param magnitude: Intensity of the blurring effect.
    :param strength: Probability of each feature being affected.
    """

    def __init__(self, prob: float, magnitude: float, strength: float):
        super().__init__()
        self.prob = prob
        self.magnitude = magnitude
        self.strength = strength
        self.rng = RandomNumberGenerator()

    def forward(self, x: torch.Tensor):
        b, d = x.shape
        gen = self.rng.get_generator(x.device)

        mask_p = (torch.rand(b, 1, device=x.device, generator=gen) < self.prob).float()
        mask_strength = (torch.rand(b, d, device=x.device, generator=gen) < self.strength).float()
        mask = mask_p * mask_strength * self.magnitude

        # rand_like has no generator=... so use rand(shape,...)
        rnd = torch.rand(x.shape, device=x.device, generator=gen)
        return x * (1 - mask) + rnd * mask


class FastObjectMask(nn.Module):
    """Randomly zeros out a subset of features in the input tensor during training.

    :param p: Probability of applying the masking transformation (0 to 1).
    """

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob
        self.rng = RandomNumberGenerator()

    def forward(self, x: torch.Tensor):
        gen = self.rng.get_generator(x.device)

        if torch.rand((), device=x.device, generator=gen).item() > self.prob:
            return x

        b, d = x.shape
        mask = (torch.rand(b, d, device=x.device, generator=gen) > self.prob).float()
        return x * mask


class FastObjectDropout(nn.Module):
    """Drop complete L1 objects instead of unrelated scalar features."""

    def __init__(
        self,
        prob: float,
        object_prob: float,
        protected_object_types: tuple[str, ...] | list[str] = ("FET", "MET"),
    ):
        super().__init__()
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1.")
        if not 0.0 <= object_prob <= 1.0:
            raise ValueError("object_prob must be between 0 and 1.")
        self.prob = float(prob)
        self.object_prob = float(object_prob)
        self.protected_object_types = set(protected_object_types)
        self.rng = RandomNumberGenerator()
        self.object_indices: list[list[int]] = []

    def set_object_feature_map(self, object_feature_map: dict | None) -> None:
        self.object_indices = []
        if not object_feature_map:
            return

        for object_type, feature_map in object_feature_map.items():
            if object_type in self.protected_object_types:
                continue
            n_objects = max((len(indices) for indices in feature_map.values()), default=0)
            for object_idx in range(n_objects):
                indices = sorted(
                    {
                        int(feature_indices[object_idx])
                        for feature_indices in feature_map.values()
                        if object_idx < len(feature_indices)
                    }
                )
                if indices:
                    self.object_indices.append(indices)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.object_indices or self.object_prob == 0.0 or self.prob == 0.0:
            return x if mask is None else (x, mask)

        gen = self.rng.get_generator(x.device)
        batch_size = x.shape[0]
        apply = torch.rand(batch_size, 1, device=x.device, generator=gen) < self.prob
        drop = (
            torch.rand(
                batch_size,
                len(self.object_indices),
                device=x.device,
                generator=gen,
            )
            < self.object_prob
        ) & apply

        x_out = x.clone()
        mask_out = None if mask is None else mask.clone()
        for object_idx, feature_indices in enumerate(self.object_indices):
            rows = torch.nonzero(drop[:, object_idx], as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            columns = torch.tensor(feature_indices, dtype=torch.long, device=x.device)
            x_out[rows[:, None], columns[None, :]] = 0.0
            if mask_out is not None:
                mask_out[rows[:, None], columns[None, :]] = 0.0

        return x_out if mask is None else (x_out, mask_out)


class FastDetectorSmearing(nn.Module):
    """Apply Gaussian noise in physical detector-resolution units."""

    def __init__(
        self,
        prob: float,
        strength: float,
        normalizer: L1DataNormalizer,
        resolution_tensor: torch.Tensor,
        nonnegative_mask: torch.Tensor | None = None,
        periodic_scale_tensor: torch.Tensor | None = None,
    ):
        super().__init__()
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1.")
        if strength < 0.0:
            raise ValueError("strength must be non-negative.")
        self.prob = float(prob)
        self.strength = float(strength)
        self.normalizer = normalizer
        self.rng = RandomNumberGenerator()
        self.register_buffer("resolution_tensor", resolution_tensor.float(), persistent=False)
        self.register_buffer(
            "nonnegative_mask",
            torch.zeros_like(resolution_tensor, dtype=torch.bool)
            if nonnegative_mask is None
            else nonnegative_mask.bool(),
            persistent=False,
        )
        self.register_buffer(
            "periodic_scale_tensor",
            torch.zeros_like(resolution_tensor, dtype=torch.float32)
            if periodic_scale_tensor is None
            else periodic_scale_tensor.float(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.prob == 0.0 or self.strength == 0.0:
            return x if mask is None else (x, mask)

        gen = self.rng.get_generator(x.device)
        batch_size = x.shape[0]
        apply = (torch.rand(batch_size, 1, device=x.device, generator=gen) < self.prob).to(x.dtype)
        valid = torch.ones_like(x) if mask is None else mask.to(dtype=x.dtype)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        noise.mul_(self.resolution_tensor.to(dtype=x.dtype))
        noise.mul_(self.strength * apply * valid)

        raw = self.normalizer.denorm_1d_tensor(x.clone())
        raw.add_(noise)
        if self.nonnegative_mask.any():
            raw[:, self.nonnegative_mask] = raw[:, self.nonnegative_mask].clamp_min(0.0)
        periodic_mask = self.periodic_scale_tensor > 0
        if periodic_mask.any():
            scales = self.periodic_scale_tensor[periodic_mask].to(dtype=raw.dtype)
            angles = raw[:, periodic_mask] * scales
            angles = torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi
            raw[:, periodic_mask] = angles / scales
        x_out = self.normalizer.norm_1d_tensor(raw)
        return x_out if mask is None else (x_out, mask)


class FastLorentzRotation(nn.Module):
    """Applies a random rotation to the phi angles of input features with given probability.

    :param p: Probability of applying the rotation to each batch item
    :param normalizer: Normalizer object used to initially normalize the data, used here to
        denormalize it such that the augmentation can be applied.
    :param phi_mask: Masks the indices corresponding to the phi angle of each object.
    """

    def __init__(
        self,
        prob: float,
        normalizer: L1DataNormalizer,
        phi_mask: torch.Tensor,
        l1_scale_phi: torch.Tensor,
    ):
        super().__init__()
        self.prob = prob
        self.rng = RandomNumberGenerator()
        self.normalizer = normalizer

        self.register_buffer("l1_scale_phi", l1_scale_phi, persistent=False)
        self.register_buffer("phi_mask", phi_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        gen = self.rng.get_generator(x.device)
        self.l1_scale_phi = self.l1_scale_phi.to(device=x.device)

        b = x.shape[0]
        bool_mask = (torch.rand(b, device=x.device, generator=gen) < self.prob).float()

        augm_x = self.normalizer.denorm_1d_tensor(x.clone())
        raw_phi = augm_x[:, self.phi_mask]

        raw_phi = raw_phi * self.l1_scale_phi
        rotation = (torch.rand(b, device=x.device, generator=gen) * 2 * torch.pi)[:, None]
        rotated_phi = torch.remainder(raw_phi + rotation + torch.pi, 2 * torch.pi) - torch.pi
        rotated_phi = rotated_phi / self.l1_scale_phi
        raw_phi = raw_phi / self.l1_scale_phi

        augm_x[:, self.phi_mask] = (
            bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * raw_phi
        ).float()

        x_out = self.normalizer.norm_1d_tensor(augm_x)
        return x_out if mask is None else (x_out, mask)
