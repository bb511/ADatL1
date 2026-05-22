import torch
from torch import nn

from src.algorithms.components.utils import RandomNumberGenerator
from src.data.components.normalization import L1DataNormalizer


class FastFeatureBlur(nn.Module):
    """
    Selectively blurs features by mixing with random values.

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
        mask_strength = (
            torch.rand(b, d, device=x.device, generator=gen) < self.strength
        ).float()
        mask = mask_p * mask_strength * self.magnitude

        # rand_like has no generator=... so use rand(shape,...)
        rnd = torch.rand(x.shape, device=x.device, generator=gen)
        return x * (1 - mask) + rnd * mask


class FastObjectMask(nn.Module):
    """
    Randomly zeros out a subset of features in the input tensor during training.

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


class FastLorentzRotation(nn.Module):
    """
    Applies a random rotation to the phi angles of input features with given probability.

    :param p: Probability of applying the rotation to each batch item
    :param normalizer: Normalizer object used to initially normalize the data, used here
        to denormalize it such that the augmentation can be applied.
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

        self.register_buffer("l1_scale_phi", l1_scale_phi)
        self.register_buffer("phi_mask", phi_mask)

    def forward(self, x: torch.Tensor):
        gen = self.rng.get_generator(x.device)
        self.l1_scale_phi = self.l1_scale_phi.to(device=x.device)

        b = x.shape[0]
        bool_mask = (torch.rand(b, device=x.device, generator=gen) < self.prob).float()

        augm_x = self.normalizer.denorm_1d_tensor(x.clone())
        raw_phi = augm_x[:, self.phi_mask]

        raw_phi = raw_phi * self.l1_scale_phi
        rotation = (torch.rand(b, device=x.device, generator=gen) * 2 * torch.pi)[
            :, None
        ]
        rotated_phi = torch.remainder(raw_phi + rotation, 2 * torch.pi)
        rotated_phi = rotated_phi / self.l1_scale_phi
        raw_phi = raw_phi / self.l1_scale_phi

        augm_x[:, self.phi_mask] = (
            bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * raw_phi
        ).float()

        return self.normalizer.norm_1d_tensor(augm_x)
