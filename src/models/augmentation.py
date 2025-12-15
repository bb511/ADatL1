import torch
from torch import nn

from src.models.utils import RandomNumberGenerator

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
        mask_strength = (torch.rand(b, d, device=x.device, generator=gen) < self.strength).float()
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
    :param norm_scale: Normalization scale factors
    :param norm_bias: Normalization bias values
    :param phi_indices: Indices of phi angles in the feature vector
    """

    def __init__(self, prob: float, norm_scale: torch.Tensor, norm_bias: torch.Tensor, phi_mask: torch.Tensor):
        super().__init__()
        self.prob = prob
        self.rng = RandomNumberGenerator()

        self.register_buffer(
            "l1_scale",
            torch.tensor([144] * 1 + [144] * 4 + [576] * 4 + [144] * 10) / (2 * torch.pi),
        )
        self.register_buffer("phi_mask", phi_mask)
        self.register_buffer("scale", torch.where(~phi_mask, norm_scale, torch.tensor(1.0)))
        self.register_buffer("bias", torch.where(~phi_mask, norm_bias, torch.tensor(0.0)))

    def forward(self, x: torch.Tensor):
        gen = self.rng.get_generator(x.device)
        b = x.shape[0]

        bool_mask = (torch.rand(b, device=x.device, generator=gen) < self.prob).float()

        original_phi = (x * self.scale + self.bias)[:, self.phi_mask]
        original_phi = original_phi / self.l1_scale

        rotation = (torch.rand(b, device=x.device, generator=gen) * 2 * torch.pi)[:, None]
        rotated_phi = torch.remainder(original_phi + rotation, 2 * torch.pi) * self.l1_scale

        result = x.clone()
        result[:, self.phi_mask] = (
            (bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * original_phi).float()
            - self.bias[self.phi_mask]
        ) / self.scale[self.phi_mask]
        return result
