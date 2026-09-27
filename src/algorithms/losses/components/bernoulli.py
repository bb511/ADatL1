from __future__ import annotations

import torch
from torch import nn


class BernoulliActivation(nn.Module):
    """Bernoulli sampling front end, factored out of the MI estimator.

    This is the piece that turns a real-valued latent activation into the
    Bernoulli parameter of a bit, exactly as hepinfo does before either
    sampling from it (``BernoulliSampling`` in ``hepinfo/qkerasV3.py``) or
    feeding it to the mutual-information estimator (``BinaryMI``):

        p = sigmoid(temperature * latent)

    with ``temperature = 6`` reproducing hepinfo's default sharpness. Set
    ``input_is_logits=False`` when the caller already provides probabilities,
    in which case the input is passed through untouched.

    :param temperature: Sharpness of the logistic map. hepinfo uses 6.
    :param input_is_logits: If False, inputs are already probabilities and are
        returned unchanged.
    """

    def __init__(
        self,
        temperature: float = 6.0,
        input_is_logits: bool = True,
    ) -> None:
        super().__init__()

        self.temperature = float(temperature)
        self.input_is_logits = bool(input_is_logits)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the Bernoulli probability of each latent unit."""

        if not self.input_is_logits:
            return latent

        return torch.sigmoid(self.temperature * latent)

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, "
            f"input_is_logits={self.input_is_logits}"
        )
