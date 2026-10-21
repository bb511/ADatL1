import torch
from torch import Tensor

# (beta, pair) float64 blocks of 2**21 elements are 16 MB; with the handful alive at
# once the peak stays near 100 MB, whatever the number of pairs.
_BLOCK_ELEMENTS = 2**21


def _log_2cosh(z: Tensor) -> Tensor:
    return torch.logaddexp(z, -z)


def posterior_agreement_curve(gap1: Tensor, gap2: Tensor, betas: Tensor) -> Tensor:
    """Return the mean two-state posterior agreement of paired scores at every beta.

    Args:
        gap1: Energy gaps D = E(p, 1) - E(p, 0) of the first paired scores [n_pairs]
        gap2: Energy gaps of the second paired scores [n_pairs]
        betas: Inverse temperatures [n_beta]

    Returns:
        mean_i k_i(beta) at every beta [n_beta]
    """
    curve = []
    for block in betas.split(max(1, _BLOCK_ELEMENTS // max(1, gap1.numel()))):
        x = 0.5 * block[:, None] * gap1
        y = 0.5 * block[:, None] * gap2
        # Eq. cap_kernel of the paper, with numerator and partition functions divided
        # by exp(-beta (E1(0) + E1(1) + E2(0) + E2(1)) / 2), so that
        #   k_i = log 2cosh(x + y) - log 2cosh(x) - log 2cosh(y) = log((1 + m1 m2) / 2),
        # m_k = tanh(beta D_k / 2). This form is finite at every beta: the exp/log
        # form overflows, and log1p(m1 m2) is -inf once both tanh of a disagreeing
        # pair round to +-1 (|beta D| above about 37 in float64).
        k = _log_2cosh(x + y) - _log_2cosh(x) - _log_2cosh(y)
        curve.append(k.mean(dim=1))
    return torch.cat(curve)
