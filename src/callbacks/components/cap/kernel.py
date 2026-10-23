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


def log_norm(gap: Tensor, beta: float) -> Tensor:
    """Return the log norm of every event's two-state posterior q at one beta.

    The squared norm is (1 + m^2) / 2 with m = tanh(beta D / 2), and its half log is
    written as (log 2cosh(beta D) - 2 log 2cosh(beta D / 2)) / 2, finite at every beta.
    """
    u = beta * gap
    return 0.5 * (_log_2cosh(u) - 2.0 * _log_2cosh(0.5 * u))


def _unit_mass(weight: Tensor | None, gap: Tensor) -> Tensor:
    if weight is None:
        return torch.full_like(gap, 1.0 / gap.numel())
    return weight / weight.sum()


def _null_agreement(
    gap1: Tensor, gap2: Tensor, beta: float, weight1: Tensor, weight2: Tensor
) -> Tensor:
    x, y = 0.5 * beta * gap1, 0.5 * beta * gap2
    rows = max(1, _BLOCK_ELEMENTS // max(1, y.numel()))
    blocks = zip(x.split(rows), weight1.split(rows))
    joint = sum(w @ (_log_2cosh(xb[:, None] + y) @ weight2) for xb, w in blocks)
    return joint - weight1 @ _log_2cosh(x) - weight2 @ _log_2cosh(y)


def null_agreement_curve(
    gap1: Tensor,
    gap2: Tensor,
    betas: Tensor,
    weight1: Tensor | None = None,
    weight2: Tensor | None = None,
) -> Tensor:
    """Return the mean kernel over every pair (i, j) of the two samples at every beta.

    The mean is weighted by weight1[i] weight2[j], each weight vector normalised here;
    the weights carry the masses of binned samples, and None weighs every event
    equally. The marginal terms of k separate, so that log 2cosh(x_i + y_j) is the
    only term evaluated on the N1 x N2 grid, in blocks of rows.

    Args:
        gap1: Energy gaps of the first sample [N1]
        gap2: Energy gaps of the second sample [N2]
        betas: Inverse temperatures [n_beta]
        weight1: Masses of the first sample's events [N1]
        weight2: Masses of the second sample's events [N2]

    Returns:
        The mean of k_ij(beta) at every beta [n_beta]
    """
    weight1, weight2 = _unit_mass(weight1, gap1), _unit_mass(weight2, gap2)
    curve = [_null_agreement(gap1, gap2, b, weight1, weight2) for b in betas.tolist()]
    return torch.stack(curve)


def indexed_agreement(gap1: Tensor, gap2: Tensor, index: Tensor, beta: float) -> Tensor:
    """Return the mean kernel of gap1 against gap2[index[r]] for every row r of index.

    Args:
        gap1: Energy gaps of the first paired sample [n_pairs]
        gap2: Energy gaps of the second sample [N2]
        index: Partners in gap2 of the events of gap1, one row per draw
            [n_draws, n_pairs]
        beta: Inverse temperature

    Returns:
        mean_i k(gap1[i], gap2[index[r, i]]; beta) for every draw r [n_draws]
    """
    x, y = 0.5 * beta * gap1, 0.5 * beta * gap2
    rows = max(1, _BLOCK_ELEMENTS // max(1, x.numel()))
    joint = torch.cat([_log_2cosh(x + y[b]).mean(dim=1) for b in index.split(rows)])
    return joint - _log_2cosh(x).mean() - _log_2cosh(y)[index].mean(dim=1)


def permuted_agreement(
    gap1: Tensor,
    gap2: Tensor,
    beta: float,
    n_permutations: int,
    generator: torch.Generator,
) -> Tensor:
    """Return the mean kernel of gap1 against random permutations of gap2.

    Args:
        gap1: Energy gaps of the first paired sample [n_pairs]
        gap2: Energy gaps of the second paired sample [n_pairs]
        beta: Inverse temperature
        n_permutations: Number of random permutations
        generator: Source of the permutations

    Returns:
        The mean kernel of every permutation [n_permutations]
    """
    # One randperm per row: three to five times faster than the argsort of uniforms.
    perms = [
        torch.randperm(gap2.numel(), generator=generator) for _ in range(n_permutations)
    ]
    return indexed_agreement(gap1, gap2, torch.stack(perms), beta)
