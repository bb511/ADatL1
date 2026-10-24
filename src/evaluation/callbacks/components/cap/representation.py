"""Fixed representations of a set's inputs: the coordinates in which the bank's
couplings pair events.

``identity`` flattens a batch [n, ...] to [n, d] and leaves every value as it is.

``colour_spectral`` maps a batch of images [n, C, H, W], per-channel normalised as the
datamodule serves them, to [n, 2 C + n_bands] in float64: the mean of every channel
over its pixels, then every channel's standard deviation (over the pixels, without
Bessel's correction), then the log energy of the grayscale image in n_bands radial
frequency bands. The grayscale image is the mean of the C normalised channels, so the
band energies depend on the normalisation statistics, which are fixed per data set.
The band edges are n_bands + 1 log-spaced radii from the lowest non-zero radial
frequency of the discrete Fourier grid, 1 / max(H, W) cycles per pixel, to the Nyquist
frequency, 1/2. A frequency of the grid whose radius r satisfies
edge_b <= r < edge_(b+1) belongs to band b, the last band holding the Nyquist radius
too; the zero frequency, which the channel means carry, and the frequencies beyond the
Nyquist radius (the corners of the grid) belong to no band. A band's energy is the sum
over its frequencies of abs(G)**2 / (H W), G the discrete Fourier transform of the
grayscale image, so the energies of all frequencies sum to the image's sum of squares
(Parseval), and eps is added inside the log.

In these coordinates the common camera changes act coordinate by coordinate: a
constant added to every pixel moves the means alone; scaling every channel about its
mean scales the standard deviations by the factor and every band energy by its square;
a blur multiplies the power of every frequency by the kernel's squared transfer
function, which is at most one and falls with the frequency.

Every representation but the identity is standardised per coordinate on a fit sample,
as the chamber datamodule standardises its readouts: centred on the median, divided by
the spread between the 0.05 and 0.95 quantiles, floored at 1e-6, and clipped at +-10.
"""

import functools
import math

import torch
from torch import Tensor

_QUANTILES = (0.05, 0.5, 0.95)
_SCALE_FLOOR = 1e-6
_CLIP = 10.0
_NYQUIST = 0.5
# Absorbs the rounding of the logs, so that a frequency on an edge falls in the band
# above it: with 8 bands on a 32 x 32 grid, the radii sqrt(2), 2, 2 sqrt(2), ... in
# units of 1/32 lie exactly on edges.
_EDGE_TOLERANCE = 1e-9


def identity(x: Tensor) -> Tensor:
    return x.reshape(len(x), -1)


def _bands(height: int, width: int, n_bands: int) -> Tensor:
    """Return the band of every frequency of the Fourier grid, -1 for none, [H, W] in
    the layout of torch.fft.fft2."""
    f_y = torch.fft.fftfreq(height, dtype=torch.float64)
    f_x = torch.fft.fftfreq(width, dtype=torch.float64)
    radius = (f_y[:, None] ** 2 + f_x[None, :] ** 2).sqrt()
    lowest = 1.0 / max(height, width)
    inside = (radius > 0) & (radius <= _NYQUIST)
    position = n_bands * (radius[inside] / lowest).log() / math.log(_NYQUIST / lowest)
    bands = torch.full(radius.shape, -1, dtype=torch.long)
    bands[inside] = (position + _EDGE_TOLERANCE).floor().long().clamp(0, n_bands - 1)
    return bands


def _band_energies(gray: Tensor, n_bands: int) -> Tensor:
    """Return the energy of every band of the images gray [n, H, W], [n, n_bands]."""
    height, width = gray.shape[-2:]
    power = torch.fft.fft2(gray).abs().square().flatten(1) / (height * width)
    bands = _bands(height, width, n_bands).flatten()
    inside = bands >= 0
    energies = power.new_zeros(len(gray), n_bands)
    return energies.index_add_(1, bands[inside], power[:, inside])


def colour_spectral(x: Tensor, n_bands: int = 8, eps: float = 1e-8) -> Tensor:
    """Return the channel means, the channel standard deviations and the log band
    energies of the images x [n, C, H, W], [n, 2 C + n_bands].

    eps keeps the log finite for a band without energy, as in a constant image. The
    default lies five orders of magnitude below the least band energy of normalised
    8-bit CIFAR-10 images, about 1e-3 even under a Gaussian blur of 6 pixels.
    """
    x = x.to(torch.float64)
    means, stds = x.mean(dim=(2, 3)), x.std(dim=(2, 3), correction=0)
    energies = _band_energies(x.mean(dim=1), n_bands)
    return torch.cat([means, stds, (energies + eps).log()], dim=1)


REPRESENTATIONS = {"identity": identity, "colour_spectral": colour_spectral}


def fit_standardisation(reference: Tensor) -> tuple[Tensor, Tensor]:
    """Return the median of every coordinate of reference [n, d] and its spread, the
    difference of the 0.95 and 0.05 quantiles floored at 1e-6, [d] each."""
    q = torch.tensor(_QUANTILES, dtype=reference.dtype)
    low, median, high = torch.quantile(reference, q, dim=0)
    # The floor keeps a coordinate that is constant over the bulk of the sample finite.
    return median, (high - low).clamp_min(_SCALE_FLOOR)


def standardise(x: Tensor, median: Tensor, spread: Tensor) -> Tensor:
    return ((x - median) / spread).clamp(-_CLIP, _CLIP)


def represent(
    name: str, params: dict | None, sets: list[Tensor], fit: Tensor
) -> list[Tensor]:
    """Return every set [n, ...] in the representation `name`, [n, d] each.

    The identity flattens the sets, changes nothing else and ignores fit. Every other
    representation, called with the keyword arguments params, is standardised on its
    image of fit, a sample of the same kind of inputs (the training set, say).
    """
    if name == "identity":
        return [identity(x) for x in sets]
    phi = functools.partial(REPRESENTATIONS[name], **(params or {}))
    median, spread = fit_standardisation(phi(fit))
    return [standardise(phi(x), median, spread) for x in sets]
