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

The two L1 representations read the L1AD inputs [n, 39, 3] of the record's basis
configuration, robust-normalised as the datamodule serves them: 39 slots in the
record's stacking order (``L1_OBJECTS``: FET, 12 e/gammas, 10 jets, 4 muons, 12 taus),
each holding (Et, eta, phi), with FET's eta padded in every event. ``l1_et`` takes the
Et of every slot, [n, 39], the identity's columns 0, 3, ..., 114, padded slots zero as
the bank callbacks collect them. ``l1_summary`` reads its inputs with every padded
value NaN (``READS_MASK``) and returns, in float64 and GeV, the coordinates it is given
from ``SUMMARY_COORDINATES``: for each collection c of jets, egammas, muons and taus,
the multiplicity n_c (the slots of c that hold an object, so at most the slots kept),
the Et sum et_c and the leading Et lead_c (0 without an object); et_fet, the FET Et,
which stands in for MET in the basis configuration (0 where the record's object filter
removed it); et_total, the sum of the four et_c; and n_total, the sum of the four n_c.
Every Et is the record's robust normalisation undone, x scale + shift in hardware
counts, rounded to the integer count the record stores, times the L1 scale in GeV per
count.

Every representation but the identity and ``l1_et`` (``UNSTANDARDISED``) is
standardised per coordinate on a fit sample, as the chamber datamodule standardises its
readouts: centred on the median, divided by the spread between the 0.05 and 0.95
quantiles, floored at 1e-6, and clipped at +-10.
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
# (slots, features) of an L1AD input, and (first slot, slots) of every object along its
# 39 slots, in the record's stacking order (alphabetical by object); Et is feature 0.
_L1_SHAPE = (39, 3)
L1_OBJECTS = {
    "FET": (0, 1),
    "egammas": (1, 12),
    "jets": (13, 10),
    "muons": (23, 4),
    "taus": (27, 12),
}
L1_COLLECTIONS = ("jets", "egammas", "muons", "taus")
SUMMARY_COORDINATES = (
    "n_jets",
    "et_jets",
    "n_egammas",
    "et_egammas",
    "n_muons",
    "et_muons",
    "n_taus",
    "et_taus",
    "et_fet",
    "lead_jets",
    "lead_egammas",
    "lead_muons",
    "lead_taus",
    "et_total",
    "n_total",
)


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


def l1_et(x: Tensor) -> Tensor:
    return x.reshape(len(x), *_L1_SHAPE)[..., 0]


def _object_et(x: Tensor, et_norm: dict, et_gev: dict) -> dict[str, Tensor]:
    """Return the Et in GeV of the slots of every object, NaN where padded.

    x holds L1AD inputs [n, 39, 3]; every object's Et is [n, slots] in float64.
    """
    et = x.reshape(len(x), *_L1_SHAPE)[..., 0].to(torch.float64)
    objects = {}
    for name, (first, slots) in L1_OBJECTS.items():
        counts = et[:, first : first + slots] * et_norm[name]["scale"]
        # The record stores Et as integer counts below 4095, which its float32
        # normalised values carry to within 3e-4: rounding recovers them exactly.
        counts = (counts + et_norm[name]["shift"]).round()
        objects[name] = counts * et_gev[name]
    return objects


def _summary(objects: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return every coordinate of ``SUMMARY_COORDINATES`` [n] from the Et by object."""
    values = {"et_fet": objects["FET"][:, 0].nan_to_num(0.0)}
    for name in L1_COLLECTIONS:
        et = objects[name]
        values[f"n_{name}"] = (~et.isnan()).sum(dim=1).to(et.dtype)
        values[f"et_{name}"] = et.nansum(dim=1)
        values[f"lead_{name}"] = et.nan_to_num(0.0).amax(dim=1)
    values["et_total"] = sum(values[f"et_{name}"] for name in L1_COLLECTIONS)
    values["n_total"] = sum(values[f"n_{name}"] for name in L1_COLLECTIONS)
    return values


def l1_summary(
    x: Tensor,
    et_norm: dict[str, dict[str, float]],
    et_gev: dict[str, float],
    coordinates: tuple[str, ...] | list[str] = SUMMARY_COORDINATES,
) -> Tensor:
    """Return the coordinates of the L1AD inputs x [n, 39, 3], every padded value NaN,
    [n, len(coordinates)] in float64.

    et_norm holds every object's Et normalisation in hardware counts, {"shift": s,
    "scale": c}, as the record's ``<object>_norm_params.pkl`` holds it under "Et", and
    et_gev every object's L1 scale of Et in GeV per count (the record's ``l1_scales``).
    Raises ValueError where FET's eta, which no event carries, is not NaN: the padded
    slots then hold values, and the multiplicities would count them.
    """
    if not x.reshape(len(x), *_L1_SHAPE)[:, 0, 1].isnan().all():
        raise ValueError("l1_summary reads padded inputs as NaN; FET's eta is not.")
    values = _summary(_object_et(x, et_norm, et_gev))
    return torch.stack([values[name] for name in coordinates], dim=1)


REPRESENTATIONS = {
    "identity": identity,
    "colour_spectral": colour_spectral,
    "l1_et": l1_et,
    "l1_summary": l1_summary,
}
# l1_et is left unstandardised as the identity it selects from: its slots that nearly
# every event leaves padded have a zero spread, which the floor would turn into +-10
# for any real object.
UNSTANDARDISED = frozenset({"identity", "l1_et"})
# The representations that read the data mask, which the bank callbacks pass them as
# NaN in every padded value of their inputs.
READS_MASK = frozenset({"l1_summary"})


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
    """Return every set [n, ...] in the representation `name`, called with the keyword
    arguments params, [n, d] each.

    The identity flattens the sets and changes nothing else; it and l1_et
    (``UNSTANDARDISED``) ignore fit. Every other representation is standardised on its
    image of fit, a sample of the same kind of inputs (the training set, say).
    """
    phi = functools.partial(REPRESENTATIONS[name], **(params or {}))
    if name in UNSTANDARDISED:
        return [phi(x) for x in sets]
    median, spread = fit_standardisation(phi(fit))
    return [standardise(phi(x), median, spread) for x in sets]
