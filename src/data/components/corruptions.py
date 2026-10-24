"""Six camera-like corruptions of CIFAR-10-C, for the second typical domain of CIFAR-10.

The functions and severity parameters are those of Hendrycks and Dietterich,
"Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
ICLR 2019, as their generation script ``ImageNet-C/create_c/make_cifar_c.py`` sets them
(github.com/hendrycks/robustness, commit ca36008 of 9 Jan 2019): sensor noise
(``gaussian_noise``, ``shot_noise``), focus (``gaussian_blur``), exposure and gain
(``brightness``, ``contrast``) and quantisation (``jpeg_compression``). Every function
maps uint8 images of shape (..., H, W, 3) to uint8 images of the same shape and treats
each image on its own.

Where this module departs from the script, no parameter changes:

- The noise is drawn from a ``numpy.random.Generator`` passed in, where the script draws
  from numpy's global state, so that a seed fixes the corrupted images.
- The script blurs with ``skimage.filters.gaussian`` (``multichannel=True``, mode
  ``nearest``, truncate 4), which calls ``scipy.ndimage.gaussian_filter`` with sigma 0
  on the channel axis; the call is made here directly. Its brightness converts with
  ``skimage.color.rgb2hsv`` and ``hsv2rgb``, ported below to numpy from scikit-image
  0.14.2, so that neither scikit-image nor OpenCV is needed.

The float output is cast to uint8 as the script casts it, with ``np.uint8``, which
truncates. The deterministic members (``gaussian_blur``, ``brightness``, ``contrast``,
``jpeg_compression``) therefore reproduce the published functions bit for bit, given
the same scipy and Pillow builds; for ``brightness`` this rests on the HSV port, which
follows scikit-image 0.14.2 line for line but has not been compared with it. The
truncation leaves the corrupted images on average half a grey level below their float
result, a property of the published benchmark kept here; JPEG is decoded to uint8
directly and never passes through the cast.
"""

import io
from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# The parameters of make_cifar_c.py for severities 1-5, quoted from its functions:
#   gaussian_noise    c = [0.04, 0.06, .08, .09, .10]   noise sd on the [0, 1] scale
#   shot_noise        c = [500, 250, 100, 75, 50]       Poisson counts at full scale
#   gaussian_blur     c = [.4, .6, 0.7, .8, 1]          sigma in pixels
#   brightness        c = [.05, .1, .15, .2, .3]        added to the HSV value
#   contrast          c = [.75, .5, .4, .3, 0.15]       factor on the deviation from
#                                                       the image's channel mean
#   jpeg_compression  c = [80, 65, 58, 50, 40]          PIL JPEG quality
SEVERITIES: dict[str, tuple[float, ...]] = {
    "gaussian_noise": (0.04, 0.06, 0.08, 0.09, 0.10),
    "shot_noise": (500, 250, 100, 75, 50),
    "gaussian_blur": (0.4, 0.6, 0.7, 0.8, 1.0),
    "brightness": (0.05, 0.1, 0.15, 0.2, 0.3),
    "contrast": (0.75, 0.5, 0.4, 0.3, 0.15),
    "jpeg_compression": (80, 65, 58, 50, 40),
}


def _parameter(kind: str, severity: int) -> float:
    if not 1 <= severity <= len(SEVERITIES[kind]):
        raise ValueError(f"Severity {severity} of {kind} is outside 1-5.")
    return SEVERITIES[kind][severity - 1]


def _to_unit(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float64) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    # make_cifar_c.py's np.uint8 cast, which truncates; rounding would change the bytes.
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """``skimage.color.rgb2hsv`` of scikit-image 0.14.2 over any leading axes."""
    out = np.empty_like(rgb)
    v = rgb.max(-1)
    delta = v - rgb.min(-1)
    # 0/0 where the three channels are equal; those entries are overwritten below.
    with np.errstate(invalid="ignore"):
        s = delta / v
        # Where two channels tie for the maximum, the later case overwrites the earlier.
        idx = rgb[..., 0] == v
        out[idx, 0] = (rgb[idx, 1] - rgb[idx, 2]) / delta[idx]
        idx = rgb[..., 1] == v
        out[idx, 0] = 2.0 + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]
        idx = rgb[..., 2] == v
        out[idx, 0] = 4.0 + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]
    s[delta == 0.0] = 0.0
    h = (out[..., 0] / 6.0) % 1.0
    h[delta == 0.0] = 0.0
    out[..., 0], out[..., 1], out[..., 2] = h, s, v
    out[np.isnan(out)] = 0
    return out


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """``skimage.color.hsv2rgb`` of scikit-image 0.14.2 over any leading axes."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hi = np.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    sectors = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    return np.choose(hi, [np.stack(sector, axis=-1) for sector in sectors])


def _jpeg_bytes(image: np.ndarray, quality: int) -> bytes:
    """Encode one (H, W, 3) uint8 image as PIL does at its default settings."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def gaussian_noise(
    images: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    x = _to_unit(images)
    sd = _parameter("gaussian_noise", severity)
    return _to_uint8(x + rng.normal(scale=sd, size=x.shape))


def shot_noise(
    images: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    counts = _parameter("shot_noise", severity)
    return _to_uint8(rng.poisson(_to_unit(images) * counts) / counts)


def gaussian_blur(images: np.ndarray, severity: int) -> np.ndarray:
    sigma = _parameter("gaussian_blur", severity)
    # Sigma 0 leaves an axis unfiltered: no smoothing across images or channels.
    sigmas = (0.0,) * (images.ndim - 3) + (sigma, sigma, 0.0)
    blurred = gaussian_filter(_to_unit(images), sigmas, mode="nearest", truncate=4.0)
    return _to_uint8(blurred)


def brightness(images: np.ndarray, severity: int) -> np.ndarray:
    hsv = _rgb_to_hsv(_to_unit(images))
    hsv[..., 2] = np.clip(hsv[..., 2] + _parameter("brightness", severity), 0.0, 1.0)
    return _to_uint8(_hsv_to_rgb(hsv))


def contrast(images: np.ndarray, severity: int) -> np.ndarray:
    x = _to_unit(images)
    means = x.mean(axis=(-3, -2), keepdims=True)
    return _to_uint8((x - means) * _parameter("contrast", severity) + means)


def jpeg_compression(images: np.ndarray, severity: int) -> np.ndarray:
    quality = int(_parameter("jpeg_compression", severity))
    flat = images.reshape(-1, *images.shape[-3:])
    decoded = [Image.open(io.BytesIO(_jpeg_bytes(image, quality))) for image in flat]
    return np.stack([np.asarray(image) for image in decoded]).reshape(images.shape)


CORRUPTIONS: dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "contrast": contrast,
    "jpeg_compression": jpeg_compression,
}
# The corruptions that draw from the generator; the others are deterministic.
NOISES = ("gaussian_noise", "shot_noise")


def corrupt(
    images: np.ndarray, kind: str, severity: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a stack of images under one corruption at one severity.

    Args:
        images: uint8 images of shape (n, H, W, 3), in the 0-255 range.
        kind: A key of ``CORRUPTIONS``.
        severity: CIFAR-10-C's severity, 1-5.
        rng: The generator of the noise kinds; the other kinds leave it untouched.

    Returns:
        uint8 images of the shape of ``images``, row i the corruption of image i.
    """
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected uint8 images (n, H, W, 3), got {images.dtype} {images.shape}."
        )
    if kind not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {kind!r}; known: {list(CORRUPTIONS)}.")
    if kind in NOISES:
        return CORRUPTIONS[kind](images, severity, rng)
    return CORRUPTIONS[kind](images, severity)
