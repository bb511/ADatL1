import numpy as np
import pytest

from src.data.components.corruptions import (
    CORRUPTIONS,
    NOISES,
    SEVERITIES,
    _hsv_to_rgb,
    _jpeg_bytes,
    _rgb_to_hsv,
    _to_uint8,
    _to_unit,
    corrupt,
)

SEVERITY_LEVELS = [1, 2, 3, 4, 5]
DETERMINISTIC = [kind for kind in CORRUPTIONS if kind not in NOISES]


def _images(n: int = 6, seed: int = 0) -> np.ndarray:
    # Gradients under pixel noise: smooth like a photograph, with high frequencies
    # for the blur and the JPEG quantisation to remove.
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 1.0, 32)
    base = 30 + 150 * ramp[None, :, None, None] * rng.uniform(0.3, 1.0, (n, 1, 1, 3))
    noisy = base + 25 * ramp[None, None, :, None] + rng.normal(0, 15, (n, 32, 32, 3))
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _by_severity(kind: str, images: np.ndarray, statistic) -> list[float]:
    return [
        statistic(corrupt(images, kind, s, np.random.default_rng(0)))
        for s in SEVERITY_LEVELS
    ]


def _decreasing(values: list[float]) -> bool:
    return all(a > b for a, b in zip(values, values[1:]))


def _increasing(values: list[float]) -> bool:
    return _decreasing(values[::-1])


@pytest.mark.parametrize("kind", list(CORRUPTIONS))
def test_same_seed_gives_the_same_bytes(kind):
    images = _images()
    first, again = (
        corrupt(images, kind, 3, np.random.default_rng(7)) for _ in range(2)
    )
    other = corrupt(images, kind, 3, np.random.default_rng(8))
    assert first.tobytes() == again.tobytes()
    assert (other.tobytes() != first.tobytes()) == (kind in NOISES)


@pytest.mark.parametrize("severity", [1, 3, 5])
@pytest.mark.parametrize("kind", list(CORRUPTIONS))
def test_output_is_uint8_of_the_input_shape(kind, severity):
    images = _images()
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    out = corrupt(images, kind, severity, rng)
    assert out.dtype == np.uint8 and out.shape == images.shape
    assert not np.array_equal(out, images)
    assert (rng.bit_generator.state == state) == (kind not in NOISES)


@pytest.mark.parametrize("kind", DETERMINISTIC)
def test_one_image_is_corrupted_as_in_the_stack(kind):
    images = _images()
    stack = corrupt(images, kind, 4, np.random.default_rng(0))
    alone = corrupt(images[2:3], kind, 4, np.random.default_rng(0))
    assert np.array_equal(stack[2:3], alone)


@pytest.mark.parametrize("value", [0, 255])
def test_noise_clips_at_the_range_instead_of_wrapping(value):
    images = np.full((4, 32, 32, 3), value, dtype=np.uint8)
    out = corrupt(images, "gaussian_noise", 5, np.random.default_rng(0)).astype(int)
    # Noise SD 0.1 of full scale; a wrapped value would sit near the other end.
    assert np.abs(out - value).max() < 6 * 0.1 * 255
    assert np.mean(out == value) > 0.4


def test_cast_truncates_and_clips_as_the_published_script_does():
    x = np.array([254.6 / 255, 0.999 / 255, 1.2, -0.3])
    assert _to_uint8(x).tolist() == [254, 0, 255, 0]


def test_exact_grey_levels_survive_the_cast_unless_float_error_lowers_them():
    levels = np.arange(256, dtype=np.uint8)
    out = _to_uint8(_to_unit(levels)).astype(int)
    exact = levels / 255.0 * 255.0 >= levels
    assert np.array_equal(out[exact], levels[exact])
    assert np.all(levels[~exact] - out[~exact] == 1)


@pytest.mark.parametrize("kind", NOISES)
def test_noise_grows_with_severity(kind):
    images = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
    spread = _by_severity(kind, images, lambda out: float(np.std(out.astype(float))))
    assert _increasing(spread)


def test_blur_removes_more_high_frequency_energy_with_severity():
    def energy(out: np.ndarray) -> float:
        return float(np.sum(np.diff(out.astype(float), axis=1) ** 2))

    assert _decreasing(_by_severity("gaussian_blur", _images(), energy))


def test_contrast_lowers_the_spread_with_severity():
    def spread(out: np.ndarray) -> float:
        return float(out.astype(float).std(axis=(1, 2)).mean())

    assert _decreasing(_by_severity("contrast", _images(), spread))


def test_brightness_raises_the_value_with_severity():
    def value(out: np.ndarray) -> float:
        return float(out.max(axis=-1).mean())

    assert _increasing(_by_severity("brightness", _images(), value))


def test_jpeg_size_shrinks_with_severity():
    images = _images()
    qualities = [int(q) for q in SEVERITIES["jpeg_compression"]]
    sizes = [sum(len(_jpeg_bytes(image, q)) for image in images) for q in qualities]
    assert _decreasing(sizes)

    def error(out: np.ndarray) -> float:
        return float(np.abs(out.astype(float) - images).mean())

    assert _increasing(_by_severity("jpeg_compression", images, error))


def test_hsv_conversion_inverts_and_keeps_skimage_conventions():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, (5, 8, 8, 3))
    rgb[0, 0, 0] = [0.5, 0.5, 0.5]
    rgb[0, 0, 1] = [0.7, 0.7, 0.2]
    assert np.allclose(_hsv_to_rgb(_rgb_to_hsv(rgb)), rgb, atol=1e-12)
    pixels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0]])
    expected = [[0, 1, 1], [1 / 3, 1, 1], [2 / 3, 1, 1], [0, 0, 0.5], [0, 0, 0]]
    assert np.allclose(_rgb_to_hsv(pixels[None].astype(float))[0], expected)


def test_brightness_scales_each_pixel_by_its_new_value():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.05, 1, (4, 8, 8, 3))
    hsv = _rgb_to_hsv(rgb)
    value = hsv[..., 2].copy()
    hsv[..., 2] = np.clip(value + 0.15, 0, 1)
    ratio = (hsv[..., 2] / value)[..., None]
    assert np.allclose(_hsv_to_rgb(hsv), rgb * ratio, atol=1e-12)


def test_invalid_requests_raise():
    images = _images(n=1)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        corrupt(images, "contrast", 6, rng)
    with pytest.raises(ValueError):
        corrupt(images, "fog", 1, rng)
    with pytest.raises(ValueError):
        corrupt(images.astype(np.float32), "contrast", 1, rng)
    with pytest.raises(ValueError):
        corrupt(images[0], "contrast", 1, rng)
