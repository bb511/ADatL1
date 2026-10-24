import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks.cap_bank import CAPBankCallback as TrainingCallback
from src.callbacks.components.cap import bank, representation
from src.evaluation.callbacks.cap_bank import CAPBankCallback as EvaluationCallback
from src.evaluation.callbacks.components.cap import bank as evaluation_bank
from src.evaluation.callbacks.components.cap import (
    representation as evaluation_representation,
)

MODULES = [representation, evaluation_representation]
ROOT = Path(__file__).resolve().parents[1]
BANK = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/bank.yaml"))
PAIRINGS = BANK["pairings"] | {"ot": BANK["pairings"]["ot"] | {"n_subsamples": 2}}
CONFIG = BANK | {"pairings": PAIRINGS}
SETS = ("normal", "pool", "reference_normal")
ARGS = ("ascore/full", *SETS)
LABEL = dict(zip(("dataset_1", "dataset_2", "reference"), SETS))


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _images(n=64, size=32, seed=0):
    # Gaussian fields with a 1/f amplitude spectrum, roughly that of natural images.
    f = torch.fft.fftfreq(size, dtype=torch.float64)
    radius = (f[:, None] ** 2 + f[None, :] ** 2).sqrt()
    amplitude = 1.0 / radius.clamp_min(1.0 / size)
    noise = _randn(n, 3, size, size, seed=seed)
    return torch.fft.ifft2(torch.fft.fft2(noise) * amplitude).real


def _blurred(x, sigma):
    # A periodic Gaussian blur of sigma pixels, applied through its transfer function.
    f = torch.fft.fftfreq(x.shape[-1], dtype=torch.float64)
    squared = f[:, None] ** 2 + f[None, :] ** 2
    transfer = torch.exp(-2.0 * math.pi**2 * sigma**2 * squared)
    return torch.fft.ifft2(torch.fft.fft2(x) * transfer).real


@pytest.mark.parametrize("module", MODULES)
def test_brightness_moves_the_means_alone(module):
    x = _images()
    before, after = module.colour_spectral(x), module.colour_spectral(x + 0.7)

    assert after.shape == (64, 14)
    torch.testing.assert_close(after[:, :3], before[:, :3] + 0.7)
    torch.testing.assert_close(after[:, 3:], before[:, 3:])


@pytest.mark.parametrize("module", MODULES)
def test_contrast_scales_the_stds_and_every_band_energy_by_one_factor(module):
    x = _images()
    mean = x.mean(dim=(2, 3), keepdim=True)
    before = module.colour_spectral(x)
    after = module.colour_spectral(mean + 1.8 * (x - mean))

    torch.testing.assert_close(after[:, :3], before[:, :3])
    torch.testing.assert_close(after[:, 3:6], 1.8 * before[:, 3:6])
    shift = after[:, 6:] - before[:, 6:]
    torch.testing.assert_close(shift, torch.full_like(shift, 2.0 * math.log(1.8)))


@pytest.mark.parametrize("module", MODULES)
def test_blur_scales_every_band_by_a_factor_falling_with_the_frequency(module):
    x = _images()
    before, after = module.colour_spectral(x), module.colour_spectral(_blurred(x, 1.0))

    torch.testing.assert_close(after[:, :3], before[:, :3])
    shift = after[:, 6:] - before[:, 6:]
    assert (shift < 0).all()
    assert (shift.diff(dim=1) < 0).all()


@pytest.mark.parametrize("module", MODULES)
def test_white_noise_raises_every_band_and_the_high_ones_most(module):
    x = _images(n=1024)
    noisy = x + 20.0 * _randn(*x.shape, seed=1)
    shift = module.colour_spectral(noisy) - module.colour_spectral(x)

    # Averaged over the images, since noise can lower one image's band by chance.
    mean_shift = shift[:, 6:].mean(dim=0)
    assert (mean_shift > 0).all()
    assert (mean_shift.diff() > 0).all()


# Frequencies (k_y, k_x) of a 32 x 32 grid, in cycles per image, and the band of their
# radius among 8, whose edges are 2 ** (b / 2) cycles per image; (16, 16) lies beyond
# the Nyquist radius.
WAVES = [(1, 0, 0), (1, 1, 1), (2, 0, 2), (2, 2, 3), (4, 0, 4), (16, 0, 7)]


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("k_y, k_x, band", [*WAVES, (16, 16, None)])
def test_a_wave_puts_all_its_energy_in_the_band_of_its_radius(module, k_y, k_x, band):
    i = torch.arange(32, dtype=torch.float64)
    wave = torch.cos(2.0 * math.pi * (k_y * i[:, None] + k_x * i[None, :]) / 32)
    energies = module.colour_spectral(wave.expand(1, 3, 32, 32), eps=0.0)[0, 6:].exp()

    expected = torch.zeros(8, dtype=torch.float64)
    if band is not None:
        expected[band] = wave.square().sum()
    torch.testing.assert_close(energies, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("module", MODULES)
def test_standardisation_centres_on_the_median_and_scales_by_the_spread(module):
    rows = [[float(i), 5.0] for i in range(101)]
    median, spread = module.fit_standardisation(torch.tensor(rows, dtype=torch.float64))
    x = torch.tensor([[50.0, 5.0], [95.0, 5.0], [2000.0, 4.0]], dtype=torch.float64)

    assert median.tolist() == [50.0, 5.0]
    assert spread.tolist() == pytest.approx([90.0, 1e-6])
    expected = torch.tensor([[0.0, 0.0], [0.5, 0.0], [10.0, -10.0]], dtype=x.dtype)
    torch.testing.assert_close(module.standardise(x, median, spread), expected)


@pytest.mark.parametrize("module", MODULES)
def test_identity_returns_the_flattened_inputs_unchanged(module):
    x = torch.randn(7, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    (flat,) = module.represent("identity", None, [x], None)

    assert flat.dtype == x.dtype and flat.shape == (7, 48)
    assert flat.numpy().tobytes() == x.numpy().tobytes()


@pytest.mark.parametrize("module", MODULES)
def test_represent_standardises_every_set_on_the_fit_sample(module):
    fit, x = _images(n=201, seed=1), _images(n=50, seed=2)
    params = {"n_bands": 6}
    out, on_fit = module.represent("colour_spectral", params, [x, fit], fit)

    phi = module.colour_spectral(fit, n_bands=6)
    median, spread = module.fit_standardisation(phi)
    expected = module.standardise(module.colour_spectral(x, n_bands=6), median, spread)
    assert out.shape == (50, 12) and torch.equal(out, expected)
    assert torch.equal(on_fit.median(dim=0).values, torch.zeros(12, dtype=out.dtype))


def _loaders(train=False):
    # 8 x 8 images as a datamodule serves them; the pool is three noisy copies of X1.
    normal = _images(n=60, size=8, seed=1)
    sets = {
        "normal": normal,
        "pool": normal.repeat(3, 1, 1, 1) + 0.05 * _randn(180, 3, 8, 8, seed=4),
        "reference_normal": _images(n=60, size=8, seed=2),
    }
    if train:
        sets["train"] = _images(n=150, size=8, seed=3)
    return {
        name: [(x, torch.zeros(len(x))) for x in v.float().split(25)]
        for name, v in sets.items()
    }


def _collected(loaders, name):
    return torch.cat([x for x, _ in loaders[name]])


def _flat(loaders, name):
    # The collection of the callbacks before representations: every batch flattened,
    # then concatenated.
    return torch.cat([x.reshape(len(x), -1) for x, _ in loaders[name]])


def _batch_ends(hook, trainer, loaders):
    for index, batches in enumerate(loaders.values()):
        for batch_idx, batch in enumerate(batches):
            outputs = {"ascore/full": torch.zeros(len(batch[0]))}
            hook(trainer, None, outputs, batch, batch_idx, index)


def _evaluation_couplings(callback, loaders, out_dir):
    trainer = SimpleNamespace(test_dataloaders=loaders, out_dir=out_dir)
    callback.on_test_epoch_start(trainer, None)
    _batch_ends(callback.on_test_batch_end, trainer, loaders)
    return callback._build_couplings(trainer)


def _training_couplings(callback, loaders):
    trainer = SimpleNamespace(val_dataloaders=loaders, sanity_checking=False)
    callback.on_fit_start(trainer, None)
    callback.on_validation_epoch_start(trainer, None)
    _batch_ends(callback.on_validation_batch_end, trainer, loaders)
    return callback._build_couplings()


def _equal(a, b) -> bool:
    a, b = asdict(a), asdict(b)
    same = [
        torch.equal(v, b[k]) if torch.is_tensor(v) else v == b[k] for k, v in a.items()
    ]
    return a.keys() == b.keys() and all(same)


def _records(directory: Path) -> dict[str, dict]:
    records = {
        path.name: json.loads(path.read_text()) for path in directory.glob("*.json")
    }
    return {name: record | {"build_seconds": None} for name, record in records.items()}


@pytest.mark.parametrize("kwargs", [{}, {"representation": "identity"}])
def test_identity_couplings_and_hashes_are_those_of_the_flattened_inputs(
    kwargs, tmp_path
):
    loaders = _loaders(train=True)
    callback = EvaluationCallback(*ARGS, bank_config=CONFIG, **kwargs)
    built = _evaluation_couplings(callback, loaders, tmp_path / "callback")
    flat = [_flat(loaders, name) for name in SETS]
    direct = evaluation_bank.build_couplings(
        *flat, PAIRINGS, tmp_path / "direct", LABEL
    )

    assert all(_equal(built[name], direct[name]) for name in PAIRINGS)
    files = [sorted(p.name for p in (tmp_path / "direct").iterdir())]
    files.append(sorted(p.name for p in (tmp_path / "callback" / "cap_bank").iterdir()))
    assert files[0] == files[1] and len(files[0]) == 6
    assert _records(tmp_path / "direct") == _records(tmp_path / "callback" / "cap_bank")


def test_training_identity_couplings_are_those_of_the_flattened_inputs():
    loaders = _loaders()
    built = _training_couplings(TrainingCallback(*ARGS, bank_config=CONFIG), loaders)
    direct = bank.build_couplings(*(_flat(loaders, name) for name in SETS), PAIRINGS)

    assert all(_equal(built[name], direct[name]) for name in PAIRINGS)


@pytest.mark.parametrize("train", [True, False])
def test_a_representation_is_fitted_on_the_training_set_when_served(train, tmp_path):
    loaders, params = _loaders(train=train), {"n_bands": 3}
    callback = EvaluationCallback(
        *ARGS,
        bank_config=CONFIG,
        representation="colour_spectral",
        representation_params=params,
    )
    built = _evaluation_couplings(callback, loaders, tmp_path)

    fit_set = "train" if train else "normal"
    sets = [_collected(loaders, name) for name in SETS]
    fit = _collected(loaders, fit_set)
    represented = evaluation_representation.represent(
        "colour_spectral", params, sets, fit
    )
    plain = evaluation_bank.build_couplings(*represented, PAIRINGS, tmp_path / "plain")
    assert all(_equal(built[name], plain[name]) for name in PAIRINGS)
    record = {"name": "colour_spectral", "params": params, "fit": fit_set}
    records = _records(tmp_path / "cap_bank")
    assert [r["label"] for r in records.values()] == [
        LABEL | {"representation": record}
    ] * 3
    # The representation enters the hash: the plain couplings sit under other names.
    assert not set(records) & set(_records(tmp_path / "plain"))


def test_training_callback_fits_a_representation_on_x1():
    loaders, params = _loaders(train=False), {"n_bands": 3}
    callback = TrainingCallback(
        *ARGS,
        bank_config=CONFIG,
        representation="colour_spectral",
        representation_params=params,
    )
    built = _training_couplings(callback, loaders)

    sets = [_collected(loaders, name) for name in SETS]
    represented = representation.represent("colour_spectral", params, sets, sets[0])
    direct = bank.build_couplings(*represented, PAIRINGS)
    assert all(_equal(built[name], direct[name]) for name in PAIRINGS)


def test_x2_blocks_reach_the_ot_and_nn_couplings(tmp_path):
    callback = EvaluationCallback(*ARGS, bank_config=CONFIG, x2_blocks=3)
    built = _evaluation_couplings(callback, _loaders(), tmp_path)

    assert built["ot"].identity_fraction == [1.0, 1.0, 1.0]
    assert sum(built["nn"].identity_fraction) == pytest.approx(1.0)
    record = json.loads(next((tmp_path / "cap_bank").glob("ot_*.json")).read_text())
    assert record["parameters"]["blocks"] == 3


@pytest.mark.parametrize("callback_cls", [TrainingCallback, EvaluationCallback])
def test_an_unknown_representation_raises(callback_cls):
    with pytest.raises(ValueError):
        callback_cls(*ARGS, bank_config=CONFIG, representation="pixels")
