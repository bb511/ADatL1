import json
import math
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
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


# The Et normalisation of every L1AD object in hardware counts and its L1 scale in GeV
# per count, as revision 1ac2b91f of the record fits (robust) and declares them.
ET_NORM = {
    "FET": {"shift": 29.0, "scale": 63.0},
    "egammas": {"shift": 6.0, "scale": 9.0},
    "jets": {"shift": 46.0, "scale": 53.0},
    "muons": {"shift": 6.0, "scale": 13.0},
    "taus": {"shift": 10.0, "scale": 19.0},
}
ET_GEV = dict.fromkeys(ET_NORM, 0.5)
L1_PARAMS = {"et_norm": ET_NORM, "et_gev": ET_GEV}
# Two events by object, each list an event's Et counts in slot order: the jet of 46,
# the e/gamma of 6 and the tau of 10 sit at their shift and are normalised to 0.
HAND = {
    "FET": [[120], []],
    "jets": [[80, 46, 30], []],
    "egammas": [[6], []],
    "taus": [[10, 12], []],
}
HAND_SUMMARY = [[3, 78, 1, 3, 0, 0, 2, 11, 60, 40, 3, 0, 6, 92, 6], [0] * 15]
RECORD_DIR = os.environ.get("ADL1T_L1AD_RECORD_DIR")
MLREADY = "data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust"
ZEROBIAS = "data_2025E+G/processed/eminimalTauFET_pdefaultTauFET/zerobias"
RECORD_ROWS = 64


def _l1_inputs(counts, seed=0):
    # L1AD inputs normalised as the record serves them, with their mask, from every
    # event's Et counts by object; eta and phi are noise, FET's eta always padded.
    n = len(next(iter(counts.values())))
    x = _randn(n, 39, 3, seed=seed).float()
    mask = torch.zeros(n, 39, 3, dtype=torch.bool)
    for name, events in counts.items():
        first, norm = representation.L1_OBJECTS[name][0], ET_NORM[name]
        for i, objects in enumerate(events):
            et = torch.tensor(objects, dtype=torch.float64)
            x[i, first : first + len(objects), 0] = (et - norm["shift"]) / norm["scale"]
            mask[i, first : first + len(objects)] = True
    mask[:, 0, 1] = False
    return x, mask


def _random_counts(n, generator):
    return {
        name: [
            torch.randint(0, 200, (int(k),), generator=generator).tolist()
            for k in torch.randint(0, slots + 1, (n,), generator=generator)
        ]
        for name, (_, slots) in representation.L1_OBJECTS.items()
    }


@pytest.mark.parametrize("module", MODULES)
def test_l1_et_keeps_the_et_columns_of_the_identity_unstandardised(module):
    x, fit = _randn(6, 39, 3).float(), _randn(50, 39, 3, seed=1).float()
    (et,) = module.represent("l1_et", None, [x], fit)

    assert et.shape == (6, 39)
    assert torch.equal(et, module.identity(x)[:, 0::3])


@pytest.mark.parametrize("module", MODULES)
def test_l1_summary_counts_the_masked_objects_and_sums_their_et_in_gev(module):
    x, mask = _l1_inputs(HAND)
    summary = module.l1_summary(torch.where(mask, x, math.nan), ET_NORM, ET_GEV)

    assert (x[0, 13:16, 0] == 0).any()
    assert summary.dtype == torch.float64
    assert summary.tolist() == HAND_SUMMARY


@pytest.mark.parametrize("module", MODULES)
def test_l1_summary_returns_the_coordinates_it_is_given_in_their_order(module):
    x, mask = _l1_inputs(_random_counts(30, torch.Generator().manual_seed(2)))
    x = torch.where(mask, x, math.nan)
    names = ["n_total", "et_fet", "lead_muons"]
    part = module.l1_summary(x, ET_NORM, ET_GEV, coordinates=names)

    full = module.l1_summary(x, ET_NORM, ET_GEV)
    columns = [module.SUMMARY_COORDINATES.index(name) for name in names]
    assert full.shape == (30, 15) and torch.equal(part, full[:, columns])


@pytest.mark.parametrize("module", MODULES)
def test_l1_summary_refuses_inputs_whose_padding_is_not_nan(module):
    x, mask = _l1_inputs(HAND)
    with pytest.raises(ValueError, match="NaN"):
        module.l1_summary(torch.where(mask, x, 0.0), ET_NORM, ET_GEV)


def _l1_loaders():
    # Random L1AD events in (x, mask, l1bit, y) batches, as the datamodule serves them.
    generator, loaders = torch.Generator().manual_seed(5), {}
    for index, (name, n) in enumerate(zip(SETS, (60, 120, 60))):
        x, mask = _l1_inputs(_random_counts(n, generator), seed=index)
        loaders[name] = [
            (xb, mb, torch.ones(len(xb), dtype=torch.bool), torch.zeros(len(xb)))
            for xb, mb in zip(x.split(25), mask.split(25))
        ]
    return loaders


def _padded(loaders, name, value):
    return torch.cat([torch.where(m, x, value) for x, m, _, _ in loaders[name]])


@pytest.mark.parametrize("name", ["l1_summary", "l1_et"])
def test_the_bank_callbacks_pass_the_l1_representations_their_padding(name, tmp_path):
    loaders, plain = _l1_loaders(), L1_PARAMS if name == "l1_summary" else {}
    # As hydra passes them: a config, whose nested nodes the cache's JSON cannot hold.
    kwargs = {"representation": name, "representation_params": OmegaConf.create(plain)}
    evaluation = EvaluationCallback(*ARGS, bank_config=CONFIG, **kwargs)
    built = _evaluation_couplings(evaluation, loaders, tmp_path)
    training = TrainingCallback(*ARGS, bank_config=CONFIG, **kwargs)
    trained = _training_couplings(training, loaders)

    padded = math.nan if name == "l1_summary" else 0.0
    sets = [_padded(loaders, s, padded) for s in SETS]
    represented = evaluation_representation.represent(name, plain, sets, sets[0])
    direct = evaluation_bank.build_couplings(*represented, PAIRINGS)
    assert all(_equal(built[p], direct[p]) for p in PAIRINGS)
    assert all(_equal(trained[p], direct[p]) for p in PAIRINGS)
    fit = "normal" if name == "l1_summary" else None
    record = {"name": name, "params": plain, "fit": fit}
    labels = [r["label"] for r in _records(tmp_path / "cap_bank").values()]
    assert labels == [LABEL | {"representation": record}] * 3


def _record_rows(zerobias: Path, n: int) -> np.ndarray:
    # The first n rows of the validation split in the merged processed zero bias, in
    # the order of the record's loader/common.split_rows.
    paths = [d / "splits.parquet" for d in sorted(zerobias.iterdir())]
    index = pa.concat_tables([pq.read_table(path) for path in paths])
    valid = pc.equal(index["split"], "valid").to_numpy(zero_copy_only=False)
    order = index["order"].to_numpy()
    rows = np.flatnonzero(valid & (order >= 0))
    return rows[np.argsort(order[rows], kind="stable")][:n]


def _record_summary(zerobias: Path, rows: np.ndarray) -> list[list[float]]:
    # The summary from the processed Et counts, which the torch stage clips per object.
    dirs, gev = sorted(zerobias.iterdir()), {}
    for name, (_, slots) in representation.L1_OBJECTS.items():
        dataset = pds.dataset([d / f"{name}.parquet" for d in dirs], format="parquet")
        events = dataset.take(pa.array(rows)).column("Et").to_pylist()
        gev[name] = [[ET_GEV[name] * et for et in event[:slots]] for event in events]
    summaries, collections = [], representation.L1_COLLECTIONS
    for i in range(len(rows)):
        values = {"et_fet": gev["FET"][i][0] if gev["FET"][i] else 0.0}
        for name in collections:
            et = gev[name][i]
            values |= {f"n_{name}": float(len(et)), f"et_{name}": sum(et)}
            values[f"lead_{name}"] = max(et, default=0.0)
        values["et_total"] = sum(values[f"et_{n}"] for n in collections)
        values["n_total"] = sum(values[f"n_{n}"] for n in collections)
        summaries.append([values[c] for c in representation.SUMMARY_COORDINATES])
    return summaries


@pytest.fixture(scope="module")
def record_summary():
    zerobias = Path(RECORD_DIR) / ZEROBIAS
    return _record_summary(zerobias, _record_rows(zerobias, RECORD_ROWS))


@pytest.mark.skipif(RECORD_DIR is None, reason="ADL1T_L1AD_RECORD_DIR names no record.")
@pytest.mark.parametrize("module", MODULES)
def test_l1_summary_reproduces_the_record_unnormalised_et(module, record_summary):
    mlready = Path(RECORD_DIR) / MLREADY
    et_norm = {
        name: pickle.loads((mlready / f"{name}_norm_params.pkl").read_bytes())["Et"]
        for name in representation.L1_OBJECTS
    }
    x, mask = (
        torch.load(mlready / "valid" / f"torch_{k}.pt", mmap=True)[:RECORD_ROWS]
        for k in ("cache", "mask")
    )
    summary = module.l1_summary(torch.where(mask, x, math.nan), et_norm, ET_GEV)

    assert et_norm == ET_NORM
    assert summary.tolist() == record_summary
