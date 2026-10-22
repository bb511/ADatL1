import math
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, TensorDataset

import tests.test_score_dump as score_dump_tests
from src.algorithms.components.flow import RealNVP
from src.algorithms.realnvp import RealNVP as RealNVPModule
from src.callbacks.autouad import EAGCallback, NPDCallback, RTMCallback, eag, npd, rtm
from src.callbacks.cap import CAPCallback
from src.callbacks.components import selection
from src.callbacks.components.rate import AnomalyRate
from src.callbacks.components.utils import row_subset
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.entropy import LossEntropyCallback, loss_entropy
from src.callbacks.goix import (
    ExcessMassCallback,
    MassVolumeCallback,
    box_volume,
    excess_mass,
    mass_volume,
)
from src.callbacks.operational import OperationalScoreCallback
from src.callbacks.score_dump import aux_inputs, input_moments
from src.callbacks.sds import (
    SDS_SETS,
    SDSCallback,
    sds_error,
    sds_inputs,
    sds_pseudo_sets,
)
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback
from src.data.utils import unpack_batch
from src.evaluation.callbacks.autouad import EAG, NPD, RTM
from src.evaluation.callbacks.entropy import LossEntropy
from src.evaluation.callbacks.goix import ExcessMass, MassVolume
from src.evaluation.callbacks.sds import SDS
from tests.test_score_dump import (
    N_FEATURES,
    ToyAE,
    assert_identical,
    dump_callback,
    fit,
    load,
    toy_sets,
)

KEY = "ascore/full"
EPOCHS = 3
# The auxiliary sets and batch size of dump_callback, so that both score the same
# points in the same batches.
AUX = {"aux_seed": 7, "aux_size": 50, "batch_size": 16}
NAMES = ["npd", "rtm", "eag", "em", "mv", "entropy", "sds", "sds_val"]
CKPT = "epoch_0003"


class Recorder(Callback):
    def __init__(self):
        self.sanity, self.epochs = None, []

    def on_validation_end(self, trainer, pl_module):
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
        if trainer.sanity_checking:
            self.sanity = metrics
        else:
            self.epochs.append(metrics)


class NegativeLossAE(ToyAE):
    def model_step(self, batch):
        outputs = super().model_step(batch)
        return {**outputs, "loss/full": outputs["loss/full"] - 100.0}


def criteria():
    return [
        NPDCallback(**AUX),
        RTMCallback(),
        EAGCallback(),
        ExcessMassCallback(**AUX),
        MassVolumeCallback(**AUX),
        LossEntropyCallback(batch_size=16),
        SDSCallback(batch_size=16),
        SDSCallback(name="sds_val", source="normal", batch_size=16),
        OperationalScoreCallback(),
    ]


def score(module, x, key=KEY):
    module.eval()
    with torch.no_grad():
        batches = [(b, torch.zeros(len(b), dtype=torch.long)) for b in x.split(16)]
        return torch.cat([module.model_step(batch)[key] for batch in batches])


def expected_sds(module, normal, pseudo_sets, target_rate=None):
    target_rate = module.hparams.target_rate if target_rate is None else target_rate
    rate = AnomalyRate(target_rate, None)
    rate.set_threshold(normal)
    targets, outliers = (
        score(module, torch.from_numpy(p).float()) for p in pseudo_sets
    )
    cut = rate.threshold.item(), rate.tie_weight.item()
    return sds_error(targets, outliers, *cut)


def expected_values(dump_dir):
    meta = load(dump_dir / "meta.pt")
    volume = box_volume(meta["aux_lower"], meta["aux_upper"])
    sets = toy_sets()
    pseudo = {
        "sds": sds_pseudo_sets(sets["train"][0].double().numpy()),
        "sds_val": sds_pseudo_sets(sets["val"]["normal"][0].double().numpy()),
    }
    rows = []
    for epoch in range(EPOCHS):
        record = load(dump_dir / f"epoch_{epoch:04d}.pt")
        module = ToyAE()
        module.load_state_dict(record["state_dict"])
        normal = record["val"]["normal"][KEY]
        aux = {name: scores[KEY] for name, scores in record["aux"].items()}
        operational = [module.compute_operational_ascore(b) for b in normal.split(32)]
        rows.append(
            {
                "npd": npd(normal, aux["autouad"]),
                "rtm": rtm(normal),
                "eag": eag(normal),
                "em": excess_mass(normal, aux["uniform"], volume)[0],
                "mv": mass_volume(normal, aux["uniform"], volume),
                "entropy": loss_entropy(record["train"]["train"]["loss/full"]),
                "sds": expected_sds(module, normal, pseudo["sds"]),
                "sds_val": expected_sds(module, normal, pseudo["sds_val"]),
                "ascore_operational": float(np.mean(operational)),
            }
        )
    return rows


def approx(value):
    # The logged values are float32.
    return pytest.approx(value, rel=1e-6, abs=1e-9, nan_ok=True)


def test_callbacks_log_the_criteria_and_their_ema(tmp_path):
    recorder = Recorder()
    callbacks = [dump_callback(tmp_path / "scores"), *criteria(), recorder]
    fit(tmp_path, callbacks, epochs=EPOCHS)
    rows = expected_values(tmp_path / "scores")

    assert not [key for key in recorder.sanity if key.startswith("val/summary/")]
    assert len(recorder.epochs) == EPOCHS
    for name in [*NAMES, "ascore_operational"]:
        raw = [row[name] for row in rows]
        for epoch, logged in enumerate(recorder.epochs):
            key = f"val/summary/{name}"
            assert logged[f"{key}_ema"] == approx(selection.ema(raw)[epoch])
            if name == "ascore_operational":
                assert logged["val/normal/ascore_operational"] == approx(raw[epoch])
                assert key not in logged
            else:
                assert logged[key] == approx(raw[epoch])


def test_sds_cuts_at_its_target_rate(tmp_path):
    recorder = Recorder()
    callbacks = [
        dump_callback(tmp_path / "scores"),
        SDSCallback(batch_size=16, target_rate=0.3),
        SDSCallback(name="sds_val", source="normal", batch_size=16, target_rate=0.3),
        recorder,
    ]
    fit(tmp_path, callbacks, epochs=EPOCHS)
    sets = toy_sets()
    pseudo = {
        "sds_0_3": sds_pseudo_sets(sets["train"][0].double().numpy()),
        "sds_val_0_3": sds_pseudo_sets(sets["val"]["normal"][0].double().numpy()),
    }
    for epoch, logged in enumerate(recorder.epochs):
        record = load(tmp_path / "scores" / f"epoch_{epoch:04d}.pt")
        module = ToyAE()
        module.load_state_dict(record["state_dict"])
        normal = record["val"]["normal"][KEY]
        for name, pseudo_sets in pseudo.items():
            value = expected_sds(module, normal, pseudo_sets, target_rate=0.3)
            assert logged[f"val/summary/{name}"] == approx(value)
        assert "val/summary/sds" not in logged


def test_callbacks_leave_the_training_trajectory_unchanged(tmp_path):
    with_criteria, _ = fit(tmp_path, criteria(), epochs=EPOCHS)
    without, _ = fit(tmp_path, [], epochs=EPOCHS)
    assert_identical(with_criteria.state_dict(), without.state_dict())


def test_loss_entropy_reads_a_seeded_subset(tmp_path):
    recorder = Recorder()
    entropy = LossEntropyCallback(max_rows=40, row_seed=3, batch_size=16)
    fit(tmp_path, [dump_callback(tmp_path / "scores"), entropy, recorder], epochs=1)
    rows = row_subset(96, 40, 3)
    assert len(rows) == 40 and rows.tolist() == sorted(set(rows.tolist()))
    assert row_subset(96, 96, 3) is None and row_subset(96, None, 3) is None
    losses = load(tmp_path / "scores" / "epoch_0000.pt")["train"]["train"]["loss/full"]
    expected = loss_entropy(losses[rows])
    assert recorder.epochs[0]["val/summary/entropy"] == approx(expected)


def test_sds_inputs_take_distinct_rows_in_the_input_shape():
    x = torch.randn(40, 2, 3, generator=torch.Generator().manual_seed(1))
    sets = sds_inputs(torch.cat([x, x[:5]]), None, 0)
    rows = torch.unique(x.reshape(40, -1), dim=0).double().numpy()
    for points, expected in zip(sets.values(), sds_pseudo_sets(rows)):
        assert points.shape == (len(expected), 2, 3) and points.dtype == x.dtype
        flat = points.reshape(len(expected), -1)
        assert torch.equal(flat, torch.from_numpy(expected).float())
    assert len(sds_inputs(x, 20, 0)["sds_targets"]) <= 20


def test_undefined_criteria_give_nan(tmp_path, monkeypatch):
    recorders = {"entropy": Recorder(), "goix": Recorder()}
    callbacks = [LossEntropyCallback(), recorders["entropy"]]
    with pytest.warns(UserWarning, match="positive losses"):
        fit(tmp_path, callbacks, module=NegativeLossAE())
    # 512 features of range about 4.5 make a box volume beyond the float64 range.
    monkeypatch.setattr(score_dump_tests, "N_FEATURES", 512)
    callbacks = [ExcessMassCallback(**AUX), MassVolumeCallback(**AUX)]
    with pytest.warns(UserWarning, match="volume"):
        fit(tmp_path, [*callbacks, recorders["goix"]])
    names = {"entropy": ["entropy"], "goix": ["em", "mv"]}
    for group, recorder in recorders.items():
        assert len(recorder.epochs) == 2
        for logged in recorder.epochs:
            for name in names[group]:
                assert math.isnan(logged[f"val/summary/{name}"])
                assert math.isnan(logged[f"val/summary/{name}_ema"])


def test_existing_callbacks_skip_the_sanity_pass(tmp_path):
    pair = {"dataset_1": "normal", "dataset_2": "reference_normal"}
    cap = {"pairing_type": "cdf", "cap_metric_config": {}}
    callbacks = [
        ThresholdDriftCallback(output_name=KEY),
        WassersteinCallback(output_name=KEY, **pair),
        CAPCallback(output_name=KEY, **pair, **cap),
        PosteriorConsistencyCallback(output_name=KEY, **pair, **cap),
        AnomalyEfficiencyCallback(output_name=KEY),
    ]
    baseline, guarded = Recorder(), Recorder()
    fit(tmp_path, [baseline], epochs=1)
    fit(tmp_path, [*callbacks, guarded], epochs=1)
    assert guarded.sanity.keys() == baseline.sanity.keys()
    added = guarded.epochs[0].keys() - baseline.epochs[0].keys()
    for key in ("operational_drift_ema", "w1dist_ema", "cap_ema", "consistency_ema"):
        assert any(key in logged for logged in added)
    assert "val/summary/eff_cvar25_ema_operational" in added


def mirror_inputs():
    gen = torch.Generator().manual_seed(3)
    sizes = {"normal": 70, "autouad": 50, "uniform": 50, "train": 96}
    inputs = {
        name: torch.randn(n, N_FEATURES, generator=gen) for name, n in sizes.items()
    }
    return {**inputs, **sds_inputs(inputs["train"], None, 0)}


def evaluate(root, module, inputs, callbacks):
    module._ckpt_path = str(root / f"{CKPT}.pt")
    labels = {name: torch.zeros(len(x), dtype=torch.long) for name, x in inputs.items()}
    loaders = {
        name: DataLoader(TensorDataset(x, labels[name]), batch_size=16)
        for name, x in inputs.items()
    }
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=list(callbacks),
        default_root_dir=root,
    )
    trainer.test(module, dataloaders=loaders, verbose=False)


def toy_with_box(inputs, lower=None, upper=None, module=None):
    module = ToyAE() if module is None else module
    train = inputs["train"]
    module.aux_lower = train.min(dim=0).values if lower is None else lower
    module.aux_upper = train.max(dim=0).values if upper is None else upper
    return module


def test_mirrors_return_the_criteria(tmp_path):
    inputs = mirror_inputs()
    module = toy_with_box(inputs)
    scores = {name: score(module, x) for name, x in inputs.items()}
    cut = scores["normal"].quantile(0.9), torch.tensor(0.5, dtype=torch.float64)
    module.register_buffer("thres_operational", cut[0])
    module.register_buffer("tie_operational", cut[1])
    volume = box_volume(module.aux_lower, module.aux_upper)
    normal, uniform = scores["normal"], scores["uniform"]
    expected = {
        NPD(): npd(normal, scores["autouad"]),
        RTM(): rtm(normal),
        EAG(): eag(normal),
        ExcessMass(): excess_mass(normal, uniform, volume)[0],
        MassVolume(): mass_volume(normal, uniform, volume),
        LossEntropy(): loss_entropy(score(module, inputs["train"], "loss/full")),
        SDS(): sds_error(
            scores["sds_targets"], scores["sds_outliers"], cut[0].item(), 0.5
        ),
    }
    evaluate(tmp_path, module, inputs, expected)
    for callback, value in expected.items():
        assert math.isfinite(value)
        assert callback.summary == {CKPT: pytest.approx(value, rel=1e-12)}
        assert callback.get_optimized_metric() == (CKPT, callback.summary[CKPT])


def test_sds_mirror_cuts_at_its_target_rate(tmp_path):
    inputs = mirror_inputs()
    # No operational buffers: the cut at a target rate must not read them.
    module = ToyAE()
    scores = {name: score(module, x) for name, x in inputs.items()}
    rate = AnomalyRate(0.3, None)
    rate.set_threshold(scores["normal"])
    cut = rate.threshold.item(), rate.tie_weight.item()
    expected = sds_error(scores["sds_targets"], scores["sds_outliers"], *cut)
    callback = SDS(target_rate=0.3)
    evaluate(tmp_path, module, inputs, [callback])
    assert callback.name == "sds_0_3"
    assert callback.summary == {CKPT: pytest.approx(expected, rel=1e-12)}


def test_mirror_needs_its_loaders(tmp_path):
    inputs = mirror_inputs()
    del inputs["autouad"]
    with pytest.raises(ValueError, match="autouad"):
        evaluate(tmp_path, ToyAE(), inputs, [NPD()])
    del inputs["normal"]
    with pytest.raises(ValueError, match="normal"):
        evaluate(tmp_path, ToyAE(), inputs, [SDS(target_rate=0.3)])


def test_mirrors_give_nan_for_undefined_criteria(tmp_path):
    inputs = mirror_inputs()
    box = torch.zeros(512), torch.full((512,), 1e3)
    module = toy_with_box(inputs, *box, module=NegativeLossAE())
    callbacks = [ExcessMass(), MassVolume(), LossEntropy()]
    with pytest.warns(UserWarning):
        evaluate(tmp_path, module, inputs, callbacks)
    for callback in callbacks:
        assert math.isnan(callback.summary[CKPT])


def physics_rows(n: int = 96, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    # Four objects of three features in the layout of the L1AD data: the first never
    # carries its second feature, as the FET carries no eta, and padding is +0.
    gen = torch.Generator().manual_seed(seed)
    mask = (torch.rand(n, 4, 1, generator=gen) < 0.6).expand(n, 4, 3).clone()
    mask[:, 0, 1] = False
    x = torch.where(mask, torch.randn(n, 4, 3, generator=gen), torch.zeros(n, 4, 3))
    return x, mask


def fit_start(callback, *tensors) -> dict[str, DataLoader]:
    labels = torch.zeros(len(tensors[0]), dtype=torch.long)
    data_train = TensorDataset(*tensors, labels)
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_train=data_train))
    callback.on_fit_start(trainer, None)
    return callback.loaders


def maskless_loader(x, batch_size):
    # placeholder_loader as at a10184b.
    labels = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, labels), batch_size=batch_size)


def maskless_sds_inputs(x, max_rows, row_seed):
    # sds_inputs as at a10184b.
    rows = torch.unique(x.reshape(len(x), -1), dim=0)
    subset = row_subset(len(rows), max_rows, row_seed)
    rows = rows if subset is None else rows[subset]
    sets = sds_pseudo_sets(rows.double().numpy())
    return {
        name: torch.from_numpy(points).to(x.dtype).reshape(-1, *x.shape[1:])
        for name, points in zip(SDS_SETS, sets)
    }


def assert_same_loaders(loaders, expected):
    assert list(loaders) == list(expected)
    for name, loader in loaders.items():
        tensors, reference = loader.dataset.tensors, expected[name].dataset.tensors
        assert loader.batch_size == expected[name].batch_size
        assert len(tensors) == len(reference)
        for a, b in zip(tensors, reference):
            assert a.dtype == b.dtype and torch.equal(a, b)


def autouad_points(x):
    seeds = {"aux_seed": AUX["aux_seed"], "aux_size": AUX["aux_size"]}
    return aux_inputs({**input_moments(x), **seeds})["autouad"]


def test_maskless_made_up_loaders_are_unchanged():
    x = toy_sets()["train"][0]
    npd_loaders = fit_start(NPDCallback(**AUX), x)
    expected = {"autouad": maskless_loader(autouad_points(x), AUX["batch_size"])}
    assert_same_loaders(npd_loaders, expected)
    for max_rows in (None, 40):
        sds_loaders = fit_start(SDSCallback(max_rows=max_rows, batch_size=16), x)
        inputs = maskless_sds_inputs(x, max_rows, 0)
        expected = {name: maskless_loader(p, 16) for name, p in inputs.items()}
        assert_same_loaders(sds_loaders, expected)


def test_sds_pseudo_points_take_the_mask_of_their_source_row():
    x, mask = physics_rows()
    x, mask = torch.cat([x, x[:5]]), torch.cat([mask, mask[:5]])
    inputs = sds_inputs(x, None, 0, mask)
    flat = x.reshape(len(x), -1)
    rows = torch.unique(flat, dim=0)
    _, _, *sources = sds_pseudo_sets(rows.double().numpy(), return_rows=True)
    maskless = sds_inputs(x, None, 0)
    for name, source in zip(SDS_SETS, sources):
        first = [int((flat == rows[i]).all(dim=1).nonzero()[0]) for i in source]
        assert torch.equal(inputs[f"{name}_mask"], mask[first])
        assert torch.equal(inputs[name], maskless[name])
    subset = sds_inputs(x, 40, 0, mask)
    for name in SDS_SETS:
        assert subset[f"{name}_mask"].shape == subset[name].shape


def test_made_up_loaders_take_the_l1ad_layout_with_zero_padding():
    x, mask = physics_rows()
    l1bit = torch.ones(len(x), dtype=torch.bool)
    loaders = {
        **fit_start(NPDCallback(**AUX), x, mask, l1bit),
        **fit_start(SDSCallback(batch_size=16), x, mask, l1bit),
    }
    train_masks = {tuple(m.flatten().tolist()) for m in mask}
    for loader in loaders.values():
        assert all(len(batch) == 4 for batch in loader)
        batches = [unpack_batch(batch) for batch in loader]
        points = torch.cat([b.x for b in batches])
        masks = torch.cat([b.mask for b in batches])
        assert {tuple(m.flatten().tolist()) for m in masks} <= train_masks
        assert not points[~masks].any() and not points[~masks].signbit().any()
        assert not torch.cat([b.l1bit for b in batches]).any()
    autouad = loaders["autouad"].dataset.tensors
    present = autouad[1]
    assert torch.equal(autouad[0][present], autouad_points(x)[present])


def test_sds_on_validation_normal_takes_its_masks(monkeypatch):
    monkeypatch.setattr(SDSCallback, "_compute_sds", lambda self, trainer, m: 0.0)
    x, mask = physics_rows()
    module = SimpleNamespace(log_dict=lambda *args, **kwargs: None)
    trainer = SimpleNamespace(val_dataloaders={"normal": None}, sanity_checking=False)
    for masked in (False, True):
        callback = SDSCallback(source="normal", batch_size=16)
        callback.on_fit_start(trainer, module)
        callback.on_validation_epoch_start(trainer, module)
        for i, (xb, mb) in enumerate(zip(x.split(32), mask.split(32))):
            y = torch.zeros(len(xb))
            batch = (xb, mb, y.bool(), y) if masked else (xb, y)
            callback.on_validation_batch_end(trainer, module, {KEY: y}, batch, i)
        callback.on_validation_epoch_end(trainer, module)
        inputs = sds_inputs(x, 10000, 0, mask if masked else None)
        for name, loader in callback.loaders.items():
            tensors, expected = loader.dataset.tensors, inputs[name]
            assert len(tensors) == (4 if masked else 2)
            if masked:
                rows_mask = inputs[f"{name}_mask"]
                expected = torch.where(rows_mask, expected, torch.zeros_like(expected))
                assert torch.equal(tensors[1], rows_mask)
            assert torch.equal(tensors[0], expected)


def test_mask_aware_realnvp_scores_the_made_up_loaders():
    x, mask = physics_rows()
    l1bit = torch.zeros(len(x), dtype=torch.bool)
    loaders = {
        **fit_start(NPDCallback(**AUX), x, mask, l1bit),
        **fit_start(SDSCallback(batch_size=16), x, mask, l1bit),
    }
    torch.manual_seed(0)
    flow = RealNVP(input_dim=12, n_flows=4, hidden_dim=16, n_hidden_layers=2)
    module = RealNVPModule(flow=flow, target_rate=0.01, mask_aware=True).eval()
    maskless = fit_start(NPDCallback(**AUX), x)["autouad"]
    with torch.no_grad():
        for loader in loaders.values():
            for batch in loader:
                assert torch.isfinite(module.model_step(batch)["ascore/full"]).all()
        with pytest.raises(ValueError, match="data mask"):
            module.model_step(next(iter(maskless)))
