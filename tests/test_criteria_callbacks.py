import math

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, TensorDataset

import tests.test_score_dump as score_dump_tests
from src.callbacks.autouad import EAGCallback, NPDCallback, RTMCallback
from src.callbacks.cap import CAPCallback
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.criterion import row_subset
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.entropy import LossEntropyCallback
from src.callbacks.goix import ExcessMassCallback, MassVolumeCallback
from src.callbacks.metrics import label_free as lfc
from src.callbacks.metrics import selection
from src.callbacks.metrics.rate import AnomalyRate
from src.callbacks.operational import OperationalScoreCallback
from src.callbacks.sds import SDSCallback, sds_inputs
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback
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


def expected_sds(module, normal, pseudo_sets):
    rate = AnomalyRate(module.hparams.target_rate, None)
    rate.set_threshold(normal)
    targets, outliers = (
        score(module, torch.from_numpy(p).float()) for p in pseudo_sets
    )
    cut = rate.threshold.item(), rate.tie_weight.item()
    return lfc.sds_error(targets, outliers, *cut)


def expected_values(dump_dir):
    meta = load(dump_dir / "meta.pt")
    volume = lfc.box_volume(meta["aux_lower"], meta["aux_upper"])
    sets = toy_sets()
    pseudo = {
        "sds": lfc.sds_pseudo_sets(sets["train"][0].double().numpy()),
        "sds_val": lfc.sds_pseudo_sets(sets["val"]["normal"][0].double().numpy()),
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
                "npd": lfc.npd(normal, aux["autouad"]),
                "rtm": lfc.rtm(normal),
                "eag": lfc.eag(normal),
                "em": lfc.excess_mass(normal, aux["uniform"], volume)[0],
                "mv": lfc.mass_volume(normal, aux["uniform"], volume),
                "entropy": lfc.loss_entropy(record["train"]["train"]["loss/full"]),
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
    expected = lfc.loss_entropy(losses[rows])
    assert recorder.epochs[0]["val/summary/entropy"] == approx(expected)


def test_sds_inputs_take_distinct_rows_in_the_input_shape():
    x = torch.randn(40, 2, 3, generator=torch.Generator().manual_seed(1))
    sets = sds_inputs(torch.cat([x, x[:5]]), None, 0)
    rows = torch.unique(x.reshape(40, -1), dim=0).double().numpy()
    for points, expected in zip(sets.values(), lfc.sds_pseudo_sets(rows)):
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
    volume = lfc.box_volume(module.aux_lower, module.aux_upper)
    normal, uniform = scores["normal"], scores["uniform"]
    expected = {
        NPD(): lfc.npd(normal, scores["autouad"]),
        RTM(): lfc.rtm(normal),
        EAG(): lfc.eag(normal),
        ExcessMass(): lfc.excess_mass(normal, uniform, volume)[0],
        MassVolume(): lfc.mass_volume(normal, uniform, volume),
        LossEntropy(): lfc.loss_entropy(score(module, inputs["train"], "loss/full")),
        SDS(): lfc.sds_error(
            scores["sds_targets"], scores["sds_outliers"], cut[0].item(), 0.5
        ),
    }
    evaluate(tmp_path, module, inputs, expected)
    for callback, value in expected.items():
        assert math.isfinite(value)
        assert callback.summary == {CKPT: pytest.approx(value, rel=1e-12)}
        assert callback.get_optimized_metric() == (CKPT, callback.summary[CKPT])


def test_mirror_needs_its_loaders(tmp_path):
    inputs = mirror_inputs()
    del inputs["autouad"]
    with pytest.raises(ValueError, match="autouad"):
        evaluate(tmp_path, ToyAE(), inputs, [NPD()])


def test_mirrors_give_nan_for_undefined_criteria(tmp_path):
    inputs = mirror_inputs()
    box = torch.zeros(512), torch.full((512,), 1e3)
    module = toy_with_box(inputs, *box, module=NegativeLossAE())
    callbacks = [ExcessMass(), MassVolume(), LossEntropy()]
    with pytest.warns(UserWarning):
        evaluate(tmp_path, module, inputs, callbacks)
    for callback in callbacks:
        assert math.isnan(callback.summary[CKPT])
