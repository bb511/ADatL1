from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import ADLightningModule
from src.callbacks import autouad
from src.callbacks.autouad import NPDCallback
from src.callbacks.components.utils import drawn_masks, placeholder_loader
from src.callbacks.score_dump import (
    ScoreDumpCallback,
    aux_inputs,
    input_moments,
    score_in_eval_mode,
)
from src.data.utils import unpack_batch

N_FEATURES = 4
KEYS = ["ascore/full", "ascore/mse", "ascore/kl", "fit/full"]
PASSES = ["initial.pt", "epoch_0000.pt", "epoch_0001.pt"]
META_KEYS = {
    "target_rate",
    "base_rate",
    "target_fpr",
    "module",
    "stored",
    "initial_sets",
    "aux_seed",
    "aux_size",
    "aux_masked",
    "aux_lower",
    "aux_upper",
    "aux_mean",
    "aux_std",
    "val",
    "test",
    "aux",
    "train",
}


def _block(gen, n, shift, label):
    x = torch.randn(n, N_FEATURES, generator=gen) + shift
    return x, torch.full((n,), label, dtype=torch.long)


def toy_sets(seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    sizes = {"val": (70, 40, 30, 20), "test": (50, 30, 25, 15)}
    sets = {"train": _block(gen, 96, 0.0, 0)}
    for split, (n_norm, n_ref, n_a, n_b) in sizes.items():
        sets[split] = {
            "normal": _block(gen, n_norm, 0.0, 0),
            "reference_normal": _block(gen, n_ref, 0.0, 0),
            "sig_a": _block(gen, n_a, 2.0, 1),
            "sig_b": _block(gen, n_b, -3.0, 2),
        }
    return sets


class CountingLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return super().__iter__()


class ToyDataModule(pl.LightningDataModule):
    def __init__(self, seed: int = 0, batch_size: int = 32):
        super().__init__()
        self.sets = toy_sets(seed)
        self.batch_size = batch_size
        self.data_train = None

    def setup(self, stage=None):
        self.data_train = TensorDataset(*self.sets["train"])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.loaders("val")

    def test_dataloader(self):
        self.test_loaders = self.loaders("test")
        return self.test_loaders

    def loaders(self, split):
        return {
            name: CountingLoader(TensorDataset(*xy), batch_size=self.batch_size)
            for name, xy in self.sets[split].items()
        }


class ToyAE(ADLightningModule):
    def __init__(
        self,
        target_rate: float = 0.1,
        base_rate: float | None = None,
        lr: float = 0.05,
        log_operational: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Linear(N_FEATURES, N_FEATURES)

    def model_step(self, batch):
        x, _ = batch
        err = (x - self.net(x)) ** 2
        score = err.mean(dim=1).detach()
        return {
            "loss": err.mean(),
            "loss/full": err.mean(dim=1).detach(),
            "ascore/full": score,
            "ascore/mse": err.sum(dim=1).detach(),
            "ascore/operational": self.compute_operational_ascore(score),
            "fit/full": score,
        }

    def outlog(self, outdict):
        logs = {"loss": outdict["loss"]}
        if self.hparams.log_operational:
            logs["ascore_operational"] = outdict["ascore/operational"]
        return logs

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def fit(
    root, callbacks, epochs=2, seed=0, module=None, datamodule=None, **trainer_kwargs
):
    pl.seed_everything(seed)
    module = ToyAE() if module is None else module
    datamodule = ToyDataModule() if datamodule is None else datamodule
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        default_root_dir=root,
        **{"enable_checkpointing": False, "num_sanity_val_steps": -1, **trainer_kwargs},
    )
    trainer.fit(module, datamodule=datamodule)
    return module, datamodule


def dump_callback(path, aux_seed=7, run_dir=None, **kwargs):
    sizes = {"aux_size": 50, "aux_batch_size": 16, **kwargs}
    return ScoreDumpCallback(
        dirpath=str(path),
        run_dir=str(path.parent if run_dir is None else run_dir),
        keys=KEYS,
        aux_seed=aux_seed,
        **sizes,
    )


def load(path):
    return torch.load(path, weights_only=True)


def assert_identical(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    else:
        assert a == b


def test_files_keys_and_shapes(tmp_path):
    module, datamodule = fit(tmp_path, [dump_callback(tmp_path / "scores")])
    files = sorted(p.name for p in (tmp_path / "scores").iterdir())
    assert files == ["epoch_0000.pt", "epoch_0001.pt", "initial.pt", "meta.pt"]

    meta = load(tmp_path / "scores" / "meta.pt")
    assert set(meta) == META_KEYS
    assert meta["target_fpr"] == pytest.approx(0.1)
    assert meta["module"].endswith("ToyAE")
    assert meta["aux_seed"] == 7 and meta["aux_size"] == 50
    assert meta["initial_sets"] == [] and meta["aux_masked"] is False
    for split in ("val", "test"):
        assert meta["stored"][split] == list(datamodule.sets[split])
        assert meta[split]["names"] == list(datamodule.sets[split])
        assert meta[split]["keys"] == ["ascore/full", "ascore/mse", "fit/full"]
        for name, (x, y) in datamodule.sets[split].items():
            sizes = [32] * (len(x) // 32) + ([len(x) % 32] if len(x) % 32 else [])
            assert meta[split]["batch_sizes"][name] == sizes
            assert meta[split]["labels"][name] == sorted(set(y.tolist()))
    assert meta["aux"]["names"] == ["gaussian", "uniform", "autouad"]
    assert meta["aux"]["batch_sizes"]["gaussian"] == [16, 16, 16, 2]
    assert meta["train"]["names"] == ["train"]
    assert meta["train"]["keys"] == ["loss/full"]

    for epoch, name in zip([-1, 0, 1], PASSES):
        record = load(tmp_path / "scores" / name)
        assert set(record) == {"epoch", "val", "test", "aux", "train", "state_dict"}
        assert record["epoch"] == epoch
        for split in ("val", "test"):
            for set_name, (x, _) in datamodule.sets[split].items():
                stored = {"ascore/full", "ascore/mse", "fit/full"}
                assert set(record[split][set_name]) == stored
                for scores in record[split][set_name].values():
                    assert scores.shape == (len(x),) and scores.dtype == torch.float32
        for set_name in ("gaussian", "uniform", "autouad"):
            assert record["aux"][set_name]["ascore/full"].shape == (50,)
        assert record["train"]["train"]["loss/full"].shape == (96,)
        assert set(record["state_dict"]) == set(module.state_dict())


def test_initial_state_holds_the_untrained_weights(tmp_path):
    module, _ = fit(tmp_path, [dump_callback(tmp_path / "scores")], epochs=1)
    pl.seed_everything(0)
    untrained = ToyAE().state_dict()
    initial = load(tmp_path / "scores" / "initial.pt")
    assert initial["epoch"] == -1
    assert_identical(initial["state_dict"], untrained)
    trained = load(tmp_path / "scores" / "epoch_0000.pt")["state_dict"]
    assert_identical(trained, module.state_dict())
    assert not torch.equal(trained["net.weight"], untrained["net.weight"])


def test_scores_come_from_the_stored_weights(tmp_path):
    _, datamodule = fit(tmp_path, [dump_callback(tmp_path / "scores")])
    meta = load(tmp_path / "scores" / "meta.pt")
    aux = {
        name: DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=16)
        for name, x in aux_inputs(meta).items()
    }
    train = {"train": DataLoader(datamodule.data_train, batch_size=16)}
    for name in PASSES:
        record = load(tmp_path / "scores" / name)
        module = ToyAE()
        module.load_state_dict(record["state_dict"])
        module.eval()
        loaders = {"val": datamodule.loaders("val"), "test": datamodule.loaders("test")}
        for split, split_loaders in {**loaders, "aux": aux, "train": train}.items():
            key = "loss/full" if split == "train" else "ascore/full"
            for set_name, loader in split_loaders.items():
                with torch.no_grad():
                    scores = [module.model_step(b)[key] for b in loader]
                expected = record[split][set_name][key]
                torch.testing.assert_close(torch.cat(scores), expected)


def test_stored_sets_restrict_val_and_test(tmp_path):
    stored = ["normal", "reference_normal"]
    paths = {"all": tmp_path / "all", "some": tmp_path / "some"}
    _, full = fit(tmp_path, [dump_callback(paths["all"])])
    _, datamodule = fit(tmp_path, [dump_callback(paths["some"], stored_sets=stored)])
    meta = {k: load(p / "meta.pt") for k, p in paths.items()}
    assert meta["some"].pop("stored") == {"val": stored, "test": stored}
    meta["all"].pop("stored")
    assert_identical(meta["some"], meta["all"])

    for name in PASSES:
        records = {k: load(p / name) for k, p in paths.items()}
        for split in ("val", "test"):
            some, every = records["some"].pop(split), records["all"].pop(split)
            assert_identical(some, {n: every[n] for n in stored})
        assert_identical(records["some"], records["all"])

    # The layout pass at fit start, then the initial pass and one per epoch.
    for name, loader in datamodule.test_loaders.items():
        assert loader.passes == (4 if name in stored else 1)
    assert all(loader.passes == 4 for loader in full.test_loaders.values())


def test_unknown_stored_set_raises(tmp_path):
    callback = dump_callback(tmp_path / "scores", stored_sets=["normal", "sig_c"])
    with pytest.raises(ValueError, match="sig_c"):
        fit(tmp_path, [callback], epochs=1)


def test_partial_sanity_check_raises(tmp_path):
    with pytest.raises(RuntimeError, match="num_sanity_val_steps"):
        fit(tmp_path, [dump_callback(tmp_path / "scores")], num_sanity_val_steps=2)


def test_existing_dump_raises(tmp_path):
    fit(tmp_path, [dump_callback(tmp_path / "scores")], epochs=1)
    with pytest.raises(RuntimeError, match="already holds"):
        fit(tmp_path, [dump_callback(tmp_path / "scores")], epochs=1)


def test_hydra_files_are_copied(tmp_path):
    hydra_dir = tmp_path / "run" / ".hydra"
    hydra_dir.mkdir(parents=True)
    for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        (hydra_dir / name).write_text(f"{name}: 1\n")
    callback = dump_callback(tmp_path / "scores", run_dir=tmp_path / "run")
    fit(tmp_path, [callback], epochs=1)
    for name in ("config.yaml", "overrides.yaml"):
        assert (tmp_path / "scores" / name).read_text() == f"{name}: 1\n"
    assert not (tmp_path / "scores" / "hydra.yaml").exists()


def test_aux_sets_depend_only_on_the_seed(tmp_path):
    metas = []
    for run, (seed, aux_seed) in enumerate([(0, 7), (1, 7), (0, 8)]):
        path = tmp_path / f"run{run}"
        fit(tmp_path, [dump_callback(path, aux_seed)], epochs=1, seed=seed)
        metas.append(load(path / "meta.pt"))

    x_train = toy_sets()["train"][0]
    lower, upper = x_train.min(dim=0).values, x_train.max(dim=0).values
    inputs = [aux_inputs(meta) for meta in metas]
    uniform = inputs[0]["uniform"]
    assert torch.equal(metas[0]["aux_lower"], lower)
    assert torch.equal(metas[0]["aux_upper"], upper)
    assert bool(((uniform >= lower) & (uniform <= upper)).all())
    for name in ("gaussian", "uniform"):
        assert torch.equal(inputs[0][name], inputs[1][name])
        assert not torch.equal(inputs[0][name], inputs[2][name])


def test_aux_sets_take_the_input_shape():
    meta = {**input_moments(torch.randn(64, 3, 4, 4)), "aux_seed": 0, "aux_size": 10}
    for inputs in aux_inputs(meta).values():
        assert inputs.shape == (10, 3, 4, 4)


def test_scoring_pass_restores_mode_and_rng():
    module = ToyAE()
    module.train()
    trainer = SimpleNamespace(
        strategy=SimpleNamespace(batch_to_device=lambda batch, device, idx: batch)
    )
    rng = torch.get_rng_state()
    score_in_eval_mode(trainer, module, ToyDataModule().loaders("test"), KEYS)
    assert module.training
    assert torch.equal(torch.get_rng_state(), rng)


def test_autouad_set_follows_the_training_moments(tmp_path):
    size = 4000
    fit(tmp_path, [dump_callback(tmp_path / "scores", aux_size=size)], epochs=1)
    meta = load(tmp_path / "scores" / "meta.pt")
    x_train = toy_sets()["train"][0]
    mean, std = x_train.mean(dim=0), x_train.std(dim=0, correction=0)
    assert torch.equal(meta["aux_mean"], mean) and torch.equal(meta["aux_std"], std)
    inputs = aux_inputs(meta)
    torch.testing.assert_close(inputs["autouad"].mean(dim=0), mean, atol=0.1, rtol=0.0)
    torch.testing.assert_close(inputs["autouad"].std(dim=0), std, atol=0.1, rtol=0.0)

    # The two sets as the dump drew them before the autouad set existed.
    gen = torch.Generator().manual_seed(7)
    lower, upper = x_train.min(dim=0).values, x_train.max(dim=0).values
    gaussian = torch.randn((size, N_FEATURES), generator=gen)
    uniform = lower + (upper - lower) * torch.rand((size, N_FEATURES), generator=gen)
    assert torch.equal(inputs["gaussian"], gaussian)
    assert torch.equal(inputs["uniform"], uniform)


@pytest.mark.parametrize("stored", [[], ["reference_normal"]])
def test_initial_sets_are_stored_at_the_sanity_pass_alone(tmp_path, stored):
    initial = ["sig_a", "normal"]
    paths = {"all": tmp_path / "all", "lean": tmp_path / "lean"}
    fit(tmp_path, [dump_callback(paths["all"])])
    kwargs = {"stored_sets": stored, "initial_sets": initial, "train_pass": False}
    fit(tmp_path, [dump_callback(paths["lean"], **kwargs)])
    meta = {k: load(p / "meta.pt") for k, p in paths.items()}
    assert meta["lean"]["stored"] == {"val": stored, "test": stored}
    assert meta["lean"]["initial_sets"] == initial
    assert "train" not in meta["lean"]
    assert_identical(meta["lean"]["aux"], meta["all"]["aux"])
    for key in ("names", "batch_sizes", "labels"):
        assert meta["lean"]["val"][key] == meta["all"]["val"][key]

    for name in PASSES:
        lean, full = load(paths["lean"] / name), load(paths["all"] / name)
        assert set(lean) == {"epoch", "val", "test", "aux", "state_dict"}
        kept = stored + initial if name == "initial.pt" else stored
        expected_val = {n: s for n, s in full["val"].items() if n in kept}
        assert_identical(lean["val"], expected_val)
        assert_identical(lean["test"], {n: full["test"][n] for n in stored})
        assert_identical(lean["aux"], full["aux"])
        assert_identical(lean["state_dict"], full["state_dict"])


def test_initial_set_outside_validation_raises(tmp_path):
    callback = dump_callback(tmp_path / "scores", initial_sets=["normal", "sig_c"])
    with pytest.raises(ValueError, match="sig_c"):
        fit(tmp_path, [callback], epochs=1)


def test_maskless_probes_are_the_unmasked_points(tmp_path):
    callback = dump_callback(tmp_path / "scores")
    fit(tmp_path, [callback], epochs=1)
    meta = load(tmp_path / "scores" / "meta.pt")
    for name, x in aux_inputs(meta).items():
        tensors = callback._loaders["aux"][name].dataset.tensors
        assert len(tensors) == 2 and torch.equal(tensors[0], x)


def masked_sets(sets: dict, seed: int = 3) -> dict:
    # The L1AD layout (x, mask, l1bit, y) with two objects of two features: an object
    # is present or padded whole, and padding is +0.
    gen = torch.Generator().manual_seed(seed)

    def with_mask(x, y):
        n = len(x)
        mask = (torch.rand(n, 2, 1, generator=gen) < 0.6).expand(n, 2, 2).clone()
        x = torch.where(mask, x.reshape(n, 2, 2), torch.zeros(n, 2, 2))
        return x, mask, torch.zeros(n, dtype=torch.bool), y

    masked = {"train": with_mask(*sets["train"])}
    for split in ("val", "test"):
        masked[split] = {name: with_mask(*xy) for name, xy in sets[split].items()}
    return masked


class MaskedDataModule(ToyDataModule):
    def __init__(self):
        super().__init__()
        self.sets = masked_sets(self.sets)


class MaskedToyAE(ToyAE):
    def model_step(self, batch):
        view = unpack_batch(batch)
        return super().model_step((view.x.reshape(len(view.x), -1), view.y))


def fit_masked(root, callbacks):
    return fit(root, callbacks, module=MaskedToyAE(), datamodule=MaskedDataModule())


def test_masked_probes_take_the_masks_of_training_rows(tmp_path):
    callback = dump_callback(tmp_path / "scores")
    _, datamodule = fit_masked(tmp_path, [callback])
    meta = load(tmp_path / "scores" / "meta.pt")
    assert meta["aux_masked"] is True
    train_mask = datamodule.sets["train"][1]
    mask = drawn_masks(train_mask, meta["aux_size"], meta["aux_seed"])
    inputs = aux_inputs(meta)
    for name, loader in callback._loaders["aux"].items():
        x, probe_mask, l1bit, _ = loader.dataset.tensors
        assert torch.equal(probe_mask, mask) and not l1bit.any()
        assert not x[~mask].any() and not x[~mask].signbit().any()
        assert torch.equal(x[mask], inputs[name][mask])

    # A reader rebuilds the probes from meta.pt and the training masks.
    for name in PASSES:
        record = load(tmp_path / "scores" / name)
        module = MaskedToyAE()
        module.load_state_dict(record["state_dict"])
        module.eval()
        for set_name, x in inputs.items():
            with torch.no_grad():
                batches = placeholder_loader(x, 16, mask)
                scores = [module.model_step(b)["ascore/full"] for b in batches]
            expected = record["aux"][set_name]["ascore/full"]
            torch.testing.assert_close(torch.cat(scores), expected)


def test_dumped_autouad_scores_are_the_ones_npd_reads(tmp_path, monkeypatch):
    seen, npd = [], autouad.npd

    def recording_npd(normal, noise, eps=autouad.EPS):
        seen.append(torch.as_tensor(noise).clone())
        return npd(normal, noise, eps)

    monkeypatch.setattr(autouad, "npd", recording_npd)
    npd_callback = NPDCallback(aux_seed=7, aux_size=50, batch_size=16)
    fit_masked(tmp_path, [npd_callback, dump_callback(tmp_path / "scores")])
    dumped = [load(tmp_path / "scores" / name)["aux"]["autouad"] for name in PASSES]
    # NPD skips the sanity pass.
    assert len(seen) == len(dumped) - 1
    for scores, record in zip(seen, dumped[1:]):
        assert torch.equal(scores, record["ascore/full"])
