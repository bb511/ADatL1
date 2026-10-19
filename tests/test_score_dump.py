from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import ADLightningModule
from src.callbacks.score_dump import ScoreDumpCallback, _score_in_eval_mode

N_FEATURES = 4
KEYS = ["ascore/full", "ascore/mse", "ascore/kl"]


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
        return self.loaders("test")

    def loaders(self, split):
        return {
            name: DataLoader(TensorDataset(*xy), batch_size=self.batch_size)
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
            "ascore/full": score,
            "ascore/mse": err.sum(dim=1).detach(),
            "ascore/operational": self.compute_operational_ascore(score),
        }

    def outlog(self, outdict):
        logs = {"loss": outdict["loss"]}
        if self.hparams.log_operational:
            logs["ascore_operational"] = outdict["ascore/operational"]
        return logs

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def fit(root, callbacks, epochs=2, seed=0, module=None, **trainer_kwargs):
    pl.seed_everything(seed)
    module = ToyAE() if module is None else module
    datamodule = ToyDataModule()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        default_root_dir=root,
        **{"enable_checkpointing": False, "num_sanity_val_steps": 2, **trainer_kwargs},
    )
    trainer.fit(module, datamodule=datamodule)
    return module, datamodule


def dump_callback(path, aux_seed=7):
    return ScoreDumpCallback(
        dirpath=str(path), keys=KEYS, aux_seed=aux_seed, aux_size=50, aux_batch_size=16
    )


def test_files_keys_and_shapes(tmp_path):
    module, datamodule = fit(tmp_path, [dump_callback(tmp_path / "scores")])
    files = sorted(p.name for p in (tmp_path / "scores").iterdir())
    assert files == ["epoch_0000.pt", "epoch_0001.pt", "meta.pt"]

    meta = torch.load(tmp_path / "scores" / "meta.pt", weights_only=True)
    record = torch.load(tmp_path / "scores" / "epoch_0001.pt", weights_only=True)
    assert record["epoch"] == 1
    assert meta["target_fpr"] == pytest.approx(0.1)
    assert meta["module"].endswith("ToyAE")
    for split in ("val", "test"):
        assert meta[split]["names"] == list(datamodule.sets[split])
        assert meta[split]["keys"] == ["ascore/full", "ascore/mse"]
        for name, (x, y) in datamodule.sets[split].items():
            sizes = [32] * (len(x) // 32) + ([len(x) % 32] if len(x) % 32 else [])
            assert meta[split]["batch_sizes"][name] == sizes
            assert meta[split]["labels"][name] == sorted(set(y.tolist()))
            assert set(record[split][name]) == {"ascore/full", "ascore/mse"}
            for scores in record[split][name].values():
                assert scores.shape == (len(x),) and scores.dtype == torch.float32

    for name in ("gaussian", "uniform"):
        assert meta["aux_inputs"][name].shape == (50, N_FEATURES)
        assert record["aux"][name]["ascore/full"].shape == (50,)
    assert set(record["state_dict"]) == set(module.state_dict())


def test_scores_come_from_the_epoch_weights(tmp_path):
    _, datamodule = fit(tmp_path, [dump_callback(tmp_path / "scores")])
    meta = torch.load(tmp_path / "scores" / "meta.pt", weights_only=True)
    aux = {
        name: DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=16)
        for name, x in meta["aux_inputs"].items()
    }
    for epoch in (0, 1):
        path = tmp_path / "scores" / f"epoch_{epoch:04d}.pt"
        record = torch.load(path, weights_only=True)
        module = ToyAE()
        module.load_state_dict(record["state_dict"])
        module.eval()
        loaders = {"val": datamodule.loaders("val"), "test": datamodule.loaders("test")}
        for split, split_loaders in {**loaders, "aux": aux}.items():
            for name, loader in split_loaders.items():
                with torch.no_grad():
                    scores = [module.model_step(b)["ascore/full"] for b in loader]
                expected = record[split][name]["ascore/full"]
                torch.testing.assert_close(torch.cat(scores), expected)


def test_aux_sets_depend_only_on_the_seed(tmp_path):
    metas = []
    for run, (seed, aux_seed) in enumerate([(0, 7), (1, 7), (0, 8)]):
        path = tmp_path / f"run{run}"
        fit(tmp_path, [dump_callback(path, aux_seed)], epochs=1, seed=seed)
        metas.append(torch.load(path / "meta.pt", weights_only=True))

    x_train = toy_sets()["train"][0]
    lower, upper = x_train.min(dim=0).values, x_train.max(dim=0).values
    uniform = metas[0]["aux_inputs"]["uniform"]
    assert torch.equal(metas[0]["aux_lower"], lower)
    assert torch.equal(metas[0]["aux_upper"], upper)
    assert bool(((uniform >= lower) & (uniform <= upper)).all())
    for name in ("gaussian", "uniform"):
        assert torch.equal(metas[0]["aux_inputs"][name], metas[1]["aux_inputs"][name])
        assert not torch.equal(
            metas[0]["aux_inputs"][name], metas[2]["aux_inputs"][name]
        )


def test_scoring_pass_restores_mode_and_rng():
    module = ToyAE()
    module.train()
    trainer = SimpleNamespace(
        strategy=SimpleNamespace(batch_to_device=lambda batch, device, idx: batch)
    )
    rng = torch.get_rng_state()
    _score_in_eval_mode(trainer, module, ToyDataModule().loaders("test"), KEYS)
    assert module.training
    assert torch.equal(torch.get_rng_state(), rng)
