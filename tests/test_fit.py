import math
from functools import partial
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.ae import AE
from src.algorithms.components.decoder import Decoder, ImageDecoder
from src.algorithms.components.dte import (
    DTEPredictor,
    ImageDTEPredictor,
    eval_corruption,
)
from src.algorithms.components.encoder import (
    DeepSetsEncoder,
    Encoder,
    ImageEncoder,
    ImageVariationalEncoder,
    VariationalEncoder,
)
from src.algorithms.components.flow import RealNVP as Flow
from src.algorithms.dsae import DeepSetsAE
from src.algorithms.dte import DTE
from src.algorithms.image_ae import ImageAE
from src.algorithms.image_dte import ImageDTE
from src.algorithms.image_svdd import ImageDeepSVDD
from src.algorithms.image_vae import ImageVAE
from src.algorithms.nae import NAE
from src.algorithms.neutral import NeuTraLAD
from src.algorithms.realnvp import RealNVP
from src.algorithms.svdd import DeepSVDD
from src.algorithms.vae import VAE
from src.callbacks.operational import FitCallback
from src.callbacks.score_dump import ScoreDumpCallback
from src.evaluation.callbacks.fit import Fit
from src.evaluation.replay import ReplayModule
from tests.test_eval_stage import Capture
from tests.test_physics_models import FEATURE_MAP, OBJECTS
from tests.test_score_dump import (
    KEYS,
    N_FEATURES,
    ToyAE,
    ToyDataModule,
    dump_callback,
    fit,
    load,
)

DIM = 5
BETA = 0.3


def _x(n: int = 32, seed: int = 1, dim: int = DIM) -> torch.Tensor:
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


def _images(n: int = 8, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(seed))


def _labels(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(len(x), dtype=torch.long)


def _step(module, batch) -> dict:
    module.eval()
    with torch.no_grad():
        return module.model_step(batch)


def _vae() -> VAE:
    torch.manual_seed(0)
    encoder = VariationalEncoder(in_dim=DIM, nodes=[8, 6, 2])
    decoder = Decoder(nodes=[2, 8], out_dim=DIM)
    module = VAE(
        encoder,
        decoder,
        kl_scale=BETA,
        target_rate=0.01,
        base_rate=None,
        eval_mean=True,
    )
    # At step 0 of the warmup the loss weighs the KL by 0; the fit takes the final beta.
    module._setup_kl_annealing(0.5, 100)
    return module


def _image_vae(eval_mean: bool = True) -> ImageVAE:
    torch.manual_seed(0)
    encoder = ImageVariationalEncoder(3, (8, 8), [4, 16, 2], [2])
    decoder = ImageDecoder([2, 4], 3, (8, 8), [2])
    module = ImageVAE(
        encoder,
        decoder,
        kl_scale=BETA,
        target_rate=0.01,
        base_rate=None,
        eval_mean=eval_mean,
    )
    module._setup_kl_annealing(0.5, 100)
    return module


def _masked_mse(x, reconstruction, mask) -> torch.Tensor:
    squared = ((x - reconstruction) ** 2 * mask).flatten(1)
    return squared.sum(dim=1) / mask.flatten(1).sum(dim=1)


def _neg_elbo(module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    z_mean, z_log_var, _ = module.encoder(x)
    n = mask.flatten(1).sum(dim=1)
    mse = _masked_mse(x, module.decoder(z_mean), mask)
    kl = -0.5 * (1 + z_log_var - z_mean**2 - z_log_var.exp()).sum(dim=1)
    return (mse + BETA * kl) / BETA + n / 2 * torch.log(math.pi * n * BETA)


def _dcl(module: NeuTraLAD, x: torch.Tensor) -> torch.Tensor:
    # Qiu et al., eq. 2, term by term, divided by K log K.
    views = module.views(x)
    z = module.encoder(views.flatten(0, 1)).unflatten(0, views.shape[:2])
    z = F.normalize(z, dim=-1)
    log_h = z @ z.transpose(1, 2) / module.temperature
    n_transforms = views.shape[1] - 1
    terms = []
    for k in range(1, n_transforms + 1):
        others = [j for j in range(n_transforms + 1) if j != k]
        terms.append(torch.logsumexp(log_h[:, k, others], dim=1) - log_h[:, k, 0])
    return torch.stack(terms).sum(dim=0) / (n_transforms * math.log(n_transforms))


def test_ae_fit_is_the_mse_over_observed_features():
    torch.manual_seed(0)
    encoder = Encoder(in_dim=DIM, nodes=[8, 3])
    decoder = Decoder(nodes=[3, 8], out_dim=DIM)
    module = AE(encoder, decoder, target_rate=0.01, base_rate=None)
    x = _x()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = ((x - module(x)[1]) ** 2).mean(dim=1)
    assert torch.equal(outputs["fit/full"], outputs["ascore/mse"])
    torch.testing.assert_close(outputs["fit/full"], expected)


@pytest.mark.parametrize("masked", [False, True])
def test_vae_fit_is_the_gaussian_neg_elbo(masked):
    module, x = _vae(), _x()
    mask = torch.ones_like(x)
    if masked:
        mask[::2, :2] = 0.0
        mask[1::4, 4] = 0.0
        batch = (x, mask, _labels(x), _labels(x))
    else:
        batch = (x, _labels(x))
    outputs = _step(module, batch)
    with torch.no_grad():
        expected = _neg_elbo(module, x, mask)
    assert module.kl_scale(0) == 0.0
    torch.testing.assert_close(outputs["fit/full"], expected)


@pytest.mark.parametrize("objective", ["one_class", "soft_boundary"])
def test_svdd_fit_is_the_distance_per_latent_coordinate(objective):
    torch.manual_seed(0)
    encoder = Encoder(in_dim=DIM, nodes=[8, 3], activation="relu", bias=False)
    module = DeepSVDD(encoder=encoder, objective=objective, center_eps=0.0)
    module.center = torch.randn(3, generator=torch.Generator().manual_seed(2))
    x = _x()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = ((encoder(x) - module.center) ** 2).sum(dim=1) / 3
    torch.testing.assert_close(outputs["fit/full"], expected)


def test_realnvp_fit_is_the_negative_log_likelihood():
    torch.manual_seed(0)
    flow = Flow(input_dim=DIM, n_flows=2, hidden_dim=8, n_hidden_layers=1)
    module = RealNVP(flow=flow, target_rate=0.01)
    x = _x()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = -module(x)
    assert torch.equal(outputs["fit/full"], outputs["ascore/full"])
    torch.testing.assert_close(outputs["fit/full"], expected)


def test_dsae_fit_is_the_mse_over_observed_features():
    torch.manual_seed(0)
    phi_nodes = {"FET": [8], "jets": [16, 8]}
    encoder = DeepSetsEncoder(object_phi_nodes=phi_nodes, rho_nodes=[32, 16], **OBJECTS)
    decoder = Decoder(nodes=[16, 32], out_dim=9)
    module = DeepSetsAE(encoder, decoder, target_rate=0.01, base_rate=None)
    module.object_feature_map = FEATURE_MAP
    mask = torch.ones(32, 9)
    mask[:, FEATURE_MAP["FET"]["eta"]] = 0.0
    # The second jet is absent from every third event.
    mask[::3, 6:] = 0.0
    x = _x(dim=9) * mask
    outputs = _step(module, (x, mask, None, _labels(x)))
    expected = _masked_mse(x, outputs["reconstructed_data"], mask)
    assert torch.equal(outputs["fit/full"], outputs["ascore/full"])
    torch.testing.assert_close(outputs["fit/full"], expected)


def test_image_ae_fit_is_the_mse_over_pixels():
    torch.manual_seed(0)
    encoder = ImageEncoder(in_channels=3, nodes=[4, 3], input_size=(8, 8), strides=[2])
    decoder = ImageDecoder([3, 4], 3, (8, 8), [2])
    module = ImageAE(encoder, decoder, target_rate=0.01, base_rate=None)
    x = _images()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = ((x - module(x)[1]) ** 2).flatten(1).mean(dim=1)
    assert torch.equal(outputs["fit/full"], outputs["ascore/full"])
    torch.testing.assert_close(outputs["fit/full"], expected)


def test_image_vae_fit_is_the_neg_elbo_of_the_posterior_mean():
    module, x = _image_vae(), _images()
    first = _step(module, (x, _labels(x)))["fit/full"]
    second = _step(module, (x, _labels(x)))["fit/full"]
    with torch.no_grad():
        expected = _neg_elbo(module, x, torch.ones_like(x))
    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected)
    sampled = _image_vae(eval_mean=False)
    draws = [_step(sampled, (x, _labels(x)))["fit/full"] for _ in range(2)]
    assert not torch.equal(*draws)


def test_image_svdd_fit_is_the_distance_per_latent_coordinate():
    torch.manual_seed(0)
    encoder = ImageEncoder(3, [4, 3], (8, 8), [2], bias=False)
    module = ImageDeepSVDD(encoder=encoder, center_eps=0.0)
    module.center = torch.randn(3, generator=torch.Generator().manual_seed(2))
    x = _images()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = ((encoder(x) - module.center) ** 2).sum(dim=1) / 3
    torch.testing.assert_close(outputs["fit/full"], expected)


def _dte(image: bool):
    torch.manual_seed(0)
    if image:
        predictor = ImageDTEPredictor(3, [4, 3], 5, (8, 8), [2])
        return ImageDTE(predictor, n_steps=300, n_bins=5, base_rate=None), _images()
    predictor = DTEPredictor(in_dim=DIM, hidden_dims=[8, 8], out_dim=5, dropout=0.0)
    return DTE(predictor, n_steps=300, n_bins=5, target_rate=0.01), _x()


def _cross_entropy(module, x, timesteps, noise=None) -> torch.Tensor:
    with torch.no_grad():
        logits = module(module.corrupt(x, timesteps, noise=noise))
    targets = module.time_to_bin(timesteps)
    return F.cross_entropy(logits, targets, reduction="none")


@pytest.mark.parametrize("image", [False, True])
def test_dte_training_fit_is_the_cross_entropy_at_a_random_timestep(image):
    module, x = _dte(image)
    module.train()
    torch.manual_seed(3)
    with torch.no_grad():
        outputs = module.model_step((x, _labels(x)))
        again = module.model_step((x, _labels(x)))
    torch.manual_seed(3)
    timesteps = torch.randint(0, 300, (len(x),))
    assert torch.equal(outputs["fit/full"], outputs["loss/full"])
    torch.testing.assert_close(
        outputs["fit/full"], _cross_entropy(module, x, timesteps)
    )
    assert not torch.equal(outputs["fit/full"], again["fit/full"])


@pytest.mark.parametrize("image", [False, True])
def test_dte_eval_fit_is_the_cross_entropy_at_a_fixed_corruption(image):
    module, x = _dte(image)
    first, second = (_step(module, (x, _labels(x)))["fit/full"] for _ in range(2))
    expected = _cross_entropy(module, x, *eval_corruption(x, 300))
    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected)


@pytest.mark.parametrize("x", [_x(64), _images(64)], ids=["tabular", "image"])
def test_eval_timesteps_are_a_function_of_each_event(x):
    timesteps, noise = eval_corruption(x, 300)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(4))
    batched = torch.cat([eval_corruption(part, 300)[0] for part in x.split(24)])
    assert torch.equal(batched, timesteps)
    assert torch.equal(eval_corruption(x[order], 300)[0], timesteps[order])
    assert 0 <= timesteps.min() and timesteps.max() < 300
    assert len(timesteps.unique()) > len(x) // 2
    assert noise.shape == x.shape and torch.equal(noise, eval_corruption(x, 300)[1])


def test_neutral_fit_is_the_normalised_contrastive_loss():
    torch.manual_seed(0)
    encoder = Encoder(in_dim=DIM, nodes=[8, 4])
    module = NeuTraLAD(encoder, in_dim=DIM, n_transforms=4, target_rate=0.01)
    x = _x()
    outputs = _step(module, (x, _labels(x)))
    with torch.no_grad():
        expected = _dcl(module, x)
    assert torch.equal(outputs["fit/full"], outputs["ascore/full"])
    torch.testing.assert_close(outputs["fit/full"], expected)


def test_nae_fit_is_the_energy_of_the_event():
    torch.manual_seed(0)
    module = NAE(
        encoder=Encoder(in_dim=DIM, nodes=[8, 4]),
        decoder=Decoder(nodes=[4, 8], out_dim=DIM),
        pretrain_epochs=1,
        temperature=1e-2,
        neg_energy_weight=1.0,
        latent_steps=3,
        latent_step_size=1e-3,
        input_steps=4,
        input_step_size=1e-3,
        target_rate=0.01,
        base_rate=None,
    )
    mask = torch.ones(32, DIM)
    mask[::2, 3:] = 0.0
    x = _x() * mask
    outputs = _step(module, (x, mask, _labels(x), _labels(x)))
    with torch.no_grad():
        reconstruction = module.decoder(F.normalize(module.encoder(x), dim=1))
    expected = _masked_mse(x, reconstruction, mask)
    assert torch.equal(outputs["fit/full"], outputs["ascore/full"])
    torch.testing.assert_close(outputs["fit/full"], expected)


class _Logs:
    def __init__(self):
        self.logged = {}

    def log_dict(self, metrics, **kwargs):
        self.logged.update(metrics)


def _callback_logs(epochs: list[torch.Tensor], batch_size: int) -> list[dict]:
    callback, module = FitCallback(), _Logs()
    loaders = {"normal": None, "sig_a": None}
    trainer = SimpleNamespace(val_dataloaders=loaders, sanity_checking=False)
    callback.on_fit_start(trainer, module)
    logs = []
    for fits in epochs:
        callback.on_validation_epoch_start(trainer, module)
        for chunk in fits.split(batch_size):
            for index, shift in enumerate((0.0, 100.0)):
                outputs = {"fit/full": chunk + shift}
                callback.on_validation_batch_end(
                    trainer, module, outputs, None, 0, index
                )
        callback.on_validation_epoch_end(trainer, module)
        logs.append(dict(module.logged))
    return logs


def test_fit_callback_logs_the_mean_over_rows_whatever_the_batching():
    gen = torch.Generator().manual_seed(0)
    epochs = [torch.rand(70, generator=gen) * 10 ** (i + 1) for i in range(2)]
    whole, small = _callback_logs(epochs, 70), _callback_logs(epochs, 8)
    assert whole == small
    means = [fits.double().mean().item() for fits in epochs]
    assert [log["val/summary/fit"] for log in whole] == means
    ema = [means[0], 0.9 * means[0] + 0.1 * means[1]]
    assert [log["val/summary/fit_ema"] for log in whole] == pytest.approx(ema)


def _mirror_records(module, x: torch.Tensor, batch_size: int, tmp_path) -> dict:
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[Fit()],
    )
    trainer.metric_records = {}
    module._ckpt_path = tmp_path / "epoch_0000.pt"
    loader = DataLoader(TensorDataset(x, _labels(x)), batch_size=batch_size)
    trainer.test(module, dataloaders={"normal": loader}, verbose=False)
    return trainer.metric_records


def test_fit_mirror_records_the_mean_and_its_error_whatever_the_batching(tmp_path):
    torch.manual_seed(0)
    module, x = ToyAE(), 20 * _x(70, dim=N_FEATURES)
    whole = _mirror_records(module, x, 70, tmp_path)
    small = _mirror_records(module, x, 8, tmp_path)
    fits = _step(module, (x, _labels(x)))["fit/full"].double()
    assert whole["fit/normal"] == pytest.approx(small["fit/normal"], rel=1e-12)
    assert whole["fit/normal"] == pytest.approx(fits.mean().item(), rel=1e-12)
    se = fits.std().item() / math.sqrt(len(fits))
    assert whole["fit/normal_se"] == pytest.approx(se, rel=1e-12)


def _served(replay: ReplayModule, split: str) -> dict:
    capture = Capture()
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[capture],
    )
    replay.serve(split)
    trainer.test(replay, dataloaders=ToyDataModule().loaders(split), verbose=False)
    return capture.outputs


def _replay(path, epoch_file: str, record: dict | None = None) -> ReplayModule:
    replay = ReplayModule(ToyAE(), load(path / "meta.pt"))
    record = load(path / epoch_file) if record is None else record
    replay.load_epoch(record, path.parent / epoch_file)
    return replay


def test_replay_rescores_the_fit_of_normal_where_the_dump_lacks_it(tmp_path):
    path = tmp_path / "scores"
    keys = [key for key in KEYS if key != "fit/full"]
    sizes = {"aux_seed": 7, "aux_size": 50, "aux_batch_size": 16}
    fit(tmp_path, [ScoreDumpCallback(str(path), str(tmp_path), keys, **sizes)])
    assert "fit/full" not in load(path / "meta.pt")["val"]["keys"]
    record = load(path / "epoch_0001.pt")
    weights = ToyAE()
    weights.load_state_dict(record["state_dict"])
    served = _served(_replay(path, "epoch_0001.pt"), "val")
    rescored = torch.cat([outputs["fit/full"] for outputs in served["normal"]])
    x, y = ToyDataModule().sets["val"]["normal"]
    torch.testing.assert_close(rescored, _step(weights, (x, y))["fit/full"])
    served.pop("normal")
    test = _served(_replay(path, "epoch_0001.pt"), "test")
    for outputs in [*served.values(), *test.values()]:
        assert not [o for o in outputs if "fit/full" in o]


def _toy_dte() -> DTE:
    predictor = DTEPredictor(in_dim=N_FEATURES, hidden_dims=[8], out_dim=5)
    optimizer = partial(torch.optim.SGD, lr=0.05)
    return DTE(predictor, n_bins=5, target_rate=0.1, optimizer=optimizer)


def test_replay_reproduces_the_rescored_dte_fit(tmp_path):
    path = tmp_path / "scores"
    keys = [key for key in KEYS if key != "fit/full"]
    sizes = {"aux_seed": 7, "aux_size": 50, "aux_batch_size": 16}
    dump = ScoreDumpCallback(str(path), str(tmp_path), keys, **sizes)
    fit(tmp_path, [dump], module=_toy_dte())
    meta, record = load(path / "meta.pt"), load(path / "epoch_0001.pt")
    fits = []
    for _ in range(2):
        replay = ReplayModule(_toy_dte(), meta)
        replay.load_epoch(record, tmp_path / "epoch_0001.pt")
        served = _served(replay, "val")["normal"]
        fits.append(torch.cat([outputs["fit/full"] for outputs in served]))
    assert torch.equal(*fits)


def test_replay_serves_a_stored_fit(tmp_path):
    path = tmp_path / "scores"
    fit(tmp_path, [dump_callback(path)])
    record = load(path / "epoch_0001.pt")
    # Shifted, so that only the stored values, not a rescoring, can give them.
    record["val"]["normal"]["fit/full"] += 100.0
    served = _served(_replay(path, "epoch_0001.pt", record), "val")
    stored = torch.cat([outputs["fit/full"] for outputs in served["normal"]])
    assert torch.equal(stored, record["val"]["normal"]["fit/full"])
