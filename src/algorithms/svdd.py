"""Deep SVDD: a bias-free network that maps typical inputs close to a fixed centre.

Ruff et al., "Deep One-Class Classification", ICML 2018, PMLR 80:4393-4402. The
anomaly score is the squared distance of the representation to the centre (eq. 5). The
centre rule (the mean output of an untrained forward pass over the training data, with
coordinates nearer zero than eps moved to +-eps) and the soft-boundary radius rule (the
(1 - nu) quantile of each training batch's distances once a warm-up has passed) follow
the authors' code, github.com/lukasruff/Deep-SVDD-PyTorch, MIT License, Copyright (c)
2018 Lukas Ruff.
"""

from itertools import islice

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.svdd import soft_boundary_loss
from src.data.utils import unpack_batch

OBJECTIVES = ("one_class", "soft_boundary")

# Logged name of every model_step value that outlog passes on; the physics loss
# checkpoint monitors val/normal/loss_distance_raw.
_LOGGED = {
    "loss": "loss",
    "loss_mean": "loss/mean",
    "loss_distance_raw": "loss/distance_raw/mean",
    "loss_reg_scaled": "loss/reg_scaled/mean",
    "ascore_operational": "ascore/operational",
    "ascore_q50": "ascore/q50",
    "ascore_q99": "ascore/q99",
    "z_squared": "z_squared",
    "center_norm": "center_norm",
    "radius": "radius",
}

# (failed, message) pairs over the saved hyperparameters, checked in this order.
_HPARAM_CHECKS = (
    (lambda hp: not 0.0 < hp.nu <= 1.0, "nu must lie in (0, 1]."),
    (lambda hp: hp.center_init_method != "mean", "center_init_method must be 'mean'."),
    (lambda hp: hp.center_eps < 0.0, "center_eps must not be negative."),
    (
        lambda hp: hp.center_max_batches is not None and hp.center_max_batches < 1,
        "center_max_batches must be None (every batch) or at least 1.",
    ),
    (lambda hp: hp.radius_warmup_epochs < 0, "radius_warmup_epochs must be >= 0."),
)


def _objective(objective: str | None, soft_boundary: bool) -> str:
    """The objective's name; without one, the boolean flag chooses it."""
    if objective is None:
        return "soft_boundary" if soft_boundary else "one_class"
    if objective not in OBJECTIVES or (soft_boundary and objective != "soft_boundary"):
        raise ValueError(
            f"objective must be one of {OBJECTIVES} and agree with soft_boundary, got "
            f"{objective!r} with soft_boundary={soft_boundary}."
        )
    return objective


def _squared_distance(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """||z - c||² per event: z has shape (batch, latent), the result (batch,)."""
    return ((z - center) ** 2).sum(dim=1)


def _away_from_zero(center: torch.Tensor, eps: float) -> torch.Tensor:
    """Move every coordinate with abs(c_i) < eps to -eps if negative, else to +eps.

    A coordinate at zero can be met by all-zero weights, the trivial solution of Ruff et
    al., Prop. 1. The authors' code leaves an exact zero in place.
    """
    pushed = torch.where(center < 0, -eps, eps).to(center.dtype)
    return torch.where(center.abs() < eps, pushed, center)


class DeepSVDD(ADLightningModule):
    """Deep SVDD on flattened inputs, one-class (eq. 4) or soft-boundary (eq. 3).

    The centre c is fixed at the start of the first fit and never trained. A bias lets
    the network map every input onto c (Ruff et al., Prop. 2), so the encoder's config
    sets bias off, and batch normalisation too, whose affine shift is a bias. The
    weight decay (λ/2) Σ ||W||² is part of the loss, with λ =
    ``network_weight_decay``, apart from any decay the optimiser applies. For both
    objectives the anomaly score is the squared distance, so a soft-boundary score is
    eq. 5's adjusted score plus R², the same ranking. ``target_rate`` is the FPR when
    ``base_rate`` is None, else a rate that ``base_rate`` divides. ``ckpt`` names a
    Lightning checkpoint whose weights, centre and radius are loaded.
    """

    # Subclasses whose encoder reads (batch, channels, height, width) set this False.
    _flat_input = True

    def __init__(
        self,
        encoder: nn.Module,
        features: nn.Module | None = None,
        center_init_method: str = "mean",
        center_eps: float = 0.1,
        center_max_batches: int | None = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        objective: str | None = None,
        soft_boundary: bool = False,
        nu: float = 0.1,
        network_weight_decay: float = 1e-6,
        radius_warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder"])
        self.objective = _objective(objective, soft_boundary)
        self._check_hparams()
        self.encoder = encoder
        self.features = nn.Identity() if features is None else features
        # Empty until the first fit fixes it; loading a state resizes it.
        self.register_buffer("center", torch.empty(0))
        self.register_buffer("radius", torch.tensor(0.0))
        if ckpt:
            self._load_checkpoint(ckpt)

    @property
    def _has_center(self) -> bool:
        return len(self.center) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The representation φ(x; W), shape (batch, latent)."""
        return self.encoder(self.features(x))

    def _encoder_input(self, batch) -> torch.Tensor:
        x = unpack_batch(batch).x
        return torch.flatten(x, start_dim=1) if self._flat_input else x

    def on_fit_start(self) -> None:
        # Before the sanity pass, so that the untrained scores have a centre too.
        if not self._has_center:
            self._fix_center(self.trainer.datamodule.train_dataloader())

    def _fix_center(self, loader) -> None:
        """Set c to the mean eval-mode representation of the training rows (Sec. 3.3).

        Only the first ``center_max_batches`` batches are read, all of them when it is
        None. Iterating ``loader`` draws one permutation from its shuffler, if any.
        """
        was_training = self.training
        self.eval()
        z_sum, n_rows = 0.0, 0
        with torch.no_grad():
            for batch in islice(loader, self.hparams.center_max_batches):
                z = self(self._encoder_input(batch).to(self.device))
                z_sum, n_rows = z_sum + z.sum(dim=0), n_rows + len(z)
        self.train(was_training)
        self.center = _away_from_zero(z_sum / n_rows, self.hparams.center_eps)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        if not self._has_center:
            raise RuntimeError("The centre is unset: fit first, or load a state.")
        z = self(self._encoder_input(batch))
        distance = _squared_distance(z, self.center)
        data_loss = self._data_loss(distance)
        weight_decay = self._weight_decay()
        loss = self._add_hgq_loss(data_loss.mean() + weight_decay)
        ascore = distance.detach()
        return {
            "loss": loss,
            "loss/mean": loss.detach(),
            "loss/distance_raw/mean": ascore.mean(),
            "loss/reg_scaled/mean": weight_decay.detach(),
            **self._summaries(z.detach(), ascore),
            "loss/full": data_loss.detach(),
            "ascore/full": ascore,
            "fit/full": ascore / z.shape[1],
        }

    def _data_loss(self, distance: torch.Tensor) -> torch.Tensor:
        if self.objective == "one_class":
            return distance
        return soft_boundary_loss(distance, self.radius, self.hparams.nu)

    def _weight_decay(self) -> torch.Tensor:
        """Return (λ/2) Σ_l ||W^l||_F², the weight decay of the encoder."""
        squares = sum(p.pow(2).sum() for p in self.encoder.parameters())
        return 0.5 * self.hparams.network_weight_decay * squares

    def _summaries(self, z: torch.Tensor, ascore: torch.Tensor) -> dict:
        q50, q99 = self.compute_score_quantiles(ascore)
        return {
            "ascore/operational": self.compute_operational_ascore(ascore),
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_squared": (z**2).sum(dim=1).mean(),
            "center_norm": self.center.norm(),
            "radius": self.radius.clone(),
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        warm = self.current_epoch >= self.hparams.radius_warmup_epochs
        if self.objective == "soft_boundary" and warm:
            # For fixed W, eq. 3 is minimised at the (1 - ν) quantile of the distances
            # (Prop. 4); the authors' code solves it on each batch after its step.
            radii = outputs["ascore/full"].sqrt()
            self.radius.copy_(torch.quantile(radii, 1.0 - self.hparams.nu))

    def outlog(self, outdict: dict) -> dict:
        return {name: outdict.get(key) for name, key in _LOGGED.items()}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # The parent copies the stored centre in place, into a buffer that must already
        # have its shape and dtype; an unfitted module's centre is empty.
        key = prefix + "center"
        if key in state_dict:
            stored = state_dict[key]
            self.center = stored.new_empty(stored.shape, device=self.center.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)

    def _add_hgq_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Add the quantisation losses that an HGQ encoder collects, if any."""
        extra = list(getattr(self.encoder, "losses", []))
        return loss + torch.stack(extra).sum() if extra else loss

    def _check_hparams(self) -> None:
        for failed, message in _HPARAM_CHECKS:
            if failed(self.hparams):
                raise ValueError(message)
