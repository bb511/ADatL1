"""NeuTraL AD: learned transformations under a deterministic contrastive loss.

Qiu et al., "Neural Transformation Learning for Deep Anomaly Detection Beyond Images",
ICML 2021, arXiv:2103.16440.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import ADLightningModule
from src.data.utils import unpack_batch

TRANSFORM_TYPES = ("residual", "multiplicative")
# Hidden width of every mask network M_k, DeepOD's tabular default
# (NeuTraL(trans_hidden_dims=50, act="LeakyReLU", bias=False)).
TRANSFORM_HIDDEN = 50

# (failed, message) pairs over the saved hyperparameters, checked in this order.
_HPARAM_CHECKS = (
    (
        lambda hp: hp.n_transforms < 2,
        "n_transforms must be at least 2, since the loss is divided by K log K.",
    ),
    (lambda hp: hp.temperature <= 0.0, "temperature must be positive."),
    (
        lambda hp: hp.transform_type not in TRANSFORM_TYPES,
        f"transform_type must be one of {TRANSFORM_TYPES}.",
    ),
)


def _linear_weight(shape: tuple[int, ...], fan_in: int) -> nn.Parameter:
    """A weight drawn from nn.Linear's default distribution, U(+-1/sqrt(fan_in))."""
    bound = 1.0 / math.sqrt(fan_in)
    return nn.Parameter(torch.empty(shape).uniform_(-bound, bound))


class _Transformations(nn.Module):
    """The K transformations T_k, batched as one network with K heads.

    Each mask network M_k is a bias-free Linear(in_dim, 50), LeakyReLU, Linear(50,
    in_dim). The residual type is T_k(x) = M_k(x) + x, the multiplicative type T_k(x) =
    sigmoid(M_k(x)) * x (Qiu et al., Sec. 4).
    """

    def __init__(self, in_dim: int, n_transforms: int, transform_type: str):
        super().__init__()
        self.transform_type = transform_type
        self.w_in = _linear_weight((n_transforms, in_dim, TRANSFORM_HIDDEN), in_dim)
        self.w_out = _linear_weight(
            (n_transforms, TRANSFORM_HIDDEN, in_dim), TRANSFORM_HIDDEN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map events of shape (batch, in_dim) to views of shape (batch, K, in_dim)."""
        hidden = F.leaky_relu(torch.einsum("bi,kih->bkh", x, self.w_in))
        masks = torch.einsum("bkh,khi->bki", hidden, self.w_out)
        if self.transform_type == "residual":
            return masks + x.unsqueeze(1)
        return torch.sigmoid(masks) * x.unsqueeze(1)


def _contrastive_loss(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per-event DCL (Qiu et al., eq. 2) divided by K log K.

    ``z`` holds the embeddings, shape (batch, K + 1, embedding), view 0 being the
    untransformed event; the result has shape (batch,). Identical views give K log K
    (Qiu et al., Prop. 3), so the normalised loss is 1 at chance for every K.
    """
    z = F.normalize(z, dim=-1)
    # log h(x_k, x_l) = cos(f(x_k), f(x_l)) / tau over every pair of views.
    log_h = z @ z.transpose(1, 2) / temperature
    n_views = z.shape[1]
    self_pairs = torch.eye(n_views, dtype=torch.bool, device=z.device)
    log_h = log_h.masked_fill(self_pairs, float("-inf"))[:, 1:]
    per_view = torch.logsumexp(log_h, dim=-1) - log_h[..., 0]
    n_transforms = n_views - 1
    return per_view.sum(dim=1) / (n_transforms * math.log(n_transforms))


class NeuTraLAD(ADLightningModule):
    """NeuTraL AD on flattened inputs, scored by its normalised training loss.

    The anomaly score and the per-event training loss are the same deterministic
    quantity, the DCL divided by K log K. When a batch carries a data mask, every view
    keeps the input's value on the padded entries. ``in_dim`` is the input width, which
    the transformations need; ``target_rate`` is the FPR when ``base_rate`` is None,
    else a rate that ``base_rate`` divides.
    """

    def __init__(
        self,
        encoder: nn.Module,
        in_dim: int,
        n_transforms: int = 11,
        transform_type: str = "residual",
        temperature: float = 0.1,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "encoder"])
        self._check_hparams()
        self.encoder = encoder
        self.transforms = _Transformations(in_dim, n_transforms, transform_type)
        self.temperature = float(temperature)

    def views(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The event and its K transformations, shape (batch, K + 1, in_dim).

        ``present`` is the boolean data mask flattened like ``x``.
        """
        transformed = self.transforms(x)
        if present is not None:
            transformed = torch.where(present.unsqueeze(1), transformed, x.unsqueeze(1))
        return torch.cat([x.unsqueeze(1), transformed], dim=1)

    def forward(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalised DCL of every event, shape (batch,)."""
        views = self.views(x, present)
        z = self.encoder(views.flatten(0, 1)).unflatten(0, views.shape[:2])
        return _contrastive_loss(z, self.temperature)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        """Score a batch; the per-event training loss is the anomaly score."""
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        present = None if b.mask is None else torch.flatten(b.mask, start_dim=1).bool()
        loss_full = self.forward(x, present)
        ascore = loss_full.detach()
        q50, q99 = self.compute_score_quantiles(ascore)
        return {
            "loss": loss_full.mean(),
            "loss/mean": ascore.mean(),
            "ascore/operational": self.compute_operational_ascore(ascore),
            "ascore/q50": q50,
            "ascore/q99": q99,
            "loss/full": ascore,
            "ascore/full": ascore,
            "fit/full": ascore,
        }

    def outlog(self, outdict: dict) -> dict:
        keys = ("loss", "loss/mean", "ascore/operational", "ascore/q50", "ascore/q99")
        return {key.replace("/", "_"): outdict.get(key) for key in keys}

    def _check_hparams(self) -> None:
        for failed, message in _HPARAM_CHECKS:
            if failed(self.hparams):
                raise ValueError(message)
