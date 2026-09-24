"""Is the MI penalty's gradient path alive, or has the sigmoid saturated?

The Bernoulli-MI estimator never sees the latent logits directly. It reads
``sigma(T z)`` and averages that over the batch, so everything it can influence
reaches the encoder through ``d/dz sigma(T z) = T sigma (1 - sigma)``. That
derivative is largest at ``z = 0`` and collapses as ``|z|`` grows: at T = 6 it is
1.5 at the origin, 0.089 at ``|z| = 1``, and 0.0044 at ``|z| = 1.5``.

If training drives the logits well outside the responsive region -- which the
reconstruction objective and the straight-through Bernoulli both encourage, since
a confident code reconstructs better -- then the MI term keeps a large *value*
while contributing essentially no *gradient*. It can be most of the loss and
still steer nothing. That is the failure this module measures.

The reported attenuation is relative to the best case,

    a(z) = [T sigma(Tz)(1 - sigma(Tz))] / (T/4) = 4 sigma(Tz) (1 - sigma(Tz))

so a = 1 at z = 0 and a -> 0 as the unit saturates. a = 0.01 means the encoder
receives a gradient from the MI term one hundred times weaker than a unit sitting
at the sigmoid's midpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np
import torch

#: Below this relative attenuation a unit contributes <1% of the gradient an
#: unsaturated unit would. Not a physical constant -- a reporting threshold.
DEAD_GRADIENT_ATTENUATION = 0.01
LOGIT_HISTOGRAM_RANGE = (-6.0, 6.0)
LOGIT_HISTOGRAM_BINS = 241
#: Quantiles are taken over |z|, not over z. A saturated code is typically
#: symmetric -- half the logits at -5, half at +5 -- whose *signed* median is
#: 0 and would report a perfectly responsive encoder while every unit is dead.
ABS_LOGIT_HISTOGRAM_MAX = 12.0
ABS_LOGIT_HISTOGRAM_BINS = 480


def gradient_attenuation(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Per-element gradient strength relative to a unit at the sigmoid midpoint."""

    probabilities = torch.sigmoid(temperature * logits.to(dtype=torch.float64))
    return 4.0 * probabilities * (1.0 - probabilities)


class LatentSaturationAccumulator:
    """Streaming summary of the logit distribution and the MI gradient path.

    Streaming because the development split is 12.5M events: the histogram and
    the quantile sketch are bounded, the raw logits are never retained.
    """

    def __init__(self, *, temperature: float, latent_width: int) -> None:
        self.temperature = float(temperature)
        self.latent_width = int(latent_width)
        self._counts = np.zeros(LOGIT_HISTOGRAM_BINS, dtype=np.int64)
        self._edges = np.linspace(*LOGIT_HISTOGRAM_RANGE, LOGIT_HISTOGRAM_BINS + 1)
        self._abs_counts = np.zeros(ABS_LOGIT_HISTOGRAM_BINS, dtype=np.int64)
        self._abs_edges = np.linspace(0.0, ABS_LOGIT_HISTOGRAM_MAX, ABS_LOGIT_HISTOGRAM_BINS + 1)
        self._abs_overflow = 0
        self._per_unit_abs_sum = np.zeros(self.latent_width, dtype=np.float64)
        self._per_unit_attenuation_sum = np.zeros(self.latent_width, dtype=np.float64)
        self._attenuation_sum = 0.0
        self._dead = 0
        self._n = 0
        self._abs_max = 0.0

    def update(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.latent_width:
            raise ValueError(
                f"Expected logits of shape [batch, {self.latent_width}], "
                f"got {tuple(logits.shape)}."
            )
        values = logits.detach().to(dtype=torch.float64, device="cpu")
        attenuation = gradient_attenuation(values, temperature=self.temperature)

        flat = values.numpy().reshape(-1)
        self._counts += np.histogram(flat, bins=self._edges)[0]
        magnitudes = np.abs(flat)
        self._abs_counts += np.histogram(magnitudes, bins=self._abs_edges)[0]
        self._abs_overflow += int((magnitudes > ABS_LOGIT_HISTOGRAM_MAX).sum())
        # Values outside the histogram range are still counted in every scalar
        # summary below, so a run that saturates past +/-6 is not hidden by the
        # axis limits.
        self._abs_max = max(self._abs_max, float(np.abs(flat).max()))
        self._per_unit_abs_sum += values.abs().numpy().sum(axis=0)
        self._per_unit_attenuation_sum += attenuation.numpy().sum(axis=0)
        self._attenuation_sum += float(attenuation.sum())
        self._dead += int((attenuation < DEAD_GRADIENT_ATTENUATION).sum())
        self._n += int(values.numel())

    def summary(self) -> dict[str, Any]:
        if self._n == 0:
            raise RuntimeError("No logits were accumulated.")
        within = self._counts.sum()

        def quantile(q: float) -> float:
            """Quantile of |z|, linearly interpolated inside the containing bin.

            Interpolated rather than snapped to a bin centre because the answer is
            compared against thresholds (|z| ~ 0.7, ~1.5) that are only a few bin
            widths apart, and because an exact tie on a cumulative boundary is a
            floating-point coin toss when the bins are read off a step function.
            """
            target = q * self._n
            cumulative = np.cumsum(self._abs_counts)
            if target > cumulative[-1]:
                # Inside the overflow tail. Reporting the range maximum keeps the
                # artifact strict JSON; abs_logit_overflow_fraction and
                # abs_logit_max carry the exact story.
                return float(ABS_LOGIT_HISTOGRAM_MAX)
            index = int(np.searchsorted(cumulative, target, side="left"))
            index = min(index, len(self._abs_counts) - 1)
            below = cumulative[index - 1] if index else 0
            count = self._abs_counts[index]
            position = 0.0 if count == 0 else (target - below) / count
            low, high = self._abs_edges[index], self._abs_edges[index + 1]
            return float(low + position * (high - low))

        events = self._n / self.latent_width
        return {
            "schema_version": 1,
            "temperature": self.temperature,
            "latent_width": self.latent_width,
            "n_events": int(events),
            "n_logits": int(self._n),
            "dead_gradient_attenuation_threshold": DEAD_GRADIENT_ATTENUATION,
            "mean_gradient_attenuation": self._attenuation_sum / self._n,
            "fraction_below_threshold": self._dead / self._n,
            "abs_logit_max": self._abs_max,
            "abs_logit_quantiles": {
                "p01": quantile(0.01), "p05": quantile(0.05), "p25": quantile(0.25),
                "p50": quantile(0.50), "p75": quantile(0.75), "p95": quantile(0.95),
                "p99": quantile(0.99),
            },
            "per_unit": [
                {
                    "index": index,
                    "mean_abs_logit": float(self._per_unit_abs_sum[index] / events),
                    "mean_gradient_attenuation": float(
                        self._per_unit_attenuation_sum[index] / events
                    ),
                }
                for index in range(self.latent_width)
            ],
            "abs_logit_overflow_fraction": self._abs_overflow / self._n,
            "histogram": {
                "edges": [float(v) for v in self._edges],
                "counts": [int(v) for v in self._counts],
                "fraction_outside_range": float(1.0 - within / self._n),
            },
        }


def write_latent_saturation(summary: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, Path]:
    """Persist the summary and draw the logit distribution against the gradient."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.pareto_plots import (
        BASELINE_MARK, GRID_INK, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "latent_saturation.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    edges = np.asarray(summary["histogram"]["edges"], dtype=float)
    counts = np.asarray(summary["histogram"]["counts"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    temperature = float(summary["temperature"])

    figure, axis = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.fill_between(
        centres, counts / max(counts.sum(), 1.0), color="#2a78d6", alpha=0.85, lw=0,
        label="latent logits $z$", zorder=2,
    )
    axis.set_xlabel("latent logit $z$", color=TEXT_PRIMARY)
    axis.set_ylabel("fraction of logits", color=TEXT_PRIMARY)
    axis.grid(True, color=GRID_INK, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(GRID_INK)
    axis.tick_params(colors=TEXT_SECONDARY)

    gradient = axis.twinx()
    grid = np.linspace(edges[0], edges[-1], 600)
    attenuation = gradient_attenuation(torch.from_numpy(grid), temperature=temperature).numpy()
    gradient.plot(grid, attenuation, color=BASELINE_MARK, linewidth=2.0, zorder=3,
                  label="MI gradient strength, relative")
    gradient.set_ylabel("relative MI gradient  $4\\sigma(Tz)(1-\\sigma(Tz))$", color=BASELINE_MARK)
    gradient.tick_params(colors=BASELINE_MARK)
    gradient.set_ylim(0, 1.05)
    for spine in gradient.spines.values():
        spine.set_visible(False)

    handles = axis.get_legend_handles_labels()[0] + gradient.get_legend_handles_labels()[0]
    labels = axis.get_legend_handles_labels()[1] + gradient.get_legend_handles_labels()[1]
    axis.legend(handles, labels, frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9, loc="upper left")
    axis.set_title(
        f"{100*summary['fraction_below_threshold']:.1f}% of logits receive under "
        f"{100*DEAD_GRADIENT_ATTENUATION:.0f}% of the available MI gradient",
        color=TEXT_PRIMARY, fontsize=12,
    )
    plot_path = destination / "latent_saturation.png"
    figure.savefig(plot_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return {"json": json_path, "png": plot_path}
