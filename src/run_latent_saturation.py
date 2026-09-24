"""Diagnostic: is the MI penalty's gradient path alive in a trained checkpoint?

Reads ``<checkpoints_dir>/<experiment_name>/<run_name>/loss_total.ckpt`` and the
development split, encodes it, and summarises the latent-logit distribution
against the sigmoid derivative the MI estimator depends on. Writes

    <run>/plots/val/loss_total/saturation/latent_saturation.json
    <run>/plots/val/loss_total/saturation/latent_saturation.png

It reuses the stage machinery so it composes exactly the config the run was
trained with and loads the checkpoint strictly; a mismatched architecture fails
loudly rather than silently measuring a different model.

Motivation: across the 96 runs of Pareto_Front_092026 the logged train/loss_mi is
lowest at epoch 0, returns to the unpenalised level by epoch 1, and ends *higher*
for gamma = 0.25 (0.294) than for gamma = 0 (0.252) -- while gamma*MI is 84% of
the total loss. Multiplying gamma by five leaves the final reconstruction loss
unchanged to five decimals. A term that large which changes nothing is the
signature of a dead gradient, and this script measures that directly.

    python3 src/run_latent_saturation.py \\
        experiment=physics/pareto_fet_train \\
        run_name=Seed123_Gamma_0.25_Bins_50_architecture_h64_32_Run01 \\
        experiment_name=Pareto_Front_092026
"""

from typing import Any, Dict, Optional, Tuple

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import hydra
import torch

from omegaconf import DictConfig
from colorama import Back

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers

register_resolvers()

from src.evaluation.latent_saturation import (
    LatentSaturationAccumulator,
    write_latent_saturation,
)
from src.utils import RankedLogger, extras
from src.utils.instrumentation import log_phase
from src.data.utils import unpack_batch
from src.utils.stage import build_stage_context

log = RankedLogger(__name__, rank_zero_only=True)

#: Saturation is a property of the representation, not of rare events; a few
#: million logits pin every number in the summary far beyond the precision that
#: matters. Capping keeps the diagnostic to a couple of minutes.
DEFAULT_MAX_BATCHES = 64


def _restore_checkpoint(model, checkpoint_path) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no state_dict.")
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    finally:
        del checkpoint
    log.info("Frozen checkpoint restored.")


def run_latent_saturation(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    context = build_stage_context(
        cfg,
        stage_name="saturation",
        strict_manifest=bool(cfg.get("manifest_strict", True)),
    )
    model = context.algorithm
    _restore_checkpoint(model, context.run_ckpts / "loss_total.ckpt")
    model.eval()

    temperature = float(cfg.algorithm.get("mi_temperature", 6.0))
    max_batches = int(cfg.get("saturation_max_batches", DEFAULT_MAX_BATCHES))

    log.info(
        Back.MAGENTA + 8 * "-" + "LATENT SATURATION DIAGNOSTIC" + 8 * "-"
    )
    with log_phase("latent saturation"):
        context.datamodule.setup_probe_split("train")
        try:
            accumulator: Optional[LatentSaturationAccumulator] = None
            with torch.no_grad():
                for index, batch in enumerate(context.datamodule.probe_dataloader()):
                    if index >= max_batches:
                        break
                    view = unpack_batch(batch)
                    model_input = torch.flatten(view.x, start_dim=1)
                    logits = model.forward_with_representations(model_input)["latent_logits"]
                    logits = torch.flatten(logits, start_dim=1)
                    if accumulator is None:
                        accumulator = LatentSaturationAccumulator(
                            temperature=temperature,
                            latent_width=int(logits.shape[1]),
                        )
                    accumulator.update(logits)
            if accumulator is None:
                raise RuntimeError("The development split yielded no batches.")
            summary = accumulator.summary()
        finally:
            context.datamodule.release_probe_split()

    summary["gamma"] = float(cfg.algorithm.get("mi_gamma"))
    summary["run_name"] = str(cfg.get("run_name"))
    written = write_latent_saturation(
        summary, output_dir=context.run_ckpts / "plots/val/loss_total/saturation"
    )

    log.info(
        f"[saturation] gamma={summary['gamma']}  "
        f"median |z|={summary['abs_logit_quantiles']['p50']:.3f}  "
        f"max |z|={summary['abs_logit_max']:.2f}  "
        f"mean MI-gradient strength={summary['mean_gradient_attenuation']:.4f}  "
        f"below {summary['dead_gradient_attenuation_threshold']:.0%}: "
        f"{summary['fraction_below_threshold']:.1%}"
    )
    log.info(f"[saturation] Wrote {written['json']} and {written['png']}.")
    return {"latent_saturation": summary}, {"paths": written}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if "run_name" in cfg and not isinstance(cfg.run_name, str):
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.run_name = str(cfg.run_name)

    extras(cfg)
    run_latent_saturation(cfg)


if __name__ == "__main__":
    main()
