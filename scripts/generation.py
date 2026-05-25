#!/usr/bin/env python3
"""Generate reproducible launch scripts for paper experiments.

The generated shell scripts are intentionally boring: they are a durable record of
the Hydra command line, while this Python file owns the typed experiment matrix and
the manifest that distinguishes tuned parameters from fixed settings and reporting
factors.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "generated"
TRAIN_ENTRYPOINT = "src/train.py"


class Dataset(str, Enum):
    PHYSICS = "physics"
    CIFAR10 = "cifar10"
    ROBUSTAD = "robustad"
    CCHAMBER = "cchamber"


class Model(str, Enum):
    AE = "ae"
    VAE = "vae"
    DSAE = "dsae"
    DSVAE = "dsvae"
    SVDD = "svdd"
    REALNVP = "realnvp"


class Strategy(str, Enum):
    SEMI_CVAR25 = "semi_cvar25"
    SEMI_CVAR10 = "semi_cvar10"
    CAP = "cap"
    CAP_METADATA_NEAREST = "cap_metadata_nearest"
    CAP_ENCODER_NEAREST = "cap_encoder_nearest"
    CAP_RANDOM = "cap_random"
    DRIFT = "drift"
    WASSERSTEIN = "wasserstein"


class Stage(str, Enum):
    SWEEP = "sweep"
    RETRAIN = "retrain"
    EVALUATE = "evaluate"
    ALL = "all"


class Launcher(str, Enum):
    NONE = "none"
    SUBMITIT_LOCAL = "submitit_local"
    SUBMITIT_SLURM_CLARIDEN = "submitit_slurm_clariden"


@dataclass(frozen=True)
class ExperimentSpecification:
    """Typed paper experiment specification.

    Attributes:
        name: Stable identifier used for generated folder names and study names.
        dataset: Dataset family used by the experiment.
        model: Model family used by the experiment.
        strategy: Validation/model-selection strategy.
        experiment: Hydra experiment config path.
        hparams_search: Hydra hparams_search config path.
        objective_name: Primary objective short name used in Optuna study names.
        objective_direction: Direction for the primary objective.
        secondary_objective_name: Model-native secondary objective short name.
        secondary_direction: Direction for the secondary objective.
        n_trials: Number of Optuna trials.
        sweep_epochs: Number of epochs per sweep trial.
        retrain_epochs: Number of epochs for selected retraining.
        seeds: Random seeds treated as reporting/statistical factors.
        fixed_overrides: Fixed Hydra overrides for the experiment.
        strategy_overrides: Strategy-specific Hydra overrides.
        sweeper_overrides: Optuna/Hydra sweeper overrides used only in sweeps.
        disabled_overrides: Overrides that disable unused callbacks/checkpoints.
        factors: Non-tuned factors reported or averaged over.
        notes: Human-readable details that should appear in manifests.
    """

    name: str
    dataset: Dataset
    model: Model
    strategy: Strategy
    experiment: str
    hparams_search: str
    objective_name: str
    objective_direction: str
    secondary_objective_name: str
    secondary_direction: str = "minimize"
    n_trials: int = 600
    sweep_epochs: int = 50
    retrain_epochs: int = 200
    seeds: tuple[int, ...] = (123,)
    fixed_overrides: tuple[str, ...] = ()
    strategy_overrides: tuple[str, ...] = ()
    sweeper_overrides: tuple[str, ...] = ()
    disabled_overrides: tuple[str, ...] = ()
    factors: Mapping[str, tuple[str, ...] | tuple[int, ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ExperimentSpecification.name must not be empty.")
        if self.n_trials <= 0:
            raise ValueError(f"{self.name}: n_trials must be positive.")
        if self.sweep_epochs <= 0:
            raise ValueError(f"{self.name}: sweep_epochs must be positive.")
        if self.retrain_epochs < self.sweep_epochs:
            raise ValueError(f"{self.name}: retrain_epochs must be >= sweep_epochs.")
        if not self.seeds:
            raise ValueError(f"{self.name}: at least one seed must be provided.")
        if self.model in {Model.DSAE, Model.DSVAE} and self.dataset != Dataset.PHYSICS:
            raise ValueError(f"{self.name}: {self.model.value} is physics-only.")
        if self.strategy in CCHAMBER_CAP_PAIRING_STRATEGIES and self.dataset != Dataset.CCHAMBER:
            raise ValueError(f"{self.name}: {self.strategy.value} is Causal Chamber-only.")
        if self.strategy in AGNOSTIC_STRATEGIES and not self.experiment.endswith("_agnostic"):
            raise ValueError(f"{self.name}: agnostic strategy requires an *_agnostic experiment.")
        if self.strategy in SEMI_STRATEGIES and self.experiment.endswith("_agnostic"):
            raise ValueError(f"{self.name}: semi-supervised strategy must not use *_agnostic.")

    @property
    def tuned_param_source(self) -> str:
        return f"configs/hparams_search/{self.hparams_search}.yaml"


MODEL_CONFIG = {
    Dataset.PHYSICS: {
        Model.AE: ("ae", "ae_optuna", "mse"),
        Model.VAE: ("vae", "vae_optuna", "kl"),
        Model.DSAE: ("dsae", "dsae_optuna", "mse"),
        Model.DSVAE: ("dsvae", "dsvae_optuna", "kl"),
        Model.SVDD: ("svdd", "svdd_optuna", "dist"),
        Model.REALNVP: ("realnvp", "realnvp_optuna", "logp"),
    },
    Dataset.CIFAR10: {
        Model.AE: ("ae", "imageae_optuna", "mse"),
        Model.VAE: ("vae", "imagevae_optuna", "kl"),
        Model.SVDD: ("svdd", "imagesvdd_optuna", "dist"),
        Model.REALNVP: ("realnvp", "imagerealnvp_optuna", "logp"),
    },
    Dataset.ROBUSTAD: {
        Model.AE: ("ae", "imageae_optuna", "mse"),
        Model.VAE: ("vae", "imagevae_optuna", "kl"),
        Model.SVDD: ("svdd", "imagesvdd_optuna", "dist"),
        Model.REALNVP: ("realnvp", "imagerealnvp_optuna", "logp"),
    },
    Dataset.CCHAMBER: {
        Model.AE: ("ae", "ae_optuna", "mse"),
        Model.VAE: ("vae", "vae_optuna", "kl"),
        Model.SVDD: ("svdd", "svdd_optuna", "dist"),
        Model.REALNVP: ("realnvp", "realnvp_optuna", "logp"),
    },
}

CCHAMBER_CAP_PAIRING_STRATEGIES = {
    Strategy.CAP_METADATA_NEAREST,
    Strategy.CAP_ENCODER_NEAREST,
    Strategy.CAP_RANDOM,
}
CAP_STRATEGIES = {Strategy.CAP, *CCHAMBER_CAP_PAIRING_STRATEGIES}
AGNOSTIC_STRATEGIES = CAP_STRATEGIES | {Strategy.DRIFT, Strategy.WASSERSTEIN}
SEMI_STRATEGIES = {Strategy.SEMI_CVAR25, Strategy.SEMI_CVAR10}
DEFAULT_PAPER_STRATEGIES = (
    Strategy.SEMI_CVAR25,
    Strategy.CAP,
    Strategy.DRIFT,
    Strategy.WASSERSTEIN,
)
ALL_STRATEGIES = DEFAULT_PAPER_STRATEGIES + (Strategy.SEMI_CVAR10,)


def build_paper_experiments(
    *,
    n_trials: int = 600,
    seeds: Sequence[int] = (123,),
    include_cvar10: bool = True,
) -> dict[str, ExperimentSpecification]:
    """Build the default paper experiment matrix."""
    specs: dict[str, ExperimentSpecification] = {}
    for dataset, model_cfg in MODEL_CONFIG.items():
        for model in model_cfg:
            for strategy in strategies_for(dataset, include_cvar10=include_cvar10):
                spec = make_experiment_specification(
                    dataset=dataset,
                    model=model,
                    strategy=strategy,
                    n_trials=n_trials,
                    seeds=tuple(int(s) for s in seeds),
                )
                specs[spec.name] = spec
    return specs


def strategies_for(dataset: Dataset, *, include_cvar10: bool) -> tuple[Strategy, ...]:
    """Return strategies supported by a dataset family in the paper matrix."""
    if dataset == Dataset.CCHAMBER:
        return (
            Strategy.CAP_METADATA_NEAREST,
            Strategy.CAP_ENCODER_NEAREST,
            Strategy.CAP_RANDOM,
            Strategy.DRIFT,
            Strategy.WASSERSTEIN,
        )
    return ALL_STRATEGIES if include_cvar10 else DEFAULT_PAPER_STRATEGIES


def make_experiment_specification(
    *,
    dataset: Dataset,
    model: Model,
    strategy: Strategy,
    n_trials: int = 600,
    seeds: tuple[int, ...] = (123,),
) -> ExperimentSpecification:
    """Create one paper experiment specification from the compact matrix."""
    if model not in MODEL_CONFIG[dataset]:
        raise ValueError(f"{model.value} is not configured for {dataset.value}.")

    experiment_stem, hparams_search, secondary = MODEL_CONFIG[dataset][model]
    agnostic = strategy in AGNOSTIC_STRATEGIES
    experiment = f"{dataset.value}/{experiment_stem}{'_agnostic' if agnostic else ''}"
    fixed = fixed_overrides_for(dataset)
    factors = factors_for(dataset)

    if strategy in SEMI_STRATEGIES:
        objective = "cvar25eff" if strategy == Strategy.SEMI_CVAR25 else "cvar10eff"
        direction = "maximize"
    elif strategy in CAP_STRATEGIES:
        objective = "cap"
        direction = "maximize"
    elif strategy == Strategy.DRIFT:
        objective = "drift"
        direction = "minimize"
    elif strategy == Strategy.WASSERSTEIN:
        objective = "wasserstein"
        direction = "minimize"
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    spec = ExperimentSpecification(
        name=f"{dataset.value}_{model.value}_{strategy.value}",
        dataset=dataset,
        model=model,
        strategy=strategy,
        experiment=experiment,
        hparams_search=hparams_search,
        objective_name=objective,
        objective_direction=direction,
        secondary_objective_name=secondary,
        n_trials=n_trials,
        seeds=seeds,
        fixed_overrides=fixed,
        strategy_overrides=strategy_overrides_for(strategy),
        sweeper_overrides=sweeper_overrides_for(strategy, secondary),
        disabled_overrides=disabled_overrides_for(dataset, strategy),
        factors=factors,
        notes=notes_for(dataset, strategy),
    )
    spec.validate()
    return spec


def fixed_overrides_for(dataset: Dataset) -> tuple[str, ...]:
    if dataset == Dataset.PHYSICS:
        return (
            "data=basis",
            "data.batch_size=16384",
            "data.max_val_batches=10",
            "algorithm.target_rate=0.25",
            "algorithm.base_rate=28608.8064",
        )
    if dataset == Dataset.CIFAR10:
        return (
            "data=cifar10",
            "data.batch_size=256",
            "data.max_val_batches=-1",
            "algorithm.target_rate=0.01",
            "algorithm.base_rate=null",
        )
    if dataset == Dataset.ROBUSTAD:
        return (
            "data=robustad",
            "data.subset=pcb",
            "data.batch_size=16",
            "data.max_val_batches=-1",
            "algorithm.target_rate=0.1",
            "algorithm.base_rate=null",
        )
    if dataset == Dataset.CCHAMBER:
        return (
            "data=causal_chamber",
            "data.batch_size=512",
            "data.max_val_batches=-1",
            "algorithm.target_rate=0.01",
            "algorithm.base_rate=null",
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def reference_dataset_for(dataset: Dataset) -> str:
    if dataset == Dataset.PHYSICS:
        return "SingleNeutrino_E-10-gun"
    if dataset == Dataset.CIFAR10:
        return "reference_normal"
    if dataset == Dataset.ROBUSTAD:
        return "shifted_normal_all"
    if dataset == Dataset.CCHAMBER:
        return "reference_normal"
    raise ValueError(f"Unknown dataset: {dataset}")


def factors_for(dataset: Dataset) -> dict[str, tuple[str, ...]]:
    if dataset == Dataset.PHYSICS:
        return {
            "validation_domains": ("normal", reference_dataset_for(dataset)),
            "reported_over": ("signal_dataset", "seed"),
            "operating_point": ("250Hz",),
        }
    if dataset == Dataset.CIFAR10:
        return {
            "normal_classes": ("0",),
            "signal_classes": tuple(str(i) for i in range(1, 10)),
            "validation_domains": ("normal", reference_dataset_for(dataset)),
            "reported_over": ("class", "seed"),
        }
    if dataset == Dataset.ROBUSTAD:
        return {
            "subsets": ("pcb",),
            "validation_domains": ("normal", reference_dataset_for(dataset)),
            "reported_over": ("shifted_anomaly_domain", "seed"),
        }
    if dataset == Dataset.CCHAMBER:
        return {
            "validation_domains": ("normal", reference_dataset_for(dataset)),
            "reported_over": ("intervention_dataset", "seed"),
            "cap_pairing": ("metadata_nearest", "encoder_nearest", "random"),
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def notes_for(dataset: Dataset, strategy: Strategy) -> tuple[str, ...]:
    notes: list[str] = []
    if dataset in {Dataset.CIFAR10, Dataset.ROBUSTAD}:
        notes.append(
            "The image hparams_search files currently contain active RobustAD search "
            "spaces and commented CIFAR-10 alternatives; this script records the "
            "active Hydra search config and applies dataset-specific storage overrides."
        )
    if dataset == Dataset.CCHAMBER:
        if strategy == Strategy.CAP_METADATA_NEAREST:
            notes.append(
                "CAP uses real normal/reference_normal rows ordered by datamodule "
                "metadata-nearest matching; callback pairing_type=none consumes that order."
            )
        elif strategy == Strategy.CAP_RANDOM:
            notes.append(
                "CAP uses real normal/reference_normal rows ordered by deterministic "
                "random datamodule pairing; callback pairing_type=none consumes that order."
            )
        elif strategy == Strategy.CAP_ENCODER_NEAREST:
            notes.append(
                "CAP uses frozen-encoder fixed pair tables via pairing_type=precomputed. "
                "Set CCHAMBER_VALID_PAIR_TABLE and CCHAMBER_TEST_PAIR_TABLE before running."
            )
        else:
            notes.append(
                "Causal Chamber validation uses real paired normal/reference_normal rows "
                "constructed by the datamodule."
            )
    if strategy == Strategy.SEMI_CVAR10:
        notes.append("Semi-supervised CVaR10 is an appendix/sensitivity strategy.")
    return tuple(notes)


def strategy_overrides_for(strategy: Strategy) -> tuple[str, ...]:
    if strategy == Strategy.SEMI_CVAR25:
        return ()
    if strategy == Strategy.SEMI_CVAR10:
        return ("evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10",)
    if strategy in CAP_STRATEGIES:
        overrides = [
            "optimized_metric_config.main_metric.callback.name=cap",
            "optimized_metric_config.main_metric.direction=maximize",
        ]
        if strategy == Strategy.CAP_METADATA_NEAREST:
            overrides.append("data.pairing_strategy=metadata_nearest")
        elif strategy == Strategy.CAP_RANDOM:
            overrides.append("data.pairing_strategy=random")
        elif strategy == Strategy.CAP_ENCODER_NEAREST:
            overrides.extend(
                [
                    "data.pairing_strategy=random",
                    "callbacks.cap_ref.pairing_type=precomputed",
                    "callbacks.cap_ref.pairing_index_path=$CCHAMBER_VALID_PAIR_TABLE",
                    "evaluation.callbacks.cap_ref.pairing_type=precomputed",
                    "evaluation.callbacks.cap_ref.pairing_index_path=$CCHAMBER_TEST_PAIR_TABLE",
                ]
            )
        return tuple(overrides)
    if strategy == Strategy.DRIFT:
        return (
            "optimized_metric_config.main_metric.callback.name=thres_drift",
            "optimized_metric_config.main_metric.direction=minimize",
        )
    if strategy == Strategy.WASSERSTEIN:
        return (
            "optimized_metric_config.main_metric.callback.name=wasserstein",
            "optimized_metric_config.main_metric.direction=minimize",
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def sweeper_overrides_for(strategy: Strategy, secondary: str) -> tuple[str, ...]:
    if strategy == Strategy.SEMI_CVAR25:
        return (
            f"hydra.sweeper.study_name=cvar25eff_vs_{secondary}",
            "hydra.sweeper.direction=[maximize,minimize]",
        )
    if strategy == Strategy.SEMI_CVAR10:
        return (
            f"hydra.sweeper.study_name=cvar10eff_vs_{secondary}",
            "hydra.sweeper.direction=[maximize,minimize]",
        )
    if strategy in CCHAMBER_CAP_PAIRING_STRATEGIES:
        return (
            f"hydra.sweeper.study_name={strategy.value}_vs_{secondary}",
            "hydra.sweeper.direction=[maximize,minimize]",
        )
    if strategy == Strategy.CAP:
        return (
            f"hydra.sweeper.study_name=cap_vs_{secondary}",
            "hydra.sweeper.direction=[maximize,minimize]",
        )
    if strategy == Strategy.DRIFT:
        return (
            f"hydra.sweeper.study_name=drift_vs_{secondary}",
            "hydra.sweeper.direction=[minimize,minimize]",
        )
    if strategy == Strategy.WASSERSTEIN:
        return (
            f"hydra.sweeper.study_name=wasserstein_vs_{secondary}",
            "hydra.sweeper.direction=[minimize,minimize]",
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def disabled_overrides_for(dataset: Dataset, strategy: Strategy) -> tuple[str, ...]:
    ref = reference_dataset_for(dataset)
    cap_key = f"cap_ema_normal_vs_{ref}"
    w1_key = f"w1dist_ema_normal_vs_{ref}"
    cap_callback_key = "cap_ref" if dataset == Dataset.CCHAMBER else "cap_sn_zb"
    cap_ckpt_key = "cap_ref_ema_ckpt" if dataset == Dataset.CCHAMBER else "cap_sn_zb_ema_ckpt"
    reco_override = ("evaluation.callbacks.reco=null",) if dataset == Dataset.PHYSICS else ()

    if strategy == Strategy.SEMI_CVAR25:
        return (
            "callbacks.max_rate_ckpt=null",
            "callbacks.cvar10_ema_ckpt=null",
            "~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational",
            "~evaluation.evaluator.ckpts.summary.cvar10_ema",
            "evaluation.callbacks.thres_drift=null",
            "evaluation.callbacks.wasserstein=null",
            *reco_override,
        )
    if strategy == Strategy.SEMI_CVAR10:
        return (
            "callbacks.max_rate_ckpt=null",
            "callbacks.cvar25_ema_ckpt=null",
            "~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational",
            "~evaluation.evaluator.ckpts.summary.cvar25_ema",
            "evaluation.callbacks.thres_drift=null",
            "evaluation.callbacks.wasserstein=null",
            *reco_override,
        )
    if strategy in CAP_STRATEGIES:
        return (
            "callbacks.anomaly_eff=null",
            "callbacks.thres_drift=null",
            "callbacks.wasserstein_dist=null",
            "callbacks.thres_drift_ema_ckpt=null",
            "callbacks.wasserstein_dist_ema_ckpt=null",
            "~evaluation.evaluator.ckpts.summary.operational_drift_ema",
            f"~evaluation.evaluator.ckpts.summary.{w1_key}",
            "evaluation.callbacks.anomaly_efficiency=null",
            "evaluation.callbacks.thres_drift=null",
            "evaluation.callbacks.wasserstein=null",
            *reco_override,
        )
    if strategy == Strategy.DRIFT:
        return (
            "callbacks.anomaly_eff=null",
            f"callbacks.{cap_callback_key}=null",
            "callbacks.wasserstein_dist=null",
            "callbacks.wasserstein_dist_ema_ckpt=null",
            f"callbacks.{cap_ckpt_key}=null",
            f"~evaluation.evaluator.ckpts.summary.{w1_key}",
            f"~evaluation.evaluator.ckpts.summary.{cap_key}",
            "evaluation.callbacks.anomaly_efficiency=null",
            f"evaluation.callbacks.{cap_callback_key}=null",
            "evaluation.callbacks.wasserstein=null",
            *reco_override,
        )
    if strategy == Strategy.WASSERSTEIN:
        return (
            "callbacks.anomaly_eff=null",
            f"callbacks.{cap_callback_key}=null",
            "callbacks.thres_drift=null",
            "callbacks.thres_drift_ema_ckpt=null",
            f"callbacks.{cap_ckpt_key}=null",
            "~evaluation.evaluator.ckpts.summary.operational_drift_ema",
            f"~evaluation.evaluator.ckpts.summary.{cap_key}",
            "evaluation.callbacks.anomaly_efficiency=null",
            f"evaluation.callbacks.{cap_callback_key}=null",
            "evaluation.callbacks.thres_drift=null",
            *reco_override,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def infer_tuned_params(hparams_search: str) -> tuple[str, ...]:
    """Infer tuned parameter paths from the active Hydra Optuna search config."""
    fpath = REPO_ROOT / "configs" / "hparams_search" / f"{hparams_search}.yaml"
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing hparams_search config: {fpath}")
    with fpath.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    params = cfg.get("hydra", {}).get("sweeper", {}).get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"{fpath}: hydra.sweeper.params must be a mapping.")
    return tuple(str(k) for k in params.keys())


def read_search_space(hparams_search: str) -> dict[str, Any]:
    fpath = REPO_ROOT / "configs" / "hparams_search" / f"{hparams_search}.yaml"
    with fpath.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    params = cfg.get("hydra", {}).get("sweeper", {}).get("params", {})
    if not isinstance(params, Mapping):
        return {}
    return {str(k): v for k, v in params.items()}


def build_sweep_overrides(
    spec: ExperimentSpecification,
    *,
    seed: int,
    launcher: Launcher,
    trainer: str,
    devices: str,
    cpus_per_task: int,
    gpus_per_node: int,
    timeout_min: int | None,
) -> list[str]:
    overrides = [
        f"experiment={spec.experiment}",
        f"experiment_name={spec.name}_search",
        f"hparams_search={spec.hparams_search}",
        f"hydra.sweeper.n_trials={spec.n_trials}",
        f"trainer.max_epochs={spec.sweep_epochs}",
        f"seed={seed}",
        "logger=none",
        f"trainer={trainer}",
        f"trainer.devices={devices}",
        storage_override_for(spec),
    ]
    if spec.dataset == Dataset.PHYSICS:
        overrides.append('paths.raw_data_dir="${RAW_DATA_DIR}"')
    if launcher != Launcher.NONE:
        overrides.extend(
            [
                launcher_override(launcher),
                f"hydra.launcher.cpus_per_task={cpus_per_task}",
                f"hydra.launcher.gpus_per_node={gpus_per_node}",
            ]
        )
        if timeout_min is not None:
            overrides.append(f"hydra.launcher.timeout_min={timeout_min}")

    overrides.extend(spec.fixed_overrides)
    overrides.extend(spec.strategy_overrides)
    overrides.extend(spec.sweeper_overrides)
    overrides.extend(spec.disabled_overrides)
    return overrides


def build_retrain_overrides(
    spec: ExperimentSpecification,
    *,
    seed: int,
    trainer: str,
    devices: str,
    selected_overrides: Sequence[str],
    run_name: str,
) -> list[str]:
    overrides = [
        f"experiment={spec.experiment}",
        f"experiment_name={spec.name}_retrain",
        f"run_name={run_name}",
        f"trainer.max_epochs={spec.retrain_epochs}",
        f"seed={seed}",
        f"trainer={trainer}",
        f"trainer.devices={devices}",
    ]
    if spec.dataset == Dataset.PHYSICS:
        overrides.append('paths.raw_data_dir="${RAW_DATA_DIR}"')
    overrides.extend(spec.fixed_overrides)
    overrides.extend(spec.strategy_overrides)
    overrides.extend(spec.disabled_overrides)
    overrides.extend(selected_overrides)
    return overrides


def build_evaluate_overrides(
    spec: ExperimentSpecification,
    *,
    seed: int,
    trainer: str,
    devices: str,
    ckpt_path: str,
    run_name: str,
) -> list[str]:
    overrides = [
        f"experiment={spec.experiment}",
        f"experiment_name={spec.name}_evaluate",
        f"run_name={run_name}",
        "train=false",
        "test=true",
        f"ckpt_path={ckpt_path}",
        f"seed={seed}",
        f"trainer={trainer}",
        f"trainer.devices={devices}",
    ]
    if spec.dataset == Dataset.PHYSICS:
        overrides.append('paths.raw_data_dir="${RAW_DATA_DIR}"')
    overrides.extend(spec.fixed_overrides)
    overrides.extend(spec.strategy_overrides)
    return overrides


def storage_override_for(spec: ExperimentSpecification) -> str:
    return (
        "hydra.sweeper.storage="
        f"sqlite:///logs/optuna/{spec.dataset.value}/{spec.model.value}.db"
    )


def launcher_override(launcher: Launcher) -> str:
    if launcher == Launcher.SUBMITIT_LOCAL:
        return "hydra/launcher=submitit_local"
    if launcher == Launcher.SUBMITIT_SLURM_CLARIDEN:
        return "hydra/launcher=submitit_slurm_clariden"
    raise ValueError(f"Unsupported launcher override: {launcher}")


def generate_scripts(
    specs: Sequence[ExperimentSpecification],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stage: Stage = Stage.SWEEP,
    launcher: Launcher = Launcher.NONE,
    trainer: str = "gpu",
    devices: str = "[0]",
    cpus_per_task: int = 1,
    gpus_per_node: int = 1,
    timeout_min: int | None = None,
    selected_overrides_file: Path | None = None,
    ckpt_manifest_file: Path | None = None,
) -> list[Path]:
    written: list[Path] = []
    for spec in specs:
        spec.validate()
        spec_dir = output_dir / spec.name
        spec_dir.mkdir(parents=True, exist_ok=True)

        stages = [Stage.SWEEP, Stage.RETRAIN, Stage.EVALUATE] if stage == Stage.ALL else [stage]
        generated_commands: dict[str, list[str]] = {}
        for requested_stage in stages:
            if requested_stage == Stage.SWEEP:
                commands = sweep_commands_for(
                    spec,
                    launcher=launcher,
                    trainer=trainer,
                    devices=devices,
                    cpus_per_task=cpus_per_task,
                    gpus_per_node=gpus_per_node,
                    timeout_min=timeout_min,
                )
                fpath = write_script(spec_dir / "sweep.sh", commands, spec)
            elif requested_stage == Stage.RETRAIN:
                commands = retrain_commands_for(
                    spec,
                    selected_overrides_file=selected_overrides_file,
                    trainer=trainer,
                    devices=devices,
                )
                fpath = write_script(spec_dir / "retrain.sh", commands, spec)
            elif requested_stage == Stage.EVALUATE:
                commands = evaluate_commands_for(
                    spec,
                    ckpt_manifest_file=ckpt_manifest_file,
                    trainer=trainer,
                    devices=devices,
                )
                fpath = write_script(spec_dir / "evaluate.sh", commands, spec)
            else:
                raise ValueError(f"Unsupported generation stage: {requested_stage}")

            written.append(fpath)
            generated_commands[requested_stage.value] = commands

        manifest = build_manifest(spec, generated_commands)
        write_manifest_json(spec_dir / "manifest.json", manifest)
        write_manifest_md(spec_dir / "manifest.md", manifest)
        written.extend([spec_dir / "manifest.json", spec_dir / "manifest.md"])
    return written


def sweep_commands_for(
    spec: ExperimentSpecification,
    *,
    launcher: Launcher,
    trainer: str,
    devices: str,
    cpus_per_task: int,
    gpus_per_node: int,
    timeout_min: int | None,
) -> list[str]:
    commands = []
    for seed in spec.seeds:
        overrides = build_sweep_overrides(
            spec,
            seed=seed,
            launcher=launcher,
            trainer=trainer,
            devices=devices,
            cpus_per_task=cpus_per_task,
            gpus_per_node=gpus_per_node,
            timeout_min=timeout_min,
        )
        commands.append(render_train_command(overrides, multirun=True))
    return commands


def retrain_commands_for(
    spec: ExperimentSpecification,
    *,
    selected_overrides_file: Path | None,
    trainer: str,
    devices: str,
) -> list[str]:
    if selected_overrides_file is None:
        return [
            "# Provide --selected-overrides with a JSON list of selected trial overrides.",
            "# Expected format:",
            "# [",
            '#   {"run_name": "cap_trial_001", "seed": 123, "overrides": ["algorithm.optimizer.lr=0.001"]}',
            "# ]",
        ]

    selections = load_json_list(selected_overrides_file)
    commands = []
    for i, item in enumerate(selections):
        overrides = tuple(str(x) for x in item.get("overrides", []))
        seed = int(item.get("seed", spec.seeds[0]))
        run_name = str(item.get("run_name", f"selected_{i:03d}"))
        cmd_overrides = build_retrain_overrides(
            spec,
            seed=seed,
            trainer=trainer,
            devices=devices,
            selected_overrides=overrides,
            run_name=run_name,
        )
        commands.append(render_train_command(cmd_overrides, multirun=False))
    return commands


def evaluate_commands_for(
    spec: ExperimentSpecification,
    *,
    ckpt_manifest_file: Path | None,
    trainer: str,
    devices: str,
) -> list[str]:
    if ckpt_manifest_file is None:
        return [
            "# Provide --ckpt-manifest with a JSON list of checkpoints to evaluate.",
            "# Expected format:",
            "# [",
            '#   {"run_name": "cap_eval_001", "seed": 123, "ckpt_path": "/path/to/model.ckpt"}',
            "# ]",
        ]

    checkpoints = load_json_list(ckpt_manifest_file)
    commands = []
    for i, item in enumerate(checkpoints):
        seed = int(item.get("seed", spec.seeds[0]))
        run_name = str(item.get("run_name", f"eval_{i:03d}"))
        ckpt_path = str(item["ckpt_path"])
        overrides = build_evaluate_overrides(
            spec,
            seed=seed,
            trainer=trainer,
            devices=devices,
            ckpt_path=ckpt_path,
            run_name=run_name,
        )
        commands.append(render_train_command(overrides, multirun=False))
    return commands


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    if not all(isinstance(item, Mapping) for item in data):
        raise ValueError(f"{path} must contain a list of objects.")
    return [dict(item) for item in data]


def render_train_command(overrides: Sequence[str], *, multirun: bool) -> str:
    command = f"uv run python {TRAIN_ENTRYPOINT}"
    if multirun:
        command = f"{command} -m"
    rendered = [command]
    rendered.extend(render_override(x) for x in overrides)
    return " \\\n    ".join(rendered)


def render_override(override: str) -> str:
    """Render one Hydra override as a safe shell token."""
    if override.startswith("'") and override.endswith("'"):
        return override
    if override.startswith("~"):
        return shlex.quote(override)
    if '"${' in override or "$" in override:
        return override
    return shlex.quote(override)


def write_script(path: Path, commands: Sequence[str], spec: ExperimentSpecification) -> Path:
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by scripts/generation.py for {spec.name}.",
        f"# Dataset: {spec.dataset.value}",
        f"# Model: {spec.model.value}",
        f"# Strategy: {spec.strategy.value}",
        "",
        f"cd {shlex.quote(str(REPO_ROOT))}",
    ]
    if spec.dataset == Dataset.PHYSICS:
        header.extend(
            [
                'RAW_DATA_DIR="${RAW_DATA_DIR:-/path/to/adl1t_data/parquet_files}"',
                "",
            ]
        )
    else:
        required_env = required_env_vars_for(spec)
        if required_env:
            header.extend(
                [
                    *[
                        f': "${{{name}:?Set {name} before running {spec.name}}}"'
                        for name in required_env
                    ],
                    "",
                ]
            )
        header.append("")

    body = []
    for idx, command in enumerate(commands, start=1):
        body.extend([f"# Command {idx}", command, ""])

    path.write_text("\n".join(header + body), encoding="utf-8")
    path.chmod(0o755)
    return path


def required_env_vars_for(spec: ExperimentSpecification) -> tuple[str, ...]:
    overrides = [
        *spec.fixed_overrides,
        *spec.strategy_overrides,
        *spec.sweeper_overrides,
        *spec.disabled_overrides,
    ]
    required = []
    for name in ("CCHAMBER_VALID_PAIR_TABLE", "CCHAMBER_TEST_PAIR_TABLE"):
        if any(f"${name}" in override or f"${{{name}}}" in override for override in overrides):
            required.append(name)
    return tuple(required)


def build_manifest(
    spec: ExperimentSpecification,
    generated_commands: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    tuned_params = infer_tuned_params(spec.hparams_search)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "experiment": serialize_spec(spec),
        "tuned_params": list(tuned_params),
        "tuned_param_source": spec.tuned_param_source,
        "search_space": read_search_space(spec.hparams_search),
        "fixed_overrides": list(spec.fixed_overrides),
        "strategy_overrides": list(spec.strategy_overrides),
        "sweeper_overrides": list(spec.sweeper_overrides),
        "disabled_overrides": list(spec.disabled_overrides),
        "factors": {k: list(v) for k, v in spec.factors.items()},
        "commands": {k: list(v) for k, v in generated_commands.items()},
    }
    return manifest


def serialize_spec(spec: ExperimentSpecification) -> dict[str, Any]:
    data = asdict(spec)
    data["dataset"] = spec.dataset.value
    data["model"] = spec.model.value
    data["strategy"] = spec.strategy.value
    data["factors"] = {k: list(v) for k, v in spec.factors.items()}
    return data


def git_metadata() -> dict[str, Any]:
    commit = run_git(["rev-parse", "HEAD"])
    short = run_git(["rev-parse", "--short", "HEAD"])
    dirty = (
        subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    )
    untracked = run_git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    return {
        "commit": commit or None,
        "short_commit": short or None,
        "dirty": bool(dirty or untracked),
        "untracked_files": untracked,
    }


def run_git(args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def write_manifest_json(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_manifest_md(path: Path, manifest: Mapping[str, Any]) -> None:
    exp = manifest["experiment"]
    lines = [
        f"# {exp['name']}",
        "",
        f"- Dataset: `{exp['dataset']}`",
        f"- Model: `{exp['model']}`",
        f"- Strategy: `{exp['strategy']}`",
        f"- Experiment config: `{exp['experiment']}`",
        f"- HParams search: `{exp['hparams_search']}`",
        f"- Trials: `{exp['n_trials']}`",
        f"- Sweep epochs: `{exp['sweep_epochs']}`",
        f"- Retrain epochs: `{exp['retrain_epochs']}`",
        f"- Seeds: `{', '.join(str(s) for s in exp['seeds'])}`",
        "",
        "## Tuned Parameters",
        "",
    ]
    lines.extend(f"- `{param}`" for param in manifest["tuned_params"])
    lines.extend(["", "## Fixed Overrides", ""])
    lines.extend(f"- `{item}`" for item in manifest["fixed_overrides"])
    lines.extend(["", "## Strategy Overrides", ""])
    lines.extend(f"- `{item}`" for item in manifest["strategy_overrides"])
    lines.extend(["", "## Sweeper Overrides", ""])
    lines.extend(f"- `{item}`" for item in manifest["sweeper_overrides"])
    lines.extend(["", "## Disabled Overrides", ""])
    lines.extend(f"- `{item}`" for item in manifest["disabled_overrides"])
    lines.extend(["", "## Reporting Factors", ""])
    for key, values in manifest["factors"].items():
        joined = ", ".join(f"`{v}`" for v in values)
        lines.append(f"- `{key}`: {joined}")
    if exp["notes"]:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in exp["notes"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible paper experiment scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known experiments.")
    add_filters(list_parser)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate shell scripts and manifests."
    )
    add_filters(generate_parser)
    generate_parser.add_argument(
        "--stage",
        choices=[s.value for s in Stage],
        default=Stage.SWEEP.value,
        help="Which script stage to generate.",
    )
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated scripts are written.",
    )
    generate_parser.add_argument(
        "--launcher",
        choices=[x.value for x in Launcher],
        default=Launcher.NONE.value,
        help="Hydra launcher override to include in sweep scripts.",
    )
    generate_parser.add_argument("--trainer", default="gpu")
    generate_parser.add_argument("--devices", default="[0]")
    generate_parser.add_argument("--cpus-per-task", type=int, default=1)
    generate_parser.add_argument("--gpus-per-node", type=int, default=1)
    generate_parser.add_argument("--timeout-min", type=int, default=None)
    generate_parser.add_argument("--n-trials", type=int, default=600)
    generate_parser.add_argument(
        "--seeds",
        default="123",
        help="Comma-separated seeds treated as reporting/statistical factors.",
    )
    generate_parser.add_argument(
        "--selected-overrides",
        type=Path,
        default=None,
        help="JSON list of selected trial overrides for retrain scripts.",
    )
    generate_parser.add_argument(
        "--ckpt-manifest",
        type=Path,
        default=None,
        help="JSON list of checkpoints for evaluation scripts.",
    )

    return parser.parse_args(argv)


def add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", help="Exact experiment specification name.")
    parser.add_argument("--dataset", choices=[x.value for x in Dataset])
    parser.add_argument("--model", choices=[x.value for x in Model])
    parser.add_argument("--strategy", choices=[x.value for x in Strategy])
    parser.add_argument(
        "--all-paper",
        action="store_true",
        help="Select the full default paper matrix.",
    )
    parser.add_argument(
        "--exclude-cvar10",
        action="store_true",
        help="Omit semi_cvar10 appendix experiments from the default registry.",
    )


def selected_specs(args: argparse.Namespace) -> list[ExperimentSpecification]:
    seeds = parse_seeds(getattr(args, "seeds", "123"))
    n_trials = int(getattr(args, "n_trials", 600))
    registry = build_paper_experiments(
        n_trials=n_trials,
        seeds=seeds,
        include_cvar10=not getattr(args, "exclude_cvar10", False),
    )
    if args.name:
        try:
            return [registry[args.name]]
        except KeyError as exc:
            raise SystemExit(f"Unknown experiment specification: {args.name}") from exc

    specs = list(registry.values())
    if not args.all_paper:
        specs = [
            spec
            for spec in specs
            if (args.dataset is None or spec.dataset.value == args.dataset)
            and (args.model is None or spec.model.value == args.model)
            and (args.strategy is None or spec.strategy.value == args.strategy)
        ]
    if not specs:
        raise SystemExit("No experiment specifications matched the requested filters.")
    return specs


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer.")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = selected_specs(args)

    if args.command == "list":
        for spec in specs:
            print(
                f"{spec.name}\t"
                f"dataset={spec.dataset.value}\t"
                f"model={spec.model.value}\t"
                f"strategy={spec.strategy.value}\t"
                f"experiment={spec.experiment}\t"
                f"hparams_search={spec.hparams_search}"
            )
        return 0

    if args.command == "generate":
        written = generate_scripts(
            specs,
            output_dir=args.output_dir,
            stage=Stage(args.stage),
            launcher=Launcher(args.launcher),
            trainer=args.trainer,
            devices=args.devices,
            cpus_per_task=args.cpus_per_task,
            gpus_per_node=args.gpus_per_node,
            timeout_min=args.timeout_min,
            selected_overrides_file=args.selected_overrides,
            ckpt_manifest_file=args.ckpt_manifest,
        )
        for path in written:
            print(display_path(path))
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
