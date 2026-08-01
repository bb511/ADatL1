#!/usr/bin/env python3
"""Create and execute a reproducible, node-packed JetCLR canary campaign.

The command deliberately does not submit jobs.  ``init`` freezes a clean Git
revision into a deployment, authenticates the data and Python environment, and
writes a reviewed Slurm launcher.  Submission remains an explicit operator step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess  # nosec B404 - commands are fixed argument vectors, never shell strings
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/iopsstor/scratch/cscs/podagiu/data")
DEFAULT_CAMPAIGN_BASE = Path("/iopsstor/scratch/cscs/vjimenez/jetclr/campaigns")
DEFAULT_DEPLOYMENT_BASE = Path("/iopsstor/scratch/cscs/vjimenez/jetclr/deployments")
DEFAULT_VENV = Path("/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden")
DEFAULT_UV = Path("/iopsstor/scratch/cscs/vjimenez/adatl1/tools/uv-0.11.32/uv")

CONFIG_PATHS = (
    "configs/algorithm/jetclr.yaml",
    "configs/data/basis.yaml",
    "configs/experiment/physics/jetclr_pairing.yaml",
    "configs/trainer/gpu.yaml",
    "configs/train.yaml",
)
DATA_CACHE_RELATIVE = Path("data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust")
DATA_FILES = tuple(
    Path(split) / name
    for split in ("train", "valid")
    for name in ("torch_cache.pt", "torch_mask.pt", "torch_l1bit.pt")
)
STAGE1_SEED = 123
STAGE1_N_CANDIDATES = 48
STAGE1_TRAIN_BATCHES = 256
STAGE2_SEED = 123
STAGE2_N_CANDIDATES = 12
STAGE2_SOURCE_CAMPAIGN = "jetclr_20260801_866638a"
STAGE2_SOURCE_SUMMARY_SHA256 = "7041ef8770d2c3cc1578255a2e2a5e88bc27977dca24719cfef4b55a36e702f1"
STAGE2_SOURCE_SUMMARY_CSV_SHA256 = (
    "a6b3441cae9c5c0ce5d9da234a78dcf839cbcf3ab68e6d6768fb65e3c1b72e12"
)
STAGE2_SOURCE_SUMMARY = DEFAULT_CAMPAIGN_BASE / STAGE2_SOURCE_CAMPAIGN / "stage1" / "summary.json"
STAGE2_SOURCE_SUMMARY_CSV = STAGE2_SOURCE_SUMMARY.with_name("summary.csv")
STAGE2_PROMOTED_IDS = (43, 3, 37, 11, 34, 27, 39)
STAGE3_SEED = 123
STAGE3_N_CANDIDATES = 12
STAGE3_SOURCE_CAMPAIGN = "jetclr_20260801_eeb36cf"
STAGE3_SOURCE_CANDIDATE_ID = 10
STAGE3_SOURCE_SPEC_SHA256 = "5c6a470270f148595bdb48def2494f80366703b4d3cdb430c54c1f9fd952e0b5"
STAGE3_SOURCE_SUMMARY_SHA256 = "6981e4bae97aea82d902694f40a575ead9eadf11a30a2bf7f2b792a573691e8b"
STAGE3_SOURCE_SUMMARY_CSV_SHA256 = (
    "aecf84a462176850b84a1054c6d8f9e0de6f73882fae7e6d0fa87c0f41deb201"
)
STAGE3_SOURCE_ROOT = DEFAULT_CAMPAIGN_BASE / STAGE3_SOURCE_CAMPAIGN
STAGE3_SOURCE_SUMMARY = STAGE3_SOURCE_ROOT / "stage2" / "summary.json"
STAGE3_SOURCE_SUMMARY_CSV = STAGE3_SOURCE_ROOT / "stage2" / "summary.csv"
STAGE4_SEED = 123
STAGE4_N_CANDIDATES = 12
STAGE4_SOURCE_CAMPAIGN = "jetclr_20260801_ebd6dd0"
STAGE4_SOURCE_SUMMARY_SHA256 = "5425fd75220750fcea277498d1bdb99f10268a3af90387a81a249d15ad13f8d7"
STAGE4_SOURCE_SUMMARY_CSV_SHA256 = (
    "feac47f5f02397d5cde14701dacdff90c6b3df1bee66d1aae598ed4cefeccbc4"
)
STAGE4_SOURCE_ROOT = DEFAULT_CAMPAIGN_BASE / STAGE4_SOURCE_CAMPAIGN
STAGE4_SOURCE_SUMMARY = STAGE4_SOURCE_ROOT / "stage3" / "summary.json"
STAGE4_SOURCE_SUMMARY_CSV = STAGE4_SOURCE_ROOT / "stage3" / "summary.csv"
STAGE4_SOURCE_ARCHITECTURES = (
    (10, "layers2", "0ef568e754d5530bd517b55ecbd6d0b19aa9c84f38bff1dfcc5b80a827b73856"),
    (9, "official_projector", "c19d089c7261c9c2055fbc218fb44333b7e99bd9020937a3282cbe5f2de50d67"),
)
STAGE5_SEEDS = (321, 777)
STAGE5_N_CANDIDATES = 8
STAGE5_SOURCE_CAMPAIGN = "jetclr_20260801_1c4c0cc"
STAGE5_SOURCE_SUMMARY_SHA256 = "e47c477ceb6529d2d497c57c216d01657ca4eee6e9a7a362a9af52e90c277c21"
STAGE5_SOURCE_SUMMARY_CSV_SHA256 = (
    "de522f251948160c2e4343be103d0fc271647363d3721e8f52ad7f7621e7bfc2"
)
STAGE5_SOURCE_ROOT = DEFAULT_CAMPAIGN_BASE / STAGE5_SOURCE_CAMPAIGN
STAGE5_SOURCE_SUMMARY = STAGE5_SOURCE_ROOT / "stage4" / "summary.json"
STAGE5_SOURCE_SUMMARY_CSV = STAGE5_SOURCE_ROOT / "stage4" / "summary.csv"
STAGE5_SOURCE_CANDIDATES = (
    (0, "layers2_var0_cov0", "0b2579c4f46e24d4b88463731c5d25ae15cd99d48ca2fe53bf69431546029d2e"),
    (
        4,
        "layers2_var0.5_cov0.005",
        "6aa762be009195995513b85cc3c437d831aedbdf0e24459ddb323fe0df564cf7",
    ),
    (
        6,
        "official_projector_var0_cov0",
        "d566dd6f0b365e5a8c100b50b02a1f85973aa1deec790d073b495bb19a81a645",
    ),
    (
        10,
        "official_projector_var0.5_cov0.005",
        "2f4c6c9486466b5dc2b77b9266cf649f4c89bfbedafab24b61cb2d7804844dc5",
    ),
)
STAGE6_SEEDS = (123, 2027, 31415)
STAGE6_EPOCHS = 16
STAGE6_N_CANDIDATES = 12
STAGE6_MILESTONES = (1, 2, 4, 8, 16)
STAGE6_SOURCE_CAMPAIGN = "jetclr_20260801_35f7899"
STAGE6_SOURCE_ROOT = DEFAULT_CAMPAIGN_BASE / STAGE6_SOURCE_CAMPAIGN
STAGE6_SOURCE_SUMMARY = STAGE6_SOURCE_ROOT / "stage5" / "summary.json"
STAGE6_SOURCE_SUMMARY_SHA256 = "2060cbf400476406363bc5c15a2a4ea201894af0c7471c7362163785ab3bc274"
STAGE6_SOURCE_SUMMARY_CSV = STAGE6_SOURCE_ROOT / "stage5" / "summary.csv"
STAGE6_SOURCE_SUMMARY_CSV_SHA256 = (
    "5ba1566a4b0569b49286ce8d33f4a4b746118c47b1608896449a22afe88bb21f"
)
STAGE6_SOURCE_PAIRED_CSV = STAGE6_SOURCE_ROOT / "stage5" / "paired_deltas.csv"
STAGE6_SOURCE_PAIRED_CSV_SHA256 = (
    "f9885959e29fc3e5a12e4e7748bc0345e216d92ee885848652b02e280800dfaa"
)
STAGE7_MILESTONES = (1, 2, 4, 8, 16)
STAGE7_N_CANDIDATES = 60
STAGE7_SOURCE_CAMPAIGN = "jetclr_20260801_5b49e71"
STAGE7_SOURCE_ROOT = DEFAULT_CAMPAIGN_BASE / STAGE7_SOURCE_CAMPAIGN
STAGE7_SOURCE_SUMMARY = STAGE7_SOURCE_ROOT / "stage6" / "summary.json"
STAGE7_SOURCE_SUMMARY_SHA256 = "0ed426f42e5c2b8c0fdac0bba0fb8778cdff24ccb9925a549f67bb124ea06f2e"
STAGE7_SOURCE_SUMMARY_CSV = STAGE7_SOURCE_ROOT / "stage6" / "summary.csv"
STAGE7_SOURCE_SUMMARY_CSV_SHA256 = (
    "fe44d567577ff1ac61a0ca422627b5560f8fb3ac9492e18d974d21d2d5996e0e"
)
STAGE7_SOURCE_PAIRED_CSV = STAGE7_SOURCE_ROOT / "stage6" / "paired_epoch16.csv"
STAGE7_SOURCE_PAIRED_CSV_SHA256 = (
    "39cb58f5d23af8e6ab52f8fbd6780cfcfd3e6aeeec44ba58be38ea3396f3703f"
)


def _stage1_base_overrides() -> dict[str, Any]:
    """Return the production augmentation and optimizer anchor."""
    return {
        "data.batch_size": 2048,
        "trainer.gradient_clip_val": 0.1,
        "algorithm.optimizer.lr": 3e-4,
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.loss.temperature": 0.1,
        "algorithm.detector_smearing.prob": 0.8,
        "algorithm.detector_smearing.strength": 0.5,
        "algorithm.object_mask.prob": 0.8,
        "algorithm.object_mask.object_prob": 0.05,
        "algorithm.lorentz_rotation.prob": 0.5,
    }


def _hydra_scalar(value: Any) -> str:
    """Render a scalar as a stable Hydra command-line value."""
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value).lower() if isinstance(value, bool) else str(value)


def stage1_specs() -> list[dict[str, Any]]:
    """Return eight anchor ablations and forty deterministic Sobol candidates."""
    base = _stage1_base_overrides()
    anchors: list[tuple[str, dict[str, Any]]] = [
        ("production", {}),
        ("no_smearing", {"algorithm.detector_smearing": None}),
        ("no_object_dropout", {"algorithm.object_mask": None}),
        ("no_rotation", {"algorithm.lorentz_rotation": None}),
        (
            "no_augmentation",
            {
                "algorithm.detector_smearing": None,
                "algorithm.object_mask": None,
                "algorithm.lorentz_rotation": None,
            },
        ),
        (
            "weak_augmentation",
            {
                "algorithm.detector_smearing.prob": 0.4,
                "algorithm.detector_smearing.strength": 0.25,
                "algorithm.object_mask.prob": 0.4,
                "algorithm.object_mask.object_prob": 0.025,
                "algorithm.lorentz_rotation.prob": 0.25,
            },
        ),
        (
            "strong_augmentation",
            {
                "algorithm.detector_smearing.prob": 1.0,
                "algorithm.detector_smearing.strength": 1.0,
                "algorithm.object_mask.prob": 1.0,
                "algorithm.object_mask.object_prob": 0.1,
                "algorithm.lorentz_rotation.prob": 1.0,
            },
        ),
        (
            "large_batch_low_temperature",
            {"data.batch_size": 4096, "algorithm.loss.temperature": 0.05},
        ),
    ]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for name, changes in anchors:
        params = dict(base)
        params.update(changes)
        candidates.append((name, "anchor", params))

    unit = qmc.Sobol(d=10, scramble=True, seed=STAGE1_SEED).random_base2(m=6)[:40]
    batches = (2048, 4096, 8192)
    clips = (0.05, 0.1, 0.5)
    weights = (1e-6, 1e-5, 1e-4, 3e-4)
    temperatures = (0.05, 0.1, 0.2)

    def choice(values: Sequence[Any], coordinate: float) -> Any:
        return values[min(int(coordinate * len(values)), len(values) - 1)]

    for index, row in enumerate(unit):
        params = {
            "data.batch_size": choice(batches, row[0]),
            "trainer.gradient_clip_val": choice(clips, row[1]),
            "algorithm.optimizer.lr": 10 ** (-4.30103 + row[2] * 1.30103),
            "algorithm.optimizer.weight_decay": choice(weights, row[3]),
            "algorithm.loss.temperature": choice(temperatures, row[4]),
            "algorithm.detector_smearing.prob": 0.2 + 0.8 * row[5],
            "algorithm.detector_smearing.strength": 0.1 + 1.4 * row[6],
            "algorithm.object_mask.prob": 0.2 + 0.8 * row[7],
            "algorithm.object_mask.object_prob": 0.15 * row[8],
            "algorithm.lorentz_rotation.prob": row[9],
        }
        candidates.append((f"sobol_{index:02d}", "sobol", params))

    specs = []
    for candidate_id, (name, kind, params) in enumerate(candidates):
        overrides = [f"{key}={_hydra_scalar(value)}" for key, value in params.items()]
        identity = {
            "candidate_id": candidate_id,
            "name": name,
            "kind": kind,
            "seed": STAGE1_SEED,
            "train_batches": STAGE1_TRAIN_BATCHES,
            "params": params,
            "overrides": overrides,
        }
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE1_N_CANDIDATES:
        raise AssertionError("Stage-1 design must contain exactly 48 candidates.")
    return specs


def stage2_specs() -> list[dict[str, Any]]:
    """Return the frozen Stage-1 Pareto front and five targeted refinements."""
    stage1 = stage1_specs()
    records: list[tuple[str, str, int | None, dict[str, Any]]] = []
    for source_id in STAGE2_PROMOTED_IDS:
        source = stage1[source_id]
        records.append(
            (
                f"stage1_{source_id:02d}_{source['name']}",
                "stage1_promoted",
                source_id,
                dict(source["params"]),
            )
        )

    base = _stage1_base_overrides()
    refinements = [
        (
            "refine_b2048_lr5e-5_t05_rot0",
            3,
            {
                "data.batch_size": 2048,
                "algorithm.optimizer.lr": 5e-5,
                "algorithm.loss.temperature": 0.05,
                "algorithm.detector_smearing.prob": 0.4,
                "algorithm.detector_smearing.strength": 0.2,
                "algorithm.object_mask.prob": 0.4,
                "algorithm.object_mask.object_prob": 0.01,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b4096_lr1e-4_t05_rot0",
            43,
            {
                "data.batch_size": 4096,
                "algorithm.optimizer.lr": 1e-4,
                "algorithm.loss.temperature": 0.05,
                "algorithm.detector_smearing.prob": 0.5,
                "algorithm.detector_smearing.strength": 0.25,
                "algorithm.object_mask.prob": 0.5,
                "algorithm.object_mask.object_prob": 0.02,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b4096_lr3e-4_t10_rot002",
            37,
            {
                "data.batch_size": 4096,
                "algorithm.optimizer.lr": 3e-4,
                "algorithm.loss.temperature": 0.1,
                "algorithm.detector_smearing.prob": 0.6,
                "algorithm.detector_smearing.strength": 0.35,
                "algorithm.object_mask.prob": 0.6,
                "algorithm.object_mask.object_prob": 0.03,
                "algorithm.lorentz_rotation.prob": 0.02,
            },
        ),
        (
            "refine_b8192_lr2e-4_t10_rot0",
            11,
            {
                "data.batch_size": 8192,
                "algorithm.optimizer.lr": 2e-4,
                "algorithm.loss.temperature": 0.1,
                "algorithm.detector_smearing.prob": 0.5,
                "algorithm.detector_smearing.strength": 0.3,
                "algorithm.object_mask.prob": 0.5,
                "algorithm.object_mask.object_prob": 0.025,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b8192_lr5e-4_t20_rot005",
            34,
            {
                "data.batch_size": 8192,
                "algorithm.optimizer.lr": 5e-4,
                "algorithm.loss.temperature": 0.2,
                "algorithm.detector_smearing.prob": 0.8,
                "algorithm.detector_smearing.strength": 0.6,
                "algorithm.object_mask.prob": 0.8,
                "algorithm.object_mask.object_prob": 0.06,
                "algorithm.lorentz_rotation.prob": 0.05,
            },
        ),
    ]
    for name, source_id, changes in refinements:
        params = dict(base)
        params.update(changes)
        records.append((name, "targeted_refinement", source_id, params))

    specs = []
    for candidate_id, (name, kind, source_id, params) in enumerate(records):
        source = stage1[source_id] if source_id is not None else None
        rationale = (
            "Stage-1 promotion preserving Pareto utility and balance-safe diversity."
            if kind == "stage1_promoted"
            else "Targeted modest-augmentation refinement around a Pareto candidate with "
            "zero or near-zero azimuthal rotation."
        )
        identity = {
            "candidate_id": candidate_id,
            "name": name,
            "kind": kind,
            "seed": STAGE2_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE2_SOURCE_CAMPAIGN,
            "source_candidate_id": source_id,
            "source_candidate_spec_sha256": source["spec_sha256"] if source else None,
            "rationale": rationale,
            "params": params,
            "overrides": [f"{key}={_hydra_scalar(value)}" for key, value in params.items()],
        }
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE2_N_CANDIDATES:
        raise AssertionError("Stage-2 design must contain exactly 12 candidates.")
    return specs


def _stage3_primary_params() -> dict[str, Any]:
    """Return the exact Stage-2 candidate-10 optimization recipe."""
    return {
        "data.batch_size": 8192,
        "trainer.gradient_clip_val": 0.1,
        "algorithm.optimizer.lr": 2e-4,
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.loss.temperature": 0.1,
        "algorithm.detector_smearing.prob": 0.5,
        "algorithm.detector_smearing.strength": 0.3,
        "algorithm.object_mask.prob": 0.5,
        "algorithm.object_mask.object_prob": 0.025,
        "algorithm.lorentz_rotation.prob": 0.0,
    }


def stage3_specs() -> list[dict[str, Any]]:
    """Return twelve pure-NTXent architecture variants on one frozen recipe."""
    encoder = {
        "algorithm.model.d_model": 128,
        "algorithm.model.out_dim": 128,
        "algorithm.model.n_heads": 8,
        "algorithm.model.n_layers": 4,
        "algorithm.model.dim_feedforward": 512,
        "algorithm.model.pooling": "cls",
        "algorithm.model.norm_first": True,
        "algorithm.model.post_pool_norm": True,
    }
    projector = {
        "algorithm.projector.nodes": [256, 256],
        "algorithm.projector.out_dim": 128,
        "algorithm.projector.batchnorm": True,
        "algorithm.projector.activation": "gelu",
    }
    variants = [
        ("current_exact", {}),
        ("sum_prenorm_postpoolnorm", {"algorithm.model.pooling": "sum"}),
        (
            "sum_prenorm_no_postpoolnorm",
            {"algorithm.model.pooling": "sum", "algorithm.model.post_pool_norm": False},
        ),
        (
            "official_sum_postnorm",
            {
                "algorithm.model.pooling": "sum",
                "algorithm.model.norm_first": False,
                "algorithm.model.post_pool_norm": False,
            },
        ),
        (
            "width256_out128",
            {
                "algorithm.model.d_model": 256,
                "algorithm.model.out_dim": 128,
                "algorithm.model.dim_feedforward": 1024,
            },
        ),
        (
            "width256_out256",
            {
                "algorithm.model.d_model": 256,
                "algorithm.model.out_dim": 256,
                "algorithm.model.dim_feedforward": 1024,
            },
        ),
        (
            "width512_out256",
            {
                "algorithm.model.d_model": 512,
                "algorithm.model.out_dim": 256,
                "algorithm.model.dim_feedforward": 2048,
            },
        ),
        ("out64", {"algorithm.model.out_dim": 64}),
        ("projector_two_linear", {"algorithm.projector.nodes": [256]}),
        (
            "official_projector",
            {
                "algorithm.projector.nodes": [256],
                "algorithm.projector.batchnorm": False,
                "algorithm.projector.activation": "relu",
            },
        ),
        ("layers2", {"algorithm.model.n_layers": 2}),
        ("layers6", {"algorithm.model.n_layers": 6}),
    ]
    specs = []
    frozen = _stage3_primary_params()
    for candidate_id, (name, changes) in enumerate(variants):
        architecture = {**encoder, **projector, **changes}
        params = {**frozen, **architecture}
        overrides = []
        for key, value in params.items():
            prefix = (
                "+"
                if key
                in {
                    "algorithm.model.norm_first",
                    "algorithm.model.post_pool_norm",
                }
                else ""
            )
            overrides.append(f"{prefix}{key}={_hydra_scalar(value)}")
        identity = {
            "candidate_id": candidate_id,
            "name": name,
            "kind": "pure_ntxent_architecture",
            "seed": STAGE3_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE3_SOURCE_CAMPAIGN,
            "source_candidate_id": STAGE3_SOURCE_CANDIDATE_ID,
            "source_candidate_spec_sha256": STAGE3_SOURCE_SPEC_SHA256,
            "rationale": "Architecture-only variant with Stage-2 optimization recipe frozen.",
            "frozen_primary_params": frozen,
            "architecture_params": architecture,
            "params": params,
            "overrides": overrides,
        }
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE3_N_CANDIDATES:
        raise AssertionError("Stage-3 design must contain exactly 12 candidates.")
    return specs


def stage4_specs() -> list[dict[str, Any]]:
    """Return the two frozen architectures crossed with six VICReg recipes."""
    stage3 = stage3_specs()
    weights = ((0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.005), (0.5, 0.02))
    specs = []
    for source_id, source_name, source_sha256 in STAGE4_SOURCE_ARCHITECTURES:
        source = stage3[source_id]
        if source["name"] != source_name or source["spec_sha256"] != source_sha256:
            raise RuntimeError("Frozen Stage-3 architecture identity changed.")
        for variance_weight, covariance_weight in weights:
            regularization = {
                "algorithm.encoder_variance_weight": variance_weight,
                "algorithm.encoder_covariance_weight": covariance_weight,
            }
            params = {**source["params"], **regularization}
            overrides = []
            for key, value in params.items():
                prefix = (
                    "+"
                    if key
                    in {
                        "algorithm.model.norm_first",
                        "algorithm.model.post_pool_norm",
                    }
                    else ""
                )
                overrides.append(f"{prefix}{key}={_hydra_scalar(value)}")
            candidate_id = len(specs)
            identity = {
                "candidate_id": candidate_id,
                "name": (
                    f"{source_name}_var{_hydra_scalar(variance_weight)}"
                    f"_cov{_hydra_scalar(covariance_weight)}"
                ),
                "kind": "encoder_vicreg_ablation",
                "seed": STAGE4_SEED,
                "full_epochs": 1,
                "source_campaign_id": STAGE4_SOURCE_CAMPAIGN,
                "source_candidate_id": source_id,
                "source_candidate_name": source_name,
                "source_candidate_spec_sha256": source_sha256,
                "rationale": (
                    "Encoder-side VICReg ablation with every Stage-3 optimization and "
                    "architecture parameter frozen."
                ),
                "is_architecture_control": variance_weight == 0.0 and covariance_weight == 0.0,
                "frozen_stage3_params": source["params"],
                "regularization_params": regularization,
                "params": params,
                "overrides": overrides,
            }
            specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE4_N_CANDIDATES:
        raise AssertionError("Stage-4 design must contain exactly 12 candidates.")
    return specs


def stage5_specs() -> list[dict[str, Any]]:
    """Return four frozen Stage-4 configurations crossed with two fresh seeds."""
    stage4 = stage4_specs()
    specs = []
    for source_id, source_name, source_sha256 in STAGE5_SOURCE_CANDIDATES:
        source = stage4[source_id]
        if source["name"] != source_name or source["spec_sha256"] != source_sha256:
            raise RuntimeError("Frozen Stage-4 confirmation source identity changed.")
        for seed in STAGE5_SEEDS:
            candidate_id = len(specs)
            identity = {
                "candidate_id": candidate_id,
                "name": f"{source_name}_seed{seed}",
                "kind": "fresh_seed_confirmation",
                "seed": seed,
                "full_epochs": 1,
                "source_campaign_id": STAGE5_SOURCE_CAMPAIGN,
                "source_candidate_id": source_id,
                "source_candidate_name": source_name,
                "source_candidate_spec_sha256": source_sha256,
                "source_architecture_id": source["source_candidate_id"],
                "source_architecture_name": source["source_candidate_name"],
                "is_architecture_control": source["is_architecture_control"],
                "regularization_params": source["regularization_params"],
                "params": source["params"],
                "overrides": source["overrides"],
                "rationale": "Fresh-seed confirmation of a frozen Stage-4 configuration.",
            }
            specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE5_N_CANDIDATES:
        raise AssertionError("Stage-5 design must contain exactly eight candidates.")
    return specs


def stage6_specs() -> list[dict[str, Any]]:
    """Return four frozen configurations crossed with three long-horizon seeds."""
    stage4 = stage4_specs()
    specs = []
    for source_id, source_name, source_sha256 in STAGE5_SOURCE_CANDIDATES:
        source = stage4[source_id]
        if source["name"] != source_name or source["spec_sha256"] != source_sha256:
            raise RuntimeError("Frozen Stage-6 source configuration identity changed.")
        for seed in STAGE6_SEEDS:
            candidate_id = len(specs)
            identity = {
                "candidate_id": candidate_id,
                "name": f"{source_name}_seed{seed}",
                "kind": "long_horizon_pilot",
                "seed": seed,
                "full_epochs": STAGE6_EPOCHS,
                "source_campaign_id": STAGE6_SOURCE_CAMPAIGN,
                "source_candidate_id": source_id,
                "source_candidate_name": source_name,
                "source_candidate_spec_sha256": source_sha256,
                "source_architecture_id": source["source_candidate_id"],
                "source_architecture_name": source["source_candidate_name"],
                "is_architecture_control": source["is_architecture_control"],
                "regularization_params": source["regularization_params"],
                "params": source["params"],
                "overrides": source["overrides"],
                "rationale": "Common 16-epoch scheduler-horizon pilot with frozen Stage-4 recipe.",
            }
            specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE6_N_CANDIDATES:
        raise AssertionError("Stage-6 design must contain exactly twelve candidates.")
    return specs


def stage7_specs(source_root: Path = STAGE7_SOURCE_ROOT) -> list[dict[str, Any]]:
    """Return 60 immutable checkpoint-evaluation specifications."""
    source_manifest = _load_campaign(source_root)
    specs = []
    for source_spec in source_manifest["stage6"]["candidates"]:
        result_path = (
            source_root / "stage6" / f"candidate_{source_spec['candidate_id']:03d}" / "result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_digest = result.pop("result_payload_sha256", None)
        if result_digest is None or _value_sha256(result) != result_digest:
            raise ValueError(f"Stage-7 source result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != source_spec["spec_sha256"]:
            raise ValueError(f"Stage-7 source result identity mismatch: {result_path}")
        inventory = {int(item["completed_epoch"]): item for item in result["checkpoint_inventory"]}
        for completed_epoch in STAGE7_MILESTONES:
            checkpoint = inventory[completed_epoch]
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint["sha256"]:
                raise ValueError(f"Stage-7 source checkpoint mismatch: {checkpoint_path}")
            candidate_id = len(specs)
            identity = {
                "candidate_id": candidate_id,
                "kind": "milestone_evaluation",
                "name": f"source_{source_spec['candidate_id']:03d}_epoch{completed_epoch:02d}",
                "seed": source_spec["seed"],
                "completed_epoch": completed_epoch,
                "epoch_index": completed_epoch - 1,
                "source_campaign_id": STAGE7_SOURCE_CAMPAIGN,
                "source_candidate_id": source_spec["candidate_id"],
                "source_candidate_spec_sha256": source_spec["spec_sha256"],
                "source_result_path": str(result_path),
                "source_result_payload_sha256": result_digest,
                "source_checkpoint_path": str(checkpoint_path),
                "source_checkpoint_sha256": checkpoint["sha256"],
                "source_architecture_name": source_spec["source_architecture_name"],
                "is_architecture_control": source_spec["is_architecture_control"],
                "params": source_spec["params"],
                "overrides": source_spec["overrides"],
            }
            specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE7_N_CANDIDATES:
        raise AssertionError("Stage-7 design must contain exactly sixty evaluations.")
    return specs


def _canonical_json(value: Any) -> str:
    """Serialize a value into the canonical representation used for identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact on its destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a non-empty CSV artifact."""
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git(source: Path, *args: str) -> str:
    """Run a read-only Git query in a selected worktree."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    return subprocess.check_output(  # nosec B603 - fixed executable and controlled arguments
        [git, *args], cwd=source, text=True
    ).strip()


def canary_specs() -> list[dict[str, Any]]:
    """Return the fixed four-recipe canary design."""
    recipes = [
        ("production", 123, []),
        (
            "no_augmentation",
            124,
            [
                "algorithm.detector_smearing=null",
                "algorithm.object_mask=null",
                "algorithm.lorentz_rotation=null",
            ],
        ),
        (
            "small_encoder",
            125,
            [
                "data.batch_size=512",
                "algorithm.model.d_model=64",
                "algorithm.model.out_dim=64",
                "algorithm.model.n_layers=2",
                "algorithm.model.n_heads=4",
                "algorithm.model.dim_feedforward=256",
            ],
        ),
        (
            "capacity_stress",
            126,
            [
                "data.batch_size=4096",
                "algorithm.model.d_model=256",
                "algorithm.model.out_dim=256",
                "algorithm.model.n_layers=6",
                "algorithm.model.n_heads=8",
                "algorithm.model.dim_feedforward=1024",
            ],
        ),
    ]
    specs = []
    for trial_id, (name, seed, overrides) in enumerate(recipes):
        identity = {"trial_id": trial_id, "name": name, "seed": seed, "overrides": overrides}
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    return specs


def _environment_record(venv: Path, uv: Path) -> dict[str, Any]:
    """Authenticate and describe the frozen ARM64 CUDA environment."""
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(python)
    if not uv.is_file():
        raise FileNotFoundError(uv)
    probe = subprocess.check_output(  # nosec B603 - authenticated venv executable
        [
            str(python),
            "-c",
            (
                "import json,platform,sys,torch,pytorch_lightning as pl;"
                "print(json.dumps({'python':platform.python_version(),"
                "'machine':platform.machine(),'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'lightning':pl.__version__}))"
            ),
        ],
        text=True,
    )
    record = json.loads(probe)
    if tuple(map(int, record["python"].split(".")[:2])) != (3, 10):
        raise RuntimeError(f"JetCLR environment must use Python 3.10, found {record['python']}.")
    if record["machine"] != "aarch64" or record["torch_cuda"] is None:
        raise RuntimeError(f"Environment is not the expected CUDA ARM64 build: {record}")
    record.update(
        {
            "venv": str(venv.resolve()),
            "python_executable": str(python.resolve()),
            "uv": str(uv.resolve()),
            "uv_sha256": _sha256(uv),
        }
    )
    record["fingerprint_sha256"] = _value_sha256(record)
    return record


def _fingerprint_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Fingerprint every required file, failing closed when one is absent."""
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _write_launcher(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write the reviewed four-way packed Clariden canary launcher."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    launcher = root / "slurm" / "canary.sbatch"
    text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-canary-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=debug
        #SBATCH --time=01:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        set -euo pipefail
        readonly REPO={deployment}
        readonly CAMPAIGN_ROOT={root}
        readonly UV={uv}
        export PROJECT_ROOT="$REPO"
        export DATA_DIR={data_dir}
        export RAW_DATA_DIR={data_dir}/raw
        export LOG_DIR="$CAMPAIGN_ROOT/logs"
        export OUTPUT_DIR="$CAMPAIGN_ROOT/outputs"
        export CHECKPOINT_DIR="$CAMPAIGN_ROOT/checkpoints"
        export WANDB_MODE=offline
        export HYDRA_FULL_ERROR=1
        export UV_PROJECT_ENVIRONMENT={manifest['environment']['venv']}

        cd "$REPO"
        test "$(git rev-parse HEAD)" = "{manifest['git']['commit']}"
        test -z "$(git status --porcelain)"
        mkdir -p "$CAMPAIGN_ROOT/slurm"
        "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py canary \\
            --root "$CAMPAIGN_ROOT"

        pids=()
        for trial_id in 0 1 2 3; do
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \\
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-trial \\
                --root "$CAMPAIGN_ROOT" --trial-id "$trial_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${{pids[@]}}"; do
            wait "$pid" || status=1
        done
        test "$status" -eq 0
        "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect \\
            --root "$CAMPAIGN_ROOT"
        """
    )
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(text, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _write_stage1_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the packed Stage-1 array, CPU collector, and dependency submitter."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    common = textwrap.dedent(
        f"""\
        set -euo pipefail
        readonly REPO={deployment}
        readonly CAMPAIGN_ROOT={root}
        readonly UV={uv}
        export PROJECT_ROOT="$REPO"
        export DATA_DIR={data_dir}
        export RAW_DATA_DIR={data_dir}/raw
        export LOG_DIR="$CAMPAIGN_ROOT/logs"
        export OUTPUT_DIR="$CAMPAIGN_ROOT/outputs"
        export CHECKPOINT_DIR="$CAMPAIGN_ROOT/checkpoints"
        export WANDB_MODE=offline
        export HYDRA_FULL_ERROR=1
        export UV_PROJECT_ENVIRONMENT={manifest['environment']['venv']}
        cd "$REPO"
        test "$(git rev-parse HEAD)" = "{manifest['git']['commit']}"
        test -z "$(git status --porcelain)"
        """
    )
    stage1 = root / "slurm" / "stage1.sbatch"
    stage1_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s1-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=12:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --array=0-11%4
        #SBATCH --output={root}/slurm/%x-%A_%a.out
        #SBATCH --error={root}/slurm/%x-%A_%a.err

        """
    )
    stage1_text += common
    stage1_text += textwrap.dedent(
        """
        base=$((SLURM_ARRAY_TASK_ID * 4))
        pids=()
        for offset in 0 1 2 3; do
            candidate_id=$((base + offset))
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-stage1 \
                --root "$CAMPAIGN_ROOT" --candidate-id "$candidate_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
    )
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_text(stage1_text, encoding="utf-8")
    stage1.chmod(0o755)

    collector = root / "slurm" / "stage1_collect.sbatch"
    collector_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s1-collect-{manifest['campaign_id'][-8:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=00:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=16G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        """
    )
    collector_text += common
    collector_text += (
        '"$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect-stage1 '
        '--root "$CAMPAIGN_ROOT"\n'
    )
    collector.write_text(collector_text, encoding="utf-8")
    collector.chmod(0o755)

    submitter = root / "slurm" / "submit_stage1.sh"
    submitter.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            readonly SCRIPT_DIR={root}/slurm
            stage1_job=$(sbatch --parsable "$SCRIPT_DIR/stage1.sbatch")
            collector_job=$(sbatch --parsable --dependency="afterok:$stage1_job" \
                "$SCRIPT_DIR/stage1_collect.sbatch")
            printf 'stage1=%s collector=%s\\n' "$stage1_job" "$collector_job"
            """
        ),
        encoding="utf-8",
    )
    submitter.chmod(0o755)
    return {"stage1": stage1, "collector": collector, "submitter": submitter}


def _write_stage2_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the three-node packed Stage-2 array and dependent CPU collector."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    common = textwrap.dedent(
        f"""\
        set -euo pipefail
        readonly REPO={deployment}
        readonly CAMPAIGN_ROOT={root}
        readonly UV={uv}
        export PROJECT_ROOT="$REPO"
        export DATA_DIR={data_dir}
        export RAW_DATA_DIR={data_dir}/raw
        export LOG_DIR="$CAMPAIGN_ROOT/logs"
        export OUTPUT_DIR="$CAMPAIGN_ROOT/outputs"
        export CHECKPOINT_DIR="$CAMPAIGN_ROOT/checkpoints"
        export WANDB_MODE=offline
        export HYDRA_FULL_ERROR=1
        export UV_PROJECT_ENVIRONMENT={manifest['environment']['venv']}
        cd "$REPO"
        test "$(git rev-parse HEAD)" = "{manifest['git']['commit']}"
        test -z "$(git status --porcelain)"
        """
    )
    stage2 = root / "slurm" / "stage2.sbatch"
    stage2_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s2-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=12:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --array=0-2%3
        #SBATCH --output={root}/slurm/%x-%A_%a.out
        #SBATCH --error={root}/slurm/%x-%A_%a.err

        """
    )
    stage2_text += common
    stage2_text += textwrap.dedent(
        """
        base=$((SLURM_ARRAY_TASK_ID * 4))
        pids=()
        for offset in 0 1 2 3; do
            candidate_id=$((base + offset))
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-stage2 \
                --root "$CAMPAIGN_ROOT" --candidate-id "$candidate_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
    )
    stage2.parent.mkdir(parents=True, exist_ok=True)
    stage2.write_text(stage2_text, encoding="utf-8")
    stage2.chmod(0o755)

    collector = root / "slurm" / "stage2_collect.sbatch"
    collector_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s2-collect-{manifest['campaign_id'][-8:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=00:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=16G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        """
    )
    collector_text += common
    collector_text += (
        '"$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect-stage2 '
        '--root "$CAMPAIGN_ROOT"\n'
    )
    collector.write_text(collector_text, encoding="utf-8")
    collector.chmod(0o755)

    submitter = root / "slurm" / "submit_stage2.sh"
    submitter.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            readonly SCRIPT_DIR={root}/slurm
            stage2_job=$(sbatch --parsable "$SCRIPT_DIR/stage2.sbatch")
            collector_job=$(sbatch --parsable --dependency="afterok:$stage2_job" \
                "$SCRIPT_DIR/stage2_collect.sbatch")
            printf 'stage2=%s collector=%s\\n' "$stage2_job" "$collector_job"
            """
        ),
        encoding="utf-8",
    )
    submitter.chmod(0o755)
    return {"stage2": stage2, "collector": collector, "submitter": submitter}


def _write_stage3_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the Stage-3 packed array and collector from the identical Stage-2 layout."""
    templates = _write_stage2_launchers(root, manifest)
    destinations = {
        "stage3": root / "slurm" / "stage3.sbatch",
        "collector": root / "slurm" / "stage3_collect.sbatch",
        "submitter": root / "slurm" / "submit_stage3.sh",
    }
    source_by_role = {
        "stage3": templates["stage2"],
        "collector": templates["collector"],
        "submitter": templates["submitter"],
    }
    for role, destination in destinations.items():
        text = source_by_role[role].read_text(encoding="utf-8")
        text = text.replace("stage2", "stage3").replace("jetclr-s2", "jetclr-s3")
        destination.write_text(text, encoding="utf-8")
        destination.chmod(0o755)
    return destinations


def _write_stage4_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the Stage-4 three-node packed array and dependent collector."""
    templates = _write_stage3_launchers(root, manifest)
    destinations = {
        "stage4": root / "slurm" / "stage4.sbatch",
        "collector": root / "slurm" / "stage4_collect.sbatch",
        "submitter": root / "slurm" / "submit_stage4.sh",
    }
    source_by_role = {
        "stage4": templates["stage3"],
        "collector": templates["collector"],
        "submitter": templates["submitter"],
    }
    for role, destination in destinations.items():
        text = source_by_role[role].read_text(encoding="utf-8")
        text = text.replace("stage3", "stage4").replace("jetclr-s3", "jetclr-s4")
        destination.write_text(text, encoding="utf-8")
        destination.chmod(0o755)
    return destinations


def _write_stage5_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write two packed four-GPU Stage-5 nodes and their dependent collector."""
    templates = _write_stage4_launchers(root, manifest)
    destinations = {
        "stage5": root / "slurm" / "stage5.sbatch",
        "collector": root / "slurm" / "stage5_collect.sbatch",
        "submitter": root / "slurm" / "submit_stage5.sh",
    }
    source_by_role = {
        "stage5": templates["stage4"],
        "collector": templates["collector"],
        "submitter": templates["submitter"],
    }
    for role, destination in destinations.items():
        text = source_by_role[role].read_text(encoding="utf-8")
        text = text.replace("stage4", "stage5").replace("jetclr-s4", "jetclr-s5")
        text = text.replace("#SBATCH --array=0-2%3", "#SBATCH --array=0-1%2")
        destination.write_text(text, encoding="utf-8")
        destination.chmod(0o755)
    return destinations


def _write_stage6_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write three packed 12-hour Stage-6 nodes and a dependent collector."""
    templates = _write_stage4_launchers(root, manifest)
    destinations = {
        "stage6": root / "slurm" / "stage6.sbatch",
        "collector": root / "slurm" / "stage6_collect.sbatch",
        "submitter": root / "slurm" / "submit_stage6.sh",
    }
    source_by_role = {
        "stage6": templates["stage4"],
        "collector": templates["collector"],
        "submitter": templates["submitter"],
    }
    for role, destination in destinations.items():
        text = source_by_role[role].read_text(encoding="utf-8")
        text = text.replace("stage4", "stage6").replace("jetclr-s4", "jetclr-s6")
        destination.write_text(text, encoding="utf-8")
        destination.chmod(0o755)
    return destinations


def _write_stage7_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the four-node-capped packed milestone evaluation array."""
    templates = _write_stage6_launchers(root, manifest)
    destinations = {
        "stage7": root / "slurm" / "stage7.sbatch",
        "collector": root / "slurm" / "stage7_collect.sbatch",
        "submitter": root / "slurm" / "submit_stage7.sh",
    }
    source_by_role = {
        "stage7": templates["stage6"],
        "collector": templates["collector"],
        "submitter": templates["submitter"],
    }
    for role, destination in destinations.items():
        text = source_by_role[role].read_text(encoding="utf-8")
        text = text.replace("stage6", "stage7").replace("jetclr-s6", "jetclr-s7")
        text = text.replace("#SBATCH --array=0-2%3", "#SBATCH --array=0-14%4")
        text = text.replace("#SBATCH --time=12:00:00", "#SBATCH --time=01:30:00")
        destination.write_text(text, encoding="utf-8")
        destination.chmod(0o755)
    return destinations


def initialize(
    root: Path,
    deployment: Path,
    source: Path,
    data_dir: Path,
    venv: Path,
    uv: Path,
    campaign_id: str | None = None,
) -> Path:
    """Freeze code and provenance, then create a non-submitting campaign."""
    root, deployment, source, data_dir = (
        item.expanduser().resolve() for item in (root, deployment, source, data_dir)
    )
    if root.exists():
        raise FileExistsError(root)
    if deployment.exists():
        raise FileExistsError(deployment)
    if _git(source, "status", "--porcelain"):
        raise RuntimeError("Refusing to snapshot a dirty source worktree; commit JetCLR first.")
    commit = _git(source, "rev-parse", "HEAD")
    campaign_id = campaign_id or f"jetclr_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{commit[:8]}"
    environment = _environment_record(venv.expanduser().resolve(), uv.expanduser().resolve())
    config_records = _fingerprint_files([source / path for path in CONFIG_PATHS])
    cache_root = data_dir / DATA_CACHE_RELATIVE
    data_records = _fingerprint_files([cache_root / path for path in DATA_FILES])
    if _sha256(STAGE2_SOURCE_SUMMARY) != STAGE2_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-1 source summary fingerprint changed.")
    if _sha256(STAGE2_SOURCE_SUMMARY_CSV) != STAGE2_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-1 source metric table fingerprint changed.")
    if _sha256(STAGE3_SOURCE_SUMMARY) != STAGE3_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-2 source summary fingerprint changed.")
    if _sha256(STAGE3_SOURCE_SUMMARY_CSV) != STAGE3_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-2 source metric table fingerprint changed.")
    source_stage2 = json.loads((STAGE3_SOURCE_ROOT / "campaign.json").read_text(encoding="utf-8"))
    source_candidate = source_stage2["stage2"]["candidates"][STAGE3_SOURCE_CANDIDATE_ID]
    if (
        source_candidate["candidate_id"] != STAGE3_SOURCE_CANDIDATE_ID
        or source_candidate["spec_sha256"] != STAGE3_SOURCE_SPEC_SHA256
        or source_candidate["params"] != _stage3_primary_params()
    ):
        raise RuntimeError("Frozen Stage-2 primary candidate identity changed.")
    if _sha256(STAGE4_SOURCE_SUMMARY) != STAGE4_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-3 source summary fingerprint changed.")
    if _sha256(STAGE4_SOURCE_SUMMARY_CSV) != STAGE4_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-3 source metric table fingerprint changed.")
    source_stage3 = json.loads((STAGE4_SOURCE_ROOT / "campaign.json").read_text(encoding="utf-8"))
    for source_id, source_name, source_sha256 in STAGE4_SOURCE_ARCHITECTURES:
        source_architecture = source_stage3["stage3"]["candidates"][source_id]
        if (
            source_architecture["candidate_id"] != source_id
            or source_architecture["name"] != source_name
            or source_architecture["spec_sha256"] != source_sha256
            or source_architecture["params"] != stage3_specs()[source_id]["params"]
        ):
            raise RuntimeError("Frozen Stage-3 promoted architecture identity changed.")
    if _sha256(STAGE5_SOURCE_SUMMARY) != STAGE5_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-4 source summary fingerprint changed.")
    if _sha256(STAGE5_SOURCE_SUMMARY_CSV) != STAGE5_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-4 source metric table fingerprint changed.")
    source_stage4 = json.loads((STAGE5_SOURCE_ROOT / "campaign.json").read_text(encoding="utf-8"))
    for source_id, source_name, source_sha256 in STAGE5_SOURCE_CANDIDATES:
        source_configuration = source_stage4["stage4"]["candidates"][source_id]
        if (
            source_configuration["candidate_id"] != source_id
            or source_configuration["name"] != source_name
            or source_configuration["spec_sha256"] != source_sha256
            or source_configuration["params"] != stage4_specs()[source_id]["params"]
        ):
            raise RuntimeError("Frozen Stage-4 confirmation candidate identity changed.")
    if _sha256(STAGE6_SOURCE_SUMMARY) != STAGE6_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-5 source summary fingerprint changed.")
    if _sha256(STAGE6_SOURCE_SUMMARY_CSV) != STAGE6_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-5 source row table fingerprint changed.")
    if _sha256(STAGE6_SOURCE_PAIRED_CSV) != STAGE6_SOURCE_PAIRED_CSV_SHA256:
        raise RuntimeError("Frozen Stage-5 paired table fingerprint changed.")
    stage5_summary = json.loads(STAGE6_SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if not all(
        stage5_summary.get("confirmations", {}).get(name, {}).get("promotion") is True
        for name in ("layers2", "official_projector")
    ):
        raise RuntimeError("Both Stage-5 architectures must be promoted before Stage 6.")
    for path, digest in (
        (STAGE7_SOURCE_SUMMARY, STAGE7_SOURCE_SUMMARY_SHA256),
        (STAGE7_SOURCE_SUMMARY_CSV, STAGE7_SOURCE_SUMMARY_CSV_SHA256),
        (STAGE7_SOURCE_PAIRED_CSV, STAGE7_SOURCE_PAIRED_CSV_SHA256),
    ):
        if _sha256(path) != digest:
            raise RuntimeError("Frozen Stage-6 milestone source fingerprint changed.")

    deployment.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    subprocess.run(  # nosec B603 - fixed Git executable and explicit paths
        [git, "clone", "--quiet", "--no-hardlinks", str(source), str(deployment)], check=True
    )
    subprocess.run(  # nosec B603 - fixed Git executable and authenticated commit
        [git, "checkout", "--quiet", "--detach", commit], cwd=deployment, check=True
    )
    if _git(deployment, "status", "--porcelain"):
        raise RuntimeError("Fresh deployment snapshot is unexpectedly dirty.")
    (deployment / ".venv").symlink_to(Path(environment["venv"]), target_is_directory=True)

    root.mkdir(parents=True)
    specs = canary_specs()
    stage1 = stage1_specs()
    stage2 = stage2_specs()
    stage3 = stage3_specs()
    stage4 = stage4_specs()
    stage5 = stage5_specs()
    stage6 = stage6_specs()
    stage7 = stage7_specs()
    _atomic_json(root / "design" / "canary_trials.json", specs)
    _atomic_json(root / "design" / "stage1_candidates.json", stage1)
    _atomic_json(root / "design" / "stage2_candidates.json", stage2)
    _atomic_json(root / "design" / "stage3_candidates.json", stage3)
    _atomic_json(root / "design" / "stage4_candidates.json", stage4)
    _atomic_json(root / "design" / "stage5_candidates.json", stage5)
    _atomic_json(root / "design" / "stage6_candidates.json", stage6)
    _atomic_json(root / "design" / "stage7_candidates.json", stage7)
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "branch": _git(source, "branch", "--show-current"),
            "source": str(source),
        },
        "deployment": {"path": str(deployment), "commit": commit},
        "config": {
            "files": config_records,
            "tree_sha256": _value_sha256(config_records),
            "uv_lock_sha256": _sha256(source / "uv.lock"),
        },
        "data": {
            "root": str(data_dir),
            "cache_root": str(cache_root),
            "files": data_records,
            "tree_sha256": _value_sha256(data_records),
        },
        "environment": environment,
        "canary": {"trials": specs, "design_sha256": _value_sha256(specs)},
        "stage1": {
            "seed": STAGE1_SEED,
            "train_batches": STAGE1_TRAIN_BATCHES,
            "candidates": stage1,
            "design_sha256": _value_sha256(stage1),
        },
        "stage2": {
            "seed": STAGE2_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE2_SOURCE_CAMPAIGN,
            "source_summary": str(STAGE2_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE2_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE2_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE2_SOURCE_SUMMARY_CSV_SHA256,
            "source_promoted_candidate_ids": list(STAGE2_PROMOTED_IDS),
            "candidates": stage2,
            "design_sha256": _value_sha256(stage2),
        },
        "stage3": {
            "seed": STAGE3_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE3_SOURCE_CAMPAIGN,
            "source_candidate_id": STAGE3_SOURCE_CANDIDATE_ID,
            "source_candidate_spec_sha256": STAGE3_SOURCE_SPEC_SHA256,
            "source_summary": str(STAGE3_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE3_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE3_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE3_SOURCE_SUMMARY_CSV_SHA256,
            "frozen_primary_params": _stage3_primary_params(),
            "candidates": stage3,
            "design_sha256": _value_sha256(stage3),
        },
        "stage4": {
            "seed": STAGE4_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE4_SOURCE_CAMPAIGN,
            "source_summary": str(STAGE4_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE4_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE4_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE4_SOURCE_SUMMARY_CSV_SHA256,
            "source_architectures": [
                {
                    "candidate_id": candidate_id,
                    "name": name,
                    "spec_sha256": spec_sha256,
                }
                for candidate_id, name, spec_sha256 in STAGE4_SOURCE_ARCHITECTURES
            ],
            "candidates": stage4,
            "design_sha256": _value_sha256(stage4),
        },
        "stage5": {
            "seeds": list(STAGE5_SEEDS),
            "full_epochs": 1,
            "source_campaign_id": STAGE5_SOURCE_CAMPAIGN,
            "source_summary": str(STAGE5_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE5_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE5_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE5_SOURCE_SUMMARY_CSV_SHA256,
            "source_candidates": [
                {"candidate_id": item[0], "name": item[1], "spec_sha256": item[2]}
                for item in STAGE5_SOURCE_CANDIDATES
            ],
            "candidates": stage5,
            "design_sha256": _value_sha256(stage5),
        },
        "stage6": {
            "seeds": list(STAGE6_SEEDS),
            "full_epochs": STAGE6_EPOCHS,
            "milestone_epochs": list(STAGE6_MILESTONES),
            "source_campaign_id": STAGE6_SOURCE_CAMPAIGN,
            "source_summary": str(STAGE6_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE6_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE6_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE6_SOURCE_SUMMARY_CSV_SHA256,
            "source_paired_csv": str(STAGE6_SOURCE_PAIRED_CSV),
            "source_paired_csv_sha256": STAGE6_SOURCE_PAIRED_CSV_SHA256,
            "source_promotions": {"layers2": True, "official_projector": True},
            "candidates": stage6,
            "design_sha256": _value_sha256(stage6),
        },
        "stage7": {
            "train": False,
            "test": False,
            "milestone_epochs": list(STAGE7_MILESTONES),
            "source_campaign_id": STAGE7_SOURCE_CAMPAIGN,
            "source_summary_sha256": STAGE7_SOURCE_SUMMARY_SHA256,
            "source_summary_csv_sha256": STAGE7_SOURCE_SUMMARY_CSV_SHA256,
            "source_paired_csv_sha256": STAGE7_SOURCE_PAIRED_CSV_SHA256,
            "candidates": stage7,
            "design_sha256": _value_sha256(stage7),
        },
    }
    manifest["manifest_payload_sha256"] = _value_sha256(manifest)
    _atomic_json(root / "campaign.json", manifest)
    launcher = _write_launcher(root, manifest)
    _write_stage1_launchers(root, manifest)
    _write_stage2_launchers(root, manifest)
    _write_stage3_launchers(root, manifest)
    _write_stage4_launchers(root, manifest)
    _write_stage5_launchers(root, manifest)
    _write_stage6_launchers(root, manifest)
    _write_stage7_launchers(root, manifest)
    return launcher


def _load_campaign(root: Path) -> dict[str, Any]:
    """Load a campaign only after authenticating its immutable payload."""
    path = root / "campaign.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("manifest_payload_sha256")
    if _value_sha256(value) != digest:
        raise ValueError("Campaign manifest fingerprint mismatch.")
    value["manifest_payload_sha256"] = digest
    return value


def _assert_runtime(manifest: Mapping[str, Any]) -> Path:
    """Require execution from the exact clean deployment recorded at init."""
    deployment = Path(manifest["deployment"]["path"])
    if _git(deployment, "rev-parse", "HEAD") != manifest["git"]["commit"]:
        raise RuntimeError("Deployment commit does not match the campaign.")
    if _git(deployment, "status", "--porcelain"):
        raise RuntimeError("Campaign deployment is dirty.")
    return deployment


def validate_campaign(root: Path) -> dict[str, Any]:
    """Re-authenticate code, configs, data, and environment before allocation use."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    current_configs = _fingerprint_files([deployment / path for path in CONFIG_PATHS])
    expected_configs = manifest["config"]["files"]
    for current, expected in zip(current_configs, expected_configs, strict=True):
        if current["size"] != expected["size"] or current["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Campaign config changed: {current['path']}")
    current_data = _fingerprint_files([Path(item["path"]) for item in manifest["data"]["files"]])
    if _value_sha256(current_data) != manifest["data"]["tree_sha256"]:
        raise RuntimeError("Campaign data cache fingerprints changed.")
    environment = _environment_record(
        Path(manifest["environment"]["venv"]), Path(manifest["environment"]["uv"])
    )
    if environment["fingerprint_sha256"] != manifest["environment"]["fingerprint_sha256"]:
        raise RuntimeError("Campaign Python environment fingerprint changed.")
    return manifest


def _last_finite_metrics(path: Path) -> dict[str, float]:
    """Extract the final finite value of each metric from a Lightning CSV log."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, float] = {}
    for row in rows:
        for name, raw in row.items():
            if name in {"epoch", "epoch_idx", "step"} or raw in (None, ""):
                continue
            value = float(raw)
            if math.isfinite(value):
                metrics[name] = value
    if "train/loss_mean" not in metrics:
        raise RuntimeError(f"Canary produced no finite train/loss_mean in {path}.")
    return metrics


def _metric_json(path: Path, required: Sequence[str]) -> dict[str, Any]:
    """Load a metric artifact and require its selection fields to be finite."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metric artifact must contain an object: {path}")
    for name in required:
        if name not in value:
            raise ValueError(f"Metric artifact {path} is missing {name!r}.")
        metric = value[name]
        if metric is None and name in {
            "value_smd_before_mean",
            "value_smd_after_mean",
            "occupancy_smd_before_mean",
            "occupancy_smd_after_mean",
        }:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Metric {name!r} in {path} must be numeric.")
        if not math.isfinite(float(metric)):
            raise ValueError(f"Metric {name!r} in {path} must be finite.")
    return value


def _validate_optional_metrics(value: Mapping[str, Any], path: Path, names: Sequence[str]) -> None:
    """Allow unavailable metrics as null while rejecting missing or non-finite values."""
    for name in names:
        if name not in value:
            raise ValueError(f"Metric artifact {path} is missing {name!r}.")
        metric = value[name]
        if metric is None:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Optional metric {name!r} in {path} must be numeric or null.")
        if not math.isfinite(float(metric)):
            raise ValueError(f"Optional metric {name!r} in {path} must be finite or null.")


def _single_artifact(root: Path, name: str) -> Path:
    """Resolve exactly one named evaluator artifact below a trial output."""
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {paths}.")
    return paths[0]


def run_trial(root: Path, trial_id: int) -> Path:
    """Run one fixed real-data canary recipe and atomically record its result."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["canary"]["trials"]
    if trial_id < 0 or trial_id >= len(specs):
        raise ValueError(f"trial-id must be between 0 and {len(specs) - 1}.")
    spec = specs[trial_id]
    trial_root = root / "canary" / f"{trial_id:02d}_{spec['name']}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing result identity mismatch: {result_path}")
        print(result_path)
        return result_path

    command = [
        sys.executable,
        "src/train.py",
        "experiment=physics/jetclr_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=4",
        "+trainer.limit_val_batches=1",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "evaluation.callbacks=null",
        "test=false",
        f"seed={spec['seed']}",
        f"experiment_name=jetclr_canary_{manifest['campaign_id']}",
        f"run_name={trial_id:02d}_{spec['name']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        *spec["overrides"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(deployment),
            "DATA_DIR": manifest["data"]["root"],
            "LOG_DIR": str(root / "logs"),
            "OUTPUT_DIR": str(root / "outputs"),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    trial_root.mkdir(parents=True, exist_ok=True)
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - argv is fixed campaign configuration
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        failure = {
            "schema_version": 1,
            "trial_id": trial_id,
            "spec_sha256": spec["spec_sha256"],
            "returncode": completed.returncode,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        }
        _atomic_json(trial_root / "failure.json", failure)
        raise subprocess.CalledProcessError(completed.returncode, command)
    metric_paths = sorted((trial_root / "output").rglob("metrics.csv"))
    if len(metric_paths) != 1:
        raise RuntimeError(f"Expected one metrics.csv for trial {trial_id}, found {metric_paths}.")
    metrics = _last_finite_metrics(metric_paths[0])
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "trial_id": trial_id,
        "name": spec["name"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "metrics_csv": str(metric_paths[0]),
        "metrics_csv_sha256": _sha256(metric_paths[0]),
        "metrics": metrics,
    }
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def run_stage1(root: Path, candidate_id: int) -> Path:
    """Run one fixed Stage-1 candidate and authenticate its three metric artifacts."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["stage1"]["candidates"]
    if candidate_id < 0 or candidate_id >= len(specs):
        raise ValueError(f"candidate-id must be between 0 and {len(specs) - 1}.")
    spec = specs[candidate_id]
    trial_root = root / "stage1" / f"candidate_{candidate_id:03d}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Existing Stage-1 result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing Stage-1 result identity mismatch: {result_path}")
        print(result_path)
        return result_path

    command = [
        sys.executable,
        "src/train.py",
        "experiment=physics/jetclr_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        f"+trainer.limit_train_batches={manifest['stage1']['train_batches']}",
        "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "test=false",
        f"seed={manifest['stage1']['seed']}",
        "data.max_val_batches=4",
        "data.max_normal_eval_batches=8",
        "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192",
        "evaluation.callbacks.embedding_anomaly.reference_size=8192",
        "evaluation.callbacks.embedding_anomaly.max_query_events=8192",
        f"experiment_name=jetclr_stage1_{manifest['campaign_id']}",
        f"run_name=candidate_{candidate_id:03d}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        *spec["overrides"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(deployment),
            "DATA_DIR": manifest["data"]["root"],
            "LOG_DIR": str(root / "logs"),
            "OUTPUT_DIR": str(root / "outputs"),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - fixed campaign argv without a shell
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        _atomic_json(
            trial_root / "failure.json",
            {
                "schema_version": 1,
                "campaign_id": manifest["campaign_id"],
                "candidate_id": candidate_id,
                "spec_sha256": spec["spec_sha256"],
                "returncode": completed.returncode,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            },
        )
        raise subprocess.CalledProcessError(completed.returncode, command)

    metrics_csv = _single_artifact(trial_root / "output", "metrics.csv")
    training = _last_finite_metrics(metrics_csv)
    pairing_path = _single_artifact(trial_root / "output", "pairing_diagnostics.json")
    pairing = _metric_json(
        pairing_path,
        (
            "selection_score",
            "raw_selection_score",
            "closure_recall_at_10",
            "mnn_coverage",
            "embedding_finite_fraction",
            "embedding_active_fraction",
            "embedding_effective_rank",
            "embedding_participation_rank",
            "embedding_top_pc_fraction",
        ),
    )
    _validate_optional_metrics(
        pairing,
        pairing_path,
        (
            "value_smd_before_mean",
            "value_smd_after_mean",
            "occupancy_smd_before_mean",
            "occupancy_smd_after_mean",
        ),
    )
    if not isinstance(pairing.get("collapse_pass"), bool) or not isinstance(
        pairing.get("collapse_failures"), list
    ):
        raise ValueError(f"Pairing collapse gate is malformed: {pairing_path}")
    anomaly_path = _single_artifact(trial_root / "output", "embedding_anomaly.json")
    anomaly = _metric_json(
        anomaly_path,
        ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"),
    )
    if not isinstance(anomaly.get("per_dataset"), dict) or not anomaly["per_dataset"]:
        raise ValueError(f"Embedding anomaly per-dataset metrics are missing: {anomaly_path}")
    for name in ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"):
        if not 0.0 <= float(anomaly[name]) <= 1.0:
            raise ValueError(f"AUROC {name!r} is outside [0, 1]: {anomaly_path}")

    artifacts = {
        "training_csv": metrics_csv,
        "pairing_json": pairing_path,
        "anomaly_json": anomaly_path,
    }
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "candidate_id": candidate_id,
        "name": spec["name"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "params": spec["params"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
        "training_metrics": training,
        "pairing_metrics": pairing,
        "anomaly_metrics": anomaly,
    }
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def _validate_stage4_training_metrics(
    training: Mapping[str, float], spec: Mapping[str, Any], metrics_path: Path
) -> None:
    """Require finite, internally consistent Stage-4 loss decomposition metrics."""
    names = (
        "train/loss_mean",
        "train/loss_ntxent",
        "train/loss_encoder_variance",
        "train/loss_encoder_covariance",
        "train/loss_encoder_variance_weighted",
        "train/loss_encoder_covariance_weighted",
    )
    missing = [name for name in names if name not in training]
    if missing:
        raise RuntimeError(f"Stage-4 training metrics are missing {missing}: {metrics_path}")
    values = {name: float(training[name]) for name in names}
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"Stage-4 loss decomposition must be non-negative: {metrics_path}")
    variance_weight = float(spec["regularization_params"]["algorithm.encoder_variance_weight"])
    covariance_weight = float(spec["regularization_params"]["algorithm.encoder_covariance_weight"])
    expected = {
        "train/loss_encoder_variance_weighted": (
            variance_weight * values["train/loss_encoder_variance"]
        ),
        "train/loss_encoder_covariance_weighted": (
            covariance_weight * values["train/loss_encoder_covariance"]
        ),
        "train/loss_mean": (
            values["train/loss_ntxent"]
            + values["train/loss_encoder_variance_weighted"]
            + values["train/loss_encoder_covariance_weighted"]
        ),
    }
    for name, expected_value in expected.items():
        if not math.isclose(values[name], expected_value, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"Stage-4 loss decomposition is inconsistent for {name}: {metrics_path}"
            )


def _run_full_epoch_stage(root: Path, candidate_id: int, stage: str) -> Path:
    """Run one authenticated full-epoch candidate for Stage 2 through 6."""
    if stage not in {"stage2", "stage3", "stage4", "stage5", "stage6"}:
        raise ValueError(f"Unsupported full-epoch stage: {stage}")
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest[stage]["candidates"]
    if candidate_id < 0 or candidate_id >= len(specs):
        raise ValueError(f"candidate-id must be between 0 and {len(specs) - 1}.")
    spec = specs[candidate_id]
    trial_root = root / stage / f"candidate_{candidate_id:03d}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Existing {stage} result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing {stage} result identity mismatch: {result_path}")
        print(result_path)
        return result_path

    epochs = int(manifest[stage].get("full_epochs", 1))
    command = [
        sys.executable,
        "src/train.py",
        "experiment=physics/jetclr_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={epochs}",
        f"trainer.max_epochs={epochs}",
        "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "test=false",
        f"seed={spec['seed']}",
        "data.max_val_batches=4",
        "data.max_normal_eval_batches=8",
        "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192",
        "evaluation.callbacks.embedding_anomaly.reference_size=8192",
        "evaluation.callbacks.embedding_anomaly.max_query_events=8192",
        f"experiment_name=jetclr_{stage}_{manifest['campaign_id']}",
        f"run_name=candidate_{candidate_id:03d}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        *(
            [
                "callbacks.last_epoch_ckpt.filename='epoch-{epoch:02d}'",
                "callbacks.last_epoch_ckpt.save_top_k=-1",
                "callbacks.last_epoch_ckpt.every_n_epochs=1",
                "callbacks.last_epoch_ckpt.save_last=true",
                "callbacks.last_epoch_ckpt.auto_insert_metric_name=false",
                "callbacks.last_epoch_ckpt.save_on_train_epoch_end=true",
            ]
            if stage == "stage6"
            else []
        ),
        *spec["overrides"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(deployment),
            "DATA_DIR": manifest["data"]["root"],
            "LOG_DIR": str(root / "logs"),
            "OUTPUT_DIR": str(root / "outputs"),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - fixed campaign argv without a shell
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        _atomic_json(
            trial_root / "failure.json",
            {
                "schema_version": 1,
                "campaign_id": manifest["campaign_id"],
                "candidate_id": candidate_id,
                "spec_sha256": spec["spec_sha256"],
                "returncode": completed.returncode,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            },
        )
        raise subprocess.CalledProcessError(completed.returncode, command)

    metrics_csv = _single_artifact(trial_root / "output", "metrics.csv")
    training = _last_finite_metrics(metrics_csv)
    if stage in {"stage4", "stage5", "stage6"}:
        _validate_stage4_training_metrics(training, spec, metrics_csv)
    pairing_path = _single_artifact(trial_root / "output", "pairing_diagnostics.json")
    required_pairing = [
        "selection_score",
        "raw_selection_score",
        "closure_recall_at_10",
        "mnn_coverage",
        "embedding_finite_fraction",
        "embedding_active_fraction",
        "embedding_effective_rank",
        "embedding_participation_rank",
        "embedding_top_pc_fraction",
        "value_smd_before_mean",
        "value_smd_after_mean",
        "occupancy_smd_before_mean",
        "occupancy_smd_after_mean",
    ]
    if stage in {"stage3", "stage4", "stage5", "stage6"}:
        required_pairing.extend(
            [
                "projector_embedding_finite_fraction",
                "projector_embedding_active_fraction",
                "projector_embedding_effective_rank",
                "projector_embedding_participation_rank",
                "projector_embedding_top_pc_fraction",
            ]
        )
    pairing = _metric_json(pairing_path, required_pairing)
    if not isinstance(pairing.get("collapse_pass"), bool) or not isinstance(
        pairing.get("collapse_failures"), list
    ):
        raise ValueError(f"Pairing collapse gate is malformed: {pairing_path}")
    if stage in {"stage3", "stage4", "stage5", "stage6"} and (
        not isinstance(pairing.get("projector_collapse_pass"), bool)
        or not isinstance(pairing.get("projector_collapse_failures"), list)
    ):
        raise ValueError(f"Projector collapse gate is malformed: {pairing_path}")
    anomaly_path = _single_artifact(trial_root / "output", "embedding_anomaly.json")
    anomaly = _metric_json(
        anomaly_path,
        ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"),
    )
    if not isinstance(anomaly.get("per_dataset"), dict) or not anomaly["per_dataset"]:
        raise ValueError(f"Embedding anomaly per-dataset metrics are missing: {anomaly_path}")
    for name in ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"):
        if not 0.0 <= float(anomaly[name]) <= 1.0:
            raise ValueError(f"AUROC {name!r} is outside [0, 1]: {anomaly_path}")
    if stage in {"stage4", "stage5", "stage6"} and not _valid_anomaly_aurocs(anomaly):
        raise ValueError(f"Stage-4 per-dataset AUROCs are malformed: {anomaly_path}")

    artifacts = {
        "training_csv": metrics_csv,
        "pairing_json": pairing_path,
        "anomaly_json": anomaly_path,
    }
    if stage in {"stage4", "stage5", "stage6"}:
        artifacts["last_checkpoint"] = _single_artifact(trial_root, "last.ckpt")
    checkpoint_inventory = None
    if stage == "stage6":
        checkpoint_paths = sorted((trial_root / "checkpoints").rglob("*.ckpt"))
        expected_names = {f"epoch-{index:02d}.ckpt" for index in range(STAGE6_EPOCHS)} | {
            "last.ckpt"
        }
        names = [path.name for path in checkpoint_paths]
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise RuntimeError(
                f"Stage-6 checkpoint inventory is ambiguous or incomplete: {checkpoint_paths}"
            )
        by_name = {path.name: path for path in checkpoint_paths}
        checkpoint_inventory = [
            {
                "completed_epoch": index + 1,
                "epoch_index": index,
                "path": str(by_name[f"epoch-{index:02d}.ckpt"]),
                "sha256": _sha256(by_name[f"epoch-{index:02d}.ckpt"]),
                "is_milestone": index + 1 in STAGE6_MILESTONES,
            }
            for index in range(STAGE6_EPOCHS)
        ]
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "candidate_id": candidate_id,
        "name": spec["name"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "source_campaign_id": spec["source_campaign_id"],
        "source_candidate_id": spec["source_candidate_id"],
        "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
        "rationale": spec["rationale"],
        "params": spec["params"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
        "training_metrics": training,
        "pairing_metrics": pairing,
        "anomaly_metrics": anomaly,
    }
    if checkpoint_inventory is not None:
        result["checkpoint_inventory"] = checkpoint_inventory
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def run_stage2(root: Path, candidate_id: int) -> Path:
    """Run one Stage-2 candidate for one complete epoch and authenticate outputs."""
    return _run_full_epoch_stage(root, candidate_id, "stage2")


def run_stage3(root: Path, candidate_id: int) -> Path:
    """Run one pure-NTXent Stage-3 architecture for one complete epoch."""
    return _run_full_epoch_stage(root, candidate_id, "stage3")


def run_stage4(root: Path, candidate_id: int) -> Path:
    """Run one Stage-4 encoder-regularization ablation for one complete epoch."""
    return _run_full_epoch_stage(root, candidate_id, "stage4")


def run_stage5(root: Path, candidate_id: int) -> Path:
    """Run one fresh-seed Stage-5 confirmation for one complete epoch."""
    return _run_full_epoch_stage(root, candidate_id, "stage5")


def run_stage6(root: Path, candidate_id: int) -> Path:
    """Run one configuration over the common 16-epoch scheduler horizon."""
    return _run_full_epoch_stage(root, candidate_id, "stage6")


def run_stage7(root: Path, candidate_id: int) -> Path:
    """Evaluate one exact Stage-6 checkpoint without fitting or touching test data."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["stage7"]["candidates"]
    if candidate_id < 0 or candidate_id >= len(specs):
        raise ValueError(f"candidate-id must be between 0 and {len(specs) - 1}.")
    spec = specs[candidate_id]
    trial_root = root / "stage7" / f"candidate_{candidate_id:03d}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Existing Stage-7 result fingerprint mismatch: {result_path}")
        print(result_path)
        return result_path

    source_result_path = Path(spec["source_result_path"])
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    source_digest = source_result.pop("result_payload_sha256", None)
    if (
        source_digest != spec["source_result_payload_sha256"]
        or _value_sha256(source_result) != source_digest
    ):
        raise ValueError(f"Stage-7 source result changed: {source_result_path}")
    source_checkpoint = Path(spec["source_checkpoint_path"])
    if (
        not source_checkpoint.is_file()
        or _sha256(source_checkpoint) != spec["source_checkpoint_sha256"]
    ):
        raise ValueError(f"Stage-7 source checkpoint changed: {source_checkpoint}")

    alias = trial_root / "checkpoint_alias" / "milestone"
    alias.mkdir(parents=True, exist_ok=True)
    alias_checkpoint = alias / "last.ckpt"
    if not alias_checkpoint.exists():
        alias_checkpoint.symlink_to(source_checkpoint)
    if _sha256(alias_checkpoint) != spec["source_checkpoint_sha256"]:
        raise ValueError(f"Stage-7 alias checkpoint mismatch: {alias_checkpoint}")
    command = [
        sys.executable,
        "src/train.py",
        "experiment=physics/jetclr_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        "train=false",
        "test=false",
        "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.clear_ckpts=null",
        "callbacks.last_epoch_ckpt=null",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        f"seed={spec['seed']}",
        "data.max_val_batches=4",
        "data.max_normal_eval_batches=8",
        "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192",
        "evaluation.callbacks.embedding_anomaly.reference_size=8192",
        "evaluation.callbacks.embedding_anomaly.max_query_events=8192",
        "experiment_name=checkpoint_alias",
        "run_name=milestone",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        *spec["overrides"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(deployment),
            "DATA_DIR": manifest["data"]["root"],
            "LOG_DIR": str(root / "logs"),
            "OUTPUT_DIR": str(root / "outputs"),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    (trial_root / "output").mkdir(parents=True)
    completed = subprocess.run(command, cwd=deployment, env=environment, check=False)  # nosec B603
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if (
        not alias_checkpoint.is_symlink()
        or _sha256(alias_checkpoint) != spec["source_checkpoint_sha256"]
    ):
        raise ValueError("Stage-7 evaluation modified its checkpoint alias.")
    pairing_path = _single_artifact(trial_root / "output", "pairing_diagnostics.json")
    anomaly_path = _single_artifact(trial_root / "output", "embedding_anomaly.json")
    pairing = _metric_json(
        pairing_path,
        (
            "raw_selection_score",
            "mnn_coverage",
            "embedding_finite_fraction",
            "embedding_effective_rank",
            "embedding_participation_rank",
            "embedding_top_pc_fraction",
            "projector_embedding_finite_fraction",
            "projector_embedding_effective_rank",
        ),
    )
    anomaly = _metric_json(
        anomaly_path, ("macro_mean_auroc", "macro_median_auroc", "worst_quartile_mean_auroc")
    )
    if not _valid_anomaly_aurocs(anomaly):
        raise ValueError(f"Stage-7 anomaly metrics are malformed: {anomaly_path}")
    artifacts = {"pairing_json": pairing_path, "anomaly_json": anomaly_path}
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "candidate_id": candidate_id,
        "spec_sha256": spec["spec_sha256"],
        "seed": spec["seed"],
        "completed_epoch": spec["completed_epoch"],
        "source_campaign_id": spec["source_campaign_id"],
        "source_candidate_id": spec["source_candidate_id"],
        "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
        "source_result_payload_sha256": source_digest,
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
        "alias_checkpoint_path": str(alias_checkpoint),
        "command": command,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
        "pairing_metrics": pairing,
        "anomaly_metrics": anomaly,
    }
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def collect(root: Path) -> Path:
    """Validate all four trial artifacts and write an atomic canary summary."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows = []
    for spec in manifest["canary"]["trials"]:
        result_path = root / "canary" / f"{spec['trial_id']:02d}_{spec['name']}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_digest = result.pop("result_payload_sha256", None)
        if result_digest is None or _value_sha256(result) != result_digest:
            raise ValueError(f"Result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Result identity mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
        ):
            raise ValueError(f"Result campaign identity mismatch: {result_path}")
        metrics_path = Path(result["metrics_csv"])
        if not metrics_path.is_file() or _sha256(metrics_path) != result["metrics_csv_sha256"]:
            raise ValueError(f"Metrics artifact mismatch: {metrics_path}")
        rows.append(
            {
                "trial_id": spec["trial_id"],
                "name": spec["name"],
                "seed": spec["seed"],
                "train_loss": result["metrics"]["train/loss_mean"],
                "git_commit": result["git_commit"],
                "spec_sha256": spec["spec_sha256"],
            }
        )
    table = root / "canary" / "summary.csv"
    _atomic_csv(table, rows)
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete",
        "n_trials": len(rows),
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "canary" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def collect_stage1(root: Path) -> Path:
    """Authenticate all Stage-1 results and rank collapse-safe candidates."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage1"]["candidates"]:
        result_path = root / "stage1" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-1 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
        ):
            raise ValueError(f"Stage-1 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-1 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        collapse_pass = bool(pairing["collapse_pass"])
        finite_pass = float(pairing["embedding_finite_fraction"]) == 1.0
        eligible = collapse_pass and finite_pass
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "seed": spec["seed"],
                "eligible": eligible,
                "collapse_pass": collapse_pass,
                "collapse_failures": ";".join(pairing["collapse_failures"]),
                "embedding_finite_fraction": pairing["embedding_finite_fraction"],
                "embedding_active_fraction": pairing["embedding_active_fraction"],
                "embedding_effective_rank": pairing["embedding_effective_rank"],
                "embedding_participation_rank": pairing["embedding_participation_rank"],
                "embedding_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "pairing_selection_score": pairing["selection_score"],
                "pairing_raw_selection_score": pairing["raw_selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": result["training_metrics"]["train/loss_mean"],
                "params_json": _canonical_json(spec["params"]),
                "spec_sha256": spec["spec_sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 1 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )
    rows.sort(
        key=lambda row: (
            bool(row["eligible"]),
            float(row["worst_quartile_mean_auroc"]),
            float(row["macro_median_auroc"]),
            float(row["pairing_selection_score"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    table = root / "stage1" / "summary.csv"
    _atomic_csv(table, rows)
    eligible = [row for row in rows if row["eligible"]]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible else "complete_no_eligible_candidates",
        "n_candidates": len(rows),
        "n_collapse_pass": sum(bool(row["collapse_pass"]) for row in rows),
        "n_eligible": len(eligible),
        "ranking": [
            "collapse_gate",
            "worst_quartile_mean_auroc",
            "macro_median_auroc",
            "pairing_selection_score",
        ],
        "best_candidate_id": eligible[0]["candidate_id"] if eligible else None,
        "best_candidate_spec_sha256": eligible[0]["spec_sha256"] if eligible else None,
        "proxy_candidate_id": rows[0]["candidate_id"],
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage1" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def _pareto_front(rows: Sequence[Mapping[str, Any]], objectives: Sequence[str]) -> set[int]:
    """Return candidate IDs not dominated when every objective is maximized."""
    front = set()
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["candidate_id"] == candidate["candidate_id"]:
                continue
            weakly_better = all(float(other[key]) >= float(candidate[key]) for key in objectives)
            strictly_better = any(float(other[key]) > float(candidate[key]) for key in objectives)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(int(candidate["candidate_id"]))
    return front


def _balance_improves(pairing: Mapping[str, Any]) -> bool:
    """Return false for unavailable SMDs and otherwise require both balances to improve."""
    names = (
        "value_smd_before_mean",
        "value_smd_after_mean",
        "occupancy_smd_before_mean",
        "occupancy_smd_after_mean",
    )
    values = [pairing.get(name) for name in names]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return False
    value_before, value_after, occupancy_before, occupancy_after = map(float, values)
    return value_after <= value_before and occupancy_after <= occupancy_before


def _balance_improvement(pairing: Mapping[str, Any]) -> float | None:
    """Return combined value/occupancy SMD improvement when all terms exist."""
    names = (
        "value_smd_before_mean",
        "value_smd_after_mean",
        "occupancy_smd_before_mean",
        "occupancy_smd_after_mean",
    )
    values = [pairing.get(name) for name in names]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    value_before, value_after, occupancy_before, occupancy_after = map(float, values)
    return (value_before - value_after) + (occupancy_before - occupancy_after)


def collect_stage2(root: Path) -> Path:
    """Authenticate Stage-2 results and report gates plus a three-objective Pareto set."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage2"]["candidates"]:
        result_path = root / "stage2" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-2 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_campaign_id") != spec["source_campaign_id"]
            or result.get("source_candidate_id") != spec["source_candidate_id"]
        ):
            raise ValueError(f"Stage-2 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-2 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        collapse_pass = bool(pairing["collapse_pass"])
        finite_pass = float(pairing["embedding_finite_fraction"]) == 1.0
        balance_pass = _balance_improves(pairing)
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "source_candidate_id": spec["source_candidate_id"],
                "rationale": spec["rationale"],
                "seed": spec["seed"],
                "collapse_eligible": collapse_pass and finite_pass,
                "collapse_pass": collapse_pass,
                "collapse_failures": ";".join(pairing["collapse_failures"]),
                "balance_pass": balance_pass,
                "embedding_finite_fraction": pairing["embedding_finite_fraction"],
                "embedding_active_fraction": pairing["embedding_active_fraction"],
                "embedding_effective_rank": pairing["embedding_effective_rank"],
                "embedding_participation_rank": pairing["embedding_participation_rank"],
                "embedding_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "raw_selection_score": pairing["raw_selection_score"],
                "gated_selection_score": pairing["selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "value_smd_before_mean": pairing["value_smd_before_mean"],
                "value_smd_after_mean": pairing["value_smd_after_mean"],
                "occupancy_smd_before_mean": pairing["occupancy_smd_before_mean"],
                "occupancy_smd_after_mean": pairing["occupancy_smd_after_mean"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": result["training_metrics"]["train/loss_mean"],
                "params_json": _canonical_json(spec["params"]),
                "spec_sha256": spec["spec_sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 2 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )
    objectives = (
        "embedding_effective_rank",
        "raw_selection_score",
        "worst_quartile_mean_auroc",
    )
    front = _pareto_front(rows, objectives)
    for row in rows:
        row["pareto_nondominated"] = row["candidate_id"] in front
    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage2" / "summary.csv"
    _atomic_csv(table, rows)
    eligible_ids = [int(row["candidate_id"]) for row in rows if row["collapse_eligible"]]
    balance_ids = [int(row["candidate_id"]) for row in rows if row["balance_pass"]]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible_ids else "complete_no_collapse_eligible_candidates",
        "n_candidates": len(rows),
        "n_collapse_eligible": len(eligible_ids),
        "collapse_eligible_candidate_ids": eligible_ids,
        "n_balance_pass": len(balance_ids),
        "balance_pass_candidate_ids": balance_ids,
        "pareto_objectives": list(objectives),
        "pareto_candidate_ids": sorted(front),
        "selection_policy": "No scalar winner is selected; preserve the validation Pareto set.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage2" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def collect_stage3(root: Path) -> Path:
    """Report encoder/projector gates and the Stage-3 four-objective Pareto set."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage3"]["candidates"]:
        result_path = root / "stage3" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-3 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_campaign_id") != STAGE3_SOURCE_CAMPAIGN
            or result.get("source_candidate_id") != STAGE3_SOURCE_CANDIDATE_ID
            or result.get("source_candidate_spec_sha256") != STAGE3_SOURCE_SPEC_SHA256
        ):
            raise ValueError(f"Stage-3 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-3 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        encoder_pass = bool(pairing["collapse_pass"])
        projector_pass = bool(pairing["projector_collapse_pass"])
        encoder_finite = float(pairing["embedding_finite_fraction"]) == 1.0
        projector_finite = float(pairing["projector_embedding_finite_fraction"]) == 1.0
        improvement = _balance_improvement(pairing)
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "seed": spec["seed"],
                "encoder_collapse_pass": encoder_pass,
                "encoder_collapse_failures": ";".join(pairing["collapse_failures"]),
                "projector_collapse_pass": projector_pass,
                "projector_collapse_failures": ";".join(pairing["projector_collapse_failures"]),
                "collapse_eligible": (
                    encoder_pass and projector_pass and encoder_finite and projector_finite
                ),
                "balance_pass": _balance_improves(pairing),
                "balance_improvement": improvement,
                "encoder_effective_rank": pairing["embedding_effective_rank"],
                "encoder_participation_rank": pairing["embedding_participation_rank"],
                "encoder_active_fraction": pairing["embedding_active_fraction"],
                "encoder_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "projector_effective_rank": pairing["projector_embedding_effective_rank"],
                "projector_participation_rank": pairing["projector_embedding_participation_rank"],
                "projector_active_fraction": pairing["projector_embedding_active_fraction"],
                "projector_top_pc_fraction": pairing["projector_embedding_top_pc_fraction"],
                "raw_selection_score": pairing["raw_selection_score"],
                "gated_selection_score": pairing["selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "value_smd_before_mean": pairing["value_smd_before_mean"],
                "value_smd_after_mean": pairing["value_smd_after_mean"],
                "occupancy_smd_before_mean": pairing["occupancy_smd_before_mean"],
                "occupancy_smd_after_mean": pairing["occupancy_smd_after_mean"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": result["training_metrics"]["train/loss_mean"],
                "architecture_json": _canonical_json(spec["architecture_params"]),
                "spec_sha256": spec["spec_sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 3 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )
    objectives = (
        "encoder_effective_rank",
        "raw_selection_score",
        "worst_quartile_mean_auroc",
        "balance_improvement",
    )
    pareto_rows = [row for row in rows if row["balance_improvement"] is not None]
    front = _pareto_front(pareto_rows, objectives) if pareto_rows else set()
    for row in rows:
        row["pareto_nondominated"] = row["candidate_id"] in front
    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage3" / "summary.csv"
    _atomic_csv(table, rows)
    eligible_ids = [int(row["candidate_id"]) for row in rows if row["collapse_eligible"]]
    balance_ids = [int(row["candidate_id"]) for row in rows if row["balance_pass"]]
    missing_balance = [
        int(row["candidate_id"]) for row in rows if row["balance_improvement"] is None
    ]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible_ids else "complete_no_collapse_eligible_candidates",
        "n_candidates": len(rows),
        "n_collapse_eligible": len(eligible_ids),
        "collapse_eligible_candidate_ids": eligible_ids,
        "n_balance_pass": len(balance_ids),
        "balance_pass_candidate_ids": balance_ids,
        "pareto_objectives": list(objectives),
        "pareto_candidate_ids": sorted(front),
        "pareto_excluded_missing_balance_candidate_ids": missing_balance,
        "selection_policy": "No scalar winner is selected; preserve the validation Pareto set.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage3" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def _valid_anomaly_aurocs(anomaly: Mapping[str, Any]) -> bool:
    """Return whether every aggregate and per-dataset AUROC is finite and bounded."""
    aggregate_names = ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc")
    values = [anomaly.get(name) for name in aggregate_names]
    per_dataset = anomaly.get("per_dataset")
    if not isinstance(per_dataset, dict) or not per_dataset:
        return False
    values.extend(
        metrics.get("auroc") if isinstance(metrics, Mapping) else None
        for metrics in per_dataset.values()
    )
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
        for value in values
    )


def collect_stage4(root: Path) -> Path:
    """Authenticate Stage-4 ablations and report paired deltas plus a Pareto set."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage4"]["candidates"]:
        result_path = root / "stage4" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-4 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_campaign_id") != STAGE4_SOURCE_CAMPAIGN
            or result.get("source_candidate_id") != spec["source_candidate_id"]
            or result.get("source_candidate_spec_sha256") != spec["source_candidate_spec_sha256"]
        ):
            raise ValueError(f"Stage-4 result identity mismatch: {result_path}")
        expected_artifacts = {"training_csv", "pairing_json", "anomaly_json", "last_checkpoint"}
        if set(result.get("artifacts", {})) != expected_artifacts:
            raise ValueError(f"Stage-4 result artifact inventory mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-4 artifact mismatch: {path}")
        checkpoint = Path(result["artifacts"]["last_checkpoint"]["path"])
        if checkpoint.name != "last.ckpt":
            raise ValueError(f"Stage-4 handoff checkpoint must be last.ckpt: {checkpoint}")

        training = result["training_metrics"]
        _validate_stage4_training_metrics(
            training, spec, Path(result["artifacts"]["training_csv"]["path"])
        )
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        encoder_pass = bool(pairing["collapse_pass"])
        projector_pass = bool(pairing["projector_collapse_pass"])
        encoder_finite = float(pairing["embedding_finite_fraction"]) == 1.0
        projector_finite = float(pairing["projector_embedding_finite_fraction"]) == 1.0
        auroc_valid = _valid_anomaly_aurocs(anomaly)
        mnn_nonzero = float(pairing["mnn_coverage"]) > 0.0
        improvement = _balance_improvement(pairing)
        weighted_regularization = float(training["train/loss_encoder_variance_weighted"]) + float(
            training["train/loss_encoder_covariance_weighted"]
        )
        total_loss = float(training["train/loss_mean"])
        regularized_fraction = weighted_regularization / total_loss if total_loss > 0.0 else None
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "source_architecture_id": spec["source_candidate_id"],
                "source_architecture_name": spec["source_candidate_name"],
                "is_architecture_control": spec["is_architecture_control"],
                "seed": spec["seed"],
                "variance_weight": spec["regularization_params"][
                    "algorithm.encoder_variance_weight"
                ],
                "covariance_weight": spec["regularization_params"][
                    "algorithm.encoder_covariance_weight"
                ],
                "scientific_eligible": (
                    encoder_pass
                    and projector_pass
                    and encoder_finite
                    and projector_finite
                    and mnn_nonzero
                    and auroc_valid
                ),
                "encoder_collapse_pass": encoder_pass,
                "encoder_collapse_failures": ";".join(pairing["collapse_failures"]),
                "projector_collapse_pass": projector_pass,
                "projector_collapse_failures": ";".join(pairing["projector_collapse_failures"]),
                "encoder_finite": encoder_finite,
                "projector_finite": projector_finite,
                "mnn_nonzero": mnn_nonzero,
                "auroc_valid": auroc_valid,
                "balance_pass": _balance_improves(pairing),
                "balance_improvement": improvement,
                "encoder_effective_rank": pairing["embedding_effective_rank"],
                "encoder_participation_rank": pairing["embedding_participation_rank"],
                "encoder_active_fraction": pairing["embedding_active_fraction"],
                "encoder_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "projector_effective_rank": pairing["projector_embedding_effective_rank"],
                "projector_participation_rank": pairing["projector_embedding_participation_rank"],
                "projector_active_fraction": pairing["projector_embedding_active_fraction"],
                "projector_top_pc_fraction": pairing["projector_embedding_top_pc_fraction"],
                "raw_selection_score": pairing["raw_selection_score"],
                "gated_selection_score": pairing["selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "value_smd_before_mean": pairing["value_smd_before_mean"],
                "value_smd_after_mean": pairing["value_smd_after_mean"],
                "occupancy_smd_before_mean": pairing["occupancy_smd_before_mean"],
                "occupancy_smd_after_mean": pairing["occupancy_smd_after_mean"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": total_loss,
                "train_loss_ntxent": training["train/loss_ntxent"],
                "train_loss_encoder_variance": training["train/loss_encoder_variance"],
                "train_loss_encoder_covariance": training["train/loss_encoder_covariance"],
                "train_loss_encoder_variance_weighted": training[
                    "train/loss_encoder_variance_weighted"
                ],
                "train_loss_encoder_covariance_weighted": training[
                    "train/loss_encoder_covariance_weighted"
                ],
                "regularized_objective_fraction": regularized_fraction,
                "regularization_dominates_objective": (
                    regularized_fraction is not None and regularized_fraction > 0.5
                ),
                "params_json": _canonical_json(spec["params"]),
                "spec_sha256": spec["spec_sha256"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": result["artifacts"]["last_checkpoint"]["sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 4 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )

    controls = {
        int(row["source_architecture_id"]): row
        for row in rows
        if bool(row["is_architecture_control"])
    }
    if set(controls) != {item[0] for item in STAGE4_SOURCE_ARCHITECTURES}:
        raise ValueError("Stage-4 design must contain one zero-weight control per architecture.")
    objectives = (
        "encoder_effective_rank",
        "raw_selection_score",
        "worst_quartile_mean_auroc",
        "balance_improvement",
    )
    for row in rows:
        control = controls[int(row["source_architecture_id"])]
        row["control_candidate_id"] = control["candidate_id"]
        for objective in objectives:
            value = row[objective]
            control_value = control[objective]
            row[f"delta_vs_control_{objective}"] = (
                None
                if value is None or control_value is None
                else float(value) - float(control_value)
            )

    pareto_rows = [
        row
        for row in rows
        if bool(row["scientific_eligible"])
        and all(row[objective] is not None for objective in objectives)
    ]
    front = _pareto_front(pareto_rows, objectives) if pareto_rows else set()
    for row in rows:
        row["pareto_nondominated"] = row["candidate_id"] in front
    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage4" / "summary.csv"
    _atomic_csv(table, rows)
    eligible_ids = [int(row["candidate_id"]) for row in rows if row["scientific_eligible"]]
    balance_ids = [int(row["candidate_id"]) for row in rows if row["balance_pass"]]
    dominated_ids = [
        int(row["candidate_id"]) for row in rows if row["regularization_dominates_objective"]
    ]
    missing_balance = [
        int(row["candidate_id"]) for row in rows if row["balance_improvement"] is None
    ]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible_ids else "complete_no_scientifically_eligible_candidates",
        "n_candidates": len(rows),
        "n_scientifically_eligible": len(eligible_ids),
        "scientifically_eligible_candidate_ids": eligible_ids,
        "scientific_eligibility": (
            "finite encoder/projector, both collapse gates, nonzero MNN coverage, valid AUROC"
        ),
        "balance_is_scientific_eligibility_gate": False,
        "n_balance_pass": len(balance_ids),
        "balance_pass_candidate_ids": balance_ids,
        "architecture_control_candidate_ids": {
            str(source_id): int(control["candidate_id"])
            for source_id, control in sorted(controls.items())
        },
        "regularization_dominates_objective_candidate_ids": dominated_ids,
        "regularization_domination_threshold": 0.5,
        "pareto_objectives": list(objectives),
        "pareto_candidate_ids": sorted(front),
        "pareto_excluded_missing_balance_candidate_ids": missing_balance,
        "selection_policy": "No scalar winner is selected; preserve the validation Pareto set.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage4" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def _stage5_result_row(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    origin: str,
    result_sha256: str,
) -> dict[str, Any]:
    """Build one authenticated confirmation row without discarding its identities."""
    pairing = result["pairing_metrics"]
    anomaly = result["anomaly_metrics"]
    training = result["training_metrics"]
    encoder_finite = float(pairing["embedding_finite_fraction"]) == 1.0
    projector_finite = float(pairing["projector_embedding_finite_fraction"]) == 1.0
    scientifically_eligible = (
        bool(pairing["collapse_pass"])
        and bool(pairing["projector_collapse_pass"])
        and encoder_finite
        and projector_finite
        and float(pairing["mnn_coverage"]) > 0.0
        and _valid_anomaly_aurocs(anomaly)
    )
    artifacts = result["artifacts"]
    return {
        "origin": origin,
        "candidate_id": spec["candidate_id"],
        "stage4_config_id": (
            spec["candidate_id"] if origin == "stage4_seed123" else spec["source_candidate_id"]
        ),
        "configuration_name": (
            spec["name"] if origin == "stage4_seed123" else spec["source_candidate_name"]
        ),
        "architecture_id": spec["source_architecture_id"]
        if origin != "stage4_seed123"
        else spec["source_candidate_id"],
        "architecture_name": spec["source_architecture_name"]
        if origin != "stage4_seed123"
        else spec["source_candidate_name"],
        "is_architecture_control": spec["is_architecture_control"],
        "seed": spec["seed"],
        "scientifically_eligible": scientifically_eligible,
        "balance_pass": _balance_improves(pairing),
        "encoder_effective_rank": pairing["embedding_effective_rank"],
        "encoder_participation_rank": pairing["embedding_participation_rank"],
        "encoder_top_pc_fraction": pairing["embedding_top_pc_fraction"],
        "projector_effective_rank": pairing["projector_embedding_effective_rank"],
        "raw_selection_score": pairing["raw_selection_score"],
        "macro_mean_auroc": anomaly["macro_mean_auroc"],
        "macro_median_auroc": anomaly["macro_median_auroc"],
        "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
        "train_loss": training["train/loss_mean"],
        "spec_sha256": spec["spec_sha256"],
        "result_payload_sha256": result_sha256,
        "checkpoint_path": artifacts["last_checkpoint"]["path"],
        "checkpoint_sha256": artifacts["last_checkpoint"]["sha256"],
        "artifact_hashes_json": _canonical_json(
            {name: artifact["sha256"] for name, artifact in sorted(artifacts.items())}
        ),
    }


def _authenticate_stage5_result(
    result_path: Path,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    source: bool,
) -> tuple[dict[str, Any], str]:
    """Authenticate one Stage-4 or Stage-5 result and every referenced artifact."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    digest = result.pop("result_payload_sha256", None)
    if digest is None or _value_sha256(result) != digest:
        raise ValueError(f"Stage-5 input result fingerprint mismatch: {result_path}")
    expected_source_campaign = STAGE4_SOURCE_CAMPAIGN if source else STAGE5_SOURCE_CAMPAIGN
    expected_source_id = spec["source_candidate_id"]
    expected_source_sha = spec["source_candidate_spec_sha256"]
    if (
        result.get("campaign_id") != manifest["campaign_id"]
        or result.get("git_commit") != manifest["git"]["commit"]
        or result.get("candidate_id") != spec["candidate_id"]
        or result.get("spec_sha256") != spec["spec_sha256"]
        or result.get("source_campaign_id") != expected_source_campaign
        or result.get("source_candidate_id") != expected_source_id
        or result.get("source_candidate_spec_sha256") != expected_source_sha
    ):
        raise ValueError(f"Stage-5 input result identity mismatch: {result_path}")
    expected_artifacts = {"training_csv", "pairing_json", "anomaly_json", "last_checkpoint"}
    if set(result.get("artifacts", {})) != expected_artifacts:
        raise ValueError(f"Stage-5 input artifact inventory mismatch: {result_path}")
    for artifact in result["artifacts"].values():
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Stage-5 input artifact mismatch: {path}")
    checkpoint = Path(result["artifacts"]["last_checkpoint"]["path"])
    if checkpoint.name != "last.ckpt":
        raise ValueError(f"Stage-5 handoff checkpoint must be last.ckpt: {checkpoint}")
    _validate_stage4_training_metrics(
        result["training_metrics"], spec, Path(result["artifacts"]["training_csv"]["path"])
    )
    return result, digest


def collect_stage5(root: Path) -> Path:
    """Confirm two hybrid recipes against matched controls over three seeds."""
    root = root.resolve()
    manifest = _load_campaign(root)
    if _sha256(STAGE5_SOURCE_SUMMARY) != STAGE5_SOURCE_SUMMARY_SHA256:
        raise ValueError("Stage-5 source summary fingerprint mismatch.")
    if _sha256(STAGE5_SOURCE_SUMMARY_CSV) != STAGE5_SOURCE_SUMMARY_CSV_SHA256:
        raise ValueError("Stage-5 source metric table fingerprint mismatch.")
    source_manifest = _load_campaign(STAGE5_SOURCE_ROOT)

    rows: list[dict[str, Any]] = []
    for source_id, source_name, source_sha256 in STAGE5_SOURCE_CANDIDATES:
        spec = source_manifest["stage4"]["candidates"][source_id]
        if spec["name"] != source_name or spec["spec_sha256"] != source_sha256:
            raise ValueError("Stage-5 frozen seed-123 source identity mismatch.")
        result_path = STAGE5_SOURCE_ROOT / "stage4" / f"candidate_{source_id:03d}" / "result.json"
        result, digest = _authenticate_stage5_result(
            result_path, source_manifest, spec, source=True
        )
        rows.append(_stage5_result_row(result, spec, "stage4_seed123", digest))

    missing = []
    for spec in manifest["stage5"]["candidates"]:
        result_path = root / "stage5" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result, digest = _authenticate_stage5_result(result_path, manifest, spec, source=False)
        rows.append(_stage5_result_row(result, spec, "stage5_fresh", digest))
    if missing:
        raise FileNotFoundError(
            f"Stage 5 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )

    paired: dict[str, list[dict[str, Any]]] = {}
    for architecture_name in ("layers2", "official_projector"):
        paired[architecture_name] = []
        architecture_rows = [row for row in rows if row["architecture_name"] == architecture_name]
        for seed in (STAGE4_SEED, *STAGE5_SEEDS):
            seed_rows = [row for row in architecture_rows if int(row["seed"]) == seed]
            controls = [row for row in seed_rows if row["is_architecture_control"]]
            hybrids = [row for row in seed_rows if not row["is_architecture_control"]]
            if len(controls) != 1 or len(hybrids) != 1:
                raise ValueError(
                    f"Stage-5 matched pair is incomplete for {architecture_name}/{seed}."
                )
            control, hybrid = controls[0], hybrids[0]
            raw_control = float(control["raw_selection_score"])
            delta = {
                "architecture_name": architecture_name,
                "seed": seed,
                "control_stage4_config_id": control["stage4_config_id"],
                "hybrid_stage4_config_id": hybrid["stage4_config_id"],
                "macro_mean_auroc_delta": float(hybrid["macro_mean_auroc"])
                - float(control["macro_mean_auroc"]),
                "worst_quartile_mean_auroc_delta": float(hybrid["worst_quartile_mean_auroc"])
                - float(control["worst_quartile_mean_auroc"]),
                "raw_selection_score_relative_delta": (
                    (float(hybrid["raw_selection_score"]) - raw_control) / abs(raw_control)
                    if raw_control != 0.0
                    else None
                ),
                "encoder_effective_rank_relative_delta": (
                    float(hybrid["encoder_effective_rank"])
                    - float(control["encoder_effective_rank"])
                )
                / float(control["encoder_effective_rank"]),
                "encoder_participation_rank_relative_delta": (
                    float(hybrid["encoder_participation_rank"])
                    - float(control["encoder_participation_rank"])
                )
                / float(control["encoder_participation_rank"]),
                "encoder_top_pc_fraction_delta": float(hybrid["encoder_top_pc_fraction"])
                - float(control["encoder_top_pc_fraction"]),
                "pair_scientifically_eligible": bool(control["scientifically_eligible"])
                and bool(hybrid["scientifically_eligible"]),
            }
            paired[architecture_name].append(delta)
            for row in (control, hybrid):
                row["paired_role"] = "control" if row is control else "hybrid"
                for key, value in delta.items():
                    if key not in {"architecture_name", "seed"}:
                        row[f"paired_{key}"] = (
                            0.0 if row is control and isinstance(value, float) else value
                        )

    confirmations = {}
    for architecture_name, deltas in paired.items():
        numeric = (
            "macro_mean_auroc_delta",
            "worst_quartile_mean_auroc_delta",
            "raw_selection_score_relative_delta",
            "encoder_effective_rank_relative_delta",
            "encoder_participation_rank_relative_delta",
            "encoder_top_pc_fraction_delta",
        )
        means = {key: sum(float(row[key]) for row in deltas) / len(deltas) for key in numeric}
        all_seed_noninferior = all(
            float(row["macro_mean_auroc_delta"]) >= -0.01
            and float(row["worst_quartile_mean_auroc_delta"]) >= -0.01
            and row["raw_selection_score_relative_delta"] is not None
            and float(row["raw_selection_score_relative_delta"]) >= -0.05
            for row in deltas
        )
        mean_noninferior = (
            means["macro_mean_auroc_delta"] >= -0.01
            and means["worst_quartile_mean_auroc_delta"] >= -0.01
            and means["raw_selection_score_relative_delta"] >= -0.05
        )
        material_geometry = (
            means["encoder_effective_rank_relative_delta"] >= 0.10
            or means["encoder_participation_rank_relative_delta"] >= 0.10
            or means["encoder_top_pc_fraction_delta"] <= -0.03
        )
        all_eligible = all(bool(row["pair_scientifically_eligible"]) for row in deltas)
        confirmations[architecture_name] = {
            "seeds": [int(row["seed"]) for row in deltas],
            "mean_deltas": means,
            "all_seed_noninferior": all_seed_noninferior,
            "mean_noninferior": mean_noninferior,
            "material_geometry": material_geometry,
            "all_pairs_scientifically_eligible": all_eligible,
            "promotion": all_eligible
            and all_seed_noninferior
            and mean_noninferior
            and material_geometry,
        }

    rows.sort(
        key=lambda row: (
            str(row["architecture_name"]),
            int(row["seed"]),
            bool(row["is_architecture_control"]),
        )
    )
    table = root / "stage5" / "summary.csv"
    _atomic_csv(table, rows)
    paired_rows = [row for values in paired.values() for row in values]
    paired_table = root / "stage5" / "paired_deltas.csv"
    _atomic_csv(paired_table, paired_rows)
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete",
        "n_seed_rows": len(rows),
        "seeds": [STAGE4_SEED, *STAGE5_SEEDS],
        "confirmations": confirmations,
        "promotion_policy": {
            "macro_mean_auroc_delta_min": -0.01,
            "worst_quartile_mean_auroc_delta_min": -0.01,
            "raw_selection_score_relative_delta_min": -0.05,
            "material_encoder_rank_relative_delta_min": 0.10,
            "material_top_pc_fraction_delta_max": -0.03,
            "requires_every_seed_noninferior": True,
        },
        "selection_policy": "No scalar global winner is selected; promotions are per architecture.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
        "paired_deltas_csv": str(paired_table),
        "paired_deltas_csv_sha256": _sha256(paired_table),
        "source_summary_sha256": STAGE5_SOURCE_SUMMARY_SHA256,
        "source_summary_csv_sha256": STAGE5_SOURCE_SUMMARY_CSV_SHA256,
    }
    output = root / "stage5" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def collect_stage6(root: Path) -> Path:
    """Authenticate long-horizon results and report epoch-16 matched comparisons."""
    root = root.resolve()
    manifest = _load_campaign(root)
    source_files = (
        (STAGE6_SOURCE_SUMMARY, STAGE6_SOURCE_SUMMARY_SHA256),
        (STAGE6_SOURCE_SUMMARY_CSV, STAGE6_SOURCE_SUMMARY_CSV_SHA256),
        (STAGE6_SOURCE_PAIRED_CSV, STAGE6_SOURCE_PAIRED_CSV_SHA256),
    )
    if any(_sha256(path) != digest for path, digest in source_files):
        raise ValueError("Stage-6 source artifact fingerprint mismatch.")
    source_summary = json.loads(STAGE6_SOURCE_SUMMARY.read_text(encoding="utf-8"))
    source_promotions = {
        name: source_summary.get("confirmations", {}).get(name, {}).get("promotion") is True
        for name in ("layers2", "official_projector")
    }
    if not all(source_promotions.values()):
        raise ValueError("Stage 6 requires both authenticated Stage-5 promotions.")

    rows = []
    missing = []
    for spec in manifest["stage6"]["candidates"]:
        result_path = root / "stage6" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_digest = result.pop("result_payload_sha256", None)
        if result_digest is None or _value_sha256(result) != result_digest:
            raise ValueError(f"Stage-6 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_campaign_id") != STAGE6_SOURCE_CAMPAIGN
            or result.get("source_candidate_id") != spec["source_candidate_id"]
            or result.get("source_candidate_spec_sha256") != spec["source_candidate_spec_sha256"]
        ):
            raise ValueError(f"Stage-6 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-6 artifact mismatch: {path}")
        inventory = result.get("checkpoint_inventory")
        if not isinstance(inventory, list) or len(inventory) != STAGE6_EPOCHS:
            raise ValueError(f"Stage-6 checkpoint inventory is incomplete: {result_path}")
        completed_epochs = [item.get("completed_epoch") for item in inventory]
        if completed_epochs != list(range(1, STAGE6_EPOCHS + 1)):
            raise ValueError(f"Stage-6 checkpoint epoch inventory is malformed: {result_path}")
        for item in inventory:
            path = Path(item["path"])
            expected_name = f"epoch-{int(item['epoch_index']):02d}.ckpt"
            if path.name != expected_name or not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError(f"Stage-6 epoch checkpoint mismatch: {path}")
        milestones = [item for item in inventory if item.get("is_milestone")]
        if [item["completed_epoch"] for item in milestones] != list(STAGE6_MILESTONES):
            raise ValueError(f"Stage-6 milestone inventory is malformed: {result_path}")

        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        row = {
            "candidate_id": spec["candidate_id"],
            "configuration_name": spec["source_candidate_name"],
            "architecture_name": spec["source_architecture_name"],
            "is_architecture_control": spec["is_architecture_control"],
            "seed": spec["seed"],
            "scientifically_eligible": bool(pairing["collapse_pass"])
            and bool(pairing["projector_collapse_pass"])
            and float(pairing["embedding_finite_fraction"]) == 1.0
            and float(pairing["projector_embedding_finite_fraction"]) == 1.0
            and float(pairing["mnn_coverage"]) > 0.0
            and _valid_anomaly_aurocs(anomaly),
            "encoder_effective_rank": pairing["embedding_effective_rank"],
            "encoder_participation_rank": pairing["embedding_participation_rank"],
            "encoder_top_pc_fraction": pairing["embedding_top_pc_fraction"],
            "raw_selection_score": pairing["raw_selection_score"],
            "macro_mean_auroc": anomaly["macro_mean_auroc"],
            "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
            "canonical_last_checkpoint_sha256": result["artifacts"]["last_checkpoint"]["sha256"],
            "milestone_checkpoint_hashes_json": _canonical_json(
                {str(item["completed_epoch"]): item["sha256"] for item in milestones}
            ),
            "all_epoch_checkpoint_hashes_json": _canonical_json(
                {str(item["completed_epoch"]): item["sha256"] for item in inventory}
            ),
            "milestone_evaluation_ready": True,
            "spec_sha256": spec["spec_sha256"],
            "result_payload_sha256": result_digest,
            "result_path": str(result_path),
        }
        rows.append(row)
    if missing:
        raise FileNotFoundError(
            f"Stage 6 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )

    comparisons = []
    for architecture in ("layers2", "official_projector"):
        for seed in STAGE6_SEEDS:
            selected = [
                row
                for row in rows
                if row["architecture_name"] == architecture and int(row["seed"]) == seed
            ]
            controls = [row for row in selected if row["is_architecture_control"]]
            hybrids = [row for row in selected if not row["is_architecture_control"]]
            if len(controls) != 1 or len(hybrids) != 1:
                raise ValueError(f"Stage-6 matched pair is incomplete for {architecture}/{seed}.")
            control, hybrid = controls[0], hybrids[0]
            comparison = {
                "architecture_name": architecture,
                "seed": seed,
                "control_candidate_id": control["candidate_id"],
                "hybrid_candidate_id": hybrid["candidate_id"],
                "both_scientifically_eligible": bool(control["scientifically_eligible"])
                and bool(hybrid["scientifically_eligible"]),
            }
            for metric in (
                "encoder_effective_rank",
                "encoder_participation_rank",
                "encoder_top_pc_fraction",
                "raw_selection_score",
                "macro_mean_auroc",
                "worst_quartile_mean_auroc",
            ):
                comparison[f"{metric}_delta"] = float(hybrid[metric]) - float(control[metric])
            comparisons.append(comparison)

    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage6" / "summary.csv"
    _atomic_csv(table, rows)
    comparisons_table = root / "stage6" / "paired_epoch16.csv"
    _atomic_csv(comparisons_table, comparisons)
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete",
        "n_candidates": len(rows),
        "epoch16_pair_count": len(comparisons),
        "milestone_epochs": list(STAGE6_MILESTONES),
        "all_candidates_milestone_evaluation_ready": all(
            row["milestone_evaluation_ready"] for row in rows
        ),
        "source_promotions": source_promotions,
        "selection_policy": "No scalar winner is selected in the long-horizon pilot.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
        "paired_epoch16_csv": str(comparisons_table),
        "paired_epoch16_csv_sha256": _sha256(comparisons_table),
    }
    output = root / "stage6" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def collect_stage7(root: Path) -> Path:
    """Aggregate milestone utility and select one global validation horizon."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows = []
    missing = []
    for spec in manifest["stage7"]["candidates"]:
        result_path = root / "stage7" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-7 result fingerprint mismatch: {result_path}")
        if (
            result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_result_payload_sha256") != spec["source_result_payload_sha256"]
            or result.get("source_checkpoint_sha256") != spec["source_checkpoint_sha256"]
        ):
            raise ValueError(f"Stage-7 result identity mismatch: {result_path}")
        source_checkpoint = Path(spec["source_checkpoint_path"])
        if (
            not source_checkpoint.is_file()
            or _sha256(source_checkpoint) != spec["source_checkpoint_sha256"]
        ):
            raise ValueError(f"Stage-7 source checkpoint mismatch: {source_checkpoint}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-7 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "source_candidate_id": spec["source_candidate_id"],
                "architecture_name": spec["source_architecture_name"],
                "is_architecture_control": spec["is_architecture_control"],
                "seed": spec["seed"],
                "completed_epoch": spec["completed_epoch"],
                "eligible": bool(pairing["collapse_pass"])
                and bool(pairing["projector_collapse_pass"])
                and float(pairing["embedding_finite_fraction"]) == 1.0
                and float(pairing["projector_embedding_finite_fraction"]) == 1.0
                and float(pairing["mnn_coverage"]) > 0.0
                and _valid_anomaly_aurocs(anomaly),
                "encoder_effective_rank": pairing["embedding_effective_rank"],
                "encoder_participation_rank": pairing["embedding_participation_rank"],
                "encoder_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "raw_selection_score": pairing["raw_selection_score"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "source_checkpoint_path": str(source_checkpoint),
                "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
                "result_payload_sha256": digest,
                "spec_sha256": spec["spec_sha256"],
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 7 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )

    paired = []
    for architecture in ("layers2", "official_projector"):
        for epoch in STAGE7_MILESTONES:
            for seed in STAGE6_SEEDS:
                selected = [
                    row
                    for row in rows
                    if row["architecture_name"] == architecture
                    and row["completed_epoch"] == epoch
                    and row["seed"] == seed
                ]
                controls = [row for row in selected if row["is_architecture_control"]]
                hybrids = [row for row in selected if not row["is_architecture_control"]]
                if len(controls) != 1 or len(hybrids) != 1:
                    raise ValueError(
                        f"Stage-7 matched pair missing: {architecture}/{epoch}/{seed}"
                    )
                control, hybrid = controls[0], hybrids[0]
                item = {
                    "architecture_name": architecture,
                    "completed_epoch": epoch,
                    "seed": seed,
                    "both_eligible": control["eligible"] and hybrid["eligible"],
                }
                for metric in (
                    "encoder_effective_rank",
                    "encoder_participation_rank",
                    "encoder_top_pc_fraction",
                    "raw_selection_score",
                    "macro_mean_auroc",
                    "worst_quartile_mean_auroc",
                ):
                    item[f"{metric}_delta"] = float(hybrid[metric]) - float(control[metric])
                paired.append(item)

    aggregates = []
    for architecture in ("layers2", "official_projector"):
        for epoch in STAGE7_MILESTONES:
            hybrid_rows = [
                row
                for row in rows
                if row["architecture_name"] == architecture
                and row["completed_epoch"] == epoch
                and not row["is_architecture_control"]
                and row["eligible"]
            ]
            values = [float(row["worst_quartile_mean_auroc"]) for row in hybrid_rows]
            aggregates.append(
                {
                    "architecture_name": architecture,
                    "completed_epoch": epoch,
                    "n_eligible_hybrids": len(values),
                    "median_worst_quartile_auroc": statistics.median(values) if values else None,
                    "mean_worst_quartile_auroc": statistics.mean(values) if values else None,
                    "se_worst_quartile_auroc": (
                        statistics.stdev(values) / math.sqrt(len(values))
                        if len(values) > 1
                        else None
                    ),
                }
            )

    epoch_stats = []
    for epoch in STAGE7_MILESTONES:
        values = [
            float(row["worst_quartile_mean_auroc"])
            for row in rows
            if row["completed_epoch"] == epoch
            and not row["is_architecture_control"]
            and row["eligible"]
        ]
        epoch_stats.append(
            {
                "completed_epoch": epoch,
                "n_eligible_hybrids": len(values),
                "median": statistics.median(values) if values else None,
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
                "standard_error": (
                    statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
                ),
            }
        )
    expected_hybrids = 2 * len(STAGE6_SEEDS)
    eligible_stats = [
        item
        for item in epoch_stats
        if item["median"] is not None and int(item["n_eligible_hybrids"]) == expected_hybrids
    ]
    if not eligible_stats:
        raise ValueError("No milestone has every hybrid seed scientifically eligible.")
    best = max(eligible_stats, key=lambda item: float(item["median"]))
    threshold = float(best["median"]) - float(best["standard_error"] or 0.0)
    selected_epoch = min(
        int(item["completed_epoch"])
        for item in eligible_stats
        if float(item["median"]) >= threshold
    )

    improvements = []
    extension_pairs_eligible = True
    for row16 in [
        row for row in rows if row["completed_epoch"] == 16 and not row["is_architecture_control"]
    ]:
        row8 = next(
            row
            for row in rows
            if row["completed_epoch"] == 8
            and row["source_candidate_id"] == row16["source_candidate_id"]
            and row["seed"] == row16["seed"]
        )
        extension_pairs_eligible = (
            extension_pairs_eligible and bool(row8["eligible"]) and bool(row16["eligible"])
        )
        improvements.append(
            float(row16["worst_quartile_mean_auroc"]) - float(row8["worst_quartile_mean_auroc"])
        )
    improvement_mean = statistics.mean(improvements)
    improvement_sd = statistics.stdev(improvements)
    improvement_se = improvement_sd / math.sqrt(len(improvements))
    extension_to_32 = extension_pairs_eligible and improvement_mean > improvement_se

    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage7" / "summary.csv"
    paired_table = root / "stage7" / "paired_deltas.csv"
    aggregate_table = root / "stage7" / "architecture_epoch.csv"
    _atomic_csv(table, rows)
    _atomic_csv(paired_table, paired)
    _atomic_csv(aggregate_table, aggregates)
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete",
        "n_evaluations": len(rows),
        "global_epoch_selection": {
            "metric": "eligible hybrid worst_quartile_mean_auroc",
            "epoch_statistics": epoch_stats,
            "best_median_epoch": best["completed_epoch"],
            "best_median": best["median"],
            "best_epoch_standard_error": best["standard_error"],
            "one_se_threshold": threshold,
            "rule": "smallest epoch whose median is at least best median minus best-epoch SE",
            "selected_epoch": selected_epoch,
        },
        "epoch8_to16": {
            "paired_improvements": improvements,
            "mean_improvement": improvement_mean,
            "standard_deviation": improvement_sd,
            "standard_error": improvement_se,
            "all_epoch8_and16_hybrids_eligible": extension_pairs_eligible,
            "extension_to_epoch32": extension_to_32,
            "decision_rule": "extend iff mean paired improvement exceeds its standard error",
        },
        "selection_policy": "One global epoch is selected; no scalar architecture winner.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
        "paired_deltas_csv": str(paired_table),
        "paired_deltas_csv_sha256": _sha256(paired_table),
        "architecture_epoch_csv": str(aggregate_table),
        "architecture_epoch_csv_sha256": _sha256(aggregate_table),
    }
    output = root / "stage7" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def status(root: Path) -> dict[str, Any]:
    """Report trial completion state without mutating the campaign."""
    root = root.resolve()
    manifest = _load_campaign(root)
    trials = []
    for spec in manifest["canary"]["trials"]:
        trial_root = root / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        trials.append({"trial_id": spec["trial_id"], "name": spec["name"], "state": state})
    stage1_trials = []
    for spec in manifest.get("stage1", {}).get("candidates", []):
        trial_root = root / "stage1" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage1_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage2_trials = []
    for spec in manifest.get("stage2", {}).get("candidates", []):
        trial_root = root / "stage2" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage2_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage3_trials = []
    for spec in manifest.get("stage3", {}).get("candidates", []):
        trial_root = root / "stage3" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage3_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage4_trials = []
    for spec in manifest.get("stage4", {}).get("candidates", []):
        trial_root = root / "stage4" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage4_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage5_trials = []
    for spec in manifest.get("stage5", {}).get("candidates", []):
        trial_root = root / "stage5" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage5_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage6_trials = []
    for spec in manifest.get("stage6", {}).get("candidates", []):
        trial_root = root / "stage6" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage6_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage7_trials = []
    for spec in manifest.get("stage7", {}).get("candidates", []):
        trial_root = root / "stage7" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage7_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    value = {
        "campaign_id": manifest["campaign_id"],
        "complete": all(item["state"] == "complete" for item in trials),
        "trials": trials,
        "canary": {
            "complete": all(item["state"] == "complete" for item in trials),
            "trials": trials,
        },
        "stage1": {
            "complete": bool(stage1_trials)
            and all(item["state"] == "complete" for item in stage1_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage1_trials),
            "n_total": len(stage1_trials),
            "trials": stage1_trials,
        },
        "stage2": {
            "complete": bool(stage2_trials)
            and all(item["state"] == "complete" for item in stage2_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage2_trials),
            "n_total": len(stage2_trials),
            "trials": stage2_trials,
        },
        "stage3": {
            "complete": bool(stage3_trials)
            and all(item["state"] == "complete" for item in stage3_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage3_trials),
            "n_total": len(stage3_trials),
            "trials": stage3_trials,
        },
        "stage4": {
            "complete": bool(stage4_trials)
            and all(item["state"] == "complete" for item in stage4_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage4_trials),
            "n_total": len(stage4_trials),
            "trials": stage4_trials,
        },
        "stage5": {
            "complete": bool(stage5_trials)
            and all(item["state"] == "complete" for item in stage5_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage5_trials),
            "n_total": len(stage5_trials),
            "trials": stage5_trials,
        },
        "stage6": {
            "complete": bool(stage6_trials)
            and all(item["state"] == "complete" for item in stage6_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage6_trials),
            "n_total": len(stage6_trials),
            "trials": stage6_trials,
        },
        "stage7": {
            "complete": bool(stage7_trials)
            and all(item["state"] == "complete" for item in stage7_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage7_trials),
            "n_total": len(stage7_trials),
            "trials": stage7_trials,
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


def main() -> None:
    """Dispatch the campaign command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--root", type=Path)
    init_parser.add_argument("--campaign-id")
    init_parser.add_argument("--deployment", type=Path)
    init_parser.add_argument("--source", type=Path, default=REPO_ROOT)
    init_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    init_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    init_parser.add_argument("--uv", type=Path, default=DEFAULT_UV)
    canary_parser = subparsers.add_parser("canary")
    canary_parser.add_argument("--root", type=Path, required=True)
    run_parser = subparsers.add_parser("run-trial")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--trial-id", type=int, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    stage1_parser = subparsers.add_parser("run-stage1")
    stage1_parser.add_argument("--root", type=Path, required=True)
    stage1_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage1_parser = subparsers.add_parser("collect-stage1")
    collect_stage1_parser.add_argument("--root", type=Path, required=True)
    stage1_status_parser = subparsers.add_parser("stage1-status")
    stage1_status_parser.add_argument("--root", type=Path, required=True)
    stage2_parser = subparsers.add_parser("run-stage2")
    stage2_parser.add_argument("--root", type=Path, required=True)
    stage2_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage2_parser = subparsers.add_parser("collect-stage2")
    collect_stage2_parser.add_argument("--root", type=Path, required=True)
    stage2_status_parser = subparsers.add_parser("stage2-status")
    stage2_status_parser.add_argument("--root", type=Path, required=True)
    stage3_parser = subparsers.add_parser("run-stage3")
    stage3_parser.add_argument("--root", type=Path, required=True)
    stage3_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage3_parser = subparsers.add_parser("collect-stage3")
    collect_stage3_parser.add_argument("--root", type=Path, required=True)
    stage3_status_parser = subparsers.add_parser("stage3-status")
    stage3_status_parser.add_argument("--root", type=Path, required=True)
    stage4_parser = subparsers.add_parser("run-stage4")
    stage4_parser.add_argument("--root", type=Path, required=True)
    stage4_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage4_parser = subparsers.add_parser("collect-stage4")
    collect_stage4_parser.add_argument("--root", type=Path, required=True)
    stage4_status_parser = subparsers.add_parser("stage4-status")
    stage4_status_parser.add_argument("--root", type=Path, required=True)
    stage5_parser = subparsers.add_parser("run-stage5")
    stage5_parser.add_argument("--root", type=Path, required=True)
    stage5_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage5_parser = subparsers.add_parser("collect-stage5")
    collect_stage5_parser.add_argument("--root", type=Path, required=True)
    stage5_status_parser = subparsers.add_parser("stage5-status")
    stage5_status_parser.add_argument("--root", type=Path, required=True)
    stage6_parser = subparsers.add_parser("run-stage6")
    stage6_parser.add_argument("--root", type=Path, required=True)
    stage6_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage6_parser = subparsers.add_parser("collect-stage6")
    collect_stage6_parser.add_argument("--root", type=Path, required=True)
    stage6_status_parser = subparsers.add_parser("stage6-status")
    stage6_status_parser.add_argument("--root", type=Path, required=True)
    stage7_parser = subparsers.add_parser("run-stage7")
    stage7_parser.add_argument("--root", type=Path, required=True)
    stage7_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage7_parser = subparsers.add_parser("collect-stage7")
    collect_stage7_parser.add_argument("--root", type=Path, required=True)
    stage7_status_parser = subparsers.add_parser("stage7-status")
    stage7_status_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "init":
        campaign_id = args.campaign_id
        if args.root is None and campaign_id is None:
            parser.error("init requires --root or --campaign-id")
        if campaign_id is None:
            campaign_id = args.root.name
        root = args.root or DEFAULT_CAMPAIGN_BASE / campaign_id
        deployment = args.deployment or DEFAULT_DEPLOYMENT_BASE / campaign_id
        launcher = initialize(
            root, deployment, args.source, args.data_dir, args.venv, args.uv, campaign_id
        )
        print(f"Campaign initialized. Review, then submit manually: sbatch {launcher}")
    elif args.command == "canary":
        manifest = validate_campaign(args.root.resolve())
        launcher = args.root.resolve() / "slurm" / "canary.sbatch"
        if not launcher.is_file():
            _write_launcher(args.root.resolve(), manifest)
        print(f"Campaign validation passed. Launcher: {launcher}")
    elif args.command == "run-trial":
        run_trial(args.root, args.trial_id)
    elif args.command == "collect":
        collect(args.root)
    elif args.command == "run-stage1":
        run_stage1(args.root, args.candidate_id)
    elif args.command == "collect-stage1":
        collect_stage1(args.root)
    elif args.command == "stage1-status":
        status(args.root)
    elif args.command == "run-stage2":
        run_stage2(args.root, args.candidate_id)
    elif args.command == "collect-stage2":
        collect_stage2(args.root)
    elif args.command == "stage2-status":
        status(args.root)
    elif args.command == "run-stage3":
        run_stage3(args.root, args.candidate_id)
    elif args.command == "collect-stage3":
        collect_stage3(args.root)
    elif args.command == "stage3-status":
        status(args.root)
    elif args.command == "run-stage4":
        run_stage4(args.root, args.candidate_id)
    elif args.command == "collect-stage4":
        collect_stage4(args.root)
    elif args.command == "stage4-status":
        status(args.root)
    elif args.command == "run-stage5":
        run_stage5(args.root, args.candidate_id)
    elif args.command == "collect-stage5":
        collect_stage5(args.root)
    elif args.command == "stage5-status":
        status(args.root)
    elif args.command == "run-stage6":
        run_stage6(args.root, args.candidate_id)
    elif args.command == "collect-stage6":
        collect_stage6(args.root)
    elif args.command == "stage6-status":
        status(args.root)
    elif args.command == "run-stage7":
        run_stage7(args.root, args.candidate_id)
    elif args.command == "collect-stage7":
        collect_stage7(args.root)
    elif args.command == "stage7-status":
        status(args.root)
    elif args.command == "status":
        status(args.root)


if __name__ == "__main__":
    main()
