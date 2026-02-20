#!/usr/bin/env python3
"""
HGQ quantization inspector for Keras v3 (.keras) models using the torch backend.

What it prints per HGQ layer:
- Quantizer configurations for input/weights/bias/output (where present)
- Configured ap_fixed<W,I> (when quantizer type exposes i0/f0)
- Actual observed quantization stats from qkernel/qbias
- Effective ap_fixed<W,I> inferred from observed quantized levels (when feasible)

Usage:
  python inspect_hgq_quant.py /path/to/model.keras
"""
import os
import sys
import json
import math
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, "/data/deodagiu/adl1t")

import keras
import keras.ops as ops
import numpy as np
from pathvalidate import sanitize_filename

import hgq  # registers HGQ objects

# >>> ADDED: matplotlib for histogram plots
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.algorithms.components.encoder import Sampling


# -----------------------------
# Helpers: robust object -> dict
# -----------------------------
def _public_attrs_dict(obj: Any, max_items: int = 200) -> Dict[str, Any]:
    out = {}
    try:
        keys = [k for k in dir(obj) if not k.startswith("_")]
    except Exception:
        return out

    for k in keys[:max_items]:
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        # Keep it printable
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
    return out


def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Best-effort serialization of HGQ Quantizer / QuantizerConfig objects.
    HGQ's QuantizerConfig is often a custom class; we try common patterns.
    """
    if obj is None:
        return {}

    # Keras-style
    if hasattr(obj, "get_config") and callable(getattr(obj, "get_config")):
        try:
            cfg = obj.get_config()
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass

    # Common custom patterns
    for m in ("to_dict", "as_dict", "dict"):
        if hasattr(obj, m) and callable(getattr(obj, m)):
            try:
                cfg = getattr(obj, m)()
                if isinstance(cfg, dict):
                    return cfg
            except Exception:
                pass

    # Fallback: __dict__ if present (can include non-serializable objects)
    if hasattr(obj, "__dict__"):
        d = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, type(None))):
                d[k] = v
            elif isinstance(v, (dict, list, tuple)):
                d[k] = v
            else:
                d[k] = f"<{v.__class__.__module__}.{v.__class__.__name__}>"
        if d:
            return d

    # Last fallback: public attributes (simple scalars only)
    d = _public_attrs_dict(obj)
    if d:
        return d

    return {"__repr__": repr(obj)}


# ---------------------------------------
# QuantizerConfig -> ap_fixed<W,I> mapping
# ---------------------------------------
def _configured_ap_fixed_from_quant_cfg(qcfg: Dict[str, Any]) -> Optional[str]:
    """
    Attempts to map an HGQ quantizer config (dict) to ap_fixed<W,I>.
    Supports the common HGQ fixed-point schemes:
      - q_type == "kif" : integer bits i0, fractional bits f0 (signed)
    If fields are not present, returns None.
    """
    q_type = qcfg.get("q_type") or qcfg.get("qtype")
    if isinstance(q_type, dict):
        q_type = q_type.get("value") or q_type.get("name")

    if q_type == "kif":
        # i0, f0 are the usual names in your YAML
        i0 = qcfg.get("i0")
        f0 = qcfg.get("f0")
        if isinstance(i0, (int, float)) and isinstance(f0, (int, float)):
            I = int(i0)
            F = int(f0)
            W = I + F
            return f"ap_fixed<{W},{I}>  (F={F}, step=2^-{F}={2**(-F):.10g})"
    # If you use other q_type variants (e.g., kbi), you can extend mapping here.
    return None


# ---------------------------------------
# Observed quant values -> effective ap_fixed
# ---------------------------------------
def _is_close_power_of_two(x: float, tol: float = 1e-6) -> Optional[int]:
    """
    If x is close to 2^-F for integer F, return F else None.
    """
    if x <= 0:
        return None
    F = -math.log(x, 2)
    F_round = int(round(F))
    if abs(F - F_round) < tol:
        return F_round
    return None


def _effective_ap_fixed_from_levels(levels: np.ndarray) -> Optional[str]:
    """
    Given observed quantized levels (float), infer an effective ap_fixed<W,I>
    when the observed step looks like a power-of-two.

    This is *effective* in real units (post any HGQ scaling), not necessarily
    the configured container used internally.
    """
    levels = np.asarray(levels, dtype=np.float64)
    levels = np.unique(np.round(levels, 12))

    if levels.size < 2:
        return "ap_fixed<1,1> (degenerate: single level)"

    sorted_levels = np.sort(levels)
    diffs = np.diff(sorted_levels)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None

    step = float(np.min(diffs))
    F = _is_close_power_of_two(step, tol=1e-4)
    if F is None:
        return None

    max_abs = float(np.max(np.abs(levels)))
    # Need I such that max_abs <= 2^(I-1) - 2^-F approximately.
    # Use a conservative bound: 2^(I-1) > max_abs
    if max_abs == 0:
        I = 1
    else:
        I = max(1, int(math.ceil(math.log(max_abs + 1e-12, 2))) + 1)

    W = I + F
    return f"ap_fixed<{W},{I}>  (effective; F={F}, step={step:.10g}, max_abs={max_abs:.10g})"


# -----------------------------
# Actual quant stats
# -----------------------------
def _to_np(x: Any) -> np.ndarray:
    return ops.convert_to_numpy(x)


def _summarize_quant(wf: np.ndarray, wq: np.ndarray) -> Dict[str, Any]:
    wf = np.asarray(wf, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64)

    diff = wf - wq
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    wf_std = float(wf.std())
    rel_rmse = float(rmse / (wf_std + 1e-12))

    qmin = float(np.min(wq))
    qmax = float(np.max(wq))
    levels = np.unique(np.round(wq, 12))
    n_levels = int(levels.size)

    # observed min step between levels
    if levels.size >= 2:
        sorted_levels = np.sort(levels)
        diffs = np.diff(sorted_levels)
        diffs = diffs[diffs > 0]
        step = float(np.min(diffs)) if diffs.size else None
    else:
        step = None

    sat_any = float(np.mean((wq == qmin) | (wq == qmax))) if qmin != qmax else 1.0
    sat_min = float(np.mean(wq == qmin)) if qmin != qmax else 1.0
    sat_max = float(np.mean(wq == qmax)) if qmin != qmax else 1.0

    return {
        "float_shape": list(wf.shape),
        "float_mean": float(wf.mean()),
        "float_std": float(wf.std()),
        "float_min": float(wf.min()),
        "float_max": float(wf.max()),
        "quant_mean": float(wq.mean()),
        "quant_std": float(wq.std()),
        "quant_min": qmin,
        "quant_max": qmax,
        "levels": n_levels,
        "observed_step": step,
        "mae": mae,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "sat_any_frac": sat_any,
        "sat_min_frac": sat_min,
        "sat_max_frac": sat_max,
        "levels_preview": levels[:20].tolist(),
    }


def _configured_ap_fixed_for_place(layer_entry, place: str) -> Optional[str]:
    q = layer_entry.get("quantizers", {}).get(place, {}).get("detail", {})
    return q.get("configured_ap_fixed")


# -----------------------------
# Model traversal and quantizer extraction
# -----------------------------
QUANT_ATTRS = {
    # Common HGQ naming patterns on Q-layers and quant modules
    "input": ("iq", "ioq", "in_q", "input_q"),
    "weights": ("kq", "wq", "kernel_q", "weight_q"),
    "bias": ("bq", "bias_q"),
    "output": ("oq", "toq", "out_q", "output_q"),
    "table": ("tq",),
}

WEIGHT_TENSORS = {
    "weights": ("kernel", "weight"),
    "bias": ("bias",),
}

QUANT_TENSORS = {
    "weights": ("qkernel", "qweight"),
    "bias": ("qbias",),
}


def _find_first_attr(obj: Any, names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _get_quantizer_detail(q: Any) -> Dict[str, Any]:
    """
    Returns:
      - quantizer wrapper config (outer)
      - quantizer inner QuantizerConfig (if present under .config)
      - configured ap_fixed (if derivable)
    """
    outer = _obj_to_dict(q)
    inner = {}
    apfx = None

    # HGQ Quantizer often has .config pointing to QuantizerConfig
    if hasattr(q, "config"):
        try:
            qc = q.config
            inner = _obj_to_dict(qc)
            apfx = _configured_ap_fixed_from_quant_cfg(inner)
        except Exception:
            pass

    return {
        "outer": outer,
        "config": inner,
        "configured_ap_fixed": apfx,
    }


def _iter_layers(model):
    if hasattr(model, "_flatten_layers"):
        yield from model._flatten_layers(include_self=False, recursive=True)
    else:
        yield from model.layers


# -----------------------------
# Build model once (materialize qkernel/qbias)
# -----------------------------
def _build_once(model):
    def make_dummy_like(inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs_list = [inputs]
        else:
            inputs_list = list(inputs)

        dummy = []
        for x in inputs_list:
            shape = [d if d is not None else 1 for d in list(x.shape)]
            dummy.append(ops.zeros(shape, dtype=x.dtype))
        return dummy if len(dummy) > 1 else dummy[0]

    dummy_inp = make_dummy_like(model.inputs)
    _ = model(dummy_inp, training=False)


def _save_histogram_pair(
    wf: np.ndarray,
    wq: np.ndarray,
    out_path: Path,
    title: str,
    n_bins: int = 120,
) -> None:
    """
    Save an overlay histogram of float vs quantized values.
    Designed to remain readable for very low-level quantization (few discrete values).
    """
    wf = np.asarray(wf, dtype=np.float64).ravel()
    wq = np.asarray(wq, dtype=np.float64).ravel()

    # If everything is identical (e.g., all zeros), still save a plot.
    wq_levels = np.unique(np.round(wq, 12))
    # If quantized has few discrete levels, reduce bins to avoid empty spam
    bins = min(n_bins, max(10, int(np.sqrt(wf.size))))
    if wq_levels.size <= 20:
        bins = max(10, min(bins, 40))

    # Align histogram range across both distributions
    lo = float(min(wf.min(initial=0.0), wq.min(initial=0.0)))
    hi = float(max(wf.max(initial=0.0), wq.max(initial=0.0)))
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    plt.figure()
    plt.hist(wf, bins=bins, range=(lo, hi), alpha=0.6, label="float")
    plt.hist(wq, bins=bins, range=(lo, hi), alpha=0.6, label="quantized")

    # Mark quantization levels when they are few (helpful for debugging)
    if wq_levels.size <= 25:
        for lv in wq_levels:
            plt.axvline(float(lv), linewidth=1)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


# -----------------------------
# Main reporting
# -----------------------------
def inspect_model(model, plots_dir: Optional[Path] = None) -> Dict[str, Any]:
    _build_once(model)

    report = {
        "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "layers": [],
    }

    for layer in _iter_layers(model):
        layer_entry: Dict[str, Any] = {
            "name": layer.name,
            "class": f"{layer.__class__.__module__}.{layer.__class__.__name__}",
            "is_hgq_layer": layer.__class__.__module__.startswith("hgq."),
            "quantizers": {},
            "actual_quant": {},
        }

        # Quantizer configs: input/weights/bias/output/table
        for place, attr_candidates in QUANT_ATTRS.items():
            attr = _find_first_attr(layer, attr_candidates)
            if attr is None:
                continue
            q = getattr(layer, attr, None)
            if q is None:
                continue
            layer_entry["quantizers"][place] = {
                "attr": attr,
                "detail": _get_quantizer_detail(q),
            }

        # Actual quantization for weights/bias (if qkernel/qbias exist)
        for wt_kind, float_names in WEIGHT_TENSORS.items():
            # find float tensor name
            float_attr = _find_first_attr(layer, float_names)
            if float_attr is None:
                continue
            wf = getattr(layer, float_attr, None)
            if wf is None:
                continue

            # find quant tensor name
            qnames = QUANT_TENSORS.get(wt_kind, ())
            q_attr = _find_first_attr(layer, qnames)
            if q_attr is None:
                # fallback: try applying quantizer callable if present
                q_attr = None

            wq = None
            if q_attr is not None:
                wq = getattr(layer, q_attr, None)

            # fallback: apply layer.kq / layer.bq if callable
            if wq is None:
                if (
                    wt_kind == "weights"
                    and hasattr(layer, "kq")
                    and callable(getattr(layer, "kq"))
                ):
                    try:
                        wq = layer.kq(wf)
                        q_attr = "kq(w)"
                    except Exception:
                        wq = None
                if (
                    wt_kind == "bias"
                    and hasattr(layer, "bq")
                    and callable(getattr(layer, "bq"))
                ):
                    try:
                        wq = layer.bq(wf)
                        q_attr = "bq(b)"
                    except Exception:
                        wq = None

            if wq is None:
                continue

            wf_np = _to_np(wf)
            wq_np = _to_np(wq)

            summary = _summarize_quant(wf_np, wq_np)
            layer_entry["actual_quant"][wt_kind] = {
                "float_attr": float_attr,
                "quant_attr": q_attr or "(unknown)",
                "summary": summary,
            }
            if plots_dir is not None:
                safe_layer = sanitize_filename(layer.name)
                fname = f"{safe_layer}__{wt_kind}__hist.png"
                out_path = plots_dir / fname
                title = (
                    f"{layer.name} / {wt_kind}\n"
                    f"levels={summary['levels']}  step={summary['observed_step']}  "
                    f"rmse={summary['rmse']:.4g}  rel_rmse={summary['rel_rmse']:.3f}\n"
                )
                _save_histogram_pair(wf_np, wq_np, out_path, title=title)

        if (
            layer_entry["is_hgq_layer"]
            or layer_entry["quantizers"]
            or layer_entry["actual_quant"]
        ):
            report["layers"].append(layer_entry)

    return report


def pretty_print(report: Dict[str, Any], max_json_chars: int = 8000) -> None:
    print("Model:", report["model_type"])
    print("Quantized/HGQ-relevant layers found:", len(report["layers"]))

    for layer in report["layers"]:
        print("\n" + "=" * 80)
        print(f"{layer['name']}  [{layer['class']}]")
        if layer["is_hgq_layer"]:
            print("HGQ layer: yes")

        # Quantizer configs
        if layer["quantizers"]:
            print("\nQuantizer configuration (by place):")
            for place, qinfo in layer["quantizers"].items():
                attr = qinfo["attr"]
                detail = qinfo["detail"]
                apfx = detail.get("configured_ap_fixed")

                print(f"  - {place}: attr={attr}")
                if apfx:
                    print(f"      configured ap_fixed: {apfx}")

                # Print a compact view: q_type/i0/f0/overflow etc. live in detail["config"]
                cfg = detail.get("config") or {}
                if cfg:
                    # Try to show only the most relevant fields first
                    key_order = [
                        "q_type",
                        "i0",
                        "f0",
                        "overflow_mode",
                        "rounding_mode",
                        "trainable",
                    ]
                    compact = {k: cfg.get(k) for k in key_order if k in cfg}
                    # plus anything else
                    for k, v in cfg.items():
                        if k not in compact:
                            compact[k] = v
                    txt = json.dumps(compact, indent=2, default=str)
                    print("      config:")
                    print(
                        "\n".join("        " + line for line in txt.splitlines())[
                            :max_json_chars
                        ]
                    )
                else:
                    # Fallback to outer wrapper dict
                    outer = detail.get("outer") or {}
                    txt = json.dumps(outer, indent=2, default=str)
                    print("      outer:")
                    print(
                        "\n".join("        " + line for line in txt.splitlines())[
                            :max_json_chars
                        ]
                    )

        # Actual observed quant
        if layer["actual_quant"]:
            print("\nActual observed quantization (weights/bias):")
            for kind, info in layer["actual_quant"].items():
                s = info["summary"]
                print(
                    f"  - {kind}: float={info['float_attr']}  quant={info['quant_attr']}"
                )
                print(
                    f"      levels={s['levels']}  "
                    f"qrange=[{s['quant_min']}, {s['quant_max']}]  "
                    f"step={s['observed_step']}  "
                    f"rmse={s['rmse']:.6f}  rel_rmse={s['rel_rmse']:.3f}  "
                    f"sat_any={100*s['sat_any_frac']:.1f}%"
                )
                cfg_apfx = None
                if kind == "weights":
                    cfg_apfx = _configured_ap_fixed_for_place(layer, "weights")
                elif kind == "bias":
                    cfg_apfx = _configured_ap_fixed_for_place(layer, "bias")

                if cfg_apfx:
                    # For KIF (and anything with i0/f0), the configured type is what matters.
                    print(
                        f"      effective ap_fixed: {cfg_apfx}  (configured; observed max_abs={max(abs(s['quant_min']), abs(s['quant_max']))})"
                    )
                else:
                    # Only infer an effective type when no configured ap_fixed is available (e.g., KBI).
                    eff = _effective_ap_fixed_from_levels(
                        np.array(
                            s["levels_preview"] + [s["quant_min"], s["quant_max"]],
                            dtype=np.float64,
                        )
                    )
                    if eff:
                        print(f"      effective ap_fixed: {eff}")
                print(f"      levels_preview: {s['levels_preview']}")

    print("\n" + "=" * 80)
    print("Done.")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python inspect_hgq_quant.py /path/to/model.keras", file=sys.stderr
        )
        sys.exit(2)

    model_path = Path(sys.argv[1])
    model = keras.saving.load_model(str(model_path), compile=False, safe_mode=True)

    # Output directory: qmodel_analysis/<model_stem>/
    out_dir = Path("qmodel_analysis") / sanitize_filename(model_path.stem)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = inspect_model(model, plots_dir=plots_dir)
    pretty_print(report)

    out_json = out_dir / "hgq_quant_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nFull JSON written to: {out_json}")
    print(f"Histogram plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
