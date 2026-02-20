# Frankenstein a model for the trigger.

import os
import sys

# IMPORTANT: set backend before importing keras
sys.path.insert(0, "/data/deodagiu/adl1t")
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras import Model
import hls4ml
import numpy as np
import torch

import hgq
from hgq.layers.core.dense import QDense  # ensures class is imported

# If you use other HGQ layers/constraints/regularizers in the model, import them too:
from hgq.constraints import Constant, MinMax
from hgq.quantizer.config import QuantizerConfig
from hgq.regularizers import MonoL1
from src.algorithms.components.encoder import Sampling


qvicreg_path = "/data/deodagiu/adl1t/checkpoints/qvicreg_best_models/best_model_5/single/knn_auprc/max/ds=haa-4b-ma15__metric=knn_auprc__value=0.640045__epoch=08.ckpt__component=model.keras"
encoder_path = "/data/deodagiu/adl1t/checkpoints/vicreg_qvae_best_pure/kl_best_model_5/single/loss_kl_raw_full_rate0.25kHz/max/ds=WtoTauto3Mu__metric=loss_kl_raw_full_rate0.25kHz__value=1.907635__epoch=12.ckpt__component=encoder.keras"
decoder_path = "/data/deodagiu/adl1t/checkpoints/vicreg_qvae_best_pure/kl_best_model_5/single/loss_kl_raw_full_rate0.25kHz/max/ds=WtoTauto3Mu__metric=loss_kl_raw_full_rate0.25kHz__value=1.907635__epoch=12.ckpt__component=decoder.keras"

# 1) Load models
qvicreg = keras.saving.load_model(qvicreg_path, compile=False)
encoder = keras.saving.load_model(encoder_path, compile=False)
decoder = keras.saving.load_model(decoder_path, compile=False)


# ----------------------------
# Helpers
# ----------------------------
def _unique_name(base: str, used: dict) -> str:
    """Return a globally-unique op name within the new graph."""
    if base not in used:
        used[base] = 1
        return base
    used[base] += 1
    return f"{base}__{used[base]}"


def clone_and_apply_layer(layer, x, name, used_names):
    """Clone a single layer (new instance), ensure unique name, apply, and copy weights."""
    cfg = layer.get_config()
    cfg["name"] = _unique_name(name, used_names)
    new_layer = layer.__class__.from_config(cfg)
    y = new_layer(x)

    w = layer.get_weights()
    if w:
        new_layer.set_weights(w)
    return y, new_layer


def inline_model_layers(
    m: keras.Model, x, prefix: str, used_names: dict, skip_layer_names=None
):
    """
    Inline a model by cloning and applying each internal layer.
    - Recurses into nested Models.
    - Skips InputLayers.
    - Optionally skips layers by *original* layer.name (not prefixed).
    This assumes the model is feed-forward in layer order (MLP-style).
    """
    if skip_layer_names is None:
        skip_layer_names = set()

    for layer in m.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if layer.name in skip_layer_names:
            continue

        if isinstance(layer, keras.Model):
            x = inline_model_layers(
                layer, x, prefix + layer.name + "__", used_names, skip_layer_names
            )
        else:
            x, _ = clone_and_apply_layer(layer, x, prefix + layer.name, used_names)
    return x


used_names = {}

inp = keras.Input(shape=(57,), name="x")
used_names["x"] = 1  # reserve

# 1) Inline VICReg (hgq_mlp) entirely: (57) -> (10)
v = inline_model_layers(qvicreg, inp, prefix="mlp__", used_names=used_names)

# 2) Inline encoder MLP trunk: (10) -> (6)
enc_mlp = encoder.get_layer("enc_mlp")  # Functional
h = inline_model_layers(enc_mlp, v, prefix="enc__", used_names=used_names)

# 3) Apply z_mean and z_log_var heads: (6) -> (4) each
z_mean_layer = encoder.get_layer("z_mean")  # QDense(4)
z_log_var_layer = encoder.get_layer("z_log_var")  # QDense(4)

mu, _ = clone_and_apply_layer(
    z_mean_layer, h, name="enc__z_mean", used_names=used_names
)
logvar, _ = clone_and_apply_layer(
    z_log_var_layer, h, name="enc__z_log_var", used_names=used_names
)

mu2 = layers.Multiply(name="mu_sq")([mu, mu])
exp_logvar = layers.Activation("exponential", name="exp_logvar")(logvar)

core = layers.Add(name="core_exp_plus_mu2")([exp_logvar, mu2])
core = layers.Subtract(name="core_minus_logvar")([core, logvar])

# KL = 0.5 * (sum(core) - d), with d=4
#    = sum(0.5*core) - 2
out_layer = layers.Dense(1, use_bias=True, trainable=False, name="out")
out = out_layer(core)
out_layer.set_weights(
    [
        0.5 * np.ones((4, 1), dtype="float32"),  # kernel
        np.array([-2.0], dtype="float32"),  # bias = -0.5*d = -2
    ]
)

flat_kl = Model(inp, out, name="flat_vicreg_encoder_kl")
flat_kl.summary()

out_dir = "/data/deodagiu/adl1t/trigger_models/hls4ml_trigger_model_kl_v1"
os.makedirs(out_dir, exist_ok=True)

out_path = out_dir + "/keras_model.keras"
keras.saving.save_model(flat_kl, out_path)
print("Saved keras model to:", out_path)

out_dir_hls = out_dir + "/hls_project"
hls_config = hls4ml.utils.config_from_keras_model(flat_kl, granularity="name")
hls_model = hls4ml.converters.convert_from_keras_model(
    flat_kl,
    hls_config=hls_config,
    output_dir=out_dir_hls,
    part="xcku115-flvb2104-2-i",
    io_type="io_parallel",
    backend="Vitis",
)

# Force project files to be written
hls_model.write()

print("hls4ml wrote to:", hls_model.config.get_output_dir())
