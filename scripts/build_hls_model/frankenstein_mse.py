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
encoder_path = "/data/deodagiu/adl1t/checkpoints/vicreg_qvae_best_pure/mse_best_model_3/single/loss_reco_full_rate0.25kHz/max/ds=GluGluHto2G_Par-MH-125__metric=loss_reco_full_rate0.25kHz__value=5944.918457__epoch=12.ckpt__component=encoder.keras"
decoder_path = "/data/deodagiu/adl1t/checkpoints/vicreg_qvae_best_pure/mse_best_model_3/single/loss_reco_full_rate0.25kHz/max/ds=GluGluHto2G_Par-MH-125__metric=loss_reco_full_rate0.25kHz__value=5944.918457__epoch=12.ckpt__component=decoder.keras"

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


# ----------------------------
# Build the flat inference graph
# ----------------------------
used_names = {}

inp = keras.Input(shape=(57,), name="input")
used_names["input"] = 1  # reserve

# 1) Inline hgq_mlp entirely (qdense_0 -> qdense_out)
x = inline_model_layers(qvicreg, inp, prefix="vicreg__", used_names=used_names)

# 2) Encoder: inline ONLY enc_mlp, then apply z_mean
enc_mlp = encoder.get_layer("enc_mlp")  # Functional
z_mean_layer = encoder.get_layer("z_mean")  # QDense

x = inline_model_layers(enc_mlp, x, prefix="enc__", used_names=used_names)

z, _ = clone_and_apply_layer(z_mean_layer, x, name="enc__z_mean", used_names=used_names)

# 3) Decoder: inline ONLY dec_mlp, then apply dec_qdense_out
dec_mlp = decoder.get_layer("dec_mlp")  # Functional
dec_out_layer = decoder.get_layer("dec_qdense_out")  # QDense

x = inline_model_layers(dec_mlp, z, prefix="dec__", used_names=used_names)

out, _ = clone_and_apply_layer(
    dec_out_layer, x, name="dec_output", used_names=used_names
)

flat_inference = Model(inp, out, name="flat_inference")

# ----------------------------
# Sanity checks
# ----------------------------
flat_inference.summary()

x_test = np.random.randn(8, 57).astype("float32")

# Reference computation from original components:
# qvicreg -> encoder (take z_mean) -> decoder
q_out = qvicreg(x_test)

# Get z_mean directly from the loaded encoder model
encoder_z_mean_ref = keras.Model(
    inputs=encoder.inputs,
    outputs=encoder.get_layer("z_mean").output,
    name="encoder_z_mean_ref",
)
z_ref = encoder_z_mean_ref(q_out)

y_ref = decoder(z_ref)

# Flat model output
y_flat = flat_inference(x_test)


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


y_ref_np = to_numpy(y_ref)
y_flat_np = to_numpy(y_flat)

diff = np.max(np.abs(y_ref_np - y_flat_np))
print("max abs diff:", diff)

inp = flat_inference.inputs[0]

vicreg_out = flat_inference.get_layer("vicreg__qdense_out").output  # (None, 10)
decoder_out = flat_inference.output  # (None, 10)
decoder_out = flat_inference.output  # (None, 10)

# Difference
diff = layers.Subtract(name="vicreg_minus_vae")([decoder_out, vicreg_out])  # (None, 10)

# Dot product with itself => sum of squares
score = layers.Dot(axes=1, name="out")([diff, diff])  # (None, 1)

# Final model
dot_model = Model(inputs=inp, outputs=score, name="flat_inference_dot")
dot_model.summary()

out_dir = "/data/deodagiu/adl1t/trigger_models/hls4ml_trigger_model_mse_v1"
os.makedirs(out_dir, exist_ok=True)

out_path = out_dir + "/keras_model.keras"
keras.saving.save_model(dot_model, out_path)
print("Saved keras model to:", out_path)

# out_dir_hls = out_dir + "/hls_project"
# hls_config = hls4ml.utils.config_from_keras_model(dot_model, granularity="name")
# hls_model = hls4ml.converters.convert_from_keras_model(
#     dot_model,
#     hls_config=hls_config,
#     output_dir=out_dir_hls,
#     part="xcku115-flvb2104-2-i",
#     io_type="io_parallel",
#     backend="Vitis",
# )

# # Force project files to be written
# hls_model.write()

# print("hls4ml wrote to:", hls_model.config.get_output_dir())
