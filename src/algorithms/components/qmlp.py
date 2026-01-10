# Quantised MLP using HGQ2.
import keras
from keras import layers

from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope

def build_hgq_mlp(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    batchnorm: bool = False,
    affine: bool = True,
):
    inputs = keras.Input(shape=(in_dim,), name="x")
    x = inputs

    with (
        QuantizerConfigScope(place="all"),        # default quantizers everywhere
        LayerConfigScope(enable_ebops=True),      # optional: resource estimation during training
    ):
        for i, hidden_dim in enumerate(nodes):
            x = QDense(hidden_dim, name=f"qdense_{i}")(x)

            if batchnorm:
                # affine=True corresponds to scale+center in Keras BN
                x = layers.BatchNormalization(scale=affine, center=affine, name=f"bn_{i}")(x)

            x = layers.ReLU(name=f"relu_{i}")(x)

        outputs = QDense(out_dim, name="qdense_out")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="hgq_mlp")

