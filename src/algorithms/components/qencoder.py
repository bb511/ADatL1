# Quantised encoder architectures using HGQv2.
import keras
from keras import layers
from keras import ops
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.algorithms.components.mlp import hgq_mlp


def hgq_variational_encoder(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    input_layer_config: dict = None,
    output_layer_config: dict = None,
    ebops: bool = False,
    name: str = "hgq_var_encoder",
):
    """Variational Encoder in HGQv2

    :param in_dim: Int specifying input dimension.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param out_dim: Int specifying output dimension.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    x_in = keras.Input(shape=(in_dim,), name="input")
    qmlp_model = hgq_mlp(
        in_dim=in_dim,
        nodes=nodes[:-1],
        out_dim=nodes[-1],
        input_layer_config=input_layer_config,
        ebops=ebops,
        final_activation=True,
        name="enc_mlp",
    )
    h = qmlp_model(x_in)

    with LayerConfigScope(enable_ebops=ebops):
        with QuantizerConfigScope(**output_layer_config):
            z_mean = QDense(out_dim, name="z_mean")(h)
            z_log_var = QDense(out_dim, name="z_log_var")(h)

    z_mean, z_log_var, z = Sampling(name="sampling")([z_mean, z_log_var])

    return keras.Model(inputs=x_in, outputs=[z_mean, z_log_var, z], name=name)


@keras.saving.register_keras_serializable(package="adl1t")
class Sampling(keras.layers.Layer):
    """Reparameterization trick layer."""

    def __init__(self, clip_min: float = -20.0, clip_max: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs

        # clamp / clip log-variance
        z_log_var = ops.clip(z_log_var, self.clip_min, self.clip_max)
        std = ops.exp(0.5 * z_log_var)

        # backend-agnostic random normal
        eps = keras.random.normal(shape=ops.shape(std), dtype=std.dtype)

        z = z_mean + std * eps
        return z_mean, z_log_var, z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"clip_min": self.clip_min, "clip_max": self.clip_max})
        return cfg
