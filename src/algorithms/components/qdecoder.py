# Quantised decoder models implemented with HGQv2.
import keras
from keras import layers
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.algorithms.components.mlp import hgq_mlp


def hgq_decoder(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    input_layer_config: dict = None,
    output_layer_config: dict = None,
    ebops: bool = False,
    name: str = "hgq_decoder",
):
    """Simple dncoder in HGQv2.

    :param in_dim: Int specifying input dimension.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param out_dim: Int specifying output dimension.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """
    z_in = keras.Input(shape=(in_dim,), name="z")

    mlp_model = hgq_mlp(
        in_dim=in_dim,
        nodes=nodes[:-1],
        out_dim=nodes[-1],
        input_layer_config=input_layer_config,
        final_activation=True,
        name="dec_mlp",
    )
    h = mlp_model(z_in)

    with LayerConfigScope(enable_ebops=False):
        with QuantizerConfigScope(**output_layer_config):
            x = QDense(out_dim, name="dec_qdense_out")(h)

    return keras.Model(inputs=z_in, outputs=x, name=name)
