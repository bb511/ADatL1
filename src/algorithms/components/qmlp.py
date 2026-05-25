# Make a quantised MLP using HGQ2.
import keras
from keras import layers

from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import Constant


def hgq_mlp(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    input_layer_config: dict = None,
    output_layer_config: dict = None,
    ebops: bool = False,
    final_activation: bool = False,
    name: str = "hgq_mlp",
):
    """Multi-layer perceptron in HGQv2.

    :param in_dim: Int for initial dimension.
    :param nodes: List of number of nodes composing each of the layers.
    :param out_dim: Int for output dimension.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    """
    inputs = keras.Input(shape=(in_dim,), name="input")
    x = inputs
    with LayerConfigScope(enable_ebops=ebops):
        for i, hidden_dim in enumerate(nodes):
            if i == 0 and input_layer_config:
                with QuantizerConfigScope(**input_layer_config, heterogeneous_axis=()):
                    x = QDense(hidden_dim, name="qdense_0", activation="relu")(x)
            else:
                x = QDense(hidden_dim, name=f"qdense_{i}", activation="relu")(x)

    if output_layer_config:
        with QuantizerConfigScope(**output_layer_config, heterogeneous_axis=()):
            if final_activation:
                outputs = QDense(out_dim, name="qdense_out", activation="relu")(x)
            else:
                outputs = QDense(out_dim, name="qdense_out")(x)
    else:
        if final_activation:
            outputs = QDense(out_dim, name="qdense_out", activation="relu")(x)
        else:
            outputs = QDense(out_dim, name="qdense_out")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    print(model.summary())

    return model
