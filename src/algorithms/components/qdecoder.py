"""HGQ/Keras decoder components for quantized VAE-style models."""

try:
    import keras
    from hgq.layers import QDense
    from hgq.config import QuantizerConfigScope, LayerConfigScope
except ImportError:  # pragma: no cover - exercised when quant extra is absent
    class _MissingKerasModel:
        pass

    class _MissingKeras:
        Model = _MissingKerasModel
        Sequential = None

    keras = _MissingKeras()
    QDense = None
    QuantizerConfigScope = None
    LayerConfigScope = None

from src.algorithms.components.qmlp import HGQMLP, require_hgq


class HGQDecoder(keras.Model):
    """Decoder implemented with HGQv2 ``QDense`` layers."""

    def __init__(
        self,
        nodes: list[int],
        input_layer_config: dict | None = None,
        output_layer_config: dict | None = None,
        ebops: bool = False,
        **kwargs,
    ):
        require_hgq()
        super().__init__(**kwargs)
        self.nodes = nodes
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config or {}
        self.ebops = ebops

        mlp = HGQMLP(
            nodes=nodes[:-1],
            input_layer_config=self.input_layer_config,
            final_activation=True,
            ebops=self.ebops,
            name="dec_mlp",
        )

        with LayerConfigScope(enable_ebops=self.ebops):
            with QuantizerConfigScope(**self.output_layer_config):
                output_layer = QDense(nodes[-1], name="qdense_out")

        self.net = keras.Sequential([mlp, output_layer], name="net")

    def call(self, z):
        return self.net(z)
