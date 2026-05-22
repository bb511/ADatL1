"""HGQ/Keras MLP components used by quantized L1AD models."""

try:
    import keras
    from keras import layers as klayers
    from hgq.layers import QDense
    from hgq.config import LayerConfigScope, QuantizerConfigScope
except ImportError:  # pragma: no cover - exercised when quant extra is absent
    class _MissingKerasModel:
        pass

    class _MissingKeras:
        Model = _MissingKerasModel

    keras = _MissingKeras()
    klayers = None
    QDense = None
    LayerConfigScope = None
    QuantizerConfigScope = None


def require_hgq() -> None:
    if QDense is None:
        raise ImportError(
            "HGQ models require the quant extra. Install with "
            "`uv sync --extra quant --group dev`."
        )


class HGQMLP(keras.Model):
    """Multi-layer perceptron implemented with HGQv2 ``QDense`` layers.

    ``nodes`` includes input and output dimensions, matching the quantized configs.
    """

    def __init__(
        self,
        nodes: list[int],
        batchnorm: bool = False,
        affine: bool = True,
        final_activation: bool = False,
        input_layer_config: dict | None = None,
        output_layer_config: dict | None = None,
        ebops: bool = False,
        **kwargs,
    ):
        require_hgq()
        super().__init__(**kwargs)
        self.nodes = nodes
        self.batchnorm = batchnorm
        self.affine = affine
        self.final_activation = final_activation
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config
        self.ebops = ebops

        self.net = self._construct_net()

    @staticmethod
    def make_qdense(out_dim: int, name: str, activation: str | None, config=None):
        if config:
            with QuantizerConfigScope(**config, heterogeneous_axis=()):
                return QDense(out_dim, name=name, activation=activation)
        return QDense(out_dim, name=name, activation=activation)

    def _construct_net(self):
        layers = []
        num_layers = len(self.nodes) - 1
        with LayerConfigScope(enable_ebops=self.ebops):
            with QuantizerConfigScope(place="all"):
                for i, out_dim in enumerate(self.nodes[1:]):
                    is_last = i == num_layers - 1
                    if is_last:
                        layers.append(
                            self.make_qdense(
                                out_dim=out_dim,
                                name="qdense_out",
                                activation="relu" if self.final_activation else None,
                                config=self.output_layer_config,
                            )
                        )
                        continue

                    layers.append(
                        self.make_qdense(
                            out_dim=out_dim,
                            name=f"qdense_{i}",
                            activation="relu",
                            config=self.input_layer_config if i == 0 else None,
                        )
                    )
                    if self.batchnorm:
                        layers.append(
                            klayers.BatchNormalization(
                                scale=self.affine,
                                center=self.affine,
                                name=f"bn_{i}",
                            )
                        )

        return layers

    def call(self, x):
        for layer in self.net:
            x = layer(x)
        return x
