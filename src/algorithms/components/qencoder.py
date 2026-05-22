"""HGQ/Keras encoder components for quantized VAE-style models."""

try:
    import keras
    from keras import ops
    from hgq.layers import QDense
    from hgq.config import QuantizerConfigScope, LayerConfigScope
except ImportError:  # pragma: no cover - exercised when quant extra is absent
    class _MissingLayer:
        pass

    class _MissingSaving:
        @staticmethod
        def register_keras_serializable(*args, **kwargs):
            def decorator(cls):
                return cls

            return decorator

    class _MissingKerasModel:
        pass

    class _MissingLayers:
        Layer = _MissingLayer

    class _MissingKeras:
        Model = _MissingKerasModel
        layers = _MissingLayers()
        saving = _MissingSaving()

    keras = _MissingKeras()
    ops = None
    QDense = None
    QuantizerConfigScope = None
    LayerConfigScope = None

from src.algorithms.components.qmlp import HGQMLP, require_hgq


@keras.saving.register_keras_serializable(package="adl1t")
class Sampling(keras.layers.Layer):
    """Reparameterization trick layer for Keras torch-backend VAEs."""

    def __init__(self, clip_min: float = -20.0, clip_max: float = 10.0, **kwargs):
        require_hgq()
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        z_log_var = ops.clip(z_log_var, self.clip_min, self.clip_max)
        std = ops.exp(0.5 * z_log_var)
        eps = keras.random.normal(shape=ops.shape(std), dtype=std.dtype)
        z = z_mean + std * eps
        return z_mean, z_log_var, z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"clip_min": self.clip_min, "clip_max": self.clip_max})
        return cfg


class HGQVariationalEncoder(keras.Model):
    """Variational encoder implemented with HGQv2 ``QDense`` layers."""

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
        self.latent_dim = nodes[-1]
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config or {}
        self.ebops = ebops

        self.net = HGQMLP(
            nodes=nodes[:-1],
            input_layer_config=self.input_layer_config,
            ebops=self.ebops,
            final_activation=True,
            name="enc_mlp",
        )

        with LayerConfigScope(enable_ebops=self.ebops):
            with QuantizerConfigScope(**self.output_layer_config):
                self.z_mean = QDense(self.latent_dim, name="z_mean")
                self.z_log_var = QDense(self.latent_dim, name="z_log_var")

        self.sampling = Sampling(name="sampling")

    def call(self, x):
        h = self.net(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z_mean, z_log_var, z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
