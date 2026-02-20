# Custom weight loader that works for both HGQ and torch models.
import torch
import numpy as np


def load_weights(model, state_dict: dict, module_key: str, has_bn: bool = False):
    """Load weights from a Lightning .ckpt into either a pt nn module or HGQ.

    Load weights from a Lightning .ckpt into either:
        - a PyTorch nn.Module (normal state_dict load), or
        - a Keras HGQ MLP (manual weight transfer).

    :param model: The torch.nn.Module OR keras.Model to load weights into
    :param module_key: String of prefix in Lightning state_dict ("model" or "projector").
    :param has_bn: Whether the MLP uses BatchNorm.
    """

    sub_sd = {
        k[len(module_key) + 1 :]: v
        for k, v in state_dict.items()
        if k.startswith(module_key + ".")
    }
    if isinstance(model, torch.nn.Module):
        model.load_state_dict(sub_sd, strict=True)
        return model

    model = load_hgq_model(model, sub_sd, has_bn)

    return model


def load_hgq_model(model, sub_sd: dict, has_bn: bool):
    """Load weights from a Lightning .ckpt module into an HGQ model."""
    import keras

    if not isinstance(model, keras.Model):
        raise TypeError("model must be torch.nn.Module or keras.Model")

    # force variable creation
    in_dim = model.input_shape[-1]
    _ = model(np.zeros((1, in_dim), dtype=np.float32))
    nodes = infer_nodes_hgq(model)

    # extract Linear / BN params in order
    linear_keys = sorted(
        k
        for k in sub_sd
        if k.endswith("weight") and not any(x in k.lower() for x in ["bn", "batchnorm"])
    )
    bn_keys = sorted(
        k
        for k in sub_sd
        if k.endswith("weight") and any(x in k.lower() for x in ["bn", "batchnorm"])
    )
    # hidden layers
    for i in range(len(nodes)):
        W = sub_sd[linear_keys[i]].numpy()
        b = sub_sd[linear_keys[i].replace("weight", "bias")].numpy()
        assign_dense_like_weights(model.get_layer(f"qdense_{i}"), W, b)
        if has_bn:
            prefix = bn_keys[i].replace("weight", "")
            gamma = sub_sd[prefix + "weight"].numpy()
            beta = sub_sd[prefix + "bias"].numpy()
            mean = sub_sd[prefix + "running_mean"].numpy()
            var = sub_sd[prefix + "running_var"].numpy()
            model.get_layer(f"bn_{i}").set_weights([gamma, beta, mean, var])

    # output layer
    W = sub_sd[linear_keys[len(nodes)]].numpy()
    b = sub_sd[linear_keys[len(nodes)].replace("weight", "bias")].numpy()
    assign_dense_like_weights(model.get_layer("qdense_out"), W, b)

    return model


def infer_nodes_hgq(model):
    """Returns hidden-layer sizes (excluding output layer) of hgq MLP."""
    qdense_layers = sorted(
        [
            l
            for l in model.layers
            if l.name.startswith("qdense_") and l.name != "qdense_out"
        ],
        key=lambda l: int(l.name.split("_")[-1]),
    )
    if not qdense_layers:
        raise ValueError("No qdense_* layers found")
    return [l.units for l in qdense_layers]


def assign_dense_like_weights(layer, W_torch, b_torch=None):
    """Assign weights and biases to QKeras dense layer."""
    W = np.asarray(W_torch, dtype=np.float32).T
    b = None if b_torch is None else np.asarray(b_torch, dtype=np.float32)
    k = next(
        w for w in layer.weights if len(w.shape) == 2 and tuple(w.shape) == W.shape
    )
    k.assign(W)

    # bias: first 1D weight that matches b.shape (if provided)
    if b is not None:
        try:
            bb = next(
                w
                for w in layer.weights
                if len(w.shape) == 1 and tuple(w.shape) == b.shape
            )
            bb.assign(b)
        except StopIteration:
            pass  # bias may be disabled
