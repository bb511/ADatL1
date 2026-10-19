import hashlib

from omegaconf import OmegaConf


def register_resolvers():
    """Register the resolvers the .yaml configs call, e.g. ${short_hash:...}."""
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
    OmegaConf.register_new_resolver("reverse", lambda xs: list(reversed(xs)))
    OmegaConf.register_new_resolver("smaller_decoder", smaller_decoder)
    OmegaConf.register_new_resolver("smaller_image_decoder", smaller_image_decoder)


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def smaller_decoder(nodes):
    """Smaller VAE decoder method.

    Rationale: for the VAE, you'd want to avoid an overly expressive decoder since you
        want the latent space to encode useful information.
    """
    nodes = list(nodes)

    # Encoder [h, z] -> Decoder [z, max(z, h//2)]
    if len(nodes) == 2:
        h, z = nodes
        return [z, max(z, h // 2)]

    # Encoder [h1, h2, z] -> Decoder [z, h2]
    if len(nodes) == 3:
        h1, h2, z = nodes
        return [z, h2]

    raise ValueError(f"smaller_decoder expects 2 or 3 encoder nodes, got {nodes}")


def smaller_image_decoder(nodes):
    """Smaller image VAE decoder compatible with decoder stride contract.

    Encoder contract:
        nodes = [conv..., hidden_dim, latent_dim]

    Decoder contract:
        nodes = [latent_dim] + reversed(conv_nodes)

    This excludes the encoder hidden projection width while preserving one
    decoder stage per encoder stride.
    """
    nodes = list(nodes)
    if len(nodes) < 3:
        raise ValueError(
            "Expected image VAE encoder nodes of form [conv..., hidden_dim, latent_dim]."
        )

    conv_nodes = nodes[:-2]
    latent_dim = nodes[-1]

    if len(conv_nodes) == 0:
        raise ValueError("Need at least one convolutional stage.")

    return [latent_dim] + list(reversed(conv_nodes))
