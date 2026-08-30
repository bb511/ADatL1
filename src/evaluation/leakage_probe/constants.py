"""Frozen constants for the leakage-probe protocol."""

from types import MappingProxyType

LEAKAGE_PROBE_PROTOCOL_VERSION = "fet-et-four-probe-v3"
PROBE_INNER_SPLIT_SEED = 12345
PROBE_TARGET_SHUFFLE_SEED = 12345
SHUFFLED_TARGET_R2_CLIPPED_MAX = 0.02
PROBE_INNER_VALIDATION_FRACTION = 0.2
PROBE_INITIALIZATION_SEEDS = (10, 123, 500)
MLP_PROBE_CONFIG = MappingProxyType(
    {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate": "constant",
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        "shuffle": True,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-4,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
    }
)
PROBE_REPRESENTATION_METRIC_NAMES = MappingProxyType(
    {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
        "latent_sample": "z_sample",
    }
)

PRIMARY_PROBE_REPRESENTATIONS = (
    "latent_logits",
    "reconstructed_data",
)
