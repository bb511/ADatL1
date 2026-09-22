"""Frozen constants for the leakage-probe protocol."""

from types import MappingProxyType

LEAKAGE_PROBE_PROTOCOL_VERSION = "fet-et-four-probe-v10"
LEAKAGE_PROBE_EVALUATION_MODES = (
    "validation",
    "final_test",
)
LEAKAGE_PROBE_INVALID_RUN_POLICY = "reject_configuration"
PROBE_EVENT_SAMPLE_SEED = 12345
PROBE_TARGET_SHUFFLE_SEED = 12345
SHUFFLED_TARGET_R2_CLIPPED_MAX = 0.02
# A finite but numerically broken linear solution must not be converted into
# zero leakage merely because its held-out R2 is negative.  This threshold is
# intentionally very loose: ordinary distribution shift remains measurable,
# while the v9 failure (held-out MSE roughly 1e12 times its development and
# held-out-target variance scales) is rejected as a failed probe.
LINEAR_PROBE_MAX_MSE_INFLATION = 1_000_000.0
# Row block for every streamed pass over a development or held-out pool.
# 65536 rows of 116 float64 features is ~61 MB, small enough that no probe
# ever materialises a scaled copy of a 12.5M-event matrix (5.8 GB at float32).
PROBE_STREAM_CHUNK_ROWS = 65536
# One frozen MLP initialization. There is no seed search and no inner
# partition: each MLP probe is fitted once on the development pool and scored
# once on held-out data, exactly like the linear probes.
PROBE_INITIALIZATION_SEED = 123
MLP_PROBE_CONFIG = MappingProxyType(
    {
        "hidden_layer_sizes": (64, 32),
        # Since v9. sklearn defaults batch_size to min(200, n_samples); on a 12.5M
        # event pool that is 62,661 optimizer steps per epoch over 200x116
        # matmuls, far too small for threaded BLAS -- Condor recorded 1.00 of
        # 7 cores busy for 108 minutes (cluster 338410). 16384 matches
        # data.batch_size in configs/experiment/physics/ae.yaml, so the probe
        # sees the same batch geometry as the autoencoder it measures.
        "batch_size": 16384,
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
    }
)

PRIMARY_PROBE_REPRESENTATIONS = (
    "latent_logits",
    "reconstructed_data",
)
