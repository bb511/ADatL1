import pickle  # nosec B403 -- writes controlled synthetic test fixtures only
from pathlib import Path

import awkward as ak
import numpy as np
import torch

from src.utils.pairing.jetclr_run_tables import ProcessedRunReader, load_run_schema
from src.utils.pairing.matching import deterministic_iterative_pairing


def _write_processed_run(root: Path) -> tuple[Path, Path]:
    run = root / "ZB_run_test"
    cache = root / "training_cache"
    run.mkdir()
    cache.mkdir()
    feature_map = {}
    offset = 0
    for name in ("FET", "egammas", "jets", "muons", "taus"):
        feature_map[name] = {
            feature: [offset + feature_index]
            for feature_index, feature in enumerate(("Et", "eta", "phi"))
        }
        offset += 3
        parameters = {
            feature: {"shift": 1.0, "scale": 2.0}
            for feature in (("Et", "phi") if name == "FET" else ("Et", "eta", "phi"))
        }
        with (cache / f"{name}_norm_params.pkl").open("wb") as handle:
            pickle.dump(parameters, handle)
        if name == "FET":
            array = ak.Array({"Et": [[3], [5]], "phi": [[7], [9]]})
        else:
            array = ak.Array(
                [
                    [{"Et": 3, "eta": 5, "phi": 7}],
                    [{"Et": 5, "eta": 7, "phi": 9}],
                ]
            )
        ak.to_parquet(array, run / f"{name}.parquet")
    (cache / "object_feature_map.json").write_text(
        __import__("json").dumps(feature_map), encoding="utf-8"
    )
    return run, cache


def test_processed_run_reader_replays_training_preprocessing(tmp_path: Path) -> None:
    run, cache = _write_processed_run(tmp_path)
    reader = ProcessedRunReader(run, load_run_schema(cache), batch_size=2)
    offset, values, masks = next(iter(reader))
    assert offset == 0
    assert values.shape == masks.shape == (2, 15)
    assert torch.equal(values[:, :3], torch.tensor([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]]))
    assert torch.equal(masks[:, :3], torch.tensor([[True, False, True], [True, False, True]]))
    assert torch.equal(values[0, 3:], torch.tensor([1.0, 2.0, 3.0] * 4))
    assert masks[:, 3:].all()


def test_iterative_flat_pairing_is_complete_unique_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(29, 8)).astype(np.float32)
    target = reference[:23] + 0.01 * rng.normal(size=(23, 8)).astype(np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    kwargs = {
        "backend": "flat",
        "k": 3,
        "nlist": 4,
        "nprobe": 4,
        "train_events": 20,
        "train_iterations": 2,
        "search_batch_size": 7,
        "add_batch_size": 11,
        "threads": 2,
        "seed": 123,
    }
    first, first_rounds = deterministic_iterative_pairing(target, reference, **kwargs)
    second, second_rounds = deterministic_iterative_pairing(target, reference, **kwargs)
    assert first.n_pairs == target.shape[0]
    assert first.n_accepted == target.shape[0]
    assert torch.unique(first.target_to_reference).numel() == target.shape[0]
    assert torch.equal(first.target_to_reference, second.target_to_reference)
    assert torch.equal(first.reference_to_target, second.reference_to_target)
    assert torch.equal(first.distance, second.distance)
    assert first_rounds == second_rounds


def test_iterative_ivf_pairing_widens_probe_to_complete() -> None:
    rng = np.random.default_rng(19)
    reference = rng.normal(size=(101, 8)).astype(np.float32)
    target = rng.normal(size=(97, 8)).astype(np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    pairing, rounds = deterministic_iterative_pairing(
        target,
        reference,
        backend="ivf_flat",
        k=2,
        nlist=8,
        nprobe=1,
        train_events=101,
        train_iterations=4,
        search_batch_size=31,
        add_batch_size=41,
        threads=2,
        seed=123,
    )
    assert pairing.n_pairs == target.shape[0]
    assert torch.unique(pairing.target_to_reference).numel() == target.shape[0]
    assert any("nprobe_after" in row for row in rounds)
