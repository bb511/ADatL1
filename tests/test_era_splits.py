import filecmp

import numpy as np
import pytest

from src.data.components import era_splits

N_EVENTS, N_FIRST = 20, 8


@pytest.fixture
def merged(tmp_path):
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    permutation = np.random.default_rng(0).permutation(N_EVENTS)
    arrays = dict(zip(era_splits.SPLITS, np.split(permutation, [12, 16])))
    np.savez(merged_dir / "zerobias.npz", **arrays)
    for seed, name in enumerate(("aux__a", "aux__b")):
        aux = np.random.default_rng(seed).permutation(10)
        np.savez(merged_dir / f"{name}.npz", ivalid=aux[:6], itest=aux[6:])

    return merged_dir, arrays


def _derive(merged_dir, era_dir, era):
    zerobias, aux = era_splits.plan_era_splits(merged_dir, era_dir, era, N_FIRST)
    era_splits.write_era_splits(era_dir, zerobias, aux)


@pytest.mark.parametrize(
    "era, keep, n_era",
    [
        ("E", lambda i: i[i < N_FIRST], N_FIRST),
        ("G", lambda i: i[i >= N_FIRST] - N_FIRST, N_EVENTS - N_FIRST),
    ],
)
def test_era_zerobias_is_the_shifted_restriction(merged, tmp_path, era, keep, n_era):
    merged_dir, arrays = merged
    _derive(merged_dir, tmp_path / "era", era)

    with np.load(tmp_path / "era" / "zerobias.npz") as written:
        splits = [written[k] for k in era_splits.SPLITS]
    for split, key in zip(splits, era_splits.SPLITS):
        assert split.dtype == arrays[key].dtype
        assert np.array_equal(split, keep(arrays[key]))
    joined = np.concatenate(splits)
    assert len(joined) == len(set(joined.tolist())) == n_era
    assert sorted(joined.tolist()) == list(range(n_era))


def test_aux_splits_are_copied_unchanged(merged, tmp_path):
    merged_dir, _ = merged
    _derive(merged_dir, tmp_path / "era", "G")

    for name in ("aux__a.npz", "aux__b.npz"):
        assert filecmp.cmp(merged_dir / name, tmp_path / "era" / name, shallow=False)


def test_refuses_to_overwrite(merged, tmp_path):
    merged_dir, _ = merged
    era_dir = tmp_path / "era"
    era_dir.mkdir()
    (era_dir / "aux__b.npz").write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _derive(merged_dir, era_dir, "E")

    assert sorted(p.name for p in era_dir.iterdir()) == ["aux__b.npz"]
    assert (era_dir / "aux__b.npz").read_bytes() == b"keep"
