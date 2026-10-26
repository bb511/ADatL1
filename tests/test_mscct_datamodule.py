import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.components import mscct
from src.data.components.mscct_pairing import oracle_partners, slide_affine
from src.data.MSCCT_datamodule import MSCCTDataModule

# The split recorded in plans/adl1t/01_data.md (26 Oct 2026).
RECORDED_SPLIT = [
    [2, 3, 4, 5, 8, 9, 11, 12, 13, 15, 18, 19, 23, 24, 26, 28, 31, 36, 37, 38, 39, 41]
    + [42, 44],
    [1, 10, 16, 20, 21, 27, 30, 32, 33, 35],
    [6, 7, 14, 17, 22, 25, 29, 34, 40, 43],
]
# Level-1 patch counts after the filter (out/v2_counts_per_slide.csv of the 26 Oct
# evidence), per scanner and class: train, validation, test.
RECORDED_COUNTS = {
    ("cs2", "dermis"): (31238, 16640, 11463),
    ("cs2", "tumour"): (22510, 5501, 9657),
    ("cs2", "inflammation_necrosis"): (421, 789, 76),
    ("nz20", "dermis"): (39635, 21112, 14668),
    ("nz210", "dermis"): (42132, 22425, 15611),
    ("p1000", "dermis"): (29514, 15703, 11316),
    ("gt450", "dermis"): (28611, 15259, 10529),
}

# The toy set: six slides on two scanners, page 0 of 192 x 256, patches of 8 at level
# 1, so 16 x 16 page-0 footprints on a 12 x 16 grid.
SLIDES = range(1, 7)
TOY_SCANNERS = ("cs2", "nz20")
PATCH, FOOTPRINT = 8, 16
HEIGHT, WIDTH = 192, 256
# Cells fully inside each rectangle: dermis rows 1-10 and cols 1-14, tumour rows 3-5
# and cols 3-6, inflammation row 7 and cols 10-11, both nested in the dermis.
POLYGONS = [
    (3, (16, 16, 239, 175)),
    (11, (48, 48, 111, 95)),
    (6, (160, 112, 191, 127)),
]
# Dermis cells overwritten on page 1, the first exactly white, the second flat.
WHITE_CELL, FLAT_CELL = (1, 1), (1, 2)
TOY_COUNTS = {"dermis": 140 - 12 - 2 - 2, "tumour": 12, "inflammation_necrosis": 2}


def _rect(x0: int, y0: int, x1: int, y1: int) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _page1(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    page0 = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    sums = page0.reshape(HEIGHT // 2, 2, WIDTH // 2, 2, 3).sum(axis=(1, 3), dtype=int)
    page1 = ((sums + 2) // 4).astype(np.uint8)
    for (r, c), value in ((WHITE_CELL, 255), (FLAT_CELL, 100)):
        page1[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH] = value
    return page0, page1


def _write_dataset(root: Path) -> None:
    images, annotations = [], []
    for slide in SLIDES:
        for scanner in TOY_SCANNERS:
            name = mscct.file_name(slide, scanner)
            rng = np.random.default_rng([slide, TOY_SCANNERS.index(scanner)])
            page0, page1 = _page1(rng)
            Image.fromarray(page0).save(
                root / name, save_all=True, append_images=[Image.fromarray(page1)]
            )
            image_id = len(images) + 1
            images.append(dict(id=image_id, file_name=name, width=WIDTH, height=HEIGHT))
            for category, rect in POLYGONS:
                annotations.append(
                    dict(
                        id=len(annotations) + 1,
                        image_id=image_id,
                        category_id=category,
                        segmentation=_rect(*rect),
                    )
                )
    coco = dict(images=images, annotations=annotations)
    (root / mscct.ANNOTATIONS).write_text(json.dumps(coco))
    files = sorted(p for p in root.iterdir() if p.suffix in (".tif", ".json"))
    record = [dict(key=p.name, checksum=f"md5:{_md5(p)}") for p in files]
    (root / mscct.RECORD).write_text(json.dumps(dict(files=record)))


@pytest.fixture
def toy_dir(tmp_path: Path) -> Path:
    _write_dataset(tmp_path)
    return tmp_path


def _toy(data_dir: Path, **overrides) -> MSCCTDataModule:
    kwargs = dict(
        data_dir=str(data_dir),
        shift_scanners=["nz20"],
        split_sizes=[2, 2, 2],
        patch_size=PATCH,
        batch_size=32,
    )
    dm = MSCCTDataModule(**(kwargs | overrides))
    dm.setup()
    return dm


def _rows(x: torch.Tensor) -> set[bytes]:
    return {row.numpy().tobytes() for row in x}


def test_split_reproduces_the_recorded_parts():
    assert mscct.slide_split(list(range(1, 45)), [24, 10, 10], 314159) == RECORDED_SPLIT


def test_smallest_polygon_wins_whatever_the_file_order():
    small, large = np.array(_rect(4, 4, 7, 7)), np.array(_rect(0, 0, 15, 15))
    polygons = [(11, small.reshape(-1, 2)), (3, large.reshape(-1, 2))]
    labels = mscct.paint_labels(16, 16, polygons)
    assert (labels[4:8, 4:8] == 11).all()
    assert (labels == 3).sum() == 16 * 16 - 16


def test_tissue_mask():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
    textured[1].reshape(64, 3)[:32] = 255
    textured[2].reshape(64, 3)[:31] = 255
    textured[3] = 120 + rng.integers(0, 3, (8, 8, 3), dtype=np.uint8)
    textured[4, ..., :2] = 120
    textured[5].reshape(64, 3)[:32, :2] = 255
    keep = mscct.tissue_mask(textured, white_share=0.5, flat_sd=2.0)
    assert keep.tolist() == [True, False, True, False, True, True]
    assert mscct.tissue_mask(textured[:0], 0.5, 2.0).shape == (0,)


def test_extract_takes_whole_cells_and_filters(toy_dir):
    image = mscct.annotations_by_file(toy_dir / mscct.ANNOTATIONS)["scc_01_cs2.tif"]
    path = toy_dir / "scc_01_cs2.tif"
    out = mscct.extract(path, image, list(TOY_COUNTS), 1, PATCH, 0.5, 2.0)
    assert {name: len(cells) for name, (_, cells) in out.items()} == TOY_COUNTS
    page = mscct.read_page(path, 1)
    patches, cells = out["tumour"]
    for patch, (r, c) in zip(patches, cells):
        tile = page[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
        assert np.array_equal(patch, tile)
    dermis = {tuple(rc) for rc in out["dermis"][1].tolist()}
    assert WHITE_CELL not in dermis and FLAT_CELL not in dermis
    assert {r for r, _ in dermis} == set(range(1, 11))


def test_sets_labels_and_sizes(toy_dir):
    dm = _toy(toy_dir)
    names = ["normal", "tumour", "inflammation_necrosis", "reference_normal", "nz20"]
    labels = [0, 1, -1, 0, 0]
    for loaders in (dm.val_dataloader(), dm.test_dataloader()):
        assert list(loaders) == names
        for name, label in zip(names, labels):
            ys = torch.cat([y for _, y in loaders[name]])
            assert set(ys.tolist()) == {label}
    sizes = {name: len(split.x) for name, split in dm._sets["valid"].items()}
    two_slides = {name: 2 * n for name, n in TOY_COUNTS.items()}
    assert sizes["normal"] == sizes["reference_normal"] == two_slides["dermis"] // 2
    assert sizes["tumour"] == two_slides["tumour"]
    assert sizes["inflammation_necrosis"] == two_slides["inflammation_necrosis"]
    assert sizes["nz20"] == two_slides["dermis"]
    assert len(dm.data_train) == two_slides["dermis"]
    assert dm._sets["train"]["normal"].x.shape[1:] == (3, PATCH, PATCH)


def test_parts_and_sets_share_no_patch(toy_dir):
    dm = _toy(toy_dir)
    slides = {
        part: set(p["normal"][:, 0].tolist()) for part, p in dm.provenance.items()
    }
    assert not (slides["train"] & slides["valid"] or slides["valid"] & slides["test"])
    assert not slides["train"] & slides["test"]
    rows = [(n, _rows(s.x)) for p in dm._sets.values() for n, s in p.items()]
    for i, (name_i, rows_i) in enumerate(rows):
        for name_j, rows_j in rows[i + 1 :]:
            assert not rows_i & rows_j, (name_i, name_j)
    for part in ("valid", "test"):
        cs2 = [dm.provenance[part][n] for n in ("normal", "reference_normal")]
        cells = [set(map(tuple, p.tolist())) for p in cs2]
        assert not cells[0] & cells[1]


def test_training_rows_are_permuted_once(toy_dir):
    dm = _toy(toy_dir)
    provenance = dm.provenance["train"]["normal"]
    assert not np.array_equal(provenance, provenance[np.lexsort(provenance.T[::-1])])
    again = _toy(toy_dir)
    assert torch.equal(dm.data_train.tensors[0], again.data_train.tensors[0])


def test_train_seed_changes_the_batch_order_only(toy_dir):
    dm_a, dm_b = (_toy(toy_dir, train_seed=s) for s in (12345, 54321))
    for part, sets in dm_a._sets.items():
        for name, split in sets.items():
            assert torch.equal(split.x, dm_b._sets[part][name].x)
    xa, xb = (torch.cat([x for x, _ in dm.train_dataloader()]) for dm in (dm_a, dm_b))
    assert not torch.equal(xa, xb)
    assert _rows(xa) == _rows(xb)


def test_normalisation_comes_from_the_training_patches(toy_dir):
    dm = _toy(toy_dir)
    scans = [dm._scan(s, "cs2") for s in dm._parts()["train"]]
    raw = np.concatenate([s["patches"]["dermis"].numpy() for s in scans]) / 255.0
    expected_mean = raw.mean(axis=(0, 1, 2))
    expected_std = raw.reshape(-1, 3).std(axis=0, ddof=1)
    torch.testing.assert_close(dm.mean, torch.tensor(expected_mean).float())
    torch.testing.assert_close(dm.std, torch.tensor(expected_std).float())
    undone = dm.data_train.tensors[0] * dm.std[:, None, None] + dm.mean[:, None, None]
    assert abs(float(undone.mean()) - float(raw.mean())) < 1e-5


def test_warm_build_reads_no_scan(toy_dir):
    cold = _toy(toy_dir)
    for path in toy_dir.glob("*.tif"):
        path.unlink()
    (toy_dir / mscct.ANNOTATIONS).unlink()
    warm = _toy(toy_dir)
    for part, sets in cold._sets.items():
        for name, split in sets.items():
            assert torch.equal(split.x, warm._sets[part][name].x)
            assert np.array_equal(
                cold.provenance[part][name], warm.provenance[part][name]
            )


def test_a_changed_key_rebuilds_the_cache(toy_dir, monkeypatch):
    _toy(toy_dir)
    calls = []
    extract = mscct.extract
    monkeypatch.setattr(mscct, "extract", lambda *a: calls.append(a[0]) or extract(*a))
    _toy(toy_dir)
    assert not calls
    _toy(toy_dir, flat_sd=1.0)
    assert len(calls) == 6 + 4
    cached = torch.load(next((toy_dir / "patches").glob("*.pt")))
    assert cached["key"]["flat_sd"] == 1.0


def test_a_scan_that_differs_from_the_record_is_refused(toy_dir):
    with open(toy_dir / "scc_03_cs2.tif", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="md5"):
        _toy(toy_dir)


def test_a_missing_record_names_the_doi(tmp_path):
    with pytest.raises(FileNotFoundError, match="zenodo.7418555"):
        _toy(tmp_path)


def test_slide_affine_recovers_the_map():
    rng = np.random.default_rng(1)
    source = [(3, rng.uniform(0, 500, (40, 2))), (11, rng.uniform(0, 500, (7, 2)))]
    affine = np.array([[1.116, 0.004, 31.0], [-0.003, 1.108, -12.5]])
    target = [(c, v @ affine[:, :2].T + affine[:, 2]) for c, v in source]
    np.testing.assert_allclose(slide_affine(source, target), affine, atol=1e-9)
    with pytest.raises(ValueError):
        slide_affine(source, target[:1])


def test_oracle_partners_follow_the_map():
    source_polygons = [(3, np.array(_rect(0, 0, 400, 300), float).reshape(-1, 2))]
    shift = np.array([2 * FOOTPRINT, FOOTPRINT])
    target_polygons = [(3, source_polygons[0][1] + shift)]
    annotations = {
        mscct.file_name(1, "cs2"): {"polygons": source_polygons},
        mscct.file_name(1, "nz20"): {"polygons": target_polygons},
    }
    source = np.array([[1, 2, 3], [1, 5, 5]])
    target = np.array([[1, 0, 0], [1, 3, 5]])
    partners = oracle_partners(annotations, source, target, ("cs2", "nz20"), FOOTPRINT)
    assert partners.tolist() == [1, -1]


def test_oracle_on_identical_geometry_pairs_each_cell_with_itself(toy_dir):
    dm = _toy(toy_dir)
    annotations = mscct.annotations_by_file(toy_dir / mscct.ANNOTATIONS)
    source, target = dm.provenance["valid"]["normal"], dm.provenance["valid"]["nz20"]
    partners = oracle_partners(annotations, source, target, ("cs2", "nz20"), 16)
    assert (partners >= 0).all()
    assert np.array_equal(target[partners], source)


@pytest.mark.skipif(
    "ADL1T_MSCCT_DIR" not in os.environ,
    reason="needs the MS-CCT record (or its patch caches) in ADL1T_MSCCT_DIR",
)
def test_counts_match_the_record_of_26_october():
    dm = MSCCTDataModule(data_dir=os.environ["ADL1T_MSCCT_DIR"])
    dm.setup()
    parts = dm._parts()
    assert [parts[part] for part in ("train", "valid", "test")] == RECORDED_SPLIT
    evaluation = [dm.provenance[part] for part in ("valid", "test")]
    for (scanner, name), counts in RECORDED_COUNTS.items():
        # The other scanners' training slides are not served, so not counted.
        if scanner != "cs2":
            assert tuple(len(sets[scanner]) for sets in evaluation) == counts[1:]
        elif name == "dermis":
            halves = [len(s["normal"]) + len(s["reference_normal"]) for s in evaluation]
            assert (len(dm.provenance["train"]["normal"]), *halves) == counts
        else:
            train = len(dm._raw(scanner, [name], parts["train"])[0])
            assert (train, *(len(sets[name]) for sets in evaluation)) == counts
