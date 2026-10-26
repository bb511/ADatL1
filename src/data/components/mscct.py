"""Patches of MS-CCT, the multi-scanner canine cutaneous squamous cell carcinoma set.

Wilm et al., BVM 2023 (arXiv 2301.04423); Zenodo record 7418555, CC BY 4.0. The same
44 slides were scanned on five scanners. ``scc_NN_<scanner>.tif`` is a pyramidal
BigTIFF whose page 0 is the scan at 16 times its native pitch (CS2 4 um/px) and whose
page L halves page L - 1, rounded down; page 1 is the half-up-rounded 2 x 2 mean of
page 0. ``scc.json`` (MS COCO) holds every image's polygons as one flat vertex list in
page-0 pixel coordinates, x the column and y the row, with integer vertices at pixel
centres; they were drawn on CS2 and carried to the other scanners by one affine map
per slide.

A patch is a cell of the non-overlapping ``patch`` x ``patch`` grid of page ``level``,
counted from the origin; the incomplete cells at the far edges are left out. A patch
belongs to a class when every page-0 pixel of its footprint carries that class after
the polygons are painted in order of decreasing area, so that the smallest covering
polygon wins: tumour, subcutis and inflammation drawn inside a dermis outline keep
their own class. The authors' file-order painting differs on at most 8.2e-4 of a
class's pixels (measured 26 Oct 2026).
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCANNERS = ("cs2", "nz20", "nz210", "p1000", "gt450")
# COCO category ids of scc.json. "tumour" is SCC (11), the only one of its seven tumour
# categories that carries annotations.
CLASS_IDS = {
    "bone": 1,
    "cartilage": 2,
    "dermis": 3,
    "epidermis": 4,
    "subcutis": 5,
    "inflammation_necrosis": 6,
    "tumour": 11,
}
DOI = "10.5281/zenodo.7418555"
ANNOTATIONS = "scc.json"
RECORD = "record.json"


def file_name(slide: int, scanner: str) -> str:
    return f"scc_{int(slide):02d}_{scanner}.tif"


def slide_split(slides: list[int], sizes: list[int], seed: int) -> list[list[int]]:
    """Return the slides of each part, each sorted, cut from one seeded permutation.

    The permutation of the sorted slide numbers is cut into consecutive parts of
    ``sizes`` (train, validation, test), so every scanner's copy of a slide falls in
    its slide's part.
    """
    if sum(sizes) != len(slides):
        raise ValueError(
            f"Part sizes {list(sizes)} do not sum to {len(slides)} slides."
        )
    perm = np.random.default_rng(seed).permutation(sorted(slides))
    bounds = np.cumsum([0, *sizes])
    return [sorted(perm[a:b].tolist()) for a, b in zip(bounds[:-1], bounds[1:])]


def record_md5s(path: Path) -> dict[str, str]:
    """Return the md5 of every file of the Zenodo record's ``record.json``, by name."""
    files = json.loads(Path(path).read_text())["files"]
    return {f["key"]: f["checksum"].removeprefix("md5:") for f in files}


def file_md5(path: Path) -> str:
    """Return a file's md5, read in chunks of 16 MiB so that no scan is held whole."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotations_by_file(path: Path) -> dict[str, dict]:
    """Return every image's page-0 size and polygons, keyed by file name.

    Returns:
        ``{file_name: {"width": int, "height": int, "polygons": [(category_id,
        vertices)]}}``, the polygons in file order and each one's vertices [n, 2] as
        (x, y). The release stores a polygon as one flat list, not COCO's list of
        lists.
    """
    data = json.loads(Path(path).read_text())
    names = {image["id"]: image["file_name"] for image in data["images"]}
    out = {
        image["file_name"]: dict(
            width=image["width"], height=image["height"], polygons=[]
        )
        for image in data["images"]
    }
    for a in data["annotations"]:
        vertices = np.asarray(a["segmentation"], dtype=np.float64).reshape(-1, 2)
        out[names[a["image_id"]]]["polygons"].append((a["category_id"], vertices))
    return out


def _shoelace_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def paint_labels(width: int, height: int, polygons: list[tuple]) -> np.ndarray:
    """Return the page-0 label map [height, width] uint8 of category ids, 0 outside
    every polygon.

    The polygons are painted, interior and outline, in order of decreasing shoelace area
    with ties kept in file order, so a pixel takes the class of the smallest polygon
    covering it.
    """
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    order = np.argsort([-_shoelace_area(v) for _, v in polygons], kind="stable")
    for i in order:
        category, vertices = polygons[i]
        draw.polygon([tuple(p) for p in vertices], fill=category, outline=category)
    return np.asarray(canvas)


def cell_extremes(labels: np.ndarray, footprint: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the least and the greatest label [rows, cols] in every complete footprint
    x footprint cell of a label map."""
    rows, cols = labels.shape[0] // footprint, labels.shape[1] // footprint
    cells = labels[: rows * footprint, : cols * footprint]
    cells = cells.reshape(rows, footprint, cols, footprint)
    return cells.min(axis=(1, 3)), cells.max(axis=(1, 3))


def read_page(path: Path, page: int) -> np.ndarray:
    """Return one page of a pyramidal TIFF as [rows, cols, 3] uint8 RGB."""
    with warnings.catch_warnings():
        # Pillow checks page 0 against its decompression-bomb threshold on opening,
        # which the largest scans exceed (90 Mpx), whichever page is then read.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.seek(page)
            return np.asarray(image.convert("RGB"))


def tissue_mask(patches: np.ndarray, white_share: float, flat_sd: float) -> np.ndarray:
    """Return True [n] for the patches [n, p, p, 3] uint8 that carry tissue.

    A patch is dropped when at least ``white_share`` of its pixels are exactly (255,
    255, 255), the fill outside the P1000 tissue, or when the standard deviation (ddof
    0) of every channel is below ``flat_sd`` grey levels, as on uniform background.
    """
    n, rows, cols = patches.shape[:3]
    flat = patches.reshape(n, rows * cols, 3)
    white = (flat == 255).all(axis=2).mean(axis=1) >= white_share
    uniform = (flat.std(axis=1) < flat_sd).all(axis=1)
    return ~(white | uniform)


def _tiles(page: np.ndarray, cells: np.ndarray, patch: int) -> np.ndarray:
    """Return the patches [n, patch, patch, 3] of a page at grid cells [n, 2]."""
    rows, cols = page.shape[0] // patch, page.shape[1] // patch
    grid = page[: rows * patch, : cols * patch].reshape(rows, patch, cols, patch, 3)
    return grid.swapaxes(1, 2)[cells[:, 0], cells[:, 1]]


def extract(
    path: Path,
    image: dict,
    classes: list[str],
    level: int,
    patch: int,
    white_share: float,
    flat_sd: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per class the tissue patches of one scan and their cells.

    Args:
        image: The scan's entry of ``annotations_by_file``.

    Returns:
        ``{class: (patches, cells)}``: patches [n, patch, patch, 3] uint8 of page
        ``level`` and cells [n, 2] int32 (row, col) on that page's grid, in row-major
        cell order.
    """
    labels = paint_labels(image["width"], image["height"], image["polygons"])
    least, greatest = cell_extremes(labels, patch << level)
    page = read_page(path, level)
    expected = (image["height"] >> level, image["width"] >> level)
    if page.shape[:2] != expected:
        raise ValueError(f"{path}: page {level} is {page.shape[:2]}, not {expected}.")
    out = {}
    for name in classes:
        cells = np.argwhere((least == CLASS_IDS[name]) & (greatest == CLASS_IDS[name]))
        patches = _tiles(page, cells, patch)
        keep = tissue_mask(patches, white_share, flat_sd)
        out[name] = (np.ascontiguousarray(patches[keep]), cells[keep].astype(np.int32))
    return out


def channel_moments(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-channel mean and standard deviation (ddof 1), in grey levels, of
    patches [n, p, p, 3] uint8, exactly from each channel's histogram."""
    levels = np.arange(256, dtype=np.float64)
    counts = np.stack(
        [np.bincount(patches[..., c].ravel(), minlength=256) for c in range(3)]
    ).astype(np.float64)
    n = counts.sum(axis=1)
    mean = counts @ levels / n
    var = (counts * (levels - mean[:, None]) ** 2).sum(axis=1) / (n - 1)
    return mean, np.sqrt(var)
