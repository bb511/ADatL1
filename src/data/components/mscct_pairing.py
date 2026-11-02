"""The coupling oracle of MS-CCT: the partner, on another scanner, of each patch.

The scanners share the glass, not the pixels. Each scanner's polygons in ``scc.json``
are one affine map of CS2's per slide, up to the rounding of both vertex sets to
integers (vertex RMS residual 0.57-0.62 px at page 0); the images follow that map to a
median 2-3 px at page 0, 95th percentile 8-11 px, measured by cross-correlation on 26
Oct 2026. The partner of a patch is the target scanner's grid cell that contains the
centre of the patch's footprint carried through its slide's map: an approximate
pairing, where CIFAR-10's identity is exact. That cell holds about half of the mapped
footprint, so a coupling's partner agrees with the oracle when it lies in any target
cell the mapped footprint overlaps (``overlap_partners``).

Coordinates are page-0 pixel coordinates with integer values at pixel centres, x the
column and y the row, as the polygons'; a grid cell (row, col) of page ``level`` covers
the page-0 pixels ``row * footprint`` to ``(row + 1) * footprint - 1``, with
``footprint = patch << level``.
"""

import numpy as np

from src.data.components.mscct import file_name

# Page-0 px. A footprint edge this close to a cell boundary is put on it: a
# least-squares fit of an identity map is exact only to rounding (at most 1.1e-11 px on
# CS2's 44 slides), and the registration is good to 2-3 px only.
BOUNDARY_TOLERANCE = 1e-6


def slide_affine(source: list[tuple], target: list[tuple]) -> np.ndarray:
    """Return the least-squares affine map [2, 3] carrying the source polygons' vertices
    (x, y) onto the target's.

    The polygons of two scans of a slide correspond one to one in file order, vertex
    for vertex; a ValueError is raised where their categories or vertex counts differ.
    The map acts as ``target = affine[:, :2] @ source + affine[:, 2]``.
    """
    if [(c, len(v)) for c, v in source] != [(c, len(v)) for c, v in target]:
        raise ValueError("The two scans' polygons do not correspond one to one.")
    x = np.concatenate([v for _, v in source])
    y = np.concatenate([v for _, v in target])
    design = np.column_stack([x, np.ones(len(x))])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return solution.T


def cell_centres(cells: np.ndarray, footprint: int) -> np.ndarray:
    """Return the page-0 centres [n, 2] (x, y) of grid cells [n, 2] (row, col)."""
    return cells[:, ::-1] * footprint + (footprint - 1) / 2


def containing_cells(points: np.ndarray, footprint: int) -> np.ndarray:
    """Return the grid cells [n, 2] (row, col) containing page-0 points [n, 2] (x, y).

    Cell r spans the coordinates [r * footprint - 0.5, (r + 1) * footprint - 0.5),
    the extent of its pixels.
    """
    return np.floor((points[:, ::-1] + 0.5) / footprint).astype(np.int64)


def map_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def oracle_partners(
    annotations: dict,
    source: np.ndarray,
    target: np.ndarray,
    scanners: tuple[str, str],
    footprint: int,
) -> np.ndarray:
    """Return, per row of the source set, the row of the target set holding its partner,
    -1 where the partner cell is not in the target set.

    Args:
        annotations: ``annotations_by_file`` of ``scc.json``.
        source: Provenance [n, 3] (slide, cell row, cell col) of the source set in
            loader order, as ``MSCCTDataModule.provenance`` holds it.
        target: The target set's provenance, likewise.
        scanners: The source and the target scanner.
        footprint: Page-0 side of a patch, ``patch_size << level``.

    Returns:
        Partner rows [n] int64. Several source rows can share a partner where the
        target's pixels are larger than the source's.
    """
    index = {tuple(p): i for i, p in enumerate(target.tolist())}
    partners = np.full(len(source), -1, dtype=np.int64)
    for slide in np.unique(source[:, 0]).tolist():
        rows = np.flatnonzero(source[:, 0] == slide)
        polygons = [annotations[file_name(slide, s)]["polygons"] for s in scanners]
        affine = slide_affine(*polygons)
        points = map_points(cell_centres(source[rows, 1:], footprint), affine)
        cells = containing_cells(points, footprint).tolist()
        partners[rows] = [index.get((slide, r, c), -1) for r, c in cells]
    return partners


def _footprint_corners(cells: np.ndarray, footprint: int) -> np.ndarray:
    """Return the page-0 corners [n, 4, 2] (x, y) of the pixel extents of the grid cells
    [n, 2] (row, col)."""
    near = cells[:, ::-1] * footprint - 0.5
    steps = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * footprint
    return near[:, None, :] + steps


def _cell_ranges(corners: np.ndarray, footprint: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the first cell [n, 2] (row, col) of the cell ranges spanned by footprints
    of corners [n, 4, 2] (x, y), and their numbers of rows and columns [n, 2].

    A footprint is half-open like a cell, so an edge on a cell boundary, to within
    ``BOUNDARY_TOLERANCE``, reaches only the cell on its inner side.
    """
    yx = corners[..., ::-1] + 0.5
    first = np.floor((yx.min(axis=1) + BOUNDARY_TOLERANCE) / footprint)
    end = np.ceil((yx.max(axis=1) - BOUNDARY_TOLERANCE) / footprint)
    return first.astype(np.int64), (end - first).astype(np.int64)


def _range_cells(first: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return every cell [m, 2] (row, col) of the ranges, range by range and row by row,
    and the index [m] of the range holding each."""
    counts = shape.prod(axis=1)
    owner = np.repeat(np.arange(len(first)), counts)
    k = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    width = shape[owner, 1]
    return first[owner] + np.column_stack([k // width, k % width]), owner


def _overlapped_cells(
    annotations: dict,
    slide: int,
    cells: np.ndarray,
    scanners: tuple[str, str],
    footprint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the target cells [m, 2] (row, col) that the footprints of a slide's source
    cells [n, 2] overlap once mapped, and the index [m] of the source cell of each."""
    polygons = [annotations[file_name(slide, s)]["polygons"] for s in scanners]
    corners = map_points(_footprint_corners(cells, footprint), slide_affine(*polygons))
    return _range_cells(*_cell_ranges(corners, footprint))


def _csr(
    owner: np.ndarray, partner: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the offsets [n + 1] and the partners >= 0, grouped by owner in a stable
    order."""
    keep = partner >= 0
    order = np.argsort(owner[keep], kind="stable")
    counts = np.bincount(owner[keep], minlength=n)
    return np.concatenate([[0], np.cumsum(counts)]), partner[keep][order]


def overlap_partners(
    annotations: dict,
    source: np.ndarray,
    target: np.ndarray,
    scanners: tuple[str, str],
    footprint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return, per row of the source set, the rows of the target set whose cells the
    footprint overlaps once carried through its slide's map.

    The overlapped cells are those containing the four mapped corners of the
    footprint's pixel extent and every cell between them, i.e. the full ranges of rows
    and of columns the corners span, without a cap. Under a rotation the ranges also
    hold cells near their corners that the footprint misses. The extent is half-open
    like a cell's, so an edge on a cell boundary, to within ``BOUNDARY_TOLERANCE``,
    reaches only the cell on its inner side, and a footprint aligned with the target
    grid overlaps one cell. The cell of ``oracle_partners`` is always one of the
    overlapped cells.

    Args:
        annotations: ``annotations_by_file`` of ``scc.json``.
        source: Provenance [n, 3] (slide, cell row, cell col) of the source set in
            loader order, as ``MSCCTDataModule.provenance`` holds it.
        target: The target set's provenance, likewise.
        scanners: The source and the target scanner.
        footprint: Page-0 side of a patch, ``patch_size << level``.

    Returns:
        Offsets [n + 1] and rows [offsets[-1]], both int64, in CSR form: source row i
        overlaps the target rows ``rows[offsets[i]:offsets[i + 1]]``, in the order of
        their cells, row by row. A cell absent from the target set contributes no
        row, so a source row's segment can be empty.
    """
    index = {tuple(p): i for i, p in enumerate(target.tolist())}
    owners, partners = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for slide in np.unique(source[:, 0]).tolist():
        rows = np.flatnonzero(source[:, 0] == slide)
        cells, owner = _overlapped_cells(
            annotations, slide, source[rows, 1:], scanners, footprint
        )
        owners.append(rows[owner])
        found = [index.get((slide, r, c), -1) for r, c in cells.tolist()]
        partners.append(np.array(found, dtype=np.int64))
    return _csr(np.concatenate(owners), np.concatenate(partners), len(source))


def agrees_with_oracle(
    offsets: np.ndarray, rows: np.ndarray, partners: np.ndarray
) -> np.ndarray:
    """Return, per source row, whether its partner, a row [n] of the target set, is one
    of the rows its footprint overlaps, given ``overlap_partners``' offsets and rows.

    A partner of -1 never agrees.
    """
    owner = np.repeat(np.arange(len(partners)), np.diff(offsets))
    hits = owner[rows == partners[owner]]
    return np.bincount(hits, minlength=len(partners)) > 0
