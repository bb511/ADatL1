"""The coupling oracle of MS-CCT: the partner, on another scanner, of each patch.

The scanners share the glass, not the pixels. Each scanner's polygons in ``scc.json``
are one affine map of CS2's per slide, up to the rounding of both vertex sets to
integers (vertex RMS residual 0.57-0.62 px at page 0); the images follow that map to a
median 2-3 px at page 0, 95th percentile 8-11 px, measured by cross-correlation on 26
Oct 2026. The partner of a patch is the target scanner's grid cell that contains the
centre of the patch's footprint carried through its slide's map: an approximate
pairing, where CIFAR-10's identity is exact.

Coordinates are page-0 pixel coordinates with integer values at pixel centres, x the
column and y the row, as the polygons'; a grid cell (row, col) of page ``level`` covers
the page-0 pixels ``row * footprint`` to ``(row + 1) * footprint - 1``, with
``footprint = patch << level``.
"""

import numpy as np

from src.data.components.mscct import file_name


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
