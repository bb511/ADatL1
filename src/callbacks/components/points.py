"""Points files: fixed inputs in a model's input space, checked against their md5s.

``load_points`` reads ``x`` and the optional ``mask`` of an ``.npz`` file and refuses
the file unless both arrays are the declared ones, so that the evaluation's replay
(``src.evaluation.components.rebuild.cached_points``) and a training callback that
scores points read a file alike.
"""

import hashlib

import numpy as np
import torch


def _md5(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    return hashlib.md5(array.tobytes(), usedforsecurity=False).hexdigest()


def _check_points(
    path: str, x: np.ndarray, mask: np.ndarray | None, md5_x: str, md5_mask: str | None
) -> None:
    found = (_md5(x), _md5(mask))
    bool_mask = mask is None or mask.dtype == np.bool_
    if x.dtype == np.float32 and bool_mask and found == (md5_x, md5_mask):
        return
    dtypes = f"x {x.dtype}" + ("" if mask is None else f", mask {mask.dtype}")
    raise ValueError(
        f"{path} holds {dtypes}, x md5 {found[0]} and mask md5 {found[1]}; "
        f"declared: x float32 of md5 {md5_x}, mask md5 {md5_mask}."
    )


def load_points(
    path: str, md5_x: str, md5_mask: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``x`` and ``mask`` of an ``.npz`` points file, the mask None where it has none.

    Each md5 is that of the C-contiguous array's bytes alone, as Theory XXXI declares
    the frozen files (``FREEZE.md``); ``md5_mask`` None declares a file without a mask.

    Raises:
        ValueError: Naming the file, the md5s found and those declared, unless x is
            float32, the mask (if any) bool, and both md5s are the declared ones.
    """
    with np.load(path) as npz:
        x = np.ascontiguousarray(npz["x"])
        mask = np.ascontiguousarray(npz["mask"]) if "mask" in npz.files else None
    _check_points(path, x, mask, md5_x, md5_mask)
    return torch.from_numpy(x), None if mask is None else torch.from_numpy(mask)
