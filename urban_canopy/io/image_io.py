"""
Image loading and canopy visualisation helpers.

``read_rgb`` is carried over from the sidewalk pipeline: it is the single place
that guarantees every downstream stage sees H x W x 3 uint8 RGB, whatever the
caller passed in. The overlays are new -- the old ones drew sidewalks and
obstacle bases, which have no meaning here.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Iterable, Sequence, Union

import cv2
import numpy as np

from urban_canopy.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "ImageLoadError",
    "read_rgb",
    "mask_overlay_bgr",
    "instances_overlay_bgr",
    "png_b64",
]

TREE_COLOR_BGR = (0, 200, 0)
VEGETATION_COLOR_BGR = (0, 190, 190)


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded."""


def read_rgb(src: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Load an image and **always** return H x W x 3 uint8 RGB.

    Parameters
    ----------
    src
        Path/str, raw encoded bytes, or an already-loaded BGR ndarray.

    Raises
    ------
    ImageLoadError
        If OpenCV cannot decode the source.
    """
    if isinstance(src, (str, Path)):
        arr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if arr is None:
            raise ImageLoadError(f"OpenCV failed to read {str(src)!r}.")
    elif isinstance(src, bytes):
        arr = cv2.imdecode(np.frombuffer(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ImageLoadError("OpenCV failed to decode in-memory bytes.")
    else:
        arr = np.asarray(src)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ImageLoadError(f"Expected an H x W x 3 array, got shape {arr.shape}.")
        arr = arr.copy()

    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def mask_overlay_bgr(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Sequence[int] = TREE_COLOR_BGR,
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Blend *mask* over *rgb* (RGB in, BGR out, ready for ``cv2.imwrite``)."""
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    m = np.asarray(mask).astype(bool)
    if m.shape != bgr.shape[:2]:
        raise ValueError(f"Mask shape {m.shape} does not match image {bgr.shape[:2]}.")
    if not m.any():
        return bgr

    out = bgr.copy()
    tint = np.asarray(color, dtype=np.float32)
    out[m] = ((1.0 - alpha) * out[m].astype(np.float32) + alpha * tint).astype(np.uint8)

    if outline:
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 1)
    return out


def _color_for_index(index: int) -> tuple[int, int, int]:
    # blake2b rather than hash(): str/int hashing is salted per process
    # (PYTHONHASHSEED), which would recolour the same instance on every run.
    digest = hashlib.blake2b(str(index).encode("utf-8"), digest_size=3).digest()
    return tuple(50 + int(b) % 206 for b in digest)  # type: ignore[return-value]


def instances_overlay_bgr(rgb: np.ndarray, instances: Iterable) -> np.ndarray:
    """
    Draw one colour per instance mask, with an index label at its centroid.

    Instances are only ever drawn when a backend actually produced them, or
    when the connected-component heuristic was explicitly requested; the caller
    decides, this function just renders whatever list it is handed.
    """
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    for index, inst in enumerate(instances):
        mask = np.asarray(getattr(inst, "mask", inst)).astype(bool)
        if mask.shape != out.shape[:2] or not mask.any():
            continue
        color = _color_for_index(index)
        out[mask] = (0.55 * out[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)
        ys, xs = np.nonzero(mask)
        cv2.putText(
            out,
            str(index),
            (int(xs.mean()), int(ys.mean())),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def png_b64(arr: np.ndarray) -> str:
    """PNG-encode a BGR/grayscale array and return it base64-encoded."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:  # pragma: no cover - cv2 only fails on malformed input
        raise ValueError("cv2.imencode failed to encode the array as PNG.")
    return base64.b64encode(buf).decode()
