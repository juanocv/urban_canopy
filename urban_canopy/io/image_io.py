"""Image decoding, explicit colour conversion and visualisation helpers."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from urban_canopy.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "ImageLoadError",
    "decode_rgb",
    "ensure_rgb_u8",
    "from_bgr_array",
    "read_rgb",
    "mask_overlay_bgr",
    "png_b64",
]

TREE_COLOR_BGR = (0, 200, 0)
VEGETATION_COLOR_BGR = (0, 190, 190)


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded."""


def ensure_rgb_u8(rgb: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Validate the public in-memory image contract: non-empty H x W x 3 uint8 RGB.

    No colour conversion, dtype coercion or range scaling is performed. This is
    deliberate: an ndarray carries no reliable metadata from which RGB versus
    BGR can be inferred.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an H x W x 3 RGB array, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected an uint8 RGB array, got dtype {arr.dtype}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Expected a non-empty RGB array.")
    if copy:
        return np.array(arr, dtype=np.uint8, order="C", copy=True)
    return np.ascontiguousarray(arr)


def from_bgr_array(bgr: np.ndarray) -> np.ndarray:
    """Explicitly convert a non-empty H x W x 3 uint8 BGR array to RGB."""
    # Reuse the same structural contract; the helper name, not array metadata,
    # supplies the colour-space meaning.
    validated = ensure_rgb_u8(bgr)
    return cv2.cvtColor(validated, cv2.COLOR_BGR2RGB)


def decode_rgb(src: str | Path | bytes) -> np.ndarray:
    """
    Decode an encoded path or byte string and return H x W x 3 uint8 RGB.

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
        raise TypeError(
            "decode_rgb accepts only a path or encoded bytes; use "
            "ensure_rgb_u8() for RGB arrays or from_bgr_array() for BGR arrays."
        )

    return from_bgr_array(arr)


def read_rgb(src: str | Path | bytes) -> np.ndarray:
    """Backward-compatible name for :func:`decode_rgb`; arrays are not accepted."""
    return decode_rgb(src)


def mask_overlay_bgr(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Sequence[int] = TREE_COLOR_BGR,
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Blend *mask* over *rgb* (RGB in, BGR out, ready for ``cv2.imwrite``)."""
    bgr = cv2.cvtColor(ensure_rgb_u8(rgb), cv2.COLOR_RGB2BGR)
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


def png_b64(arr: np.ndarray) -> str:
    """PNG-encode a BGR/grayscale array and return it base64-encoded."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:  # pragma: no cover - cv2 only fails on malformed input
        raise ValueError("cv2.imencode failed to encode the array as PNG.")
    return base64.b64encode(buf.tobytes()).decode()
