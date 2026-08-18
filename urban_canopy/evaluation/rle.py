"""
Uncompressed COCO run-length encoding.

Masks have to travel between the analysis step and the evaluation step, and
between machines, without dragging in ``pycocotools`` -- which needs a compiler
on Windows and is the single most common reason an evaluation script will not
run on someone else's laptop. The uncompressed RLE form ("counts" as a list of
integers) is part of the COCO spec, so what this module writes can be read by
pycocotools, and what pycocotools writes in that form can be read here.

Column-major (Fortran) order and a leading run of zeros, exactly as COCO
specifies.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["encode_rle", "decode_rle", "is_rle"]


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean H x W mask as uncompressed COCO RLE."""
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {binary.shape}.")
    height, width = binary.shape

    flat = binary.reshape(-1, order="F").astype(np.uint8)
    if flat.size == 0:
        return {"size": [height, width], "counts": []}

    # Boundaries between runs, plus the implicit ones at each end.
    changes = np.flatnonzero(np.diff(flat)) + 1
    edges = np.concatenate(([0], changes, [flat.size]))
    lengths = np.diff(edges).astype(int)

    counts = lengths.tolist()
    # COCO's first run is always the run of zeros; prepend an empty one when the
    # mask starts with foreground.
    if flat[0] == 1:
        counts = [0] + counts

    return {"size": [int(height), int(width)], "counts": [int(c) for c in counts]}


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode uncompressed COCO RLE back into a boolean H x W mask."""
    size = rle.get("size")
    counts = rle.get("counts")
    if size is None or counts is None:
        raise ValueError("RLE needs both 'size' and 'counts'.")
    if isinstance(counts, (str, bytes)):
        raise ValueError(
            "This is compressed RLE (counts is a string). Decode it with pycocotools "
            "and pass the resulting mask, or export uncompressed RLE."
        )

    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=bool)

    position = 0
    value = False
    for run in counts:
        run = int(run)
        if run < 0:
            raise ValueError("RLE counts must be non-negative.")
        end = position + run
        if end > flat.size:
            raise ValueError(
                f"RLE counts overrun the declared size: {end} > {flat.size} for {height}x{width}."
            )
        if value and run:
            flat[position:end] = True
        position = end
        value = not value

    if position != flat.size:
        raise ValueError(
            f"RLE counts cover {position} of {flat.size} pixels for a {height}x{width} mask."
        )

    return flat.reshape((height, width), order="F")


def is_rle(value: Any) -> bool:
    """True when *value* looks like a COCO RLE dict."""
    return (
        isinstance(value, Mapping)
        and "counts" in value
        and "size" in value
        and not isinstance(value.get("size"), (str, bytes))
        and isinstance(value.get("size"), Sequence)
    )
