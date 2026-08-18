"""
COCO run-length encoding, without ``pycocotools``.

Masks have to travel between the analysis step and the evaluation step, and
between machines, without dragging in ``pycocotools`` -- which needs a compiler
on Windows and is the single most common reason an evaluation script will not
run on someone else's laptop.

This module writes the **uncompressed** form ("counts" as a list of integers),
which is part of the COCO spec, so what it produces can be read by pycocotools.
It reads **both** forms: uncompressed, and the compressed string form that
Roboflow and pycocotools emit, decoded here with a port of upstream's
``rleFrString`` (a LEB128-style variable-length encoding of run lengths, where
runs from the third onwards are stored as deltas against the run two positions
back, sign-extended when negative).

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


def _decode_compressed_counts(encoded: str | bytes) -> list[int]:
    """
    Decode COCO's compressed RLE string into integer run lengths.

    Port of upstream ``rleFrString``. Each run is a variable-length group of
    6-bit chunks (bit 0x20 continues the group, bit 0x10 of the final chunk
    means the value is negative and must be sign-extended); from the third run
    onwards the decoded value is a delta against the run two positions back.
    """
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Compressed RLE counts must be ASCII.") from exc

    runs: list[int] = []
    position = 0

    while position < len(encoded):
        value = 0
        shift = 0
        more = True

        while more:
            if position >= len(encoded):
                # Upstream reads a NUL-terminated buffer and cannot run past the
                # end; a truncated string here would otherwise raise IndexError
                # from the middle of the loop with nothing pointing at the file.
                raise ValueError(
                    "Compressed RLE string ends mid-run; it is truncated or not "
                    "COCO compressed RLE."
                )
            chunk = ord(encoded[position]) - 48
            value |= (chunk & 0x1F) << (5 * shift)
            more = bool(chunk & 0x20)

            position += 1
            shift += 1

            if not more and (chunk & 0x10):
                value |= -1 << (5 * shift)

        if len(runs) > 2:
            value += runs[-2]

        runs.append(int(value))

    return runs


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """
    Decode COCO RLE into a boolean H x W mask.

    Accepts both the uncompressed form (``counts`` a list of integers) and the
    compressed string form that pycocotools and Roboflow emit.
    """
    size = rle.get("size")
    counts = rle.get("counts")

    if size is None or counts is None:
        raise ValueError("RLE needs both 'size' and 'counts'.")

    if isinstance(counts, (str, bytes)):
        counts = _decode_compressed_counts(counts)

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
                f"RLE counts overrun the declared size: "
                f"{end} > {flat.size} for {height}x{width}."
            )

        if value and run:
            flat[position:end] = True

        position = end
        value = not value

    if position != flat.size:
        raise ValueError(
            f"RLE counts cover {position} of {flat.size} pixels " f"for a {height}x{width} mask."
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
