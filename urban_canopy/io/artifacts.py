"""
Audit artifacts.

Qualitative inspection is not an afterthought here: the whole point of keeping
``raw_mask`` and ``refined_mask`` separate through the pipeline is that both can
be written next to the frame they came from. One directory per view, predictable
filenames, and a JSON with the numbers, so sorting the CSV by
``tree_coverage_pct`` and opening the two extremes is a two-step operation.

Everything one invocation produces lives under a single **run directory**::

    artifacts_out/
      20260818-104512_oneformer/
        run.json                 manifest, aggregate, every view
        views.csv
        predictions.json         only when asked for
        views/
          000_street/
            rgb.png  mask_raw.png  mask_refined.png  overlay_tree.png  metrics.json

The run directory is what makes results comparable. Naming view folders after
the image alone meant that analysing one frame with OneFormer and then with
Detectron2 wrote both into ``artifacts_out/street/``, and the second silently
destroyed the first -- losing exactly the comparison this project exists to
make. A run is identified by timestamp and backend, so runs accumulate instead
of overwriting, sort chronologically, and say what produced them.

Nothing here is required for the metrics; a run with artifacts disabled produces
identical numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from urban_canopy.io.image_io import (
    TREE_COLOR_BGR,
    VEGETATION_COLOR_BGR,
    instances_overlay_bgr,
    mask_overlay_bgr,
)
from urban_canopy.io.atomic import atomic_write_bytes, atomic_write_text
from urban_canopy.io.json_io import json_dumps
from urban_canopy.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "ArtifactConfig",
    "RunLayout",
    "make_run_id",
    "slugify",
    "write_view_artifacts",
    "write_run_artifacts",
    "write_json",
    "artifact_stem",
]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    """Collapse anything filesystem-hostile into single hyphens."""
    return _UNSAFE.sub("-", str(text).strip()).strip("-.") or "run"


def make_run_id(backend: str, *, name: str | None = None, now: datetime | None = None) -> str:
    """
    Identifier for one invocation: ``20260818-104512_oneformer``.

    Timestamp first so runs sort chronologically, backend second so a directory
    listing answers "which model produced this" without opening anything.
    """
    if name:
        return slugify(name)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{slugify(backend)}"


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Where everything one invocation produces is written."""

    root: Path
    views: Path
    run_json: Path
    views_csv: Path
    predictions_json: Path

    @classmethod
    def create(cls, outdir: str | Path, run_id: str) -> "RunLayout":
        """
        Reserve a fresh run directory under *outdir*.

        A run id already taken gets a numeric suffix rather than being written
        into: two runs started within the same second on the same backend are
        rare, and silently merging their artifacts would reintroduce exactly the
        overwrite this layout exists to prevent.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        base = outdir / run_id
        root = base
        attempt = 2
        while True:
            try:
                root.mkdir(exist_ok=False)
                break
            except FileExistsError:
                root = base.with_name(f"{base.name}-{attempt}")
                attempt += 1

        views = root / "views"
        views.mkdir()
        return cls(
            root=root,
            views=views,
            run_json=root / "run.json",
            views_csv=root / "views.csv",
            predictions_json=root / "predictions.json",
        )


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Which artifacts to write for each view, and where the view folders go."""

    outdir: Path
    enabled: bool = True
    save_rgb: bool = True
    save_raw_mask: bool = True
    save_refined_mask: bool = True
    save_overlay: bool = True
    save_vegetation_overlay: bool = False
    save_instances: bool = True
    save_metrics_json: bool = True


def artifact_stem(result, *, index: int | None = None) -> str:
    """
    Stable, filesystem-safe name for one view's artifact directory.

    The index comes first and is always present, so a multi-view run lists in
    acquisition order rather than alphabetically by latitude. Street View frames
    then carry the heading, which is the parameter a reader actually scans for
    when comparing views of one location; the full capture parameters are in the
    view's ``metrics.json``, not crammed into the folder name.
    """
    capture = result.capture
    if capture.source == "streetview" and capture.lat is not None and capture.lon is not None:
        stem = f"sv_{capture.lat:.6f}_{capture.lon:.6f}_h{(capture.heading or 0):03d}"
    elif capture.image_path:
        stem = Path(capture.image_path).stem
    else:
        stem = "view"

    stem = slugify(stem)
    return f"{(index or 0):03d}_{stem}" if index is not None else stem


def _write_mask(path: Path, mask: np.ndarray) -> None:
    _write_image(path, np.asarray(mask).astype(bool).astype(np.uint8) * 255)


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError(f"Unsupported artifact image extension: {suffix!r}.")
    ok, encoded = cv2.imencode(suffix, np.asarray(image))
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode artifact image {path.name!r}.")
    atomic_write_bytes(path, encoded.tobytes())


def write_view_artifacts(
    result,
    config: ArtifactConfig,
    *,
    index: int | None = None,
) -> dict[str, str]:
    """
    Write one view's artifacts and record their paths on the result.

    Returns the mapping of artifact name to path, which is also stored in
    ``result.artifacts`` so the JSON and CSV exports can point at them.
    """
    if not config.enabled:
        return {}

    stem = artifact_stem(result, index=index)
    target = Path(config.outdir) / stem
    target.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    rgb = result.rgb_image

    if config.save_rgb and rgb is not None:
        path = target / "rgb.png"
        _write_image(path, cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
        written["rgb"] = str(path)

    if config.save_raw_mask:
        path = target / "mask_raw.png"
        _write_mask(path, result.raw_mask)
        written["mask_raw"] = str(path)

    if config.save_refined_mask:
        path = target / "mask_refined.png"
        _write_mask(path, result.refined_mask)
        written["mask_refined"] = str(path)

    if config.save_overlay and rgb is not None:
        path = target / "overlay_tree.png"
        _write_image(path, mask_overlay_bgr(rgb, result.refined_mask, color=TREE_COLOR_BGR))
        written["overlay_tree"] = str(path)

    if config.save_vegetation_overlay and rgb is not None and result.vegetation_mask is not None:
        path = target / "overlay_vegetation.png"
        _write_image(
            path,
            mask_overlay_bgr(rgb, result.vegetation_mask, color=VEGETATION_COLOR_BGR),
        )
        written["overlay_vegetation"] = str(path)

    if config.save_instances and rgb is not None and result.instances:
        path = target / "instances.png"
        _write_image(path, instances_overlay_bgr(rgb, result.instances))
        written["instances"] = str(path)

    if config.save_metrics_json:
        path = target / "metrics.json"
        payload = result.to_dict(include_artifacts=False)
        payload["artifacts"] = written
        atomic_write_text(path, json_dumps(payload, indent=2))
        written["metrics_json"] = str(path)

    result.artifacts.update(written)
    return written


def write_run_artifacts(
    results: Sequence,
    config: ArtifactConfig,
) -> list[dict[str, str]]:
    """Write artifacts for a whole run, numbering the views in order."""
    return [write_view_artifacts(r, config, index=i) for i, r in enumerate(results)]


def write_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON document, creating parent directories as needed."""
    target = Path(path)
    return atomic_write_text(target, json_dumps(payload, indent=2))
