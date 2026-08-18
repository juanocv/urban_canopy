"""
Audit artifacts.

Qualitative inspection is not an afterthought here: the whole point of keeping
``raw_mask`` and ``refined_mask`` separate through the pipeline is that both can
be written next to the frame they came from. One directory per view, predictable
filenames, and a JSON with the numbers, so sorting the CSV by
``tree_coverage_pct`` and opening the two extremes is a two-step operation.

Nothing here is required for the metrics; a run with artifacts disabled produces
identical numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from urban_canopy.log import get_logger

logger = get_logger(__name__)

__all__ = ["ArtifactConfig", "write_view_artifacts", "write_json", "artifact_stem"]


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Which artifacts to write for each view."""

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

    Street View frames are named after the parameters that identify them, so two
    runs of the same plan overwrite rather than accumulate near-duplicates.
    """
    capture = result.capture
    if capture.source == "streetview" and capture.lat is not None and capture.lon is not None:
        stem = (
            f"sv_{capture.lat:.6f}_{capture.lon:.6f}"
            f"_h{(capture.heading or 0):03d}"
            f"_p{(capture.pitch or 0):+03d}"
            f"_f{capture.fov or 0}"
        )
    elif capture.image_path:
        stem = Path(capture.image_path).stem
    else:
        stem = "view"
    if index is not None:
        stem = f"{index:03d}_{stem}"
    return stem.replace(" ", "_")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.asarray(mask).astype(bool).astype(np.uint8) * 255))


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
        cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
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
        cv2.imwrite(str(path), mask_overlay_bgr(rgb, result.refined_mask, color=TREE_COLOR_BGR))
        written["overlay_tree"] = str(path)

    if config.save_vegetation_overlay and rgb is not None and result.vegetation_mask is not None:
        path = target / "overlay_vegetation.png"
        cv2.imwrite(
            str(path),
            mask_overlay_bgr(rgb, result.vegetation_mask, color=VEGETATION_COLOR_BGR),
        )
        written["overlay_vegetation"] = str(path)

    if config.save_instances and rgb is not None and result.instances:
        path = target / "instances.png"
        cv2.imwrite(str(path), instances_overlay_bgr(rgb, result.instances))
        written["instances"] = str(path)

    if config.save_metrics_json:
        path = target / "metrics.json"
        payload = result.to_dict(include_artifacts=False)
        payload["artifacts"] = written
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return target
