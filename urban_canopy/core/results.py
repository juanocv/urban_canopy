"""
Result types.

Every number this project publishes travels with the conditions that produced
it: which coordinates, which heading, which field of view, which backend, which
taxonomy, whether the mask was refined, and which artifacts were written. A
coverage ratio without its heading is not reproducible, and a ratio measured
through a vegetation proxy is not the same quantity as one measured from a tree
class -- so both facts ride along in the result rather than in the operator's
memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from urban_canopy.io.atomic import atomic_write_text
from urban_canopy.processing.aggregate import MultiViewAggregate
from urban_canopy.processing.coverage import CoverageMetrics
from urban_canopy.processing.refinement import RefinementStats
from urban_canopy.validation import (
    validate_fov,
    validate_heading,
    validate_image_size,
    validate_latitude,
    validate_longitude,
    validate_pitch,
)

__all__ = ["CaptureParams", "ViewFailure", "ViewResult", "MultiViewResult", "QualityFlag"]


class QualityFlag:
    """Named reasons a view deserves a second look."""

    EMPTY_TREE_MASK = "empty_tree_mask"
    TREE_UNAVAILABLE = "tree_coverage_unavailable"
    TREE_FROM_PROXY = "tree_from_vegetation_proxy"
    REFINEMENT_DISABLED = "refinement_disabled"
    GROWTH_GUARD = "refinement_growth_guard_triggered"
    HEURISTIC_INSTANCES = "instances_are_heuristic"
    NEAR_TOTAL_COVERAGE = "coverage_above_90pct"


@dataclass(frozen=True, slots=True)
class CaptureParams:
    """Exactly how one frame was obtained."""

    source: str  # "streetview" | "local"
    lat: float | None = None
    lon: float | None = None
    heading: int | None = None
    pitch: int | None = None
    fov: int | None = None
    size: str | None = None
    address: str | None = None
    pano_id: str | None = None
    capture_date: str | None = None
    image_path: str | None = None

    def __post_init__(self) -> None:
        if self.source not in ("streetview", "local"):
            raise ValueError(f"source must be 'streetview' or 'local'; got {self.source!r}.")
        if self.lat is not None:
            validate_latitude(self.lat)
        if self.lon is not None:
            validate_longitude(self.lon)
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must either both be present or both be absent.")
        if self.heading is not None:
            validate_heading(self.heading)
        if self.pitch is not None:
            validate_pitch(self.pitch)
        if self.fov is not None:
            validate_fov(self.fov)
        if self.size is not None:
            validate_image_size(self.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "lat": self.lat,
            "lon": self.lon,
            "heading": self.heading,
            "pitch": self.pitch,
            "fov": self.fov,
            "size": self.size,
            "address": self.address,
            "pano_id": self.pano_id,
            "capture_date": self.capture_date,
            "image_path": self.image_path,
        }


@dataclass(frozen=True, slots=True)
class ViewFailure:
    """One planned heading that failed before producing a usable result."""

    heading: int
    stage: str  # "fetch" | "analysis"
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(slots=True)
class ViewResult:
    """Everything one image produced."""

    coverage: CoverageMetrics
    capture: CaptureParams
    backend: str
    class_space: str
    raw_mask: np.ndarray  # H x W uint8, straight from the segmenter
    refined_mask: np.ndarray  # H x W uint8, equal to raw_mask when refinement is off
    refinement: RefinementStats
    vegetation_mask: np.ndarray | None = None
    rgb_image: np.ndarray | None = None
    instances: list | None = None
    instances_supported: bool = False
    instance_source: str | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    backend_notes: tuple[str, ...] = field(default_factory=tuple)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def instance_count(self) -> int | None:
        return None if self.instances is None else len(self.instances)

    def to_dict(self, *, include_artifacts: bool = True) -> dict[str, Any]:
        """JSON-safe view of the result; arrays stay out of it by design."""
        payload: dict[str, Any] = {
            "backend": self.backend,
            "class_space": self.class_space,
            "capture": self.capture.to_dict(),
            "coverage": self.coverage.to_dict(),
            "refinement": self.refinement.to_dict(),
            "instances": {
                "count": self.instance_count,
                "supported": self.instances_supported,
                "source": self.instance_source,
            },
            "quality_flags": list(self.quality_flags),
            "backend_notes": list(self.backend_notes),
        }
        if include_artifacts and self.artifacts:
            payload["artifacts"] = dict(self.artifacts)
        return payload


@dataclass(slots=True)
class MultiViewResult:
    """A set of views of one location or street segment, plus their aggregate."""

    views: list[ViewResult]
    aggregate: MultiViewAggregate
    lat: float | None = None
    lon: float | None = None
    address: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    failures: list[ViewFailure] = field(default_factory=list)

    def to_dict(self, *, include_artifacts: bool = True) -> dict[str, Any]:
        return {
            "location": {"lat": self.lat, "lon": self.lon, "address": self.address},
            "plan": dict(self.plan),
            "aggregate": self.aggregate.to_dict(),
            "views": [v.to_dict(include_artifacts=include_artifacts) for v in self.views],
            "failures": [failure.to_dict() for failure in self.failures],
        }


def results_to_rows(results: Sequence[ViewResult]) -> list[dict[str, Any]]:
    """Flatten view results into CSV-friendly rows."""
    rows: list[dict[str, Any]] = []
    for result in results:
        capture = result.capture
        coverage = result.coverage
        rows.append(
            {
                "image_path": capture.image_path or "",
                "source": capture.source,
                "lat": capture.lat,
                "lon": capture.lon,
                "heading": capture.heading,
                "pitch": capture.pitch,
                "fov": capture.fov,
                "fov_size": capture.size,
                "pano_id": capture.pano_id,
                "capture_date": capture.capture_date,
                "backend": result.backend,
                "class_space": result.class_space,
                "tree_source": coverage.tree_source,
                "tree_coverage_ratio": coverage.tree_coverage_ratio,
                "tree_coverage_pct": coverage.tree_coverage_pct,
                "vegetation_coverage_ratio": coverage.vegetation_coverage_ratio,
                "vegetation_coverage_pct": coverage.vegetation_coverage_pct,
                "valid_pixels": coverage.valid_pixels,
                "total_pixels": coverage.total_pixels,
                "instance_count": result.instance_count,
                "instance_source": result.instance_source or "",
                "refinement_enabled": result.refinement.enabled,
                "quality_flags": "|".join(result.quality_flags),
            }
        )
    return rows


def write_rows_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write CSV rows with a stable column order."""
    import csv

    target = Path(path)
    if not rows:
        return atomic_write_text(target, "")

    fieldnames = list(rows[0].keys())
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(target, handle.getvalue())
