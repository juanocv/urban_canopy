"""
The prediction interchange file.

Analysis and evaluation are separate steps on purpose: inference needs a GPU, a
model download and possibly a paid API, while evaluation needs none of those and
gets re-run every time a threshold or a matching rule is questioned. The bridge
between them is one self-contained JSON holding, per image, the coverage number,
the refined tree mask as uncompressed RLE, and the instances when the backend
produced any -- plus the run manifest, so a report can always name the model and
configuration behind the numbers it summarises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .rle import decode_rle, encode_rle

__all__ = [
    "PREDICTIONS_SCHEMA",
    "PredictionRecord",
    "PredictionsFile",
    "build_predictions",
    "write_predictions",
    "load_predictions",
]

PREDICTIONS_SCHEMA = "urban_canopy/predictions/1"


@dataclass(slots=True)
class PredictionRecord:
    """One image's prediction."""

    file_name: str
    height: int
    width: int
    tree_coverage_ratio: float | None
    tree_coverage_pct: float | None
    tree_source: str
    valid_pixels: int
    total_pixels: int
    exclude_bottom_px: int = 0
    mask: dict[str, Any] | None = None
    instances: list[dict[str, Any]] | None = None
    instance_source: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    backend: str | None = None
    class_space: str | None = None

    def tree_mask(self) -> np.ndarray:
        """Decoded refined tree mask; all-false when the record carries none."""
        if self.mask is None:
            return np.zeros((self.height, self.width), dtype=bool)
        return decode_rle(self.mask)

    def instance_masks(self) -> list[np.ndarray]:
        return [decode_rle(inst["mask"]) for inst in (self.instances or [])]

    def instance_scores(self) -> list[float | None]:
        return [inst.get("score") for inst in (self.instances or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
            "tree_coverage_ratio": self.tree_coverage_ratio,
            "tree_coverage_pct": self.tree_coverage_pct,
            "tree_source": self.tree_source,
            "valid_pixels": self.valid_pixels,
            "total_pixels": self.total_pixels,
            "exclude_bottom_px": self.exclude_bottom_px,
            "mask": self.mask,
            "instances": self.instances,
            "instance_source": self.instance_source,
            "quality_flags": list(self.quality_flags),
            "backend": self.backend,
            "class_space": self.class_space,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        return cls(
            file_name=str(data["file_name"]),
            height=int(data["height"]),
            width=int(data["width"]),
            tree_coverage_ratio=data.get("tree_coverage_ratio"),
            tree_coverage_pct=data.get("tree_coverage_pct"),
            tree_source=str(data.get("tree_source", "tree_class")),
            valid_pixels=int(data.get("valid_pixels", 0)),
            total_pixels=int(data.get("total_pixels", 0)),
            exclude_bottom_px=int(data.get("exclude_bottom_px", 0)),
            mask=data.get("mask"),
            instances=data.get("instances"),
            instance_source=data.get("instance_source"),
            quality_flags=list(data.get("quality_flags", [])),
            backend=data.get("backend"),
            class_space=data.get("class_space"),
        )


@dataclass(slots=True)
class PredictionsFile:
    """A loaded predictions document."""

    records: list[PredictionRecord]
    manifest: dict[str, Any] = field(default_factory=dict)
    schema: str = PREDICTIONS_SCHEMA

    @property
    def by_file_name(self) -> dict[str, PredictionRecord]:
        # Keyed on the basename: an annotation export names images without the
        # directory the analysis run happened to read them from.
        return {Path(r.file_name).name: r for r in self.records}


def _record_from_result(result, *, include_mask: bool, include_instances: bool) -> PredictionRecord:
    capture = result.capture
    coverage = result.coverage
    height, width = np.asarray(result.refined_mask).shape[:2]

    instances = None
    if include_instances and result.instances is not None:
        instances = [
            {
                "label": inst.label,
                "score": None if inst.score is None else float(inst.score),
                "source": inst.source,
                "mask": encode_rle(inst.mask),
            }
            for inst in result.instances
        ]

    file_name = capture.image_path or ""
    return PredictionRecord(
        file_name=Path(file_name).name if file_name else "",
        height=int(height),
        width=int(width),
        tree_coverage_ratio=coverage.tree_coverage_ratio,
        tree_coverage_pct=coverage.tree_coverage_pct,
        tree_source=coverage.tree_source,
        valid_pixels=coverage.valid_pixels,
        total_pixels=coverage.total_pixels,
        # Filled in by build_predictions from the run configuration: the
        # evaluator rebuilds the same valid-pixel mask from it.
        exclude_bottom_px=0,
        mask=encode_rle(result.refined_mask) if include_mask else None,
        instances=instances,
        instance_source=result.instance_source,
        quality_flags=list(result.quality_flags),
        backend=result.backend,
        class_space=result.class_space,
    )


def build_predictions(
    results: Sequence,
    *,
    manifest: dict[str, Any] | None = None,
    include_masks: bool = True,
    include_instances: bool = True,
    exclude_bottom_px: int = 0,
) -> dict[str, Any]:
    """Assemble the predictions document from a list of ``ViewResult``."""
    records = []
    for result in results:
        record = _record_from_result(
            result, include_mask=include_masks, include_instances=include_instances
        )
        record.exclude_bottom_px = int(exclude_bottom_px)
        records.append(record)

    return {
        "schema": PREDICTIONS_SCHEMA,
        "manifest": manifest or {},
        "images": [record.to_dict() for record in records],
    }


def write_predictions(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def load_predictions(path: str | Path) -> PredictionsFile:
    """Read a predictions file, rejecting an unknown schema loudly."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    if schema != PREDICTIONS_SCHEMA:
        raise ValueError(
            f"{Path(path).name} declares schema {schema!r}; this build reads "
            f"{PREDICTIONS_SCHEMA!r}."
        )
    return PredictionsFile(
        records=[PredictionRecord.from_dict(item) for item in data.get("images", [])],
        manifest=dict(data.get("manifest", {})),
        schema=schema,
    )
