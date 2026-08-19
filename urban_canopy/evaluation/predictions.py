"""
The prediction interchange file.

Analysis and evaluation are separate steps on purpose: inference needs a GPU, a
model download and possibly a paid API, while evaluation needs none of those and
gets re-run every time a threshold or a matching rule is questioned. The bridge
between them is one self-contained JSON holding, per image, the coverage number,
the refined tree mask as uncompressed RLE -- plus the run manifest, so a report can always name the model and
configuration behind the numbers it summarises. ``mask_status`` distinguishes a
real empty prediction from a tree mask the backend cannot express, and from a
mask intentionally omitted during export.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from urban_canopy.io.atomic import atomic_write_text
from urban_canopy.io.json_io import json_dumps

from .rle import decode_rle, encode_rle

__all__ = [
    "PREDICTIONS_SCHEMA",
    "PredictionValidationError",
    "PredictionRecord",
    "PredictionsFile",
    "build_predictions",
    "write_predictions",
    "load_predictions",
]

PREDICTIONS_SCHEMA = "urban_canopy/predictions/3"

MaskStatus = Literal["available", "unavailable", "omitted"]


class PredictionValidationError(ValueError):
    """Raised when a predictions document would be ambiguous to evaluate."""


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
    mask_status: MaskStatus
    mask: dict[str, Any] | None = None
    quality_flags: list[str] = field(default_factory=list)
    backend: str | None = None
    class_space: str | None = None

    def tree_mask(self) -> np.ndarray:
        """Decoded refined tree mask, only when semantic evaluation is possible."""
        if self.mask_status != "available" or self.mask is None:
            raise PredictionValidationError(
                f"{self.file_name!r} has mask_status={self.mask_status!r}; "
                "no tree mask is available for semantic evaluation."
            )
        mask = decode_rle(self.mask)
        if mask.shape != (self.height, self.width):
            raise PredictionValidationError(
                f"{self.file_name!r} carries a mask of shape {mask.shape}, but declares "
                f"{self.height}x{self.width}."
            )
        return mask

    def validate(self) -> None:
        if not self.file_name or not Path(self.file_name).name:
            raise PredictionValidationError("Every prediction needs a non-empty file_name.")
        if self.height <= 0 or self.width <= 0:
            raise PredictionValidationError(
                f"{self.file_name!r} declares a non-positive image size."
            )
        expected_pixels = self.height * self.width
        if self.total_pixels != expected_pixels or self.valid_pixels != expected_pixels:
            raise PredictionValidationError(
                f"{self.file_name!r} must measure the complete image: expected "
                f"valid_pixels=total_pixels={expected_pixels}, got "
                f"{self.valid_pixels} and {self.total_pixels}."
            )
        if self.mask_status not in ("available", "unavailable", "omitted"):
            raise PredictionValidationError(
                f"{self.file_name!r} has unknown mask_status={self.mask_status!r}."
            )
        if self.mask_status == "available":
            if self.mask is None:
                raise PredictionValidationError(
                    f"{self.file_name!r} says its mask is available but carries none."
                )
            if self.tree_source == "unavailable":
                raise PredictionValidationError(
                    f"{self.file_name!r} cannot carry an available tree mask when "
                    "tree_source='unavailable'."
                )
            self.tree_mask()
        elif self.mask is not None:
            raise PredictionValidationError(
                f"{self.file_name!r} has mask_status={self.mask_status!r} but still "
                "carries mask data."
            )

        if self.mask_status == "unavailable":
            if self.tree_source != "unavailable":
                raise PredictionValidationError(
                    f"{self.file_name!r} marks its mask unavailable but declares "
                    f"tree_source={self.tree_source!r}."
                )
            if self.tree_coverage_ratio is not None or self.tree_coverage_pct is not None:
                raise PredictionValidationError(
                    f"{self.file_name!r} cannot publish tree coverage when the tree "
                    "class is unavailable."
                )
        else:
            if self.tree_coverage_ratio is None or self.tree_coverage_pct is None:
                raise PredictionValidationError(
                    f"{self.file_name!r} has a tree class but carries incomplete "
                    "coverage metrics."
                )
            ratio = float(self.tree_coverage_ratio)
            pct = float(self.tree_coverage_pct)
            if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise PredictionValidationError(
                    f"{self.file_name!r} has invalid tree_coverage_ratio={ratio!r}."
                )
            if not np.isfinite(pct) or not np.isclose(pct, 100.0 * ratio, atol=1e-10):
                raise PredictionValidationError(
                    f"{self.file_name!r} has tree_coverage_pct={pct!r}, inconsistent "
                    f"with tree_coverage_ratio={ratio!r}."
                )

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
            "mask_status": self.mask_status,
            "mask": self.mask,
            "quality_flags": list(self.quality_flags),
            "backend": self.backend,
            "class_space": self.class_space,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        return cls(
            file_name=str(data["file_name"]),
            height=int(data["height"]),
            width=int(data["width"]),
            tree_coverage_ratio=data.get("tree_coverage_ratio"),
            tree_coverage_pct=data.get("tree_coverage_pct"),
            tree_source=str(data.get("tree_source", "tree_class")),
            valid_pixels=int(data.get("valid_pixels", 0)),
            total_pixels=int(data.get("total_pixels", 0)),
            mask_status=cast(MaskStatus, str(data["mask_status"])),
            mask=data.get("mask"),
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
        index: dict[str, PredictionRecord] = {}
        for record in self.records:
            record.validate()
            name = Path(record.file_name).name
            if name in index:
                raise PredictionValidationError(
                    f"Multiple predictions resolve to basename {name!r}: "
                    f"{index[name].file_name!r} and {record.file_name!r}."
                )
            index[name] = record
        return index


def _record_from_result(result, *, include_mask: bool) -> PredictionRecord:
    capture = result.capture
    coverage = result.coverage
    height, width = np.asarray(result.refined_mask).shape[:2]

    file_name = capture.image_path or ""
    tree_available = coverage.tree_source != "unavailable"
    if not tree_available:
        mask_status: MaskStatus = "unavailable"
        mask = None
    elif include_mask:
        mask_status = "available"
        mask = encode_rle(result.refined_mask)
    else:
        mask_status = "omitted"
        mask = None

    return PredictionRecord(
        file_name=Path(file_name).name if file_name else "",
        height=int(height),
        width=int(width),
        tree_coverage_ratio=coverage.tree_coverage_ratio,
        tree_coverage_pct=coverage.tree_coverage_pct,
        tree_source=coverage.tree_source,
        valid_pixels=coverage.valid_pixels,
        total_pixels=coverage.total_pixels,
        mask_status=mask_status,
        mask=mask,
        quality_flags=list(result.quality_flags),
        backend=result.backend,
        class_space=result.class_space,
    )


def build_predictions(
    results: Sequence,
    *,
    manifest: dict[str, Any] | None = None,
    include_masks: bool = True,
) -> dict[str, Any]:
    """Assemble the predictions document from a list of ``ViewResult``."""
    records = []
    for result in results:
        record = _record_from_result(result, include_mask=include_masks)
        record.validate()
        records.append(record)

    return {
        "schema": PREDICTIONS_SCHEMA,
        "manifest": manifest or {},
        "images": [record.to_dict() for record in records],
    }


def write_predictions(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    return atomic_write_text(target, json_dumps(payload))


def load_predictions(path: str | Path) -> PredictionsFile:
    """Read a predictions file, rejecting an unknown schema loudly."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    legacy = {
        "urban_canopy/predictions/1": (
            "could exclude part of the Street View frame from its denominator"
        ),
        "urban_canopy/predictions/2": (
            "carried per-instance predictions, which this build no longer evaluates"
        ),
    }
    if schema in legacy:
        raise ValueError(
            f"{Path(path).name} uses a legacy predictions schema, which "
            f"{legacy[schema]}. Regenerate it with this build to produce "
            f"{PREDICTIONS_SCHEMA!r}."
        )
    if schema != PREDICTIONS_SCHEMA:
        raise ValueError(
            f"{Path(path).name} declares schema {schema!r}; this build reads "
            f"{PREDICTIONS_SCHEMA!r}."
        )
    records: list[PredictionRecord] = []
    for index, item in enumerate(data.get("images", [])):
        try:
            records.append(PredictionRecord.from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionValidationError(
                f"{Path(path).name} has an invalid prediction at images[{index}]: {exc}"
            ) from exc
    predictions = PredictionsFile(
        records=records, manifest=dict(data.get("manifest", {})), schema=schema
    )
    # Force validation at the file boundary rather than much later during a join.
    _ = predictions.by_file_name
    return predictions
