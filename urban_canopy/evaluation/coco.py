"""
Ground-truth loading: COCO Instance Segmentation.

The annotation format is fixed by how the labels are produced -- each image is
labelled in Roboflow and exported as COCO Instance Segmentation, one mask per
tree. This module reads that export, checks it, and exposes it two ways, because
the project evaluates two different things from the same file:

* ``instance_masks(image_id)`` -- one mask per annotated tree, for the instance
  metrics;
* ``semantic_mask(image_id)`` -- the union of those masks, for the pixel metrics
  and for the ground-truth coverage ratio.

Deriving the semantic mask from the instances rather than annotating it twice is
deliberate: two separately drawn ground truths would disagree, and then no one
could say which of the two the pixel metrics were measured against.

Polygon, uncompressed-RLE and compressed-RLE segmentations are all decoded here
with no external dependency (see :mod:`urban_canopy.evaluation.rle`), so a
Roboflow export works whichever segmentation form it was produced with.

Roboflow rewrites ``file_name`` to an export-specific hashed name and keeps the
original under ``extra.name``. Matching predictions to annotations uses the
original name when present, so a prediction file produced from the source images
still joins after a re-export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from urban_canopy.log import get_logger

from .rle import decode_rle, is_rle

logger = get_logger(__name__)

__all__ = [
    "CocoImage",
    "CocoAnnotation",
    "CocoDataset",
    "DatasetValidationError",
    "DEFAULT_TREE_CATEGORIES",
]

#: Category names treated as individual trees unless the caller says otherwise.
DEFAULT_TREE_CATEGORIES = ("tree", "arvore", "árvore")


class DatasetValidationError(ValueError):
    """Raised when an annotation file cannot be used as ground truth."""


@dataclass(frozen=True, slots=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    original_file_name: str | None = None

    @property
    def match_name(self) -> str:
        return Path(self.original_file_name or self.file_name).name


@dataclass(frozen=True, slots=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: Any
    iscrowd: int = 0
    area: float | None = None

    def to_mask(self, height: int, width: int) -> np.ndarray:
        """Rasterise this annotation into a boolean mask."""
        seg = self.segmentation

        if is_rle(seg):
            mask = decode_rle(seg)
            if mask.shape != (height, width):
                raise DatasetValidationError(
                    f"Annotation {self.id} has an RLE of shape {mask.shape}, "
                    f"but its image is {height}x{width}."
                )
            return mask

        if isinstance(seg, Mapping):
            # is_rle() already took every well-formed RLE, compressed or not, so
            # reaching here means the dict is missing 'size' or 'counts'.
            raise DatasetValidationError(
                f"Annotation {self.id} has a dict segmentation that is not valid COCO "
                f"RLE (keys: {sorted(seg)}); it needs both 'size' and 'counts'."
            )

        if not isinstance(seg, Sequence) or not len(seg):
            raise DatasetValidationError(f"Annotation {self.id} has an empty segmentation.")

        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in seg:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if points.shape[0] < 3:
                # A two-point "polygon" has no area; Roboflow occasionally emits
                # one from a mis-click. Skipping it beats crashing the run.
                logger.warning("Annotation %s has a polygon with < 3 points; skipped.", self.id)
                continue
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool)


@dataclass(slots=True)
class CocoDataset:
    """A loaded COCO instance-segmentation ground truth."""

    images: dict[int, CocoImage]
    annotations: dict[int, list[CocoAnnotation]]
    categories: dict[int, str]
    tree_category_ids: frozenset[int]
    path: Path | None = None
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        tree_categories: Iterable[str] = DEFAULT_TREE_CATEGORIES,
    ) -> "CocoDataset":
        """Read a COCO JSON export and index it by image."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))

        for key in ("images", "annotations", "categories"):
            if key not in data:
                raise DatasetValidationError(f"{source.name} has no {key!r} array.")

        images: dict[int, CocoImage] = {}
        for raw in data["images"]:
            extra = raw.get("extra") or {}

            original_name = None
            if isinstance(extra, Mapping):
                value = extra.get("name")
                if value:
                    original_name = str(value)

            images[int(raw["id"])] = CocoImage(
                id=int(raw["id"]),
                file_name=str(raw["file_name"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                original_file_name=original_name,
            )

        categories = {int(c["id"]): str(c["name"]) for c in data["categories"]}
        wanted = {name.strip().lower() for name in tree_categories}
        tree_ids = frozenset(cid for cid, name in categories.items() if name.lower() in wanted)

        annotations: dict[int, list[CocoAnnotation]] = {image_id: [] for image_id in images}
        for raw in data["annotations"]:
            image_id = int(raw["image_id"])
            annotations.setdefault(image_id, []).append(
                CocoAnnotation(
                    id=int(raw["id"]),
                    image_id=image_id,
                    category_id=int(raw["category_id"]),
                    segmentation=raw.get("segmentation"),
                    iscrowd=int(raw.get("iscrowd", 0)),
                    area=float(raw["area"]) if raw.get("area") is not None else None,
                )
            )

        return cls(
            images=images,
            annotations=annotations,
            categories=categories,
            tree_category_ids=tree_ids,
            path=source,
            info=dict(data.get("info", {})),
        )

    # ------------------------------------------------------------------ #
    def validate(self, *, strict: bool = False) -> list[str]:
        """
        Basic dataset checks.

        Returns the list of problems found. With ``strict=True`` the first
        problem raises instead -- useful in CI, unhelpful while a dataset is
        still being labelled.
        """
        problems: list[str] = []

        if not self.images:
            problems.append("The dataset has no images.")
        if not self.tree_category_ids:
            problems.append(
                "No category matched the tree category names; "
                f"the file declares {sorted(self.categories.values())}."
            )

        orphans = sorted(set(self.annotations) - set(self.images))
        if orphans:
            problems.append(
                f"Annotations reference {len(orphans)} unknown image ids: {orphans[:5]}"
            )

        for image_id, image in self.images.items():
            if image.width <= 0 or image.height <= 0:
                problems.append(f"Image {image.file_name!r} declares a non-positive size.")
            anns = self.annotations.get(image_id, [])
            if not anns:
                # Legitimate: an image with no trees is a needed negative case.
                logger.debug("Image %s has no annotations (treated as no trees).", image.file_name)
            for ann in anns:
                if ann.category_id not in self.categories:
                    problems.append(
                        f"Annotation {ann.id} uses unknown category id {ann.category_id}."
                    )
                if ann.iscrowd:
                    problems.append(
                        f"Annotation {ann.id} is marked iscrowd; crowd regions have no "
                        "individual instance and are excluded from instance matching."
                    )

        # Checked on the *join* key, not on the raw file_name: with a Roboflow
        # export those differ, and a collision that only shows up in match_name
        # would otherwise pass validation and then abort the evaluation from
        # inside by_file_name, long after `validate-dataset` said it was fine.
        duplicates = [
            name
            for name, count in _counter(image.match_name for image in self.images.values()).items()
            if count > 1
        ]
        if duplicates:
            problems.append(
                f"Several images share the same name after resolving extra.name, "
                f"so predictions cannot be joined unambiguously: {duplicates[:5]}"
            )

        if strict and problems:
            raise DatasetValidationError("; ".join(problems))
        return problems

    # ------------------------------------------------------------------ #
    @property
    def by_file_name(self) -> dict[str, CocoImage]:
        index: dict[str, CocoImage] = {}

        for image in self.images.values():
            name = image.match_name

            if name in index:
                raise DatasetValidationError(
                    f"Multiple annotation images resolve to {name!r}: "
                    f"{index[name].file_name!r} and {image.file_name!r}."
                )

            index[name] = image

        return index

    def tree_annotations(self, image_id: int) -> list[CocoAnnotation]:
        """Non-crowd annotations of a tree category for one image."""
        return [
            ann
            for ann in self.annotations.get(image_id, [])
            if ann.category_id in self.tree_category_ids and not ann.iscrowd
        ]

    def instance_masks(self, image_id: int) -> list[np.ndarray]:
        """One boolean mask per annotated tree."""
        image = self.images[image_id]
        return [ann.to_mask(image.height, image.width) for ann in self.tree_annotations(image_id)]

    def semantic_mask(self, image_id: int) -> np.ndarray:
        """Union of the tree instances: the pixel-level ground truth."""
        image = self.images[image_id]
        out = np.zeros((image.height, image.width), dtype=bool)
        for mask in self.instance_masks(image_id):
            out |= mask
        return out

    def coverage_ratio(self, image_id: int, valid_mask: np.ndarray | None = None) -> float:
        """Ground-truth ``tree_coverage_ratio`` for one image."""
        from urban_canopy.processing.coverage import coverage_from_mask

        return coverage_from_mask(self.semantic_mask(image_id), valid_mask)

    def summary(self) -> dict[str, Any]:
        total = sum(len(self.tree_annotations(i)) for i in self.images)
        empty = sum(1 for i in self.images if not self.tree_annotations(i))
        return {
            "path": str(self.path) if self.path else None,
            "n_images": len(self.images),
            "n_tree_instances": total,
            "n_images_without_trees": empty,
            "categories": dict(sorted(self.categories.items())),
            "tree_category_ids": sorted(self.tree_category_ids),
        }


def _counter(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out
