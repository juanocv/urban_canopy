"""
The project's primary indicator.

    tree_coverage_ratio = tree pixels / valid pixels
    tree_coverage_pct   = 100 * tree_coverage_ratio

Three decisions are deliberate and worth stating, because they are what makes
the number comparable across frames:

* The denominator is *valid* pixels, not ``H * W``. Anything excluded (the
  Street View watermark strip, a letterboxed border) leaves both numerator and
  denominator, so the ratio stays a proper fraction of what was actually looked
  at.
* ``tree`` and ``vegetation`` are computed from separate masks and reported
  separately. Nothing here folds grass or shrubs into the tree number.
* When the backend's class space has no tree class, the ratio is ``None`` -- not
  zero, and not quietly the vegetation number. A caller may opt into the
  vegetation proxy, and then every result carries ``tree_source`` saying so.

No qualitative banding ("low", "medium", "high") is produced anywhere. The
continuous ratio is the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "TREE_SOURCE_CLASS",
    "TREE_SOURCE_PROXY",
    "TREE_SOURCE_UNAVAILABLE",
    "CoverageMetrics",
    "resolve_tree_mask",
    "compute_coverage",
]

#: The backend has a real tree class and it was used.
TREE_SOURCE_CLASS = "tree_class"
#: A wider vegetation class stood in for trees, at the caller's explicit request.
TREE_SOURCE_PROXY = "vegetation_proxy"
#: The class space cannot express trees and no proxy was authorised.
TREE_SOURCE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Per-image visual coverage indicators."""

    valid_pixels: int
    total_pixels: int
    tree_pixels: int | None
    tree_coverage_ratio: float | None
    tree_coverage_pct: float | None
    tree_source: str
    vegetation_pixels: int | None = None
    vegetation_coverage_ratio: float | None = None
    vegetation_coverage_pct: float | None = None
    #: ratio per taxonomy group, so grass/shrub coverage stays inspectable
    #: without ever being added to the tree number.
    group_ratios: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_pixels": self.valid_pixels,
            "total_pixels": self.total_pixels,
            "tree_pixels": self.tree_pixels,
            "tree_coverage_ratio": self.tree_coverage_ratio,
            "tree_coverage_pct": self.tree_coverage_pct,
            "tree_source": self.tree_source,
            "vegetation_pixels": self.vegetation_pixels,
            "vegetation_coverage_ratio": self.vegetation_coverage_ratio,
            "vegetation_coverage_pct": self.vegetation_coverage_pct,
            "group_ratios": dict(self.group_ratios),
        }


def resolve_tree_mask(
    output,
    *,
    allow_vegetation_proxy: bool = False,
) -> tuple[np.ndarray | None, str]:
    """
    Pick the mask that may be called "tree", and say where it came from.

    Returns ``(mask, tree_source)``. The mask is None exactly when
    *tree_source* is :data:`TREE_SOURCE_UNAVAILABLE`.
    """
    taxonomy = output.taxonomy
    if taxonomy.tree_group is not None:
        mask = output.group(taxonomy.tree_group)
        if mask is not None:
            return np.asarray(mask).astype(bool), TREE_SOURCE_CLASS

    if allow_vegetation_proxy and taxonomy.tree_proxy_group is not None:
        proxy = output.group(taxonomy.tree_proxy_group)
        if proxy is not None:
            return np.asarray(proxy).astype(bool), TREE_SOURCE_PROXY

    return None, TREE_SOURCE_UNAVAILABLE


def _ratio(count: int, denominator: int) -> float:
    return float(count) / float(denominator)


def compute_coverage(
    *,
    tree_mask: np.ndarray | None,
    vegetation_mask: np.ndarray | None,
    valid_mask: np.ndarray,
    tree_source: str,
    group_masks: Mapping[str, np.ndarray] | None = None,
) -> CoverageMetrics:
    """
    Turn masks into the coverage indicators.

    Every mask is intersected with *valid_mask* first, so a prediction that
    spills into an excluded region cannot push the ratio above 1.
    """
    valid = np.asarray(valid_mask).astype(bool)
    total_pixels = int(valid.size)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        raise ValueError(
            "The valid-pixel mask is empty, so no coverage ratio is defined. "
            "Check exclude_bottom_px against the image height."
        )

    if tree_mask is None:
        tree_pixels = None
        tree_ratio = None
        tree_pct = None
    else:
        tree = np.asarray(tree_mask).astype(bool) & valid
        tree_pixels = int(np.count_nonzero(tree))
        tree_ratio = _ratio(tree_pixels, valid_pixels)
        tree_pct = 100.0 * tree_ratio

    if vegetation_mask is None:
        veg_pixels = None
        veg_ratio = None
        veg_pct = None
    else:
        veg = np.asarray(vegetation_mask).astype(bool) & valid
        veg_pixels = int(np.count_nonzero(veg))
        veg_ratio = _ratio(veg_pixels, valid_pixels)
        veg_pct = 100.0 * veg_ratio

    ratios: dict[str, float] = {}
    for name, mask in (group_masks or {}).items():
        group = np.asarray(mask).astype(bool) & valid
        ratios[name] = _ratio(int(np.count_nonzero(group)), valid_pixels)

    return CoverageMetrics(
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
        tree_pixels=tree_pixels,
        tree_coverage_ratio=tree_ratio,
        tree_coverage_pct=tree_pct,
        tree_source=tree_source,
        vegetation_pixels=veg_pixels,
        vegetation_coverage_ratio=veg_ratio,
        vegetation_coverage_pct=veg_pct,
        group_ratios=ratios,
    )


def coverage_from_mask(mask: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """
    Bare ratio helper, used by the evaluation code on ground-truth masks.

    Shares the definition above so a predicted and an annotated ratio are never
    computed two different ways.
    """
    m = np.asarray(mask).astype(bool)
    valid = (
        np.ones_like(m, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    )
    denominator = int(np.count_nonzero(valid))
    if denominator == 0:
        raise ValueError("The valid-pixel mask is empty, so no coverage ratio is defined.")
    return _ratio(int(np.count_nonzero(m & valid)), denominator)
