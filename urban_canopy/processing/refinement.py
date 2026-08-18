"""
Conservative refinement of the canopy mask.

None of the sidewalk project's refinement is reused, and not for want of
convenience: ``refine_sidewalk_mask`` and its helpers exist to reconstruct a
continuous ground band from a broken one -- row-wise edge interpolation, RANSAC
curb lines, bridge-filling across occlusions. Applied to a canopy those
operations would invent leaves. Filling the sky between two branches is exactly
the failure mode that inflates a coverage ratio, and it would inflate it most on
the sparse, patchy crowns where the measurement matters.

So this module only ever does four things, all of them local, all optional, and
all off by default except the two cheapest:

1. morphological opening -- removes speckle (off by default);
2. small-component removal -- drops isolated blobs below an area floor;
3. small-hole filling -- closes gaps *strictly smaller* than an area ceiling;
4. morphological closing -- joins near-touching foliage (off by default).

On top of that sits a growth guard: if the operations that can *add* area push
the mask more than ``max_area_growth_frac`` above the raw mask, the result is
rolled back to the removal-only stage and the frame is flagged. The guard is
what keeps a mis-set kernel from quietly becoming a finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from urban_canopy.log import debug_event, get_logger

logger = get_logger(__name__)

__all__ = ["RefinementConfig", "RefinementStats", "refine_canopy_mask"]


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """
    Knobs for :func:`refine_canopy_mask`.

    Defaults are intentionally timid: remove specks, close pinholes, touch
    nothing else. ``enabled=False`` passes the raw mask through untouched, which
    is the comparison baseline every experiment should report against.
    """

    enabled: bool = True
    #: Connected components smaller than this are dropped. Absolute pixels.
    min_component_area_px: int = 64
    #: Alternative area floor as a fraction of the frame; the larger of the two
    #: wins when both are set.
    min_component_area_frac: float | None = None
    #: Holes strictly smaller than this are filled. Absolute pixels.
    max_hole_area_px: int = 64
    #: Square structuring element for the opening step; 0 disables it.
    open_kernel_px: int = 0
    #: Square structuring element for the closing step; 0 disables it.
    close_kernel_px: int = 0
    #: Guard: maximum fractional area growth the additive steps may cause,
    #: relative to the raw mask. Exceeding it rolls back to removal-only.
    max_area_growth_frac: float = 0.05

    def area_floor(self, shape: tuple[int, int]) -> int:
        floor = int(self.min_component_area_px)
        if self.min_component_area_frac is not None:
            floor = max(floor, int(self.min_component_area_frac * shape[0] * shape[1]))
        return floor


@dataclass(frozen=True, slots=True)
class RefinementStats:
    """What refinement actually did, kept for the audit trail."""

    enabled: bool
    area_raw: int
    area_refined: int
    components_removed: int = 0
    holes_filled: int = 0
    area_growth_frac: float = 0.0
    growth_guard_triggered: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_uint8(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool).astype(np.uint8)


def _square(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (int(size), int(size)))


def _remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    if min_area <= 0:
        return mask, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    removed = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
        else:
            removed += 1
    return out, removed


def _fill_small_holes(mask: np.ndarray, max_area: int) -> tuple[np.ndarray, int]:
    """
    Fill background components that are fully enclosed by foreground.

    A hole touching the image border is not enclosed -- it is the outside world
    seen through the crown -- so it is never filled, whatever its size.
    """
    if max_area <= 0:
        return mask, 0

    background = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    height, width = mask.shape

    out = mask.copy()
    filled = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= max_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        right = left + int(stats[label, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[label, cv2.CC_STAT_HEIGHT])
        if left == 0 or top == 0 or right >= width or bottom >= height:
            continue
        out[labels == label] = 1
        filled += 1
    return out, filled


def refine_canopy_mask(
    mask: np.ndarray,
    config: RefinementConfig | None = None,
) -> tuple[np.ndarray, RefinementStats]:
    """
    Refine a binary canopy mask conservatively.

    Returns ``(refined_uint8, stats)``. The input is never modified, so the raw
    mask remains available for the audit artifacts and for the
    refinement-disabled comparison.
    """
    cfg = config or RefinementConfig()
    raw = _as_uint8(mask)
    area_raw = int(raw.sum())

    if not cfg.enabled:
        return raw.copy(), RefinementStats(
            enabled=False,
            area_raw=area_raw,
            area_refined=area_raw,
            config=asdict(cfg),
        )

    work = raw.copy()

    if cfg.open_kernel_px and cfg.open_kernel_px > 1:
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, _square(cfg.open_kernel_px))

    work, removed = _remove_small_components(work, cfg.area_floor(raw.shape))

    # Everything up to here can only shrink the mask, so it is the safe state to
    # fall back to if the additive steps overshoot.
    subtractive = work.copy()

    work, filled = _fill_small_holes(work, cfg.max_hole_area_px)

    if cfg.close_kernel_px and cfg.close_kernel_px > 1:
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, _square(cfg.close_kernel_px))

    area_refined = int(work.sum())
    growth = 0.0 if area_raw == 0 else (area_refined - area_raw) / float(area_raw)

    guard_triggered = False
    if growth > cfg.max_area_growth_frac:
        guard_triggered = True
        work = subtractive
        area_refined = int(work.sum())
        growth = 0.0 if area_raw == 0 else (area_refined - area_raw) / float(area_raw)
        filled = 0
        logger.warning(
            "Canopy refinement grew the mask by more than %.1f%%; rolled back to "
            "component removal only (raw=%d px)",
            100.0 * cfg.max_area_growth_frac,
            area_raw,
        )

    stats = RefinementStats(
        enabled=True,
        area_raw=area_raw,
        area_refined=area_refined,
        components_removed=removed,
        holes_filled=filled,
        area_growth_frac=float(growth),
        growth_guard_triggered=guard_triggered,
        config=asdict(cfg),
    )
    debug_event(logger, "canopy_refinement", stats.to_dict())
    return work, stats
