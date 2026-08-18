"""
Tree instances, and the thing that is not tree instances.

Splitting a semantic canopy mask into connected components is **not** instance
segmentation, and this module does not pretend otherwise. Two crowns that touch
in image space -- the common case on a treed street -- become one component. One
crown occluded by a pole becomes two. Neither error is visible in the output,
which is why every mask produced here is stamped
``source = "connected_components_heuristic"`` and why the pipeline leaves the
heuristic off unless it is asked for by name.

It is offered at all because it is a useful *approximation* to look at while
building an annotation set, and because its failure modes are worth measuring:
run the instance evaluation against it and the gap to a real instance model is a
number rather than an intuition.
"""

from __future__ import annotations

import cv2
import numpy as np

from urban_canopy.models.base import HEURISTIC_INSTANCES, InstanceMask

__all__ = ["instances_from_components", "HEURISTIC_LIMITATIONS"]

HEURISTIC_LIMITATIONS = (
    "Connected components of a semantic mask: touching crowns merge into one "
    "component and occluded crowns split into several. Counts from this path are "
    "not a count of trees."
)


def instances_from_components(
    mask: np.ndarray,
    *,
    min_area_px: int = 64,
    label: str = "tree",
) -> list[InstanceMask]:
    """
    Split a binary mask into connected components, one candidate per component.

    Components smaller than *min_area_px* are dropped. Returned masks carry no
    score -- there is no model confidence behind them, and inventing one would
    make the AP curves meaningless.
    """
    binary = np.asarray(mask).astype(bool).astype(np.uint8)
    if not binary.any():
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[InstanceMask] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < int(min_area_px):
            continue
        out.append(
            InstanceMask(
                label=label,
                mask=(labels == index),
                score=None,
                source=HEURISTIC_INSTANCES,
            )
        )
    # Largest first: stable ordering makes artifacts and reports diffable.
    out.sort(key=lambda inst: inst.area, reverse=True)
    return out
