"""
Multi-view aggregation.

Robust statistics over the per-view coverage ratios of one location or street
segment: mean, median, IQR, p25/p75, and the two counts that let a reader judge
them (views attempted, views that produced a usable ratio).

The one thing this module refuses to do is add up instance counts across views.
A tree photographed from four headings is one tree and four detections; summing
them answers no question anyone asked. Counts therefore stay per view, and
``instance_counts`` carries the note explaining why. Cross-view association --
matching the same crown between headings -- is not implemented, and until it is,
no total is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AggregateStats", "MultiViewAggregate", "aggregate_values", "aggregate_views"]

NO_CROSS_VIEW_ASSOCIATION_NOTE = (
    "Instance counts are per view and are never summed: the same tree can appear "
    "in several headings, and no cross-view association is implemented."
)


@dataclass(frozen=True, slots=True)
class AggregateStats:
    """Robust summary of one indicator across views."""

    n_views: int
    n_valid_views: int
    mean: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    iqr: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_views": self.n_views,
            "n_valid_views": self.n_valid_views,
            "mean": self.mean,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
            "iqr": self.iqr,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


def aggregate_values(
    values: Sequence[float | None], *, n_views: int | None = None
) -> AggregateStats:
    """
    Summarise a list of per-view values, ignoring ``None`` and non-finite entries.

    *n_views* defaults to ``len(values)``; pass it explicitly when some views
    failed before producing any value at all, so the "attempted" count stays
    honest.
    """
    total = len(values) if n_views is None else int(n_views)
    if total < len(values):
        raise ValueError(
            f"n_views ({total}) cannot be smaller than the {len(values)} supplied value(s)."
        )
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]

    if not finite:
        return AggregateStats(n_views=total, n_valid_views=0)

    arr = np.asarray(finite, dtype=float)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return AggregateStats(
        n_views=total,
        n_valid_views=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p25=p25,
        p75=p75,
        iqr=p75 - p25,
        # ddof=0: this is the spread of the views actually taken, not an
        # estimate of a population the views were sampled from.
        std=float(arr.std(ddof=0)),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
    )


@dataclass(frozen=True, slots=True)
class MultiViewAggregate:
    """Aggregated indicators for one location or street segment."""

    tree_coverage: AggregateStats
    vegetation_coverage: AggregateStats
    #: One entry per view, in view order. Never summed; see the module docstring.
    instance_counts: tuple[int | None, ...] = ()
    instances_supported: bool = False
    instance_source: str | None = None
    headings: tuple[int | None, ...] = ()
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_coverage": self.tree_coverage.to_dict(),
            "tree_coverage_pct": {
                key: (None if value is None else 100.0 * value)
                for key, value in self.tree_coverage.to_dict().items()
                if key not in ("n_views", "n_valid_views")
            },
            "vegetation_coverage": self.vegetation_coverage.to_dict(),
            "instance_counts_per_view": list(self.instance_counts),
            "instances_supported": self.instances_supported,
            "instance_source": self.instance_source,
            "headings": list(self.headings),
            "quality_flags": list(self.quality_flags),
            "notes": list(self.notes),
        }


def aggregate_views(views: Iterable, *, n_planned: int | None = None) -> MultiViewAggregate:
    """
    Aggregate a sequence of :class:`~urban_canopy.core.results.ViewResult`.

    Views whose tree ratio is unavailable still count towards ``n_views``: a
    heading that produced no usable number is information, and hiding it would
    make a two-of-eight run look like a two-of-two one. *n_planned* extends that
    to headings that never produced a result at all -- a failed download is the
    same kind of missing data as an unusable frame.
    """
    views = list(views)
    total = len(views) if n_planned is None else max(int(n_planned), len(views))

    tree_values = [v.coverage.tree_coverage_ratio for v in views]
    veg_values = [v.coverage.vegetation_coverage_ratio for v in views]

    counts: list[int | None] = []
    supported = False
    sources: set[str] = set()
    for view in views:
        if view.instances is None:
            counts.append(None)
            continue
        counts.append(len(view.instances))
        supported = supported or bool(view.instances_supported)
        if view.instance_source is not None:
            sources.add(view.instance_source)

    source = next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else None)

    flags = sorted({flag for view in views for flag in view.quality_flags})
    notes = [NO_CROSS_VIEW_ASSOCIATION_NOTE] if any(c is not None for c in counts) else []

    return MultiViewAggregate(
        tree_coverage=aggregate_values(tree_values, n_views=total),
        vegetation_coverage=aggregate_values(veg_values, n_views=total),
        instance_counts=tuple(counts),
        instances_supported=supported,
        instance_source=source,
        headings=tuple(view.capture.heading for view in views),
        quality_flags=tuple(flags),
        notes=tuple(notes),
    )
