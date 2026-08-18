import numpy as np
import pytest

from urban_canopy.processing.aggregate import (
    NO_CROSS_VIEW_ASSOCIATION_NOTE,
    aggregate_values,
    aggregate_views,
)


def test_aggregate_values_robust_stats():
    stats = aggregate_values([0.1, 0.2, 0.3, 0.4])
    assert stats.n_views == 4
    assert stats.n_valid_views == 4
    assert stats.mean == pytest.approx(0.25)
    assert stats.median == pytest.approx(0.25)
    assert stats.p25 == pytest.approx(0.175)
    assert stats.p75 == pytest.approx(0.325)
    assert stats.iqr == pytest.approx(0.15)


def test_aggregate_values_ignores_none_but_counts_views():
    stats = aggregate_values([0.2, None, 0.4, None])
    assert stats.n_views == 4
    assert stats.n_valid_views == 2
    assert stats.mean == pytest.approx(0.3)


def test_aggregate_values_all_none():
    stats = aggregate_values([None, None])
    assert stats.n_views == 2
    assert stats.n_valid_views == 0
    assert stats.mean is None
    assert stats.median is None


def test_aggregate_values_planned_views_beyond_delivered():
    stats = aggregate_values([0.5], n_views=8)
    assert stats.n_views == 8
    assert stats.n_valid_views == 1


class _Coverage:
    def __init__(self, tree, veg):
        self.tree_coverage_ratio = tree
        self.vegetation_coverage_ratio = veg


class _Capture:
    def __init__(self, heading):
        self.heading = heading


class _View:
    def __init__(self, tree, veg=None, heading=0, instances=None, flags=()):
        self.coverage = _Coverage(tree, veg)
        self.capture = _Capture(heading)
        self.instances = instances
        self.instances_supported = instances is not None
        self.instance_source = "model" if instances is not None else None
        self.quality_flags = tuple(flags)


def test_aggregate_views_never_sums_instance_counts():
    views = [
        _View(0.3, heading=0, instances=[object(), object()]),
        _View(0.5, heading=90, instances=[object(), object(), object()]),
    ]
    aggregate = aggregate_views(views)
    # Per-view counts are preserved; there is no "total" field anywhere.
    assert aggregate.instance_counts == (2, 3)
    payload = aggregate.to_dict()
    assert payload["instance_counts_per_view"] == [2, 3]
    assert "total_instances" not in payload
    assert NO_CROSS_VIEW_ASSOCIATION_NOTE in aggregate.notes


def test_aggregate_views_mixed_instance_support():
    views = [_View(0.3, instances=None), _View(0.5, instances=[object()])]
    aggregate = aggregate_views(views)
    assert aggregate.instance_counts == (None, 1)


def test_aggregate_views_collects_quality_flags_and_headings():
    views = [
        _View(0.2, heading=0, flags=("empty_tree_mask",)),
        _View(None, veg=0.4, heading=90, flags=("tree_coverage_unavailable",)),
    ]
    aggregate = aggregate_views(views)
    assert aggregate.headings == (0, 90)
    assert set(aggregate.quality_flags) == {"empty_tree_mask", "tree_coverage_unavailable"}
    assert aggregate.tree_coverage.n_views == 2
    assert aggregate.tree_coverage.n_valid_views == 1
    assert aggregate.vegetation_coverage.n_valid_views == 1


def test_aggregate_views_n_planned_counts_failures():
    aggregate = aggregate_views([_View(0.2)], n_planned=4)
    assert aggregate.tree_coverage.n_views == 4
    assert aggregate.tree_coverage.n_valid_views == 1


def test_percent_projection_in_to_dict():
    aggregate = aggregate_views([_View(0.25), _View(0.75)])
    payload = aggregate.to_dict()
    assert payload["tree_coverage_pct"]["median"] == pytest.approx(50.0)
    assert payload["tree_coverage"]["median"] == pytest.approx(0.5)
    assert np.isfinite(payload["tree_coverage"]["iqr"])
