import numpy as np
import pytest

from urban_canopy.processing.coverage import (
    TREE_SOURCE_CLASS,
    TREE_SOURCE_PROXY,
    TREE_SOURCE_UNAVAILABLE,
    compute_coverage,
    coverage_from_mask,
    resolve_tree_mask,
)


def _valid(shape):
    return np.ones(shape, dtype=bool)


def test_ratio_is_tree_pixels_over_valid_pixels():
    tree = np.zeros((10, 10), bool)
    tree[:5, :] = True  # 50 of 100
    metrics = compute_coverage(
        tree_mask=tree,
        vegetation_mask=None,
        valid_mask=_valid((10, 10)),
        tree_source=TREE_SOURCE_CLASS,
    )
    assert metrics.tree_coverage_ratio == pytest.approx(0.5)
    assert metrics.tree_coverage_pct == pytest.approx(50.0)
    assert metrics.valid_pixels == 100


def test_empty_mask_is_zero_not_none():
    metrics = compute_coverage(
        tree_mask=np.zeros((8, 8), bool),
        vegetation_mask=None,
        valid_mask=_valid((8, 8)),
        tree_source=TREE_SOURCE_CLASS,
    )
    assert metrics.tree_coverage_ratio == 0.0
    assert metrics.tree_pixels == 0


def test_unavailable_tree_is_none_not_zero():
    metrics = compute_coverage(
        tree_mask=None,
        vegetation_mask=np.ones((8, 8), bool),
        valid_mask=_valid((8, 8)),
        tree_source=TREE_SOURCE_UNAVAILABLE,
    )
    assert metrics.tree_coverage_ratio is None
    assert metrics.tree_coverage_pct is None
    assert metrics.vegetation_coverage_ratio == pytest.approx(1.0)


def test_excluded_pixels_leave_numerator_and_denominator():
    # Tree covers the whole frame, but the bottom 2 rows are excluded: the
    # ratio must still be exactly 1.0, not >1 and not diluted.
    tree = np.ones((10, 10), bool)
    valid = np.ones((10, 10), bool)
    valid[-2:, :] = False
    metrics = compute_coverage(
        tree_mask=tree,
        vegetation_mask=None,
        valid_mask=valid,
        tree_source=TREE_SOURCE_CLASS,
    )
    assert metrics.tree_coverage_ratio == pytest.approx(1.0)
    assert metrics.valid_pixels == 80


def test_empty_valid_mask_raises():
    with pytest.raises(ValueError):
        compute_coverage(
            tree_mask=np.zeros((4, 4), bool),
            vegetation_mask=None,
            valid_mask=np.zeros((4, 4), bool),
            tree_source=TREE_SOURCE_CLASS,
        )


def test_group_ratios_keep_grass_separate_from_tree():
    tree = np.zeros((10, 10), bool)
    tree[:2, :] = True
    grass = np.zeros((10, 10), bool)
    grass[5:, :] = True
    metrics = compute_coverage(
        tree_mask=tree,
        vegetation_mask=tree | grass,
        valid_mask=_valid((10, 10)),
        tree_source=TREE_SOURCE_CLASS,
        group_masks={"tree": tree, "grass": grass},
    )
    assert metrics.tree_coverage_ratio == pytest.approx(0.2)
    assert metrics.group_ratios["grass"] == pytest.approx(0.5)
    assert metrics.vegetation_coverage_ratio == pytest.approx(0.7)


def test_coverage_from_mask_matches_compute_coverage():
    mask = np.zeros((6, 6), bool)
    mask[0, :3] = True
    assert coverage_from_mask(mask) == pytest.approx(3 / 36)


class _FakeOutput:
    def __init__(self, taxonomy, group_masks):
        self.taxonomy = taxonomy
        self.group_masks = group_masks

    def group(self, name):
        return self.group_masks.get(name)


def test_resolve_tree_mask_uses_tree_class_when_available():
    from urban_canopy.models.taxonomy import ADE20K

    tree = np.ones((4, 4), bool)
    output = _FakeOutput(ADE20K, {"tree": tree, "grass": np.zeros((4, 4), bool)})
    mask, source = resolve_tree_mask(output)
    assert source == TREE_SOURCE_CLASS
    assert mask.all()


def test_resolve_tree_mask_without_tree_class_needs_explicit_proxy():
    from urban_canopy.models.taxonomy import CITYSCAPES

    veg = np.ones((4, 4), bool)
    output = _FakeOutput(CITYSCAPES, {"vegetation": veg, "terrain": np.zeros((4, 4), bool)})

    mask, source = resolve_tree_mask(output, allow_vegetation_proxy=False)
    assert mask is None
    assert source == TREE_SOURCE_UNAVAILABLE

    mask, source = resolve_tree_mask(output, allow_vegetation_proxy=True)
    assert source == TREE_SOURCE_PROXY
    assert mask.all()
