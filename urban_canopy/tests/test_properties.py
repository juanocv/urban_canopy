"""Property tests for scientific invariants that examples alone cannot cover."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from urban_canopy.evaluation.rle import decode_rle, encode_rle
from urban_canopy.io.geo import Coordinate, bearing, destination, haversine
from urban_canopy.processing.aggregate import aggregate_values
from urban_canopy.processing.coverage import TREE_SOURCE_CLASS, compute_coverage, coverage_from_mask


@st.composite
def boolean_masks(draw, *, max_side=48):
    shape = (
        draw(st.integers(min_value=1, max_value=max_side)),
        draw(st.integers(min_value=1, max_value=max_side)),
    )
    return draw(arrays(np.bool_, shape, elements=st.booleans()))


@st.composite
def equal_shape_mask_pairs(draw, *, max_side=48):
    shape = (
        draw(st.integers(min_value=1, max_value=max_side)),
        draw(st.integers(min_value=1, max_value=max_side)),
    )
    return (
        draw(arrays(np.bool_, shape, elements=st.booleans())),
        draw(arrays(np.bool_, shape, elements=st.booleans())),
    )


@given(boolean_masks())
def test_uncompressed_rle_round_trip_preserves_every_pixel(mask):
    encoded = encode_rle(mask)
    decoded = decode_rle(encoded)

    assert decoded.dtype == np.bool_
    assert decoded.shape == mask.shape
    np.testing.assert_array_equal(decoded, mask)
    assert sum(encoded["counts"]) == mask.size


@given(boolean_masks(), boolean_masks())
def test_coverage_rejects_incompatible_shapes_instead_of_broadcasting(mask, valid):
    assume(mask.shape != valid.shape)
    with pytest.raises(ValueError, match="shape"):
        coverage_from_mask(mask, valid)


@given(equal_shape_mask_pairs())
def test_coverage_is_the_exact_valid_pixel_fraction(pair):
    tree, valid = pair
    assume(valid.any())

    metrics = compute_coverage(
        tree_mask=tree,
        vegetation_mask=None,
        valid_mask=valid,
        tree_source=TREE_SOURCE_CLASS,
    )
    expected_pixels = int(np.count_nonzero(tree & valid))
    expected_ratio = expected_pixels / int(np.count_nonzero(valid))

    assert metrics.tree_pixels == expected_pixels
    assert metrics.tree_coverage_ratio == pytest.approx(expected_ratio)
    assert metrics.tree_coverage_pct == pytest.approx(100.0 * expected_ratio)
    assert 0.0 <= metrics.tree_coverage_ratio <= 1.0


finite_or_missing = st.one_of(
    st.none(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@given(st.lists(finite_or_missing, min_size=1, max_size=50), st.integers(0, 20))
def test_aggregation_obeys_order_and_count_invariants(values, extra_planned):
    result = aggregate_values(values, n_views=len(values) + extra_planned)
    finite = [float(value) for value in values if value is not None]

    assert result.n_views == len(values) + extra_planned
    assert result.n_valid_views == len(finite)
    if not finite:
        assert result.mean is None
        return

    assert result.minimum <= result.p25 <= result.median <= result.p75 <= result.maximum
    assert (
        result.minimum <= result.mean <= result.maximum
        or np.isclose(result.mean, result.minimum)
        or np.isclose(result.mean, result.maximum)
    )
    assert result.iqr >= 0.0


@pytest.mark.parametrize(
    "corrupt",
    [
        {"size": [2], "counts": [4]},
        {"size": [-1, 2], "counts": []},
        {"size": [2, 2], "counts": [5]},
        {"size": [2, 2], "counts": [1, -1, 4]},
        {"size": [2.5, 2], "counts": [4]},
        {"size": [2, 2], "counts": [1.5, 2.5]},
        {"size": [2, 2], "counts": [True, 3]},
        {"size": [2, 2], "counts": {"not": "a sequence"}},
        {"size": [2, 2], "counts": "P"},  # compressed run truncated mid-value
    ],
)
def test_corrupt_rle_is_rejected_with_a_domain_error(corrupt):
    with pytest.raises(ValueError, match="RLE|size|counts"):
        decode_rle(corrupt)


def test_aggregation_rejects_an_impossible_view_count():
    with pytest.raises(ValueError, match="cannot be smaller"):
        aggregate_values([0.1, 0.2], n_views=1)


@given(
    st.floats(min_value=-80, max_value=80, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=100_000, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=360, exclude_max=True, allow_nan=False),
)
def test_geographic_destination_preserves_requested_spherical_distance(
    lat, lon, distance_m, heading
):
    origin = Coordinate(lat, lon)
    reached = destination(origin, distance_m, heading)
    assert haversine(origin, reached) == pytest.approx(distance_m, abs=1e-6)
    assert origin | reached == pytest.approx(distance_m, abs=1e-6)


def test_bearing_uses_clockwise_degrees_from_north():
    origin = Coordinate(0.0, 0.0)
    assert bearing(origin, Coordinate(1.0, 0.0)) == pytest.approx(0.0)
    assert bearing(origin, Coordinate(0.0, 1.0)) == pytest.approx(90.0)
