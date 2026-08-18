import pytest

from urban_canopy.core.viewplan import ViewPlanConfig, plan_headings


def test_offsets_mode_is_deterministic():
    config = ViewPlanConfig(mode="offsets", reference_heading=30, offsets=(0, 90, 180, 270))
    first = plan_headings(config)
    second = plan_headings(config)
    assert first == second == [30, 120, 210, 300]


def test_offsets_wrap_around():
    config = ViewPlanConfig(mode="offsets", reference_heading=350, offsets=(0, 20, -20))
    assert plan_headings(config) == [350, 10, 330]


def test_equiangular_mode():
    config = ViewPlanConfig(mode="equiangular", reference_heading=0, n_views=4)
    assert plan_headings(config) == [0, 90, 180, 270]


def test_equiangular_from_reference():
    config = ViewPlanConfig(mode="equiangular", reference_heading=45, n_views=3)
    assert plan_headings(config) == [45, 165, 285]


def test_fixed_mode_uses_exact_headings():
    config = ViewPlanConfig(mode="fixed", headings=(15, 195))
    assert plan_headings(config) == [15, 195]


def test_duplicate_headings_are_removed():
    config = ViewPlanConfig(mode="fixed", headings=(0, 360, 90))
    assert plan_headings(config) == [0, 90]


def test_fixed_mode_needs_headings():
    with pytest.raises(ValueError):
        plan_headings(ViewPlanConfig(mode="fixed", headings=()))


def test_offsets_mode_needs_offsets():
    with pytest.raises(ValueError):
        plan_headings(ViewPlanConfig(mode="offsets", offsets=()))


def test_plan_serialises_for_the_manifest():
    config = ViewPlanConfig(mode="offsets", reference_heading=10, offsets=(0, 180))
    payload = config.to_dict()
    assert payload["mode"] == "offsets"
    assert payload["reference_heading"] == 10
    assert payload["offsets"] == [0, 180]
