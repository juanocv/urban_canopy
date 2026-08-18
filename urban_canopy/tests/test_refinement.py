import numpy as np

from urban_canopy.processing.refinement import RefinementConfig, refine_canopy_mask


def test_disabled_refinement_returns_raw_mask():
    raw = np.zeros((20, 20), np.uint8)
    raw[2:5, 2:5] = 1
    refined, stats = refine_canopy_mask(raw, RefinementConfig(enabled=False))
    assert (refined == raw).all()
    assert stats.enabled is False
    assert stats.area_raw == stats.area_refined == 9


def test_small_components_are_removed():
    raw = np.zeros((40, 40), np.uint8)
    raw[5:25, 5:25] = 1  # 400 px, kept
    raw[35, 35] = 1  # 1 px speck, dropped
    refined, stats = refine_canopy_mask(raw, RefinementConfig(min_component_area_px=16))
    assert refined[35, 35] == 0
    assert refined[10, 10] == 1
    assert stats.components_removed == 1


def test_small_enclosed_holes_are_filled():
    raw = np.ones((30, 30), np.uint8)
    raw[10:12, 10:12] = 0  # 4 px hole, enclosed
    refined, stats = refine_canopy_mask(
        raw, RefinementConfig(min_component_area_px=0, max_hole_area_px=16)
    )
    assert refined[10, 10] == 1
    assert stats.holes_filled == 1


def test_large_holes_are_not_filled():
    raw = np.ones((40, 40), np.uint8)
    raw[10:30, 10:30] = 0  # 400 px hole: gaps between crowns, not noise
    refined, _ = refine_canopy_mask(
        raw, RefinementConfig(min_component_area_px=0, max_hole_area_px=64)
    )
    assert refined[20, 20] == 0


def test_border_touching_gap_is_never_filled():
    # Sky seen past the edge of a crown is "outside", however small it is here.
    raw = np.ones((20, 20), np.uint8)
    raw[0:2, 9:11] = 0  # touches the top border
    refined, _ = refine_canopy_mask(
        raw, RefinementConfig(min_component_area_px=0, max_hole_area_px=100)
    )
    assert refined[0, 9] == 0


def test_growth_guard_rolls_back_aggressive_closing():
    # A sparse dotted pattern that a large closing kernel would weld together.
    raw = np.zeros((60, 60), np.uint8)
    for row in range(4, 56, 8):
        for col in range(4, 56, 8):
            raw[row : row + 2, col : col + 2] = 1

    config = RefinementConfig(
        min_component_area_px=0,
        max_hole_area_px=0,
        close_kernel_px=15,
        max_area_growth_frac=0.05,
    )
    refined, stats = refine_canopy_mask(raw, config)
    assert stats.growth_guard_triggered is True
    # Rolled back: no more area than the raw mask.
    assert int(refined.sum()) <= int(raw.sum())


def test_empty_mask_stays_empty():
    refined, stats = refine_canopy_mask(np.zeros((10, 10), np.uint8), RefinementConfig())
    assert not refined.any()
    assert stats.area_raw == 0
    assert stats.area_refined == 0


def test_input_is_not_mutated():
    raw = np.zeros((20, 20), np.uint8)
    raw[0, 0] = 1  # a speck that refinement will remove
    before = raw.copy()
    refine_canopy_mask(raw, RefinementConfig(min_component_area_px=4))
    assert (raw == before).all()
