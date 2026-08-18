import numpy as np

from urban_canopy.models.base import HEURISTIC_INSTANCES
from urban_canopy.processing.instances import instances_from_components


def test_components_become_separate_instances():
    mask = np.zeros((40, 40), bool)
    mask[5:15, 5:15] = True
    mask[25:35, 25:35] = True
    instances = instances_from_components(mask, min_area_px=10)
    assert len(instances) == 2
    assert all(inst.source == HEURISTIC_INSTANCES for inst in instances)
    assert all(inst.score is None for inst in instances)


def test_touching_crowns_merge_into_one_component():
    # The documented limitation, pinned as behaviour: two touching squares are
    # one component, hence one "instance".
    mask = np.zeros((20, 40), bool)
    mask[5:15, 5:20] = True
    mask[5:15, 20:35] = True  # shares the x=20 boundary
    instances = instances_from_components(mask, min_area_px=10)
    assert len(instances) == 1


def test_min_area_filters_specks():
    mask = np.zeros((20, 20), bool)
    mask[0, 0] = True
    mask[5:15, 5:15] = True
    instances = instances_from_components(mask, min_area_px=10)
    assert len(instances) == 1
    assert instances[0].area == 100


def test_empty_mask_yields_no_instances():
    assert instances_from_components(np.zeros((10, 10), bool)) == []


def test_partially_visible_tree_at_border_is_kept():
    # A crown cut by the image edge still counts for coverage and appears as a
    # component; nothing removes border-touching foreground.
    mask = np.zeros((20, 20), bool)
    mask[0:8, 0:8] = True  # touches two borders
    instances = instances_from_components(mask, min_area_px=10)
    assert len(instances) == 1


def test_ordering_is_deterministic_largest_first():
    mask = np.zeros((40, 40), bool)
    mask[0:4, 0:4] = True  # 16 px
    mask[10:30, 10:30] = True  # 400 px
    instances = instances_from_components(mask, min_area_px=1)
    assert [inst.area for inst in instances] == [400, 16]
