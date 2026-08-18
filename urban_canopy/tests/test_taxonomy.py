import json

import pytest

from urban_canopy.models.taxonomy import (
    ADE20K,
    CITYSCAPES,
    COCO_PANOPTIC,
    Taxonomy,
    default_taxonomy,
    load_taxonomy,
    normalise_label,
)


def test_ade20k_separates_tree_from_grass_and_plant():
    assert ADE20K.group_for_label("tree") == "tree"
    assert ADE20K.group_for_label("palm, palm tree") == "tree"
    assert ADE20K.group_for_label("grass") == "grass"
    assert ADE20K.group_for_label("plant") == "plant_shrub"
    assert ADE20K.group_for_label("building") is None
    assert ADE20K.has_tree_class


def test_coco_panoptic_maps_merged_names():
    assert COCO_PANOPTIC.group_for_label("tree-merged") == "tree"
    assert COCO_PANOPTIC.group_for_label("grass-merged") == "grass"
    assert COCO_PANOPTIC.has_tree_class


def test_cityscapes_has_no_tree_class():
    assert CITYSCAPES.tree_group is None
    assert not CITYSCAPES.has_tree_class
    assert CITYSCAPES.group_for_label("vegetation") == "vegetation"
    # terrain is not counted as vegetation coverage
    assert "terrain" not in CITYSCAPES.vegetation_groups
    assert CITYSCAPES.tree_proxy_group == "vegetation"


def test_normalise_label_splits_ade20k_synonyms():
    assert "palm" in normalise_label("palm, palm tree")
    assert "tree" in normalise_label("Tree")


def test_vegetation_union_is_explicit_not_automatic():
    # The union is exactly what the taxonomy declares -- nothing else is folded in.
    assert set(ADE20K.vegetation_groups) == {"tree", "grass", "plant_shrub"}


def test_roundtrip_through_json(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(ADE20K.to_dict()), encoding="utf-8")
    loaded = load_taxonomy(path, class_space="ade20k")
    assert loaded == ADE20K


def test_load_rejects_mismatched_class_space(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(ADE20K.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="class_space"):
        load_taxonomy(path, class_space="cityscapes")


def test_default_taxonomy_unknown_space():
    with pytest.raises(ValueError):
        default_taxonomy("imagenet")


def test_invalid_group_reference_fails_fast():
    with pytest.raises(ValueError):
        Taxonomy(
            class_space="x",
            groups=(),
            tree_group="tree",
            vegetation_groups=(),
        )
