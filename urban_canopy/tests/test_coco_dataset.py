import json

import numpy as np
import pytest

from urban_canopy.evaluation.coco import CocoDataset, DatasetValidationError
from urban_canopy.evaluation.rle import decode_rle, encode_rle


def _dataset_dict():
    """Two images: one with two trees (one polygon, one RLE), one with none."""
    square = encode_rle(_square_mask())
    return {
        "images": [
            {"id": 1, "file_name": "street_a.jpg", "width": 40, "height": 30},
            {"id": 2, "file_name": "no_trees.jpg", "width": 40, "height": 30},
        ],
        "categories": [
            {"id": 1, "name": "tree"},
            {"id": 2, "name": "shrub"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                # Closed 10x10 square polygon at (5,5).
                "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]],
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "segmentation": square,
                "iscrowd": 0,
            },
            {
                "id": 12,
                "image_id": 1,
                "category_id": 2,  # shrub: not a tree instance
                "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3]],
                "iscrowd": 0,
            },
        ],
    }


def _square_mask():
    mask = np.zeros((30, 40), bool)
    mask[20:28, 25:35] = True
    return mask


def _write(tmp_path, data):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_and_summary(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    summary = dataset.summary()
    assert summary["n_images"] == 2
    assert summary["n_tree_instances"] == 2  # the shrub does not count
    assert summary["n_images_without_trees"] == 1


def test_instance_masks_are_separate(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    masks = dataset.instance_masks(1)
    assert len(masks) == 2
    # Non-overlapping annotations: separate masks, disjoint pixels.
    assert not (masks[0] & masks[1]).any()
    assert masks[0].shape == (30, 40)


def test_semantic_mask_is_union_of_instances(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    semantic = dataset.semantic_mask(1)
    masks = dataset.instance_masks(1)
    assert (semantic == (masks[0] | masks[1])).all()


def test_image_without_trees_has_empty_semantic_mask(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    assert not dataset.semantic_mask(2).any()
    assert dataset.instance_masks(2) == []


def test_coverage_ratio_from_ground_truth(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    ratio = dataset.coverage_ratio(2)
    assert ratio == 0.0


def test_rle_annotation_roundtrip(tmp_path):
    dataset = CocoDataset.load(_write(tmp_path, _dataset_dict()))
    rle_masks = [m for m in dataset.instance_masks(1) if m[24, 30]]
    assert len(rle_masks) == 1
    assert (rle_masks[0] == _square_mask()).all()


def test_validate_flags_missing_tree_category(tmp_path):
    data = _dataset_dict()
    data["categories"] = [{"id": 1, "name": "car"}]
    dataset = CocoDataset.load(_write(tmp_path, data))
    problems = dataset.validate()
    assert any("tree" in p.lower() for p in problems)


def test_validate_flags_orphan_annotations(tmp_path):
    data = _dataset_dict()
    data["annotations"][0]["image_id"] = 999
    dataset = CocoDataset.load(_write(tmp_path, data))
    problems = dataset.validate()
    assert any("unknown image" in p for p in problems)


def test_validate_strict_raises(tmp_path):
    data = _dataset_dict()
    data["categories"] = [{"id": 1, "name": "car"}]
    dataset = CocoDataset.load(_write(tmp_path, data))
    with pytest.raises(DatasetValidationError):
        dataset.validate(strict=True)


def test_missing_top_level_key_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        CocoDataset.load(_write(tmp_path, {"images": [], "annotations": []}))


def test_custom_tree_category_names(tmp_path):
    data = _dataset_dict()
    data["categories"][0]["name"] = "arbol"
    path = _write(tmp_path, data)
    dataset = CocoDataset.load(path, tree_categories=("arbol",))
    assert dataset.summary()["n_tree_instances"] == 2


def test_rle_encode_decode_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((37, 23)) > 0.6
    assert (decode_rle(encode_rle(mask)) == mask).all()


def test_rle_empty_and_full():
    empty = np.zeros((5, 7), bool)
    full = np.ones((5, 7), bool)
    assert not decode_rle(encode_rle(empty)).any()
    assert decode_rle(encode_rle(full)).all()


def test_compressed_rle_is_rejected_with_guidance():
    with pytest.raises(ValueError, match="pycocotools"):
        decode_rle({"size": [5, 5], "counts": "abc"})

def test_roboflow_original_filename_is_used_for_matching(tmp_path):
    data = _dataset_dict()

    data["images"][0]["file_name"] = (
        "street_jpg.rf.QIGwbVMOsqXHSgUMaJPz.jpg"
    )
    data["images"][0]["extra"] = {
        "name": "street.jpg"
    }

    dataset = CocoDataset.load(_write(tmp_path, data))

    assert "street.jpg" in dataset.by_file_name
    assert (
        dataset.by_file_name["street.jpg"].file_name
        == "street_jpg.rf.QIGwbVMOsqXHSgUMaJPz.jpg"
    )