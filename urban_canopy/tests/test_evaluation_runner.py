"""End-to-end evaluation: synthetic predictions file vs synthetic COCO export."""

import json

import numpy as np
import pytest

from urban_canopy.evaluation.coco import CocoDataset
from urban_canopy.evaluation.predictions import (
    PREDICTIONS_SCHEMA,
    load_predictions,
    write_predictions,
)
from urban_canopy.evaluation.rle import encode_rle
from urban_canopy.evaluation.runner import evaluate

HEIGHT, WIDTH = 30, 40


def _tree_mask():
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[5:15, 5:15] = True
    return mask


def _coco(tmp_path):
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": WIDTH, "height": HEIGHT},
            {"id": 2, "file_name": "empty.jpg", "width": WIDTH, "height": HEIGHT},
        ],
        "categories": [{"id": 1, "name": "tree"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": encode_rle(_tree_mask()),
                "iscrowd": 0,
            }
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return CocoDataset.load(path)


def _predictions(tmp_path, *, mask=None, with_instances=False, extra_record=None):
    mask = _tree_mask() if mask is None else mask
    ratio = float(mask.sum()) / (HEIGHT * WIDTH)
    record = {
        "file_name": "a.jpg",
        "height": HEIGHT,
        "width": WIDTH,
        "tree_coverage_ratio": ratio,
        "tree_coverage_pct": 100.0 * ratio,
        "tree_source": "tree_class",
        "valid_pixels": HEIGHT * WIDTH,
        "total_pixels": HEIGHT * WIDTH,
        "exclude_bottom_px": 0,
        "mask": encode_rle(mask),
        "instances": (
            [{"label": "tree", "score": 0.9, "source": "model", "mask": encode_rle(mask)}]
            if with_instances
            else None
        ),
        "instance_source": "model" if with_instances else None,
        "quality_flags": [],
        "backend": "stub",
        "class_space": "ade20k",
    }
    empty = {
        "file_name": "empty.jpg",
        "height": HEIGHT,
        "width": WIDTH,
        "tree_coverage_ratio": 0.0,
        "tree_coverage_pct": 0.0,
        "tree_source": "tree_class",
        "valid_pixels": HEIGHT * WIDTH,
        "total_pixels": HEIGHT * WIDTH,
        "exclude_bottom_px": 0,
        "mask": encode_rle(np.zeros((HEIGHT, WIDTH), bool)),
        "instances": [] if with_instances else None,
        "instance_source": "model" if with_instances else None,
        "quality_flags": ["empty_tree_mask"],
        "backend": "stub",
        "class_space": "ade20k",
    }
    images = [record, empty]
    if extra_record:
        images.append(extra_record)
    payload = {"schema": PREDICTIONS_SCHEMA, "manifest": {"note": "test"}, "images": images}
    path = tmp_path / "predictions.json"
    write_predictions(path, payload)
    return load_predictions(path)


def test_perfect_predictions_score_perfectly(tmp_path):
    report = evaluate(_predictions(tmp_path, with_instances=True), _coco(tmp_path))
    assert report.n_matched_images == 2
    assert report.semantic["micro"]["iou"] == pytest.approx(1.0)
    assert report.coverage["mae_pp"] == pytest.approx(0.0)
    assert report.instances["recall"] == pytest.approx(1.0)
    assert report.instances["mean_matched_iou"] == pytest.approx(1.0)
    assert report.instances["AP50"] == pytest.approx(1.0)


def test_shifted_predictions_show_error(tmp_path):
    shifted = np.zeros((HEIGHT, WIDTH), bool)
    shifted[5:15, 10:20] = True  # half-overlap with the GT square
    report = evaluate(_predictions(tmp_path, mask=shifted), _coco(tmp_path))
    micro = report.semantic["micro"]
    assert micro["iou"] == pytest.approx(50 / 150)
    # Same area, so the coverage indicator agrees even though the pixels do not:
    # exactly why level 1 and level 3 are separate metrics.
    assert report.coverage["mae_pp"] == pytest.approx(0.0)


def test_semantic_only_predictions_skip_instances_with_reason(tmp_path):
    report = evaluate(_predictions(tmp_path, with_instances=False), _coco(tmp_path))
    assert report.instances is None
    assert "instance" in report.instances_skipped_reason


def test_unmatched_images_are_listed_not_dropped(tmp_path):
    extra = {
        "file_name": "unlabelled.jpg",
        "height": HEIGHT,
        "width": WIDTH,
        "tree_coverage_ratio": 0.5,
        "tree_coverage_pct": 50.0,
        "tree_source": "tree_class",
        "valid_pixels": HEIGHT * WIDTH,
        "total_pixels": HEIGHT * WIDTH,
        "exclude_bottom_px": 0,
        "mask": None,
        "instances": None,
        "instance_source": None,
        "quality_flags": [],
        "backend": "stub",
        "class_space": "ade20k",
    }
    report = evaluate(_predictions(tmp_path, extra_record=extra), _coco(tmp_path))
    assert report.unmatched_predictions == ["unlabelled.jpg"]
    assert report.n_matched_images == 2


def test_no_shared_images_raises(tmp_path):
    predictions = _predictions(tmp_path)
    dataset = _coco(tmp_path)
    dataset.images = {
        i: type(img)(
            id=img.id, file_name=f"other_{img.file_name}", width=img.width, height=img.height
        )
        for i, img in dataset.images.items()
    }
    with pytest.raises(ValueError, match="basename"):
        evaluate(predictions, dataset)


def test_wrong_schema_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something/else", "images": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_predictions(path)


def test_manifest_travels_into_the_report(tmp_path):
    report = evaluate(_predictions(tmp_path), _coco(tmp_path))
    assert report.manifest == {"note": "test"}
