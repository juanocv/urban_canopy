"""End-to-end evaluation: synthetic predictions file vs synthetic COCO export."""

import json
from dataclasses import replace

import numpy as np
import pytest

from urban_canopy.evaluation.coco import CocoDataset, DatasetValidationError
from urban_canopy.evaluation.predictions import (
    PREDICTIONS_SCHEMA,
    PredictionValidationError,
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
        "mask_status": "available",
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
        "mask_status": "available",
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


def test_unavailable_tree_class_is_not_scored_as_an_empty_prediction(tmp_path):
    predictions = _predictions(tmp_path)
    record = predictions.records[0]
    record.tree_source = "unavailable"
    record.tree_coverage_ratio = None
    record.tree_coverage_pct = None
    record.mask_status = "unavailable"
    record.mask = None

    report = evaluate(predictions, _coco(tmp_path))

    # The other, genuinely empty prediction is still evaluated. The unavailable
    # ADE/Cityscapes-style record is disclosed, not converted into false negatives.
    assert report.semantic["n_images"] == 1
    assert report.semantic_skipped_images == {"a.jpg": "unavailable"}
    assert report.semantic["micro"]["fn"] == 0


def test_duplicate_prediction_basenames_are_rejected(tmp_path):
    predictions = _predictions(tmp_path)
    predictions.records.append(replace(predictions.records[0], file_name="other/a.jpg"))
    with pytest.raises(PredictionValidationError, match="basename 'a.jpg'"):
        predictions.by_file_name


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
        "mask_status": "omitted",
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


def test_legacy_schema_requires_regeneration(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"schema": "urban_canopy/predictions/1", "images": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy"):
        load_predictions(path)


def test_evaluation_refuses_dataset_without_a_tree_category(tmp_path):
    dataset = _coco(tmp_path)
    dataset.tree_category_ids = frozenset()
    with pytest.raises(DatasetValidationError, match="No category"):
        evaluate(_predictions(tmp_path), dataset)


def test_manifest_travels_into_the_report(tmp_path):
    report = evaluate(_predictions(tmp_path), _coco(tmp_path))
    assert report.manifest == {"note": "test"}
