"""End-to-end evaluation: synthetic predictions file vs synthetic COCO export."""

import json
from dataclasses import replace
from pathlib import Path

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
from urban_canopy.evaluation.runner import evaluate, join_image_names

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
    assert report.instance_eligibility == {
        "n_shared": 2,
        "n_eligible": 0,
        "excluded_images": {
            "a.jpg": "instances_unavailable",
            "empty.jpg": "instances_unavailable",
        },
    }


def test_instance_evaluation_discloses_ineligible_shared_images(tmp_path):
    predictions = _predictions(tmp_path, with_instances=True)
    predictions.records[1].instances = None
    predictions.records[1].instance_source = None

    report = evaluate(predictions, _coco(tmp_path))

    assert report.instances["n_images"] == 1
    assert report.instances["n_shared"] == 2
    assert report.instances["n_eligible"] == 1
    assert report.instances["excluded_images"] == {"empty.jpg": "instances_unavailable"}


def test_instance_evaluation_rejects_mixed_prediction_origins(tmp_path):
    predictions = _predictions(tmp_path, with_instances=True)
    predictions.records[1].instance_source = "connected_components_heuristic"

    with pytest.raises(ValueError, match="cannot mix prediction origins"):
        evaluate(predictions, _coco(tmp_path))


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
        _ = predictions.by_file_name


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


# --------------------------------------------------------------- name join ---
def test_exact_basenames_join_unchanged():
    join = join_image_names(["a.jpg", "b.jpg"], ["b.jpg", "a.jpg"])
    assert join.matches == (("a.jpg", "a.jpg"), ("b.jpg", "b.jpg"))
    assert join.joined_across_extensions == ()
    assert join.unmatched_predictions == ()
    assert join.unmatched_annotations == ()


def test_extension_mismatch_still_joins():
    # Roboflow re-encodes a JPEG frame and exports it as PNG. A strict basename
    # join reported both sides unmatched and evaluated nothing.
    join = join_image_names(["street.jpg"], ["street.png"])
    assert join.matches == (("street.jpg", "street.png"),)
    assert join.joined_across_extensions == (("street.jpg", "street.png"),)
    assert join.unmatched_predictions == ()
    assert join.unmatched_annotations == ()


def test_exact_match_wins_over_the_extension_fallback():
    # frame.jpg and frame.png may be genuinely different images; the exact pair
    # must claim each other before the fallback sees them.
    join = join_image_names(["frame.jpg", "frame.png"], ["frame.png", "frame.jpg"])
    assert dict(join.matches) == {"frame.jpg": "frame.jpg", "frame.png": "frame.png"}
    assert join.joined_across_extensions == ()


def test_leftover_after_exact_match_still_uses_the_fallback():
    join = join_image_names(["frame.jpg", "other.jpg"], ["frame.jpg", "other.tif"])
    assert dict(join.matches) == {"frame.jpg": "frame.jpg", "other.jpg": "other.tif"}
    assert join.joined_across_extensions == (("other.jpg", "other.tif"),)


def test_ambiguous_stem_is_refused_rather_than_guessed():
    # Pairing the wrong one scores an image against another image's ground
    # truth and still produces a plausible-looking number.
    with pytest.raises(ValueError, match="without guessing"):
        join_image_names(["frame.jpg", "frame.bmp"], ["frame.png"])
    with pytest.raises(ValueError, match="without guessing"):
        join_image_names(["frame.jpg"], ["frame.png", "frame.tif"])


def test_unmatched_names_are_reported_on_both_sides():
    join = join_image_names(["a.jpg", "only_pred.jpg"], ["a.jpg", "only_gt.jpg"])
    assert join.unmatched_predictions == ("only_pred.jpg",)
    assert join.unmatched_annotations == ("only_gt.jpg",)


def test_extensions_are_compared_case_sensitively():
    # Not casefolded: on Linux "Frame.jpg" and "frame.jpg" are different files,
    # and inventing a case-insensitive match would collide them.
    join = join_image_names(["Frame.jpg"], ["frame.png"])
    assert join.matches == ()
    assert join.unmatched_predictions == ("Frame.jpg",)


def test_evaluation_joins_across_extensions_end_to_end(tmp_path):
    dataset = _coco(tmp_path)
    # Re-encode the annotation side, exactly as an annotation tool would.
    dataset.images = {
        i: type(img)(
            id=img.id,
            file_name=f"{Path(img.file_name).stem}.png",
            width=img.width,
            height=img.height,
        )
        for i, img in dataset.images.items()
    }

    report = evaluate(_predictions(tmp_path), dataset)

    assert report.n_matched_images == 2
    assert report.semantic["micro"]["iou"] == pytest.approx(1.0)
    assert report.unmatched_predictions == []
    assert report.unmatched_annotations == []
    # The pairing is surfaced, never silent.
    assert ["a.jpg", "a.png"] in report.settings["joined_across_extensions"]
