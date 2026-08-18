import numpy as np
import pytest

from urban_canopy.evaluation.instance_metrics import (
    average_precision,
    evaluate_instances,
    mask_iou,
    match_instances,
)


def _blob(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_mask_iou_basic():
    a = _blob((10, 10), 0, 5, 0, 10)  # 50 px
    b = _blob((10, 10), 0, 10, 0, 10)  # 100 px
    assert mask_iou(a, b) == pytest.approx(0.5)


def test_mask_iou_disjoint_and_empty():
    a = _blob((10, 10), 0, 2, 0, 2)
    b = _blob((10, 10), 8, 10, 8, 10)
    assert mask_iou(a, b) == 0.0
    assert mask_iou(np.zeros((5, 5), bool), np.zeros((5, 5), bool)) == 0.0


def test_perfect_match():
    gt = [_blob((20, 20), 0, 10, 0, 10), _blob((20, 20), 12, 20, 12, 20)]
    result = match_instances(gt, gt, iou_threshold=0.5)
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.mean_matched_iou == pytest.approx(1.0)


def test_each_gt_matches_at_most_one_prediction():
    # Two identical predictions of one tree: 1 TP and 1 FP, never 2 TPs.
    gt = [_blob((20, 20), 0, 10, 0, 10)]
    preds = [gt[0].copy(), gt[0].copy()]
    result = match_instances(preds, gt, iou_threshold=0.5, scores=[0.9, 0.8])
    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 0


def test_below_threshold_is_fp_and_fn():
    gt = [_blob((20, 20), 0, 10, 0, 10)]
    preds = [_blob((20, 20), 8, 18, 8, 18)]  # small overlap, IoU < 0.5
    result = match_instances(preds, gt, iou_threshold=0.5)
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1
    assert np.isnan(result.mean_matched_iou)


def test_missed_tree_is_fn():
    gt = [_blob((20, 20), 0, 10, 0, 10), _blob((20, 20), 12, 20, 12, 20)]
    preds = [gt[0].copy()]
    result = match_instances(preds, gt, iou_threshold=0.5)
    assert result.tp == 1
    assert result.fn == 1
    assert result.recall == pytest.approx(0.5)


def test_no_predictions_no_gt():
    result = match_instances([], [], iou_threshold=0.5)
    assert result.tp == result.fp == result.fn == 0
    assert np.isnan(result.precision)
    assert np.isnan(result.recall)


def test_score_ordering_decides_contested_gt():
    gt = [_blob((20, 20), 0, 10, 0, 10)]
    good = gt[0].copy()
    slightly_off = _blob((20, 20), 0, 10, 1, 10)
    # The higher-scored, worse prediction claims the GT first (greedy by score,
    # COCO's protocol) and its IoU still clears the threshold.
    result = match_instances([good, slightly_off], gt, iou_threshold=0.5, scores=[0.4, 0.9])
    assert result.tp == 1
    match = result.matches[0]
    assert match.pred_index == 1
    assert result.ranked_by == "score"


def test_unscored_predictions_rank_by_area():
    gt = [_blob((20, 20), 0, 10, 0, 10)]
    result = match_instances([gt[0].copy()], gt, iou_threshold=0.5, scores=[None])
    assert result.ranked_by == "area"


def test_average_precision_perfect_detector():
    ap = average_precision([([0.9, 0.8], [True, True], 2)])
    assert ap == pytest.approx(1.0)


def test_average_precision_no_gt_is_nan():
    assert np.isnan(average_precision([([0.9], [False], 0)]))


def test_evaluate_instances_pools_counts_across_images():
    shape = (20, 20)
    gt_a = [_blob(shape, 0, 10, 0, 10)]
    gt_b = [_blob(shape, 0, 10, 0, 10), _blob(shape, 12, 20, 12, 20)]
    samples = [
        ("a.jpg", [gt_a[0].copy()], [0.9], gt_a),
        ("b.jpg", [gt_b[0].copy()], [0.8], gt_b),  # misses the second tree
    ]
    report = evaluate_instances(samples, iou_threshold=0.5)
    assert report.tp == 2
    assert report.fp == 0
    assert report.fn == 1
    assert report.recall == pytest.approx(2 / 3)
    assert report.precision == pytest.approx(1.0)
    assert report.ap50 is not None


def test_evaluate_instances_without_scores_skips_ap_with_reason():
    shape = (20, 20)
    gt = [_blob(shape, 0, 10, 0, 10)]
    report = evaluate_instances([("a.jpg", [gt[0].copy()], [None], gt)])
    assert report.ap50 is None
    assert report.ap_unavailable_reason is not None
    assert "score" in report.ap_unavailable_reason
    # The threshold metrics are still defined.
    assert report.recall == pytest.approx(1.0)
