import numpy as np
import pytest

from urban_canopy.evaluation.semantic import (
    binary_confusion,
    evaluate_semantic,
    macro_average,
    pool,
)


def _mask(shape, rows):
    out = np.zeros(shape, bool)
    out[rows, :] = True
    return out


def test_iou_dice_precision_recall():
    pred = _mask((10, 10), slice(0, 6))  # 60 px
    gt = _mask((10, 10), slice(2, 8))  # 60 px, overlap 40
    confusion = binary_confusion(pred, gt)
    assert confusion.tp == 40
    assert confusion.fp == 20
    assert confusion.fn == 20
    assert confusion.iou == pytest.approx(40 / 80)
    assert confusion.dice == pytest.approx(80 / 120)
    assert confusion.f1 == confusion.dice
    assert confusion.precision == pytest.approx(40 / 60)
    assert confusion.recall == pytest.approx(40 / 60)


def test_empty_pred_and_gt_has_undefined_iou():
    confusion = binary_confusion(np.zeros((5, 5), bool), np.zeros((5, 5), bool))
    assert confusion.is_empty
    assert np.isnan(confusion.iou)
    assert confusion.tn == 25


def test_valid_mask_limits_the_comparison():
    pred = np.ones((10, 10), bool)
    gt = np.ones((10, 10), bool)
    gt[-2:, :] = False  # disagreement only in the excluded strip
    valid = np.ones((10, 10), bool)
    valid[-2:, :] = False
    confusion = binary_confusion(pred, gt, valid)
    assert confusion.fp == 0
    assert confusion.iou == pytest.approx(1.0)


def test_pool_is_micro_average():
    a = binary_confusion(_mask((10, 10), slice(0, 5)), _mask((10, 10), slice(0, 5)))
    b = binary_confusion(np.zeros((10, 10), bool), _mask((10, 10), slice(0, 5)))
    pooled = pool([a, b])
    assert pooled.tp == 50
    assert pooled.fn == 50
    assert pooled.recall == pytest.approx(0.5)


def test_macro_average_skips_undefined_images():
    defined = binary_confusion(_mask((10, 10), slice(0, 5)), _mask((10, 10), slice(0, 5)))
    empty = binary_confusion(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
    macro = macro_average([defined, empty])
    assert macro["iou"] == pytest.approx(1.0)
    assert macro["iou_n_images"] == 1


def test_evaluate_semantic_counts_empty_images():
    shape = (10, 10)
    pairs = [
        ("a.jpg", _mask(shape, slice(0, 5)), _mask(shape, slice(0, 5)), None),
        ("no_trees.jpg", np.zeros(shape, bool), np.zeros(shape, bool), None),
    ]
    report = evaluate_semantic(pairs)
    assert report.n_images == 2
    assert report.n_empty_images == 1
    assert report.micro.iou == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        binary_confusion(np.zeros((5, 5), bool), np.zeros((6, 6), bool))
