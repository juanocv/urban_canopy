import numpy as np
import pytest

from urban_canopy.evaluation.coverage_error import evaluate_coverage


def test_perfect_agreement():
    report = evaluate_coverage([("a", 20.0, 20.0), ("b", 35.0, 35.0)])
    assert report.mae_pp == 0.0
    assert report.rmse_pp == 0.0
    assert report.bias_pp == 0.0


def test_mae_rmse_bias():
    # errors: +2, -4  ->  MAE 3, RMSE sqrt(10), bias -1
    report = evaluate_coverage([("a", 12.0, 10.0), ("b", 16.0, 20.0)])
    assert report.mae_pp == pytest.approx(3.0)
    assert report.rmse_pp == pytest.approx(np.sqrt(10.0))
    assert report.bias_pp == pytest.approx(-1.0)
    assert report.max_abs_error_pp == pytest.approx(4.0)


def test_bias_is_signed():
    # Systematic over-estimation must show as positive bias, not cancel out.
    report = evaluate_coverage([("a", 15.0, 10.0), ("b", 25.0, 20.0)])
    assert report.bias_pp == pytest.approx(5.0)


def test_correlation_perfect_but_biased_model():
    # pred = 2 * gt: r == 1.0 while the errors are large. Both must be visible.
    samples = [("a", 20.0, 10.0), ("b", 40.0, 20.0), ("c", 60.0, 30.0)]
    report = evaluate_coverage(samples)
    assert report.pearson_r == pytest.approx(1.0)
    assert report.mae_pp == pytest.approx(20.0)


def test_correlation_undefined_without_variance():
    report = evaluate_coverage([("a", 10.0, 10.0), ("b", 10.0, 10.0)])
    assert report.pearson_r is None


def test_empty_samples_raise():
    with pytest.raises(ValueError):
        evaluate_coverage([])


def test_per_image_rows():
    report = evaluate_coverage([("a", 12.0, 10.0)])
    assert report.per_image is not None
    row = report.per_image[0]
    assert row["error_pp"] == pytest.approx(2.0)
    assert row["abs_error_pp"] == pytest.approx(2.0)
