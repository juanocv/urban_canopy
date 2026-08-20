"""
Level 3: how far the published indicator is from the annotated one.

    tree_coverage_pred  vs  tree_coverage_gt

Error metrics first, in percentage points, because they are the ones that bound
what the indicator can be used for: MAE, RMSE and the mean bias (signed, so
systematic over- or under-estimation is visible rather than averaged away).

Pearson correlation is reported too and is deliberately listed last. A model
that predicts exactly twice the true coverage everywhere correlates at 1.0 and
is wrong by a factor of two; correlation cannot substitute for error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CoverageErrorReport", "evaluate_coverage"]


@dataclass(frozen=True, slots=True)
class CoverageErrorReport:
    """Agreement between predicted and annotated coverage."""

    n: int
    mae_pp: float
    rmse_pp: float
    bias_pp: float
    mean_pred_pct: float
    mean_gt_pct: float
    max_abs_error_pp: float
    pearson_r: float | None = None
    per_image: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae_pp": self.mae_pp,
            "rmse_pp": self.rmse_pp,
            "bias_pp": self.bias_pp,
            "max_abs_error_pp": self.max_abs_error_pp,
            "mean_pred_pct": self.mean_pred_pct,
            "mean_gt_pct": self.mean_gt_pct,
            "pearson_r": self.pearson_r,
            "per_image": self.per_image,
        }


def _pearson(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Pearson r, or None when one side has no variance (r is undefined)."""
    if pred.size < 2:
        return None
    if np.isclose(pred.std(), 0.0) or np.isclose(gt.std(), 0.0):
        return None
    return float(np.corrcoef(pred, gt)[0, 1])


def evaluate_coverage(
    samples: Sequence[tuple[str, float, float]],
    *,
    keep_per_image: bool = True,
) -> CoverageErrorReport:
    """
    Evaluate ``(name, predicted_pct, ground_truth_pct)`` triples.

    Both inputs are **percentages**, and every error is reported in percentage
    points, so "MAE 3.2 pp" reads directly against a coverage of, say, 18%.
    """
    if not samples:
        raise ValueError("No coverage samples to evaluate.")

    names = [name for name, _, _ in samples]
    pred = np.asarray([float(p) for _, p, _ in samples], dtype=float)
    gt = np.asarray([float(g) for _, _, g in samples], dtype=float)

    errors = pred - gt
    per_image = None
    if keep_per_image:
        per_image = [
            {
                "image": name,
                "tree_coverage_pred_pct": float(p),
                "tree_coverage_gt_pct": float(g),
                "error_pp": float(p - g),
                "abs_error_pp": float(abs(p - g)),
            }
            for name, p, g in zip(names, pred, gt, strict=True)
        ]

    return CoverageErrorReport(
        n=int(pred.size),
        mae_pp=float(np.mean(np.abs(errors))),
        rmse_pp=float(np.sqrt(np.mean(errors**2))),
        bias_pp=float(np.mean(errors)),
        mean_pred_pct=float(pred.mean()),
        mean_gt_pct=float(gt.mean()),
        max_abs_error_pp=float(np.max(np.abs(errors))),
        pearson_r=_pearson(pred, gt),
        per_image=per_image,
    )
