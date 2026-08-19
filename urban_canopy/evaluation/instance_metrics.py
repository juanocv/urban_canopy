"""
Level 2: per-instance metrics for individual trees.

Each prediction is matched to **at most one** ground-truth instance, and each
ground-truth instance accepts at most one prediction. Matching is greedy by
confidence (COCO's rule): predictions are considered in descending score order,
and each takes the unmatched ground truth it overlaps most, provided the IoU
clears the threshold. Predictions with no score -- everything coming out of the
connected-component heuristic -- are ordered by area instead, which is stated in
the report because it is a different, weaker protocol.

Output supports statements of the form:

    "recall of 0.80 for individual trees, and the correctly matched instances
    had a mean IoU of 0.84"

AP50 and AP50:95 are computed only when every prediction carries a score. A
ranking metric over unranked predictions is meaningless, so the report says
"unavailable" and why, rather than inventing an ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from urban_canopy.validation import validate_probability

__all__ = [
    "InstanceMatch",
    "MatchResult",
    "match_instances",
    "average_precision",
    "evaluate_instances",
    "DEFAULT_IOU_THRESHOLD",
    "COCO_IOU_THRESHOLDS",
]

DEFAULT_IOU_THRESHOLD = 0.50
#: The COCO AP50:95 sweep.
COCO_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks; 0.0 when both are empty."""
    x = np.asarray(a).astype(bool)
    y = np.asarray(b).astype(bool)
    if x.shape != y.shape:
        raise ValueError(f"Mask shapes differ: {x.shape} vs {y.shape}.")
    union = int(np.count_nonzero(x | y))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(x & y)) / union


@dataclass(frozen=True, slots=True)
class InstanceMatch:
    """One prediction paired with one ground-truth instance."""

    pred_index: int
    gt_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Instance metrics for one image at one IoU threshold."""

    iou_threshold: float
    n_pred: int
    n_gt: int
    matches: tuple[InstanceMatch, ...] = ()
    ranked_by: str = "score"

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def precision(self) -> float:
        return float("nan") if self.n_pred == 0 else self.tp / self.n_pred

    @property
    def recall(self) -> float:
        return float("nan") if self.n_gt == 0 else self.tp / self.n_gt

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return float("nan") if denominator == 0 else (2 * self.tp) / denominator

    @property
    def mean_matched_iou(self) -> float:
        """Mean IoU over the correctly matched pairs only."""
        if not self.matches:
            return float("nan")
        return float(np.mean([m.iou for m in self.matches]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "ranked_by": self.ranked_by,
        }


def _order(
    preds: Sequence[np.ndarray], scores: Sequence[float | None] | None
) -> tuple[list[int], str]:
    """Prediction order for greedy matching, and the criterion used."""
    if scores is not None and len(scores) != len(preds):
        raise ValueError(
            f"scores length ({len(scores)}) must match instances length ({len(preds)})."
        )
    if scores is not None and all(s is not None and np.isfinite(s) for s in scores):
        numeric_scores = [float(s) for s in scores if s is not None]
        order = sorted(range(len(preds)), key=lambda i: -numeric_scores[i])
        return order, "score"
    order = sorted(range(len(preds)), key=lambda i: -int(np.count_nonzero(preds[i])))
    return order, "area"


def match_instances(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    scores: Sequence[float | None] | None = None,
) -> MatchResult:
    """
    Greedy one-to-one matching between predicted and annotated instances.

    Every prediction is matched to at most one ground truth and vice versa, so
    two predictions covering the same tree yield one TP and one FP -- never two
    TPs.
    """
    iou_threshold = validate_probability(iou_threshold, name="iou_threshold")
    order, ranked_by = _order(preds, scores)

    taken: set[int] = set()
    matches: list[InstanceMatch] = []

    for pred_index in order:
        best_gt = -1
        best_iou = 0.0
        for gt_index in range(len(gts)):
            if gt_index in taken:
                continue
            iou = mask_iou(preds[pred_index], gts[gt_index])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt >= 0 and best_iou >= iou_threshold:
            taken.add(best_gt)
            matches.append(InstanceMatch(pred_index=pred_index, gt_index=best_gt, iou=best_iou))

    # Sorted by prediction index so the artifact overlays and the report agree.
    matches.sort(key=lambda m: m.pred_index)
    return MatchResult(
        iou_threshold=float(iou_threshold),
        n_pred=len(preds),
        n_gt=len(gts),
        matches=tuple(matches),
        ranked_by=ranked_by,
    )


def average_precision(
    per_image: Sequence[tuple[Sequence[float], Sequence[bool], int]],
) -> float:
    """
    COCO-style AP from ``(scores, is_tp, n_gt)`` triples, one per image.

    Uses the 101-point interpolated precision-recall curve. Returns ``nan`` when
    the dataset contains no ground-truth instances.
    """
    total_gt = sum(int(n) for _, _, n in per_image)
    if total_gt == 0:
        return float("nan")

    scores: list[float] = []
    flags: list[bool] = []
    for image_scores, image_flags, _ in per_image:
        scores.extend(float(s) for s in image_scores)
        flags.extend(bool(f) for f in image_flags)

    if not scores:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    sorted_flags = np.asarray(flags, dtype=bool)[order]

    tp_cumulative = np.cumsum(sorted_flags)
    fp_cumulative = np.cumsum(~sorted_flags)
    recalls = tp_cumulative / total_gt
    precisions = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)

    # Make precision monotonically non-increasing from the right, then read it
    # off at 101 evenly spaced recall levels.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, 101)
    indices = np.searchsorted(recalls, levels, side="left")
    sampled = np.where(
        indices < precisions.size, precisions[np.minimum(indices, precisions.size - 1)], 0.0
    )
    return float(sampled.mean())


@dataclass(frozen=True, slots=True)
class InstanceReport:
    """Dataset-level instance metrics."""

    iou_threshold: float
    n_images: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    ranked_by: str
    ap50: float | None = None
    ap50_95: float | None = None
    ap_unavailable_reason: str | None = None
    per_image: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "n_images": self.n_images,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "ranked_by": self.ranked_by,
            "AP50": self.ap50,
            "AP50:95": self.ap50_95,
            "ap_unavailable_reason": self.ap_unavailable_reason,
            "per_image": self.per_image,
        }


def evaluate_instances(
    samples: Sequence[
        tuple[str, Sequence[np.ndarray], Sequence[float | None] | None, Sequence[np.ndarray]]
    ],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    compute_ap: bool = True,
) -> InstanceReport:
    """
    Evaluate ``(name, pred_masks, pred_scores, gt_masks)`` samples over a dataset.

    Counts are pooled across images before the ratios are computed, so a single
    image with many trees does not weigh the same as one with a single tree.
    """
    results: list[MatchResult] = []
    rows: list[dict[str, Any]] = []
    ap_records: list[tuple[list[float], list[bool], int]] = []
    scores_complete = True

    for name, preds, scores, gts in samples:
        result = match_instances(preds, gts, iou_threshold=iou_threshold, scores=scores)
        results.append(result)
        rows.append({"image": name, **result.to_dict()})

        if scores is None or any(s is None or not np.isfinite(s) for s in scores):
            scores_complete = False
        else:
            matched = {m.pred_index for m in result.matches}
            numeric_scores = [float(s) for s in scores if s is not None]
            ap_records.append(
                (
                    numeric_scores,
                    [index in matched for index in range(len(preds))],
                    len(gts),
                )
            )

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    matched_ious = [m.iou for r in results for m in r.matches]

    ap50: float | None = None
    ap50_95: float | None = None
    reason: str | None = None
    if not compute_ap:
        reason = "AP computation was disabled for this run."
    elif not scores_complete:
        reason = (
            "At least one prediction carries no confidence score, so no ranking "
            "exists over the predictions and AP is undefined. This is expected for "
            "the connected-component heuristic."
        )
    else:
        ap50 = average_precision(ap_records)
        sweep = []
        for threshold in COCO_IOU_THRESHOLDS:
            records: list[tuple[list[float], list[bool], int]] = []
            for _name, preds, scores, gts in samples:
                result = match_instances(preds, gts, iou_threshold=threshold, scores=scores)
                matched = {m.pred_index for m in result.matches}
                numeric_scores = [float(s) for s in (scores or []) if s is not None]
                records.append(
                    (
                        numeric_scores,
                        [index in matched for index in range(len(preds))],
                        len(gts),
                    )
                )
            sweep.append(average_precision(records))
        finite = [v for v in sweep if np.isfinite(v)]
        ap50_95 = float(np.mean(finite)) if finite else float("nan")

    ranking_modes = {result.ranked_by for result in results}
    ranked_by = (
        next(iter(ranking_modes))
        if len(ranking_modes) == 1
        else ("mixed" if ranking_modes else "score")
    )
    denominator = 2 * tp + fp + fn
    return InstanceReport(
        iou_threshold=float(iou_threshold),
        n_images=len(results),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float("nan") if (tp + fp) == 0 else tp / (tp + fp),
        recall=float("nan") if (tp + fn) == 0 else tp / (tp + fn),
        f1=float("nan") if denominator == 0 else (2 * tp) / denominator,
        mean_matched_iou=float(np.mean(matched_ious)) if matched_ious else float("nan"),
        ranked_by=ranked_by,
        ap50=ap50,
        ap50_95=ap50_95,
        ap_unavailable_reason=reason,
        per_image=rows,
    )
