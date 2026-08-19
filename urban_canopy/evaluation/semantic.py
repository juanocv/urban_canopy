"""
Level 1: pixel-level segmentation metrics.

IoU, Dice/F1, precision and recall for the binary tree mask, per image and
pooled over a dataset.

Empty cases are handled explicitly rather than by convention-by-accident. An
image with no annotated trees where the model also predicts none has an
undefined IoU -- there is no union to divide by -- so the per-image value is
``nan`` and the image still counts in ``n_images``. Pooling (micro-averaging)
adds the confusion counts first and divides once, which is why the dataset-level
numbers stay defined even when many images are empty, and why they are the ones
to report. Macro averages over the finite per-image values are also provided,
with the count of images that contributed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["BinaryConfusion", "binary_confusion", "pool", "macro_average", "SemanticReport"]


def _safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


@dataclass(frozen=True, slots=True)
class BinaryConfusion:
    """Pixel counts for one image, or pooled over many."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __add__(self, other: "BinaryConfusion") -> "BinaryConfusion":
        return BinaryConfusion(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )

    @property
    def iou(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp + self.fn)

    @property
    def dice(self) -> float:
        """Dice coefficient, identical to pixel-level F1."""
        return _safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def f1(self) -> float:
        return self.dice

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def is_empty(self) -> bool:
        """True when neither prediction nor ground truth marks any pixel."""
        return (self.tp + self.fp + self.fn) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "iou": self.iou,
            "dice": self.dice,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }


def binary_confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> BinaryConfusion:
    """
    Confusion counts between a predicted and an annotated mask.

    *valid_mask* is available for generic metric use. The Urban Canopy runner
    passes ``None`` and evaluates the complete image.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"Prediction shape {p.shape} does not match ground truth {g.shape}.")

    if valid_mask is None:
        valid = np.ones_like(p, dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != p.shape:
            raise ValueError(f"Valid mask shape {valid.shape} does not match {p.shape}.")

    p = p & valid
    g = g & valid
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    return BinaryConfusion(tp=tp, fp=fp, fn=fn, tn=tn)


def pool(confusions: Iterable[BinaryConfusion]) -> BinaryConfusion:
    """Micro-average: add the counts, then compute the ratios once."""
    total = BinaryConfusion(0, 0, 0, 0)
    for confusion in confusions:
        total = total + confusion
    return total


def macro_average(confusions: Sequence[BinaryConfusion]) -> dict[str, Any]:
    """Mean of the per-image metrics, ignoring images where they are undefined."""
    keys = ("iou", "dice", "f1", "precision", "recall")
    out: dict[str, Any] = {}
    for key in keys:
        values = [getattr(c, key) for c in confusions]
        finite = [v for v in values if np.isfinite(v)]
        out[key] = float(np.mean(finite)) if finite else float("nan")
        out[f"{key}_n_images"] = len(finite)
    return out


@dataclass(frozen=True, slots=True)
class SemanticReport:
    """Dataset-level pixel metrics."""

    n_images: int
    n_empty_images: int
    micro: BinaryConfusion
    macro: dict[str, Any]
    per_image: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_images_without_trees_in_both": self.n_empty_images,
            "micro": self.micro.to_dict(),
            "macro": self.macro,
            "per_image": self.per_image,
        }


def evaluate_semantic(
    pairs: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]],
) -> SemanticReport:
    """
    Evaluate ``(name, pred, gt, valid)`` tuples.

    Returns micro and macro summaries plus the per-image rows, which is what the
    qualitative pass needs: sort by IoU, open the worst five.
    """
    confusions: list[BinaryConfusion] = []
    rows: list[dict[str, Any]] = []
    empty = 0

    for name, pred, gt, valid in pairs:
        confusion = binary_confusion(pred, gt, valid)
        confusions.append(confusion)
        if confusion.is_empty:
            empty += 1
        rows.append({"image": name, **confusion.to_dict()})

    return SemanticReport(
        n_images=len(confusions),
        n_empty_images=empty,
        micro=pool(confusions),
        macro=macro_average(confusions),
        per_image=rows,
    )
