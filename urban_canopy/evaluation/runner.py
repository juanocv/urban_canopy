"""
The evaluation runner: predictions file + COCO annotations -> full report.

Joins the two inputs on image name -- the annotation side resolves Roboflow's
``extra.name`` back to the original filename first -- evaluates the three levels
(pixels, instances, coverage indicator), and returns one JSON-ready document.
Images present in only one of the two inputs are listed, not silently dropped:
a join that quietly shrinks is the classic way an evaluation flatters itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urban_canopy.io.image_io import valid_pixel_mask
from urban_canopy.log import get_logger

from .coco import CocoDataset
from .coverage_error import evaluate_coverage
from .instance_metrics import DEFAULT_IOU_THRESHOLD, evaluate_instances
from .predictions import PredictionsFile
from .semantic import evaluate_semantic

logger = get_logger(__name__)

__all__ = ["EvaluationReport", "evaluate"]


@dataclass(slots=True)
class EvaluationReport:
    """Everything one evaluation run produced."""

    semantic: dict[str, Any]
    coverage: dict[str, Any] | None
    instances: dict[str, Any] | None
    instances_skipped_reason: str | None
    n_matched_images: int
    unmatched_predictions: list[str] = field(default_factory=list)
    unmatched_annotations: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "urban_canopy/evaluation/1",
            "settings": self.settings,
            "n_matched_images": self.n_matched_images,
            "unmatched_predictions": self.unmatched_predictions,
            "unmatched_annotations": self.unmatched_annotations,
            "semantic": self.semantic,
            "coverage": self.coverage,
            "instances": self.instances,
            "instances_skipped_reason": self.instances_skipped_reason,
            "manifest": self.manifest,
        }


def evaluate(
    predictions: PredictionsFile,
    dataset: CocoDataset,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    keep_per_image: bool = True,
) -> EvaluationReport:
    """
    Run all three evaluation levels over the images both inputs share.

    Instance metrics run only over predictions that actually carry instances;
    when none do, that level is skipped with the reason recorded, never
    silently reported as zero.
    """
    pred_by_name = predictions.by_file_name
    gt_by_name = dataset.by_file_name

    shared = sorted(set(pred_by_name) & set(gt_by_name))
    only_pred = sorted(set(pred_by_name) - set(gt_by_name))
    only_gt = sorted(set(gt_by_name) - set(pred_by_name))
    if only_pred:
        logger.warning("%d predicted images have no annotation: %s", len(only_pred), only_pred[:5])
    if only_gt:
        logger.warning("%d annotated images have no prediction: %s", len(only_gt), only_gt[:5])
    if not shared:
        raise ValueError(
            "No image appears in both the predictions file and the annotations. "
            "The join is on file basenames, with Roboflow's extra.name preferred "
            "over the exported file_name on the annotation side; check that both "
            "sides name files the same way."
        )

    semantic_pairs = []
    coverage_samples = []
    instance_samples = []
    instances_present = False

    for name in shared:
        record = pred_by_name[name]
        image = gt_by_name[name]

        gt_mask = dataset.semantic_mask(image.id)
        pred_mask = record.tree_mask()
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"{name}: predicted mask is {pred_mask.shape}, annotation is "
                f"{gt_mask.shape}. Evaluate against the same resolution the "
                "model saw, or re-export the annotations at that size."
            )

        # Rebuild the same denominator the prediction used, so the ground-truth
        # ratio is measured over the same pixels.
        valid = valid_pixel_mask(gt_mask.shape, exclude_bottom_px=record.exclude_bottom_px)

        semantic_pairs.append((name, pred_mask, gt_mask, valid))

        if record.tree_coverage_pct is not None:
            from urban_canopy.processing.coverage import coverage_from_mask

            gt_pct = 100.0 * coverage_from_mask(gt_mask, valid)
            coverage_samples.append((name, float(record.tree_coverage_pct), gt_pct))

        if record.instances is not None:
            instances_present = True
            instance_samples.append(
                (
                    name,
                    record.instance_masks(),
                    record.instance_scores(),
                    dataset.instance_masks(image.id),
                )
            )

    semantic_report = evaluate_semantic(semantic_pairs)

    coverage_report = None
    if coverage_samples:
        coverage_report = evaluate_coverage(
            coverage_samples, keep_per_image=keep_per_image
        ).to_dict()

    instances_report = None
    skipped_reason = None
    if instances_present:
        instances_report = evaluate_instances(
            instance_samples, iou_threshold=iou_threshold
        ).to_dict()
        sources = {pred_by_name[n].instance_source for n, *_ in instance_samples}
        sources.discard(None)
        if sources:
            instances_report["instance_source"] = sorted(sources)
    else:
        skipped_reason = (
            "No prediction carries instances. The selected backend produces "
            "semantic/stuff masks only; instance metrics need an instance model "
            "or the explicitly requested connected-component heuristic."
        )

    semantic_dict = semantic_report.to_dict()
    if not keep_per_image:
        semantic_dict.pop("per_image", None)
        if instances_report:
            instances_report.pop("per_image", None)

    return EvaluationReport(
        semantic=semantic_dict,
        coverage=coverage_report,
        instances=instances_report,
        instances_skipped_reason=skipped_reason,
        n_matched_images=len(shared),
        unmatched_predictions=only_pred,
        unmatched_annotations=only_gt,
        manifest=dict(predictions.manifest),
        settings={
            "iou_threshold": iou_threshold,
            "annotations": str(dataset.path) if dataset.path else None,
            "join_key": "file basename",
        },
    )


def evaluate_files(
    predictions_path: str | Path,
    annotations_path: str | Path,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    keep_per_image: bool = True,
) -> EvaluationReport:
    """Convenience wrapper: load both files, validate, evaluate."""
    from .predictions import load_predictions

    dataset = CocoDataset.load(annotations_path)
    problems = dataset.validate(strict=False)
    for problem in problems:
        logger.warning("Annotation check: %s", problem)

    return evaluate(
        load_predictions(predictions_path),
        dataset,
        iou_threshold=iou_threshold,
        keep_per_image=keep_per_image,
    )
