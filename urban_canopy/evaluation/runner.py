"""
The evaluation runner: predictions file + COCO annotations -> full report.

Joins the two inputs on image name -- the annotation side resolves Roboflow's
``extra.name`` back to the original filename first -- evaluates the three levels
(pixels, instances, coverage indicator), and returns one JSON-ready document.
Images present in only one of the two inputs are listed, not silently dropped:
a join that quietly shrinks is the classic way an evaluation flatters itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urban_canopy.log import get_logger
from urban_canopy.validation import validate_probability

from .coco import CocoDataset
from .coverage_error import evaluate_coverage
from .instance_metrics import DEFAULT_IOU_THRESHOLD, evaluate_instances
from .predictions import PredictionsFile
from .semantic import evaluate_semantic

logger = get_logger(__name__)

__all__ = ["EvaluationReport", "NameJoin", "evaluate", "join_image_names"]


@dataclass(frozen=True, slots=True)
class NameJoin:
    """How prediction names were paired with annotation names."""

    #: ``(prediction name, annotation name)`` pairs, prediction order.
    matches: tuple[tuple[str, str], ...]
    unmatched_predictions: tuple[str, ...]
    unmatched_annotations: tuple[str, ...]
    #: Subset of *matches* paired only after dropping the file extension.
    joined_across_extensions: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_matched": len(self.matches),
            "joined_across_extensions": [list(pair) for pair in self.joined_across_extensions],
        }


def join_image_names(
    prediction_names: Iterable[str],
    annotation_names: Iterable[str],
) -> NameJoin:
    """
    Pair predicted images with annotated ones.

    Exact basenames are matched first. Whatever is left over is then matched on
    the name without its extension, because annotation tools re-encode: Roboflow
    exports a JPEG frame as ``frame.png``, and a strict basename join silently
    reports both sides as unmatched and evaluates nothing.

    Exact-first matters. A dataset may legitimately hold ``frame.jpg`` and
    ``frame.png`` as different images; matching those first means the fallback
    only ever rescues names nothing else claimed. When the fallback is
    ambiguous -- two leftovers on either side sharing a stem -- it refuses
    rather than picking one, because a wrong pairing scores one image's
    prediction against another image's ground truth and looks like a plausible
    result. Extensions are compared case-sensitively, matching the annotation
    files rather than any one filesystem's rules.
    """
    predictions = list(dict.fromkeys(prediction_names))
    annotations = set(annotation_names)

    matches: list[tuple[str, str]] = []
    across_extensions: list[tuple[str, str]] = []

    matched_annotations = {name for name in predictions if name in annotations}
    remaining_predictions = [name for name in predictions if name not in matched_annotations]
    remaining_annotations = sorted(annotations - matched_annotations)

    def by_stem(names: Iterable[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for name in names:
            grouped.setdefault(Path(name).stem, []).append(name)
        return grouped

    prediction_stems = by_stem(remaining_predictions)
    annotation_stems = by_stem(remaining_annotations)

    paired_predictions: set[str] = set()
    paired_annotations: set[str] = set()
    for stem in sorted(set(prediction_stems) & set(annotation_stems)):
        candidates_pred = prediction_stems[stem]
        candidates_gt = annotation_stems[stem]
        if len(candidates_pred) > 1 or len(candidates_gt) > 1:
            raise ValueError(
                f"Cannot join {stem!r} across file extensions without guessing: "
                f"predictions {sorted(candidates_pred)} and annotations "
                f"{sorted(candidates_gt)} all share that name. Rename them so each "
                "prediction has one unambiguous annotation."
            )
        paired_predictions.add(candidates_pred[0])
        paired_annotations.add(candidates_gt[0])
        across_extensions.append((candidates_pred[0], candidates_gt[0]))

    extension_pairs = dict(across_extensions)
    for name in predictions:
        if name in matched_annotations:
            matches.append((name, name))
        elif name in extension_pairs:
            matches.append((name, extension_pairs[name]))

    return NameJoin(
        matches=tuple(matches),
        unmatched_predictions=tuple(
            name
            for name in predictions
            if name not in matched_annotations and name not in paired_predictions
        ),
        unmatched_annotations=tuple(
            name for name in remaining_annotations if name not in paired_annotations
        ),
        joined_across_extensions=tuple(across_extensions),
    )


@dataclass(slots=True)
class EvaluationReport:
    """Everything one evaluation run produced."""

    semantic: dict[str, Any]
    coverage: dict[str, Any] | None
    instances: dict[str, Any] | None
    instances_skipped_reason: str | None
    instance_eligibility: dict[str, Any]
    n_matched_images: int
    semantic_skipped_images: dict[str, str] = field(default_factory=dict)
    unmatched_predictions: list[str] = field(default_factory=list)
    unmatched_annotations: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "urban_canopy/evaluation/2",
            "settings": self.settings,
            "n_matched_images": self.n_matched_images,
            "semantic_skipped_images": self.semantic_skipped_images,
            "unmatched_predictions": self.unmatched_predictions,
            "unmatched_annotations": self.unmatched_annotations,
            "semantic": self.semantic,
            "coverage": self.coverage,
            "instances": self.instances,
            "instances_skipped_reason": self.instances_skipped_reason,
            "instance_eligibility": self.instance_eligibility,
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
    iou_threshold = validate_probability(iou_threshold, name="iou_threshold")

    # Evaluation is a publication boundary: warnings are insufficient when a
    # malformed dataset would change the ground truth or the join cardinality.
    dataset.validate(strict=True)
    pred_by_name = predictions.by_file_name
    gt_by_name = dataset.by_file_name

    join = join_image_names(pred_by_name, gt_by_name)
    shared = [pred_name for pred_name, _ in join.matches]
    only_pred = list(join.unmatched_predictions)
    only_gt = list(join.unmatched_annotations)
    if only_pred:
        logger.warning("%d predicted images have no annotation: %s", len(only_pred), only_pred[:5])
    if only_gt:
        logger.warning("%d annotated images have no prediction: %s", len(only_gt), only_gt[:5])
    for pred_name, gt_name in join.joined_across_extensions:
        logger.warning(
            "Joined %r to %r: same name, different extension. Confirm the annotation "
            "tool re-encoded the frame rather than labelling a different image.",
            pred_name,
            gt_name,
        )
    if not join.matches:
        raise ValueError(
            "No image appears in both the predictions file and the annotations. "
            "The join is on file basenames, falling back to the name without its "
            "extension, and Roboflow's extra.name is preferred over the exported "
            "file_name on the annotation side; check that both sides name files "
            "the same way."
        )

    semantic_pairs = []
    coverage_samples = []
    instance_samples = []
    instance_excluded: dict[str, str] = {}
    semantic_skipped: dict[str, str] = {}

    for name, gt_name in join.matches:
        record = pred_by_name[name]
        image = gt_by_name[gt_name]
        if (record.height, record.width) != (image.height, image.width):
            raise ValueError(
                f"{name}: prediction declares {record.height}x{record.width}, annotation "
                f"declares {image.height}x{image.width}. Evaluate the same image resolution."
            )

        gt_mask = dataset.semantic_mask(image.id)
        if record.mask_status == "available":
            pred_mask = record.tree_mask()
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"{name}: predicted mask is {pred_mask.shape}, annotation is "
                    f"{gt_mask.shape}. Evaluate against the same resolution the "
                    "model saw, or re-export the annotations at that size."
                )
            semantic_pairs.append((name, pred_mask, gt_mask, None))

            # The interchange file carries both mask and published ratio. Refuse
            # an internally inconsistent record instead of evaluating two
            # different predictions at levels 1 and 3.
            ratio_from_mask = float(pred_mask.sum()) / float(pred_mask.size)
            if (
                record.tree_coverage_ratio is None
                or not abs(float(record.tree_coverage_ratio) - ratio_from_mask) <= 1e-12
            ):
                raise ValueError(
                    f"{name}: tree_coverage_ratio does not match the serialized mask "
                    f"({record.tree_coverage_ratio!r} vs {ratio_from_mask})."
                )
        else:
            semantic_skipped[name] = record.mask_status

        if record.tree_coverage_pct is not None:
            from urban_canopy.processing.coverage import coverage_from_mask

            gt_pct = 100.0 * coverage_from_mask(gt_mask)
            coverage_samples.append((name, float(record.tree_coverage_pct), gt_pct))

        if record.instances is not None:
            instance_samples.append(
                (
                    name,
                    record.instance_masks(),
                    record.instance_scores(),
                    dataset.instance_masks(image.id),
                )
            )
        else:
            instance_excluded[name] = "instances_unavailable"

    semantic_report = evaluate_semantic(semantic_pairs)

    coverage_report = None
    if coverage_samples:
        coverage_report = evaluate_coverage(
            coverage_samples, keep_per_image=keep_per_image
        ).to_dict()

    instances_report = None
    skipped_reason = None
    instance_eligibility = {
        "n_shared": len(shared),
        "n_eligible": len(instance_samples),
        "excluded_images": instance_excluded,
    }
    if instance_samples:
        sources = {pred_by_name[n].instance_source for n, *_ in instance_samples}
        if len(sources) != 1:
            raise ValueError(
                "Instance evaluation cannot mix prediction origins in one metric; "
                f"found {sorted(str(source) for source in sources)}. Evaluate model "
                "and connected_components_heuristic predictions separately."
            )
        instances_report = evaluate_instances(
            instance_samples, iou_threshold=iou_threshold
        ).to_dict()
        instances_report.update(instance_eligibility)
        instances_report["instance_source"] = next(iter(sources))
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
        instance_eligibility=instance_eligibility,
        n_matched_images=len(shared),
        semantic_skipped_images=semantic_skipped,
        unmatched_predictions=only_pred,
        unmatched_annotations=only_gt,
        manifest=dict(predictions.manifest),
        settings={
            "iou_threshold": iou_threshold,
            "annotations": str(dataset.path) if dataset.path else None,
            "join_key": "file basename, falling back to the name without extension",
            "joined_across_extensions": [list(pair) for pair in join.joined_across_extensions],
            "coverage_denominator": "complete image",
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

    return evaluate(
        load_predictions(predictions_path),
        dataset,
        iou_threshold=iou_threshold,
        keep_per_image=keep_per_image,
    )
