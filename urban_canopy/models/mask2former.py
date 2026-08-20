"""
Mask2Former adapter (HuggingFace ``transformers``).

Integrated through ``transformers`` rather than through
`facebookresearch/Mask2Former <https://github.com/facebookresearch/Mask2Former>`_
directly. The upstream repository is built on Detectron2 and needs its custom
CUDA ops (MultiScaleDeformableAttention) compiled from source; the ``transformers``
port is the same architecture and the same published weights, installs with pip,
and shares the post-processing API this project already uses for OneFormer.

Class-space audit -- and the reason this backend earns its place beside
OneFormer rather than duplicating it: Mask2Former publishes weights for
**several datasets**, so the same architecture can be compared across class
spaces, isolating "which classes exist" from "which model predicts them".

* ``*-ade-*``        -- ADE20K-150: ``tree`` (4), ``grass`` (9), ``plant`` (17),
  ``palm`` (72). Trees separable from other vegetation. ``tree`` is *stuff*.
* ``*-coco-*``       -- COCO-panoptic: ``tree-merged`` (184), also *stuff*, and
  already merged across the frame by construction.
* ``*-cityscapes-*`` -- Cityscapes-19: **no tree class at all**; ``vegetation``
  merges trees with bushes. Coverage is reported as unavailable unless the
  caller explicitly enables the vegetation proxy.

In every one of those spaces the tree class is stuff, so no checkpoint separates
individual trees. Instance-task checkpoints exist (``*-instance``) but their
thing classes do not include ``tree`` -- which is why this project measures
coverage only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from urban_canopy.log import get_logger
from urban_canopy.validation import validate_probability

from .base import Segment, SegmentationOutput, build_group_masks
from .taxonomy import (
    Taxonomy,
    default_taxonomy,
    infer_class_space,
    validate_taxonomy_class_space,
)

logger = get_logger(__name__)

#: ADE20K semantic: the only published space with a real tree class, and the
#: semantic task avoids panoptic thresholds influencing a coverage ratio.
DEFAULT_MODEL = "facebook/mask2former-swin-large-ade-semantic"

Task = Literal["semantic", "panoptic"]


def infer_task(model_name: str) -> Task:
    """
    Read the task off the checkpoint name.

    Unlike OneFormer, a Mask2Former checkpoint is trained for one task and its
    name says which (``...-ade-semantic``, ``...-coco-panoptic``). Asking a
    semantic checkpoint for panoptic output produces something, but not
    something meaningful, so the default follows the weights rather than a
    project-wide preference.
    """
    lowered = model_name.lower()
    if "panoptic" in lowered:
        return "panoptic"
    if "instance" in lowered:
        # Instance checkpoints exist but no published space has tree as a thing
        # class, so there is nothing to individualise; the semantic reading of
        # their masks is still a valid coverage measure.
        logger.warning(
            "%s is an instance checkpoint, but no Mask2Former class space has tree as a "
            "thing class. Reading it as semantic coverage.",
            model_name,
        )
        return "semantic"
    return "semantic"


class Mask2FormerSegmenter:
    """Vegetation segmentation through Mask2Former."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        taxonomy: Taxonomy | None = None,
        task: Task | None = None,
        panoptic_threshold: float = 0.50,
        mask_threshold: float = 0.50,
        overlap_mask_area_threshold: float = 0.80,
    ) -> None:
        self.backend_name = "mask2former"
        self.model_name = model_name
        self.class_space = infer_class_space(model_name)
        self.taxonomy = validate_taxonomy_class_space(
            taxonomy or default_taxonomy(self.class_space),
            self.class_space,
            context=f"Mask2Former checkpoint {model_name!r}",
        )
        self.task: Task = task or infer_task(model_name)
        if self.task not in ("semantic", "panoptic"):
            raise ValueError(f"task must be 'semantic' or 'panoptic'; got {self.task!r}")

        self._panoptic_threshold = validate_probability(
            panoptic_threshold, name="panoptic_threshold"
        )
        self._mask_threshold = validate_probability(mask_threshold, name="mask_threshold")
        self._overlap_mask_area_threshold = validate_probability(
            overlap_mask_area_threshold,
            name="overlap_mask_area_threshold",
        )

        import torch
        from PIL import Image
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self._torch = torch
        self._Image = Image
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @property
    def _id2label(self) -> dict[int, str]:
        return getattr(self.model.config, "id2label", {}) or {}

    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        with self._torch.inference_mode():
            return self._segment(img_rgb)

    def _segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        pil = self._Image.fromarray(np.asarray(img_rgb))
        height, width = img_rgb.shape[:2]

        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if self.task == "semantic":
            label_map, segments, labelled = self._semantic(outputs, (height, width))
        else:
            label_map, segments, labelled = self._panoptic(outputs, (height, width))

        notes = [
            f"{self.model_name} predicts the {self.class_space} class space "
            f"({self.task} task).",
        ]
        if self.class_space == "ade20k":
            notes.append(
                "ADE20K treats 'tree' as a stuff class, so this backend cannot "
                "individualise trees; coverage only."
            )
        elif self.class_space == "coco_panoptic":
            notes.append(
                "COCO-panoptic merges all trees into the stuff class 'tree-merged'; "
                "coverage only, no individual trees."
            )
        elif self.class_space == "cityscapes":
            notes.append(
                "Cityscapes has no tree class: 'vegetation' merges trees with bushes. "
                "Tree coverage is unavailable unless the vegetation proxy is enabled."
            )

        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks=build_group_masks(self.taxonomy, labelled, (height, width)),
            label_map=label_map,
            segments=segments,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------ #
    def _semantic(self, outputs, size: tuple[int, int]):
        semantic = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[size])[
            0
        ]
        label_map = semantic.cpu().numpy().astype(np.int32)

        segments: list[Segment] = []
        labelled: list[tuple[str, np.ndarray]] = []
        for class_id in np.unique(label_map):
            name = self._id2label.get(int(class_id), str(int(class_id)))
            segments.append(Segment(id=int(class_id), label=name, is_thing=False))
            labelled.append((name, label_map == class_id))
        return label_map, segments, labelled

    def _panoptic(self, outputs, size: tuple[int, int]):
        panoptic = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[size],
            threshold=self._panoptic_threshold,
            mask_threshold=self._mask_threshold,
            overlap_mask_area_threshold=self._overlap_mask_area_threshold,
        )[0]

        label_map = panoptic["segmentation"].cpu().numpy().astype(np.int32)

        segments: list[Segment] = []
        labelled: list[tuple[str, np.ndarray]] = []
        for raw in panoptic["segments_info"]:
            class_id = raw.get("label_id", raw.get("category_id"))
            name = self._id2label.get(int(class_id), str(class_id))
            segment_id = int(raw["id"])
            # transformers' panoptic post-processing does not report isthing, so
            # segments are recorded as stuff rather than guessed at; nothing
            # downstream distinguishes them.
            segments.append(
                Segment(
                    id=segment_id,
                    label=name,
                    is_thing=False,
                    score=float(raw["score"]) if raw.get("score") is not None else None,
                )
            )
            labelled.append((name, label_map == segment_id))
        return label_map, segments, labelled
