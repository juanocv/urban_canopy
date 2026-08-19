"""
OneFormer adapter (HuggingFace ``transformers``).

Class-space audit, checked against the OneFormer dataset registration files
rather than assumed:

* ``shi-labs/oneformer_ade20k_*`` predicts the ADE20K-150 space, which has
  ``tree`` (id 4), ``grass`` (9), ``plant`` (17), ``flower`` (66) and
  ``palm`` (72) -- so trees *are* separable from other vegetation here.
* In ADE20K panoptic, ``tree`` carries ``isthing = 0``: it is a **stuff** class,
  and every tree in a frame collapses into one segment. The 100-class ADE20K
  instance set contains ``palm`` and ``flower`` but **not** ``tree``.

Consequence, and the reason this adapter never fills ``instances``: OneFormer on
ADE20K is a sound baseline for *visible canopy coverage* and cannot support any
claim about the number of individual trees detected. The panoptic task is still
available because its segment list is useful for auditing, but the default is
the semantic task -- panoptic post-processing applies confidence and overlap
thresholds that silently drop pixels, and a coverage ratio should not depend on
them.
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

DEFAULT_MODEL = "shi-labs/oneformer_ade20k_swin_large"

#: Kept as a name so existing imports still work; the logic is shared with the
#: other HuggingFace backend, since both name their checkpoints after the
#: dataset they were trained on.
class_space_for_model = infer_class_space


class OneFormerSegmenter:
    """Vegetation segmentation through OneFormer."""

    supports_tree_instances = False

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        taxonomy: Taxonomy | None = None,
        task: Literal["semantic", "panoptic"] = "semantic",
        panoptic_threshold: float = 0.50,
        mask_threshold: float = 0.50,
        overlap_mask_area_threshold: float = 0.80,
    ) -> None:
        if task not in ("semantic", "panoptic"):
            raise ValueError(f"task must be 'semantic' or 'panoptic'; got {task!r}")

        self.backend_name = "oneformer"
        self.model_name = model_name
        self.class_space = class_space_for_model(model_name)
        self.taxonomy = validate_taxonomy_class_space(
            taxonomy or default_taxonomy(self.class_space),
            self.class_space,
            context=f"OneFormer checkpoint {model_name!r}",
        )
        self.task = task
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
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        self._torch = torch
        self._Image = Image
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
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

        inputs = self.processor(images=pil, task_inputs=[self.task], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if self.task == "semantic":
            label_map, segments, labelled = self._semantic(outputs, (height, width))
        else:
            label_map, segments, labelled = self._panoptic(outputs, (height, width))

        group_masks = build_group_masks(self.taxonomy, labelled, (height, width))

        notes = [
            f"{self.model_name} predicts the {self.class_space} class space "
            f"({self.task} task).",
        ]
        if self.class_space == "ade20k":
            notes.append(
                "ADE20K treats 'tree' as a stuff class, so this backend cannot "
                "individualise trees; coverage only."
            )

        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks=group_masks,
            label_map=label_map,
            segments=segments,
            instances=None,
            supports_tree_instances=False,
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
            # transformers' panoptic post-processing does not report isthing, and
            # this adapter never emits instances anyway, so segments are recorded
            # as stuff rather than guessed at.
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
