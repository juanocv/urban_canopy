"""
Detectron2 adapter (model zoo ``COCO-PanopticSegmentation/*``).

The COCO-panoptic space has ``tree-merged`` (id 184), ``grass-merged`` (193) and
``flower`` (119), all with ``isthing = 0``. ``tree-merged`` is not only stuff --
the name says it outright -- every tree in the frame is merged into a single
segment.

This adapter is panoptic-only. It once carried an "instance" mode for a
fine-tuned Mask R-CNN, which was removed with the rest of the project's instance
support: no published Detectron2 checkpoint has tree as a thing class (COCO-80
has only ``potted plant``; LVIS v1's 1203 categories have only ``Christmas_tree``;
Cityscapes-instance has eight person/vehicle classes), so the mode could only
ever run against weights that do not exist. See ``docs/evaluation.md``.
"""

from __future__ import annotations

import numpy as np

from urban_canopy.log import get_logger
from urban_canopy.validation import validate_probability

from .base import Segment, SegmentationOutput, build_group_masks
from .taxonomy import Taxonomy, default_taxonomy, validate_taxonomy_class_space

logger = get_logger(__name__)

DEFAULT_PANOPTIC_CFG = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"


class Detectron2Segmenter:
    """Vegetation segmentation through Detectron2."""

    def __init__(
        self,
        config_yml: str,
        weights_path: str,
        *,
        class_space: str = "coco_panoptic",
        taxonomy: Taxonomy | None = None,
        score_thresh: float = 0.50,
        device: str | None = None,
    ) -> None:
        self.taxonomy = validate_taxonomy_class_space(
            taxonomy or default_taxonomy(class_space),
            class_space,
            context="Detectron2 model",
        )
        score_thresh = validate_probability(score_thresh, name="score_thresh")

        import torch
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor

        self._torch = torch
        cfg = get_cfg()
        cfg.merge_from_file(config_yml)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.backend_name = "detectron2"
        self.class_space = class_space
        self.predictor = DefaultPredictor(cfg)

        dataset = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None
        self._meta = MetadataCatalog.get(dataset) if dataset else None
        self._thing_classes = list(getattr(self._meta, "thing_classes", []) if self._meta else [])
        self._stuff_classes = list(getattr(self._meta, "stuff_classes", []) if self._meta else [])

    @classmethod
    def from_zoo(
        cls,
        cfg_name: str = DEFAULT_PANOPTIC_CFG,
        *,
        score_thresh: float = 0.50,
        taxonomy: Taxonomy | None = None,
        device: str | None = None,
    ) -> Detectron2Segmenter:
        """Build the panoptic baseline from a model-zoo config string."""
        taxonomy = validate_taxonomy_class_space(
            taxonomy or default_taxonomy("coco_panoptic"),
            "coco_panoptic",
            context="Detectron2 model-zoo checkpoint",
        )
        score_thresh = validate_probability(score_thresh, name="score_thresh")

        from detectron2 import model_zoo

        return cls(
            config_yml=model_zoo.get_config_file(cfg_name),
            weights_path=model_zoo.get_checkpoint_url(cfg_name),
            class_space="coco_panoptic",
            taxonomy=taxonomy,
            score_thresh=score_thresh,
            device=device,
        )

    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        with self._torch.inference_mode():
            return self._segment(img_rgb)

    def _segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        # Detectron2 predictors expect BGR when cfg.INPUT.FORMAT is BGR, which is
        # the default for every zoo config.
        img = np.asarray(img_rgb)
        bgr = img[:, :, ::-1] if self.cfg.INPUT.FORMAT == "BGR" else img
        outputs = self.predictor(bgr)

        return self._from_panoptic(outputs, img.shape[:2])

    # ------------------------------------------------------------------ #
    def _from_panoptic(self, outputs, shape: tuple[int, int]) -> SegmentationOutput:
        seg_map, seg_info = outputs["panoptic_seg"]
        label_map = seg_map.cpu().numpy().astype(np.int32)

        segments: list[Segment] = []
        labelled: list[tuple[str, np.ndarray]] = []
        for raw in seg_info:
            is_thing = bool(raw.get("isthing", False))
            cat_id = int(raw["category_id"])
            pool = self._thing_classes if is_thing else self._stuff_classes
            name = pool[cat_id] if cat_id < len(pool) else f"class_{cat_id}"
            segment_id = int(raw["id"])
            segments.append(Segment(id=segment_id, label=name, is_thing=is_thing))
            labelled.append((name, label_map == segment_id))

        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks=build_group_masks(self.taxonomy, labelled, shape),
            label_map=label_map,
            segments=segments,
            notes=(
                "COCO-panoptic merges all trees into the stuff class 'tree-merged'; "
                "coverage only, no individual trees.",
            ),
        )
