"""
Detectron2 adapter, in two modes.

**panoptic** (default, model zoo ``COCO-PanopticSegmentation/*``)
    The COCO-panoptic space has ``tree-merged`` (id 184), ``grass-merged`` (193)
    and ``flower`` (119), all with ``isthing = 0``. ``tree-merged`` is not only
    stuff -- the name says it outright -- every tree in the frame is merged into
    a single segment. Coverage baseline only; ``instances`` stays None.

**instance** (custom weights)
    A Mask R-CNN whose thing classes include a tree class -- in practice a model
    fine-tuned on the project's own COCO instance annotations -- does produce one
    mask per tree, with a confidence score. That is the only configuration in
    which this project reports per-instance metrics, and it is the reason the
    contract carries scores at all: AP50 needs them.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from urban_canopy.log import get_logger

from .base import MODEL_INSTANCES, InstanceMask, Segment, SegmentationOutput, build_group_masks
from .taxonomy import Taxonomy, default_taxonomy

logger = get_logger(__name__)

DEFAULT_PANOPTIC_CFG = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"


class Detectron2Segmenter:
    """Vegetation segmentation through Detectron2."""

    def __init__(
        self,
        config_yml: str,
        weights_path: str,
        *,
        mode: Literal["panoptic", "instance"] = "panoptic",
        class_space: str = "coco_panoptic",
        taxonomy: Taxonomy | None = None,
        score_thresh: float = 0.50,
        thing_classes: Sequence[str] | None = None,
        device: str | None = None,
    ) -> None:
        if mode not in ("panoptic", "instance"):
            raise ValueError(f"mode must be 'panoptic' or 'instance'; got {mode!r}")

        cfg = get_cfg()
        cfg.merge_from_file(config_yml)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.mode = mode
        self.backend_name = "detectron2"
        self.class_space = class_space
        self.taxonomy = taxonomy or default_taxonomy(class_space)
        self.predictor = DefaultPredictor(cfg)

        dataset = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None
        self._meta = MetadataCatalog.get(dataset) if dataset else None
        self._thing_classes = list(
            thing_classes
            if thing_classes is not None
            else (getattr(self._meta, "thing_classes", []) if self._meta else [])
        )
        self._stuff_classes = list(getattr(self._meta, "stuff_classes", []) if self._meta else [])

        if mode == "instance":
            tree_group = self.taxonomy.tree_group
            tree_things = [
                name
                for name in self._thing_classes
                if tree_group is not None and self.taxonomy.group_for_label(name) == tree_group
            ]
            if not tree_things:
                raise ValueError(
                    "mode='instance' needs a thing class that the taxonomy maps to the "
                    f"tree group; the model's thing classes are {self._thing_classes!r}. "
                    "Pass --thing-classes or a taxonomy that covers them."
                )
            self._tree_things = tree_things
            self.supports_tree_instances = True
        else:
            self._tree_things = []
            self.supports_tree_instances = False

    @classmethod
    def from_zoo(
        cls,
        cfg_name: str = DEFAULT_PANOPTIC_CFG,
        *,
        score_thresh: float = 0.50,
        taxonomy: Taxonomy | None = None,
        device: str | None = None,
    ) -> "Detectron2Segmenter":
        """Build the panoptic baseline from a model-zoo config string."""
        from detectron2 import model_zoo

        return cls(
            config_yml=model_zoo.get_config_file(cfg_name),
            weights_path=model_zoo.get_checkpoint_url(cfg_name),
            mode="panoptic",
            class_space="coco_panoptic",
            taxonomy=taxonomy,
            score_thresh=score_thresh,
            device=device,
        )

    @torch.inference_mode()
    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        # Detectron2 predictors expect BGR when cfg.INPUT.FORMAT is BGR, which is
        # the default for every zoo config.
        img = np.asarray(img_rgb)
        bgr = img[:, :, ::-1] if self.cfg.INPUT.FORMAT == "BGR" else img
        outputs = self.predictor(bgr)

        if self.mode == "panoptic":
            return self._from_panoptic(outputs, img.shape[:2])
        return self._from_instances(outputs, img.shape[:2])

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
            instances=None,
            supports_tree_instances=False,
            notes=(
                "COCO-panoptic merges all trees into the stuff class 'tree-merged'; "
                "coverage only, no individual trees.",
            ),
        )

    def _from_instances(self, outputs, shape: tuple[int, int]) -> SegmentationOutput:
        inst = outputs["instances"].to("cpu")
        masks = inst.pred_masks.numpy().astype(bool) if inst.has("pred_masks") else None
        classes = inst.pred_classes.numpy() if inst.has("pred_classes") else np.empty(0, int)
        scores = inst.scores.numpy() if inst.has("scores") else np.full(len(classes), np.nan)

        instances: list[InstanceMask] = []
        labelled: list[tuple[str, np.ndarray]] = []
        segments: list[Segment] = []

        for index in range(len(classes)):
            cat_id = int(classes[index])
            name = (
                self._thing_classes[cat_id]
                if cat_id < len(self._thing_classes)
                else f"class_{cat_id}"
            )
            mask = masks[index] if masks is not None else np.zeros(shape, bool)
            score = float(scores[index]) if index < len(scores) else None
            segments.append(Segment(id=index, label=name, is_thing=True, score=score))
            labelled.append((name, mask))
            if name in self._tree_things:
                instances.append(
                    InstanceMask(label=name, mask=mask, score=score, source=MODEL_INSTANCES)
                )

        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks=build_group_masks(self.taxonomy, labelled, shape),
            label_map=None,
            segments=segments,
            instances=instances,
            supports_tree_instances=True,
            notes=(
                "Instance segmentation: each tree carries its own mask and score. "
                "Overlapping instances are kept as predicted; the semantic mask "
                "used for coverage is their union.",
            ),
        )
