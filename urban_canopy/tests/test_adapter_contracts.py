"""Adapters reject semantic/configuration mismatches before loading ML stacks."""

import pytest

from urban_canopy.models.taxonomy import ADE20K, CITYSCAPES


def test_oneformer_rejects_taxonomy_from_another_class_space():
    from urban_canopy.models.oneformer import OneFormerSegmenter

    with pytest.raises(ValueError, match="class_space"):
        OneFormerSegmenter(taxonomy=CITYSCAPES)


def test_mask2former_rejects_taxonomy_from_another_class_space():
    from urban_canopy.models.mask2former import Mask2FormerSegmenter

    with pytest.raises(ValueError, match="class_space"):
        Mask2FormerSegmenter(taxonomy=CITYSCAPES)


def test_deeplab_rejects_taxonomy_from_another_class_space():
    from urban_canopy.models.deeplab import DeepLabSegmenter

    with pytest.raises(ValueError, match="class_space"):
        DeepLabSegmenter(object(), taxonomy=ADE20K)


def test_detectron_rejects_taxonomy_from_another_class_space():
    from urban_canopy.models.detectron2 import Detectron2Segmenter

    with pytest.raises(ValueError, match="class_space"):
        Detectron2Segmenter("config.yaml", "weights.pth", taxonomy=CITYSCAPES)


@pytest.mark.parametrize(
    ("module_name", "class_name", "kwargs"),
    [
        (
            "urban_canopy.models.oneformer",
            "OneFormerSegmenter",
            {"panoptic_threshold": float("nan")},
        ),
        (
            "urban_canopy.models.mask2former",
            "Mask2FormerSegmenter",
            {"mask_threshold": 1.1},
        ),
    ],
)
def test_huggingface_adapters_reject_invalid_thresholds_before_loading(
    module_name,
    class_name,
    kwargs,
):
    module = __import__(module_name, fromlist=[class_name])
    adapter = getattr(module, class_name)
    with pytest.raises(ValueError, match="threshold"):
        adapter(**kwargs)
