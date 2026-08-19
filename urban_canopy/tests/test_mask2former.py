"""
Mask2Former adapter: class-space and task inference.

The adapter itself needs weights, so what is tested offline is the part that
decides *what the numbers mean* -- which dataset the checkpoint speaks and
therefore which classes count as trees. Getting that wrong applies an ADE20K
taxonomy to a Cityscapes model and reports a tree ratio for a class space with
no tree class, which is the one failure mode that would be invisible in the
output.
"""

import pytest

from urban_canopy.models.factory import (
    BACKEND_CLASS_SPACE,
    BACKENDS,
    CHECKPOINT_DEFINES_CLASS_SPACE,
)
from urban_canopy.models.mask2former import DEFAULT_MODEL, infer_task
from urban_canopy.models.taxonomy import CITYSCAPES, infer_class_space


# ------------------------------------------------------------ registration ---
def test_mask2former_is_a_registered_backend():
    assert "mask2former" in BACKENDS
    assert len(BACKENDS) == 4
    assert BACKEND_CLASS_SPACE["mask2former"] == "ade20k"


def test_checkpoint_defines_class_space_for_the_huggingface_backends():
    assert set(CHECKPOINT_DEFINES_CLASS_SPACE) == {"oneformer", "mask2former"}


# ------------------------------------------------------------- class space ---
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("facebook/mask2former-swin-large-ade-semantic", "ade20k"),
        ("facebook/mask2former-swin-tiny-ade-semantic", "ade20k"),
        ("facebook/mask2former-swin-base-IN21k-ade-semantic", "ade20k"),
        ("facebook/mask2former-swin-large-coco-panoptic", "coco_panoptic"),
        ("facebook/mask2former-swin-tiny-cityscapes-semantic", "cityscapes"),
        ("shi-labs/oneformer_ade20k_swin_large", "ade20k"),
    ],
)
def test_class_space_is_read_from_the_checkpoint_name(model, expected):
    assert infer_class_space(model) == expected


def test_default_model_is_ade20k_which_has_a_tree_class():
    # The default must be a space where "tree" exists; otherwise the backend
    # would ship reporting coverage as unavailable out of the box.
    assert infer_class_space(DEFAULT_MODEL) == "ade20k"


def test_unknown_dataset_refuses_to_guess():
    # Mapillary Vistas has no built-in taxonomy here. Silently defaulting would
    # mislabel every pixel; the error names the way out.
    with pytest.raises(ValueError, match="--taxonomy"):
        infer_class_space("facebook/mask2former-swin-large-mapillary-vistas-semantic")


def test_ambiguous_name_refuses_to_guess():
    with pytest.raises(ValueError, match="more than one dataset"):
        infer_class_space("someone/model-ade-and-cityscapes")


def test_cityscapes_checkpoint_selects_a_taxonomy_without_trees():
    # The consequence that matters: a Cityscapes Mask2Former reports vegetation,
    # and tree coverage comes back unavailable rather than silently equal to it.
    space = infer_class_space("facebook/mask2former-swin-tiny-cityscapes-semantic")
    assert space == CITYSCAPES.class_space
    assert CITYSCAPES.tree_group is None
    assert CITYSCAPES.tree_proxy_group == "vegetation"


# -------------------------------------------------------------------- task ---
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("facebook/mask2former-swin-large-ade-semantic", "semantic"),
        ("facebook/mask2former-swin-large-coco-panoptic", "panoptic"),
        ("facebook/mask2former-swin-large-cityscapes-panoptic", "panoptic"),
    ],
)
def test_task_follows_the_checkpoint(model, expected):
    # A Mask2Former checkpoint is trained for one task; the name states it.
    assert infer_task(model) == expected


def test_instance_checkpoint_falls_back_to_semantic_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert infer_task("facebook/mask2former-swin-large-coco-instance") == "semantic"
    assert "thing class" in caplog.text
