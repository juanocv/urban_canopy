"""
Backend factory.

Kept deliberately thin, and kept lazy: importing ``urban_canopy`` must not drag
in torch, transformers or Detectron2, so every backend module is imported inside
the branch that needs it and a missing optional dependency is reported with the
install command that fixes it.
"""

from __future__ import annotations

from typing import Any, Literal

BACKENDS = ("oneformer", "mask2former", "detectron2", "deeplab")

#: Backends whose class space follows the checkpoint rather than the backend:
#: they publish weights for several datasets, so the taxonomy has to be resolved
#: from the model name (:func:`urban_canopy.models.taxonomy.infer_class_space`)
#: instead of read off a fixed table.
CHECKPOINT_DEFINES_CLASS_SPACE = ("oneformer", "mask2former")

#: Class space each backend predicts with its *default* checkpoint. For the
#: backends above this is only the default; see BACKEND_DEFAULT_MODEL.
BACKEND_CLASS_SPACE = {
    "oneformer": "ade20k",
    "mask2former": "ade20k",
    "detectron2": "coco_panoptic",
    "deeplab": "cityscapes",
}

#: Default HuggingFace checkpoint for the backends that take one.
BACKEND_DEFAULT_MODEL = {
    "oneformer": "shi-labs/oneformer_ade20k_swin_large",
    "mask2former": "facebook/mask2former-swin-large-ade-semantic",
}


def _optional_import_error(component: str, install_hint: str, exc: ModuleNotFoundError) -> None:
    missing = exc.name or "unknown"
    raise ModuleNotFoundError(
        f"{component} requires optional dependencies that are not installed. "
        f"Missing module: {missing}. {install_hint}"
    ) from exc


DETECTRON2_INSTALL_HINT = (
    "Install Detectron2 following its upstream instructions; see docs/reproducibility.md."
)


def _detectron2_hint(exc: ModuleNotFoundError) -> str:
    """
    Turn a Detectron2 import failure into the instruction that fixes it.

    ``pkg_resources`` deserves its own message because the failure reads as a
    broken Detectron2 build and is nothing of the sort: Detectron2's model_zoo
    imports it, setuptools removed it in version 81, and the break is identical
    on every operating system.
    """
    if (exc.name or "") == "pkg_resources":
        return (
            "Detectron2's model_zoo imports pkg_resources, which setuptools removed in "
            'version 81. Restore it with `python -m pip install "setuptools<81"`. '
            "This is a Detectron2/setuptools incompatibility, not a broken Detectron2 "
            "build or a Windows-specific problem."
        )
    return DETECTRON2_INSTALL_HINT


def build_segmenter(
    backend: Literal["oneformer", "mask2former", "detectron2", "deeplab"] = "oneformer",
    **kwargs: Any,
):
    """Build one segmentation backend by name."""
    if backend == "oneformer":
        try:
            from .oneformer import OneFormerSegmenter

            return OneFormerSegmenter(**kwargs)
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The OneFormer segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`.',
                exc,
            )

    if backend == "mask2former":
        try:
            from .mask2former import Mask2FormerSegmenter

            return Mask2FormerSegmenter(**kwargs)
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The Mask2Former segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`.',
                exc,
            )

    if backend == "detectron2":
        # Construction is inside the guard, not just the import: Detectron2 defers
        # part of its own import chain to call time (model_zoo imports
        # pkg_resources only when from_zoo runs), so guarding the import alone let
        # those failures escape as a bare traceback with no hint attached.
        try:
            from .detectron2 import Detectron2Segmenter

            return Detectron2Segmenter.from_zoo(**kwargs)
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The Detectron2 segmentation backend",
                _detectron2_hint(exc),
                exc,
            )

    if backend == "deeplab":
        from .taxonomy import validate_taxonomy_class_space

        if kwargs.get("taxonomy") is not None:
            validate_taxonomy_class_space(
                kwargs["taxonomy"],
                "cityscapes",
                context="DeepLab checkpoint",
            )
        try:
            from .deeplab import DeepLabSegmenter, load_deeplab_checkpoint
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The DeepLab segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`; '
                "see docs/reproducibility.md.",
                exc,
            )

        try:
            ckpt = kwargs.pop("ckpt_path")
        except KeyError:
            raise ValueError("The deeplab backend needs ckpt_path (a .pth checkpoint).") from None

        loader_keys = {"model_name", "num_classes", "output_stride", "allow_pickle", "repo_path"}
        loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}
        # Read, don't pop: the loader materialises the model on a device and the
        # segmenter moves it again, so both need the caller's choice.
        if "device" in kwargs:
            loader_kwargs["device"] = kwargs["device"]
        try:
            model = load_deeplab_checkpoint(ckpt, **loader_kwargs)
            return DeepLabSegmenter(model, **kwargs)
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The DeepLab segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`; '
                "see docs/reproducibility.md.",
                exc,
            )

    raise ValueError(f"Unknown backend: {backend!r}; choose from {', '.join(BACKENDS)}")
