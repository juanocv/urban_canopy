"""
Backend factory.

Kept deliberately thin, and kept lazy: importing ``urban_canopy`` must not drag
in torch, transformers or Detectron2, so every backend module is imported inside
the branch that needs it and a missing optional dependency is reported with the
install command that fixes it.
"""

from __future__ import annotations

from typing import Any, Literal

BACKENDS = ("oneformer", "detectron2", "deeplab")

#: Class space each backend predicts, before any per-run override.
BACKEND_CLASS_SPACE = {
    "oneformer": "ade20k",
    "detectron2": "coco_panoptic",
    "deeplab": "cityscapes",
}

#: Whether a backend, in its default configuration, can separate individual
#: trees. All three cannot: see the audit in each adapter's module docstring.
BACKEND_SUPPORTS_TREE_INSTANCES = {
    "oneformer": False,
    "detectron2": False,  # True only in mode="instance" with custom weights
    "deeplab": False,
}


def _optional_import_error(component: str, install_hint: str, exc: ModuleNotFoundError) -> None:
    missing = exc.name or "unknown"
    raise ModuleNotFoundError(
        f"{component} requires optional dependencies that are not installed. "
        f"Missing module: {missing}. {install_hint}"
    ) from exc


def build_segmenter(
    backend: Literal["oneformer", "detectron2", "deeplab"] = "oneformer",
    **kwargs: Any,
):
    """Build one segmentation backend by name."""
    if backend == "oneformer":
        try:
            from .oneformer import OneFormerSegmenter
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The OneFormer segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`.',
                exc,
            )
        return OneFormerSegmenter(**kwargs)

    if backend == "detectron2":
        try:
            from .detectron2 import Detectron2Segmenter
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The Detectron2 segmentation backend",
                "Install Detectron2 following its upstream instructions; see "
                "docs/reproducibility.md.",
                exc,
            )
        # Custom weights select the instance mode; without them the model-zoo
        # panoptic baseline is built.
        if kwargs.get("weights_path"):
            return Detectron2Segmenter(**kwargs)
        kwargs.pop("weights_path", None)
        kwargs.pop("config_yml", None)
        kwargs.pop("mode", None)
        kwargs.pop("thing_classes", None)
        kwargs.pop("class_space", None)
        return Detectron2Segmenter.from_zoo(**kwargs)

    if backend == "deeplab":
        try:
            from .deeplab import DeepLabSegmenter, load_deeplab_checkpoint
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The DeepLab segmentation backend",
                "Install the ML extra and make the local DeepLab `network` package "
                "importable; see docs/reproducibility.md.",
                exc,
            )

        try:
            ckpt = kwargs.pop("ckpt_path")
        except KeyError:
            raise ValueError("The deeplab backend needs ckpt_path (a .pth checkpoint).") from None

        loader_keys = {"model_name", "num_classes", "output_stride", "allow_pickle"}
        loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}
        # Read, don't pop: the loader materialises the model on a device and the
        # segmenter moves it again, so both need the caller's choice.
        if "device" in kwargs:
            loader_kwargs["device"] = kwargs["device"]
        model = load_deeplab_checkpoint(ckpt, **loader_kwargs)
        return DeepLabSegmenter(model, **kwargs)

    raise ValueError(f"Unknown backend: {backend!r}; choose from {', '.join(BACKENDS)}")
