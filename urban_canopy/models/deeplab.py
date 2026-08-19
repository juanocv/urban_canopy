"""
DeepLabV3+ adapter (VainF's Cityscapes checkpoints).

Class-space audit: Cityscapes-19 has **no tree class**. ``vegetation`` (trainId
8) merges trees with bushes and hedges, and ``terrain`` (9) merges grass with
soil and sand. Output is semantic only -- no panoptic map, no instances.

So this backend cannot answer "how much of the frame is tree". It answers "how
much is woody vegetation", which is a different and wider quantity. The adapter
therefore reports ``tree_group = None`` through its taxonomy, and the pipeline
refuses to publish a tree ratio for it unless the caller explicitly opts into
the vegetation proxy -- in which case every result says so. It stays in the
project as the cheap, fast comparison baseline it is (sub-second on CPU with the
MobileNet checkpoint, against tens of seconds for OneFormer).
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from urban_canopy.log import get_logger
from urban_canopy.validation import MAX_IMAGE_DIMENSION, validate_int_range

from .base import Segment, SegmentationOutput, build_group_masks
from .taxonomy import Taxonomy, default_taxonomy, validate_taxonomy_class_space

logger = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CITYSCAPES_LABELS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


class DeepLabSegmenter:
    """Semantic vegetation segmentation through a loaded DeepLab model."""

    supports_tree_instances = False

    def __init__(
        self,
        dl_model: Any,
        *,
        taxonomy: Taxonomy | None = None,
        device: str | None = None,
        input_size: tuple[int, int] = (512, 1024),
    ) -> None:
        self.backend_name = "deeplab"
        self.class_space = "cityscapes"
        self.model_name = getattr(dl_model, "_urban_canopy_model_name", None)
        self.checkpoint_sha256 = getattr(dl_model, "_urban_canopy_checkpoint_sha256", None)
        self.taxonomy = validate_taxonomy_class_space(
            taxonomy or default_taxonomy(self.class_space),
            self.class_space,
            context="DeepLab Cityscapes checkpoint",
        )
        if len(input_size) != 2:
            raise ValueError("input_size must be a (height, width) pair.")
        for name, value in zip(("input height", "input width"), input_size, strict=True):
            validate_int_range(
                value,
                name=name,
                minimum=1,
                maximum=MAX_IMAGE_DIMENSION,
            )

        import torch
        from PIL import Image
        from torchvision import transforms

        self._torch = torch
        self._Image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = dl_model.to(self.device).eval()
        self.id2label = dict(CITYSCAPES_LABELS)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        with self._torch.inference_mode():
            return self._segment(img_rgb)

    def _segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        img = np.asarray(img_rgb)
        height, width = img.shape[:2]

        tensor = self.transform(self._Image.fromarray(img)).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        if isinstance(out, dict):
            logits = out.get("out", next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out

        pred = logits.softmax(1).argmax(1).squeeze(0).cpu().numpy().astype(np.int32)
        if pred.shape != (height, width):
            pred = np.array(
                self._Image.fromarray(pred.astype("uint8")).resize(
                    (width, height), self._Image.NEAREST
                )
            ).astype(np.int32)

        segments: list[Segment] = []
        labelled: list[tuple[str, np.ndarray]] = []
        for class_id in np.unique(pred):
            name = self.id2label.get(int(class_id), f"class_{int(class_id)}")
            segments.append(Segment(id=int(class_id), label=name, is_thing=False))
            labelled.append((name, pred == class_id))

        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks=build_group_masks(self.taxonomy, labelled, (height, width)),
            label_map=pred,
            segments=segments,
            instances=None,
            supports_tree_instances=False,
            notes=(
                "Cityscapes has no tree class: 'vegetation' merges trees with bushes "
                "and hedges. Tree coverage is unavailable unless the vegetation proxy "
                "is explicitly enabled.",
            ),
        )


# --------------------------------------------------------------------------- #
# Checkpoint loading                                                           #
# --------------------------------------------------------------------------- #

# Architectures exposed by VainF's `network.modeling`. Longest architecture
# prefix first so "deeplabv3plus_*" is never mistaken for "deeplabv3_*".
_ARCHITECTURES = ("deeplabv3plus", "deeplabv3")
_BACKBONES = (
    "mobilenet",
    "resnet50",
    "resnet101",
    "hrnetv2_32",
    "hrnetv2_48",
    "xception",
)


def infer_model_name(ckpt_path: str | Path) -> str | None:
    """
    Guess the ``network.modeling`` entry point from a checkpoint filename.

    Upstream names its weights after the architecture they belong to, e.g.
    ``best_deeplabv3plus_mobilenet_cityscapes_os16.pth``. Loading a checkpoint
    into the wrong backbone leaves most of the network randomly initialised, so
    reading the name it advertises beats defaulting to a fixed architecture.
    """
    stem = Path(ckpt_path).stem.lower()
    for architecture in _ARCHITECTURES:
        for backbone in _BACKBONES:
            name = f"{architecture}_{backbone}"
            if name in stem:
                return name
    return None


def import_deeplab_modeling(repo_path: str | Path | None = None):
    """
    Import VainF's ``network.modeling``, optionally from a checkout directory.

    Upstream ships a research repository, not a Python package: there is no
    ``setup.py`` or ``pyproject.toml``, so ``pip install -e`` on it fails with
    "does not appear to be a Python project". The supported ways to reach it are
    therefore to point *repo_path* at the clone, or to put the clone on
    ``sys.path`` yourself. Nothing is installed and the checkout is not modified.
    """
    import importlib

    if repo_path is not None:
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DeepLab repository not found: {root}")
        if not (root / "network").is_dir():
            raise FileNotFoundError(
                f"{root} has no 'network' directory, so it is not a "
                "DeepLabV3Plus-Pytorch checkout. Point --deeplab-repo at the "
                "repository root (the folder containing network/, datasets/, main.py)."
            )
        # Prepend so a checkout always wins over a stale copy already on the path.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    try:
        return importlib.import_module("network.modeling")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The DeepLab backend needs VainF's DeepLabV3Plus-Pytorch checkout, whose "
            "'network' package is not importable. Clone it and pass its path with "
            "--deeplab-repo (or deeplab_repo=...):\n"
            "  git clone https://github.com/VainF/DeepLabV3Plus-Pytorch\n"
            "  tree-ai --seg deeplab --deeplab-repo ./DeepLabV3Plus-Pytorch --ckpt <weights.pth>\n"
            "Do not run `pip install -e` on it: upstream ships no setup.py, so that "
            "fails with 'does not appear to be a Python project'."
        ) from exc


def load_deeplab_checkpoint(
    ckpt_path: str | Path,
    *,
    model_name: str = "deeplabv3plus_resnet101",
    num_classes: int = 19,
    output_stride: int = 16,
    device: str | None = None,
    allow_pickle: bool = False,
    repo_path: str | Path | None = None,
) -> Any:
    """Load a DeepLabV3+ checkpoint into the architecture it belongs to."""
    import pickle

    import torch

    ckpt = Path(ckpt_path).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    modeling = import_deeplab_modeling(repo_path)
    if model_name not in modeling.__dict__:
        raise ValueError(f"{model_name} not found in network.modeling")

    model = modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)

    try:
        # Explicit on every supported Torch version: relying on Torch's changing
        # default would make allow_pickle=False unsafe on older installations.
        raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not allow_pickle:
            raise ValueError(
                "This checkpoint requires Python pickle, which can execute code while "
                "loading. Use a weights-only checkpoint, or pass --trust-checkpoint "
                "(allow_pickle=True in Python) only if you trust its source."
            ) from exc
        logger.warning("Loading trusted legacy checkpoint %s with pickle enabled", ckpt)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)

    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint {ckpt.name!r} did not contain a mapping of tensor weights.")
    if "model_state" in raw:
        state = raw["model_state"]
    elif "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Checkpoint {ckpt.name!r} has a state entry that is not a tensor mapping."
        )
    model_keys = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if (k in model_keys) and hasattr(v, "shape") and (v.shape == model_keys[k].shape)
    }

    # A mismatched backbone still matches a handful of tensors -- a mobilenet
    # checkpoint fills 44 of resnet101's 674 -- so "not empty" is far too weak a
    # test. Carried over from the sidewalk project, where it caught a 93%
    # randomly initialised network that segmented almost nothing and surfaced
    # only as an implausible downstream number.
    missing = [k for k in model_keys if k not in filtered]
    if missing:
        raise RuntimeError(
            f"Checkpoint {ckpt.name!r} does not fit model_name={model_name!r}: "
            f"{len(filtered)} of {len(model_keys)} tensors matched, "
            f"{len(missing)} would stay randomly initialised "
            f"(first missing: {missing[:3]}). "
            "Pass the --deeplab-model that matches the checkpoint's backbone."
        )

    model.load_state_dict(filtered, strict=False)
    # Keep immutable provenance with the loaded object. The adapter copies these
    # values into the run manifest; the path itself is deliberately omitted
    # because it is machine-specific and may reveal local directory names.
    model._urban_canopy_model_name = model_name
    model._urban_canopy_checkpoint_sha256 = _sha256_file(ckpt)
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(target).eval()
