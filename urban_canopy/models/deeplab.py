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

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from urban_canopy.log import get_logger

from .base import Segment, SegmentationOutput, build_group_masks
from .taxonomy import Taxonomy, default_taxonomy

logger = get_logger(__name__)

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
        dl_model: torch.nn.Module,
        *,
        taxonomy: Taxonomy | None = None,
        device: str | None = None,
        input_size: tuple[int, int] = (512, 1024),
    ) -> None:
        self.backend_name = "deeplab"
        self.class_space = "cityscapes"
        self.taxonomy = taxonomy or default_taxonomy(self.class_space)
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

    @torch.inference_mode()
    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput:
        img = np.asarray(img_rgb)
        height, width = img.shape[:2]

        tensor = self.transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
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
                Image.fromarray(pred.astype("uint8")).resize((width, height), Image.NEAREST)
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


def load_deeplab_checkpoint(
    ckpt_path: str | Path,
    *,
    model_name: str = "deeplabv3plus_resnet101",
    num_classes: int = 19,
    output_stride: int = 16,
    device: str | None = None,
    allow_pickle: bool = True,
) -> torch.nn.Module:
    """Load a DeepLabV3+ checkpoint into the architecture it belongs to."""
    import importlib
    import pickle

    ckpt = Path(ckpt_path).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    modeling = importlib.import_module("network.modeling")
    if model_name not in modeling.__dict__:
        raise ValueError(f"{model_name} not found in network.modeling")

    model = modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)

    try:
        raw = torch.load(ckpt, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if not allow_pickle:
            raise ValueError(
                "Failed to load checkpoint with pickle disabled. "
                "Set allow_pickle=True to enable it."
            ) from exc
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)

    state = raw.get("model_state") or raw.get("state_dict") or raw
    model_keys = model.state_dict()
    filtered = {
        k: v for k, v in state.items() if (k in model_keys) and (v.shape == model_keys[k].shape)
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
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(target).eval()
