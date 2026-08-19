"""
Run configuration and the reproducibility manifest.

One object carries every decision that can move a coverage number, and one
function snapshots the environment that produced it. Both are serialised into
prediction files and evaluation reports, so a result can be traced back to the
backend, taxonomy, refinement settings and seed it came from without consulting
anyone's shell history.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from urban_canopy.processing.refinement import RefinementConfig

__all__ = ["CanopyConfig", "build_manifest", "set_seed"]

InstanceMode = Literal["auto", "none", "heuristic"]


@dataclass(frozen=True, slots=True)
class CanopyConfig:
    """Everything the pipeline needs beyond the model and the imagery."""

    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    #: Allow a wider vegetation class to stand in for trees when the backend's
    #: class space has none. Off by default; when on, every result says so.
    allow_vegetation_proxy: bool = False
    #: "auto" keeps model instances when the backend produces them and nothing
    #: otherwise; "heuristic" derives connected components from the semantic mask
    #: and flags them as such; "none" never reports instances.
    instance_mode: InstanceMode = "auto"
    #: Area floor for the connected-component heuristic.
    heuristic_min_area_px: int = 64
    #: Keep the decoded RGB frame on the result. Needed for artifacts and
    #: overlays; turn it off for long batch runs to bound memory.
    keep_rgb: bool = True
    #: Recorded in the manifest and applied by :func:`set_seed`.
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement": asdict(self.refinement),
            "allow_vegetation_proxy": self.allow_vegetation_proxy,
            "instance_mode": self.instance_mode,
            "heuristic_min_area_px": self.heuristic_min_area_px,
            "keep_rgb": self.keep_rgb,
            "seed": self.seed,
        }


def set_seed(seed: int) -> None:
    """
    Seed the RNGs that can affect a run.

    Inference here is deterministic in principle, but backends do sample (mask
    post-processing tie-breaks, any augmentation a custom model carries), and a
    seed costs nothing to set and everything to reconstruct after the fact.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(
    *,
    config: CanopyConfig,
    backend: str,
    class_space: str,
    taxonomy: Any | None = None,
    model_name: str | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot of the run: versions, device, config, taxonomy, seed, timestamp."""
    torch_version = None
    cuda_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = getattr(torch.version, "cuda", None)
    except ModuleNotFoundError:
        pass

    manifest: dict[str, Any] = {
        "schema": "urban_canopy/manifest/1",
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "urban_canopy_version": _version("urban-canopy"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "packages": {
            "numpy": _version("numpy"),
            "opencv-python": _version("opencv-python"),
            "transformers": _version("transformers"),
            "torch": torch_version,
        },
        "cuda": cuda_version,
        "device": device,
        "model": {"backend": backend, "name": model_name, "class_space": class_space},
        "taxonomy": taxonomy.to_dict() if taxonomy is not None else None,
        "config": config.to_dict(),
        "seed": config.seed,
    }
    if extra:
        manifest.update(extra)
    return manifest
