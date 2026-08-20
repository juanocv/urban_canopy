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
from typing import Any

from urban_canopy.processing.refinement import RefinementConfig
from urban_canopy.validation import validate_bool, validate_int_range

__all__ = ["CanopyConfig", "build_manifest", "set_seed"]

_LAST_REPRODUCIBILITY: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CanopyConfig:
    """Everything the pipeline needs beyond the model and the imagery."""

    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    #: Allow a wider vegetation class to stand in for trees when the backend's
    #: class space has none. Off by default; when on, every result says so.
    allow_vegetation_proxy: bool = False
    #: Keep the decoded RGB frame on the result. Needed for artifacts and
    #: overlays; turn it off for long batch runs to bound memory.
    keep_rgb: bool = False
    #: Recorded in the manifest and applied by :func:`set_seed`.
    seed: int = 0
    #: Request deterministic Torch/CUDA algorithms. This is stricter than RNG
    #: seeding, but still cannot promise identical bits across hardware/stacks.
    deterministic: bool = False

    def __post_init__(self) -> None:
        validate_bool(self.allow_vegetation_proxy, name="allow_vegetation_proxy")
        validate_bool(self.keep_rgb, name="keep_rgb")
        validate_bool(self.deterministic, name="deterministic")
        validate_int_range(self.seed, name="seed", minimum=0, maximum=2**32 - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement": asdict(self.refinement),
            "allow_vegetation_proxy": self.allow_vegetation_proxy,
            "keep_rgb": self.keep_rgb,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }


def set_seed(seed: int, *, deterministic: bool = False) -> dict[str, Any]:
    """
    Seed the RNGs that can affect a run.

    Seeding controls random-number streams. ``deterministic=True`` additionally
    asks Torch to reject nondeterministic algorithms and configures CUDA/cuDNN
    knobs. Neither mode promises identical bits across library versions or
    hardware. ``PYTHONHASHSEED`` is only observed: changing it after interpreter
    startup would not affect this process, so this function never pretends to.
    """
    global _LAST_REPRODUCIBILITY

    seed = validate_int_range(seed, name="seed", minimum=0, maximum=2**32 - 1)
    if deterministic:
        # Effective when set before CUDA initialisation, which the CLI ensures
        # by calling this function before device resolution or model loading.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    status: dict[str, Any] = {
        "rng_seeded": True,
        "seed": seed,
        "deterministic_algorithms_requested": bool(deterministic),
        "python_hash_seed_env": os.getenv("PYTHONHASHSEED"),
        "python_hash_seed_changed_at_runtime": False,
        "python_hash_seed_effective_only_if_set_before_start": True,
        "bitwise_determinism_guaranteed": False,
    }
    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy_seeded"] = True
    except ModuleNotFoundError:  # pragma: no cover - numpy is a hard dependency
        status["numpy_seeded"] = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(bool(deterministic))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = False
        status.update(
            {
                "torch_seeded": True,
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
                "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
                "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except ModuleNotFoundError:
        status["torch_seeded"] = False

    _LAST_REPRODUCIBILITY = status
    return dict(status)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _reproducibility_for(config: CanopyConfig) -> dict[str, Any]:
    if (
        _LAST_REPRODUCIBILITY is not None
        and _LAST_REPRODUCIBILITY.get("seed") == config.seed
        and _LAST_REPRODUCIBILITY.get("deterministic_algorithms_requested") == config.deterministic
    ):
        return dict(_LAST_REPRODUCIBILITY)
    return {
        "rng_seeded": False,
        "seed": config.seed,
        "deterministic_algorithms_requested": config.deterministic,
        "python_hash_seed_env": os.getenv("PYTHONHASHSEED"),
        "python_hash_seed_changed_at_runtime": False,
        "python_hash_seed_effective_only_if_set_before_start": True,
        "bitwise_determinism_guaranteed": False,
    }


def build_manifest(
    *,
    config: CanopyConfig,
    backend: str,
    class_space: str,
    taxonomy: Any | None = None,
    model_name: str | None = None,
    model_sha256: str | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot of the run: versions, device, config, taxonomy, seed, timestamp."""
    torch_version = None
    cuda_version = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
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
        "model": {
            "backend": backend,
            "name": model_name,
            "class_space": class_space,
            "checkpoint_sha256": model_sha256,
        },
        "taxonomy": taxonomy.to_dict() if taxonomy is not None else None,
        "config": config.to_dict(),
        "seed": config.seed,
        "reproducibility": _reproducibility_for(config),
    }
    if extra:
        manifest.update(extra)
    return manifest
