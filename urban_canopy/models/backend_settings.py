"""Unified backend configuration and construction for every interface."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from urban_canopy.log import get_logger
from urban_canopy.validation import validate_probability

from .factory import (
    BACKEND_CLASS_SPACE,
    CHECKPOINT_DEFINES_CLASS_SPACE,
    build_segmenter,
)
from .taxonomy import infer_class_space, load_taxonomy

logger = get_logger(__name__)

BackendName = Literal["oneformer", "mask2former", "detectron2", "deeplab"]
DeviceName = Literal["auto", "cpu", "cuda"]

__all__ = [
    "BackendSettings",
    "backend_provenance",
    "build_segmenter_from_settings",
    "resolve_device",
]


class BackendSettings(BaseSettings):
    """Backend settings resolved identically for CLI and API startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    backend: BackendName = Field("oneformer", validation_alias="UC_SEG_BACKEND")
    model_name: str | None = Field(None, validation_alias="UC_SEG_MODEL")
    device: DeviceName = Field("auto", validation_alias="UC_DEVICE")
    taxonomy_path: Path | None = Field(None, validation_alias="UC_TAXONOMY")
    oneformer_task: Literal["semantic", "panoptic"] = Field(
        "semantic", validation_alias="UC_SEG_TASK"
    )

    d2_config: Path | None = Field(None, validation_alias="UC_D2_CONFIG")
    d2_weights: Path | None = Field(None, validation_alias="UC_D2_WEIGHTS")
    d2_score_threshold: float = Field(0.50, validation_alias="UC_D2_SCORE_THRESH")

    deeplab_checkpoint: Path | None = Field(None, validation_alias="UC_DEEPLAB_CKPT")
    deeplab_repo: Path | None = Field(None, validation_alias="UC_DEEPLAB_REPO")
    deeplab_model: str | None = Field(None, validation_alias="UC_DEEPLAB_MODEL")
    trust_checkpoint: bool = Field(False, validation_alias="UC_TRUST_CHECKPOINT")

    @field_validator(
        "model_name",
        "taxonomy_path",
        "d2_config",
        "d2_weights",
        "deeplab_checkpoint",
        "deeplab_repo",
        "deeplab_model",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("d2_score_threshold")
    @classmethod
    def _score_threshold(cls, value):
        return validate_probability(value, name="d2_score_threshold")

    @model_validator(mode="after")
    def _paired_detectron_files(self):
        if (self.d2_config is None) != (self.d2_weights is None):
            raise ValueError("UC_D2_CONFIG and UC_D2_WEIGHTS must be configured together.")
        return self

    @classmethod
    def from_cli_args(cls, args, *, device: str) -> BackendSettings:
        """Apply explicit CLI values over environment-backed defaults."""
        defaults = cls()  # pyright: ignore[reportCallIssue] -- BaseSettings env sources
        # Only the device is unconditional: the CLI always resolves it. Every
        # other flag carries an argparse default of None, so "absent" is
        # distinguishable from "explicitly set". Writing an argparse default
        # here instead would overwrite the environment-backed value and make the
        # matching UC_* setting silently inert.
        updates: dict[str, Any] = {"device": device}
        optional = {
            "backend": args.seg,
            "oneformer_task": args.seg_task,
            "d2_score_threshold": args.d2_score_thresh,
            "trust_checkpoint": args.trust_checkpoint,
            "model_name": args.seg_model,
            "taxonomy_path": args.taxonomy,
            "d2_config": args.d2_config,
            "d2_weights": args.d2_weights,
            "deeplab_checkpoint": args.ckpt,
            "deeplab_repo": args.deeplab_repo,
            "deeplab_model": args.deeplab_model,
        }
        updates.update({key: value for key, value in optional.items() if value is not None})
        payload = defaults.model_dump()
        payload.update(updates)
        return cls(**payload)  # pyright: ignore[reportCallIssue] -- dynamic Pydantic signature

    def checkpoint_identifier(self) -> str | None:
        if self.backend in CHECKPOINT_DEFINES_CLASS_SPACE:
            return self.model_name
        if self.backend == "deeplab" and self.deeplab_checkpoint is not None:
            return self.deeplab_checkpoint.name
        if self.backend == "detectron2" and self.d2_weights is not None:
            return self.d2_weights.name
        if self.backend == "detectron2":
            return "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
        return None


def resolve_device(requested: DeviceName | str) -> str:
    """Resolve auto/cpu/cuda without depending on a CLI parser."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown device {requested!r}; choose auto, cpu or cuda.")
    try:
        import torch
    except ModuleNotFoundError:
        if requested == "cuda":
            raise ValueError(
                "device='cuda' needs PyTorch. Install the ML extra before startup."
            ) from None
        return "cpu"

    available = bool(torch.cuda.is_available())
    if requested == "auto":
        resolved = "cuda" if available else "cpu"
        logger.info("Device auto-selected: %s (torch %s)", resolved, torch.__version__)
        return resolved
    if requested == "cuda" and not available:
        raise ValueError(
            f"CUDA was requested, but the installed PyTorch ({torch.__version__}) "
            "cannot use CUDA on this machine."
        )
    return requested


def _require_existing(path: Path | None, *, setting: str, directory: bool = False) -> Path:
    if path is None:
        raise ValueError(f"{setting} is required for this backend.")
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{setting} does not exist: {resolved}")
    if directory and not resolved.is_dir():
        raise ValueError(f"{setting} must point to a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"{setting} must point to a file: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_segmenter_from_settings(
    settings: BackendSettings,
    *,
    builder=build_segmenter,
):
    """Validate and build the configured backend using one shared code path."""
    device = resolve_device(settings.device)
    model_name = settings.model_name
    if settings.backend in CHECKPOINT_DEFINES_CLASS_SPACE and model_name:
        class_space = infer_class_space(model_name)
    else:
        class_space = BACKEND_CLASS_SPACE[settings.backend]
    taxonomy = load_taxonomy(settings.taxonomy_path, class_space=class_space)

    if settings.backend in ("oneformer", "mask2former"):
        kwargs: dict[str, Any] = {"device": device, "taxonomy": taxonomy}
        if model_name:
            kwargs["model_name"] = model_name
        if settings.backend == "oneformer":
            kwargs["task"] = settings.oneformer_task
        return builder(settings.backend, **kwargs)

    if settings.backend == "detectron2":
        kwargs = {
            "taxonomy": taxonomy,
            "score_thresh": settings.d2_score_threshold,
            "device": device,
        }
        if settings.d2_config is not None:
            config = _require_existing(settings.d2_config, setting="UC_D2_CONFIG/--d2-config")
            weights = _require_existing(settings.d2_weights, setting="UC_D2_WEIGHTS/--d2-weights")
            kwargs.update(
                config_yml=str(config),
                weights_path=str(weights),
                mode="instance",
            )
        return builder("detectron2", **kwargs)

    checkpoint = _require_existing(
        settings.deeplab_checkpoint,
        setting="UC_DEEPLAB_CKPT/--ckpt",
    )
    model = settings.deeplab_model
    if model is None:
        from .deeplab import infer_model_name

        model = infer_model_name(checkpoint)
        if model is None:
            raise ValueError(
                f"Could not infer the DeepLab architecture from {checkpoint.name!r}; "
                "configure UC_DEEPLAB_MODEL or --deeplab-model."
            )
    repo = settings.deeplab_repo
    if repo is not None:
        repo = _require_existing(repo, setting="UC_DEEPLAB_REPO/--deeplab-repo", directory=True)
    return builder(
        "deeplab",
        ckpt_path=str(checkpoint),
        model_name=model,
        taxonomy=taxonomy,
        allow_pickle=settings.trust_checkpoint,
        device=device,
        repo_path=str(repo) if repo else None,
    )


def backend_provenance(segmenter: Any, settings: BackendSettings) -> dict[str, Any]:
    """Public, non-secret description of the backend actually serving requests."""
    model_name = getattr(segmenter, "model_name", None)
    checkpoint = settings.checkpoint_identifier() or model_name
    device = getattr(segmenter, "device", None) or resolve_device(settings.device)
    checkpoint_hash = getattr(segmenter, "checkpoint_sha256", None)
    if checkpoint_hash is None:
        local_checkpoint = (
            settings.deeplab_checkpoint
            if settings.backend == "deeplab"
            else settings.d2_weights if settings.backend == "detectron2" else None
        )
        if local_checkpoint is not None and local_checkpoint.is_file():
            checkpoint_hash = _sha256_file(local_checkpoint)
    taxonomy = getattr(segmenter, "taxonomy", None)
    if taxonomy is None:
        raise ValueError("The configured segmenter does not expose its taxonomy.")
    return {
        "backend": getattr(segmenter, "backend_name", settings.backend),
        "checkpoint": checkpoint,
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_hash,
        "class_space": getattr(segmenter, "class_space", None),
        "taxonomy": taxonomy.to_dict(),
        "taxonomy_source": (
            str(settings.taxonomy_path) if settings.taxonomy_path is not None else "built-in"
        ),
        "device": device,
    }
