"""Turn parsed CLI arguments into a configured pipeline."""

from __future__ import annotations

from urban_canopy.core.config import CanopyConfig
from urban_canopy.core.pipeline import CanopyPipeline
from urban_canopy.core.viewplan import ViewPlanConfig
from urban_canopy.models.backend_settings import (
    BackendSettings,
    build_segmenter_from_settings,
    resolve_device as resolve_backend_device,
)
from urban_canopy.models.factory import build_segmenter
from urban_canopy.processing.refinement import RefinementConfig


def resolve_device(requested: str, parser) -> str:
    """
    Turn ``--device`` into a concrete device, failing early and readably.

    Asking for CUDA on a CPU-only PyTorch build otherwise surfaces as an
    assertion from deep inside ``Module.to()``, after the weights have already
    been downloaded.
    """
    try:
        return resolve_backend_device(requested)
    except ValueError as exc:
        parser.error(str(exc))


def build_segmenter_from_args(args, device: str):
    """Construct the requested backend with its taxonomy."""
    settings = BackendSettings.from_cli_args(args, device=device)
    return build_segmenter_from_settings(settings, builder=build_segmenter)


def config_from_args(args) -> CanopyConfig:
    return CanopyConfig(
        refinement=RefinementConfig(
            enabled=args.refine,
            min_component_area_px=args.min_component_px,
            max_hole_area_px=args.max_hole_px,
            open_kernel_px=args.open_px,
            close_kernel_px=args.close_px,
        ),
        allow_vegetation_proxy=args.allow_vegetation_proxy,
        instance_mode=args.instances,
        keep_rgb=bool(args.save_artifacts),
        seed=args.seed,
        deterministic=args.deterministic,
    )


def viewplan_from_args(args) -> ViewPlanConfig:
    reference = args.reference_heading if args.reference_heading is not None else args.heading

    if args.view_mode == "fixed":
        if not args.headings:
            raise ValueError("--view-mode fixed needs --headings (comma-separated degrees).")
        headings = tuple(int(h.strip()) for h in args.headings.split(",") if h.strip())
        return ViewPlanConfig(
            mode="fixed",
            headings=headings,
            pitch=args.pitch,
            fov=args.fov,
            size=args.size,
            min_successful_views=args.min_successful_views,
        )

    if args.view_mode == "equiangular":
        return ViewPlanConfig(
            mode="equiangular",
            reference_heading=reference,
            n_views=args.n_views,
            pitch=args.pitch,
            fov=args.fov,
            size=args.size,
            min_successful_views=args.min_successful_views,
        )

    offsets = tuple(int(o.strip()) for o in args.offsets.split(",") if o.strip())
    return ViewPlanConfig(
        mode="offsets",
        reference_heading=reference,
        offsets=offsets,
        pitch=args.pitch,
        fov=args.fov,
        size=args.size,
        min_successful_views=args.min_successful_views,
    )


def build_pipeline(args, device: str, *, needs_streetview: bool) -> CanopyPipeline:
    import urban_canopy as uc

    segmenter = build_segmenter_from_args(args, device)
    streetview = uc.StreetViewClient() if needs_streetview else None
    return CanopyPipeline(
        segmenter=segmenter,
        streetview=streetview,
        config=config_from_args(args),
    )
