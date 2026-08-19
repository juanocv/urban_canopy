"""Turn parsed CLI arguments into a configured pipeline."""

from __future__ import annotations

from pathlib import Path

from urban_canopy.core.config import CanopyConfig
from urban_canopy.core.pipeline import CanopyPipeline
from urban_canopy.core.viewplan import ViewPlanConfig
from urban_canopy.log import get_logger
from urban_canopy.models.factory import (
    BACKEND_CLASS_SPACE,
    CHECKPOINT_DEFINES_CLASS_SPACE,
    build_segmenter,
)
from urban_canopy.models.taxonomy import infer_class_space, load_taxonomy
from urban_canopy.processing.refinement import RefinementConfig

logger = get_logger(__name__)


def resolve_device(requested: str, parser) -> str:
    """
    Turn ``--device`` into a concrete device, failing early and readably.

    Asking for CUDA on a CPU-only PyTorch build otherwise surfaces as an
    assertion from deep inside ``Module.to()``, after the weights have already
    been downloaded.
    """
    try:
        import torch
    except ModuleNotFoundError:
        if requested == "cuda":
            parser.error(
                "--device cuda needs PyTorch, which is not installed. "
                'Install the ML extra with `python -m pip install -e ".[ml]"`.'
            )
        return "cpu"

    available = torch.cuda.is_available()
    if requested == "auto":
        resolved = "cuda" if available else "cpu"
        logger.info("Device auto-selected: %s (torch %s)", resolved, torch.__version__)
        return resolved

    if requested == "cuda" and not available:
        parser.error(
            f"--device cuda was requested, but the installed PyTorch ({torch.__version__}) "
            "cannot use CUDA on this machine. Re-run with --device cpu or --device auto, "
            "or install a CUDA build from https://pytorch.org/get-started/locally/."
        )

    return requested


def build_segmenter_from_args(args, device: str):
    """Construct the requested backend with its taxonomy."""
    model_name = getattr(args, "seg_model", None)

    # OneFormer and Mask2Former publish weights for several datasets, so the
    # class space -- and therefore which classes count as trees -- follows the
    # checkpoint, not the backend. Reading it off a fixed per-backend table would
    # silently apply an ADE20K taxonomy to a Cityscapes checkpoint, which has no
    # tree class at all.
    if args.seg in CHECKPOINT_DEFINES_CLASS_SPACE and model_name:
        class_space = infer_class_space(model_name)
        logger.info("Class space inferred from %s: %s", model_name, class_space)
    else:
        class_space = BACKEND_CLASS_SPACE[args.seg]

    taxonomy = load_taxonomy(args.taxonomy, class_space=class_space)

    if args.seg in ("oneformer", "mask2former"):
        kwargs = {"device": device, "taxonomy": taxonomy}
        if model_name:
            kwargs["model_name"] = model_name
        if args.seg == "oneformer":
            # OneFormer takes the task per call; Mask2Former's is fixed by the
            # checkpoint and inferred from its name.
            kwargs["task"] = args.seg_task
        return build_segmenter(args.seg, **kwargs)

    if args.seg == "detectron2":
        if args.d2_config and args.d2_weights:
            return build_segmenter(
                "detectron2",
                config_yml=str(args.d2_config),
                weights_path=str(args.d2_weights),
                mode="instance",
                taxonomy=taxonomy,
                score_thresh=args.d2_score_thresh,
                device=device,
            )
        return build_segmenter(
            "detectron2",
            taxonomy=taxonomy,
            score_thresh=args.d2_score_thresh,
            device=device,
        )

    if args.seg == "deeplab":
        from urban_canopy.models.deeplab import get_settings, infer_model_name

        # A flag beats the standing default; the default beats nothing at all.
        defaults = get_settings()
        ckpt = args.ckpt if args.ckpt is not None else defaults.ckpt
        repo = args.deeplab_repo if args.deeplab_repo is not None else defaults.repo

        if ckpt is None:
            raise ValueError(
                "The deeplab backend needs a checkpoint. Pass --ckpt, or set it once "
                "for this machine with UC_DEEPLAB_CKPT (in the environment or .env) "
                "so later runs do not have to repeat it."
            )
        ckpt = Path(ckpt).expanduser()
        if not ckpt.exists():
            source = "--ckpt" if args.ckpt is not None else "UC_DEEPLAB_CKPT"
            raise FileNotFoundError(f"DeepLab checkpoint from {source} does not exist: {ckpt}")
        if args.ckpt is None:
            logger.info("DeepLab checkpoint from UC_DEEPLAB_CKPT: %s", ckpt)

        model_name = args.deeplab_model or defaults.model
        if model_name is None:
            model_name = infer_model_name(ckpt)
            if model_name is None:
                raise ValueError(
                    f"Could not infer the DeepLab architecture from {ckpt.name!r}. "
                    "Pass --deeplab-model explicitly (e.g. deeplabv3plus_mobilenet), "
                    "or set UC_DEEPLAB_MODEL."
                )
            logger.info("DeepLab architecture inferred from the checkpoint name: %s", model_name)

        return build_segmenter(
            "deeplab",
            ckpt_path=str(ckpt),
            model_name=model_name,
            taxonomy=taxonomy,
            allow_pickle=args.trust_checkpoint,
            device=device,
            repo_path=str(Path(repo).expanduser()) if repo else None,
        )

    raise ValueError(f"Unknown backend: {args.seg!r}")


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
        seed=args.seed,
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
