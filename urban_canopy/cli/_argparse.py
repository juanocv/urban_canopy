"""CLI argument definitions for ``tree-ai``."""

from __future__ import annotations

import argparse
from pathlib import Path

#: Stored when an export flag is given without a path; the export then lands in
#: the run directory instead of the working directory.
DEFAULT_EXPORT = "<run-dir>"


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Defaults to UC_LOG_LEVEL or INFO.",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["text", "json"],
        help="Logging format. Defaults to UC_LOG_FORMAT or text.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file path for logs. Can also be set with UC_LOG_FILE.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose console output")


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seg",
        default="oneformer",
        choices=["oneformer", "detectron2", "deeplab"],
        help="Segmentation backend (default: oneformer)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. 'auto' (default) uses CUDA when the installed "
        "PyTorch build can, otherwise CPU. Naming 'cuda' explicitly fails "
        "loudly when it cannot.",
    )
    parser.add_argument(
        "--seg-task",
        default="semantic",
        choices=["semantic", "panoptic"],
        help="OneFormer task. Semantic (default) avoids panoptic post-processing "
        "thresholds influencing the coverage ratio.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="JSON taxonomy overriding the built-in class mapping for the backend",
    )
    parser.add_argument(
        "--allow-vegetation-proxy",
        action="store_true",
        help="Let a wider vegetation class stand in for trees on backends whose "
        "class space has no tree class (DeepLab/Cityscapes). Results are "
        "flagged tree_from_vegetation_proxy.",
    )
    # Detectron2 custom instance model
    parser.add_argument("--d2-config", type=Path, help="Detectron2 config .yaml (instance mode)")
    parser.add_argument("--d2-weights", type=Path, help="Detectron2 weights .pth (instance mode)")
    parser.add_argument(
        "--d2-score-thresh",
        type=float,
        default=0.50,
        help="Detectron2 score threshold (default 0.50)",
    )
    # DeepLab. All three have standing defaults (UC_DEEPLAB_CKPT / _REPO / _MODEL,
    # settable in .env), so on a configured machine none of them need repeating.
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to DeepLab checkpoint (.pth). Defaults to UC_DEEPLAB_CKPT.",
    )
    parser.add_argument(
        "--deeplab-repo",
        type=Path,
        default=None,
        help="Path to a VainF DeepLabV3Plus-Pytorch checkout (the folder holding "
        "network/). Upstream ships no setup.py, so it cannot be pip-installed; "
        "this puts it on sys.path for the run instead. Defaults to UC_DEEPLAB_REPO.",
    )
    parser.add_argument(
        "--deeplab-model",
        default=None,
        help="Entry point inside network.modeling (e.g. deeplabv3plus_mobilenet). "
        "Defaults to UC_DEEPLAB_MODEL, then to the architecture named in the "
        "checkpoint filename.",
    )


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        help="Apply conservative mask refinement (default)",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Use the raw segmenter mask, for comparison runs",
    )
    parser.set_defaults(refine=True)
    parser.add_argument(
        "--min-component-px",
        type=int,
        default=64,
        help="Refinement: drop connected components smaller than this (default 64)",
    )
    parser.add_argument(
        "--max-hole-px",
        type=int,
        default=64,
        help="Refinement: fill enclosed holes strictly smaller than this (default 64)",
    )
    parser.add_argument(
        "--open-px",
        type=int,
        default=0,
        help="Refinement: morphological opening kernel in px (0 = off, default)",
    )
    parser.add_argument(
        "--close-px",
        type=int,
        default=0,
        help="Refinement: morphological closing kernel in px (0 = off, default)",
    )
    parser.add_argument(
        "--instances",
        default="auto",
        choices=["auto", "none", "heuristic"],
        help="Instance reporting. 'auto' (default) keeps model instances when the "
        "backend produces them and nothing otherwise; 'heuristic' splits the "
        "semantic mask into connected components and FLAGS them as a heuristic, "
        "not a tree count.",
    )
    parser.add_argument(
        "--exclude-bottom-px",
        type=int,
        default=None,
        help="Exclude this bottom strip (Street View watermark) from the "
        "valid-pixel denominator. Defaults to UC_IMG_EXCLUDE_BOTTOM_PX.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed recorded in the manifest")


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts_out"),
        help="Root folder holding run directories (default: artifacts_out). Each "
        "run writes to <outdir>/<timestamp>_<backend>/ so runs never overwrite "
        "one another.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name this run's directory instead of <timestamp>_<backend>",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write the full audit bundle for this run: per-view images (RGB, raw "
        "mask, refined mask, overlays, metrics.json) plus run.json, views.csv and "
        "predictions.json. The three export flags below are for asking for one of "
        "them on its own.",
    )
    parser.add_argument(
        "--metrics-json",
        nargs="?",
        const=DEFAULT_EXPORT,
        default=None,
        help="Write run metrics as JSON (implied by --save-artifacts). Without a "
        "path, writes run.json inside the run directory.",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const=DEFAULT_EXPORT,
        default=None,
        help="Write per-view rows as CSV (implied by --save-artifacts). Without a "
        "path, writes views.csv inside the run directory.",
    )
    parser.add_argument(
        "--predictions-json",
        nargs="?",
        const=DEFAULT_EXPORT,
        default=None,
        help="Write a predictions file with RLE masks for `tree-ai evaluate` "
        "(implied by --save-artifacts). Without a path, writes predictions.json "
        "inside the run directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tree-ai",
        description="Visible tree-canopy coverage from Street View or local imagery.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---------------- analyse (default) ----------------
    analyse = subparsers.add_parser(
        "analyse",
        help="Analyse imagery (default when no sub-command is given)",
        description="Analyse Street View or local imagery for visible tree coverage.",
    )
    analyse.add_argument("address", nargs="?", help="Free-form address string")
    analyse.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Analyse an existing image (repeatable for a batch)",
    )
    analyse.add_argument("--lat", type=float, default=None, help="Latitude (decimal degrees)")
    analyse.add_argument("--lon", type=float, default=None, help="Longitude (decimal degrees)")

    view = analyse.add_mutually_exclusive_group()
    view.add_argument(
        "--single-view",
        dest="multi_view",
        action="store_false",
        help="One frame at --heading (default)",
    )
    view.add_argument(
        "--multi-view",
        dest="multi_view",
        action="store_true",
        help="Sample several headings and aggregate",
    )
    analyse.set_defaults(multi_view=False)

    analyse.add_argument("--heading", type=int, default=0, help="Heading in degrees (0-359)")
    analyse.add_argument("--pitch", type=int, default=0, help="Camera pitch (-90 to 90)")
    analyse.add_argument("--fov", type=int, default=90, help="Field of view (10-120)")
    analyse.add_argument("--size", default="640x640", help="Street View image size (WxH)")

    analyse.add_argument(
        "--view-mode",
        default="offsets",
        choices=["offsets", "equiangular", "fixed"],
        help="Multi-view heading strategy (default: offsets)",
    )
    analyse.add_argument(
        "--reference-heading",
        type=int,
        default=None,
        help="Reference heading for offsets/equiangular modes. Defaults to "
        "--heading. Deterministic: no mask-driven street-center search.",
    )
    analyse.add_argument(
        "--offsets",
        default="0,90,180,270",
        help="Comma-separated offsets from the reference heading (offsets mode)",
    )
    analyse.add_argument(
        "--n-views",
        type=int,
        default=4,
        help="Number of equiangular views (equiangular mode, default 4)",
    )
    analyse.add_argument(
        "--headings",
        default=None,
        help="Comma-separated absolute headings (fixed mode)",
    )

    _add_backend_arguments(analyse)
    _add_processing_arguments(analyse)
    _add_output_arguments(analyse)
    _add_logging_arguments(analyse)

    # ---------------- evaluate ----------------
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a predictions file against COCO ground truth",
        description="Semantic, instance and coverage-error evaluation against a "
        "COCO Instance Segmentation export (e.g. from Roboflow).",
    )
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSON written by `tree-ai analyse --predictions-json`",
    )
    evaluate.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="COCO instance segmentation JSON (ground truth)",
    )
    evaluate.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for instance matching (default 0.50)",
    )
    evaluate.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the full evaluation report to this path",
    )
    evaluate.add_argument(
        "--no-per-image",
        action="store_true",
        help="Omit per-image rows from the report",
    )
    _add_logging_arguments(evaluate)

    # ---------------- validate-dataset ----------------
    validate = subparsers.add_parser(
        "validate-dataset",
        help="Check a COCO annotations file without evaluating anything",
    )
    validate.add_argument("--annotations", type=Path, required=True)
    validate.add_argument(
        "--strict", action="store_true", help="Exit non-zero on the first problem"
    )
    _add_logging_arguments(validate)

    return parser
