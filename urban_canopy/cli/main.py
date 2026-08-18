#!/usr/bin/env python
"""
Command-line front-end for the canopy pipeline.

Examples
--------
  Local image, single view:
      tree-ai --image street.jpg --single-view --seg oneformer --device cpu

  Coordinates, multi-view:
      tree-ai --lat -23.678479 --lon -46.559621 --multi-view --seg oneformer

  Evaluation against Roboflow COCO ground truth:
      tree-ai evaluate --predictions predictions.json --annotations annotations.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from urban_canopy.log import configure_logging, get_logger

logger = get_logger(__name__)


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _print_view(result, *, prefix: str = "") -> None:
    coverage = result.coverage
    heading = result.capture.heading
    heading_str = f" heading={heading}" if heading is not None else ""
    print(
        f"{prefix}TREE COVERAGE {_fmt_pct(coverage.tree_coverage_pct)}"
        f"  (source={coverage.tree_source}){heading_str}"
    )
    if coverage.vegetation_coverage_pct is not None:
        print(f"{prefix}  vegetation coverage: {_fmt_pct(coverage.vegetation_coverage_pct)}")
    if result.instance_count is not None:
        print(f"{prefix}  instances: {result.instance_count}" f" (source={result.instance_source})")
    if result.quality_flags:
        print(f"{prefix}  flags: {', '.join(result.quality_flags)}")


def _print_aggregate(aggregate) -> None:
    stats = aggregate.tree_coverage
    print("\nMULTI-VIEW AGGREGATE (tree coverage ratio):")
    print(f"  views: {stats.n_valid_views}/{stats.n_views} valid")
    if stats.n_valid_views:
        print(f"  mean   = {100 * stats.mean:.2f}%")
        print(f"  median = {100 * stats.median:.2f}%")
        print(f"  p25    = {100 * stats.p25:.2f}%   p75 = {100 * stats.p75:.2f}%")
        print(f"  IQR    = {100 * stats.iqr:.2f} pp")
    counts = [c for c in aggregate.instance_counts if c is not None]
    if counts:
        print(f"  instances per view: {counts} (per view only; never summed across views)")
    for note in aggregate.notes:
        print(f"  note: {note}")


def _run_analyse(args, parser) -> int:
    from urban_canopy.cli._builder import (
        build_pipeline,
        resolve_device,
        viewplan_from_args,
    )
    from urban_canopy.core.config import build_manifest, set_seed
    from urban_canopy.core.results import results_to_rows, write_rows_csv
    from urban_canopy.io.artifacts import ArtifactConfig, write_json, write_view_artifacts

    has_location = args.address or (args.lat is not None and args.lon is not None)
    if not (args.image or has_location):
        parser.error("provide an address, --lat/--lon, or --image")
    if args.image and has_location:
        parser.error("--image and an address/--lat/--lon are mutually exclusive")
    if args.multi_view and args.image:
        parser.error("--multi-view needs Street View (an address or --lat/--lon)")

    set_seed(args.seed)

    # Resolve before loading any weights, so an impossible request fails in
    # milliseconds instead of after a multi-gigabyte download.
    device = resolve_device(args.device, parser)
    args.device = device

    started = time.time()
    pipe = build_pipeline(args, device, needs_streetview=bool(has_location))
    logger.info("Pipeline building took %.2f seconds", time.time() - started)

    # ---------------- run ----------------
    results = []
    multi = None
    if args.image:
        results = pipe.analyse_images([p.resolve() for p in args.image])
        if not results:
            print("No image could be analysed", file=sys.stderr)
            return 1
    elif args.multi_view:
        plan = viewplan_from_args(args)
        if args.address:
            multi = pipe.analyse_address_multiview(args.address, plan=plan)
        else:
            multi = pipe.analyse_multiview(args.lat, args.lon, plan=plan)
        results = multi.views
    else:
        if args.address:
            result = pipe.analyse_address(
                args.address, heading=args.heading, pitch=args.pitch, fov=args.fov, size=args.size
            )
        else:
            result = pipe.analyse_coords(
                args.lat,
                args.lon,
                heading=args.heading,
                pitch=args.pitch,
                fov=args.fov,
                size=args.size,
            )
        results = [result]
    logger.info("Analysis took %.2f seconds", time.time() - started)

    # ---------------- print ----------------
    for index, result in enumerate(results):
        prefix = f"[{index}] " if len(results) > 1 else ""
        _print_view(result, prefix=prefix)
    if multi is not None:
        _print_aggregate(multi.aggregate)

    # ---------------- artifacts / exports ----------------
    if args.save_artifacts:
        artifact_config = ArtifactConfig(outdir=args.outdir)
        for index, result in enumerate(results):
            write_view_artifacts(result, artifact_config, index=index if len(results) > 1 else None)
        print(f"\nArtifacts written to {args.outdir}")

    manifest = build_manifest(
        config=pipe.config,
        backend=args.seg,
        class_space=pipe.segmenter.class_space,
        taxonomy=pipe.segmenter.taxonomy,
        model_name=getattr(pipe.segmenter, "model_name", None),
        device=device,
    )

    if args.metrics_json:
        if multi is not None:
            payload = {"manifest": manifest, **multi.to_dict()}
        else:
            payload = {
                "manifest": manifest,
                "views": [r.to_dict() for r in results],
            }
        write_json(payload, args.metrics_json)
        print(f"Metrics written to {args.metrics_json}")

    if args.csv:
        write_rows_csv(results_to_rows(results), args.csv)
        print(f"CSV written to {args.csv}")

    if args.predictions_json:
        from urban_canopy.evaluation.predictions import build_predictions, write_predictions
        from urban_canopy.io.image_io import get_exclude_bottom_px

        exclude = (
            args.exclude_bottom_px
            if args.exclude_bottom_px is not None
            else get_exclude_bottom_px()
        )
        payload = build_predictions(results, manifest=manifest, exclude_bottom_px=exclude)
        write_predictions(args.predictions_json, payload)
        print(f"Predictions written to {args.predictions_json}")

    return 0


def _run_evaluate(args) -> int:
    from urban_canopy.evaluation.runner import evaluate_files
    from urban_canopy.io.artifacts import write_json

    report = evaluate_files(
        args.predictions,
        args.annotations,
        iou_threshold=args.iou_threshold,
        keep_per_image=not args.no_per_image,
    )
    payload = report.to_dict()

    micro = payload["semantic"]["micro"]
    print(f"Matched images: {payload['n_matched_images']}")
    print("\nSEMANTIC (pixel level, micro-averaged):")
    print(f"  IoU       = {micro['iou']:.4f}")
    print(f"  Dice/F1   = {micro['dice']:.4f}")
    print(f"  precision = {micro['precision']:.4f}")
    print(f"  recall    = {micro['recall']:.4f}")

    if payload["coverage"] is not None:
        cov = payload["coverage"]
        print("\nCOVERAGE INDICATOR (tree_coverage_pred vs tree_coverage_gt):")
        print(f"  MAE  = {cov['mae_pp']:.2f} pp")
        print(f"  RMSE = {cov['rmse_pp']:.2f} pp")
        print(f"  bias = {cov['bias_pp']:+.2f} pp")
        if cov.get("pearson_r") is not None:
            print(f"  Pearson r = {cov['pearson_r']:.3f} (complementary; not an error metric)")

    if payload["instances"] is not None:
        inst = payload["instances"]
        print(f"\nINSTANCES (IoU >= {inst['iou_threshold']:.2f}, ranked by {inst['ranked_by']}):")
        print(f"  TP={inst['tp']}  FP={inst['fp']}  FN={inst['fn']}")
        print(f"  precision = {inst['precision']:.4f}")
        print(f"  recall    = {inst['recall']:.4f}")
        print(f"  F1        = {inst['f1']:.4f}")
        print(f"  mean matched IoU = {inst['mean_matched_iou']:.4f}")
        if inst.get("AP50") is not None:
            print(f"  AP50    = {inst['AP50']:.4f}")
            print(f"  AP50:95 = {inst['AP50:95']:.4f}")
        elif inst.get("ap_unavailable_reason"):
            print(f"  AP: unavailable -- {inst['ap_unavailable_reason']}")
    elif payload["instances_skipped_reason"]:
        print(f"\nINSTANCES: skipped -- {payload['instances_skipped_reason']}")

    if args.report_json:
        write_json(payload, args.report_json)
        print(f"\nFull report written to {args.report_json}")
    return 0


def _run_validate(args) -> int:
    from urban_canopy.evaluation.coco import CocoDataset, DatasetValidationError

    try:
        dataset = CocoDataset.load(args.annotations)
        problems = dataset.validate(strict=args.strict)
    except DatasetValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    summary = dataset.summary()
    print(f"Images: {summary['n_images']}")
    print(f"Tree instances: {summary['n_tree_instances']}")
    print(f"Images without trees: {summary['n_images_without_trees']}")
    print(f"Categories: {summary['categories']}")
    if problems:
        print("\nProblems found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nDataset looks usable.")
    return 0


def main(argv: list[str] | None = None) -> int:
    from urban_canopy.cli._argparse import build_parser

    argv = list(sys.argv[1:] if argv is None else argv)
    # `tree-ai --image x.jpg` without a sub-command means `analyse`.
    if argv and argv[0] not in ("analyse", "evaluate", "validate-dataset", "-h", "--help"):
        argv = ["analyse"] + argv
    elif not argv:
        argv = ["analyse"]

    parser = build_parser()
    args = parser.parse_args(argv)

    # A Windows console is cp1252 by default, where an unencodable character
    # raises UnicodeEncodeError mid-print. Degrade to "?" instead of dying.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass

    configure_logging(
        debug=args.debug,
        level=args.log_level,
        fmt=args.log_format,
        log_file=args.log_file,
    )

    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "validate-dataset":
        return _run_validate(args)
    if hasattr(args, "outdir"):
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    return _run_analyse(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
