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
from contextlib import suppress
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
    for note in aggregate.notes:
        print(f"  note: {note}")


def _export_path(value, default: Path) -> Path:
    """Resolve an export flag to a path, defaulting inside the run directory."""
    from urban_canopy.cli._argparse import DEFAULT_EXPORT

    return default if value == DEFAULT_EXPORT else Path(value)


def _run_analyse(args, parser) -> int:
    from urban_canopy.cli._builder import (
        build_pipeline,
        resolve_device,
        viewplan_from_args,
    )
    from urban_canopy.core.config import build_manifest, set_seed
    from urban_canopy.core.results import results_to_rows, write_rows_csv
    from urban_canopy.io.artifacts import (
        ArtifactConfig,
        RunLayout,
        make_run_id,
        write_json,
        write_view_artifacts,
    )

    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be provided together")
    has_location = args.address or (args.lat is not None and args.lon is not None)
    if not (args.image or has_location):
        parser.error("provide an address, --lat/--lon, or --image")
    if args.image and has_location:
        parser.error("--image and an address/--lat/--lon are mutually exclusive")
    if args.multi_view and args.image:
        parser.error("--multi-view needs Street View (an address or --lat/--lon)")

    plan = None
    if args.multi_view:
        from urban_canopy.core.viewplan import plan_headings

        try:
            plan = viewplan_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        if plan.min_successful_views < 1:
            parser.error("--min-successful-views must be at least 1")
        planned_count = len(plan_headings(plan))
        if plan.min_successful_views > planned_count:
            parser.error(
                "--min-successful-views cannot exceed the number of distinct "
                f"planned headings ({planned_count})"
            )

    set_seed(args.seed, deterministic=args.deterministic)

    # Resolve before loading any weights, so an impossible request fails in
    # milliseconds instead of after a multi-gigabyte download.
    device = resolve_device(args.device, parser)
    args.device = device

    started = time.time()
    pipe = build_pipeline(args, device, needs_streetview=bool(has_location))
    logger.info("Pipeline building took %.2f seconds", time.time() - started)

    from urban_canopy.cli._argparse import DEFAULT_EXPORT

    metrics_target = args.metrics_json
    csv_target = args.csv
    predictions_target = args.predictions_json
    if args.save_artifacts:
        metrics_target = metrics_target if metrics_target is not None else DEFAULT_EXPORT
        csv_target = csv_target if csv_target is not None else DEFAULT_EXPORT
        predictions_target = (
            predictions_target if predictions_target is not None else DEFAULT_EXPORT
        )
    wants_output = bool(args.save_artifacts or metrics_target or csv_target or predictions_target)
    layout = None

    # ---------------- run ----------------
    results = []
    multi = None
    if args.image:
        artifact_config = None
        for result in pipe.iter_analyse_images([p.resolve() for p in args.image]):
            if args.save_artifacts:
                if layout is None:
                    layout = RunLayout.create(
                        args.outdir,
                        make_run_id(pipe.segmenter.backend_name, name=args.run_name),
                    )
                    print(f"\nRun directory: {layout.root}")
                    artifact_config = ArtifactConfig(outdir=layout.views)
                assert artifact_config is not None
                write_view_artifacts(result, artifact_config, index=len(results))
                # RGB is the dominant per-view allocation. Once its artifacts
                # exist, keeping it until the whole batch finishes serves no
                # consumer; masks and compact metrics remain available for the
                # run-level exports below.
                result.rgb_image = None
            results.append(result)
        if not results:
            print("No image could be analysed", file=sys.stderr)
            return 1
    elif args.multi_view:
        from urban_canopy.core.pipeline import MultiViewAnalysisError

        assert plan is not None
        try:
            if args.address:
                multi = pipe.analyse_address_multiview(args.address, plan=plan)
            else:
                multi = pipe.analyse_multiview(args.lat, args.lon, plan=plan)
        except MultiViewAnalysisError as exc:
            print(f"Multi-view analysis failed: {exc}", file=sys.stderr)
            for failure in exc.failures:
                print(
                    f"  heading={failure.heading} stage={failure.stage}: "
                    f"{failure.error_type}: {failure.message}",
                    file=sys.stderr,
                )
            return 2
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

    manifest = build_manifest(
        config=pipe.config,
        backend=pipe.segmenter.backend_name,
        class_space=pipe.segmenter.class_space,
        taxonomy=pipe.segmenter.taxonomy,
        model_name=getattr(pipe.segmenter, "model_name", None),
        model_sha256=getattr(pipe.segmenter, "checkpoint_sha256", None),
        device=device,
    )

    # ---------------- artifacts / exports ----------------
    # The run directory is created only when the run actually writes something,
    # so a plain analysis leaves no empty folders behind.
    if not wants_output:
        return 0

    if layout is None:
        layout = RunLayout.create(
            args.outdir, make_run_id(pipe.segmenter.backend_name, name=args.run_name)
        )
        print(f"\nRun directory: {layout.root}")

    if args.save_artifacts and not args.image:
        artifact_config = ArtifactConfig(outdir=layout.views)
        for index, result in enumerate(results):
            write_view_artifacts(result, artifact_config, index=index)
            result.rgb_image = None
    if args.save_artifacts:
        print(f"  views/      {len(results)} view folder(s)")

    if metrics_target is not None:
        if multi is not None:
            payload = {"manifest": manifest, **multi.to_dict()}
        else:
            payload = {"manifest": manifest, "views": [r.to_dict() for r in results]}
        target = _export_path(metrics_target, layout.run_json)
        write_json(payload, target)
        print(f"  {target.name:<12} run metrics")

    if csv_target is not None:
        target = _export_path(csv_target, layout.views_csv)
        write_rows_csv(results_to_rows(results), target)
        print(f"  {target.name:<12} per-view rows")

    if predictions_target is not None:
        from urban_canopy.evaluation.predictions import build_predictions, write_predictions

        payload = build_predictions(results, manifest=manifest)
        target = _export_path(predictions_target, layout.predictions_json)
        write_predictions(target, payload)
        print(f"  {target.name:<12} for `tree-ai evaluate`")

    return 0


def _run_evaluate(args) -> int:
    from urban_canopy.evaluation.runner import evaluate_files
    from urban_canopy.io.artifacts import write_json

    report = evaluate_files(
        args.predictions,
        args.annotations,
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
            with suppress(ValueError, OSError):  # pragma: no cover - exotic streams
                reconfigure(errors="replace")

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
    # The output directory is created by the run that writes into it, not here:
    # creating it up front left an empty artifacts_out/ behind on every analysis
    # that was never asked to save anything.
    return _run_analyse(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
