"""
Pipeline orchestration.

    acquire -> segment -> refine -> measure -> (aggregate) -> artifacts

The dependency-injected shape of the sidewalk pipeline is kept -- a segmenter, a
Street View client and a config go in, and the object is reusable across
requests -- because that is what makes the CLI, the Web API and the tests share
one code path. What is gone is everything downstream of the mask: no depth
estimator, no camera model, no metric geometry. The measurement here is a pixel
ratio, and pixel ratios need no scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from urban_canopy.io.image_io import decode_rgb, ensure_rgb_u8
from urban_canopy.io.streetview import ImageRequest, StreetViewClient
from urban_canopy.log import get_logger
from urban_canopy.models.base import HEURISTIC_INSTANCES, MODEL_INSTANCES
from urban_canopy.processing.aggregate import aggregate_views
from urban_canopy.processing.coverage import (
    TREE_SOURCE_PROXY,
    TREE_SOURCE_UNAVAILABLE,
    compute_coverage,
    resolve_tree_mask,
)
from urban_canopy.processing.instances import instances_from_components
from urban_canopy.processing.refinement import refine_canopy_mask

from .config import CanopyConfig
from .results import CaptureParams, MultiViewResult, QualityFlag, ViewFailure, ViewResult
from .viewplan import ViewPlanConfig, plan_headings

logger = get_logger(__name__)

__all__ = ["CanopyPipeline", "MultiViewAnalysisError"]


class MultiViewAnalysisError(RuntimeError):
    """Raised when a view plan produces fewer usable views than required."""

    def __init__(
        self,
        *,
        min_successful_views: int,
        planned_headings: Sequence[int],
        successful_headings: Sequence[int],
        failures: Sequence[ViewFailure],
    ) -> None:
        self.min_successful_views = int(min_successful_views)
        self.planned_headings = tuple(int(h) for h in planned_headings)
        self.successful_headings = tuple(int(h) for h in successful_headings)
        self.failures = tuple(failures)
        super().__init__(
            f"Multi-view analysis produced {len(self.successful_headings)} usable view(s), "
            f"but at least {self.min_successful_views} were required."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "min_successful_views": self.min_successful_views,
            "planned_headings": list(self.planned_headings),
            "successful_headings": list(self.successful_headings),
            "failures": [failure.to_dict() for failure in self.failures],
        }


class CanopyPipeline:
    """
    High-level canopy analysis.

    Parameters
    ----------
    segmenter
        Any object fulfilling :class:`~urban_canopy.models.base.Segmenter`.
    streetview
        A :class:`~urban_canopy.io.streetview.StreetViewClient`. Optional: local
        images need none. Create one and reuse it, for HTTP keep-alive and the
        on-disk cache.
    config
        A :class:`~urban_canopy.core.config.CanopyConfig`.
    """

    def __init__(
        self,
        *,
        segmenter: Any,
        streetview: StreetViewClient | None = None,
        config: CanopyConfig | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.sv = streetview
        self.config = config or CanopyConfig()

    # ------------------------------------------------------------------ #
    # Public entry points                                                #
    # ------------------------------------------------------------------ #
    def analyse_image(
        self,
        img_path: str | Path,
        *,
        capture: CaptureParams | None = None,
    ) -> ViewResult:
        """Analyse an image already on disk, skipping Street View entirely."""
        path = Path(img_path)
        rgb = decode_rgb(path)
        params = capture or CaptureParams(source="local", image_path=str(path))
        if params.image_path is None:
            params = CaptureParams(**{**params.to_dict(), "image_path": str(path)})
        return self.analyse_rgb_array(rgb, capture=params)

    def analyse_rgb_array(self, img_rgb: np.ndarray, *, capture: CaptureParams) -> ViewResult:
        """Analyse a validated RGB frame without guessing or changing its colour space."""
        return self._analyse(ensure_rgb_u8(img_rgb), capture)

    def analyse_array(self, img_rgb: np.ndarray, *, capture: CaptureParams) -> ViewResult:
        """Compatibility alias for :meth:`analyse_rgb_array`; the input is RGB."""
        return self.analyse_rgb_array(img_rgb, capture=capture)

    def analyse_coords(
        self,
        lat: float,
        lon: float,
        *,
        heading: int = 0,
        pitch: int = 0,
        fov: int = 90,
        size: str | None = None,
        record_metadata: bool = True,
    ) -> ViewResult:
        """Single-view analysis of one Street View heading."""
        client = self._require_streetview()
        request = ImageRequest(
            lat=lat,
            lon=lon,
            heading=heading,
            pitch=pitch,
            fov=fov,
            size=size or client.settings.default_size,
        )
        path = client.fetch(request)
        capture = self._capture_for(request, path, record_metadata=record_metadata)
        return self.analyse_image(path, capture=capture)

    def analyse_address(
        self,
        address: str,
        *,
        heading: int = 0,
        pitch: int = 0,
        fov: int = 90,
        size: str | None = None,
    ) -> ViewResult:
        """Single-view analysis of a geocoded address."""
        client = self._require_streetview()
        lat, lon = client.geocode(address)
        result = self.analyse_coords(lat, lon, heading=heading, pitch=pitch, fov=fov, size=size)
        result.capture = CaptureParams(**{**result.capture.to_dict(), "address": address})
        return result

    def analyse_multiview(
        self,
        lat: float,
        lon: float,
        *,
        plan: ViewPlanConfig | None = None,
        address: str | None = None,
        record_metadata: bool = True,
    ) -> MultiViewResult:
        """
        Analyse several headings of one location and aggregate them.

        A heading that fails to download or segment is logged and skipped; it
        still counts towards ``n_views`` in the aggregate so the reader can see
        how much of the plan actually landed.
        """
        client = self._require_streetview()
        config = plan or ViewPlanConfig()
        headings = plan_headings(config)
        if config.min_successful_views < 1:
            raise ValueError("min_successful_views must be at least 1.")
        if config.min_successful_views > len(headings):
            raise ValueError(
                f"min_successful_views={config.min_successful_views} exceeds the "
                f"{len(headings)} distinct planned heading(s)."
            )

        pano_meta: dict[str, Any] = {}
        if record_metadata:
            pano_meta = self._panorama_metadata(client, lat, lon)

        views: list[ViewResult] = []
        failures: list[ViewFailure] = []
        for heading in headings:
            request = ImageRequest(
                lat=lat,
                lon=lon,
                heading=heading,
                pitch=config.pitch,
                fov=config.fov,
                size=config.size,
            )
            try:
                path = client.fetch(request)
            except Exception as exc:
                logger.warning("Heading %s: Street View fetch failed: %s", heading, exc)
                failures.append(
                    ViewFailure(
                        heading=heading,
                        stage="fetch",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            capture = CaptureParams(
                source="streetview",
                lat=lat,
                lon=lon,
                heading=heading,
                pitch=config.pitch,
                fov=config.fov,
                size=config.size,
                address=address,
                pano_id=pano_meta.get("pano_id"),
                capture_date=pano_meta.get("date"),
                image_path=str(path),
            )
            try:
                views.append(self.analyse_image(path, capture=capture))
            except Exception as exc:
                logger.warning("Heading %s: analysis failed: %s", heading, exc)
                failures.append(
                    ViewFailure(
                        heading=heading,
                        stage="analysis",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

        if len(views) < config.min_successful_views:
            raise MultiViewAnalysisError(
                min_successful_views=config.min_successful_views,
                planned_headings=headings,
                successful_headings=[
                    int(view.capture.heading) for view in views if view.capture.heading is not None
                ],
                failures=failures,
            )

        # n_views reflects the plan, not the survivors: two usable frames out of
        # eight planned is a different result from two out of two.
        aggregate = aggregate_views(views, n_planned=len(headings))

        return MultiViewResult(
            views=views,
            aggregate=aggregate,
            lat=lat,
            lon=lon,
            address=address,
            plan={**config.to_dict(), "planned_headings": headings},
            failures=failures,
        )

    def analyse_address_multiview(
        self,
        address: str,
        *,
        plan: ViewPlanConfig | None = None,
    ) -> MultiViewResult:
        """Multi-view analysis of a geocoded address."""
        client = self._require_streetview()
        lat, lon = client.geocode(address)
        return self.analyse_multiview(lat, lon, plan=plan, address=address)

    def analyse_images(self, paths: Sequence[str | Path]) -> list[ViewResult]:
        """Analyse a batch of local images, skipping the ones that fail to load."""
        return list(self.iter_analyse_images(paths))

    def iter_analyse_images(self, paths: Sequence[str | Path]) -> Iterator[ViewResult]:
        """Yield local-image results one at a time, skipping unreadable inputs."""
        for path in paths:
            try:
                yield self.analyse_image(path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

    # ------------------------------------------------------------------ #
    # Core implementation                                                #
    # ------------------------------------------------------------------ #
    def _analyse(self, img_rgb: np.ndarray, capture: CaptureParams) -> ViewResult:
        config = self.config
        rgb = ensure_rgb_u8(img_rgb)
        height, width = rgb.shape[:2]

        output = self.segmenter.segment(rgb)
        output.validate((height, width))

        # Street View imagery is measured exactly as delivered. In particular,
        # its attribution/watermark remains part of the frame and denominator.
        valid = np.ones((height, width), dtype=bool)

        tree_raw, tree_source = resolve_tree_mask(
            output, allow_vegetation_proxy=config.allow_vegetation_proxy
        )
        vegetation = output.union(output.taxonomy.vegetation_groups)

        flags: list[str] = []
        if tree_source == TREE_SOURCE_UNAVAILABLE:
            flags.append(QualityFlag.TREE_UNAVAILABLE)
        elif tree_source == TREE_SOURCE_PROXY:
            flags.append(QualityFlag.TREE_FROM_PROXY)

        base = np.zeros((height, width), dtype=np.uint8) if tree_raw is None else tree_raw
        refined, refinement = refine_canopy_mask(base, config.refinement)
        if not config.refinement.enabled:
            flags.append(QualityFlag.REFINEMENT_DISABLED)
        if refinement.growth_guard_triggered:
            flags.append(QualityFlag.GROWTH_GUARD)

        coverage = compute_coverage(
            tree_mask=None if tree_raw is None else refined,
            vegetation_mask=vegetation,
            valid_mask=valid,
            tree_source=tree_source,
            group_masks=output.group_masks,
        )

        if coverage.tree_pixels == 0 and tree_raw is not None:
            flags.append(QualityFlag.EMPTY_TREE_MASK)
        if coverage.tree_coverage_ratio is not None and coverage.tree_coverage_ratio > 0.90:
            flags.append(QualityFlag.NEAR_TOTAL_COVERAGE)

        instances, instance_source = self._resolve_instances(
            output, refined, tree_mask_available=tree_raw is not None
        )
        if instance_source == HEURISTIC_INSTANCES:
            flags.append(QualityFlag.HEURISTIC_INSTANCES)

        return ViewResult(
            coverage=coverage,
            capture=capture,
            backend=output.backend,
            class_space=output.class_space,
            raw_mask=np.asarray(base).astype(np.uint8),
            refined_mask=np.asarray(refined).astype(np.uint8),
            refinement=refinement,
            vegetation_mask=None if vegetation is None else vegetation.astype(np.uint8),
            rgb_image=rgb if config.keep_rgb else None,
            instances=instances,
            instances_supported=bool(output.supports_tree_instances),
            instance_source=instance_source,
            quality_flags=tuple(dict.fromkeys(flags)),
            backend_notes=tuple(output.notes),
        )

    def _resolve_instances(self, output, refined_mask: np.ndarray, *, tree_mask_available: bool):
        mode = self.config.instance_mode
        if mode == "none" or not tree_mask_available:
            return None, None

        if output.supports_tree_instances and output.instances is not None:
            return list(output.instances), MODEL_INSTANCES

        if mode == "heuristic":
            return (
                instances_from_components(
                    refined_mask, min_area_px=self.config.heuristic_min_area_px
                ),
                HEURISTIC_INSTANCES,
            )

        # mode == "auto" and the backend cannot individualise trees: report
        # nothing rather than something that looks like a count.
        return None, None

    def _require_streetview(self) -> StreetViewClient:
        if self.sv is None:
            raise RuntimeError(
                "This pipeline was built without a StreetViewClient; pass one to "
                "analyse coordinates or addresses."
            )
        return self.sv

    def _capture_for(
        self,
        request: ImageRequest,
        path: Path,
        *,
        record_metadata: bool,
    ) -> CaptureParams:
        meta: dict[str, Any] = {}
        if record_metadata:
            meta = self._panorama_metadata(self._require_streetview(), request.lat, request.lon)
        return CaptureParams(
            source="streetview",
            lat=request.lat,
            lon=request.lon,
            heading=request.heading,
            pitch=request.pitch,
            fov=request.fov,
            size=request.size,
            pano_id=meta.get("pano_id"),
            capture_date=meta.get("date"),
            image_path=str(path),
        )

    @staticmethod
    def _panorama_metadata(client: StreetViewClient, lat: float, lon: float) -> dict[str, Any]:
        """
        Panorama id and capture date, or an empty dict.

        Google does not bill this endpoint, but it can still fail, and a missing
        pano id is no reason to abandon an otherwise good frame.
        """
        try:
            meta = client.metadata(lat, lon)
        except Exception as exc:
            logger.debug("Street View metadata unavailable for %s,%s: %s", lat, lon, exc)
            return {}
        return {"pano_id": meta.get("pano_id"), "date": meta.get("date")}
