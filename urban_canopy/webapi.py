"""
Urban Canopy Web API.

Run it with::

    python -m pip install -e ".[api,ml]"
    uvicorn urban_canopy.webapi:app --host 127.0.0.1 --port 8000

Endpoints
---------
* ``POST /analyse/single`` -- one Street View frame -> coverage metrics
* ``POST /analyse/multi``  -- several headings -> per-view metrics + aggregate
* ``GET /ping``            -- liveness probe

Dataset evaluation is CLI-only (``tree-ai evaluate``): it reads local files and
produces large reports, neither of which belongs in a request/response cycle.

Concurrency
-----------
The endpoints are synchronous, so Starlette runs them in a worker thread pool
and several requests can overlap. The segmentation model behind them is neither
thread-safe nor cheap in VRAM, so model work is serialised through a semaphore
sized by ``UC_API_MAX_CONCURRENCY`` (default 1).

The API has no authentication and calls a paid Google API on every request --
keep it behind a proxy or bound to localhost.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from urban_canopy.log import configure_logging, get_logger

configure_logging(force=False)
logger = get_logger(__name__)

DEFAULT_BACKEND = os.getenv("UC_SEG_BACKEND", "oneformer")

# Serialises model work; see the module docstring.
MAX_CONCURRENCY = max(1, int(os.getenv("UC_API_MAX_CONCURRENCY", "1")))
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)

CORS_ORIGINS = [o.strip() for o in os.getenv("UC_API_CORS_ORIGINS", "*").split(",") if o.strip()]


@contextmanager
def _inference_slot():
    """Hold one of the model-inference slots for the duration of the block."""
    acquired = _inference_slots.acquire(timeout=float(os.getenv("UC_API_QUEUE_TIMEOUT_S", "300")))
    if not acquired:
        raise HTTPException(503, "Server busy: inference queue timed out")
    try:
        yield
    finally:
        _inference_slots.release()


class PipelineRegistry:
    """
    Lazily builds and caches one pipeline per configuration knob that changes
    behaviour (refinement on/off, vegetation proxy on/off). The segmenter and
    Street View client are shared across all of them: they are the expensive
    parts and they are configuration-independent.
    """

    def __init__(self, segmenter: Any, streetview: Any) -> None:
        self._segmenter = segmenter
        self._streetview = streetview
        self._pipes: dict[tuple[bool, bool], Any] = {}
        self._lock = threading.Lock()

    def get(self, *, refine: bool, allow_vegetation_proxy: bool):
        from urban_canopy.core.config import CanopyConfig
        from urban_canopy.core.pipeline import CanopyPipeline
        from urban_canopy.processing.refinement import RefinementConfig

        key = (refine, allow_vegetation_proxy)
        with self._lock:
            pipe = self._pipes.get(key)
            if pipe is None:
                pipe = CanopyPipeline(
                    segmenter=self._segmenter,
                    streetview=self._streetview,
                    config=CanopyConfig(
                        refinement=RefinementConfig(enabled=refine),
                        allow_vegetation_proxy=allow_vegetation_proxy,
                        keep_rgb=True,
                    ),
                )
                self._pipes[key] = pipe
            return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    import urban_canopy as uc

    logger.info(
        "Starting Urban Canopy API (backend=%s, max_concurrency=%s)",
        DEFAULT_BACKEND,
        MAX_CONCURRENCY,
    )
    segmenter = uc.build_segmenter(DEFAULT_BACKEND)
    streetview = uc.StreetViewClient()

    app.state.registry = PipelineRegistry(segmenter, streetview)
    app.state.registry.get(refine=True, allow_vegetation_proxy=False)
    logger.info("Urban Canopy API ready")
    yield


app = FastAPI(
    title="Urban-Canopy",
    version="0.1.0",
    description="Visible tree-canopy coverage from Google Street View imagery.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / response schemas                                         #
# ------------------------------------------------------------------ #
class SingleViewRequest(BaseModel):
    address: str | None = Field(
        default=None,
        json_schema_extra={"example": "Av. Paulista 1578, Sao Paulo"},
        description="Ignored if lat+lon are given",
    )
    lat: float | None = Field(None, description="Latitude (decimal deg)")
    lon: float | None = Field(None, description="Longitude (decimal deg)")
    heading: int = Field(0, ge=0, le=359)
    pitch: int = Field(0, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)
    size: str = Field("640x640", pattern=r"^\d{2,4}x\d{2,4}$")
    refine: bool = True
    allow_vegetation_proxy: bool = False
    return_overlays: bool = Field(
        False, description="Include base64 PNG overlays (RGB, tree overlay, mask)"
    )


class MultiViewRequest(BaseModel):
    address: str | None = None
    lat: float | None = None
    lon: float | None = None
    reference_heading: int = Field(0, ge=0, le=359)
    mode: str = Field("offsets", pattern="^(offsets|equiangular)$")
    offsets: list[int] = Field(default_factory=lambda: [0, 90, 180, 270])
    n_views: int = Field(4, ge=1, le=16)
    pitch: int = Field(0, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)
    size: str = Field("640x640", pattern=r"^\d{2,4}x\d{2,4}$")
    refine: bool = True
    allow_vegetation_proxy: bool = False


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _resolve_location(req) -> tuple[float, float]:
    if req.lat is not None and req.lon is not None:
        return req.lat, req.lon
    if req.address:
        try:
            return app.state.registry._streetview.geocode(req.address)
        except Exception as exc:
            raise HTTPException(422, f"Geocoding failed: {exc}") from exc
    raise HTTPException(422, "Either address or lat+lon is required")


def _overlays(result) -> dict[str, str]:
    import cv2
    import numpy as np

    from urban_canopy.io.image_io import mask_overlay_bgr, png_b64

    if result.rgb_image is None:
        return {}
    bgr = cv2.cvtColor(np.asarray(result.rgb_image), cv2.COLOR_RGB2BGR)
    return {
        "rgb_png_b64": png_b64(bgr),
        "overlay_tree_png_b64": png_b64(mask_overlay_bgr(result.rgb_image, result.refined_mask)),
        "mask_refined_png_b64": png_b64(result.refined_mask.astype("uint8") * 255),
    }


# ------------------------------------------------------------------ #
# Endpoints                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyse/single")
def analyse_single(req: SingleViewRequest) -> dict[str, Any]:
    lat, lon = _resolve_location(req)
    pipe = app.state.registry.get(
        refine=req.refine, allow_vegetation_proxy=req.allow_vegetation_proxy
    )
    with _inference_slot():
        try:
            result = pipe.analyse_coords(
                lat, lon, heading=req.heading, pitch=req.pitch, fov=req.fov, size=req.size
            )
        except Exception as exc:
            logger.exception("Single-view analysis failed")
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    payload = result.to_dict()
    if req.address:
        payload["capture"]["address"] = req.address
    if req.return_overlays:
        payload["overlays"] = _overlays(result)
    return payload


@app.post("/analyse/multi")
def analyse_multi(req: MultiViewRequest) -> dict[str, Any]:
    from urban_canopy.core.viewplan import ViewPlanConfig

    lat, lon = _resolve_location(req)
    plan = ViewPlanConfig(
        mode=req.mode,
        reference_heading=req.reference_heading,
        offsets=tuple(req.offsets),
        n_views=req.n_views,
        pitch=req.pitch,
        fov=req.fov,
        size=req.size,
    )
    pipe = app.state.registry.get(
        refine=req.refine, allow_vegetation_proxy=req.allow_vegetation_proxy
    )
    with _inference_slot():
        try:
            result = pipe.analyse_multiview(lat, lon, plan=plan, address=req.address)
        except Exception as exc:
            logger.exception("Multi-view analysis failed")
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return result.to_dict()
