"""
Urban Canopy -- visible tree-canopy coverage from Street View imagery
=====================================================================

Main entry points
-----------------
* :class:`~urban_canopy.core.pipeline.CanopyPipeline` -- high-level, one-call API
* :func:`~urban_canopy.models.factory.build_segmenter` -- backend factory
* :class:`~urban_canopy.io.streetview.StreetViewClient` -- Google Street View I/O
* :func:`~urban_canopy.evaluation.runner.evaluate_files` -- ground-truth evaluation
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from urban_canopy.core.config import CanopyConfig
    from urban_canopy.core.pipeline import CanopyPipeline
    from urban_canopy.core.viewplan import ViewPlanConfig
    from urban_canopy.io.streetview import StreetViewClient
    from urban_canopy.models.factory import build_segmenter
    from urban_canopy.processing.refinement import RefinementConfig

__all__ = [
    "CanopyPipeline",
    "CanopyConfig",
    "ViewPlanConfig",
    "RefinementConfig",
    "build_segmenter",
    "StreetViewClient",
    "Coordinate",
    "haversine",
]

# ------------------------------------------------------------
# 1)  Re-export light helpers immediately
# ------------------------------------------------------------
from .io.geo import Coordinate, haversine  # noqa: E402

# ------------------------------------------------------------
# 2)  Lazy re-exports for anything that can pull in cv2/torch
# ------------------------------------------------------------
_lazy_map: dict[str, str] = {
    "CanopyPipeline": "urban_canopy.core.pipeline",
    "CanopyConfig": "urban_canopy.core.config",
    "ViewPlanConfig": "urban_canopy.core.viewplan",
    "RefinementConfig": "urban_canopy.processing.refinement",
    "build_segmenter": "urban_canopy.models.factory",
    "StreetViewClient": "urban_canopy.io.streetview",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _lazy_map:
        mod: ModuleType = import_module(_lazy_map[name])
        obj = getattr(mod, name)
        globals()[name] = obj  # cache for next time
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
