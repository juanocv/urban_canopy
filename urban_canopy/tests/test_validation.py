"""Runtime configuration validation is shared across public entry points."""

import math

import numpy as np
import pytest

from urban_canopy.cli._argparse import build_parser
from urban_canopy.core.config import CanopyConfig
from urban_canopy.io.streetview import ImageRequest
from urban_canopy.processing.refinement import RefinementConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_component_area_px": -1},
        {"max_hole_area_px": -1},
        {"min_component_area_frac": -0.01},
        {"min_component_area_frac": 1.01},
        {"max_area_growth_frac": float("nan")},
        {"open_kernel_px": 256},
        {"close_kernel_px": -1},
        {"open_kernel_px": np.float32(1.5)},
        {"enabled": "yes"},
    ],
)
def test_refinement_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RefinementConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instance_mode": "invented"},
        {"heuristic_min_area_px": -1},
        {"seed": -1},
        {"seed": 2**32},
        {"keep_rgb": 1},
        {"deterministic": "yes"},
    ],
)
def test_canopy_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CanopyConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lat": math.nan, "lon": 0},
        {"lat": 91, "lon": 0},
        {"lat": 0, "lon": 181},
        {"lat": 0, "lon": 0, "heading": 360},
        {"lat": 0, "lon": 0, "pitch": -91},
        {"lat": 0, "lon": 0, "fov": 121},
        {"lat": 0, "lon": 0, "size": "640*640"},
        {"lat": 0, "lon": 0, "size": "5000x640"},
    ],
)
def test_image_request_rejects_invalid_capture_parameters(kwargs):
    with pytest.raises(ValueError):
        ImageRequest(**kwargs)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--lat", "nan", "--lon", "0"],
        ["--lat", "91", "--lon", "0"],
        ["--heading", "360"],
        ["--pitch", "-91"],
        ["--fov", "9"],
        ["--size", "640*640"],
        ["--open-px", "256"],
        ["--min-component-px", "-1"],
        ["--d2-score-thresh", "1.1"],
    ],
)
def test_cli_rejects_invalid_analysis_values(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyse", "--image", "x.jpg", *arguments])


@pytest.mark.parametrize("value", ["nan", "-0.1", "1.1"])
def test_cli_rejects_invalid_iou_threshold(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "evaluate",
                "--predictions",
                "p.json",
                "--annotations",
                "a.json",
                "--iou-threshold",
                value,
            ]
        )
