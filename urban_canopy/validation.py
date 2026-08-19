"""Dependency-free runtime validation shared by library, CLI and API."""

from __future__ import annotations

import math
import re
from typing import Any, TypeVar

__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MAX_MORPH_KERNEL_PX",
    "validate_bool",
    "validate_choice",
    "validate_finite_float",
    "validate_fov",
    "validate_heading",
    "validate_image_size",
    "validate_int_range",
    "validate_latitude",
    "validate_longitude",
    "validate_pitch",
    "validate_probability",
]

MAX_IMAGE_DIMENSION = 4096
MAX_MORPH_KERNEL_PX = 255
_SIZE_RE = re.compile(r"^([1-9]\d*)x([1-9]\d*)$")
T = TypeVar("T")


def validate_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean; got {value!r}.")
    return value


def validate_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number; got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number; got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return number


def validate_probability(value: Any, *, name: str) -> float:
    number = validate_finite_float(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {number}.")
    return number


def validate_int_range(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}.") from exc
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}.") from exc
    if not math.isfinite(numeric_value) or numeric_value != number:
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; got {number}.")
    return number


def validate_choice(value: T, *, name: str, choices: tuple[T, ...]) -> T:
    if value not in choices:
        rendered = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {rendered}; got {value!r}.")
    return value


def validate_latitude(value: Any) -> float:
    number = validate_finite_float(value, name="latitude")
    if not -90.0 <= number <= 90.0:
        raise ValueError(f"latitude must be between -90 and 90; got {number}.")
    return number


def validate_longitude(value: Any) -> float:
    number = validate_finite_float(value, name="longitude")
    if not -180.0 <= number <= 180.0:
        raise ValueError(f"longitude must be between -180 and 180; got {number}.")
    return number


def validate_heading(value: Any) -> int:
    return validate_int_range(value, name="heading", minimum=0, maximum=359)


def validate_pitch(value: Any) -> int:
    return validate_int_range(value, name="pitch", minimum=-90, maximum=90)


def validate_fov(value: Any) -> int:
    return validate_int_range(value, name="fov", minimum=10, maximum=120)


def validate_image_size(value: Any) -> str:
    text = str(value).strip().lower()
    match = _SIZE_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"size must use WIDTHxHEIGHT with positive integers; got {value!r}.")
    width, height = (int(part) for part in match.groups())
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"size dimensions must not exceed {MAX_IMAGE_DIMENSION}px; got {text!r}.")
    return f"{width}x{height}"
