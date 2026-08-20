"""Strict JSON conversion: undefined numeric metrics become JSON ``null``."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["json_dumps", "json_safe"]


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers and NumPy values with JSON values."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_dumps(
    payload: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False,
) -> str:
    """Serialize RFC-compliant JSON, rejecting any non-finite value left over."""
    return json.dumps(
        json_safe(payload),
        indent=indent,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
        default=str,
    )
