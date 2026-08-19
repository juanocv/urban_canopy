"""
Light-weight geographic helpers (no external deps).

The module is **pure**: it never prints, reads, or writes files, so every
function here is unit-testable in microseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "EARTH_RADIUS_M",
    "Coordinate",
    "haversine",
    "bearing",
    "destination",
]

EARTH_RADIUS_M: float = 6_371_000.0  # IUGG mean Earth radius (metres)


@dataclass(frozen=True, slots=True)
class Coordinate:
    """Immutable (lat, lon) pair in *decimal degrees*."""

    lat: float
    lon: float

    def __or__(self, other: Coordinate) -> float:
        """``c1 | c2`` -> great-circle distance in metres."""
        return haversine(self, other)


def haversine(a: Coordinate | tuple[float, float], b: Coordinate | tuple[float, float]) -> float:
    """
    Great-circle distance in **metres** (error < 1 m up to 200 km).

    >>> sao = Coordinate(-23.5505, -46.6333)
    >>> rio = Coordinate(-22.9068, -43.1729)
    >>> round(haversine(sao, rio) / 1000)
    357
    """
    lat1, lon1 = a if isinstance(a, tuple) else (a.lat, a.lon)
    lat2, lon2 = b if isinstance(b, tuple) else (b.lat, b.lon)

    phi1, phi2 = map(math.radians, (lat1, lat2))
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    s = math.sin(d_phi * 0.5)
    c = math.sin(d_lambda * 0.5)
    h = s * s + math.cos(phi1) * math.cos(phi2) * c * c
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def bearing(a: Coordinate, b: Coordinate) -> float:
    """Initial course *a -> b* in degrees (0 = North, clockwise)."""
    phi1, phi2 = map(math.radians, (a.lat, b.lat))
    d_lambda = math.radians(b.lon - a.lon)

    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def destination(origin: Coordinate, distance_m: float, heading_deg: float) -> Coordinate:
    """
    Point reached after travelling *distance_m* at *heading_deg* from *origin*.

    Used to walk a street segment at a fixed spacing before sampling views at
    each stop.
    """
    d = distance_m / EARTH_RADIUS_M
    theta = math.radians(heading_deg)

    phi1 = math.radians(origin.lat)
    lambda1 = math.radians(origin.lon)

    phi2 = math.asin(math.sin(phi1) * math.cos(d) + math.cos(phi1) * math.sin(d) * math.cos(theta))
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2),
    )

    return Coordinate(lat=math.degrees(phi2), lon=(math.degrees(lambda2) + 540) % 360 - 180)
