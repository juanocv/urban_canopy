"""
Deterministic heading selection for multi-view analysis.

The sidewalk pipeline chose headings by *looking*: it segmented four probe
frames, scored them on whether two curbs were visible and how balanced the
sidewalk pixels were, and centred the sweep on the winner. None of that is
reusable here, and it would be actively wrong if it were -- picking the heading
from the segmentation of the very class being measured makes the measurement a
function of its own outcome. A location that happens to segment well would get
sampled where the model is confident, and the resulting coverage would be biased
upwards by construction.

So heading selection here is blind to the canopy mask and fully determined by
configuration:

* ``fixed``       -- exactly the headings the caller listed;
* ``offsets``     -- a reference heading plus a configured set of offsets;
* ``equiangular`` -- N headings evenly spaced around the reference.

Same config, same headings, every run, on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

__all__ = ["ViewPlanConfig", "plan_headings", "DEFAULT_OFFSETS"]

#: Both sides of the street plus both directions along it: the sweep most street
#: level canopy studies use when nothing better is known about the geometry.
DEFAULT_OFFSETS: tuple[int, ...] = (0, 90, 180, 270)

PlanMode = Literal["fixed", "offsets", "equiangular"]


@dataclass(frozen=True, slots=True)
class ViewPlanConfig:
    """How to turn a location into a list of headings."""

    mode: PlanMode = "offsets"
    #: Reference heading in degrees; typically the street bearing. Defaults to 0
    #: (north) when the caller has nothing better, which keeps runs comparable.
    reference_heading: int = 0
    #: Used by ``offsets``.
    offsets: tuple[int, ...] = DEFAULT_OFFSETS
    #: Used by ``equiangular``.
    n_views: int = 4
    #: Used by ``fixed``.
    headings: tuple[int, ...] = ()
    pitch: int = 0
    fov: int = 90
    size: str = "640x640"
    #: Abort the run when fewer headings than this produce a usable result.
    min_successful_views: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reference_heading": self.reference_heading,
            "offsets": list(self.offsets),
            "n_views": self.n_views,
            "headings": list(self.headings),
            "pitch": self.pitch,
            "fov": self.fov,
            "size": self.size,
            "min_successful_views": self.min_successful_views,
        }


def _wrap(value: float) -> int:
    """Normalise a heading to the integer range [0, 360)."""
    return int(round(value)) % 360


def plan_headings(config: ViewPlanConfig) -> list[int]:
    """
    Headings for one location, in acquisition order.

    Duplicates are removed while preserving order -- two offsets that land on
    the same heading would otherwise pay for the same frame twice and count it
    twice in the aggregate.
    """
    if config.mode == "fixed":
        if not config.headings:
            raise ValueError("mode='fixed' needs at least one heading.")
        raw: Sequence[float] = [float(h) for h in config.headings]

    elif config.mode == "offsets":
        if not config.offsets:
            raise ValueError("mode='offsets' needs at least one offset.")
        raw = [config.reference_heading + float(off) for off in config.offsets]

    elif config.mode == "equiangular":
        if config.n_views < 1:
            raise ValueError("mode='equiangular' needs n_views >= 1.")
        step = 360.0 / config.n_views
        raw = [config.reference_heading + index * step for index in range(config.n_views)]

    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown view plan mode: {config.mode!r}")

    return list(dict.fromkeys(_wrap(value) for value in raw))


@dataclass(frozen=True, slots=True)
class SingleViewConfig:
    """Capture parameters for a one-frame analysis."""

    heading: int = 0
    pitch: int = 0
    fov: int = 90
    size: str = "640x640"
    metadata: dict[str, Any] = field(default_factory=dict)
