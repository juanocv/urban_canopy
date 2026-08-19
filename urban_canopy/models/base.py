"""
The common contract every segmentation backend fulfils.

This replaces the sidewalk project's four-tuple return
``(mask, seg_map, seg_info, obstacles)``, which encoded one target class and one
downstream consumer. A canopy backend has to say more than "here is the mask":
it must report which class space it speaks and which vegetation groups it could
separate, so no consumer has to guess.

The contract is deliberately semantic-only. Individual trees are not part of it:
no published checkpoint for any class space this project supports carries tree as
a *thing* class, so a backend could never honestly fill such a field. See
``docs/evaluation.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .taxonomy import Taxonomy

__all__ = [
    "Segment",
    "SegmentationOutput",
    "Segmenter",
]


@dataclass(frozen=True, slots=True)
class Segment:
    """One segment of a panoptic/semantic map."""

    id: int
    label: str
    is_thing: bool = False
    score: float | None = None


@dataclass(slots=True)
class SegmentationOutput:
    """What every backend returns, whatever it is underneath."""

    backend: str
    class_space: str
    taxonomy: Taxonomy
    #: group name -> H x W bool mask. Groups the taxonomy declares but the frame
    #: does not contain are present and empty, so callers never KeyError.
    group_masks: dict[str, np.ndarray]
    #: Per-pixel segment/class id map, kept for auditing and debug overlays.
    label_map: np.ndarray | None = None
    segments: list[Segment] | None = None
    #: Free-form provenance notes surfaced in results and logs.
    notes: tuple[str, ...] = field(default_factory=tuple)

    def group(self, name: str) -> np.ndarray | None:
        """Mask for a group, or None when this class space has no such group."""
        return self.group_masks.get(name)

    def validate(self, expected_shape: tuple[int, int]) -> None:
        """Reject malformed backend output before NumPy can broadcast it silently."""
        shape = (int(expected_shape[0]), int(expected_shape[1]))
        if min(shape) <= 0:
            raise ValueError(f"Expected a positive output shape, got {shape}.")
        if self.taxonomy.class_space != self.class_space:
            raise ValueError(
                f"Segmentation output declares class_space={self.class_space!r}, but its "
                f"taxonomy declares {self.taxonomy.class_space!r}."
            )

        expected_groups = set(self.taxonomy.group_names)
        actual_groups = set(self.group_masks)
        if actual_groups != expected_groups:
            missing = sorted(expected_groups - actual_groups)
            extra = sorted(actual_groups - expected_groups)
            raise ValueError(
                "Segmentation output groups do not match the taxonomy "
                f"(missing={missing}, extra={extra})."
            )

        for name, mask in self.group_masks.items():
            arr = np.asarray(mask)
            if arr.ndim != 2 or arr.shape != shape:
                raise ValueError(
                    f"Segmentation group {name!r} has shape {arr.shape}, expected {shape}."
                )

        if self.label_map is not None:
            label_map = np.asarray(self.label_map)
            if label_map.ndim != 2 or label_map.shape != shape:
                raise ValueError(
                    f"Segmentation label_map has shape {label_map.shape}, expected {shape}."
                )

    def union(self, names: Iterable[str]) -> np.ndarray | None:
        """Union of several group masks; None when none of them exist."""
        masks = [self.group_masks[n] for n in names if n in self.group_masks]
        if not masks:
            return None
        out = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            out |= m
        return out

    @property
    def shape(self) -> tuple[int, int]:
        for mask in self.group_masks.values():
            return tuple(mask.shape[:2])  # type: ignore[return-value]
        if self.label_map is not None:
            return tuple(self.label_map.shape[:2])  # type: ignore[return-value]
        raise ValueError("SegmentationOutput carries neither group masks nor a label map.")


class Segmenter(Protocol):
    """
    Common contract for all vegetation segmentation backends.

    The target classes are fixed at construction through a
    :class:`~urban_canopy.models.taxonomy.Taxonomy`, not passed per call: which
    classes count as trees is a property of the study, not of the frame.
    """

    backend_name: str
    class_space: str
    taxonomy: Taxonomy

    def segment(self, img_rgb: np.ndarray) -> SegmentationOutput: ...


def build_group_masks(
    taxonomy: Taxonomy,
    labelled: Sequence[tuple[str, Any]],
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """
    Fold ``(class_name, mask)`` pairs into one boolean mask per taxonomy group.

    Every declared group gets an entry, empty when the frame contains none of
    its classes: an absent key and an all-false mask mean different things
    downstream (unavailable vs. zero coverage), and only the taxonomy may say
    which applies.
    """
    out = {name: np.zeros(shape, dtype=bool) for name in taxonomy.group_names}
    for label, mask in labelled:
        group = taxonomy.group_for_label(label)
        if group is None:
            continue
        out[group] |= np.asarray(mask).astype(bool)
    return out
