"""
Vegetation class taxonomy.

The one thing that must never be implicit in this project is *which* predicted
classes are counted as trees. Every backend speaks a different class space, and
some of those spaces cannot express "tree" at all, so the mapping lives here as
data -- inspectable, serialisable, and overridable from a JSON file -- instead
of being scattered through the adapters as string comparisons.

Two rules are enforced by construction:

* ``tree`` and the wider vegetation groups (``grass``, ``plant_shrub``) are kept
  apart. They are never silently merged; ``vegetation_groups`` states the union
  explicitly, and a caller can redefine it.
* A class space with no tree class (Cityscapes) reports ``tree_group is None``.
  Coverage then comes back as "unavailable" unless the caller *asks* for the
  proxy, and the result is flagged as a proxy wherever it travels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ClassGroup",
    "Taxonomy",
    "ADE20K",
    "COCO_PANOPTIC",
    "CITYSCAPES",
    "default_taxonomy",
    "load_taxonomy",
    "normalise_label",
]

CLASS_SPACES = ("ade20k", "coco_panoptic", "cityscapes")


def normalise_label(label: str) -> tuple[str, ...]:
    """
    Split a dataset class name into the synonyms it advertises.

    ADE20K names carry their synonyms inline ("plant, flora, plant life") and
    different releases expose different truncations of the same class, so
    matching on the full string alone misses. Returns the full normalised string
    first, then each comma-separated token.
    """
    text = " ".join(str(label).strip().lower().split())
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    out: list[str] = [text]
    out.extend(t for t in tokens if t != text)
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True, slots=True)
class ClassGroup:
    """A named bucket of dataset class names."""

    name: str
    aliases: tuple[str, ...]

    def matches(self, label: str) -> bool:
        candidates = set(normalise_label(label))
        return any(alias in candidates for alias in self.aliases)


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """How one backend's class space maps onto this project's groups."""

    class_space: str
    groups: tuple[ClassGroup, ...]
    #: Group whose pixels are counted as trees. ``None`` when the class space has
    #: no tree class of its own.
    tree_group: str | None
    #: Groups whose union defines ``vegetation_coverage_ratio``.
    vegetation_groups: tuple[str, ...]
    #: Group offered as an *explicit* stand-in when ``tree_group`` is None.
    tree_proxy_group: str | None = None

    def __post_init__(self) -> None:
        names = {g.name for g in self.groups}
        if self.tree_group is not None and self.tree_group not in names:
            raise ValueError(f"tree_group={self.tree_group!r} is not one of {sorted(names)}")
        if self.tree_proxy_group is not None and self.tree_proxy_group not in names:
            raise ValueError(
                f"tree_proxy_group={self.tree_proxy_group!r} is not one of {sorted(names)}"
            )
        unknown = [g for g in self.vegetation_groups if g not in names]
        if unknown:
            raise ValueError(f"vegetation_groups names unknown groups: {unknown}")

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups)

    @property
    def has_tree_class(self) -> bool:
        return self.tree_group is not None

    def group_for_label(self, label: str) -> str | None:
        """Group a predicted class name belongs to, or None if it is not vegetation."""
        for group in self.groups:
            if group.matches(label):
                return group.name
        return None

    def with_overrides(self, **kwargs: Any) -> "Taxonomy":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_space": self.class_space,
            "groups": [{"name": g.name, "aliases": list(g.aliases)} for g in self.groups],
            "tree_group": self.tree_group,
            "vegetation_groups": list(self.vegetation_groups),
            "tree_proxy_group": self.tree_proxy_group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Taxonomy":
        groups = tuple(
            ClassGroup(name=str(g["name"]), aliases=tuple(str(a).lower() for a in g["aliases"]))
            for g in data["groups"]
        )
        return cls(
            class_space=str(data["class_space"]),
            groups=groups,
            tree_group=data.get("tree_group"),
            vegetation_groups=tuple(data.get("vegetation_groups", ())),
            tree_proxy_group=data.get("tree_proxy_group"),
        )


# --------------------------------------------------------------------------- #
# Built-in taxonomies                                                          #
# --------------------------------------------------------------------------- #

#: ADE20K-150 (OneFormer). "tree" (id 4) is a *stuff* class; "palm" (id 72) is a
#: thing class, and it is a tree species, so it is counted as a tree here.
ADE20K = Taxonomy(
    class_space="ade20k",
    groups=(
        ClassGroup("tree", ("tree", "palm", "palm tree")),
        ClassGroup("grass", ("grass",)),
        ClassGroup("plant_shrub", ("plant", "flora", "plant life", "flower", "bush", "shrub")),
    ),
    tree_group="tree",
    vegetation_groups=("tree", "grass", "plant_shrub"),
)

#: COCO-panoptic 133 (Detectron2 panoptic FPN). "tree-merged" (id 184) is a
#: *stuff* class that already merges every tree in the frame into one segment.
COCO_PANOPTIC = Taxonomy(
    class_space="coco_panoptic",
    groups=(
        ClassGroup("tree", ("tree-merged", "tree")),
        ClassGroup("grass", ("grass-merged", "grass")),
        ClassGroup(
            "plant_shrub",
            ("bush", "flower", "leaves", "branch", "potted plant", "plant-other", "moss"),
        ),
    ),
    tree_group="tree",
    vegetation_groups=("tree", "grass", "plant_shrub"),
)

#: Cityscapes-19 (DeepLabV3+). There is **no tree class**: "vegetation" (id 8)
#: merges trees with bushes and hedges, and "terrain" (id 9) merges grass with
#: soil and sand. "terrain" is deliberately left out of vegetation_groups for
#: that reason -- it is ground cover that is only sometimes vegetation.
CITYSCAPES = Taxonomy(
    class_space="cityscapes",
    groups=(
        ClassGroup("vegetation", ("vegetation",)),
        ClassGroup("terrain", ("terrain",)),
    ),
    tree_group=None,
    vegetation_groups=("vegetation",),
    tree_proxy_group="vegetation",
)

_BUILTIN = {t.class_space: t for t in (ADE20K, COCO_PANOPTIC, CITYSCAPES)}


def default_taxonomy(class_space: str) -> Taxonomy:
    """Built-in taxonomy for a class space."""
    try:
        return _BUILTIN[class_space]
    except KeyError:
        raise ValueError(
            f"No built-in taxonomy for class space {class_space!r}; "
            f"known spaces: {', '.join(sorted(_BUILTIN))}. "
            "Pass a JSON taxonomy instead."
        ) from None


def load_taxonomy(source: str | Path | dict[str, Any] | None, *, class_space: str) -> Taxonomy:
    """
    Resolve the taxonomy for a run.

    ``None`` selects the built-in mapping for *class_space*; a path or dict lets
    a study pin its own class mapping and ship it alongside its results.
    """
    if source is None:
        return default_taxonomy(class_space)
    if isinstance(source, dict):
        return Taxonomy.from_dict(source)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    taxonomy = Taxonomy.from_dict(data)
    if taxonomy.class_space != class_space:
        raise ValueError(
            f"Taxonomy file declares class_space={taxonomy.class_space!r} but the "
            f"selected backend speaks {class_space!r}."
        )
    return taxonomy


def group_masks_from_labels(
    label_masks: Iterable[tuple[str, Any]],
    taxonomy: Taxonomy,
) -> dict[str, list[Any]]:
    """Bucket ``(class_name, mask)`` pairs into taxonomy groups."""
    buckets: dict[str, list[Any]] = {name: [] for name in taxonomy.group_names}
    for label, mask in label_masks:
        group = taxonomy.group_for_label(label)
        if group is not None:
            buckets[group].append(mask)
    return buckets
