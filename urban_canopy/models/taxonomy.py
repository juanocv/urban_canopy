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
import re
from collections.abc import Mapping
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
    "infer_class_space",
    "load_taxonomy",
    "normalise_label",
    "validate_taxonomy_class_space",
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
    if not text:
        return ()
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    out: list[str] = [text]
    out.extend(t for t in tokens if t != text)
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True, slots=True)
class ClassGroup:
    """A named bucket of dataset class names."""

    name: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Taxonomy group names cannot be empty.")
        aliases = tuple(
            dict.fromkeys(
                normalised for alias in self.aliases for normalised in normalise_label(alias)
            )
        )
        if not aliases:
            raise ValueError(f"Taxonomy group {name!r} needs at least one non-empty alias.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)

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
    #: Explicit resolution for an alias intentionally shared by groups.
    #: Stored as ``((alias, preferred_group), ...)`` to keep the frozen object
    #: immutable and JSON-serialisable.
    alias_priority: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        class_space = str(self.class_space).strip().lower()
        if not class_space:
            raise ValueError("class_space cannot be empty.")
        object.__setattr__(self, "class_space", class_space)

        groups = tuple(self.groups)
        if not all(isinstance(group, ClassGroup) for group in groups):
            raise ValueError("groups must contain ClassGroup objects only.")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "vegetation_groups", tuple(self.vegetation_groups))

        canonical_names: dict[str, str] = {}
        for group in self.groups:
            key = group.name.casefold()
            if key in canonical_names:
                raise ValueError(f"Duplicate taxonomy group name: {group.name!r}.")
            canonical_names[key] = group.name
        names = set(canonical_names.values())

        priorities: dict[str, str] = {}
        raw_priorities = (
            self.alias_priority.items()
            if isinstance(self.alias_priority, Mapping)
            else self.alias_priority
        )
        for raw_alias, raw_group in raw_priorities:
            aliases = normalise_label(raw_alias)
            if len(aliases) != 1:
                raise ValueError(
                    f"alias_priority key {raw_alias!r} must identify one normalised alias."
                )
            try:
                group_name = canonical_names[str(raw_group).strip().casefold()]
            except KeyError:
                raise ValueError(
                    f"alias_priority maps {raw_alias!r} to unknown group {raw_group!r}."
                ) from None
            if aliases[0] in priorities:
                raise ValueError(f"Duplicate alias_priority entry for {aliases[0]!r}.")
            priorities[aliases[0]] = group_name
        object.__setattr__(self, "alias_priority", tuple(priorities.items()))

        owners: dict[str, list[str]] = {}
        for group in self.groups:
            for alias in group.aliases:
                owners.setdefault(alias, []).append(group.name)
        for alias, preferred in priorities.items():
            alias_owners = owners.get(alias, [])
            if not alias_owners:
                raise ValueError(f"alias_priority names unknown alias {alias!r}; no group owns it.")
            if preferred not in alias_owners:
                raise ValueError(
                    f"alias_priority for {alias!r} selects {preferred!r}, which does "
                    f"not own that alias ({alias_owners})."
                )
        for alias, groups in owners.items():
            if len(groups) <= 1:
                continue
            preferred = priorities.get(alias)
            if preferred is None:
                raise ValueError(
                    f"Alias {alias!r} belongs to multiple groups {groups}; add an "
                    "alias_priority entry to resolve it explicitly."
                )
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
        candidates = normalise_label(label)
        priorities = dict(self.alias_priority)
        for alias in candidates:
            if alias in priorities:
                return priorities[alias]
        for group in self.groups:
            if any(alias in candidates for alias in group.aliases):
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
            "alias_priority": dict(self.alias_priority),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Taxonomy":
        groups = tuple(
            ClassGroup(name=str(g["name"]), aliases=tuple(str(a) for a in g["aliases"]))
            for g in data["groups"]
        )
        raw_priority = data.get("alias_priority", {})
        if not isinstance(raw_priority, dict):
            raise ValueError("alias_priority must be a JSON object mapping alias to group.")
        return cls(
            class_space=str(data["class_space"]),
            groups=groups,
            tree_group=data.get("tree_group"),
            vegetation_groups=tuple(data.get("vegetation_groups", ())),
            tree_proxy_group=data.get("tree_proxy_group"),
            alias_priority=tuple((str(alias), str(group)) for alias, group in raw_priority.items()),
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

#: Dataset tokens that appear in HuggingFace checkpoint names, mapped to the
#: class space they imply. Both naming conventions in use are covered:
#: ``shi-labs/oneformer_ade20k_swin_large`` and
#: ``facebook/mask2former-swin-large-ade-semantic``.
_DATASET_TOKENS = {
    "ade": "ade20k",
    "ade20k": "ade20k",
    "coco": "coco_panoptic",
    "cityscapes": "cityscapes",
}


def infer_class_space(model_name: str) -> str:
    """
    Work out which class space a checkpoint predicts, from its name.

    Matching is on whole tokens rather than substrings: ``"ade"`` is three
    characters and would otherwise match inside unrelated words, silently
    selecting a taxonomy whose tree class does not exist in the model.

    Raises when the name says nothing recognisable -- guessing here would
    mislabel every pixel the run reports.
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", str(model_name).lower()) if t}
    matched = {space for token, space in _DATASET_TOKENS.items() if token in tokens}

    if len(matched) == 1:
        return matched.pop()
    if not matched:
        raise ValueError(
            f"Cannot tell which dataset the checkpoint {model_name!r} was trained on, "
            f"so the tree classes are unknown. Recognised tokens: "
            f"{', '.join(sorted(_DATASET_TOKENS))}. Pass an explicit taxonomy "
            "(--taxonomy) to state the mapping yourself."
        )
    raise ValueError(
        f"The checkpoint name {model_name!r} names more than one dataset "
        f"({', '.join(sorted(matched))}); pass --taxonomy to disambiguate."
    )


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
        taxonomy = Taxonomy.from_dict(source)
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        taxonomy = Taxonomy.from_dict(data)
    return validate_taxonomy_class_space(taxonomy, class_space, context="selected backend")


def validate_taxonomy_class_space(
    taxonomy: Taxonomy,
    class_space: str,
    *,
    context: str,
) -> Taxonomy:
    """Reject a taxonomy that names a different checkpoint label space."""
    expected = str(class_space).strip().lower()
    if taxonomy.class_space != expected:
        raise ValueError(
            f"Taxonomy declares class_space={taxonomy.class_space!r} but {context} "
            f"speaks {expected!r}."
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
