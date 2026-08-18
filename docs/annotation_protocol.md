# Annotation Protocol

Ground truth is produced manually in Roboflow and exported as **COCO Instance
Segmentation**, one annotation per individual tree. Polygon, uncompressed RLE
and compressed COCO RLE segmentations are all accepted.

Roboflow replaces `file_name` with an export-specific hashed name and keeps the
original under `images[].extra.name`. The evaluator joins on the original name
when it is present and keeps the hashed one for provenance, so predictions
produced from the source images still match after a re-export.

This document is the labelling contract: every annotator follows it, and every
metric in `docs/evaluation.md` is defined against it.

## 1. What counts as a tree

A **tree** is a perennial woody plant with a distinguishable trunk (visible or
plausibly occluded) supporting an elevated crown. Palms count as trees.

Explicitly **not** trees:

| Category | Examples | Annotate? |
|---|---|---|
| Shrubs/bushes | hedges, ornamental bushes, shrubs under ~2 m without a clear trunk | No |
| Grass | lawns, verges, grass strips | No |
| Climbing/potted plants | vines on walls, planters, flower beds | No |
| Dead trunks without crown | bare poles of removed trees | No |
| Trees on private property visible from the street | backyard crowns over a wall | **Yes** — the indicator is *visible* canopy, whatever the land tenure |
| Trees in planters/pits | street trees in tree pits | Yes |

When the trunk is hidden but the crown's size, height and texture make "tree"
the only reasonable reading, annotate it. When it is genuinely ambiguous
between tree and shrub, do **not** annotate it, and log the image in the
ambiguity list — consistent under-inclusion is measurable, oscillation is not.

## 2. What the mask includes

The mask covers **crown and trunk together** — every pixel that visually
belongs to the tree: foliage, branches, visible trunk. It excludes:

- sky visible **through** the crown where actual gaps are discernible at
  labelling zoom. Small internal gaps (< ~100 px at native resolution) may be
  bridged by the polygon; this matches the pipeline's small-hole refinement
  ceiling, so ground truth and prediction err in the same direction;
- supports, stakes, guards, signage attached to the tree;
- shadows on the ground;
- fallen leaves / ground litter.

## 3. Occlusion

Annotate **only visible pixels**. A crown split by a pole or a bus into
several visible fragments is **one annotation** (COCO polygons allow multiple
parts per instance) as long as the fragments clearly belong to one tree.
Never guess pixels behind an occluder.

If two trees' crowns overlap each other, split at the best visual boundary;
when no boundary is discernible, assign the ambiguous pixels to the nearer
(lower-in-image) tree. The pixel union — which is what coverage and the
semantic metrics use — is unaffected by where that internal boundary lands.

## 4. Trees partially outside the frame

Annotate the visible part, however small, if it passes the visibility floor
below. A crown entering from the top edge of the image is still visible canopy.

## 5. Minimum visibility

Skip vegetation that is:

- smaller than **~0.1% of the image area** (≈ 400 px in a 640×640 frame); or
- so distant/blurred that tree vs. other vegetation cannot be decided at 2×
  zoom.

These floors exist so the ground truth does not depend on annotator patience.
They are part of the definition of `tree_coverage_gt`: predicted pixels on
sub-floor vegetation will count as false positives, which is the intended,
conservative reading.

## 6. Images with no trees

Keep them, with zero annotations. Negative images are required — they are the
only way false-positive behaviour is measured, and `tree-ai validate-dataset`
reports how many the set contains.

## 7. Categories

Export with a single category named `tree`. If the workspace also labels
`shrub`/`grass` for other purposes, keep them as separate categories; the
loader (`urban_canopy/evaluation/coco.py`) only treats categories named
`tree`/`arvore`/`árvore` (configurable) as tree instances.

`iscrowd` regions are not used: a stand of trees too dense to separate should
be annotated as one instance per *distinguishable* crown, or skipped and
logged if none are distinguishable. The validator flags any `iscrowd`
annotation it finds.

## 8. Process

- Label at native image resolution; the export must keep the same width and
  height the pipeline analysed, or evaluation will refuse the pair.
- One annotator per image plus a reviewer pass on ~20% is the minimum; record
  disagreements in the ambiguity list.
- Version every export (`annotations_vN.json`) and never edit an export that
  has been evaluated against.
