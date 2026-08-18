# Evaluation Methodology

Evaluation runs offline from two files:

```bash
# 1. inference: writes artifacts_out/<timestamp>_<backend>/predictions.json
tree-ai --image street.jpg --predictions-json

# 2. evaluation, offline and repeatable
tree-ai evaluate --predictions artifacts_out/<run>/predictions.json \
                 --annotations annotations.json
```

The predictions file embeds the run manifest (model, versions, taxonomy,
refinement config, seed), so every reported number is traceable.

The join is on the original image basename. For Roboflow exports that means
`images[].extra.name`, which is preferred whenever present because Roboflow
replaces `file_name` with an export-specific hashed name; otherwise the COCO
`file_name` basename is used. Two images resolving to the same name is a fatal
error rather than a silent overwrite, and `tree-ai validate-dataset` reports it
before an evaluation is ever run. Images present on only one side are listed in
the report, never silently dropped.

Three independent levels are computed. They answer different questions and are
never merged into one score.

## Level 1 — Semantic segmentation (pixels)

Binary tree-vs-rest comparison between the predicted refined mask and the
union of the annotated instances, restricted to the **valid pixels** the
prediction was measured over (the excluded watermark strip leaves both sides).

Reported per image and pooled:

- **IoU** = TP / (TP + FP + FN)
- **Dice / F1** = 2·TP / (2·TP + FP + FN)
- **precision** = TP / (TP + FP)
- **recall** = TP / (TP + FN)

Conventions: pooled ("micro") metrics add the confusion counts over the whole
set first — they are the headline numbers. Macro averages are also reported
with the count of images that contributed. An image where neither prediction
nor ground truth has any tree pixel has *undefined* per-image IoU (reported as
NaN, counted in `n_images_without_trees_in_both`), rather than a flattering
1.0 or a punishing 0.0.

## Level 2 — Individual trees (instances)

Runs only when predictions carry instances: a real instance model (e.g. a
fine-tuned Mask R-CNN) or the explicitly requested connected-component
heuristic. The report names which (`instance_source`), because they are
different claims — the heuristic merges touching crowns and splits occluded
ones by construction.

Matching: greedy one-to-one by descending confidence (COCO protocol), IoU
threshold configurable via `--iou-threshold` (default **0.50**). Each
prediction matches at most one ground-truth instance and vice versa; a second
prediction on an already-claimed tree is a false positive. Predictions
without scores are ranked by area instead, and the report says so
(`ranked_by`).

Reported: TP, FP, FN, precision, recall, F1, and **mean matched IoU** (the
mean IoU of true-positive pairs only — localisation quality, separate from
detection rate). This supports statements like:

> recall 0.80 for individual trees; correctly matched instances had mean IoU 0.84.

**AP50** and **AP50:95** (101-point interpolation, COCO thresholds) are
computed only when *every* prediction carries a finite confidence score.
Otherwise the report states why AP is unavailable instead of inventing an
ordering.

## Level 3 — The coverage indicator

Direct comparison of the published number, `tree_coverage_pred` vs
`tree_coverage_gt`, both in percent, with `tree_coverage_gt` computed from the
annotation union using the same valid-pixel denominator as the prediction.

- **MAE** in percentage points (headline)
- **RMSE** in percentage points
- **bias** (mean signed error — systematic over/under-estimation)
- max absolute error, mean of both sides

**Pearson r** is reported as a complementary diagnostic only. A model
predicting exactly twice the true coverage has r = 1.0 and is wrong by a
factor of two; correlation never substitutes for the error metrics. r is
omitted when either side has no variance.

Levels 1 and 3 are deliberately separate: a mask shifted sideways can have
poor IoU and perfect coverage agreement. Both facts matter and both are
reported.

## Experimental split

- Zero-shot pre-trained backends with default settings: the whole labelled set
  may serve as test.
- The moment any parameter is tuned by looking at results — refinement sizes,
  score thresholds, taxonomy edits, `--exclude-bottom-px` — the set must be
  split: a **calibration/validation** subset for tuning and a **held-out test**
  subset touched once, at the end. Record the split as file lists next to the
  annotations.
- Any fine-tuning needs the full train/val/test discipline, with no street
  segment shared between splits (frames of the same location are near
  duplicates).
- Seeds, model names and configuration land in the manifest automatically;
  keep the split lists and the annotation export version in the same commit as
  the reported numbers.

## Qualitative audit

`--save-artifacts` writes, per view: the RGB frame, raw mask, refined mask,
tree overlay, instance visualisation (when instances exist) and a metrics
JSON. The evaluation report carries per-image rows; sorting them by IoU or by
absolute coverage error and opening the corresponding artifact folders is the
intended workflow for collecting success and failure cases.
