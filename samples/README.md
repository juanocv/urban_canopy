# Sample images

A small, deliberately varied set for trying the pipeline, running the notebooks
and smoke-testing a new backend without spending Google API quota.

## Contents

| File | Measured tree coverage | Why it is here |
|---|---|---|
| `images/streetview_id4_heading30.jpg` | 0.00% | **Negative case** — no trees at all |
| `images/streetview_id1_heading90.jpg` | 3.8% | Sparse: a few distant crowns |
| `images/streetview_id3_heading15.jpg` | 14.9% | Moderate |
| `images/streetview_id1_heading180.jpg` | 16.0% | Moderate |
| `images/streetview_id1_heading270.jpg` | 23.1% | Substantial |
| `images/streetview_id1_heading0.jpg` | 36.2% | Heavy canopy |
| `images/streetview_id3_heading165.jpg` | 40.2% | Heaviest in the set |

Coverage measured with `--seg detectron2` (COCO-panoptic, `tree-merged`) at
default settings. They are indicative, not ground truth: no image here is
annotated, so nothing in this folder can be used to *evaluate* a model — only to
exercise the pipeline and to eyeball its output.

The four `id1` files are a 90-degree sweep of one location (headings 0, 90, 180,
270), which is what the multi-view notebook aggregates. Their spread — 3.8% to
36.2% at the same point — is itself the argument for multi-view: a single
heading is not a property of the location.

The negative case is not filler. An image where the model should predict nothing
is the only kind that exposes false positives, and it is the first thing to
check after changing a taxonomy or a refinement setting.

## Provenance and terms

These frames come from the Google Street View Static API, retrieved for the
predecessor project (`sidewalk_analysis`). They are included for convenience of
testing.

Google's terms restrict redistribution of Street View imagery, so treat this
folder as a local development convenience rather than a redistributable dataset:
if this repository is published or the imagery is used in a publication, review
the [Google Maps Platform Terms of
Service](https://cloud.google.com/maps-platform/terms) first, and consider
replacing these with imagery you are licensed to redistribute.

## Reproducing the measurements

```bash
tree-ai --image samples/images/streetview_id1_heading0.jpg --seg detectron2 --device cpu
```

Or the whole folder at once, into one run directory with a CSV:

```bash
tree-ai --image samples/images/streetview_id1_heading0.jpg \
        --image samples/images/streetview_id1_heading90.jpg \
        --image samples/images/streetview_id1_heading180.jpg \
        --image samples/images/streetview_id1_heading270.jpg \
        --seg detectron2 --save-artifacts --csv
```
