# Architecture

Urban Canopy inherits the modular architecture of
[`sidewalk_analysis`](https://github.com/juanocv/sidewalk_analysis) and removes
everything specific to metric sidewalk measurement. This document records the
structure and the exact mapping from the parent project.

## Pipeline

```
acquisition (Street View / local)
        │  ImageRequest, caching, pano metadata
        ▼
view plan (single / multi-view)          ← deterministic, config-driven
        │  headings, pitch, fov
        ▼
segmentation backend            ← OneFormer | Mask2Former | Detectron2 | DeepLab
        │  SegmentationOutput: group masks per taxonomy, instances?, notes
        ▼
tree-mask resolution                     ← tree class, or explicit vegetation proxy
        ▼
conservative refinement                  ← optional; growth-guarded
        ▼
coverage indicators                      ← tree pixels / all image pixels
        ▼
aggregation (multi-view)                 ← median/IQR/p25/p75; counts stay per view
        ▼
exports                                  ← artifacts, metrics JSON, CSV, predictions
```

Evaluation is a separate, offline stage joining a predictions file with a COCO
ground-truth export (see `docs/evaluation.md`).

## Modules

| Module | Responsibility |
|---|---|
| `core/pipeline.py` | Orchestration; dependency-injected segmenter + Street View client |
| `core/viewplan.py` | Deterministic heading plans (fixed / offsets / equiangular) |
| `core/config.py` | `CanopyConfig`, seeds, run manifest |
| `validation.py` | Dependency-free limits shared by dataclasses, CLI and API |
| `core/results.py` | View results, structured heading failures, CSV rows |
| `io/streetview.py` | GSV client, cache, geocoding, `ImageRequest`, pano metadata |
| `io/image_io.py` | Decoding to RGB and overlays |
| `io/artifacts.py` | Per-view audit artifacts with checked atomic writes |
| `io/atomic.py`, `io/json_io.py` | Atomic file replacement and strict JSON conversion |
| `io/geo.py` | Pure geographic helpers |
| `models/taxonomy.py` | Class-space → group mapping; tree vs vegetation kept apart |
| `models/base.py` | `SegmentationOutput` contract, instance provenance constants |
| `models/{oneformer,mask2former,detectron2,deeplab}.py` | Backend adapters |
| `models/factory.py` | Lazy backend construction; optional ML imports happen at construction |
| `processing/coverage.py` | The indicator; proxy/unavailable semantics |
| `processing/refinement.py` | Conservative canopy cleanup with growth guard |
| `processing/instances.py` | Connected-component heuristic, explicitly flagged |
| `processing/aggregate.py` | Robust multi-view statistics |
| `evaluation/*` | COCO loading, RLE, three metric levels, runner, interchange |
| `cli/`, `webapi.py` | Interfaces |

## Mapping from sidewalk_analysis

### Reused (unchanged or lightly adapted)

| Component | Disposition |
|---|---|
| `StreetViewClient`, `ImageRequest`, geocoding, joblib cache | **Reused**; added pano-id/date recording, hashable request |
| `io/geo.py` | **Reused** verbatim |
| `read_rgb` normalisation | **Reused**; Street View frames remain intact |
| `log.py` (text/JSON logging) | **Reused** (`SWAI_*` → `UC_*`) |
| `diagnostics.py` | **Reused**, trimmed to relevant deps |
| Settings pattern (pydantic-settings, `.env`) | **Reused**, including the `extra="ignore"` regression fix and its test |
| Lazy package exports (PEP 562) | **Reused** |
| Factory with lazy imports + install hints | **Reused** |
| Device management (`--device auto/cpu/cuda`, early failure) | **Reused** |
| DeepLab checkpoint loader (architecture inference, tensor-match guard) | **Reused** |
| CLI structure, Windows console fixes | **Reused**, plus sub-commands |
| Web API structure (lifespan, registry, semaphore, CORS) | **Reused**, re-keyed on canopy config |
| Offline/CPU-only test philosophy, `gpu`/`network` markers | **Reused** |
| CI workflow, check/setup scripts | **Reused** |

### Adapted (same idea, new domain)

| Component | Change |
|---|---|
| `Segmenter` protocol (4-tuple) | → `SegmentationOutput` dataclass: taxonomy-driven group masks, honest instance support, provenance notes |
| Backend wrappers | Target decoupled from `sidewalk`; per-backend class-space audit; **no refinement inside adapters** (the parent called `shave_above_top_envelope` there; refinement is now one explicit pipeline stage) |
| `AliasSegmenter` label synonyms | → `Taxonomy` (data, serialisable, per-study override) |
| Multi-view aggregation (median width) | → robust stats over coverage ratios; per-view instance counts |
| Debug artifacts | → structured per-view artifact directories |

### Removed (not carried into the inference path)

Sidewalk segmentation and refinement (`refine_sidewalk_mask`, curb-line
RANSAC, bridge fill, `shave_above_top_envelope`); obstacle extraction and
contact bases; MiDaS; ZoeDepth; every depth path (`DepthScale`,
`to_metric_depth`, fallback scales); camera model and pixel→metre conversion;
width estimation (`WidthResult`, `compute_width`); clearances
(`ClearanceResult`, `compute_clearances`); NBR 9050 accessibility metrics and
ratings; ensemble mask fusion (fusing masks from different class spaces is not
meaningful for a measurement whose value *is* the mask area — comparison across
backends happens in evaluation instead); and the mask-driven street-centre
search (`_find_street_center`).

That last removal is a correctness point, not a simplification: choosing
headings by segmenting probe frames would make the sample depend on the model
being measured. Heading plans here are deterministic and blind to the imagery.

## Contracts worth knowing

- **`tree_source`** on every result: `tree_class`, `vegetation_proxy`
  (explicitly requested, flagged) or `unavailable` (never silently zero).
- **Class space follows the checkpoint**, not the backend, for OneFormer and
  Mask2Former — both publish weights for several datasets. `infer_class_space()`
  reads the dataset token out of the model name and selects the taxonomy from
  it, so a Cityscapes checkpoint gets a taxonomy with no tree group and reports
  coverage as unavailable. A name that matches no known dataset is refused
  rather than defaulted: applying an ADE20K taxonomy to unknown classes would
  mislabel every pixel silently.
- **Taxonomy consistency**: every adapter rejects a taxonomy from another class
  space before importing or downloading its model. Aliases use the same
  normalization as predicted labels; duplicate group names and ambiguous
  aliases are rejected unless `alias_priority` resolves the conflict.
- **`InstanceMask.source`**: `model` or `connected_components_heuristic`;
  metrics and reports carry it through.
- **Complete-frame denominator**: Street View frames, including attribution and
  watermark pixels, remain intact. Coverage is always divided by `H * W`.
- **Prediction mask status**: `available`, `unavailable` (the class space has no
  tree class) or `omitted` (mask export disabled). Only available masks enter
  semantic metrics.
- **Predictions interchange**: uncompressed COCO RLE, readable with or without
  pycocotools, manifest embedded.
- **Multi-view minimum**: a plan requires `min_successful_views >= 1`. Each
  failed heading records stage, exception type and message; falling below the
  minimum raises `MultiViewAnalysisError` rather than returning an empty run.
- **Strict JSON**: undefined numeric metrics remain undefined in memory and are
  exported as JSON `null`; `NaN` and `Infinity` are never written.
- **Atomic outputs**: cache frames, JSON, CSV and image artifacts are written to
  a sibling temporary file and atomically replace the target only after success.
- **Bounded RGB retention**: `keep_rgb` defaults to false. CLI batches yield one
  result at a time and release RGB after per-view artifacts; the API retains it
  only for `/analyse/single` requests that ask for overlays.
- **Reproducibility levels**: RNG seeding and deterministic Torch algorithms are
  separate manifest fields. `PYTHONHASHSEED` is observed, never assigned after
  startup, and cross-stack bitwise identity is explicitly not guaranteed.
