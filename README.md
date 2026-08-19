# Urban Canopy

Urban Canopy estimates the **visible tree-canopy coverage** of urban streets from
Google Street View imagery, using semantic/panoptic/instance segmentation. The
production package lives in `urban_canopy/`; third-party model checkouts
(OneFormer via HuggingFace, Detectron2, DeepLab) stay outside the package
boundary.

The primary indicator is continuous and per image:

```
tree_coverage_ratio = tree pixels / all image pixels      (in [0, 1])
tree_coverage_pct   = 100 * tree_coverage_ratio
```

A wider `vegetation_coverage_ratio` is reported separately when the model can
distinguish it. Tree, grass and shrub classes are **never merged silently** —
the mapping from model classes to these groups is explicit, inspectable and
overridable (`urban_canopy/models/taxonomy.py`). No qualitative bands ("low /
medium / high greenery") are produced: the continuous ratio is the output.

## What it does

1. **Acquisition** — Google Street View (cached, with panorama id + capture
   date recorded) or local images.
2. **View strategy** — single view, or a deterministic multi-view plan
   (reference heading + offsets, or equiangular sampling). Heading selection is
   configuration-driven and independent of the segmentation output.
3. **Segmentation** — OneFormer (ADE20K), Mask2Former (ADE20K, COCO or
   Cityscapes), Detectron2 (COCO-panoptic, or a custom instance model), DeepLab
   (Cityscapes), behind one common contract.
4. **Refinement** — conservative, optional cleanup of the canopy mask (speck
   removal, small-hole filling), with a growth guard that prevents any setting
   from inflating the mask by more than a configured fraction.
5. **Indicators** — coverage ratios per image, with quality flags and full
   capture provenance.
6. **Aggregation** — mean / median / IQR / p25 / p75 across the views of a
   location. Instance counts stay per view and are never summed across views.
7. **Evaluation** — three independent levels against manual COCO ground truth:
   pixels (IoU, Dice/F1, precision, recall), instances (TP/FP/FN, precision,
   recall, F1, mean matched IoU, AP50/AP50:95 when scores exist), and the
   coverage indicator itself (MAE, RMSE, bias in percentage points).
8. **Audit artifacts** — per view: RGB, raw mask, refined mask, overlays,
   instance visualisation, metrics JSON; plus CSV/JSON exports per run.

### What the backends can and cannot claim

| Backend | Pretraining | Tree class | Individual trees? |
|---|---|---|---|
| OneFormer | ADE20K-150 | `tree` (stuff) + `palm` | No — coverage only |
| Mask2Former | ADE20K / COCO / Cityscapes | depends on the checkpoint | No — coverage only |
| Detectron2 panoptic FPN | COCO-panoptic 133 | `tree-merged` (stuff) | No — coverage only |
| Detectron2 Mask R-CNN (custom weights) | your fine-tune | your `tree` thing class | **Yes** — masks + scores |
| DeepLab V3+ | Cityscapes-19 | none (`vegetation` merges trees+bushes) | No — and no tree ratio unless `--allow-vegetation-proxy` |

Mask2Former is the one backend published for **several class spaces**, so it can
hold the architecture fixed and vary the label set — which separates "the model
disagrees" from "the dataset has no such class". On one sample frame:

```text
oneformer     tree 31.97%   vegetation 42.68%     (ADE20K)
mask2former   tree 32.69%   vegetation 42.88%     (ADE20K)
detectron2    tree 36.21%   vegetation 46.00%     (COCO-panoptic)
deeplab       tree    n/a   vegetation 34.54%     (Cityscapes — no tree class)

mask2former --seg-model facebook/mask2former-swin-tiny-cityscapes-semantic
              tree    n/a   vegetation 36.24%     (same model, no tree class)
```

That last line is the point: the same architecture reports no tree ratio when
pointed at a class space that cannot express one, rather than quietly returning
the vegetation number.

Splitting a semantic mask into connected components is available as an
**explicitly flagged heuristic** (`--instances heuristic`), not as instance
segmentation: touching crowns merge, occluded crowns split, and the counts say
so in every output.

## Repository layout

```text
urban_canopy/              Python package used by the CLI and API
urban_canopy/core/         Pipeline orchestration, config, results, view plans
urban_canopy/io/           Street View, image and geospatial I/O, artifacts
urban_canopy/models/       Backend adapters, taxonomy, factory
urban_canopy/processing/   Coverage, refinement, aggregation, instance heuristic
urban_canopy/evaluation/   COCO ground truth, metrics, prediction interchange
urban_canopy/tests/        Offline, CPU-only unit tests
docs/                      Architecture, annotation protocol, evaluation method
notebooks/                 Two worked examples, runnable without an API key
samples/images/            Small curated image set for trying the pipeline
```

## Trying it without an API key

`samples/images/` holds seven curated frames spanning 0% to 40% canopy —
including a no-trees negative case and a four-heading sweep of one location.
`notebooks/` walks through them:

```bash
python -m pip install -e ".[ml,notebooks]"
jupyter lab notebooks/
```

- [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) — one image
  end to end: the indicator, the tree/vegetation split, raw vs refined masks,
  and the refinement growth guard.
- [`02_multiview_and_evaluation.ipynb`](notebooks/02_multiview_and_evaluation.ipynb)
  — multi-view aggregation and the three evaluation levels.

## Setup

Needs Python 3.10 or newer on Windows or Linux (CI tests 3.10 and 3.13).

**Linux** — Debian/Ubuntu do not ship `venv` with the interpreter, and OpenCV
links against libGL:

```bash
sudo apt install python3-venv libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The base install is enough for the unit tests and package imports: adapter
modules keep Torch, Transformers, Pillow, Torchvision and Detectron2 imports at
construction time. Running real segmentation needs the ML layer:

```bash
python -m pip install -e ".[ml]"
```

PyTorch itself is left to you: install the CPU or CUDA build matching your
machine from [pytorch.org](https://pytorch.org/get-started/locally/). If
`nvidia-smi` prints nothing, take the CPU build and run with `--device cpu`.
A venv does not inherit another venv's torch build, so check per environment —
a `+cpu` version string means `--device cuda` will fail there whatever the GPU
can do.

OneFormer and Mask2Former install with the `ml` extra and need nothing else;
their weights download into `HF_HOME` on first use (~1.7 GB for the OneFormer
default, ~850 MB for the Mask2Former one, far less for the `swin-tiny`
checkpoints). Detectron2 compiles from source (needs `build-essential
python3-dev` on Ubuntu, or Visual Studio Build Tools on Windows). See
[`docs/reproducibility.md`](docs/reproducibility.md) for all of them.

**DeepLab users:** VainF's `DeepLabV3Plus-Pytorch` is research code, not a
package — it has no `setup.py`, so `pip install -e` on it fails. You do not need
to clone it either: the pipeline imports only its self-contained `network`
package, which a helper fetches at a pinned commit (~65 KB, no git):

```bash
python scripts/fetch-deeplab.py
tree-ai --image street.jpg --seg deeplab \
        --deeplab-repo ./DeepLabV3Plus-network --ckpt <cityscapes-weights.pth>
```

`--deeplab-repo` also accepts a full or sparse clone if you have one.

Both paths are properties of the machine rather than of a run, so set them once
in `.env` and later calls need neither flag:

```ini
UC_DEEPLAB_CKPT=C:/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
UC_DEEPLAB_REPO=./DeepLabV3Plus-network
```

```bash
tree-ai --image street.jpg --seg deeplab        # both resolved from .env
```

`--ckpt` and `--deeplab-repo` still win when passed, for a one-off override. See
[`docs/reproducibility.md`](docs/reproducibility.md#backend-specific-setup).

**Detectron2 users:** it imports `pkg_resources`, which setuptools removed in
version 81, so a current environment fails with
`ModuleNotFoundError: No module named 'pkg_resources'`. Fix it with:

```bash
python -m pip install "setuptools<81"
```

This breaks identically on Linux and WSL — it is not a Windows problem.
[`docs/detectron2-windows.md`](docs/detectron2-windows.md) covers it, the
torch/`_C` coupling that forces a rebuild when torch changes, and when WSL is
actually worth the move (usually: not, if the Windows build already works).

Or use the helper:

```bash
./scripts/setup-dev.sh --api --ml                                          # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1 -WithApi -WithMl  # Windows
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` before Street View
calls. Importing modules and running unit tests never need the key.

## Running

The editable install exposes a `tree-ai` console script
(`python -m urban_canopy.cli.main` is the same entry point).

Local image, single view:

```bash
tree-ai --image street.jpg --single-view --seg oneformer --device cpu
```

Coordinates, multi-view (0/90/180/270 around a reference heading by default):

```bash
tree-ai --lat -23.678479 --lon -46.559621 --multi-view --seg oneformer
```

Address, multi-view with a known street bearing:

```bash
tree-ai "Av. Paulista 1578, Sao Paulo" --multi-view --reference-heading 45 --offsets 90,270
```

Multi-view requires at least one usable heading by default. Set a stricter
study rule with `--min-successful-views N`; failed headings are returned with
their stage (`fetch` or `analysis`) and error type instead of disappearing into
the logs.

Export everything an evaluation or audit needs — one flag, everything lands in
this run's directory:

```bash
tree-ai --image street.jpg --save-artifacts
```

### Where results go

Each invocation gets its own directory under `--outdir` (default
`artifacts_out/`), named after the timestamp and the backend:

```text
artifacts_out/
  20260818-104512_oneformer/
    run.json            manifest, aggregate, every view
    views.csv           one row per view
    predictions.json    for `tree-ai evaluate`
    views/
      000_street/       rgb.png  mask_raw.png  mask_refined.png
                        overlay_tree.png  instances.png  metrics.json
      001_...           further views, in acquisition order
```

`--save-artifacts` writes that whole bundle. The three export flags exist for
asking for one piece on its own — `--csv` alone writes the rows and no images,
which is what a large batch usually wants — and any of them accepts an explicit
path (`--csv results.csv`) to place that file elsewhere.

Local-image batches are consumed as an iterator. RGB is disabled unless image
artifacts are requested; when requested, each view is written immediately and
its RGB allocation is released before the next result accumulates.

Runs accumulate instead of overwriting, so analysing one image with OneFormer
and then with Detectron2 leaves both results side by side — which is the whole
point of supporting several backends. Name a run yourself with `--run-name`.
Nothing is written unless an output flag asks for it.

Evaluate against Roboflow COCO ground truth:

```bash
tree-ai evaluate --predictions artifacts_out/<run>/predictions.json \
                 --annotations annotations.json --report-json report.json
```

Check an annotation export before labelling more:

```bash
tree-ai validate-dataset --annotations annotations.json
```

Knobs worth knowing:

- `--no-refine` feeds the raw segmenter mask downstream (the comparison
  baseline every refinement experiment should report against).
- `--instances heuristic` derives connected components from the semantic mask;
  results carry the `instances_are_heuristic` flag.
- `--allow-vegetation-proxy` lets DeepLab's `vegetation` class stand in for
  trees; results carry `tree_from_vegetation_proxy` and
  `tree_source="vegetation_proxy"`.
- `--view-mode offsets|equiangular|fixed` with `--offsets`, `--n-views` or
  `--headings` controls the multi-view plan deterministically.
- `--min-successful-views N` aborts a multi-view run that produced too little
  imagery for the study protocol.
- DeepLab loads weights-only checkpoints by default. `--trust-checkpoint`
  enables legacy pickle loading and must only be used for a trusted file.
- Successful DeepLab runs record the checkpoint SHA-256 in the manifest.
- `--deterministic` additionally requests deterministic Torch/CUDA algorithms.
  This is stricter than `--seed`, but the manifest deliberately does not claim
  bitwise identity across different hardware or library versions.

## Web API

```bash
python -m pip install -e ".[api,ml]"
uvicorn urban_canopy.webapi:app --host 127.0.0.1 --port 8000
```

`POST /analyse/single` and `POST /analyse/multi` return the coverage metrics
(with optional base64 overlays on `/single`); `GET /ping` is a liveness probe.
Interactive docs at `/docs`. Dataset evaluation stays in the CLI.

The API has no authentication and calls a paid Google API on every request —
keep it behind a proxy or bound to localhost.

## Ground truth and evaluation

Labelling happens in Roboflow, exported as **COCO Instance Segmentation**, one
polygon/mask per tree. The pixel-level ground truth is the union of the
instances, so the two levels can never disagree about what a tree pixel is.

- Annotation policy (what counts as a tree, crowns vs trunks, occlusions,
  partial trees, minimum visibility): [`docs/annotation_protocol.md`](docs/annotation_protocol.md)
- Detectron2 on Windows, and the WSL question:
  [`docs/detectron2-windows.md`](docs/detectron2-windows.md)
- Metrics, matching rules, empty-case conventions and the validation/test
  split policy: [`docs/evaluation.md`](docs/evaluation.md)
- Architecture and the mapping from `sidewalk_analysis` components:
  [`docs/architecture.md`](docs/architecture.md)

Every prediction file embeds a manifest (package versions, model name, device,
taxonomy, refinement config, RNG seed and deterministic-runtime flags), so any
reported number can be traced to the run that produced it.

## Quality checks

```bash
python -m compileall urban_canopy -q
python -m pytest
python -m ruff check urban_canopy
python -m black --check urban_canopy
```

Or all of them at once:

```bash
./scripts/check.sh                                             # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1    # Windows
```

The default pytest suite is offline and CPU-only, enforced by `pyproject.toml`
deselecting the `gpu` and `network` markers. Run the excluded checks
deliberately with `pytest -m gpu` / `pytest -m network`, and add new
heavyweight tests under one of those markers.

## Citation

```bibtex
@misc{urban_canopy_2026,
  author = {Juan Oliveira de Carvalho},
  title = {Urban Canopy: Visible Street-Level Tree Coverage from Street View Imagery Using Semantic Segmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/juanocv/urban_canopy}
}
```
