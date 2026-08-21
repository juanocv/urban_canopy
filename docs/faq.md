> 🇧🇷 **Português:** [Leia esta página em português](pt-br/faq.md)

# FAQ

The README says what to run. This says why the project behaves the way it does,
and covers the things that go wrong on a fresh machine.

- [Installation](#installation)
- [Choosing a backend](#choosing-a-backend)
- [The indicator](#the-indicator)
- [Ground truth and evaluation](#ground-truth-and-evaluation)
- [Running and outputs](#running-and-outputs)
- [Development](#development)

---

## Installation

### `pip install -e` on DeepLabV3Plus-Pytorch fails with "does not appear to be a Python project"

Because it is not one. VainF's `DeepLabV3Plus-Pytorch` is research code with no
`setup.py`, so pip has nothing to install.

You do not need to clone it either. The pipeline imports exactly one thing from
it — the self-contained `network` package — and a helper fetches that at a
pinned commit, without git, at about 65 KB:

```bash
python scripts/fetch-deeplab.py
tree-ai --image street.jpg --seg deeplab \
        --deeplab-repo ./DeepLabV3Plus-network --ckpt <cityscapes-weights.pth>
```

`--deeplab-repo` also accepts a full or sparse clone if you already have one.

### Do I have to pass `--ckpt` and `--deeplab-repo` on every DeepLab call?

No. Both are properties of the machine rather than of a run, so set them once in
`.env`:

```ini
UC_DEEPLAB_CKPT=C:/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
UC_DEEPLAB_REPO=./DeepLabV3Plus-network
```

```bash
tree-ai --image street.jpg --seg deeplab        # both resolved from .env
```

The flags still win when passed, for a one-off override.

### Detectron2 fails with `ModuleNotFoundError: No module named 'pkg_resources'`

Detectron2 imports `pkg_resources`, which setuptools removed in version 81:

```bash
python -m pip install "setuptools<81"
```

This breaks identically on Linux and WSL — it is not a Windows problem.

### Should I move to WSL to get Detectron2 working?

Usually not, if the Windows build already works. The genuinely painful part is
the same everywhere: Detectron2 compiles a `_C` extension against the exact
torch build present at compile time, so changing torch forces a rebuild.

[`detectron2-windows.md`](detectron2-windows.md) covers the whole decision, the
torch/`_C` coupling, and the cases where WSL does pay for itself.

### `--device cuda` fails even though `nvidia-smi` works

Check the torch build **inside the environment you are running from**. A venv
does not inherit another venv's torch, so a `+cpu` version string means CUDA is
unavailable there whatever the GPU can do:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `+cpu`, reinstall torch from
[pytorch.org](https://pytorch.org/get-started/locally/) with the CUDA build for
your driver, or run with `--device cpu`.

### How much gets downloaded on first use?

OneFormer and Mask2Former install with the `ml` extra and need nothing else;
their weights land in `HF_HOME` on first use:

| Checkpoint | Approximate size |
|---|---|
| OneFormer ADE20K Swin-L (default) | 1.7 GB |
| Mask2Former ADE20K Swin-L (default) | 850 MB |
| `swin-tiny` variants | far less |

Detectron2 downloads its COCO-panoptic weights on first use as well, but it
compiles from source first — `build-essential python3-dev` on Ubuntu, Visual
Studio Build Tools on Windows.

### Why does `--trust-checkpoint` exist?

DeepLab loads weights-only checkpoints by default. The upstream VainF
checkpoints need Python pickle, which can execute code while loading, so they
require an explicit `--trust-checkpoint` (or `UC_TRUST_CHECKPOINT=1` in `.env`).
Use it only for a file you trust. Successful DeepLab runs record the checkpoint
SHA-256 in the manifest.

---

## Choosing a backend

### Which backend should I use?

OneFormer, unless you have a reason not to. Measured against the manual ground
truth on the seven annotated sample frames:

| Backend | IoU | Precision | Recall | Coverage MAE | Bias |
|---|---|---|---|---|---|
| OneFormer | 0.880 | 0.935 | 0.937 | 0.96 pp | +0.02 pp |
| Mask2Former | 0.833 | 0.914 | 0.904 | 1.46 pp | −0.16 pp |
| Detectron2 | 0.714 | 0.721 | 0.986 | 5.16 pp | +5.16 pp |
| DeepLab | — | — | — | — | — |

Seven frames is a small sample, and the honest reading is a claim about the
direction and rough size of the error, not a ranking between backends whose
errors overlap. What it does show clearly: Detectron2's bias equals its MAE, so
it errs the same way every time — recall 0.986 against precision 0.721 means it
paints tree coverage over things the annotator did not label. That is a calibratable
offset rather than noise.

### Why does DeepLab report no tree coverage at all?

Because Cityscapes has no tree class. Its `vegetation` class merges trees with
bushes, so there is no honest tree ratio to report, and the pipeline says
`tree_source="unavailable"` rather than passing the vegetation number off as
trees.

`--allow-vegetation-proxy` overrides this when you want it; results then carry
`tree_from_vegetation_proxy` and `tree_source="vegetation_proxy"`, so the
substitution is visible in every downstream file.

### Why is Mask2Former listed with three class spaces?

It is the one backend published for several of them, which lets you hold the
architecture fixed and vary the label set — separating "the model disagrees"
from "the dataset has no such class". On one sample frame:

```text
oneformer     tree 31.97%   vegetation 42.68%     (ADE20K)
mask2former   tree 32.69%   vegetation 42.88%     (ADE20K)
detectron2    tree 36.21%   vegetation 46.00%     (COCO-panoptic)
deeplab       tree    n/a   vegetation 34.54%     (Cityscapes — no tree class)

mask2former --seg-model facebook/mask2former-swin-tiny-cityscapes-semantic
              tree    n/a   vegetation 36.24%     (same model, no tree class)
```

The last line is the point: the same architecture reports no tree ratio when
pointed at a class space that cannot express one.

---

## The indicator

### Why does the project measure area and never count trees?

Because no model available for these class spaces can count them. That is a
finding rather than a simplification:

- COCO-80 has only `potted plant`.
- COCO-panoptic's `tree-merged` is a *stuff* class — all trees in a frame form
  one region by construction.
- LVIS v1's 1203 categories contain only `Christmas_tree`.
- Cityscapes-instance has eight person/vehicle classes.
- The ADE20K instance set (100 things) has `palm` and `flower`, but not `tree`.

Every downloadable tree instance-segmentation model — detectree2, DeepForest,
`restor/tcd-mask-rcnn-r50` — is trained on **overhead aerial** imagery, where
crowns are separated blobs. From the street they overlap and occlude, and the
recent work that tackles that does not release weights.

So a per-instance metric could only have been computed against a model that does
not exist. Individual-tree support was removed rather than left as an
unreachable code path.

### Why a continuous ratio instead of "low / medium / high greenery"?

Bands hide the threshold choice inside the result. Two studies quoting "medium
greenery" cannot be compared unless both publish their cut points, and a value
sitting near a boundary flips category on measurement noise. The continuous
ratio is the output; anyone who needs bands can apply their own thresholds to it
and state them.

### What is the denominator?

Every pixel of the image. Frames are used as delivered — no cropping of sky,
road or vehicle hood — because a denominator that varies per image makes two
measurements incomparable.

### Why is the same location reported four times?

Because a single heading is a property of the photograph, not of the place. On
the four-heading sweep in `samples/images/`, the manually labelled coverage of
one point ranges from 1.2% to 29.3% — a 28 pp swing that is not model error.

That is why multi-view runs report median and IQR over a deterministic set of
headings, and why headings come from configuration and never from the
segmentation output: choosing the view by how well the model segments it would
bias the measurement by construction.

---

## Ground truth and evaluation

### Why annotate one polygon per tree if the metric is per pixel?

Two reasons. It is what Roboflow produces naturally, and the pixel ground truth
is simply the union of those polygons — drawing a second, region-level ground
truth over the same pixels would produce two versions that disagree.

Keeping the instances costs nothing and leaves per-tree work possible later.

### Why does a frame with no trees matter?

It is the only kind of frame on which a false positive is measurable. Every
other image can punish a model for missing coverage; only a zero-tree frame can
punish it for inventing coverage.

Roboflow exports nothing for such a frame, so it has to be reinstated by hand as
an image entry with zero annotations. Keep that list somewhere visible in the
evaluation code rather than hiding it in a data file: a negative case that
disappears silently is indistinguishable from one that was never labelled.

### Why merge the annotation exports before evaluating?

Roboflow exports one COCO file per labelling job. Evaluating them one at a time
gives one report per image at *n* = 1 each, and those cannot be averaged: a
micro-averaged IoU is pooled over pixels, not over per-image scores. Merging
first gives one result at the real sample size.

### Why two evaluation levels instead of one number?

They answer different questions, and they disagree in an informative way. A mask
shifted sideways scores poorly on IoU while agreeing almost exactly on coverage:

- **Pixel level** — IoU, Dice/F1, precision, recall. Is the mask in the right
  place?
- **Indicator level** — MAE, RMSE, bias in percentage points. Is the number the
  study will publish right?

Reporting only one hides half the story. Full conventions, matching rules and
empty-case handling: [`evaluation.md`](evaluation.md).

---

## Running and outputs

### Nothing was written after my run

Nothing is written unless an output flag asks for it. `--save-artifacts` writes
the whole bundle; `--csv`, `--json` and the image flags each ask for one piece.
`--csv` alone writes the rows and no images, which is what a large batch usually
wants, and any of them accepts an explicit path (`--csv results.csv`).

### Why do runs accumulate instead of overwriting?

Because comparing backends is the point. Analysing one image with OneFormer and
then with Detectron2 leaves both results side by side, each in its own
timestamped directory under `--outdir`. Name a run yourself with `--run-name`.

### Does a large local-image batch hold every image in memory?

No. Batches are consumed as an iterator, RGB is disabled unless image artifacts
are requested, and when they are, each view is written immediately and its RGB
allocation released before the next result accumulates.

### What happens when some headings fail in a multi-view run?

Multi-view requires at least one usable heading by default; `--min-successful-views N`
sets a stricter study rule. Failed headings are returned with their stage
(`fetch` or `analysis`) and error type rather than disappearing into the logs.

### Other flags worth knowing

- `--no-refine` feeds the raw segmenter mask downstream — the comparison
  baseline every refinement experiment should report against.
- `--view-mode offsets|equiangular|fixed`, with `--offsets`, `--n-views` or
  `--headings`, controls the multi-view plan deterministically.
- `--deterministic` additionally requests deterministic Torch/CUDA algorithms.
  It is stricter than `--seed`, but the manifest deliberately does not claim
  bitwise identity across different hardware or library versions.

### Why does the Web API refuse to start?

Invalid or incomplete backend configuration aborts startup before the server
reports readiness — a server that answers `/ping` but cannot segment is worse
than one that never came up. Check the `UC_*` values against `.env.example`.

Note that the API has no authentication and calls a paid Google API on every
request. Keep it behind a proxy or bound to localhost.

---

## Development

### What do the quality gates actually enforce?

- **pytest** — 80% aggregate branch coverage, plus a per-module floor of 60%
  through `scripts/check_coverage.py`, so a well-covered package cannot hide one
  untested module.
- **Hypothesis** — property tests over RLE, mask/coverage, aggregation and
  geographic invariants.
- **Ruff** — bugbear, import sorting, modernization, simplification and
  NumPy-specific rules on top of fatal errors and undefined names.
- **Pyright** — the dependency-light public scientific contracts.

CI tests Python 3.10 and 3.13. A weekly workflow separately installs the
declared minimum dependency set and the latest compatible releases, so lower
bounds and upstream updates are executable claims rather than untested metadata.

### Why is the default test suite offline and CPU-only?

So that a clone can be verified without a GPU, an API key or a 1.7 GB download.
`pyproject.toml` deselects the `gpu` and `network` markers by default; run those
deliberately with `pytest -m gpu` / `pytest -m network`, and put new heavyweight
tests under one of the markers.
