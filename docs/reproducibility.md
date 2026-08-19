# Reproducibility Notes

## Model assets and caches

| Backend | Weights source | Cache location |
|---|---|---|
| OneFormer | `shi-labs/oneformer_ade20k_swin_large` via HuggingFace | `HF_HOME` (~1.7 GB) |
| Mask2Former | `facebook/mask2former-swin-large-ade-semantic` via HuggingFace | `HF_HOME` (~850 MB) |
| Detectron2 panoptic | model zoo `COCO-PanopticSegmentation/panoptic_fpn_R_50_3x` | `FVCORE_CACHE`/torch cache |
| DeepLab | VainF `DeepLabV3Plus-Pytorch` Cityscapes checkpoint (manual download) | wherever you keep `--ckpt` |

Set `HF_HOME` and `TORCH_HOME` in `.env` if `~/.cache` is not where model
downloads belong. They are cached; only the first run pays.

## Backend-specific setup

**OneFormer** needs only the `ml` extra (`transformers`).

**Mask2Former** also needs only the `ml` extra. It is deliberately *not*
installed from
[facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former):
that repository builds on Detectron2 and compiles custom CUDA ops
(MultiScaleDeformableAttention) from source, which on Windows means the whole
MSVC toolchain story again. The `transformers` port is the same architecture
loading the same published weights, with no build step.

Its value here is that it publishes weights for several datasets, so the class
space is a property of the checkpoint rather than of the backend:

```bash
tree-ai --seg mask2former                                  # ADE20K, has a tree class
tree-ai --seg mask2former --seg-model facebook/mask2former-swin-large-coco-panoptic
tree-ai --seg mask2former --seg-model facebook/mask2former-swin-tiny-cityscapes-semantic
```

`--seg-model` also works for `--seg oneformer`. The dataset token in the name
(`ade`, `coco`, `cityscapes`) selects the taxonomy, and the task token
(`semantic`, `panoptic`) selects the post-processing — a Mask2Former checkpoint
is trained for one task, so the name decides rather than a flag. A checkpoint
naming no recognised dataset, such as the Mapillary Vistas ones, is refused
rather than guessed at; pass `--taxonomy` to state the mapping yourself.

Use `swin-tiny` checkpoints when trying things out: same interface, a fraction
of the download.

**Detectron2** compiles from source:

```bash
# Linux
sudo apt install build-essential python3-dev
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"
```

On Windows use the upstream instructions (Visual Studio Build Tools required).
Instance mode additionally needs your fine-tuned config and weights
(`--d2-config`, `--d2-weights`); its thing classes must include a class the
taxonomy maps to `tree`.

See [Detectron2 on Windows](detectron2-windows.md) for the `pkg_resources`
failure, the torch/`_C` coupling, and an assessment of whether WSL is worth it.

**DeepLab** needs VainF's repository plus a Cityscapes checkpoint. The
repository is research code, **not an installable package** — it ships no
`setup.py` and no `pyproject.toml`, so `pip install -e` on it fails with:

```
ERROR: ... does not appear to be a Python project:
neither 'setup.py' nor 'pyproject.toml' found.
```

Point `--deeplab-repo` at a checkout instead. Nothing is installed, and the
checkout is not modified.

You do not need the whole repository. The pipeline imports exactly one thing
from it — `network.modeling` — and that package is self-contained: it depends on
torch and numpy, and on nothing else in the repository. Everything else there is
training scaffolding (`datasets/`, `metrics/`, `main.py`) and 2.1 MB of demo
images.

**Recommended — fetch only what is needed (no git, pinned commit):**

```bash
python scripts/fetch-deeplab.py            # -> ./DeepLabV3Plus-network, ~65 KB

tree-ai --image street.jpg --seg deeplab \
        --deeplab-repo ./DeepLabV3Plus-network \
        --ckpt best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```

The script downloads one pinned commit as a tarball and extracts `network/` plus
the upstream `LICENSE`, so two machines get byte-identical model code.

**With git, if you prefer a real checkout** — sparse and shallow, 262 KB instead
of 11 MB:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/VainF/DeepLabV3Plus-Pytorch
cd DeepLabV3Plus-Pytorch && git sparse-checkout set network
```

**Full clone** (`git clone https://github.com/VainF/DeepLabV3Plus-Pytorch`) also
works and is the right choice only if you intend to train or evaluate with
upstream's own scripts.

| Approach | Downloaded | On disk | Needs git | Pinned |
|---|---|---|---|---|
| `scripts/fetch-deeplab.py` | 2.2 MB | **65 KB** | no | yes |
| Sparse + shallow clone | 262 KB | 262 KB | yes (2.25+) | no |
| Full clone | ~11 MB | 11 MB | yes | no |

All three produce identical results; the choice is only about footprint and
whether you want upstream's git history.

Checkpoints come from the [upstream
README](https://github.com/VainF/DeepLabV3Plus-Pytorch#results); the loader
infers the architecture from the filename and refuses a checkpoint that does not
fit the chosen backbone, so a mobilenet checkpoint cannot be silently loaded
into a resnet101.

Checkpoint loading is explicitly `weights_only=True`, independent of the
installed Torch version. A checkpoint that requires Python pickle is rejected,
because pickle can execute arbitrary code while loading.

**The upstream checkpoints linked above are exactly that kind of file**, so the
documented DeepLab path needs the opt-in on every run:

```
ValueError: This checkpoint requires Python pickle, which can execute code while
loading. Use a weights-only checkpoint, or pass --trust-checkpoint ...
```

They are published by the repository this backend is built around, so trusting
them is a reasonable decision — but it should be a decision, taken once and
recorded, rather than a silent default:

```bash
tree-ai --image street.jpg --seg deeplab --ckpt best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
        --trust-checkpoint
```

Set it once for the machine instead of repeating the flag, alongside the other
DeepLab defaults below:

```ini
# .env
UC_TRUST_CHECKPOINT=1
```

Either way the run logs a warning naming the file it trusted. Verify the file
first if it did not come from the upstream release — the SHA-256 recorded in the
manifest is what makes that verifiable afterwards.

The equivalent library option is `allow_pickle=True`; its default is `False`.
For successful DeepLab runs, the manifest also records the checkpoint's SHA-256
digest without exposing its machine-specific local path, so the exact weights
can be verified independently of the filename.

### Standing defaults

The checkpoint and the checkout sit at the same path for weeks while every other
flag changes run to run, so they are configuration rather than arguments. Set
them once — environment or `.env` — and the flags become optional:

| Variable | Replaces | Falls back to |
|---|---|---|
| `UC_DEEPLAB_CKPT` | `--ckpt` | — (required one way or the other) |
| `UC_DEEPLAB_REPO` | `--deeplab-repo` | whatever already imports as `network` |
| `UC_DEEPLAB_MODEL` | `--deeplab-model` | architecture inferred from the checkpoint filename |

```ini
# .env
UC_DEEPLAB_CKPT=C:/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
UC_DEEPLAB_REPO=./DeepLabV3Plus-network
```

```bash
tree-ai --image street.jpg --seg deeplab                       # uses both
tree-ai --image street.jpg --seg deeplab --ckpt other.pth      # flag overrides
```

Precedence is flag, then variable, then nothing. A missing file is reported
against whichever supplied it, so `does not exist` names either `--ckpt` or
`UC_DEEPLAB_CKPT` rather than leaving you to guess which one was in effect.
Blank values (`UC_DEEPLAB_CKPT=`, as shipped in `.env.example`) count as unset.

For library or notebook use, pass the same paths directly:

```python
build_segmenter("deeplab", ckpt_path=ckpt, repo_path="./DeepLabV3Plus-network")
```

A `.pth` file in site-packages is a third option, if you would rather the
checkout be importable to everything in the venv:

```bash
python -c "import sysconfig, pathlib; \
  pathlib.Path(sysconfig.get_paths()['purelib'], 'deeplab.pth') \
  .write_text(str(pathlib.Path('DeepLabV3Plus-network').resolve()))"
```

Remember what this backend can and cannot report: Cityscapes has no tree class,
so tree coverage comes back as `unavailable` and flagged. Only
`--allow-vegetation-proxy` produces a number, sourced as `vegetation_proxy`:

```text
TREE COVERAGE n/a  (source=unavailable)      # default: honest
  vegetation coverage: 34.54%
  flags: tree_coverage_unavailable

TREE COVERAGE 34.49%  (source=vegetation_proxy)   # with the proxy enabled
  flags: tree_from_vegetation_proxy
```

## Determinism

- Heading plans are pure functions of configuration (`core/viewplan.py`).
- `--seed` seeds Python, NumPy and torch. It does **not** assign
  `PYTHONHASHSEED`: Python reads that variable before interpreter startup, so a
  runtime assignment would be misleading and ineffective for the current
  process.
- `--deterministic` calls `torch.use_deterministic_algorithms(True)`, disables
  cuDNN benchmarking, enables deterministic cuDNN behavior and configures the
  cuBLAS workspace before model/CUDA initialization. An operation without a
  deterministic implementation may then fail loudly.
- The manifest separates `rng_seeded` from
  `deterministic_algorithms_requested`, records the effective Torch/cuDNN/CUDA
  flags, and always states `bitwise_determinism_guaranteed=false`: versions,
  drivers and hardware can still change floating-point results.
- Street View frames are cached by their full parameter set, and the panorama
  id + capture date are recorded per view: Google re-shoots streets, so two
  runs months apart can legitimately differ — the pano id is what tells you
  whether they should have.
- Cache entries are decoded before reuse. Downloads are decoded before an
  atomic replace, so a corrupt or interrupted write is never published as a
  valid cached frame.
- Google may serve different imagery for the same coordinates over time. For a
  frozen study, archive the fetched frames (the cache directory) alongside the
  predictions file.

## Validation and clean installations

Runtime configuration rejects non-finite coordinates, out-of-range capture
parameters and thresholds, malformed/oversized image dimensions, invalid modes
and negative or excessive morphology kernels. The same dependency-free
validators back dataclasses, CLI parsing and API schemas.

The regular CI job installs only `dev,api` and verifies that adapter modules
import without ML dependencies. A separate `ml-import-smoke` job installs the
`ml` extra. Built wheels exclude `urban_canopy.tests` and are inspected for that
contract during CI.

## Google API usage

Image requests are billed; metadata requests are not. The pipeline calls
metadata once per location, not per heading. The on-disk cache means repeated
runs of the same plan hit Google only once.
