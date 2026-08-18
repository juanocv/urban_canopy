# Reproducibility Notes

## Model assets and caches

| Backend | Weights source | Cache location |
|---|---|---|
| OneFormer | `shi-labs/oneformer_ade20k_swin_large` via HuggingFace | `HF_HOME` (~1.7 GB) |
| Detectron2 panoptic | model zoo `COCO-PanopticSegmentation/panoptic_fpn_R_50_3x` | `FVCORE_CACHE`/torch cache |
| DeepLab | VainF `DeepLabV3Plus-Pytorch` Cityscapes checkpoint (manual download) | wherever you keep `--ckpt` |

Set `HF_HOME` and `TORCH_HOME` in `.env` if `~/.cache` is not where model
downloads belong. They are cached; only the first run pays.

## Backend-specific setup

**OneFormer** needs only the `ml` extra (`transformers`).

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

For library or notebook use, pass the same path as `repo_path`:

```python
build_segmenter("deeplab", ckpt_path=ckpt, repo_path="./DeepLabV3Plus-Pytorch")
```

To avoid repeating it, drop a `.pth` file into the environment instead — this
persists for the venv without touching the checkout or the third-party code:

```bash
python -c "import sysconfig, pathlib; \
  pathlib.Path(sysconfig.get_paths()['purelib'], 'deeplab.pth') \
  .write_text(str(pathlib.Path('DeepLabV3Plus-Pytorch').resolve()))"
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
- `--seed` seeds Python, NumPy and torch; the value lands in the manifest.
- Street View frames are cached by their full parameter set, and the panorama
  id + capture date are recorded per view: Google re-shoots streets, so two
  runs months apart can legitimately differ — the pano id is what tells you
  whether they should have.
- Google may serve different imagery for the same coordinates over time. For a
  frozen study, archive the fetched frames (the cache directory) alongside the
  predictions file.

## Google API usage

Image requests are billed; metadata requests are not. The pipeline calls
metadata once per location, not per heading. The on-disk cache means repeated
runs of the same plan hit Google only once.
