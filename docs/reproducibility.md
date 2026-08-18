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

**DeepLab** needs VainF's repository importable as `network`:

```bash
git clone https://github.com/VainF/DeepLabV3Plus-Pytorch
pip install -e ./DeepLabV3Plus-Pytorch    # or add it to PYTHONPATH
```

plus a Cityscapes checkpoint passed with `--ckpt`. The loader infers the
architecture from the filename and refuses a checkpoint that does not fit the
chosen backbone. Remember: Cityscapes has no tree class — this backend reports
tree coverage only under `--allow-vegetation-proxy`, flagged as such.

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
