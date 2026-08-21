> 🇧🇷 **Português:** [Leia esta página em português](../docs/pt-br/notebooks.md)

# Example notebooks

Worked examples for someone who wants to understand what the pipeline produces
before wiring it into anything. Both run on the images in `samples/images/`, so
**no Google API key is needed**, and both fall back to CPU when no GPU is
present.

| Notebook | What it covers |
|---|---|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | One image end to end: building a backend, reading the coverage indicator, the tree/vegetation split, inspecting raw vs refined masks, and the refinement growth guard |
| [`02_multiview_and_evaluation.ipynb`](02_multiview_and_evaluation.ipynb) | Aggregating four headings of one location, why median and IQR are reported, and the two evaluation levels run end to end |

## Running them

```bash
python -m pip install -e ".[ml,notebooks]"
jupyter lab notebooks/
```

The first cell of each notebook resolves the repository root whether the
notebook is launched from `notebooks/` or from the project root.

Both default to `BACKEND = "oneformer"`, which downloads ~1.7 GB of weights on
first use and caches them. Change that variable to `"detectron2"` if you already
have a compiled Detectron2 install (see
[`../docs/detectron2-windows.md`](../docs/detectron2-windows.md)).

## Outputs are committed

The notebooks are stored with their outputs so the figures and numbers are
readable on GitHub without running anything. They were executed against the
sample images on CUDA; re-running them on a different backend or device will
change the numbers, which is expected. All seven sample frames are manually
annotated, so every evaluation number is scored against ground truth.

To re-execute after editing:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
