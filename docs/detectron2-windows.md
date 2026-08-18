# Detectron2 on Windows

Detectron2 is the one backend upstream does not officially support on Windows
("we do not provide official support for Windows"). That warning is real but
narrower than it sounds, and it is worth separating the problems that are
actually about Windows from the ones that merely happen there.

## The `pkg_resources` failure

```
File "...\detectron2\model_zoo\model_zoo.py", line 4, in <module>
    import pkg_resources
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause.** `pkg_resources` ships as part of `setuptools`, and **setuptools
removed it in version 81**. Detectron2 0.6 predates that removal and still
imports it, in one place, to locate the model-zoo config files bundled inside
the installed package.

**Fix.** Restore it in the environment:

```bash
python -m pip install "setuptools<81"
```

**This is not a Windows problem.** The same setuptools release breaks the same
Detectron2 import identically on Linux, macOS and WSL. Nothing about migrating
operating systems would have avoided it, and it is worth being explicit about
that because the traceback arrives looking like a broken native build.

Two related notes:

- Python 3.12 and newer no longer create virtualenvs with `setuptools`
  preinstalled, so a fresh venv can lack `pkg_resources` even when the system
  Python has it. The failure then looks environment-specific and is not.
- `urban_canopy` does not depend on `setuptools` and deliberately does not pin
  it: the constraint belongs to Detectron2, not to this project, and pinning it
  in `pyproject.toml` would force an obsolete build tool on everyone installing
  the OneFormer-only path. Instead, `build_segmenter("detectron2")` detects this
  exact failure and prints the command above.

## The compiled extension is tied to one torch version

A working install contains a native extension named for the exact interpreter
and platform it was built against:

```
.venv\Lib\site-packages\detectron2\_C.cp313-win_amd64.pyd
```

That file is linked against the libtorch ABI of the torch release present at
build time. **Changing torch afterwards requires rebuilding Detectron2.** This
matters most when moving from a CPU build to a CUDA one: installing a CUDA torch
into an environment whose `_C` was compiled against a CPU torch is the usual
cause of an import error or a hard crash appearing "for no reason" later.

Check what you have:

```bash
python -c "import torch, detectron2; from detectron2 import _C; print(torch.__version__, _C.__file__)"
```

## Is WSL worth it?

Not automatically, and usually not once Windows is already working. The honest
trade:

**Reasons to stay on Windows**

- Compiling `_C` with MSVC is the genuinely hard step, and it is a one-time
  cost. Once that `.pyd` exists and imports, the day-to-day experience is
  identical to Linux. Migrating discards work already paid for.
- The project lives on a Windows drive. From WSL2 that drive is reachable at
  `/mnt/...` over a translation layer whose small-file I/O is slow — and this
  pipeline reads image datasets file by file. Avoiding that means copying the
  repository into the WSL filesystem, which leaves two working copies to keep in
  sync.
- Model caches (~1.7 GB for OneFormer alone, plus zoo weights) would be
  downloaded and stored twice unless `HF_HOME`/`TORCH_HOME` are pointed across
  the boundary, which reintroduces the slow path.
- Everything except Detectron2 — OneFormer, DeepLab, the CLI, the API, the test
  suite — has no Windows caveat at all.

**Reasons WSL genuinely helps**

- **You need to change the torch version.** Rebuilding Detectron2 is
  substantially easier on Linux, and prebuilt wheels exist for common
  torch/CUDA combinations. On Windows every torch change means another MSVC
  build.
- **You want parity with CI or a deployment target**, both of which are Linux
  here.
- **The MSVC build has not succeeded yet.** If you are still fighting the
  compiler, WSL is the shorter path — that is the scenario upstream's warning is
  really about.

**Recommendation.** If `import detectron2` and `from detectron2 import _C` both
succeed on Windows, stay there and pin `setuptools<81`. Reach for WSL when a
torch change forces a rebuild, or if a future Detectron2/Python upgrade will not
compile — not because of the `pkg_resources` error, which WSL does not fix.

## Checking the environment

`tree-ai-diagnostics` reports the interpreter, torch build, CUDA availability
and whether each backend imports. The two things worth confirming for Detectron2
specifically:

```bash
python -c "import pkg_resources; print('pkg_resources ok')"
python -c "from detectron2 import model_zoo; print('model_zoo ok')"
```

A CPU-only torch reports `+cpu` in its version string, and `--device cuda` will
fail there regardless of what the GPU can do — worth checking per environment,
since a venv does not inherit the torch build of another one on the same
machine.
