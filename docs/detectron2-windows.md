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

A mismatch surfaces as an import error that names nothing useful:

```
ImportError: DLL load failed while importing _C: The specified procedure could not be found.
```

## Moving a venv from CPU torch to CUDA torch

Worked example, and the order matters: torch first, then rebuild Detectron2
against it.

**1. Pick a CUDA build your GPU still supports.** This is the step that decides
everything else. Maxwell cards (GTX 9xx, compute capability 5.2) are supported
by CUDA 11.8 and dropped by CUDA 12.8, and PyTorch stopped shipping cu118
wheels after 2.6. Check what a candidate build targets:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

A capability of `(5, 2)` is covered by `sm_50` in the arch list — CUDA binaries
are compatible within an architecture family, so `sm_50` code runs on `sm_52`.

**2. Install torch and torchvision as a matched pair**, into the venv only:

```bash
.venv/Scripts/python.exe -m pip install "torch==2.6.0+cu118" "torchvision==0.21.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
```

**3. Rebuild Detectron2 against the new torch**, from the same commit, in a
shell where MSVC is on the path:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_VISIBLE_DEVICES=-1
set DISTUTILS_USE_SDK=1
set MAX_JOBS=2
.venv\Scripts\python.exe -m pip install --no-build-isolation --force-reinstall --no-deps ^
  "git+https://github.com/facebookresearch/detectron2.git@<commit>"
```

`--no-build-isolation` is required: Detectron2's `setup.py` imports the torch
you just installed to decide what to compile, and an isolated build environment
would not have it. `MAX_JOBS` bounds memory — an unbounded parallel MSVC build
is a common cause of the compiler being killed mid-run.

### Why `CUDA_VISIBLE_DEVICES=-1` during the build

It makes Detectron2 compile its ops for CPU (`CppExtension`) instead of CUDA.
That is deliberate here, for two reasons:

- `setup.py` selects a CUDA build when `torch.cuda.is_available()` and a CUDA
  toolkit are both found. The toolkit it finds is whatever is installed —
  **12.8** on this machine — while torch is built against **11.8**. Compiling an
  extension with a different CUDA major version than torch does not produce a
  working binary.
- CUDA 12.8 cannot generate code for `sm_52` anyway, so a "successful" CUDA
  build would not run on the GPU.

Setting the variable to `-1` makes `torch.cuda.is_available()` false for the
build process only, which is enough to select the CPU path. Clearing
`CUDA_HOME`/`CUDA_PATH` does **not** work: torch falls back to scanning the
default toolkit install directory.

**What this costs.** The network still runs on the GPU — this only affects
Detectron2's own custom kernels (rotated boxes, deformable convolution,
accelerated COCO eval). The panoptic path this project uses never calls them:
NMS and ROIAlign come from torchvision. If a future model does need one, it
fails with a clear "not compiled with GPU support" rather than the cryptic DLL
error above. Getting full CUDA ops would mean installing the CUDA 11.8 toolkit
alongside 12.8 and rebuilding without `CUDA_VISIBLE_DEVICES=-1`.

**Measured result** on a GTX 970, `panoptic_fpn_R_50_3x`, 600x400 frame,
segmentation only, after warm-up:

| Device | Median inference |
|---|---|
| CPU | 2.09 s |
| CUDA | 0.23 s |

About 9x, and identical coverage output on both — which is the check that
matters: the device must not change the number.

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
  build. (That rebuild has since been done here successfully — see the section
  above — so the cost is known rather than hypothetical: one `pip install` with
  the right environment variables, a few minutes of compilation.)
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

## Environment isolation

Changing torch in a venv cannot affect another project on the same machine,
provided the venv was created without system site-packages:

```bash
grep include-system-site-packages .venv/pyvenv.cfg   # must say false
python -c "import sys; print(sys.prefix != sys.base_prefix)"   # must say True
```

With that, the only way to disturb a neighbouring project is to run `pip`
against the wrong interpreter, so address the venv explicitly
(`.venv/Scripts/python.exe -m pip ...`) rather than relying on which one is
active. The NVIDIA driver is shared and is not touched by any of this; CUDA
itself arrives inside the torch wheel, which is why two venvs on one machine can
hold different CUDA versions without conflict.

Take a snapshot before a torch migration so the environment can be restored:

```bash
.venv/Scripts/python.exe -m pip freeze > venv-freeze-backup.txt
```
