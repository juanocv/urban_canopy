"""A missing optional backend must explain itself, not raise a bare traceback."""

import builtins

import pytest

from urban_canopy.models.factory import build_segmenter


@pytest.fixture()
def block_import(monkeypatch):
    """Make one module name unimportable, wherever in the chain it is reached."""

    def _block(name: str):
        real_import = builtins.__import__

        def fake_import(module, *args, **kwargs):
            if module == name or module.startswith(f"{name}."):
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return real_import(module, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    return _block


def test_unknown_backend_lists_the_valid_ones():
    with pytest.raises(ValueError, match="oneformer"):
        build_segmenter("resnet")


def test_missing_detectron2_reports_the_install_hint(block_import):
    block_import("detectron2")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        build_segmenter("detectron2")
    message = str(excinfo.value)
    assert "Detectron2 segmentation backend" in message
    assert "docs/reproducibility.md" in message


def test_pkg_resources_hint_names_the_actual_fix():
    from urban_canopy.models.factory import _detectron2_hint

    hint = _detectron2_hint(
        ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
    )
    assert "setuptools<81" in hint
    # The message must not let the user conclude this is a Windows problem: the
    # same removal breaks Detectron2 identically on Linux.
    assert "Windows-specific problem" in hint


def test_other_detectron2_failures_keep_the_generic_hint():
    from urban_canopy.models.factory import DETECTRON2_INSTALL_HINT, _detectron2_hint

    exc = ModuleNotFoundError("No module named 'detectron2'", name="detectron2")
    assert _detectron2_hint(exc) == DETECTRON2_INSTALL_HINT


def test_failure_raised_at_construction_time_is_still_wrapped(monkeypatch):
    """
    Detectron2 defers part of its import chain to call time: `model_zoo` imports
    `pkg_resources` only when `from_zoo` runs. Guarding the import alone let that
    escape as a bare traceback, so the guard has to cover construction too.
    """
    import sys
    import types

    stub = types.ModuleType("urban_canopy.models.detectron2")

    class _Segmenter:
        @classmethod
        def from_zoo(cls, **kwargs):
            raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")

    stub.Detectron2Segmenter = _Segmenter
    monkeypatch.setitem(sys.modules, "urban_canopy.models.detectron2", stub)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        build_segmenter("detectron2")
    assert "setuptools<81" in str(excinfo.value)


def test_missing_transformers_reports_the_ml_extra(block_import):
    block_import("transformers")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        build_segmenter("oneformer")
    assert '".[ml]"' in str(excinfo.value)


def test_deeplab_without_checkpoint_says_so():
    with pytest.raises(ValueError, match="ckpt_path"):
        build_segmenter("deeplab")
