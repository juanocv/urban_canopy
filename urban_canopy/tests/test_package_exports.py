import urban_canopy as uc


def test_package_import_keeps_heavy_exports_lazy():
    assert "CanopyPipeline" in uc.__all__
    assert "build_segmenter" in uc.__all__
    assert "CanopyPipeline" not in uc.__dict__


def test_light_helpers_are_immediate():
    assert uc.Coordinate(0.0, 0.0) is not None
    assert callable(uc.haversine)


def test_model_factory_import_keeps_backends_lazy():
    from urban_canopy.models.factory import build_segmenter

    assert callable(build_segmenter)


def test_adapter_modules_import_without_ml_dependencies():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import builtins
        real_import = builtins.__import__
        blocked = ("torch", "torchvision", "transformers", "PIL", "detectron2")
        def guarded(name, *args, **kwargs):
            if name in blocked or name.startswith(tuple(item + "." for item in blocked)):
                raise ModuleNotFoundError(name, name=name)
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        import urban_canopy.models.oneformer
        import urban_canopy.models.mask2former
        import urban_canopy.models.deeplab
        import urban_canopy.models.detectron2
        """)
    subprocess.run([sys.executable, "-c", script], check=True)
