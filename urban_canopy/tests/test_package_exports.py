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
