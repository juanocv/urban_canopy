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


def test_valid_pixel_mask_helpers():
    import numpy as np

    from urban_canopy.io.image_io import valid_pixel_mask

    valid = valid_pixel_mask((10, 10), exclude_bottom_px=3)
    assert valid[:7].all()
    assert not valid[7:].any()

    extra = np.zeros((10, 10), bool)
    extra[0, 0] = True
    valid = valid_pixel_mask((10, 10), exclude_bottom_px=0, extra_invalid=extra)
    assert not valid[0, 0]
    assert valid.sum() == 99
