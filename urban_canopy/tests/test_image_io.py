"""The ndarray colour contract is explicit: encoded input, RGB and BGR stay distinct."""

import cv2
import numpy as np
import pytest

from urban_canopy.io.image_io import decode_rgb, ensure_rgb_u8, from_bgr_array, read_rgb


def test_decode_path_and_bytes_return_rgb(tmp_path):
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    path = tmp_path / "colours.png"
    assert cv2.imwrite(str(path), bgr)
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok

    expected = np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_rgb(path), expected)
    np.testing.assert_array_equal(decode_rgb(encoded.tobytes()), expected)
    np.testing.assert_array_equal(read_rgb(path), expected)


def test_bgr_array_conversion_is_explicit():
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(from_bgr_array(bgr), [[[3, 2, 1]]])
    # Validating an RGB array does not swap channels.
    np.testing.assert_array_equal(ensure_rgb_u8(bgr), bgr)


@pytest.mark.parametrize(
    "invalid,match",
    [
        (np.zeros((2, 2), dtype=np.uint8), "H x W x 3"),
        (np.zeros((2, 2, 4), dtype=np.uint8), "H x W x 3"),
        (np.zeros((2, 2, 3), dtype=np.float32), "uint8"),
        (np.zeros((0, 2, 3), dtype=np.uint8), "non-empty"),
    ],
)
def test_rgb_validation_rejects_ambiguous_or_malformed_arrays(invalid, match):
    with pytest.raises(ValueError, match=match):
        ensure_rgb_u8(invalid)


def test_decoder_refuses_to_guess_an_array_colour_space():
    with pytest.raises(TypeError, match="from_bgr_array"):
        decode_rgb(np.zeros((2, 2, 3), dtype=np.uint8))  # type: ignore[arg-type]
