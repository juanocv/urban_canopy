"""Street View cache writes are atomic and corrupt entries are never reused."""

import cv2
import numpy as np
import pytest

from urban_canopy.io.streetview import ImageRequest, Settings, StreetViewClient


def _jpeg_bytes():
    image = np.zeros((640, 640, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok and len(encoded) > 1024
    return encoded.tobytes()


class _Response:
    def __init__(self, content, content_type="image/jpeg"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _Session:
    def __init__(self, response):
        self.response = response
        self.params = {}
        self.headers = {}
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.response


def _client(tmp_path, response):
    session = _Session(response)
    client = StreetViewClient(
        cache_dir=tmp_path,
        session=session,
        settings=Settings(google_api_key="test-key"),
    )
    return client, session


def test_fetch_writes_a_decodable_cache_entry(tmp_path):
    content = _jpeg_bytes()
    client, session = _client(tmp_path, _Response(content))
    path = client.fetch(ImageRequest(-23.0, -46.0))
    assert path.read_bytes() == content
    assert session.calls == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_corrupt_cache_entry_is_replaced_instead_of_reused(tmp_path):
    content = _jpeg_bytes()
    request = ImageRequest(-23.0, -46.0)
    path = tmp_path / request.filename
    path.write_bytes(b"corrupt")
    client, session = _client(tmp_path, _Response(content))

    assert client.fetch(request) == path
    assert path.read_bytes() == content
    assert session.calls == 1


def test_non_image_response_is_never_cached(tmp_path):
    client, _ = _client(tmp_path, _Response(b"x" * 2048, "application/json"))
    request = ImageRequest(-23.0, -46.0)
    with pytest.raises(RuntimeError, match="Content-Type"):
        client.fetch(request)
    assert not (tmp_path / request.filename).exists()
