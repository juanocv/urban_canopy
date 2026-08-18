"""Web API tests with a stub segmenter and stubbed Street View I/O."""

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import urban_canopy as uc  # noqa: E402
from urban_canopy import webapi  # noqa: E402
from urban_canopy.io.streetview import Settings, StreetViewClient  # noqa: E402
from urban_canopy.models.base import SegmentationOutput  # noqa: E402
from urban_canopy.models.taxonomy import ADE20K  # noqa: E402


class StubSegmenter:
    backend_name = "stub"
    class_space = "ade20k"
    taxonomy = ADE20K
    supports_tree_instances = False

    def segment(self, img_rgb):
        height, width = img_rgb.shape[:2]
        tree = np.zeros((height, width), bool)
        tree[: height // 2, :] = True
        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks={
                "tree": tree,
                "grass": np.zeros_like(tree),
                "plant_shrub": np.zeros_like(tree),
            },
        )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    frame = tmp_path / "sv.jpg"
    cv2.imwrite(str(frame), np.zeros((40, 60, 3), np.uint8))

    monkeypatch.setattr(uc, "build_segmenter", lambda *a, **k: StubSegmenter())
    monkeypatch.setattr(
        uc,
        "StreetViewClient",
        lambda *a, **k: StreetViewClient(
            cache_dir=tmp_path / "cache", settings=Settings(google_api_key="test-key")
        ),
    )
    monkeypatch.setattr(StreetViewClient, "fetch", lambda self, req: frame)
    monkeypatch.setattr(StreetViewClient, "geocode", lambda self, addr: (-23.0, -46.0))
    monkeypatch.setattr(StreetViewClient, "metadata", lambda self, lat, lon: {})

    with TestClient(webapi.app) as test_client:
        yield test_client


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_view(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0, "heading": 90})
    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["tree_coverage_pct"] == pytest.approx(50.0)
    assert payload["coverage"]["tree_source"] == "tree_class"
    assert payload["capture"]["heading"] == 90


def test_single_view_by_address(client):
    response = client.post("/analyse/single", json={"address": "Av. Paulista 1578"})
    assert response.status_code == 200
    assert response.json()["capture"]["address"] == "Av. Paulista 1578"


def test_single_view_needs_a_location(client):
    response = client.post("/analyse/single", json={})
    assert response.status_code == 422


def test_single_view_overlays(client):
    response = client.post(
        "/analyse/single",
        json={"lat": -23.0, "lon": -46.0, "return_overlays": True},
    )
    assert response.status_code == 200
    overlays = response.json()["overlays"]
    assert set(overlays) == {"rgb_png_b64", "overlay_tree_png_b64", "mask_refined_png_b64"}


def test_multi_view(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90, 180, 270]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["tree_coverage"]["n_valid_views"] == 4
    assert payload["aggregate"]["tree_coverage"]["median"] == pytest.approx(0.5)
    assert len(payload["views"]) == 4
    assert "instance_counts_per_view" in payload["aggregate"]


def test_multi_view_equiangular(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "mode": "equiangular", "n_views": 3},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["planned_headings"] == [0, 120, 240]
