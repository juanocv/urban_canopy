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

    monkeypatch.setattr(webapi, "build_segmenter_from_settings", lambda settings: StubSegmenter())
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


def test_readiness_exposes_backend_provenance(client):
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["backend"]["backend"] == "stub"
    assert payload["backend"]["class_space"] == "ade20k"
    assert payload["backend"]["taxonomy"]["class_space"] == "ade20k"


def test_single_view(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0, "heading": 90})
    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["tree_coverage_pct"] == pytest.approx(50.0)
    assert payload["coverage"]["tree_source"] == "tree_class"
    assert payload["capture"]["heading"] == 90
    assert payload["backend_provenance"]["backend"] == "stub"


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
    assert (True, False, True) in webapi.app.state.registry._pipes


def test_single_view_without_overlays_uses_non_rgb_pipeline(client):
    response = client.post("/analyse/single", json={"lat": -23.0, "lon": -46.0})
    assert response.status_code == 200
    assert (True, False, False) in webapi.app.state.registry._pipes


@pytest.mark.parametrize(
    "payload",
    [
        {"lat": 91, "lon": 0},
        {"lat": 0, "lon": 181},
        {"lat": 0},
        {"lat": 0, "lon": 0, "size": "640*640"},
        {"lat": 0, "lon": 0, "size": "5000x640"},
    ],
)
def test_single_view_rejects_invalid_capture_configuration(client, payload):
    assert client.post("/analyse/single", json=payload).status_code == 422


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
    assert payload["backend_provenance"]["backend"] == "stub"


def test_multi_view_equiangular(client):
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "mode": "equiangular", "n_views": 3},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["planned_headings"] == [0, 120, 240]


def test_multi_view_total_failure_is_a_bad_gateway(client, monkeypatch):
    def fail(self, req):
        raise RuntimeError("imagery unavailable")

    monkeypatch.setattr(StreetViewClient, "fetch", fail)
    response = client.post(
        "/analyse/multi",
        json={"lat": -23.0, "lon": -46.0, "offsets": [0, 90]},
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["successful_headings"] == []
    assert [failure["heading"] for failure in detail["failures"]] == [0, 90]


def test_multi_view_rejects_impossible_success_minimum(client):
    response = client.post(
        "/analyse/multi",
        json={
            "lat": -23.0,
            "lon": -46.0,
            "offsets": [0, 90],
            "min_successful_views": 3,
        },
    )
    assert response.status_code == 422
    assert "distinct planned headings (2)" in response.json()["detail"]
