"""Pipeline orchestration tests with a stub segmenter -- no models, no network."""

import cv2
import numpy as np
import pytest

from urban_canopy.core.config import CanopyConfig
from urban_canopy.core.pipeline import CanopyPipeline, MultiViewAnalysisError
from urban_canopy.core.results import QualityFlag
from urban_canopy.core.viewplan import ViewPlanConfig
from urban_canopy.io.streetview import Settings, StreetViewClient
from urban_canopy.models.base import SegmentationOutput
from urban_canopy.models.taxonomy import ADE20K, CITYSCAPES
from urban_canopy.processing.refinement import RefinementConfig


class StubSegmenter:
    """Marks the top-left quadrant as tree and a bottom strip as grass."""

    backend_name = "stub"
    class_space = "ade20k"
    taxonomy = ADE20K
    supports_tree_instances = False

    def segment(self, img_rgb):
        height, width = img_rgb.shape[:2]
        tree = np.zeros((height, width), bool)
        tree[: height // 2, : width // 2] = True
        grass = np.zeros((height, width), bool)
        grass[-height // 10 :, :] = True
        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks={"tree": tree, "grass": grass, "plant_shrub": np.zeros_like(tree)},
            supports_tree_instances=False,
        )


class NoTreeClassSegmenter:
    """Cityscapes-like: only a merged vegetation mask."""

    backend_name = "stub-cityscapes"
    class_space = "cityscapes"
    taxonomy = CITYSCAPES
    supports_tree_instances = False

    def segment(self, img_rgb):
        height, width = img_rgb.shape[:2]
        veg = np.zeros((height, width), bool)
        veg[:, : width // 4] = True
        return SegmentationOutput(
            backend=self.backend_name,
            class_space=self.class_space,
            taxonomy=self.taxonomy,
            group_masks={"vegetation": veg, "terrain": np.zeros_like(veg)},
            supports_tree_instances=False,
        )


def _image(tmp_path, name="frame.jpg", size=(80, 120)):
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((*size, 3), np.uint8))
    return path


def test_single_view_local_image(tmp_path):
    pipe = CanopyPipeline(segmenter=StubSegmenter())
    result = pipe.analyse_image(_image(tmp_path))
    # Quadrant = 25% of the frame.
    assert result.coverage.tree_coverage_ratio == pytest.approx(0.25)
    assert result.coverage.tree_source == "tree_class"
    assert result.capture.source == "local"
    assert result.raw_mask.shape == (80, 120)
    assert result.instances is None  # auto mode, backend has none


def test_no_tree_class_without_proxy_is_flagged(tmp_path):
    pipe = CanopyPipeline(segmenter=NoTreeClassSegmenter())
    result = pipe.analyse_image(_image(tmp_path))
    assert result.coverage.tree_coverage_ratio is None
    assert QualityFlag.TREE_UNAVAILABLE in result.quality_flags
    # Vegetation is still reported.
    assert result.coverage.vegetation_coverage_ratio == pytest.approx(0.25)


def test_unavailable_tree_class_exports_no_semantic_mask(tmp_path):
    from urban_canopy.evaluation.predictions import build_predictions

    result = CanopyPipeline(segmenter=NoTreeClassSegmenter()).analyse_image(_image(tmp_path))
    record = build_predictions([result])["images"][0]

    assert record["tree_source"] == "unavailable"
    assert record["mask_status"] == "unavailable"
    assert record["mask"] is None


def test_unavailable_tree_class_cannot_produce_heuristic_instances(tmp_path):
    pipe = CanopyPipeline(
        segmenter=NoTreeClassSegmenter(),
        config=CanopyConfig(instance_mode="heuristic"),
    )
    result = pipe.analyse_image(_image(tmp_path))
    assert result.instances is None
    assert result.instance_source is None


def test_no_tree_class_with_proxy_is_flagged_differently(tmp_path):
    pipe = CanopyPipeline(
        segmenter=NoTreeClassSegmenter(),
        config=CanopyConfig(allow_vegetation_proxy=True),
    )
    result = pipe.analyse_image(_image(tmp_path))
    assert result.coverage.tree_coverage_ratio == pytest.approx(0.25)
    assert result.coverage.tree_source == "vegetation_proxy"
    assert QualityFlag.TREE_FROM_PROXY in result.quality_flags


def test_refinement_disabled_keeps_raw_mask(tmp_path):
    pipe = CanopyPipeline(
        segmenter=StubSegmenter(),
        config=CanopyConfig(refinement=RefinementConfig(enabled=False)),
    )
    result = pipe.analyse_image(_image(tmp_path))
    assert (result.raw_mask == result.refined_mask).all()
    assert QualityFlag.REFINEMENT_DISABLED in result.quality_flags


def test_heuristic_instances_are_flagged(tmp_path):
    pipe = CanopyPipeline(
        segmenter=StubSegmenter(),
        config=CanopyConfig(instance_mode="heuristic"),
    )
    result = pipe.analyse_image(_image(tmp_path))
    assert result.instances is not None
    assert len(result.instances) == 1  # one connected quadrant
    assert result.instance_source == "connected_components_heuristic"
    assert QualityFlag.HEURISTIC_INSTANCES in result.quality_flags
    # The backend still reports that it cannot do real instances.
    assert result.instances_supported is False


def test_instance_mode_none(tmp_path):
    pipe = CanopyPipeline(segmenter=StubSegmenter(), config=CanopyConfig(instance_mode="none"))
    result = pipe.analyse_image(_image(tmp_path))
    assert result.instances is None


def test_complete_frame_is_the_coverage_denominator(tmp_path):
    pipe = CanopyPipeline(segmenter=StubSegmenter())
    result = pipe.analyse_image(_image(tmp_path))
    assert result.coverage.valid_pixels == 80 * 120
    assert result.coverage.total_pixels == 80 * 120
    assert result.coverage.tree_coverage_ratio == pytest.approx(0.25)


class WrongShapeSegmenter(StubSegmenter):
    def segment(self, img_rgb):
        output = super().segment(img_rgb)
        output.group_masks["tree"] = output.group_masks["tree"][:1, :]
        return output


def test_pipeline_rejects_backend_masks_with_the_wrong_shape(tmp_path):
    pipe = CanopyPipeline(segmenter=WrongShapeSegmenter())
    with pytest.raises(ValueError, match="shape"):
        pipe.analyse_image(_image(tmp_path))


def _stubbed_streetview(tmp_path, monkeypatch):
    frame = _image(tmp_path, "sv.jpg")
    client = StreetViewClient(
        cache_dir=tmp_path / "cache",
        settings=Settings(google_api_key="test-key"),
    )
    monkeypatch.setattr(StreetViewClient, "fetch", lambda self, req: frame)
    monkeypatch.setattr(StreetViewClient, "geocode", lambda self, addr: (-23.0, -46.0))
    monkeypatch.setattr(StreetViewClient, "metadata", lambda self, lat, lon: {})
    return client


def test_multiview_aggregates_headings(tmp_path, monkeypatch):
    pipe = CanopyPipeline(
        segmenter=StubSegmenter(),
        streetview=_stubbed_streetview(tmp_path, monkeypatch),
    )
    plan = ViewPlanConfig(mode="offsets", reference_heading=0, offsets=(0, 90, 180, 270))
    result = pipe.analyse_multiview(-23.0, -46.0, plan=plan)

    assert len(result.views) == 4
    assert result.aggregate.tree_coverage.n_views == 4
    assert result.aggregate.tree_coverage.n_valid_views == 4
    assert result.aggregate.tree_coverage.median == pytest.approx(0.25)
    assert result.plan["planned_headings"] == [0, 90, 180, 270]
    headings = [view.capture.heading for view in result.views]
    assert headings == [0, 90, 180, 270]


def test_multiview_counts_failed_headings(tmp_path, monkeypatch):
    client = _stubbed_streetview(tmp_path, monkeypatch)
    calls = {"n": 0}

    def flaky_fetch(self, req):
        calls["n"] += 1
        if req.heading == 90:
            raise RuntimeError("no imagery here")
        return _image(tmp_path, "sv.jpg")

    monkeypatch.setattr(StreetViewClient, "fetch", flaky_fetch)
    pipe = CanopyPipeline(segmenter=StubSegmenter(), streetview=client)
    result = pipe.analyse_multiview(
        -23.0, -46.0, plan=ViewPlanConfig(mode="offsets", offsets=(0, 90, 180))
    )
    assert len(result.views) == 2
    assert result.aggregate.tree_coverage.n_views == 3
    assert result.aggregate.tree_coverage.n_valid_views == 2
    assert [failure.to_dict() for failure in result.failures] == [
        {
            "heading": 90,
            "stage": "fetch",
            "error_type": "RuntimeError",
            "message": "no imagery here",
        }
    ]


def test_multiview_raises_when_every_heading_fails(tmp_path, monkeypatch):
    client = _stubbed_streetview(tmp_path, monkeypatch)
    monkeypatch.setattr(
        StreetViewClient,
        "fetch",
        lambda self, req: (_ for _ in ()).throw(RuntimeError("quota unavailable")),
    )
    pipe = CanopyPipeline(segmenter=StubSegmenter(), streetview=client)

    with pytest.raises(MultiViewAnalysisError) as excinfo:
        pipe.analyse_multiview(
            -23.0,
            -46.0,
            plan=ViewPlanConfig(mode="offsets", offsets=(0, 90)),
        )

    error = excinfo.value
    assert error.successful_headings == ()
    assert [failure.heading for failure in error.failures] == [0, 90]
    assert {failure.stage for failure in error.failures} == {"fetch"}


def test_multiview_enforces_configured_minimum(tmp_path, monkeypatch):
    client = _stubbed_streetview(tmp_path, monkeypatch)

    def one_success(self, req):
        if req.heading != 0:
            raise RuntimeError("missing")
        return _image(tmp_path, "sv.jpg")

    monkeypatch.setattr(StreetViewClient, "fetch", one_success)
    pipe = CanopyPipeline(segmenter=StubSegmenter(), streetview=client)
    plan = ViewPlanConfig(
        mode="offsets",
        offsets=(0, 90, 180),
        min_successful_views=2,
    )
    with pytest.raises(MultiViewAnalysisError, match="at least 2"):
        pipe.analyse_multiview(-23.0, -46.0, plan=plan)


def test_coords_analysis_needs_a_client(tmp_path):
    pipe = CanopyPipeline(segmenter=StubSegmenter())
    with pytest.raises(RuntimeError, match="StreetViewClient"):
        pipe.analyse_coords(-23.0, -46.0)


def test_address_records_the_address(tmp_path, monkeypatch):
    pipe = CanopyPipeline(
        segmenter=StubSegmenter(),
        streetview=_stubbed_streetview(tmp_path, monkeypatch),
    )
    result = pipe.analyse_address("Av. Paulista 1578")
    assert result.capture.address == "Av. Paulista 1578"
    assert result.capture.lat == pytest.approx(-23.0)


def test_view_result_serialises_without_arrays(tmp_path):
    import json

    pipe = CanopyPipeline(segmenter=StubSegmenter())
    result = pipe.analyse_image(_image(tmp_path))
    payload = json.dumps(result.to_dict())
    assert "tree_coverage_ratio" in payload
