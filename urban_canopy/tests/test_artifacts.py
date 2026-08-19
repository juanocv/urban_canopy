"""Run-directory layout and artifact naming."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from urban_canopy.core.results import CaptureParams
from urban_canopy.io.atomic import atomic_write_text
from urban_canopy.io.artifacts import (
    ArtifactConfig,
    RunLayout,
    artifact_stem,
    make_run_id,
    slugify,
    write_json,
    write_view_artifacts,
)
from urban_canopy.processing.coverage import TREE_SOURCE_CLASS, CoverageMetrics
from urban_canopy.processing.refinement import RefinementStats


class _Result:
    """Minimal stand-in for ViewResult, enough for the artifact writer."""

    def __init__(self, capture, *, instances=None, shape=(20, 30)):
        self.capture = capture
        self.raw_mask = np.zeros(shape, np.uint8)
        self.raw_mask[2:8, 2:8] = 1
        self.refined_mask = self.raw_mask.copy()
        self.vegetation_mask = None
        self.rgb_image = np.zeros((*shape, 3), np.uint8)
        self.instances = instances
        self.instances_supported = False
        self.instance_source = None
        self.quality_flags = ()
        self.backend = "stub"
        self.class_space = "ade20k"
        self.backend_notes = ()
        self.artifacts = {}
        self.refinement = RefinementStats(enabled=True, area_raw=36, area_refined=36)
        self.coverage = CoverageMetrics(
            valid_pixels=600,
            total_pixels=600,
            tree_pixels=36,
            tree_coverage_ratio=0.06,
            tree_coverage_pct=6.0,
            tree_source=TREE_SOURCE_CLASS,
        )

    @property
    def instance_count(self):
        return None if self.instances is None else len(self.instances)

    def to_dict(self, *, include_artifacts=True):
        return {"backend": self.backend, "coverage": self.coverage.to_dict()}


def _local(path="street.jpg"):
    return _Result(CaptureParams(source="local", image_path=path))


def _streetview(heading):
    return _Result(
        CaptureParams(
            source="streetview",
            lat=-23.678479,
            lon=-46.559621,
            heading=heading,
            pitch=0,
            fov=90,
        )
    )


# ---------------------------------------------------------------- run ids ---
def test_run_id_is_timestamp_then_backend():
    run_id = make_run_id("oneformer", now=datetime(2026, 8, 18, 10, 45, 12))
    assert run_id == "20260818-104512_oneformer"


def test_run_ids_sort_chronologically():
    early = make_run_id("x", now=datetime(2026, 8, 18, 9, 0, 0))
    late = make_run_id("x", now=datetime(2026, 8, 18, 17, 0, 0))
    assert sorted([late, early]) == [early, late]


def test_explicit_name_wins_and_is_slugified():
    assert make_run_id("oneformer", name="baseline run/2") == "baseline-run-2"


def test_slugify_never_returns_empty():
    assert slugify("///") == "run"
    assert slugify("  spaced  name ") == "spaced-name"


# ---------------------------------------------------------------- layout ---
def test_layout_creates_the_run_tree(tmp_path):
    layout = RunLayout.create(tmp_path, "run-a")
    assert layout.root == tmp_path / "run-a"
    assert layout.views.is_dir()
    assert layout.run_json == layout.root / "run.json"
    assert layout.views_csv == layout.root / "views.csv"
    assert layout.predictions_json == layout.root / "predictions.json"


def test_layout_never_reuses_an_existing_run_directory(tmp_path):
    first = RunLayout.create(tmp_path, "same")
    second = RunLayout.create(tmp_path, "same")
    third = RunLayout.create(tmp_path, "same")
    assert [p.root.name for p in (first, second, third)] == ["same", "same-2", "same-3"]


def test_layout_reservation_is_safe_under_concurrency(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        layouts = list(pool.map(lambda _: RunLayout.create(tmp_path, "parallel"), range(8)))
    roots = [layout.root for layout in layouts]
    assert len(set(roots)) == 8
    assert all(root.is_dir() for root in roots)


def test_two_backends_on_one_image_do_not_collide(tmp_path):
    """The regression this layout exists for."""
    result = _local()
    written = []
    for backend in ("oneformer", "detectron2"):
        layout = RunLayout.create(tmp_path, make_run_id(backend))
        written.append(
            write_view_artifacts(result, ArtifactConfig(outdir=layout.views), index=0)[
                "mask_refined"
            ]
        )
    assert written[0] != written[1]
    assert all(Path(p).exists() for p in written)


# ------------------------------------------------------------------ stems ---
def test_local_image_stem_uses_the_filename():
    assert artifact_stem(_local("some street.jpg"), index=0) == "000_some-street"


def test_streetview_stem_carries_the_heading():
    stem = artifact_stem(_streetview(90), index=3)
    assert stem.startswith("003_sv_")
    assert stem.endswith("_h090")


def test_multiview_stems_sort_in_acquisition_order():
    # Headings sampled out of numeric order must still list in the order they
    # were captured, which is what the index prefix guarantees.
    results = [_streetview(h) for h in (270, 0, 90)]
    stems = [artifact_stem(r, index=i) for i, r in enumerate(results)]
    assert sorted(stems) == stems
    assert [s.split("_h")[-1] for s in stems] == ["270", "000", "090"]


def test_stem_without_index_has_no_prefix():
    assert artifact_stem(_local(), index=None) == "street"


# ----------------------------------------------------------------- writes ---
def test_view_artifacts_written_and_recorded(tmp_path):
    layout = RunLayout.create(tmp_path, "run")
    result = _local()
    written = write_view_artifacts(result, ArtifactConfig(outdir=layout.views), index=0)

    view_dir = layout.views / "000_street"
    for name in ("rgb.png", "mask_raw.png", "mask_refined.png", "overlay_tree.png", "metrics.json"):
        assert (view_dir / name).exists(), name
    # Paths travel on the result so the CSV/JSON exports can point at them.
    assert result.artifacts == written
    assert "mask_refined" in written


def test_disabled_config_writes_nothing(tmp_path):
    layout = RunLayout.create(tmp_path, "run")
    result = _local()
    assert write_view_artifacts(result, ArtifactConfig(outdir=layout.views, enabled=False)) == {}
    assert not any(layout.views.iterdir())


def test_failed_image_encoding_is_reported_and_not_recorded(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, "run")
    result = _local()
    monkeypatch.setattr("urban_canopy.io.artifacts.cv2.imencode", lambda *args: (False, None))
    config = ArtifactConfig(
        outdir=layout.views,
        save_rgb=False,
        save_refined_mask=False,
        save_overlay=False,
        save_instances=False,
        save_metrics_json=False,
    )
    with pytest.raises(RuntimeError, match="failed to encode"):
        write_view_artifacts(result, config, index=0)
    assert result.artifacts == {}


def test_atomic_write_preserves_existing_target_on_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("disk failure")

    monkeypatch.setattr("urban_canopy.io.atomic.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_json_writer_emits_null_for_undefined_metrics(tmp_path):
    target = write_json(
        {"nan": float("nan"), "positive_inf": np.float32("inf")},
        tmp_path / "report.json",
    )
    raw = target.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"nan": None, "positive_inf": None}


@pytest.mark.parametrize("heading", [0, 5, 90, 359])
def test_heading_is_zero_padded_for_sorting(heading):
    assert artifact_stem(_streetview(heading), index=0).endswith(f"_h{heading:03d}")
