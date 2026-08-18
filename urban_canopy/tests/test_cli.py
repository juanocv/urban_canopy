"""CLI wiring tests: local images and the evaluate/validate sub-commands.

These use the stub segmenter through monkeypatching, so they exercise argument
parsing, orchestration and export formats without any model or network.
"""

import json

import cv2
import numpy as np
import pytest

import urban_canopy.cli._builder as builder
import urban_canopy.cli.main as cli_main
from urban_canopy.evaluation.rle import encode_rle
from urban_canopy.models.base import SegmentationOutput
from urban_canopy.models.taxonomy import ADE20K


class StubSegmenter:
    backend_name = "stub"
    class_space = "ade20k"
    taxonomy = ADE20K
    supports_tree_instances = False
    model_name = "stub-model"

    def segment(self, img_rgb):
        height, width = img_rgb.shape[:2]
        tree = np.zeros((height, width), bool)
        tree[: height // 2, : width // 2] = True
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
def stub_backend(monkeypatch):
    monkeypatch.setattr(builder, "build_segmenter_from_args", lambda args, device: StubSegmenter())
    monkeypatch.setattr(builder, "resolve_device", lambda requested, parser: "cpu")


def _image(tmp_path, name="frame.jpg"):
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((40, 60, 3), np.uint8))
    return path


def test_analyse_local_image(tmp_path, capsys, stub_backend):
    image = _image(tmp_path)
    code = cli_main.main(["--image", str(image), "--single-view", "--outdir", str(tmp_path / "o")])
    assert code == 0
    out = capsys.readouterr().out
    assert "TREE COVERAGE 25.00%" in out


def test_analyse_writes_all_exports(tmp_path, stub_backend):
    image = _image(tmp_path)
    metrics = tmp_path / "metrics.json"
    csv_path = tmp_path / "rows.csv"
    predictions = tmp_path / "predictions.json"
    code = cli_main.main(
        [
            "--image",
            str(image),
            "--outdir",
            str(tmp_path / "o"),
            "--metrics-json",
            str(metrics),
            "--csv",
            str(csv_path),
            "--predictions-json",
            str(predictions),
            "--save-artifacts",
        ]
    )
    assert code == 0

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["manifest"]["model"]["backend"] == "oneformer"
    assert payload["views"][0]["coverage"]["tree_coverage_pct"] == pytest.approx(25.0)

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "tree_coverage_pct" in header

    preds = json.loads(predictions.read_text(encoding="utf-8"))
    assert preds["schema"] == "urban_canopy/predictions/1"
    assert preds["images"][0]["file_name"] == "frame.jpg"
    assert preds["images"][0]["mask"] is not None

    artifact_dir = tmp_path / "o" / "frame"
    assert (artifact_dir / "mask_raw.png").exists()
    assert (artifact_dir / "mask_refined.png").exists()
    assert (artifact_dir / "overlay_tree.png").exists()
    assert (artifact_dir / "metrics.json").exists()


def test_analyse_requires_some_input(tmp_path, capsys, stub_backend):
    with pytest.raises(SystemExit):
        cli_main.main(["--outdir", str(tmp_path)])


def test_multiview_rejects_local_images(tmp_path, stub_backend):
    image = _image(tmp_path)
    with pytest.raises(SystemExit):
        cli_main.main(["--image", str(image), "--multi-view"])


def _write_eval_fixtures(tmp_path):
    mask = np.zeros((30, 40), bool)
    mask[5:15, 5:15] = True
    annotations = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 40, "height": 30}],
        "categories": [{"id": 1, "name": "tree"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": encode_rle(mask),
                "iscrowd": 0,
            }
        ],
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")

    predictions = {
        "schema": "urban_canopy/predictions/1",
        "manifest": {},
        "images": [
            {
                "file_name": "a.jpg",
                "height": 30,
                "width": 40,
                "tree_coverage_ratio": 100 / 1200,
                "tree_coverage_pct": 100 * 100 / 1200,
                "tree_source": "tree_class",
                "valid_pixels": 1200,
                "total_pixels": 1200,
                "exclude_bottom_px": 0,
                "mask": encode_rle(mask),
                "instances": None,
                "instance_source": None,
                "quality_flags": [],
                "backend": "stub",
                "class_space": "ade20k",
            }
        ],
    }
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    return pred_path, ann_path


def test_evaluate_subcommand(tmp_path, capsys):
    pred_path, ann_path = _write_eval_fixtures(tmp_path)
    report_path = tmp_path / "report.json"
    code = cli_main.main(
        [
            "evaluate",
            "--predictions",
            str(pred_path),
            "--annotations",
            str(ann_path),
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "IoU       = 1.0000" in out
    assert "MAE  = 0.00 pp" in out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["semantic"]["micro"]["iou"] == pytest.approx(1.0)


def test_validate_dataset_subcommand(tmp_path, capsys):
    _, ann_path = _write_eval_fixtures(tmp_path)
    code = cli_main.main(["validate-dataset", "--annotations", str(ann_path)])
    assert code == 0
    assert "Dataset looks usable" in capsys.readouterr().out
