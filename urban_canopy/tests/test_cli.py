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
    # The backend that actually ran, not the one requested: these tests stub
    # the segmenter, and the manifest has to name what produced the numbers.
    assert payload["manifest"]["model"]["backend"] == StubSegmenter.backend_name
    assert payload["views"][0]["coverage"]["tree_coverage_pct"] == pytest.approx(25.0)

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "tree_coverage_pct" in header

    preds = json.loads(predictions.read_text(encoding="utf-8"))
    assert preds["schema"] == "urban_canopy/predictions/2"
    assert preds["images"][0]["file_name"] == "frame.jpg"
    assert preds["images"][0]["mask_status"] == "available"
    assert preds["images"][0]["mask"] is not None

    run_dir = _only_run_dir(tmp_path / "o")
    artifact_dir = run_dir / "views" / "000_frame"
    assert (artifact_dir / "mask_raw.png").exists()
    assert (artifact_dir / "mask_refined.png").exists()
    assert (artifact_dir / "overlay_tree.png").exists()
    assert (artifact_dir / "metrics.json").exists()


def _only_run_dir(outdir):
    runs = sorted(p for p in outdir.iterdir() if p.is_dir())
    assert len(runs) == 1, f"expected exactly one run directory, found {runs}"
    return runs[0]


def test_run_directory_is_named_after_timestamp_and_backend(tmp_path, stub_backend):
    cli_main.main(
        ["--image", str(_image(tmp_path)), "--outdir", str(tmp_path / "o"), "--save-artifacts"]
    )
    run_dir = _only_run_dir(tmp_path / "o")
    assert run_dir.name.endswith(f"_{StubSegmenter.backend_name}")
    # e.g. 20260818-104512_stub
    assert len(run_dir.name.split("_")[0]) == len("20260818-104512")


def test_run_name_overrides_the_generated_id(tmp_path, stub_backend):
    cli_main.main(
        [
            "--image",
            str(_image(tmp_path)),
            "--outdir",
            str(tmp_path / "o"),
            "--run-name",
            "baseline run/1",
            "--save-artifacts",
        ]
    )
    run_dir = _only_run_dir(tmp_path / "o")
    assert run_dir.name == "baseline-run-1"


def test_two_runs_never_overwrite_each_other(tmp_path, stub_backend):
    """
    The regression this layout exists for: analysing one image with two backends
    used to write both into <outdir>/<image>/, and the second destroyed the first.
    """
    image = _image(tmp_path)
    outdir = tmp_path / "o"
    for name in ("run-a", "run-b"):
        cli_main.main(
            [
                "--image",
                str(image),
                "--outdir",
                str(outdir),
                "--run-name",
                name,
                "--save-artifacts",
                "--metrics-json",
            ]
        )

    runs = sorted(p.name for p in outdir.iterdir() if p.is_dir())
    assert runs == ["run-a", "run-b"]
    for name in runs:
        assert (outdir / name / "run.json").exists()
        assert (outdir / name / "views" / "000_frame" / "metrics.json").exists()


def test_reused_run_name_gets_a_suffix_instead_of_merging(tmp_path, stub_backend):
    image = _image(tmp_path)
    outdir = tmp_path / "o"
    for _ in range(2):
        cli_main.main(
            [
                "--image",
                str(image),
                "--outdir",
                str(outdir),
                "--run-name",
                "same",
                "--save-artifacts",
            ]
        )
    assert sorted(p.name for p in outdir.iterdir()) == ["same", "same-2"]


def test_exports_without_a_path_land_in_the_run_directory(tmp_path, stub_backend):
    outdir = tmp_path / "o"
    code = cli_main.main(
        [
            "--image",
            str(_image(tmp_path)),
            "--outdir",
            str(outdir),
            "--metrics-json",
            "--csv",
            "--predictions-json",
        ]
    )
    assert code == 0
    run_dir = _only_run_dir(outdir)
    assert (run_dir / "run.json").exists()
    assert (run_dir / "views.csv").exists()
    assert (run_dir / "predictions.json").exists()
    # Nothing was scattered into the working directory.
    assert not (tmp_path / "run.json").exists()


def test_no_output_requested_creates_no_directory(tmp_path, stub_backend):
    outdir = tmp_path / "o"
    code = cli_main.main(["--image", str(_image(tmp_path)), "--outdir", str(outdir)])
    assert code == 0
    assert not outdir.exists()


def test_cli_retains_rgb_only_for_artifact_images():
    from urban_canopy.cli._argparse import build_parser
    from urban_canopy.cli._builder import config_from_args

    parser = build_parser()
    plain = parser.parse_args(["analyse", "--image", "x.jpg"])
    artifacts = parser.parse_args(["analyse", "--image", "x.jpg", "--save-artifacts"])
    deterministic = parser.parse_args(["analyse", "--image", "x.jpg", "--deterministic"])
    assert config_from_args(plain).keep_rgb is False
    assert config_from_args(artifacts).keep_rgb is True
    assert config_from_args(deterministic).deterministic is True


def test_analyse_requires_some_input(tmp_path, capsys, stub_backend):
    with pytest.raises(SystemExit):
        cli_main.main(["--outdir", str(tmp_path)])


def test_multiview_rejects_local_images(tmp_path, stub_backend):
    image = _image(tmp_path)
    with pytest.raises(SystemExit):
        cli_main.main(["--image", str(image), "--multi-view"])


def test_multiview_rejects_impossible_success_minimum(tmp_path, stub_backend):
    with pytest.raises(SystemExit):
        cli_main.main(
            [
                "--lat",
                "-23",
                "--lon",
                "-46",
                "--multi-view",
                "--offsets",
                "0,90",
                "--min-successful-views",
                "3",
            ]
        )


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
        "schema": "urban_canopy/predictions/2",
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
                "mask_status": "available",
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


def test_save_artifacts_implies_every_export(tmp_path, stub_backend):
    """
    --save-artifacts is the "give me everything" flag. Asking for the audit
    bundle should not also require spelling out the three export flags.
    """
    outdir = tmp_path / "o"
    code = cli_main.main(
        ["--image", str(_image(tmp_path)), "--outdir", str(outdir), "--save-artifacts"]
    )
    assert code == 0

    run_dir = _only_run_dir(outdir)
    assert (run_dir / "run.json").exists()
    assert (run_dir / "views.csv").exists()
    assert (run_dir / "predictions.json").exists()
    assert (run_dir / "views" / "000_frame" / "overlay_tree.png").exists()


def test_export_flags_still_work_alone(tmp_path, stub_backend):
    # --csv on its own must stay cheap: rows, no images, no prediction masks.
    outdir = tmp_path / "o"
    cli_main.main(["--image", str(_image(tmp_path)), "--outdir", str(outdir), "--csv"])

    run_dir = _only_run_dir(outdir)
    assert (run_dir / "views.csv").exists()
    assert not (run_dir / "run.json").exists()
    assert not (run_dir / "predictions.json").exists()
    assert not any((run_dir / "views").iterdir())


def test_explicit_path_overrides_the_implied_export(tmp_path, stub_backend):
    outdir = tmp_path / "o"
    custom = tmp_path / "elsewhere" / "rows.csv"
    cli_main.main(
        [
            "--image",
            str(_image(tmp_path)),
            "--outdir",
            str(outdir),
            "--save-artifacts",
            "--csv",
            str(custom),
        ]
    )
    run_dir = _only_run_dir(outdir)
    assert custom.exists()
    assert not (run_dir / "views.csv").exists()
    # The other two implied exports still land in the run directory.
    assert (run_dir / "run.json").exists()
    assert (run_dir / "predictions.json").exists()


def test_seg_model_drives_the_taxonomy(monkeypatch, tmp_path):
    """
    A Cityscapes checkpoint must select the Cityscapes taxonomy, not the
    backend's default ADE20K one. Getting this wrong would report a tree ratio
    for a class space that has no tree class.
    """
    from urban_canopy.cli._argparse import build_parser
    from urban_canopy.cli._builder import build_segmenter_from_args

    captured = {}
    monkeypatch.setattr(
        "urban_canopy.cli._builder.build_segmenter",
        lambda backend, **kwargs: captured.update(backend=backend, **kwargs) or object(),
    )

    args = build_parser().parse_args(
        [
            "analyse",
            "--image",
            "x.jpg",
            "--seg",
            "mask2former",
            "--seg-model",
            "facebook/mask2former-swin-tiny-cityscapes-semantic",
        ]
    )
    build_segmenter_from_args(args, "cpu")

    assert captured["backend"] == "mask2former"
    assert captured["model_name"] == "facebook/mask2former-swin-tiny-cityscapes-semantic"
    assert captured["taxonomy"].class_space == "cityscapes"
    assert captured["taxonomy"].tree_group is None


def test_mask2former_defaults_to_ade20k(monkeypatch):
    from urban_canopy.cli._argparse import build_parser
    from urban_canopy.cli._builder import build_segmenter_from_args

    captured = {}
    monkeypatch.setattr(
        "urban_canopy.cli._builder.build_segmenter",
        lambda backend, **kwargs: captured.update(kwargs) or object(),
    )
    args = build_parser().parse_args(["analyse", "--image", "x.jpg", "--seg", "mask2former"])
    build_segmenter_from_args(args, "cpu")

    assert captured["taxonomy"].class_space == "ade20k"
    # No task is forwarded: a Mask2Former checkpoint's task is fixed by its name.
    assert "task" not in captured
    assert "model_name" not in captured
