"""
Standing defaults for the DeepLab backend (UC_DEEPLAB_*).

The checkpoint and the code checkout live at a fixed path on a given machine
while every other flag changes run to run, so they are configuration rather
than arguments. These tests pin the precedence -- flag beats default beats
nothing -- and the blank-value handling that a copied ``.env.example`` depends
on.
"""

import pytest

from urban_canopy.cli._argparse import build_parser
from urban_canopy.cli._builder import build_segmenter_from_args
from urban_canopy.models.deeplab import get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ("UC_DEEPLAB_CKPT", "UC_DEEPLAB_REPO", "UC_DEEPLAB_MODEL"):
        monkeypatch.delenv(key, raising=False)
    # Settings reads .env from the working directory; keep tests off the repo's.
    monkeypatch.chdir(tmp_path)


def _args(*argv):
    return build_parser().parse_args(["analyse", "--image", "x.jpg", "--seg", "deeplab", *argv])


def _checkpoint(tmp_path, name="best_deeplabv3plus_mobilenet_cityscapes_os16.pth"):
    path = tmp_path / name
    path.write_bytes(b"not a real checkpoint")
    return path


# --------------------------------------------------------------- settings ---
def test_unset_settings_are_none():
    settings = get_settings()
    assert settings.ckpt is None
    assert settings.repo is None
    assert settings.model is None


def test_blank_values_count_as_unset(monkeypatch):
    # Copying .env.example to .env leaves these keys present but empty. Without
    # special handling an empty path parses as Path('.'), which reads as
    # "configured" and fails later with a confusing message.
    monkeypatch.setenv("UC_DEEPLAB_CKPT", "")
    monkeypatch.setenv("UC_DEEPLAB_REPO", "   ")
    monkeypatch.setenv("UC_DEEPLAB_MODEL", "")
    settings = get_settings()
    assert (settings.ckpt, settings.repo, settings.model) == (None, None, None)


def test_settings_read_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(tmp_path / "w.pth"))
    monkeypatch.setenv("UC_DEEPLAB_MODEL", "deeplabv3plus_resnet101")
    settings = get_settings()
    assert settings.ckpt == tmp_path / "w.pth"
    assert settings.model == "deeplabv3plus_resnet101"


# ------------------------------------------------------------- precedence ---
def test_missing_checkpoint_everywhere_names_both_ways_to_set_it():
    with pytest.raises(ValueError) as excinfo:
        build_segmenter_from_args(_args(), "cpu")
    message = str(excinfo.value)
    assert "--ckpt" in message
    assert "UC_DEEPLAB_CKPT" in message


def test_default_checkpoint_is_used_when_the_flag_is_absent(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(checkpoint))

    captured = {}

    def fake_build(backend, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("urban_canopy.cli._builder.build_segmenter", fake_build)
    build_segmenter_from_args(_args(), "cpu")

    assert captured["ckpt_path"] == str(checkpoint)
    # Inference from the filename still runs on a default-supplied checkpoint.
    assert captured["model_name"] == "deeplabv3plus_mobilenet"


def test_flag_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(tmp_path / "ignored.pth"))
    wanted = _checkpoint(tmp_path, "best_deeplabv3plus_resnet101_cityscapes_os16.pth")

    captured = {}
    monkeypatch.setattr(
        "urban_canopy.cli._builder.build_segmenter",
        lambda backend, **kwargs: captured.update(kwargs) or object(),
    )
    build_segmenter_from_args(_args("--ckpt", str(wanted)), "cpu")

    assert captured["ckpt_path"] == str(wanted)
    assert captured["model_name"] == "deeplabv3plus_resnet101"


def test_repo_and_model_defaults_are_applied(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, "weights.pth")  # name carries no architecture
    repo = tmp_path / "DeepLabV3Plus-network"
    repo.mkdir()
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(checkpoint))
    monkeypatch.setenv("UC_DEEPLAB_REPO", str(repo))
    monkeypatch.setenv("UC_DEEPLAB_MODEL", "deeplabv3plus_mobilenet")

    captured = {}
    monkeypatch.setattr(
        "urban_canopy.cli._builder.build_segmenter",
        lambda backend, **kwargs: captured.update(kwargs) or object(),
    )
    build_segmenter_from_args(_args(), "cpu")

    assert captured["repo_path"] == str(repo)
    assert captured["model_name"] == "deeplabv3plus_mobilenet"


def test_default_pointing_at_a_missing_file_names_its_source(monkeypatch, tmp_path):
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(tmp_path / "gone.pth"))
    with pytest.raises(FileNotFoundError, match="UC_DEEPLAB_CKPT"):
        build_segmenter_from_args(_args(), "cpu")


def test_flag_pointing_at_a_missing_file_names_the_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"--ckpt"):
        build_segmenter_from_args(_args("--ckpt", str(tmp_path / "gone.pth")), "cpu")
