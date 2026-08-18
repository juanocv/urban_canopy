"""
Reaching VainF's DeepLabV3Plus-Pytorch checkout.

Upstream ships research code with no ``setup.py`` and no ``pyproject.toml``, so
``pip install -e`` on it fails outright. These tests pin the supported path --
point at the checkout, get ``network.modeling`` -- and pin the guidance the
failure modes must carry, because the raw errors point nowhere useful.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from urban_canopy.models.deeplab import import_deeplab_modeling


@pytest.fixture()
def fake_checkout(tmp_path):
    """A directory shaped like the upstream repository, minus the deep learning."""
    repo = tmp_path / "DeepLabV3Plus-Pytorch"
    network = repo / "network"
    network.mkdir(parents=True)
    (network / "__init__.py").write_text("", encoding="utf-8")
    (network / "modeling.py").write_text(
        textwrap.dedent("""
            def deeplabv3plus_mobilenet(num_classes=19, output_stride=16):
                return ("mobilenet", num_classes, output_stride)
            """),
        encoding="utf-8",
    )
    yield repo
    # Undo the sys.path mutation so ordering between tests cannot matter.
    sys.path[:] = [p for p in sys.path if p != str(repo.resolve())]
    for name in [m for m in sys.modules if m == "network" or m.startswith("network.")]:
        del sys.modules[name]


def test_repo_path_makes_modeling_importable(fake_checkout):
    modeling = import_deeplab_modeling(fake_checkout)
    assert hasattr(modeling, "deeplabv3plus_mobilenet")
    assert str(fake_checkout.resolve()) in sys.path


def test_repo_path_is_not_added_twice(fake_checkout):
    import_deeplab_modeling(fake_checkout)
    import_deeplab_modeling(fake_checkout)
    assert sys.path.count(str(fake_checkout.resolve())) == 1


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        import_deeplab_modeling(tmp_path / "nope")


def test_directory_without_network_is_reported(tmp_path):
    wrong = tmp_path / "some-other-repo"
    wrong.mkdir()
    with pytest.raises(FileNotFoundError, match="no 'network' directory"):
        import_deeplab_modeling(wrong)


def test_absent_checkout_explains_how_to_get_one(monkeypatch):
    # Simulate 'network' being unavailable however the interpreter was set up.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "network" or name.startswith("network."):
            raise ModuleNotFoundError("No module named 'network'", name="network")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "network", raising=False)
    monkeypatch.delitem(sys.modules, "network.modeling", raising=False)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        import_deeplab_modeling(None)

    message = str(excinfo.value)
    assert "--deeplab-repo" in message
    assert "git clone" in message
    # The instruction that actually failed for users must be contradicted here,
    # not left for them to try again.
    assert "pip install -e" in message


def test_cli_exposes_the_repo_flag():
    from urban_canopy.cli._argparse import build_parser

    args = build_parser().parse_args(
        ["analyse", "--image", "x.jpg", "--seg", "deeplab", "--deeplab-repo", "some/path"]
    )
    assert str(args.deeplab_repo) == str(Path("some/path"))
