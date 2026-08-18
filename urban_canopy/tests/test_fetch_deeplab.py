"""
The minimal DeepLab fetch helper.

Network access is not needed here: the download is one function, and everything
worth testing is what happens to the archive afterwards -- which members are
kept, and whether a hostile path can escape the destination.
"""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "fetch-deeplab.py"


@pytest.fixture(scope="module")
def fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_deeplab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(names: dict[str, bytes]) -> bytes:
    """A gzipped tarball shaped like GitHub's, with a <repo>-<sha>/ prefix."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in names.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_keeps_only_network_and_licence(fetch_module):
    wanted = fetch_module._wanted
    assert wanted("network/modeling.py")
    assert wanted("network/backbone/resnet.py")
    assert wanted("LICENSE")
    # Everything else in the repository is training scaffolding we never import.
    assert not wanted("datasets/cityscapes.py")
    assert not wanted("samples/1_image.png")
    assert not wanted("main.py")
    assert not wanted("README.md")


def test_extracts_the_wanted_members(fetch_module, tmp_path, monkeypatch):
    prefix = "DeepLabV3Plus-Pytorch-abc123"
    payload = _archive(
        {
            f"{prefix}/network/modeling.py": b"def deeplabv3plus_mobilenet(): pass\n",
            f"{prefix}/network/backbone/resnet.py": b"# backbone\n",
            f"{prefix}/LICENSE": b"MIT\n",
            f"{prefix}/samples/big.png": b"x" * 5000,
            f"{prefix}/main.py": b"# training entry point\n",
        }
    )

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", lambda *a, **k: _Response())

    written = fetch_module.fetch(tmp_path)
    relative = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in written)

    assert relative == ["LICENSE", "network/backbone/resnet.py", "network/modeling.py"]
    assert not (tmp_path / "samples").exists()
    assert not (tmp_path / "main.py").exists()
    assert (tmp_path / "network" / "modeling.py").read_text().startswith("def deeplabv3plus")


def test_path_traversal_is_refused(fetch_module, tmp_path, monkeypatch):
    # A tar member is attacker-controlled input, and Python's extract does not
    # check containment for you.
    payload = _archive({"repo-sha/network/../../escaped.py": b"pwned\n"})

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", lambda *a, **k: _Response())

    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="traversal"):
        fetch_module.fetch(dest)
    assert not (tmp_path / "escaped.py").exists()


def test_commit_is_pinned(fetch_module):
    # An unpinned fetch would make two machines disagree about the model code.
    assert len(fetch_module.COMMIT) == 40
    assert fetch_module.COMMIT in fetch_module.TARBALL
