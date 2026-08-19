"""DeepLab checkpoints are weights-only unless trust is explicitly granted."""

import hashlib
import pickle
from types import SimpleNamespace

import torch
import pytest

import urban_canopy.models.deeplab as deeplab


class _Model:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"weight": torch.zeros(1)}

    def load_state_dict(self, state, strict=False):
        self.loaded = (state, strict)

    def to(self, device):
        return self

    def eval(self):
        return self


def _setup(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"fixture")
    model = _Model()
    modeling = SimpleNamespace(model=lambda **kwargs: model)
    monkeypatch.setattr(deeplab, "import_deeplab_modeling", lambda repo_path: modeling)
    return checkpoint, model


def test_checkpoint_loading_is_explicitly_weights_only(tmp_path, monkeypatch):
    checkpoint, model = _setup(tmp_path, monkeypatch)
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", fake_load)
    loaded = deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")

    assert loaded is model
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert model._urban_canopy_model_name == "model"
    assert model._urban_canopy_checkpoint_sha256 == hashlib.sha256(b"fixture").hexdigest()


def test_pickle_checkpoint_is_rejected_by_default(tmp_path, monkeypatch):
    checkpoint, _ = _setup(tmp_path, monkeypatch)
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        raise pickle.UnpicklingError("unsafe global")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(ValueError, match="trust-checkpoint"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_trusted_pickle_fallback_is_explicit(tmp_path, monkeypatch):
    checkpoint, model = _setup(tmp_path, monkeypatch)
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if kwargs["weights_only"]:
            raise pickle.UnpicklingError("legacy checkpoint")
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", fake_load)
    loaded = deeplab.load_deeplab_checkpoint(
        checkpoint,
        model_name="model",
        device="cpu",
        allow_pickle=True,
    )

    assert loaded is model
    assert [call["weights_only"] for call in calls] == [True, False]
