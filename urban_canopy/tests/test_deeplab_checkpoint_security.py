"""DeepLab checkpoints are weights-only unless trust is explicitly granted.

This file runs **without PyTorch installed**. The loader touches torch in
exactly two places -- ``torch.load`` and ``torch.cuda.is_available`` -- and
everything checked here is plain Python around them: the pickle refusal, the
backbone-mismatch guard, the state-key variants, the provenance fields. Stubbing
those two calls keeps the security contract covered by the offline, CPU-only job
that runs on every push, instead of only on machines that installed the ml
extra.

Nothing is lost by not using the real library: these tests always replaced
``torch.load`` with a fake, because the point is which arguments the loader
passes it, not what it returns.
"""

import hashlib
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

import urban_canopy.models.deeplab as deeplab


class _Weight:
    """Stands in for a tensor: the loader only ever reads ``.shape``."""

    def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
        self.shape = shape


class _Model:
    def __init__(self, keys: tuple[str, ...] = ("weight",)) -> None:
        self.loaded = None
        self.device = None
        self._keys = keys

    def state_dict(self):
        return {key: _Weight() for key in self._keys}

    def load_state_dict(self, state, strict=False):
        self.loaded = (state, strict)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


@pytest.fixture
def torch_stub(monkeypatch):
    """Install a torch exposing only what the loader calls."""
    stub = ModuleType("torch")
    stub.cuda = SimpleNamespace(is_available=lambda: False)

    def _unstubbed(path, **kwargs):
        raise AssertionError("this test reached torch.load without stubbing it")

    stub.load = _unstubbed
    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


def _setup(tmp_path, monkeypatch, model: _Model | None = None):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"fixture")
    model = model or _Model()
    modeling = SimpleNamespace(model=lambda **kwargs: model)
    monkeypatch.setattr(deeplab, "import_deeplab_modeling", lambda repo_path: modeling)
    return checkpoint, model


def _returning(payload, calls: list):
    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return payload

    return fake_load


# --------------------------------------------------------------------------- #
# Pickle policy
# --------------------------------------------------------------------------- #


def test_checkpoint_loading_is_explicitly_weights_only(tmp_path, monkeypatch, torch_stub):
    checkpoint, model = _setup(tmp_path, monkeypatch)
    calls = []
    torch_stub.load = _returning({"state_dict": {"weight": _Weight()}}, calls)

    loaded = deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")

    assert loaded is model
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert model._urban_canopy_model_name == "model"
    assert model._urban_canopy_checkpoint_sha256 == hashlib.sha256(b"fixture").hexdigest()


def test_pickle_checkpoint_is_rejected_by_default(tmp_path, monkeypatch, torch_stub):
    checkpoint, _ = _setup(tmp_path, monkeypatch)
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        raise pickle.UnpicklingError("unsafe global")

    torch_stub.load = fake_load
    with pytest.raises(ValueError, match="trust-checkpoint"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_trusted_pickle_fallback_is_explicit(tmp_path, monkeypatch, torch_stub):
    checkpoint, model = _setup(tmp_path, monkeypatch)
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if kwargs["weights_only"]:
            raise pickle.UnpicklingError("legacy checkpoint")
        return {"state_dict": {"weight": _Weight()}}

    torch_stub.load = fake_load
    loaded = deeplab.load_deeplab_checkpoint(
        checkpoint,
        model_name="model",
        device="cpu",
        allow_pickle=True,
    )

    assert loaded is model
    assert [call["weights_only"] for call in calls] == [True, False]


# --------------------------------------------------------------------------- #
# Inputs the loader must refuse
# --------------------------------------------------------------------------- #


def test_missing_checkpoint_names_the_path(tmp_path, monkeypatch, torch_stub):
    _setup(tmp_path, monkeypatch)
    absent = tmp_path / "not-there.pth"

    with pytest.raises(FileNotFoundError, match="not-there.pth"):
        deeplab.load_deeplab_checkpoint(absent, model_name="model", device="cpu")


def test_unknown_architecture_is_rejected(tmp_path, monkeypatch, torch_stub):
    checkpoint, _ = _setup(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="network.modeling"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="no_such_model", device="cpu")


def test_checkpoint_that_is_not_a_mapping_is_rejected(tmp_path, monkeypatch, torch_stub):
    checkpoint, _ = _setup(tmp_path, monkeypatch)
    torch_stub.load = _returning(["not", "a", "mapping"], [])

    with pytest.raises(ValueError, match="mapping of tensor weights"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")


def test_state_entry_that_is_not_a_mapping_is_rejected(tmp_path, monkeypatch, torch_stub):
    checkpoint, _ = _setup(tmp_path, monkeypatch)
    torch_stub.load = _returning({"model_state": ["not", "a", "mapping"]}, [])

    with pytest.raises(ValueError, match="not a tensor mapping"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")


def test_mismatched_backbone_refuses_a_partial_load(tmp_path, monkeypatch, torch_stub):
    """The guard that matters: a wrong checkpoint still matches *some* tensors.

    Loading it would leave the rest randomly initialised and produce a plausible
    but meaningless coverage number, so the loader refuses rather than warns.
    """
    checkpoint, _ = _setup(tmp_path, monkeypatch, _Model(keys=("a", "b", "c")))
    torch_stub.load = _returning({"a": _Weight()}, [])

    with pytest.raises(RuntimeError, match="1 of 3 tensors matched") as excinfo:
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")

    # The message has to say what to do about it, not only that it failed.
    assert "--deeplab-model" in str(excinfo.value)


def test_tensors_of_the_wrong_shape_do_not_count_as_matched(tmp_path, monkeypatch, torch_stub):
    checkpoint, _ = _setup(tmp_path, monkeypatch)
    torch_stub.load = _returning({"weight": _Weight(shape=(7, 7))}, [])

    with pytest.raises(RuntimeError, match="0 of 1 tensors matched"):
        deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu")


# --------------------------------------------------------------------------- #
# Accepted state layouts and device resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"model_state": {"weight": _Weight()}}, id="model_state"),
        pytest.param({"state_dict": {"weight": _Weight()}}, id="state_dict"),
        pytest.param({"weight": _Weight()}, id="bare-state"),
    ],
)
def test_every_published_state_layout_loads(tmp_path, monkeypatch, torch_stub, payload):
    """VainF's own releases use all three; none of them is an error."""
    checkpoint, model = _setup(tmp_path, monkeypatch)
    torch_stub.load = _returning(payload, [])

    assert deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device="cpu") is model
    assert model.loaded is not None
    state, strict = model.loaded
    assert set(state) == {"weight"}
    assert strict is False


def test_device_falls_back_to_cpu_when_cuda_is_absent(tmp_path, monkeypatch, torch_stub):
    checkpoint, model = _setup(tmp_path, monkeypatch)
    torch_stub.load = _returning({"weight": _Weight()}, [])

    deeplab.load_deeplab_checkpoint(checkpoint, model_name="model", device=None)

    assert model.device == "cpu"
