"""Offline adapter smoke tests with simulated framework processors and models."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _install_torch_and_pil(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)

    image = SimpleNamespace(fromarray=lambda array: np.asarray(array), NEAREST=0)
    pil = ModuleType("PIL")
    pil.Image = image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    return image


class _HFProcessor:
    @classmethod
    def from_pretrained(cls, model_name):
        instance = cls()
        instance.model_name = model_name
        return instance

    def __call__(self, **kwargs):
        self.call = kwargs
        return {"pixel_values": _Tensor(np.zeros((1, 3, 2, 3), dtype=np.float32))}

    def post_process_semantic_segmentation(self, outputs, *, target_sizes):
        assert outputs is _HFModel.OUTPUT
        assert target_sizes == [(2, 3)]
        return [_Tensor([[4, 4, 0], [0, 4, 0]])]


class _HFModel:
    OUTPUT = object()

    def __init__(self):
        self.config = SimpleNamespace(id2label={0: "building", 4: "tree"})

    @classmethod
    def from_pretrained(cls, model_name):
        instance = cls()
        instance.model_name = model_name
        return instance

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **inputs):
        assert inputs["pixel_values"].to("cpu") is inputs["pixel_values"]
        return self.OUTPUT


@pytest.mark.parametrize(
    ("module_name", "adapter_name", "model_name"),
    [
        (
            "urban_canopy.models.oneformer",
            "OneFormerSegmenter",
            "shi-labs/oneformer_ade20k_swin_large",
        ),
        (
            "urban_canopy.models.mask2former",
            "Mask2FormerSegmenter",
            "facebook/mask2former-swin-tiny-ade-semantic",
        ),
    ],
)
def test_huggingface_adapter_smoke_without_real_weights(
    monkeypatch, module_name, adapter_name, model_name
):
    _install_torch_and_pil(monkeypatch)
    transformers = ModuleType("transformers")
    transformers.OneFormerProcessor = _HFProcessor
    transformers.OneFormerForUniversalSegmentation = _HFModel
    transformers.Mask2FormerImageProcessor = _HFProcessor
    transformers.Mask2FormerForUniversalSegmentation = _HFModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    module = __import__(module_name, fromlist=[adapter_name])
    adapter = getattr(module, adapter_name)(model_name=model_name, device="cpu")
    output = adapter.segment(np.zeros((2, 3, 3), dtype=np.uint8))

    output.validate((2, 3))
    np.testing.assert_array_equal(
        output.group_masks["tree"], [[True, True, False], [False, True, False]]
    )
    assert output.instances is None
    assert adapter.model.evaluated is True


def test_detectron2_panoptic_adapter_smoke_without_detectron2(monkeypatch):
    _install_torch_and_pil(monkeypatch)

    class _Cfg:
        def __init__(self):
            self.MODEL = SimpleNamespace(
                WEIGHTS=None,
                ROI_HEADS=SimpleNamespace(SCORE_THRESH_TEST=None),
                DEVICE=None,
            )
            self.DATASETS = SimpleNamespace(TRAIN=("fake-dataset",))
            self.INPUT = SimpleNamespace(FORMAT="BGR")

        def merge_from_file(self, path):
            self.config_path = path

    class _MetadataCatalog:
        @staticmethod
        def get(_dataset):
            return SimpleNamespace(thing_classes=[], stuff_classes=["tree-merged"])

    class _Predictor:
        def __init__(self, cfg):
            self.cfg = cfg
            self.received = None

        def __call__(self, bgr):
            self.received = bgr.copy()
            return {
                "panoptic_seg": (
                    _Tensor([[1, 1, 0], [0, 1, 0]]),
                    [{"id": 1, "category_id": 0, "isthing": False}],
                )
            }

    detectron2 = ModuleType("detectron2")
    detectron2.__path__ = []
    config = ModuleType("detectron2.config")
    config.get_cfg = _Cfg
    data = ModuleType("detectron2.data")
    data.MetadataCatalog = _MetadataCatalog
    engine = ModuleType("detectron2.engine")
    engine.DefaultPredictor = _Predictor
    monkeypatch.setitem(sys.modules, "detectron2", detectron2)
    monkeypatch.setitem(sys.modules, "detectron2.config", config)
    monkeypatch.setitem(sys.modules, "detectron2.data", data)
    monkeypatch.setitem(sys.modules, "detectron2.engine", engine)

    from urban_canopy.models.detectron2 import Detectron2Segmenter

    adapter = Detectron2Segmenter("config.yaml", "weights.pth", device="cpu")
    rgb = np.array([[[10, 20, 30], [1, 2, 3], [4, 5, 6]]] * 2, dtype=np.uint8)
    output = adapter.segment(rgb)

    output.validate((2, 3))
    np.testing.assert_array_equal(adapter.predictor.received, rgb[:, :, ::-1])
    assert output.group_masks["tree"].sum() == 3
    assert output.instances is None


def test_deeplab_adapter_smoke_without_torchvision(monkeypatch):
    _install_torch_and_pil(monkeypatch)

    class _InputTensor:
        def unsqueeze(self, _axis):
            return self

        def to(self, _device):
            return self

    class _Transform:
        def __call__(self, _image):
            return _InputTensor()

    transforms = ModuleType("torchvision.transforms")
    transforms.Compose = lambda _steps: _Transform()
    transforms.Resize = lambda size: ("resize", size)
    transforms.ToTensor = lambda: "tensor"
    transforms.Normalize = lambda **kwargs: ("normalize", kwargs)
    torchvision = ModuleType("torchvision")
    torchvision.__path__ = []
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)

    class _Logits(_Tensor):
        def softmax(self, _axis):
            return self

        def argmax(self, _axis):
            return self

        def squeeze(self, _axis):
            return self

    class _Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

        def __call__(self, _tensor):
            return _Logits([[8, 8, 0], [0, 8, 0]])

    from urban_canopy.models.deeplab import DeepLabSegmenter

    adapter = DeepLabSegmenter(_Model(), device="cpu", input_size=(2, 3))
    output = adapter.segment(np.zeros((2, 3, 3), dtype=np.uint8))

    output.validate((2, 3))
    assert output.group_masks["vegetation"].sum() == 3
    assert output.taxonomy.tree_group is None
