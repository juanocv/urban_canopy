"""Seeding and deterministic execution are distinct, auditable settings.

This file runs **without PyTorch installed**. `set_seed` and `build_manifest`
touch torch only to flip global knobs and read version strings, so a stub covers
the real contract: which knobs are set, what the manifest records, and that a
missing torch degrades to `torch_seeded=False` instead of failing.

Stubbing is better than the real library here, not merely cheaper. It reaches
the CUDA branch on a machine without a GPU, it reaches the "torch absent" branch
on a machine that has torch, and it does not leave
`torch.use_deterministic_algorithms(True)` switched on for whatever test runs
next -- the earlier version of this file needed a `try/finally` to undo exactly
that.
"""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from urban_canopy.core import config as config_module
from urban_canopy.core.config import CanopyConfig, build_manifest, set_seed


@pytest.fixture
def isolated_env(monkeypatch):
    """Give the test its own environment copy.

    `set_seed` writes CUBLAS_WORKSPACE_CONFIG through `os.environ.setdefault`,
    which would otherwise persist for the rest of the session and change what a
    later test observes.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    # build_manifest reads this cache; reset it so each test states its own.
    monkeypatch.setattr(config_module, "_LAST_REPRODUCIBILITY", None)
    return os.environ


@pytest.fixture
def torch_stub(monkeypatch, isolated_env):
    """Install a torch exposing only the knobs `set_seed` touches."""
    calls: list[tuple] = []
    stub = ModuleType("torch")
    stub.__version__ = "2.6.0+cu118"
    stub.version = SimpleNamespace(cuda="11.8")
    stub.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
    stub.calls = calls

    stub.manual_seed = lambda seed: calls.append(("manual_seed", seed))
    stub.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: calls.append(("manual_seed_all", seed)),
    )

    def use_deterministic_algorithms(flag):
        stub._enabled = bool(flag)
        calls.append(("use_deterministic_algorithms", bool(flag)))

    stub.use_deterministic_algorithms = use_deterministic_algorithms
    stub.are_deterministic_algorithms_enabled = lambda: getattr(stub, "_enabled", False)

    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


@pytest.fixture
def torch_absent(monkeypatch, isolated_env):
    """Make `import torch` raise, whether or not torch is installed here."""

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])


# --------------------------------------------------------------------------- #
# PYTHONHASHSEED is observed, never assigned
# --------------------------------------------------------------------------- #


def test_seed_does_not_claim_to_change_python_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    status = set_seed(123)
    assert "PYTHONHASHSEED" not in os.environ
    assert status["rng_seeded"] is True
    assert status["python_hash_seed_env"] is None
    assert status["python_hash_seed_changed_at_runtime"] is False
    assert status["bitwise_determinism_guaranteed"] is False


def test_default_config_does_not_retain_rgb():
    assert CanopyConfig().keep_rgb is False


# --------------------------------------------------------------------------- #
# What deterministic=True actually switches
# --------------------------------------------------------------------------- #


def test_deterministic_torch_flags_are_recorded(torch_stub):
    status = set_seed(7, deterministic=True)

    assert status["deterministic_algorithms_requested"] is True
    assert status["torch_seeded"] is True
    assert status["torch_deterministic_algorithms"] is True
    assert status["cudnn_deterministic"] is True
    # Benchmarking picks the fastest algorithm per input shape, which is a
    # nondeterministic choice; it is switched off whatever was asked for.
    assert status["cudnn_benchmark"] is False
    assert torch_stub.backends.cudnn.benchmark is False
    assert ("manual_seed", 7) in torch_stub.calls


def test_deterministic_sets_the_cublas_workspace_before_cuda_starts(torch_stub, isolated_env):
    isolated_env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    status = set_seed(7, deterministic=True)

    assert isolated_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert status["cublas_workspace_config"] == ":4096:8"


def test_an_existing_cublas_workspace_choice_is_left_alone(torch_stub, isolated_env):
    isolated_env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_seed(7, deterministic=True)

    assert isolated_env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_seeding_without_determinism_leaves_the_flags_off(torch_stub, isolated_env):
    status = set_seed(11)

    assert status["deterministic_algorithms_requested"] is False
    assert status["torch_deterministic_algorithms"] is False
    assert status["cudnn_deterministic"] is False
    assert "CUBLAS_WORKSPACE_CONFIG" not in isolated_env


def test_cuda_generators_are_seeded_when_a_device_is_present(torch_stub):
    torch_stub.cuda.is_available = lambda: True
    status = set_seed(5)

    assert ("manual_seed_all", 5) in torch_stub.calls
    assert status["cuda_available"] is True


def test_seeding_degrades_without_torch(torch_absent):
    status = set_seed(3)

    assert status["rng_seeded"] is True
    assert status["torch_seeded"] is False
    assert "torch_deterministic_algorithms" not in status


# --------------------------------------------------------------------------- #
# What the manifest records
# --------------------------------------------------------------------------- #


def test_manifest_records_the_torch_and_cuda_versions(torch_stub):
    config = CanopyConfig(seed=7, deterministic=True)
    set_seed(7, deterministic=True)
    manifest = build_manifest(config=config, backend="stub", class_space="ade20k")

    assert manifest["packages"]["torch"] == "2.6.0+cu118"
    assert manifest["cuda"] == "11.8"
    assert manifest["reproducibility"]["rng_seeded"] is True
    assert manifest["reproducibility"]["bitwise_determinism_guaranteed"] is False


def test_manifest_reports_no_torch_rather_than_failing(torch_absent):
    manifest = build_manifest(config=CanopyConfig(seed=1), backend="stub", class_space="ade20k")

    assert manifest["packages"]["torch"] is None
    assert manifest["cuda"] is None


def test_manifest_does_not_claim_a_seeding_that_did_not_happen(isolated_env):
    """An unseeded run, or one seeded with different settings, must say so."""
    manifest = build_manifest(
        config=CanopyConfig(seed=99, deterministic=True),
        backend="stub",
        class_space="ade20k",
    )

    assert manifest["reproducibility"]["rng_seeded"] is False
    assert manifest["reproducibility"]["seed"] == 99
    assert manifest["reproducibility"]["deterministic_algorithms_requested"] is True


def test_manifest_merges_caller_supplied_fields(isolated_env):
    manifest = build_manifest(
        config=CanopyConfig(),
        backend="stub",
        class_space="ade20k",
        extra={"study": "pilot-2026", "operator": "jc"},
    )

    assert manifest["study"] == "pilot-2026"
    assert manifest["operator"] == "jc"
    assert manifest["schema"] == "urban_canopy/manifest/1"


def test_version_of_an_absent_package_is_none_not_an_error():
    """The manifest lists optional packages; missing ones read as null."""
    assert config_module._version("urban-canopy-not-a-real-distribution") is None
