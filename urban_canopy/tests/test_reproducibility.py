"""Seeding and deterministic execution are distinct, auditable settings."""

import os

import pytest

from urban_canopy.core.config import CanopyConfig, build_manifest, set_seed


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


def test_deterministic_torch_flags_are_recorded():
    torch = pytest.importorskip("torch")
    try:
        status = set_seed(7, deterministic=True)
        assert status["deterministic_algorithms_requested"] is True
        assert status["torch_deterministic_algorithms"] is True
        assert status["cudnn_deterministic"] is True

        config = CanopyConfig(seed=7, deterministic=True)
        manifest = build_manifest(config=config, backend="stub", class_space="ade20k")
        assert manifest["reproducibility"]["rng_seeded"] is True
        assert manifest["reproducibility"]["bitwise_determinism_guaranteed"] is False
    finally:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
