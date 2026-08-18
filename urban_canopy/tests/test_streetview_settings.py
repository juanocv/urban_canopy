"""io.streetview.Settings must tolerate a real .env file, not just a bare one.

Inherited regression from the sidewalk project: this class reads GOOGLE_API_KEY,
which has no UC_ prefix, so its model_config points env_file at ".env" directly.
pydantic-settings' default extra="forbid" would then reject every unrelated UC_*
key the same file carries for the other Settings classes, and the module would
fail to import as soon as someone actually filled in their .env.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from urban_canopy.io.streetview import ImageRequest, Settings

ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


def test_settings_accept_the_full_env_example_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        ENV_EXAMPLE.read_text(encoding="utf-8").replace(
            "GOOGLE_API_KEY=", "GOOGLE_API_KEY=test-key"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = Settings()
    assert cfg.google_api_key == "test-key"


def test_settings_ignore_keys_meant_for_other_settings_classes(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=test-key\n"
        "UC_DEBUG=1\n"
        "UC_LOG_LEVEL=DEBUG\n"
        "UC_SEG_BACKEND=deeplab\n"
        "UC_IMG_EXCLUDE_BOTTOM_PX=20\n"
        "TORCH_HOME=/models/torch\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = Settings()
    assert cfg.google_api_key == "test-key"
    assert cfg.default_fov == 90


def test_settings_still_validate_their_own_fields(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=test-key\nDEFAULT_FOV=not-a-number\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError):
        Settings()


def test_image_request_filename_is_deterministic():
    a = ImageRequest(lat=-23.5, lon=-46.6, heading=90, pitch=0, fov=90, size="640x640")
    b = ImageRequest(lat=-23.5, lon=-46.6, heading=90, pitch=0, fov=90, size="640x640")
    assert a.filename == b.filename
    assert "090" in a.filename
