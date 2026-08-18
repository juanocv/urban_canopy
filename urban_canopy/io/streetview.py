"""
Google Street View I/O: a thin, cached client with **no business logic**.

Carried over from the sidewalk pipeline essentially unchanged -- image
acquisition has nothing to do with what is measured in the frame -- with two
additions this project needs for reproducibility: the metadata endpoint is used
to record the panorama id and capture date of every frame, and ``ImageRequest``
is hashable so a view plan can be de-duplicated before spending quota.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from joblib import Memory
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from urban_canopy.log import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        # This class reads GOOGLE_API_KEY, which carries no UC_ prefix, so it
        # must load the whole .env file rather than a prefixed slice of it.
        # extra="ignore" then keeps the unrelated UC_* keys in the same file
        # from failing validation and breaking the import.
        extra="ignore",
    )

    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    cache_dir: Path = Path.home() / ".urban_canopy" / "cache" / "streetview"
    default_fov: int = 90
    default_size: str = "640x640"
    timeout_s: int = 10


cfg = Settings()


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """Every parameter that identifies one Street View frame."""

    lat: float
    lon: float
    heading: int = 0
    pitch: int = 0
    fov: int = cfg.default_fov
    size: str = cfg.default_size

    @property
    def filename(self) -> str:
        return (
            f"sv_{self.lat:.6f}_{self.lon:.6f}_{self.heading:03d}_"
            f"{self.pitch:02d}_{self.fov}_{self.size}.jpg"
        )


class StreetViewClient:
    """
    Fetch Street View images and metadata with local on-disk caching.

    >>> client = StreetViewClient()                       # doctest: +SKIP
    >>> path = client.fetch(ImageRequest(-23.68, -46.54))  # doctest: +SKIP
    """

    _BASE = "https://maps.googleapis.com/maps/api/streetview"
    _META = "https://maps.googleapis.com/maps/api/streetview/metadata"
    _GEO = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        settings: Settings = cfg,
    ):
        self.settings = settings
        self.cache = Memory(location=cache_dir or settings.cache_dir, compress=True)
        self.session = session or requests.Session()
        if settings.google_api_key:
            self.session.params = {"key": settings.google_api_key}
        self.session.headers.update({"User-Agent": "urban-canopy/0.1"})
        os.makedirs(self.cache.location, exist_ok=True)

    def geocode(self, address: str) -> tuple[float, float]:
        """Return (lat, lon) for a free-form address string."""
        resp = self._get(self._GEO, params={"address": address})
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"No coordinates found for address: {address!r}.")
        loc = results[0]["geometry"]["location"]
        return loc["lat"], loc["lng"]

    def fetch(self, req: ImageRequest | str) -> Path:
        """Download one Street View image, or return the cached copy."""
        if isinstance(req, str):
            lat, lon = self.geocode(req)
            req = ImageRequest(lat, lon)

        local_path = Path(self.cache.location) / req.filename
        if local_path.exists():
            logger.debug("Street View cache hit: %s", local_path)
            return local_path

        params = {
            "location": f"{req.lat},{req.lon}",
            "heading": req.heading,
            "pitch": req.pitch,
            "fov": req.fov,
            "size": req.size,
        }
        # Google occasionally answers HTTP 200 with an error placeholder image,
        # which is far smaller than any real frame.
        content = self._get(self._BASE, params=params).content
        if len(content) < 1024:
            raise RuntimeError("Street View returned an empty image.")

        local_path.write_bytes(content)
        return local_path

    def metadata(self, lat: float, lon: float) -> dict:
        """
        Panorama metadata for a coordinate.

        Google does not bill this endpoint, and it is the only way to record
        *which* panorama (and from which month) a coverage number came from, so
        the pipeline calls it once per location rather than per heading.
        """
        resp = self._get(self._META, params={"location": f"{lat},{lon}"})
        return resp.json()

    def _get(self, url: str, params: dict) -> requests.Response:
        if not self.session.params.get("key"):
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Street View API calls. "
                "Set it in the environment or in a local .env file."
            )
        start = time.perf_counter()
        resp = self.session.get(url, params=params, timeout=self.settings.timeout_s)
        try:
            resp.raise_for_status()
        except Exception:  # pragma: no cover - network failure path
            msg = resp.json().get("error_message", "")
            raise RuntimeError(f"Street View API error: {msg or resp.text}") from None
        finally:
            logger.info(
                "Street View GET %s took %.4f seconds",
                url.split("/")[-1],
                time.perf_counter() - start,
            )
        return resp
