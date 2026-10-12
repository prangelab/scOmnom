"""Checksum-verified human NicheNet models from Zenodo record 7074291."""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
from pathlib import Path
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)
_BASE_URL = "https://zenodo.org/records/7074291/files/"
# Published byte counts and MD5 checksums, not checksums learned from downloads.
_MODELS = {
    "ligand_target_matrix": ("ligand_target_matrix_nsga2r_final.rds", 262116605,
                             "b09606b04b2d4490418d9028c0e58b9f"),
    "lr_network": ("lr_network_human_21122021.rds", 20724,
                   "2a155f81e9ffffd5d5e709fe66bcc465"),
    "weighted_networks": ("weighted_networks_nsga2r_final.rds", 73662019,
                          "80014dee22df42e98d1c608731c685b5"),
}


def _cache_root() -> Path:
    override = os.environ.get("SCOMNOM_NICHENET_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else (
        Path.home() / "Library" / "Caches" if sys.platform == "darwin"
        else Path.home() / ".cache"
    )
    return (base / "scOmnom" / "nichenet").resolve()


def _verify(path: Path, size: int, expected_md5: str) -> str:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    actual_size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            actual_size += len(block)
            md5.update(block)
            sha256.update(block)
    if actual_size != size or md5.hexdigest() != expected_md5:
        raise RuntimeError(
            f"NicheNet model integrity failure: {path}; expected {size} bytes / "
            f"MD5 {expected_md5}, got {actual_size} bytes / MD5 {md5.hexdigest()}. "
            "Inspect the file before removing or replacing it; inference is stopped."
        )
    return sha256.hexdigest()


def _download(url: str, destination: Path, size: int, md5: str) -> str:
    for attempt in range(1, 4):
        # Unique same-directory staging prevents partial files becoming cache hits.
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part",
                                    dir=destination.parent)
        partial = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                LOGGER.info("Downloading NicheNet model %s (attempt %d/3)", url, attempt)
                with urllib.request.urlopen(url, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                        if handle.tell() > size:
                            raise RuntimeError(f"NicheNet model exceeds declared size: {url}")
            sha256 = _verify(partial, size, md5)
            os.replace(partial, destination)
            return sha256
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead) as exc:
            transient = not isinstance(exc, urllib.error.HTTPError) or (
                exc.code in {408, 429, 500, 502, 503, 504}
            )
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLError):
                transient = False
            if not transient or attempt == 3:
                raise RuntimeError(
                    f"NicheNet model download failed after {attempt} attempt(s): {url}: {exc}"
                ) from exc
            LOGGER.warning("NicheNet download attempt %d failed: %s; retrying", attempt, exc)
            time.sleep(2 ** attempt)
        finally:
            partial.unlink(missing_ok=True)
    raise AssertionError("Unreachable download state")


def prepare_models(organism: str) -> list[dict[str, object]]:
    """Return local paths and provenance after verifying every model's bytes."""
    if organism.lower() != "human":
        raise ValueError("scOmnom NicheNet currently supports organism='human' only.")
    root = _cache_root() / "7074291"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for model, (filename, size, md5) in _MODELS.items():
        path = root / filename
        url = _BASE_URL + filename
        cached = path.exists()
        sha256 = _verify(path, size, md5) if cached else _download(url, path, size, md5)
        LOGGER.info("Verified NicheNet model %s (%s), SHA256 %s",
                    model, "cache" if cached else "download", sha256)
        records.append({"model": model, "record": "7074291", "url": url,
                        "path": str(path), "size_bytes": size,
                        "expected_md5": md5, "sha256": sha256, "cache_hit": cached})
    return records
