"""Exercise model acquisition without network access or real model downloads."""

import hashlib
import io
import ssl
from urllib.error import HTTPError, URLError

import pytest

from scomnom import nichenet_resources as resources


@pytest.fixture
def model(monkeypatch, tmp_path):
    payload = b"synthetic model bytes"
    monkeypatch.setenv("SCOMNOM_NICHENET_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(resources, "_MODELS", {
        "lr_network": ("model.rds", len(payload), hashlib.md5(payload).hexdigest())
    })
    monkeypatch.setattr(resources.time, "sleep", lambda seconds: None)
    return payload, tmp_path / "7074291" / "model.rds"


def test_download_then_offline_reuse(model, monkeypatch):
    payload, path = model
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    first = resources.prepare_models("human")[0]
    assert path.read_bytes() == payload
    assert first["sha256"] == hashlib.sha256(payload).hexdigest()
    assert first["cache_hit"] is False

    def no_network(*args, **kwargs):
        pytest.fail("A verified cache must not need network access")

    monkeypatch.setattr(resources.urllib.request, "urlopen", no_network)
    second = resources.prepare_models("human")[0]
    assert second["cache_hit"] is True
    assert second["sha256"] == first["sha256"]
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(RuntimeError, match="integrity failure"):
        resources.prepare_models("human")
    assert path.read_bytes() == b"x" * len(payload)


@pytest.mark.parametrize("bad", [b"truncated", b"x" * 21, b"x" * 100])
def test_bad_download_never_published(model, monkeypatch, bad):
    _, path = model
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(bad))
    with pytest.raises(RuntimeError, match="integrity failure|exceeds declared size"):
        resources.prepare_models("human")
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


@pytest.mark.parametrize("failures", [1, 3])
def test_transient_retry_bound_and_cleanup(model, monkeypatch, caplog, failures):
    payload, path = model
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        assert kwargs["timeout"] == 60
        if len(calls) <= failures:
            raise HTTPError(url, 504, "Gateway Timeout", {}, None)
        return io.BytesIO(payload)

    monkeypatch.setattr(resources.urllib.request, "urlopen", download)
    if failures == 3:
        with pytest.raises(RuntimeError, match="3 attempt.*504"):
            resources.prepare_models("human")
        assert len(calls) == 3
        assert list(path.parent.iterdir()) == []
    else:
        resources.prepare_models("human")
        assert len(calls) == 2
    assert "504" in caplog.text


@pytest.mark.parametrize("error", [HTTPError("https://zenodo.org", 404, "Not Found", {}, None),
                                   URLError(ssl.SSLCertVerificationError("certificate failed"))])
def test_permanent_or_tls_failure_not_retried(model, monkeypatch, error):
    calls = []

    def download(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(resources.urllib.request, "urlopen", download)
    with pytest.raises(RuntimeError, match="1 attempt"):
        resources.prepare_models("human")
    assert calls == [1]


def test_partial_network_failure_clean_retry(model, monkeypatch):
    payload, path = model

    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError("read timeout")
            return super().read(3)

    responses = iter([Broken(payload), io.BytesIO(payload)])
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: next(responses))
    resources.prepare_models("human")
    assert path.read_bytes() == payload
    assert list(path.parent.iterdir()) == [path]


def test_unsupported_organism_does_not_acquire(model, monkeypatch):
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="human"):
        resources.prepare_models("mouse")


def test_cache_override_and_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("SCOMNOM_NICHENET_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert resources._cache_root() == tmp_path / "xdg" / "scOmnom" / "nichenet"
    monkeypatch.setenv("SCOMNOM_NICHENET_CACHE_DIR", str(tmp_path / "shared"))
    assert resources._cache_root() == tmp_path / "shared"
