"""Missing optional CCC backends must never trigger dependency installation."""

import importlib
from types import SimpleNamespace

import pytest

md = importlib.import_module("scomnom.markers_and_de")


@pytest.fixture
def no_process(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dependency check attempted to launch a subprocess")
    monkeypatch.setattr(md.subprocess, "run", forbidden)


@pytest.mark.parametrize("install_missing", [False, True])
def test_missing_mebocost_never_installs(monkeypatch, no_process, install_missing):
    def missing(name):
        raise ImportError("missing backend")
    monkeypatch.setattr(md.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Automatic installation during analysis is disabled") as exc:
        md._import_mebocost_api(install_missing=install_missing)
    assert "missing backend" in str(exc.value)
    assert "installation/" in str(exc.value)


@pytest.mark.parametrize("install_missing", [False, True])
def test_existing_mebocost_remains_usable(monkeypatch, no_process, install_missing):
    api = SimpleNamespace(create_obj=lambda: None)
    monkeypatch.setattr(md.importlib, "import_module", lambda name: api)
    assert md._import_mebocost_api(install_missing=install_missing) is api


@pytest.mark.parametrize("install_missing", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_mebocost_resources_never_download(monkeypatch, tmp_path, no_process, install_missing, complete):
    repo = tmp_path / "MEBOCOST"
    config = tmp_path / "mebocost.conf"
    monkeypatch.setattr(md, "_MEBOCOST_RESOURCE_REPO", repo)
    monkeypatch.setattr(md, "_MEBOCOST_RESOURCE_CONFIG", config)
    files = md._mebocost_resource_file_map(repo)
    if complete:
        for path in files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test resource\n")
        assert md._ensure_mebocost_resource_config(install_missing=install_missing) == config
        before = config.read_bytes()
        assert md._ensure_mebocost_resource_config(install_missing=install_missing) == config
        assert config.read_bytes() == before
    else:
        with pytest.raises(RuntimeError, match="Automatic installation during analysis is disabled"):
            md._ensure_mebocost_resource_config(install_missing=install_missing)
        assert not config.exists() and not repo.exists()
