"""Exercise installer environment routing without running R or fetching code."""

import importlib
from types import SimpleNamespace

import pytest

md = importlib.import_module("scomnom.markers_and_de")


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    lib = tmp_path / "r-library"
    monkeypatch.setattr(md, "_NICHENET_R_LIB_DIR", lib)
    monkeypatch.setattr(md, "_resolve_rscript", lambda: "/test/bin/Rscript")
    monkeypatch.setattr(md, "_resolve_r_binary", lambda: "/test/bin/R")
    monkeypatch.setattr(md.shutil, "which", lambda name: "/test/bin/git")
    monkeypatch.setenv("R_LIBS", "/unrelated/lib")
    monkeypatch.setenv("R_LIBS_SITE", "/unrelated/site")
    calls = []

    def responses(codes):
        codes = iter(codes)

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=next(codes), stdout="", stderr="")

        monkeypatch.setattr(md.subprocess, "run", run)

    return lib, calls, responses


def test_existing_nichenet_runtime_does_not_install(runtime):
    lib, calls, responses = runtime
    responses([0])
    assert md._ensure_nichenet_r_runtime(install_missing=False) == lib
    assert len(calls) == 1
    env = calls[0][1]["env"]
    assert env["R_LIBS_USER"] == str(lib)
    assert "R_LIBS" not in env


def test_missing_nichenet_runtime_requires_opt_in(runtime):
    _, calls, responses = runtime
    responses([1])
    with pytest.raises(RuntimeError, match="install-missing-r-deps"):
        md._ensure_nichenet_r_runtime(install_missing=False)
    assert len(calls) == 1


def test_nichenet_install_uses_isolated_library_for_every_install_step(runtime):
    lib, calls, responses = runtime
    responses([1, 0, 0, 0, 0])
    assert md._ensure_nichenet_r_runtime(install_missing=True) == lib
    assert len(calls) == 5
    assert calls[2][0][1] == "clone"
    assert calls[3][0][1:3] == ["CMD", "INSTALL"]
    for _, kwargs in calls[1:4]:
        env = kwargs["env"]
        assert env["R_LIBS_USER"] == env["R_LIBS"] == str(lib)
        assert "R_LIBS_SITE" not in env
    assert "R_LIBS" not in calls[-1][1]["env"]


def test_nichenet_bootstrap_failure_stops_before_clone(runtime):
    _, calls, responses = runtime
    responses([1, 1])
    with pytest.raises(RuntimeError, match="bootstrapping required CRAN packages"):
        md._ensure_nichenet_r_runtime(install_missing=True)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "codes, message",
    [
        ([1, 0, 1], "source download failed"),
        ([1, 0, 0, 1], "dependency installation failed"),
        ([1, 0, 0, 0, 1], "without a detectable"),
    ],
)
def test_nichenet_later_install_failures_are_reported(runtime, codes, message):
    _, calls, responses = runtime
    responses(codes)
    with pytest.raises(RuntimeError, match=message):
        md._ensure_nichenet_r_runtime(install_missing=True)
    assert len(calls) == len(codes)
