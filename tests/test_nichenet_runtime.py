"""Exercise installer environment routing without running R or fetching code."""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
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


@pytest.mark.parametrize("returncode", [0, 1])
def test_native_helper_keeps_resource_provenance_and_r_logs(monkeypatch, tmp_path, caplog, returncode):
    monkeypatch.setattr(md, "_resolve_rscript", lambda: "/test/bin/Rscript")
    models = [{"model": "lr_network", "path": str(tmp_path / "model.rds"),
               "expected_md5": "pinned-md5", "sha256": "observed-sha256", "size_bytes": 12}]
    monkeypatch.setattr(md, "_prepare_nichenet_models", lambda organism: models)

    def run(args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        assert config["model_resources"] == models
        assert config["signal_scope"] == "all"
        assert kwargs["env"] == {"EXAMPLE": "value"}
        for name in ("ligand_activity", "ligand_target_links", "ligand_receptor_links", "potential_ligands"):
            (Path(config["output_dir"]) / f"{name}.tsv").write_text("gene\tvalue\nSTAT1\t0.5\n")
        return SimpleNamespace(returncode=returncode, stdout="R progress\n", stderr="R warning\n")

    monkeypatch.setattr(md.subprocess, "run", run)
    kwargs = dict(sender_expressed_genes=["STAT1"], receiver_expressed_genes=["STAT1"],
                  geneset_oi=["STAT1"], background_genes=["STAT1"], top_n_ligands=30,
                  top_n_targets=200, organism="human", signal_scope="all",
                  r_env={"EXAMPLE": "value"}, evidence_dir=tmp_path / "evidence")
    with caplog.at_level("INFO"):
        if returncode:
            with pytest.raises(RuntimeError, match="R helper failed"):
                md._run_nichenet_sender_focused(**kwargs)
        else:
            result = md._run_nichenet_sender_focused(**kwargs)
            assert result["model_resources"].to_dict(orient="records") == models
            assert len(result["ligand_activity"]) == 1
    evidence = kwargs["evidence_dir"]
    assert (evidence / "nichenet_r.stdout.log").read_text() == "R progress\n"
    assert (evidence / "nichenet_r.stderr.log").read_text() == "R warning\n"
    assert "observed-sha256" in (evidence / "nichenet_model_resources.tsv").read_text()
    assert "R warning" in caplog.text and "R progress" in caplog.text


def test_successful_namespace_probe_warnings_are_visible(runtime, monkeypatch, caplog):
    lib, _, _ = runtime
    monkeypatch.setattr(md.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout="loaded", stderr="namespace warning"))
    with caplog.at_level("INFO"):
        assert md._ensure_nichenet_r_runtime(install_missing=False) == lib
    assert "namespace warning" in caplog.text and "loaded" in caplog.text


def test_r_loader_verifies_before_reading(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript not installed")
    # Parse the entire native helper, but execute only its base-R loader function.
    code = r'''
    args <- commandArgs(TRUE)
    for (expr in parse(file=args[[1]])) {
      if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
          identical(expr[[2]], as.name("load_verified_model"))) eval(expr)
    }
    path <- args[[2]]
    saveRDS(list(test=42), path)
    cfg <- list(organism="human", model_resources=data.frame(
      model="lr_network", path=path, size_bytes=file.info(path)$size,
      expected_md5=unname(tools::md5sum(path))))
    stopifnot(load_verified_model("lr_network")$test == 42)
    cfg$model_resources$size_bytes <- cfg$model_resources$size_bytes + 1
    err <- tryCatch(load_verified_model("lr_network"), error=identity)
    stopifnot(inherits(err, "error"), grepl("checksum mismatch", conditionMessage(err)))
    cfg$model_resources$size_bytes <- file.info(path)$size
    cfg$model_resources$expected_md5 <- paste(rep("0", 32), collapse="")
    err <- tryCatch(load_verified_model("lr_network"), error=identity)
    stopifnot(inherits(err, "error"), grepl("checksum mismatch", conditionMessage(err)))
    err <- tryCatch(load_verified_model("absent"), error=identity)
    stopifnot(inherits(err, "error"), grepl("Missing verified", conditionMessage(err)))
    cat("Local model accepted; altered and missing models rejected\n")
    '''
    result = subprocess.run([rscript, "--vanilla", "-e", code,
                             str(md._NICHENET_R_HELPER), str(tmp_path / "tiny.rds")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "altered and missing models rejected" in result.stdout
