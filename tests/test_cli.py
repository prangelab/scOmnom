import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from scomnom.cli import app

runner = CliRunner()


# ---------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------
def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scOmnom CLI" in result.output


# ---------------------------------------------------------
# cell-qc
# ---------------------------------------------------------
def test_cell_qc_help():
    result = runner.invoke(app, ["cell-qc", "--help"])
    assert result.exit_code == 0
    assert "Generate QC comparisons" in result.output


def test_cell_qc_requires_any_input():
    result = runner.invoke(app, ["cell-qc", "--out", "outdir"])
    # All inputs missing
    assert result.exit_code != 0
    assert "Provide at least one input" in result.output


@patch("scomnom.cli.run_cell_qc")
def test_cell_qc_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "cell-qc",
            "--raw", "raw_dir",
            "--out", "outdir"
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.raw_sample_dir == "raw_dir"
    assert cfg.output_dir.name == "outdir"


# ---------------------------------------------------------
# load-and-filter
# ---------------------------------------------------------
def test_load_and_filter_help():
    result = runner.invoke(app, ["load-and-filter", "--help"])
    assert result.exit_code == 0
    assert "preprocessing pipeline" in result.output


def test_load_and_filter_requires_output_and_metadata():
    result = runner.invoke(app, ["load-and-filter"])
    assert result.exit_code != 0
    assert "Output directory (required)" in result.output


def test_load_and_filter_mutually_exclusive_raw_filtered():
    result = runner.invoke(
        app,
        [
            "load-and-filter",
            "--raw-sample-dir", "rawdir",
            "--filtered-sample-dir", "filtered",
            "--metadata-tsv", "meta.tsv",
            "--out", "out"
        ]
    )
    assert result.exit_code != 0
    assert "Cannot specify both" in result.output


@patch("scomnom.cli.run_load_and_filter")
def test_load_and_filter_dispatch(mock_run, tmp_path):
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample\tcol\nA\t1\n")

    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "load-and-filter",
            "--raw-sample-dir", "raw",
            "--metadata-tsv", str(meta),
            "--out", str(out),
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.raw_sample_dir == "raw"
    assert cfg.metadata_tsv == meta


# ---------------------------------------------------------
# integrate
# ---------------------------------------------------------
def test_integrate_help():
    result = runner.invoke(app, ["integrate", "--help"])
    assert result.exit_code == 0
    assert "batch correction" in result.output


def test_integrate_requires_input():
    result = runner.invoke(app, ["integrate"])
    assert result.exit_code != 0
    assert "Input h5ad" in result.output


def test_integrate_invalid_method_rejected():
    result = runner.invoke(
        app,
        [
            "integrate",
            "--input-path", "adata.h5ad",
            "--methods", "BadTool"
        ]
    )
    assert result.exit_code != 0
    assert "Invalid method" in result.output


@patch("scomnom.cli.run_integration")
def test_integrate_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "integrate",
            "--input-path", "adata.h5ad",
            "--methods", "Scanorama",
            "--methods", "Harmony",
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.methods == ["Scanorama", "Harmony"]
    assert cfg.input_path == "adata.h5ad"


# ---------------------------------------------------------
# cluster-and-annotate
# ---------------------------------------------------------
def test_cluster_help():
    result = runner.invoke(app, ["cluster-and-annotate", "--help"])
    assert result.exit_code == 0
    assert "Perform clustering" in result.output


def test_cluster_requires_input():
    # Missing --input, but not using --list-models or --download-models
    result = runner.invoke(app, ["cluster-and-annotate"])
    assert result.exit_code != 0
    assert "Missing required option --input" in result.output


# --- list models ----------------------------------------------------------
@patch("scomnom.cli.get_available_celltypist_models", return_value=[{"name": "Test.pkl"}])
def test_cluster_list_models(mock_models):
    result = runner.invoke(app, ["cluster-and-annotate", "--list-models"])
    assert result.exit_code == 0
    assert "Available CellTypist models" in result.output
    mock_models.assert_called_once()


# --- download models ------------------------------------------------------
@patch("scomnom.cli.download_all_celltypist_models")
def test_cluster_download_models(mock_dl):
    result = runner.invoke(app, ["cluster-and-annotate", "--download-models"])
    assert result.exit_code == 0
    mock_dl.assert_called_once()


# --- dispatch full clustering ---------------------------------------------
@patch("scomnom.cli.run_clustering")
def test_cluster_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "cluster-and-annotate",
            "--input", "integrated.h5ad",
            "--out", "out.h5ad",
            "--res-min", "0.2",
            "--res-max", "1.2",
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.input_path == "integrated.h5ad"
    assert cfg.output_path == "out.h5ad"
    assert cfg.res_min == 0.2
    assert cfg.res_max == 1.2
