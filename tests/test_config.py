import pytest
from pathlib import Path

from scomnom.config import (
    LoadAndQCConfig,
    IntegrationConfig,
    CellQCConfig,
    ClusterAnnotateConfig,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def tmpfile(tmp_path, name="x.txt", content=""):
    p = tmp_path / name
    p.write_text(content)
    return p


# -------------------------------------------------------------------------
# LoadAndQCConfig
# -------------------------------------------------------------------------
def test_loadandqc_requires_exactly_one_input(tmp_path):
    out = tmp_path

    # 0 inputs -> error
    with pytest.raises(ValueError):
        LoadAndQCConfig(
            raw_sample_dir=None,
            filtered_sample_dir=None,
            cellbender_dir=None,
            metadata_tsv=None,
            output_dir=out,
        )

    # 2 inputs -> error
    with pytest.raises(ValueError):
        LoadAndQCConfig(
            raw_sample_dir="raw",
            filtered_sample_dir="filtered",
            cellbender_dir=None,
            metadata_tsv=None,
            output_dir=out,
        )

    # 1 valid input -> OK
    cfg = LoadAndQCConfig(
        raw_sample_dir="raw",
        filtered_sample_dir=None,
        cellbender_dir=None,
        metadata_tsv=None,
        output_dir=out,
    )
    assert cfg.raw_sample_dir == Path("raw")


def test_loadandqc_output_name_coerced_to_h5ad(tmp_path):
    cfg1 = LoadAndQCConfig(
        raw_sample_dir="raw",
        output_dir=tmp_path,
        output_name="mydata",
    )
    assert cfg1.output_name == "mydata.h5ad"

    cfg2 = LoadAndQCConfig(
        raw_sample_dir="raw",
        output_dir=tmp_path,
        output_name="already.h5ad",
    )
    assert cfg2.output_name == "already.h5ad"


def test_loadandqc_figure_format_validation(tmp_path):
    # valid
    cfg = LoadAndQCConfig(
        raw_sample_dir="raw",
        output_dir=tmp_path,
        figure_formats=["png", "pdf"],
    )
    assert cfg.figure_formats == ["png", "pdf"]

    # invalid
    with pytest.raises(ValueError):
        LoadAndQCConfig(
            raw_sample_dir="raw",
            output_dir=tmp_path,
            figure_formats=["not_a_format"],
        )


def test_loadandqc_figdir_property(tmp_path):
    cfg = LoadAndQCConfig(
        raw_sample_dir="raw",
        output_dir=tmp_path,
        figdir_name="figs",
    )
    assert cfg.figdir == tmp_path / "figs"


# -------------------------------------------------------------------------
# IntegrationConfig
# -------------------------------------------------------------------------
def test_integration_normalizes_methods(tmp_path):
    cfg = IntegrationConfig(
        input_path="in.h5ad",
        methods=[" Scanorama ", "Harmony"],
    )
    assert cfg.methods == ["Scanorama", "Harmony"]


def test_integration_default_output(tmp_path):
    cfg = IntegrationConfig(input_path=tmp_path/"adata.h5ad")
    # Default output_path stays None (resolution happens in CLI)
    assert cfg.output_path is None
    assert cfg.input_path == tmp_path / "adata.h5ad"


# -------------------------------------------------------------------------
# CellQCConfig
# -------------------------------------------------------------------------
def test_cellqc_valid_formats(tmp_path):
    cfg = CellQCConfig(
        output_dir=tmp_path,
        figure_formats=["png", "pdf"],
    )
    assert cfg.output_dir == tmp_path


def test_cellqc_invalid_format(tmp_path):
    with pytest.raises(ValueError):
        CellQCConfig(
            output_dir=tmp_path,
            figure_formats=["foo"],
        )


def test_cellqc_defaults(tmp_path):
    cfg = CellQCConfig(output_dir=tmp_path)
    assert cfg.raw_pattern.endswith("raw_feature_bc_matrix")
    assert cfg.make_figures is True
    assert cfg.figdir_name == "figures"


# -------------------------------------------------------------------------
# ClusterAnnotateConfig
# -------------------------------------------------------------------------
def test_clusterannotate_figdir_uses_output_when_present(tmp_path):
    inp = tmp_path / "integrated.h5ad"
    out = tmp_path / "res.h5ad"

    cfg = ClusterAnnotateConfig(
        input_path=inp,
        output_path=out,
        figdir_name="figs",
    )
    assert cfg.figdir == out.parent / "figs"


def test_clusterannotate_figdir_falls_back_to_input_parent(tmp_path):
    inp = tmp_path / "integrated.h5ad"

    cfg = ClusterAnnotateConfig(
        input_path=inp,
        output_path=None,
        figdir_name="plots",
    )
    assert cfg.figdir == inp.parent / "plots"


def test_clusterannotate_format_validation(tmp_path):
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "in.h5ad",
        figure_formats=["png", "pdf"],
    )
    assert cfg.figure_formats == ["png", "pdf"]

    with pytest.raises(ValueError):
        ClusterAnnotateConfig(
            input_path=tmp_path / "in.h5ad",
            figure_formats=["abc"],
        )


def test_clusterannotate_defaults(tmp_path):
    cfg = ClusterAnnotateConfig(input_path=tmp_path / "a.h5ad")
    assert cfg.embedding_key == "X_integrated"
    assert cfg.label_key == "leiden"
    assert cfg.w_stab == 0.50
    assert cfg.w_sil == 0.35
    assert cfg.w_tiny == 0.15
    assert cfg.bio_guided_clustering is True
