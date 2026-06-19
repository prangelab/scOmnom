import pytest
from pathlib import Path

from scomnom.config import (
    LoadAndFilterConfig,
    IntegrateConfig,
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
# LoadAndFilterConfig
# -------------------------------------------------------------------------
def test_loadandqc_requires_exactly_one_input(tmp_path):
    out = tmp_path

    # 0 inputs -> error
    with pytest.raises(ValueError):
        LoadAndFilterConfig(
            raw_sample_dir=None,
            filtered_sample_dir=None,
            cellbender_dir=None,
            metadata_tsv=tmpfile(tmp_path, "meta0.tsv"),
            output_dir=out,
        )

    # filtered cannot be combined with raw/cellbender
    with pytest.raises(ValueError):
        LoadAndFilterConfig(
            raw_sample_dir="raw",
            filtered_sample_dir="filtered",
            cellbender_dir=None,
            metadata_tsv=tmpfile(tmp_path, "meta2.tsv"),
            output_dir=out,
        )

    # 1 valid input -> OK
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        filtered_sample_dir=None,
        cellbender_dir=None,
        metadata_tsv=tmpfile(tmp_path, "meta1.tsv"),
        output_dir=out,
    )
    assert cfg.raw_sample_dir == Path("raw")


def test_loadandqc_output_name_coerced_to_h5ad(tmp_path):
    cfg1 = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta1.tsv"),
        output_dir=tmp_path,
        output_name="mydata",
    )
    assert cfg1.output_name == "mydata"

    cfg2 = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta2.tsv"),
        output_dir=tmp_path,
        output_name="already.h5ad",
    )
    assert cfg2.output_name == "already.h5ad"


def test_loadandqc_figure_format_validation(tmp_path):
    # valid
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta.tsv"),
        output_dir=tmp_path,
        figure_formats=["png", "pdf"],
    )
    assert cfg.figure_formats == ["png", "pdf"]

    # invalid
    with pytest.raises(ValueError):
        LoadAndFilterConfig(
            raw_sample_dir="raw",
            metadata_tsv=tmpfile(tmp_path, "meta_bad.tsv"),
            output_dir=tmp_path,
            figure_formats=["not_a_format"],
        )


def test_loadandqc_figdir_property(tmp_path):
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta.tsv"),
        output_dir=tmp_path,
        figdir_name="figs",
    )
    assert cfg.figdir == tmp_path / "figs"


def test_loadandfilter_min_counts_defaults_to_none(tmp_path):
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta.tsv"),
        output_dir=tmp_path,
    )
    assert cfg.min_counts is None
    assert cfg.min_counts_mad == 5.0
    assert cfg.min_counts_quantile is None
    assert cfg.min_counts_auto_activate_quantile == 0.01
    assert cfg.min_counts_auto_activate_below == 1000


def test_loadandfilter_accepts_min_counts(tmp_path):
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta.tsv"),
        output_dir=tmp_path,
        min_counts=1000,
    )
    assert cfg.min_counts == 1000


def test_loadandfilter_accepts_disabling_auto_min_counts(tmp_path):
    cfg = LoadAndFilterConfig(
        raw_sample_dir="raw",
        metadata_tsv=tmpfile(tmp_path, "meta.tsv"),
        output_dir=tmp_path,
        min_counts_mad=None,
        min_counts_quantile=None,
        min_counts_auto_activate_quantile=None,
        min_counts_auto_activate_below=None,
    )
    assert cfg.min_counts_mad is None
    assert cfg.min_counts_quantile is None
    assert cfg.min_counts_auto_activate_quantile is None
    assert cfg.min_counts_auto_activate_below is None


# -------------------------------------------------------------------------
# IntegrateConfig
# -------------------------------------------------------------------------
def test_integration_normalizes_methods(tmp_path):
    cfg = IntegrateConfig(
        input_path="in.h5ad",
        output_dir=tmp_path,
        methods=[" Scanorama ", "Harmony"],
    )
    assert cfg.methods == ["scanorama", "harmony"]


def test_integration_default_output(tmp_path):
    cfg = IntegrateConfig(input_path=tmp_path / "adata.h5ad", output_dir=tmp_path)
    assert cfg.output_name == "adata.integrated"
    assert cfg.input_path == tmp_path / "adata.h5ad"


# -------------------------------------------------------------------------
# ClusterAnnotateConfig
# -------------------------------------------------------------------------
def test_clusterannotate_figdir_uses_output_when_present(tmp_path):
    inp = tmp_path / "integrated.h5ad"

    cfg = ClusterAnnotateConfig(
        input_path=inp,
        output_dir=tmp_path / "out",
        figdir_name="figs",
    )
    assert cfg.figdir == (tmp_path / "out").resolve() / "figs"


def test_clusterannotate_figdir_falls_back_to_input_parent(tmp_path):
    inp = tmp_path / "integrated.h5ad"

    cfg = ClusterAnnotateConfig(
        input_path=inp,
        output_dir=None,
        figdir_name="plots",
    )
    assert cfg.figdir == inp.parent.resolve() / "plots"


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
    assert cfg.force_celltypist_recompute is False
