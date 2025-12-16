# tests/test_plot_utils.py

import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

import scomnom.plot_utils as pu


# ---------------------------------------------------------------------
# Small AnnData helper
# ---------------------------------------------------------------------
def synthetic_adata(n=50, g=10):
    from anndata import AnnData
    X = np.random.poisson(1.0, (n, g))
    adata = AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(n)]
    adata.var_names = [f"gene{i}" for i in range(g)]
    return adata


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def reset_root_figdir(monkeypatch):
    # reset ROOT_FIGDIR between tests
    monkeypatch.setattr(pu, "ROOT_FIGDIR", None, raising=False)
    yield
    monkeypatch.setattr(pu, "ROOT_FIGDIR", None, raising=False)


@pytest.fixture
def mock_save_multi(monkeypatch):
    calls = []

    def _save(stem, figdir, fig=None):
        calls.append((stem, Path(figdir)))

    monkeypatch.setattr(pu, "save_multi", _save)
    return calls


@pytest.fixture
def mock_scanpy_plots(monkeypatch):
    # stub scanpy plotting functions to avoid heavy work
    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(pu.sc.pl, "umap", _noop, raising=False)
    monkeypatch.setattr(pu.sc.pl, "violin", _noop, raising=False)
    monkeypatch.setattr(pu.sc.pl, "scatter", _noop, raising=False)
    monkeypatch.setattr(pu.sc.pl, "pca_variance_ratio", _noop, raising=False)
    return True


# ---------------------------------------------------------------------
# Basic configuration / save_multi
# ---------------------------------------------------------------------
def test_setup_scanpy_figs_and_save_multi(tmp_path, reset_root_figdir):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir, formats=["png"])

    # create a simple figure
    fig = plt.figure()
    pu.save_multi("test_plot", figdir, fig=fig)

    # png directory should exist & file created
    out = tmp_path / "figs" / "png" / "test_plot.png"
    assert out.exists()


def test_save_multi_without_setup_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pu, "ROOT_FIGDIR", None, raising=False)
    fig = plt.figure()
    with pytest.raises(RuntimeError):
        pu.save_multi("x", tmp_path, fig=fig)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def test_res_key_and_sorted_resolutions():
    assert pu._res_key(0.2) == "0.200"
    assert pu._res_key("0.2") == "0.200"
    assert pu._sorted_resolutions([0.6, "0.2", 0.4]) == [0.2, 0.4, 0.6]


def test_extract_series():
    res = [0.2, 0.4, 0.6]
    values = {"0.200": 1.0, "0.600": 3.0}
    arr = pu._extract_series(res, values, default=-1.0)
    # 0.2 -> 1.0, 0.4 missing -> -1.0, 0.6 -> 3.0
    assert np.allclose(arr, [1.0, -1.0, 3.0])


def test_normalize_array_basic():
    x = np.array([2.0, 4.0, 6.0])
    y = pu._normalize_array(x)
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_normalize_array_all_nan():
    x = np.array([np.nan, np.nan])
    y = pu._normalize_array(x)
    assert np.allclose(y, [0.0, 0.0])


def test_plateau_spans():
    plateaus = [
        {"resolutions": [0.2, 0.4, 0.6]},
        {"resolutions": [0.8, 1.0]},
    ]
    spans = pu._plateau_spans(plateaus)
    assert spans == [(0.2, 0.6), (0.8, 1.0)]


# ---------------------------------------------------------------------
# QC wrappers
# ---------------------------------------------------------------------
def test_umap_by(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["group"] = "A"

    pu.umap_by(adata, keys="group", figdir=figdir, stem="my_umap")

    assert ("my_umap", figdir) in mock_save_multi


def test_plot_elbow_knee(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.layers["counts_raw"] = adata.X.copy()

    pu.plot_elbow_knee(adata, figpath_stem="knee", figdir=figdir)

    assert ("knee", figdir) in mock_save_multi


def test_plot_read_comparison(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    ref = {"A": 100, "B": 200}
    other = {"A": 50, "B": 300}
    pu.plot_read_comparison(ref, other, "ref", "other", figdir=figdir, stem="reads")
    assert ("reads", figdir) in mock_save_multi


def test_plot_final_cell_counts_missing_batch(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    class Cfg:
        batch_key = "batch"
        figdir = figdir

    adata = synthetic_adata()
    # no "batch" column → should just warn and not call save_multi
    pu.plot_final_cell_counts(adata, Cfg)
    assert mock_save_multi == []


def test_plot_final_cell_counts_ok(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    class Cfg:
        batch_key = "batch"
        figdir = figdir

    adata = synthetic_adata()
    adata.obs["batch"] = ["A"] * adata.n_obs

    pu.plot_final_cell_counts(adata, Cfg)
    # called exactly once with stem="final_cell_counts"
    assert any(stem == "final_cell_counts" for stem, _ in mock_save_multi)


def test_plot_mt_histogram(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    class Cfg:
        figdir = figdir

    adata = synthetic_adata()
    adata.obs["pct_counts_mt"] = np.random.rand(adata.n_obs) * 20

    pu.plot_mt_histogram(adata, Cfg, suffix="prefilter")
    assert any(stem == "prefilter_QC_hist_pct_mt" for stem, _ in mock_save_multi)


def test_run_qc_plots_pre_filter(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    class Cfg:
        make_figures = True
        batch_key = "batch"
        figdir = figdir

    adata = synthetic_adata()
    adata.obs["batch"] = ["A"] * adata.n_obs
    adata.obs["doublet_score"] = np.random.rand(adata.n_obs)
    adata.obs["pct_counts_mt"] = np.random.rand(adata.n_obs) * 10

    pu.run_qc_plots_pre_filter(adata, Cfg)
    # at least MT histogram should be saved
    assert any("prefilter_QC_hist_pct_mt" in stem for stem, _ in mock_save_multi)


def test_run_qc_plots_postfilter(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    class Cfg:
        make_figures = True
        batch_key = "batch"
        figdir = figdir
        max_pcs_plot = 5

    adata = synthetic_adata()
    adata.obs["batch"] = ["A"] * adata.n_obs
    adata.obs["leiden"] = ["0"] * adata.n_obs
    adata.obs["n_genes_by_counts"] = np.random.randint(50, 200, size=adata.n_obs)
    adata.obs["total_counts"] = np.random.randint(100, 1000, size=adata.n_obs)
    adata.obs["pct_counts_mt"] = np.random.rand(adata.n_obs) * 10

    pu.run_qc_plots_postfilter(adata, Cfg)
    # check that some expected stems were used
    stems = [s for s, _ in mock_save_multi]
    assert "QC_violin_mt_counts_postfilter" in stems
    assert "postfilter_QC_hist_pct_mt" in stems
    assert "QC_umap_sample" in stems or "QC_umap_leiden" in stems


# ---------------------------------------------------------------------
# scIB results table
# ---------------------------------------------------------------------
def test_plot_scib_results_table(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    # rows: methods; columns: metrics
    scaled = pd.DataFrame(
        {
            "cLISI": [0.8, 0.6],
            "iLISI": [0.7, 0.5],
            "Batch correction": [0.9, 0.4],
            "Total": [0.85, 0.45],
        },
        index=["Unintegrated", "Scanorama"],
    )

    pu.plot_scib_results_table(scaled, figdir)
    assert any(stem == "scIB_results_table" for stem, _ in mock_save_multi)


# ---------------------------------------------------------------------
# Clustering / stability plots
# ---------------------------------------------------------------------
def test_plot_clustering_resolution_sweep(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = np.array([0.2, 0.4, 0.6])
    sil = [0.1, 0.3, 0.2]
    ncl = [5, 8, 10]
    pen = [0.05, 0.15, 0.1]

    pu.plot_clustering_resolution_sweep(res, sil, ncl, pen, figdir)
    assert any(stem == "clustering_resolution_sweep" for stem, _ in mock_save_multi)


def test_plot_clustering_stability_ari(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    pu.plot_clustering_stability_ari([0.8, 0.9, 0.85], figdir)
    assert any(stem == "clustering_stability_ari" for stem, _ in mock_save_multi)


def test_plot_cluster_umaps(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster"] = ["0"] * adata.n_obs
    adata.obs["batch"] = ["A"] * adata.n_obs

    pu.plot_cluster_umaps(adata, label_key="cluster", batch_key="batch", figdir=figdir)
    stems = [s for s, _ in mock_save_multi]
    assert f"cluster_umap_cluster" in stems
    assert f"cluster_umap_batch" in stems
    assert f"cluster_umap_batch_and_cluster" in stems


# ---------------------------------------------------------------------
# Cluster tree
# ---------------------------------------------------------------------
def test_plot_cluster_tree(tmp_path, reset_root_figdir, mock_save_multi):
    # do NOT rely on ROOT_FIGDIR; save_multi is mocked
    labels_per_res = {
        "0.200": np.array([0, 0, 1, 1]),
        "0.400": np.array([0, 1, 1, 2]),
    }
    resolutions = [0.2, 0.4]
    pu.plot_cluster_tree(labels_per_res, resolutions, figdir=tmp_path, stem="tree")

    assert any(stem == "tree" for stem, _ in mock_save_multi)


# ---------------------------------------------------------------------
# Stability curves with and without biological metrics
# ---------------------------------------------------------------------
def test_plot_stability_curves_structural_only(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    sil = {"0.200": 0.1, "0.400": 0.3, "0.600": 0.2}
    stab = {"0.200": 0.9, "0.400": 0.85, "0.600": 0.8}
    comp = {"0.200": 0.2, "0.400": 0.5, "0.600": 0.4}
    tiny = {"0.200": 0.6, "0.400": 0.7, "0.600": 0.5}
    plateaus = [{"resolutions": [0.2, 0.4]}]

    pu.plot_stability_curves(
        resolutions=res,
        silhouette=sil,
        stability=stab,
        composite=comp,
        tiny_cluster_penalty=tiny,
        best_resolution=0.4,
        plateaus=plateaus,
        figdir=figdir,
        selection_config={"use_bio": False},
    )

    assert any(stem == "cluster_selection_stability" for stem, _ in mock_save_multi)


def test_plot_stability_curves_with_bio(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    sil = {"0.200": 0.1, "0.400": 0.3, "0.600": 0.2}
    stab = {"0.200": 0.9, "0.400": 0.85, "0.600": 0.8}
    comp = {"0.200": 0.2, "0.400": 0.5, "0.600": 0.4}
    tiny = {"0.200": 0.6, "0.400": 0.7, "0.600": 0.5}
    bio_h = {"0.200": 0.7, "0.400": 0.9, "0.600": 0.8}
    bio_f = {"0.200": 0.3, "0.400": 0.1, "0.600": 0.2}
    bio_a = {"0.200": 0.6, "0.400": 0.8, "0.600": 0.7}
    plateaus = [{"resolutions": [0.2, 0.4]}]
    sel_cfg = {"use_bio": True, "w_hom": 0.5, "w_frag": 0.3, "w_bioari": 0.2}

    pu.plot_stability_curves(
        resolutions=res,
        silhouette=sil,
        stability=stab,
        composite=comp,
        tiny_cluster_penalty=tiny,
        best_resolution=0.4,
        plateaus=plateaus,
        figdir=figdir,
        bio_homogeneity=bio_h,
        bio_fragmentation=bio_f,
        bio_ari=bio_a,
        selection_config=sel_cfg,
    )

    assert any(stem == "cluster_selection_stability" for stem, _ in mock_save_multi)


def test_plot_biological_metrics(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    bio_h = {"0.200": 0.7, "0.400": 0.9, "0.600": 0.8}
    bio_f = {"0.200": 0.3, "0.400": 0.1, "0.600": 0.2}
    bio_a = {"0.200": 0.6, "0.400": 0.8, "0.600": 0.7}
    plateaus = [{"resolutions": [0.2, 0.4]}]
    sel_cfg = {"w_hom": 0.5, "w_frag": 0.3, "w_bioari": 0.2}

    pu.plot_biological_metrics(
        resolutions=res,
        bio_homogeneity=bio_h,
        bio_fragmentation=bio_f,
        bio_ari=bio_a,
        selection_config=sel_cfg,
        best_resolution=0.4,
        plateaus=plateaus,
        figdir=figdir,
    )

    assert any(stem == "biological_metrics" for stem, _ in mock_save_multi)


def test_plot_plateau_highlights(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    sil = {"0.200": 0.1, "0.400": 0.3, "0.600": 0.2}
    stab = {"0.200": 0.9, "0.400": 0.85, "0.600": 0.8}
    comp = {"0.200": 0.2, "0.400": 0.5, "0.600": 0.4}
    plateaus = [{"resolutions": [0.2, 0.4]}]

    pu.plot_plateau_highlights(
        resolutions=res,
        silhouette=sil,
        stability=stab,
        composite=comp,
        best_resolution=0.4,
        plateaus=plateaus,
        figdir=figdir,
    )

    assert any(stem == "plateau_highlights" for stem, _ in mock_save_multi)
