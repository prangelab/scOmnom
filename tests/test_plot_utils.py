# tests/test_plot_utils.py

import numpy as np
import pandas as pd
import scanpy as sc
import pytest
import sys
from scipy import sparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
mpl.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

import scomnom.plot_utils as pu
import scomnom.plotting as plotting
import scomnom.de_plot_utils as dpu
import scomnom.io_utils as iu


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

    fig = plt.figure()
    pu.save_multi("test_plot", Path("overview"), fig=fig)

    out = tmp_path / "figs" / "png" / "overview_round1" / "test_plot.png"
    assert out.exists()


def test_save_multi_without_setup_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pu, "ROOT_FIGDIR", None, raising=False)
    fig = plt.figure()
    with pytest.raises(RuntimeError):
        pu.save_multi("x", tmp_path, fig=fig)


def test_persist_plot_artifacts_clears_figure_reference(monkeypatch):
    calls = []

    def _save(stem, figdir, fig=None, savefig_kwargs=None):
        calls.append((stem, Path(figdir), fig is not None))

    monkeypatch.setattr(pu, "save_multi", _save)
    fig = plt.figure()
    artifact = pu.PlotArtifact(stem="demo", figdir=Path("figs"), fig=fig)

    pu.persist_plot_artifacts([artifact])

    assert calls == [("demo", Path("figs"), True)]
    assert artifact.fig is None


def test_use_system_tar_zstd_prefers_python_path_on_macos(monkeypatch):
    monkeypatch.setattr(iu.sys, "platform", "darwin", raising=False)
    monkeypatch.setenv("SCOMNOM_FORCE_SYSTEM_TAR_ZSTD", "")
    monkeypatch.setattr(iu.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setitem(sys.modules, "zstandard", object())

    assert iu._use_system_tar_zstd() is False


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
def test_umap_by(tmp_path, reset_root_figdir, monkeypatch):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["group"] = "A"

    calls = []

    def _fake_umap(*args, **kwargs):
        calls.append(dict(kwargs))
        ax = kwargs["ax"]
        ax.scatter([0.0], [0.0], s=4.0)
        return None

    monkeypatch.setattr(pu.sc.pl, "umap", _fake_umap, raising=False)

    pu.umap_by(adata, keys="group", figdir=figdir, stem="my_umap")

    assert len(calls) == 1
    assert "rasterized" not in calls[0]
    assert "edgecolors" not in calls[0]


def test_plot_elbow_knee(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.X = sparse.csr_matrix(adata.X)
    adata.layers["counts_raw"] = adata.X.copy()

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_elbow_knee(adata, figpath_stem="knee", figdir=figdir)

    assert [(a.stem, a.figdir) for a in artifacts] == [("knee", figdir)]


def test_plotting_api_plot_decoupler_payload_returns_figures():
    activity = pd.DataFrame(
        [[1.0, 0.2, -0.4], [0.3, -1.2, 0.7]],
        index=["C00", "C01"],
        columns=["PATH_A", "PATH_B", "PATH_C"],
    )
    payload = {
        "cluster_display_map": {
            "C00": "C00: Alpha",
            "C01": "C01: Beta",
        },
        "cluster_display_labels": ["C00: Alpha", "C01: Beta"],
        "progeny": {
            "activity": activity,
            "config": {"round_id": "r5_archetypes"},
        },
    }

    figs = plotting.plot_decoupler_payload(
        payload,
        net_name="progeny",
        display=False,
        return_fig=True,
    )

    assert isinstance(figs, list)
    assert len(figs) >= 1
    for fig in figs:
        plt.close(fig)


def test_plotting_api_plot_de_decoupler_payload_accepts_source_payload():
    activity = pd.DataFrame(
        [[1.0, -0.5], [-0.4, 0.8]],
        index=["C00", "C01"],
        columns=["PATH_A", "PATH_B"],
    )
    payload = {
        "source": "pseudobulk",
        "condition_key": "sex",
        "contrast": "female_vs_male",
        "nets": {
            "progeny": {
                "activity": activity,
                "config": {"input": "pseudobulk:sex:female_vs_male:stat"},
            }
        },
    }

    figs = plotting.plot_de_decoupler_payload(
        payload,
        net_name="progeny",
        display=False,
        return_fig=True,
    )

    assert isinstance(figs, list)
    assert len(figs) >= 1
    for fig in figs:
        plt.close(fig)


def test_plot_decoupler_dotplot_uses_numeric_positions():
    activity = pd.DataFrame(
        [[1.2, -0.8, 0.4], [0.7, -1.1, 0.2]],
        index=["1", "2"],
        columns=["TERM_A", "TERM_B", "TERM_C"],
    )

    artifacts = pu.plot_decoupler_dotplot(activity, net_name="msigdb", figdir=Path("decoupler"), top_k=3)

    assert len(artifacts) == 1
    fig = artifacts[0].fig
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()

    assert offsets.shape[0] > 0
    assert np.issubdtype(offsets.dtype, np.number)
    assert [tick.get_text() for tick in ax.get_xticklabels()][:2] == ["1", "2"]
    plt.close(fig)


def test_plot_de_gsea_payload_returns_artifact():
    payload = {
        "results": pd.DataFrame(
            {
                "cluster": ["C00", "C00", "C00", "C01"],
                "pathway": [
                    "HALLMARK_INFLAMMATORY_RESPONSE",
                    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                    "REACTOME_SIGNALING_BY_FGFR",
                    "REACTOME_MAPK_TARGETS",
                ],
                "NES": [1.8, -1.6, 1.2, 1.4],
                "ES": [0.5, -0.4, 0.25, 0.3],
                "pval": [0.01, 0.02, 0.03, 0.03],
                "padj": [1.20, 0.04, 0.05, 0.05],
                "leading_edge_preview": ["G1, G2", "G3, G4", "G5, G6", "G7, G8"],
                "leading_edge_n": [2, 2, 2, 2],
            }
        )
    }

    artifacts = dpu.plot_de_gsea_payload(payload, figdir=Path("gsea"), title_prefix="sex female_vs_male")

    assert isinstance(artifacts, list)
    assert len(artifacts) == 3
    assert sorted(a.stem for a in artifacts) == [
        "gsea_summary_HALLMARK_C00",
        "gsea_summary_REACTOME_C00",
        "gsea_summary_REACTOME_C01",
    ]
    for art in artifacts:
        scatter = art.fig.axes[0].collections[0]
        color_values = np.asarray(scatter.get_array(), dtype=float)
        assert np.all(color_values >= 0.0)
        clim = scatter.get_clim()
        assert clim[0] == 0.0
        assert clim[1] >= 0.0


def test_plot_de_msigdb_joint_payload_returns_artifact():
    payload = {
        "results": pd.DataFrame(
            {
                "cluster": ["C00", "C00", "C00", "C01"],
                "pathway": [
                    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
                    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                    "REACTOME_SIGNALING_BY_EGFR",
                    "REACTOME_MAPK_TARGETS",
                ],
                "decoupler_score": [2.0, -1.5, 1.1, 1.2],
                "NES": [1.9, -1.7, 1.0, 1.3],
                "padj": [0.01, 0.02, 0.04, 0.04],
                "sign_concordant": [True, True, True, True],
                "gsea_sig": [True, True, True, True],
                "leading_edge_n": [3, 4, 2, 2],
            }
        )
    }

    artifacts = dpu.plot_de_msigdb_joint_payload(payload, figdir=Path("joint"), title_prefix="sex female_vs_male")

    assert isinstance(artifacts, list)
    assert len(artifacts) == 3
    assert sorted(a.stem for a in artifacts) == [
        "msigdb_joint_concordant_HALLMARK_C00",
        "msigdb_joint_concordant_REACTOME_C00",
        "msigdb_joint_concordant_REACTOME_C01",
    ]


def test_plotting_api_plot_module_score_summary_heatmap_returns_figure():
    summary = pd.DataFrame(
        [[1.2, -0.3], [-0.8, 0.9]],
        index=["C00: Alpha", "C01: Beta"],
        columns=["Inflammation", "Stress"],
    )

    fig = plotting.plot_module_score_summary_heatmap(
        summary,
        display=False,
        return_fig=True,
    )

    assert fig is not None
    plt.close(fig)


def test_plot_final_cell_counts_missing_batch(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    cfg = SimpleNamespace(batch_key="batch", figdir=figdir)

    adata = synthetic_adata()
    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_final_cell_counts(adata, cfg)
    assert artifacts == []


def test_plot_final_cell_counts_ok(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    cfg = SimpleNamespace(batch_key="batch", figdir=figdir)

    adata = synthetic_adata()
    adata.obs["batch"] = ["A"] * adata.n_obs

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_final_cell_counts(adata, cfg)
    assert [a.stem for a in artifacts] == ["final_cell_counts"]


def test_plot_mt_histogram(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    cfg = SimpleNamespace(figdir=figdir)

    adata = synthetic_adata()
    adata.obs["pct_counts_mt"] = np.random.rand(adata.n_obs) * 20

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_mt_histogram(adata, cfg, suffix="prefilter")
    assert [a.stem for a in artifacts] == ["prefilter_QC_hist_pct_mt"]


def test_run_qc_plots_pre_filter(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    cfg = SimpleNamespace(
        make_figures=True,
        batch_key="sample",
        figdir=figdir,
        min_counts=None,
        min_counts_mad=5.0,
        min_counts_quantile=None,
        min_counts_auto_activate_quantile=0.01,
        min_counts_auto_activate_below=1000,
        min_genes=0,
        max_pct_mt=100.0,
        max_genes_mad=5.0,
        max_genes_quantile=0.999,
        max_counts_mad=5.0,
        max_counts_quantile=0.999,
    )
    qc_df = pd.DataFrame(
        {
            "sample": ["A"] * 10,
            "total_counts": np.random.randint(100, 1000, size=10),
            "n_genes_by_counts": np.random.randint(50, 200, size=10),
            "pct_counts_mt": np.random.rand(10) * 10,
        }
    )

    artifacts = pu.run_qc_plots_pre_filter_df(qc_df, cfg)
    assert any("prefilter_QC_hist_pct_mt" in artifact.stem for artifact in artifacts)


def test_run_qc_plots_postfilter(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    cfg = SimpleNamespace(
        make_figures=True,
        batch_key="batch",
        figdir=figdir,
        max_pcs_plot=5,
        min_genes=0,
        max_pct_mt=100.0,
        max_genes_mad=5.0,
        max_genes_quantile=0.999,
        max_counts_mad=5.0,
        max_counts_quantile=0.999,
    )

    adata = synthetic_adata()
    adata.X = sparse.csr_matrix(adata.X)
    adata.layers["counts_raw"] = adata.X.copy()
    adata.obs["batch"] = ["A"] * adata.n_obs
    adata.obs["leiden"] = ["0"] * adata.n_obs
    adata.obs["n_genes_by_counts"] = np.random.randint(50, 200, size=adata.n_obs)
    adata.obs["total_counts"] = np.random.randint(100, 1000, size=adata.n_obs)
    adata.obs["pct_counts_mt"] = np.random.rand(adata.n_obs) * 10

    artifacts = pu.run_qc_plots_postfilter(adata, cfg)
    stems = [artifact.stem for artifact in artifacts]
    assert "QC_violin_counts_postfilter_raw" in stems
    assert "QC_violin_mt_postfilter_raw" in stems
    assert "postfilter_raw_QC_hist_pct_mt" in stems


def test_plot_qc_filter_stack_accepts_global_only_stats(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obs["sample_id"] = pd.Categorical(["pbmc3k"] * adata.n_obs)
    adata.uns["qc_filter_stats"] = pd.DataFrame(
        {
            "filter": ["min_genes", "max_pct_mt"],
            "scope": ["cell", "cell"],
            "batch": ["ALL", "ALL"],
            "n_before": [100, 90],
            "n_after": [90, 85],
            "n_removed": [10, 5],
            "frac_removed": [0.10, 0.055],
        }
    )

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_qc_filter_stack(
            adata,
            batch_key="sample_id",
            figdir=Path("QC_plots") / "overview",
        )

    assert [artifact.stem for artifact in artifacts] == ["QC Filter effects"]


def test_plot_qc_filter_stack_aggregates_duplicate_global_stats(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obs["sample_id"] = pd.Categorical(["s1", "s2"] * (adata.n_obs // 2))
    adata.uns["qc_filter_stats"] = pd.DataFrame(
        {
            "filter": ["min_genes", "max_pct_mt", "min_genes", "max_pct_mt"],
            "scope": ["cell", "cell", "cell", "cell"],
            "batch": ["ALL", "ALL", "ALL", "ALL"],
            "n_before": [100, 80, 200, 150],
            "n_after": [80, 75, 150, 140],
            "n_removed": [20, 5, 50, 10],
            "frac_removed": [0.20, 0.0625, 0.25, 0.0667],
        }
    )

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_qc_filter_stack(
            adata,
            batch_key="sample_id",
            figdir=Path("QC_plots") / "overview",
        )

    assert [artifact.stem for artifact in artifacts] == ["QC Filter effects"]


def test_de_heatmap_drops_sample_key_from_annotations(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata(n=12, g=4)
    adata.obs["donor_id"] = pd.Categorical(["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4)
    adata.obs["condition"] = pd.Categorical(["ctrl", "stim"] * 6)

    with pu.capture_plot_artifacts() as artifacts:
        dpu.heatmap_top_genes_by_sample(
            adata,
            genes=["gene0", "gene1", "gene2"],
            sample_key="donor_id",
            condition_key="condition",
            annotation_keys=["condition", "donor_id"],
            artifact_stem="de_heatmap",
            artifact_figdir=Path("DE") / "heatmaps",
        )

    assert [artifact.stem for artifact in artifacts] == ["de_heatmap"]


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

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_scib_results_table(scaled)
    assert [a.stem for a in artifacts] == ["scIB_results_table"]


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

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_clustering_resolution_sweep(res, sil, ncl, pen, figdir)
    assert [a.stem for a in artifacts] == ["clustering_resolution_sweep"]


def test_plot_clustering_stability_ari(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_clustering_stability_ari([0.8, 0.9, 0.85], figdir)
    assert [a.stem for a in artifacts] == ["clustering_stability_ari"]


def test_plot_cluster_umaps(tmp_path, reset_root_figdir, mock_scanpy_plots, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster"] = ["0"] * adata.n_obs
    adata.obs["batch"] = ["A"] * adata.n_obs

    artifacts = pu.plot_cluster_umaps(adata, label_key="cluster", batch_key="batch", figdir=figdir)
    stems = [artifact.stem for artifact in artifacts]
    assert f"cluster_umap_cluster" in stems
    assert f"cluster_umap_batch" in stems
    assert f"cluster_umap_batch_and_cluster" in stems


def test_plot_cluster_umaps_does_not_forward_conflicting_scanpy_style_kwargs(tmp_path, reset_root_figdir, monkeypatch):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster"] = ["0"] * adata.n_obs
    adata.obs["batch"] = ["A"] * adata.n_obs

    calls = []

    def _fake_umap(*args, **kwargs):
        calls.append(dict(kwargs))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter([0.0], [0.0], s=4.0)
        return fig

    monkeypatch.setattr(pu.sc.pl, "umap", _fake_umap, raising=False)

    pu.plot_cluster_umaps(adata, label_key="cluster", batch_key="batch", figdir=figdir)

    assert len(calls) == 3
    for kwargs in calls:
        assert "rasterized" not in kwargs
        assert "edgecolors" not in kwargs
        assert "linewidths" not in kwargs


def test_umap_by_two_legend_styles_does_not_forward_conflicting_scanpy_style_kwargs(tmp_path, reset_root_figdir, monkeypatch):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    adata = synthetic_adata(n=5000)
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster_label"] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "C00 alpha", "C01 beta"))

    calls = []

    def _fake_umap(*args, **kwargs):
        calls.append(dict(kwargs))
        return None

    monkeypatch.setattr(pu.sc.pl, "umap", _fake_umap, raising=False)
    monkeypatch.setattr(pu, "save_umap_multi", lambda *args, **kwargs: None)

    pu.umap_by_two_legend_styles(
        adata,
        key="cluster_label",
        figdir=figdir,
        stem="umap_pretty_cluster_label",
    )

    assert len(calls) == 3
    for kwargs in calls:
        assert kwargs["alpha"] == pytest.approx(0.94)
        assert 5.0 <= float(kwargs["size"]) <= 16.0
        assert "edgecolors" not in kwargs
        assert "linewidths" not in kwargs
        assert "rasterized" not in kwargs


def test_plot_de_umap_single_uses_shared_umap_style_defaults(monkeypatch):
    adata = synthetic_adata(n=5000)
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster_label"] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "C00", "C01"))

    fig = plt.figure()
    calls = []

    def _fake_umap(*args, **kwargs):
        calls.append(dict(kwargs))
        return fig

    monkeypatch.setattr(dpu.sc.pl, "umap", _fake_umap, raising=False)

    artifacts = dpu.umap_single(
        adata,
        color="cluster_label",
        show=False,
        artifact_figdir=Path("figs"),
        artifact_stem="umap_single_test",
    )

    assert len(artifacts) == 1
    assert artifacts[0].stem == "umap_single_test"
    assert calls
    kwargs = calls[0]
    assert kwargs["alpha"] == pytest.approx(0.94)
    assert 5.0 <= float(kwargs["size"]) <= 16.0
    assert kwargs["edges"] is False


def test_plot_de_umap_single_styles_on_data_labels(monkeypatch):
    adata = synthetic_adata(n=50)
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obs["cluster_label"] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "C00", "C01"))

    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(5), s=5)
    txt = ax.text(0.5, 0.5, "C00")

    def _fake_umap(*args, **kwargs):
        return fig

    monkeypatch.setattr(dpu.sc.pl, "umap", _fake_umap, raising=False)

    artifacts = dpu.umap_single(
        adata,
        color="cluster_label",
        legend_loc="on data",
        show=False,
        artifact_figdir=Path("figs"),
        artifact_stem="umap_single_on_data",
    )

    assert len(artifacts) == 1
    bbox = txt.get_bbox_patch()
    assert bbox is not None
    assert bbox.get_alpha() == pytest.approx(0.65)


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
    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_cluster_tree(labels_per_res, resolutions, figdir=tmp_path, stem="tree")

    assert [a.stem for a in artifacts] == ["tree"]


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

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_stability_curves(
            resolutions=res,
            silhouette=sil,
            stability=stab,
            composite=comp,
            tiny_cluster_penalty=tiny,
            best_resolution=0.4,
            plateaus=plateaus,
            figdir=figdir,
        )

    assert [a.stem for a in artifacts] == ["cluster_selection_stability"]


def test_plot_stability_curves_with_bio(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    sil = {"0.200": 0.1, "0.400": 0.3, "0.600": 0.2}
    stab = {"0.200": 0.9, "0.400": 0.85, "0.600": 0.8}
    comp = {"0.200": 0.2, "0.400": 0.5, "0.600": 0.4}
    tiny = {"0.200": 0.6, "0.400": 0.7, "0.600": 0.5}
    plateaus = [{"resolutions": [0.2, 0.4]}]

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_stability_curves(
            resolutions=res,
            silhouette=sil,
            stability=stab,
            composite=comp,
            tiny_cluster_penalty=tiny,
            best_resolution=0.4,
            plateaus=plateaus,
            figdir=figdir,
            stem="cluster_selection_stability_alt",
        )

    assert [a.stem for a in artifacts] == ["cluster_selection_stability_alt"]


def test_plot_biological_metrics(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    bio_h = {"0.200": 0.7, "0.400": 0.9, "0.600": 0.8}
    bio_f = {"0.200": 0.3, "0.400": 0.1, "0.600": 0.2}
    bio_a = {"0.200": 0.6, "0.400": 0.8, "0.600": 0.7}
    plateaus = [{"resolutions": [0.2, 0.4]}]
    sel_cfg = {"w_hom": 0.5, "w_frag": 0.3, "w_bioari": 0.2}

    with pu.capture_plot_artifacts() as artifacts:
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

    assert [a.stem for a in artifacts] == ["biological_metrics"]


def test_plot_plateau_highlights(tmp_path, reset_root_figdir, mock_save_multi):
    figdir = tmp_path / "figs"
    pu.setup_scanpy_figs(figdir)

    res = [0.2, 0.4, 0.6]
    sil = {"0.200": 0.1, "0.400": 0.3, "0.600": 0.2}
    stab = {"0.200": 0.9, "0.400": 0.85, "0.600": 0.8}
    comp = {"0.200": 0.2, "0.400": 0.5, "0.600": 0.4}
    plateaus = [{"resolutions": [0.2, 0.4]}]

    with pu.capture_plot_artifacts() as artifacts:
        pu.plot_plateau_highlights(
            resolutions=res,
            silhouette=sil,
            stability=stab,
            composite=comp,
            best_resolution=0.4,
            plateaus=plateaus,
            figdir=figdir,
        )

    assert [a.stem for a in artifacts] == ["plateau_highlights"]
