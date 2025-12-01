# tests/test_load_and_filter.py

import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from pathlib import Path

from scomnom.load_and_filter import (
    add_metadata,
    compute_qc_metrics,
    filter_and_doublets,
    normalize_and_hvg,
    pca_neighbors_umap,
    cluster_and_cleanup_qc,
    run_load_and_filter,
)
from scomnom.config import LoadAndQCConfig


# -------------------------------------------------------------------------
# Synthetic AnnData Factory
# -------------------------------------------------------------------------
def synthetic_adata(n_cells=300, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n_cells, n_genes))

    adata = sc.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(n_cells)]

    # Required keys
    adata.obs["sample"] = pd.Categorical(
        rng.integers(0, 3, size=n_cells).astype(str)
    )
    return adata


# -------------------------------------------------------------------------
# Fixtures: full mock of IO + plotting + scrublet
# -------------------------------------------------------------------------
@pytest.fixture
def mock_scrublet(monkeypatch):
    # Scanpy Scrublet modifies adata in-place; mock with deterministic flags
    def fake_scrublet(adata, *args, **kwargs):
        adata.obs["doublet_score"] = 0.01
        adata.obs["predicted_doublet"] = False
        return adata

    monkeypatch.setattr("scanpy.pp.scrublet", fake_scrublet)
    return True


@pytest.fixture
def mock_io(monkeypatch):
    import scomnom.io_utils as io

    # Simulate minimal raw loading: return simple sample_map & read_counts
    monkeypatch.setattr(
        io,
        "load_raw_data",
        lambda cfg, plot_dir=None: (
            {"A": synthetic_adata(100), "B": synthetic_adata(80)},
            {"A": 10_000, "B": 8_000},
            None,
        ),
    )
    monkeypatch.setattr(
        io,
        "load_filtered_data",
        lambda cfg: ({"A": synthetic_adata(100)}, {"A": 10_000}),
    )
    monkeypatch.setattr(
        io,
        "load_cellbender_data",
        lambda cfg: ({"A": synthetic_adata(100)}, {"A": 10_000}),
    )

    monkeypatch.setattr(
        io,
        "infer_batch_key_from_metadata_tsv",
        lambda m, k: "sample",  # always use .obs["sample"]
    )
    monkeypatch.setattr(
        io,
        "merge_samples",
        lambda sm, batch_key="sample": sc.concat(sm, label=batch_key, merge="same"),
    )
    monkeypatch.setattr(
        io,
        "save_adata",
        lambda *args, **kw: None,
    )

    return True


@pytest.fixture
def mock_plots(monkeypatch):
    import scomnom.plot_utils as pu
    monkeypatch.setattr(pu, "setup_scanpy_figs", lambda *a, **k: None)
    monkeypatch.setattr(pu, "save_multi", lambda *a, **k: None)
    monkeypatch.setattr(pu, "run_qc_plots_pre_filter", lambda *a, **k: None)
    monkeypatch.setattr(pu, "run_qc_plots_postfilter", lambda *a, **k: None)
    monkeypatch.setattr(pu, "plot_elbow_knee", lambda *a, **k: None)
    monkeypatch.setattr(pu, "plot_final_cell_counts", lambda *a, **k: None)
    return True


# -------------------------------------------------------------------------
# add_metadata tests
# -------------------------------------------------------------------------
def test_add_metadata_basic(tmp_path):
    adata = synthetic_adata()

    # write metadata.tsv
    meta = pd.DataFrame({
        "sample": ["0", "1", "2"],
        "patient": ["P1", "P2", "P3"],
    })
    mpath = tmp_path / "meta.tsv"
    meta.to_csv(mpath, sep="\t", index=False)

    out = add_metadata(adata, mpath, "sample")

    # new col created
    assert "patient" in out.obs
    assert set(out.obs["patient"].cat.categories) == {"P1", "P2", "P3"}


def test_add_metadata_missing_column(tmp_path):
    adata = synthetic_adata()

    meta = pd.DataFrame({"not_sample": ["A"]})
    mpath = tmp_path / "meta.tsv"
    meta.to_csv(mpath, sep="\t", index=False)

    with pytest.raises(KeyError):
        add_metadata(adata, mpath, "sample")


def test_add_metadata_missing_sample(tmp_path):
    adata = synthetic_adata()

    meta = pd.DataFrame({
        "sample": ["100"],  # does not match any obs
        "patient": ["X"],
    })
    mpath = tmp_path / "meta.tsv"
    meta.to_csv(mpath, sep="\t", index=False)

    with pytest.raises(ValueError):
        add_metadata(adata, mpath, "sample")


# -------------------------------------------------------------------------
# compute_qc_metrics
# -------------------------------------------------------------------------
def test_compute_qc_metrics():
    adata = synthetic_adata(n_genes=20)
    # prefix patterns
    cfg = type("cfg", (), dict(
        mt_prefix="g",
        ribo_prefixes=["r"],
        hb_regex="hb",
    ))

    out = compute_qc_metrics(adata, cfg)
    assert "total_counts" in out.obs
    assert "pct_counts_mt" in out.obs
    assert "mt" in out.var


# -------------------------------------------------------------------------
# filter_and_doublets
# -------------------------------------------------------------------------
def test_filter_and_doublets(mock_scrublet):
    cfg = type("cfg", (), dict(
        batch_key="sample",
        min_genes=1,
        min_cells=1,
        n_jobs=1,
    ))

    adata = synthetic_adata()
    out = filter_and_doublets(adata, cfg)

    # doublet flags must exist (from mocked scrublet)
    assert "predicted_doublet" in out.obs
    assert out.obs["predicted_doublet"].dtype.name == "bool"
    assert out.n_obs > 0


# -------------------------------------------------------------------------
# normalize_and_hvg
# -------------------------------------------------------------------------
def test_normalize_and_hvg():
    cfg = type("cfg", (), dict(
        batch_key="sample",
        n_top_genes=10,
    ))

    adata = synthetic_adata()
    out = normalize_and_hvg(adata, cfg)

    assert "highly_variable" in out.var
    assert "counts" in out.layers


# -------------------------------------------------------------------------
# pca_neighbors_umap
# -------------------------------------------------------------------------
def test_pca_neighbors_umap():
    cfg = type("cfg", (), dict(
        max_pcs_plot=5,
    ))

    adata = synthetic_adata()
    out = pca_neighbors_umap(adata, cfg)

    assert "X_umap" in out.obsm
    assert "n_pcs_elbow" in out.uns
    assert isinstance(out.uns["n_pcs_elbow"], int)


# -------------------------------------------------------------------------
# cluster_and_cleanup_qc
# -------------------------------------------------------------------------
def test_cluster_and_cleanup_qc(mock_scrublet):
    cfg = type("cfg", (), dict(
        batch_key="sample",
        max_pct_mt=100,
        min_cells_per_sample=0,
    ))

    adata = synthetic_adata()
    sc.pp.scrublet(adata)  # so predicted_doublet exists
    adata.obs["pct_counts_mt"] = 0.5

    out = cluster_and_cleanup_qc(adata, cfg)
    assert "leiden" in out.obs


# -------------------------------------------------------------------------
# Full orchestrator: run_load_and_filter
# -------------------------------------------------------------------------
def test_run_load_and_filter(
    tmp_path,
    mock_io,
    mock_plots,
    mock_scrublet,
):
    # Create metadata.tsv
    meta = pd.DataFrame({
        "sample": ["A", "B"],
        "patient": ["P1", "P2"],
    })
    mpath = tmp_path / "meta.tsv"
    meta.to_csv(mpath, sep="\t", index=False)

    cfg = LoadAndQCConfig(
        metadata_tsv=mpath,
        batch_key="sample",
        raw_sample_dir=tmp_path / "raw",     # triggers load_raw_data mock
        filtered_sample_dir=None,
        cellbender_dir=None,
        figdir=tmp_path / "figs",
        output_dir=tmp_path,
        output_name="out.h5ad",
        mt_prefix="g",
        ribo_prefixes=["r"],
        hb_regex="hb",
        min_genes=1,
        min_cells=1,
        min_cells_per_sample=0,
        max_pct_mt=100,
        n_jobs=1,
    )

    # Ensure input directory exists
    (tmp_path / "raw").mkdir(exist_ok=True)

    out = run_load_and_filter(cfg, logfile=None)

    # Basic checks
    assert isinstance(out, sc.AnnData)
    assert "leiden" in out.obs
    assert "X_umap" in out.obsm
    assert "batch_key" in out.uns
