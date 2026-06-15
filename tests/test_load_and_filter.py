# tests/test_load_and_filter.py

import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from pathlib import Path
from scipy import sparse

from scomnom.load_and_filter import (
    add_metadata,
    compute_qc_metrics,
    call_doublets,
    cleanup_after_solo,
    normalize_and_hvg,
    pca_neighbors_umap,
    run_load_and_filter,
    sparse_filter_cells_and_genes,
)
from scomnom.config import LoadAndFilterConfig


# -------------------------------------------------------------------------
# Synthetic AnnData Factory
# -------------------------------------------------------------------------
def synthetic_adata(n_cells=300, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_cells, n_genes)))

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
    def fake_run_solo_with_scvi(adata, *args, **kwargs):
        adata.obs["doublet_score"] = 0.01
        return adata

    monkeypatch.setattr("scomnom.load_and_filter.run_solo_with_scvi", fake_run_solo_with_scvi)
    return True


@pytest.fixture
def mock_io(monkeypatch):
    import scomnom.io_utils as io

    # Simulate minimal raw loading: return simple sample_map & read_counts
    monkeypatch.setattr(
        io,
        "load_raw_data",
        lambda cfg, plot_dir=None: (
            {"A": synthetic_adata(100, n_genes=120), "B": synthetic_adata(80, n_genes=120)},
            {"A": 10_000, "B": 8_000},
            None,
        ),
    )
    monkeypatch.setattr(
        io,
        "load_filtered_data",
        lambda cfg: ({"A": synthetic_adata(100, n_genes=120)}, {"A": 10_000}),
    )
    monkeypatch.setattr(
        io,
        "load_cellbender_filtered_data",
        lambda cfg: ({"A": synthetic_adata(100, n_genes=120)}, {"A": 10_000}),
    )

    monkeypatch.setattr(
        io,
        "infer_batch_key_from_metadata_tsv",
        lambda m, k: "sample",  # always use .obs["sample"]
    )
    monkeypatch.setattr(
        io,
        "merge_samples",
        lambda sm, batch_key="sample", input_layer_name=None: sc.concat(sm, label=batch_key, merge="same"),
    )
    monkeypatch.setattr(
        io,
        "save_dataset",
        lambda *args, **kw: None,
    )
    monkeypatch.setattr(
        io,
        "attach_raw_counts_postfilter",
        lambda cfg, adata: adata,
    )

    return True


@pytest.fixture
def mock_plots(monkeypatch):
    import scomnom.plot_utils as pu
    import scomnom.reporting as reporting
    monkeypatch.setattr(pu, "setup_scanpy_figs", lambda *a, **k: None)
    monkeypatch.setattr(pu, "persist_plot_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(pu, "run_qc_plots_pre_filter_df", lambda *a, **k: [])
    monkeypatch.setattr(pu, "run_qc_plots_postfilter", lambda *a, **k: [])
    monkeypatch.setattr(pu, "plot_final_cell_counts", lambda *a, **k: [])
    monkeypatch.setattr(pu, "plot_cellbender_effects", lambda *a, **k: [])
    monkeypatch.setattr(pu, "plot_qc_filter_stack", lambda *a, **k: [])
    monkeypatch.setattr(pu, "doublet_plots", lambda *a, **k: [])
    monkeypatch.setattr(pu, "umap_plots", lambda *a, **k: [])
    monkeypatch.setattr(pu, "capture_plot_artifacts", lambda *a, **k: __import__("contextlib").nullcontext([]))
    monkeypatch.setattr(reporting, "generate_qc_report", lambda *a, **k: None)
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


def test_sparse_filter_cells_and_genes_applies_min_counts():
    X = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [2, 2, 1],
                [5, 1, 0],
            ],
            dtype=np.int64,
        )
    )
    adata = sc.AnnData(X)
    adata.var_names = ["g0", "g1", "g2"]
    adata.obs_names = ["c0", "c1", "c2"]
    adata.obs["sample"] = pd.Categorical(["A", "A", "A"])
    adata.obs["pct_counts_mt"] = 0.0

    qc_rows = []
    out = sparse_filter_cells_and_genes(
        adata,
        min_genes=1,
        min_cells=1,
        min_counts=4,
        max_pct_mt=100,
        batch_key="sample",
        qc_rows=qc_rows,
    )

    assert list(out.obs_names) == ["c1", "c2"]
    assert any(row["filter"] == "min_counts" for row in qc_rows)


def test_sparse_filter_cells_and_genes_min_counts_can_remove_all_cells():
    X = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    adata = sc.AnnData(X)
    adata.var_names = ["g0", "g1", "g2"]
    adata.obs_names = ["c0", "c1"]
    adata.obs["sample"] = pd.Categorical(["A", "A"])
    adata.obs["pct_counts_mt"] = 0.0

    with pytest.raises(ValueError, match="All cells removed by min_counts=10"):
        sparse_filter_cells_and_genes(
            adata,
            min_genes=1,
            min_cells=1,
            min_counts=10,
            max_pct_mt=100,
            batch_key="sample",
            qc_rows=[],
        )


# -------------------------------------------------------------------------
# call_doublets / cleanup_after_solo
# -------------------------------------------------------------------------
def test_call_doublets_and_cleanup_after_solo():
    adata = synthetic_adata()
    adata.obs["doublet_score"] = np.linspace(0.0, 1.0, adata.n_obs)
    call_doublets(adata, batch_key="sample", expected_doublet_rate=0.1)

    assert "predicted_doublet" in adata.obs
    assert adata.obs["predicted_doublet"].dtype.name == "bool"

    out = cleanup_after_solo(adata, batch_key="sample", min_cells_per_sample=0)
    assert out.n_obs <= adata.n_obs


# -------------------------------------------------------------------------
# normalize_and_hvg
# -------------------------------------------------------------------------
def test_normalize_and_hvg():
    adata = synthetic_adata()
    out = normalize_and_hvg(adata, n_top_genes=10, batch_key="sample")

    assert "highly_variable" in out.var
    assert np.issubdtype(out.X.dtype, np.floating)


# -------------------------------------------------------------------------
# pca_neighbors_umap
# -------------------------------------------------------------------------
def test_pca_neighbors_umap():
    adata = synthetic_adata(n_genes=60)
    adata = normalize_and_hvg(adata, n_top_genes=30, batch_key="sample")
    out = pca_neighbors_umap(adata, var_explained=0.85, min_pcs=5, max_pcs=20)

    assert "X_umap" in out.obsm
    assert "n_pcs" in out.uns
    assert isinstance(out.uns["n_pcs"], int)


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

    cfg = LoadAndFilterConfig(
        metadata_tsv=mpath,
        batch_key="sample",
        raw_sample_dir=tmp_path / "raw",     # triggers load_raw_data mock
        filtered_sample_dir=None,
        cellbender_dir=None,
        output_dir=tmp_path,
        output_name="out.h5ad",
        mt_prefix="g",
        ribo_prefixes=["r"],
        hb_regex="hb",
        n_top_genes=60,
        min_genes=1,
        min_cells=1,
        min_cells_per_sample=0,
        max_pct_mt=100,
        n_jobs=1,
    )

    # Ensure input directory exists
    (tmp_path / "raw").mkdir(exist_ok=True)

    out = run_load_and_filter(cfg)

    # Basic checks
    assert isinstance(out, sc.AnnData)
    assert "leiden" in out.obs
    assert "X_umap" in out.obsm
    assert "batch_key" in out.uns
