# tests/test_integration.py

import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from pathlib import Path

from scomnom.integrate import (
    run_integration,
    _subset_to_hvgs,
    _run_all_embeddings,
)
from scomnom.config import IntegrationConfig


# -----------------------------------------------------------------------------
# Synthetic AnnData fixture
# -----------------------------------------------------------------------------
def synthetic_adata(n_cells=300, n_genes=50, n_batches=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_cells, n_genes))

    adata = sc.AnnData(X)
    adata.obs["batch"] = pd.Categorical(
        rng.integers(0, n_batches, size=n_cells).astype(str)
    )
    adata.obs["label"] = pd.Categorical(
        rng.integers(0, 4, size=n_cells).astype(str)
    )

    # mark all genes as HVGs (simple)
    adata.var["highly_variable"] = True

    # initial PCA
    adata.obsm["X_pca"] = rng.normal(0, 1, size=(n_cells, 10))

    return adata


# -----------------------------------------------------------------------------
# Global monkeypatch: mock scanorama, harmony, scVI, BBKNN, scANVI
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_integration_methods(monkeypatch):
    # --- Scanorama ------------------------------------
    monkeypatch.setattr(
        "scanorama.integrate_scanpy",
        lambda ad_list: [a.obsm.setdefault("X_scanorama", np.random.randn(a.n_obs, 10)) for a in ad_list]
    )

    # --- Harmony ---------------------------------------
    class DummyHO:
        def __init__(self, Z):
            self.Z_corr = Z

    monkeypatch.setattr(
        "harmonypy.run_harmony",
        lambda X, obs, batch_key: DummyHO(np.random.randn(X.shape[0], 10))
    )

    # --- BBKNN -----------------------------------------
    import scomnom.integrate as integ
    monkeypatch.setattr(
        integ,
        "_run_bbknn_embedding",
        lambda adata, batch_key: np.random.randn(adata.n_obs, 2)
    )

    # --- SCVI ------------------------------------------
    class DummySCVI:
        def __init__(self, adata):
            self.adata = adata

        @staticmethod
        def setup_anndata(*args, **kwargs):
            return

        def train(self, *a, **kw):
            return

        def get_latent_representation(self):
            return np.random.randn(self.adata.n_obs, 10)

    monkeypatch.setattr(
        "scvi.model.SCVI",
        lambda *a, **kw: DummySCVI(a[0])
    )

    # --- SCANVI ----------------------------------------
    class DummySCANVI:
        @staticmethod
        def from_scvi_model(scvi_model, adata, labels_key, unlabeled_category):
            return DummySCANVI()

        def train(self, *a, **k):
            return

        def get_latent_representation(self):
            return np.random.randn(300, 10)

    monkeypatch.setattr(
        "scvi.model.SCANVI",
        DummySCANVI
    )

    return True


# -----------------------------------------------------------------------------
# Mock scIB benchmarking
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_scib(monkeypatch):
    class DummyBenchmarker:
        def __init__(self, adata, batch_key, label_key, embedding_obsm_keys,
                     bio_conservation_metrics, batch_correction_metrics, n_jobs):
            self.adata = adata
            self.keys = embedding_obsm_keys

        def benchmark(self):
            return

        def get_results(self, min_max_scale=False):
            # Return deterministic mock results
            vals = {
                k: [0.8, 0.6, 0.9] if "Scanorama" in k else [0.5, 0.3, 0.4]
                for k in self.keys
            }
            df = pd.DataFrame(vals, index=["clisi", "ilisi", "pcr"])
            df.loc["Metric Type", :] = "metric"
            return df

    monkeypatch.setattr(
        "scib_metrics.benchmark.Benchmarker",
        DummyBenchmarker
    )
    monkeypatch.setattr(
        "scib_metrics.benchmark.BioConservation",
        lambda: None
    )
    monkeypatch.setattr(
        "scib_metrics.benchmark.BatchCorrection",
        lambda: None
    )

    return True


# -----------------------------------------------------------------------------
# Mock plot saving and IO (no filesystem writes)
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_io(monkeypatch):
    import scomnom.io_utils as io
    monkeypatch.setattr(io, "save_adata", lambda *args, **kw: None)
    monkeypatch.setattr(io, "infer_batch_key", lambda ad, k: "batch")

    # avoid saving figures
    import scomnom.plot_utils as pu
    monkeypatch.setattr(pu, "save_multi", lambda *args, **kw: None)

    return True


# -----------------------------------------------------------------------------
# Basic HVG subsetting test
# -----------------------------------------------------------------------------
def test_subset_to_hvgs():
    adata = synthetic_adata()
    out = _subset_to_hvgs(adata)
    assert out.n_vars == adata.n_vars  # all HVGs True
    assert out.n_obs == adata.n_obs


# -----------------------------------------------------------------------------
# Test embedding orchestrator (mocked)
# -----------------------------------------------------------------------------
def test_run_all_embeddings(mock_integration_methods):
    adata = synthetic_adata()
    keys = _run_all_embeddings(
        adata,
        methods=["Scanorama", "Harmony", "BBKNN", "scVI", "scANVI"],
        batch_key="batch",
        label_key="label",
    )
    # must include PCA + all methods
    assert "Unintegrated" in keys
    assert "Scanorama" in keys
    assert "Harmony" in keys
    assert "BBKNN" in keys
    assert "scVI" in keys
    assert "scANVI" in keys


# -----------------------------------------------------------------------------
# Full pipeline test: run_integration()
# -----------------------------------------------------------------------------
def test_run_integration_full(
    tmp_path,
    mock_integration_methods,
    mock_scib,
    mock_io,
):
    adata = synthetic_adata()
    in_path = tmp_path / "input.h5ad"
    adata.write_h5ad(in_path)

    cfg = IntegrationConfig(
        input_path=in_path,
        output_path=tmp_path / "out.h5ad",
        batch_key="batch",
        label_key="label",
        methods=["Scanorama", "Harmony", "scVI"],
        benchmark_n_jobs=1,
        make_figures=False,
    )

    out = run_integration(cfg)

    # basic existence checks
    assert "integration" in out.uns
    meta = out.uns["integration"]
    assert meta["best_embedding"] in meta["methods"]

    # UMAP must exist
    assert "X_umap" in out.obsm

    # integrated embedding stored
    assert "X_integrated" in out.obsm
    assert out.obsm["X_integrated"].shape[0] == out.n_obs

    # embeddings copied
    for k in meta["methods"]:
        assert k in out.obsm
