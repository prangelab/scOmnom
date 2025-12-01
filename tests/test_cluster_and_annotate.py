# tests/test_cluster_and_annotate.py
import numpy as np
import pandas as pd
import scanpy as sc
import pytest

from scomnom.cluster_and_annotate import run_clustering, _compute_resolutions
from scomnom.config import ClusterAnnotateConfig


# ----------------------------------------------------------------------
# Synthetic data generator
# ----------------------------------------------------------------------
def synthetic_adata(n_cells=400, n_genes=50, n_batches=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_cells, n_genes))

    batch = rng.integers(0, n_batches, size=n_cells).astype(str)

    adata = sc.AnnData(X=X)
    adata.obs["batch"] = pd.Categorical(batch)

    # simple fake PCA + neighbors + UMAP embedding
    adata.obsm["X_pca"] = rng.normal(0, 1, size=(n_cells, 10))
    adata.obsm["X_umap"] = rng.normal(0, 1, size=(n_cells, 2))

    return adata


# ----------------------------------------------------------------------
# Mock CellTypist helper for biological mode
# ----------------------------------------------------------------------
@pytest.fixture
def mock_celltypist(monkeypatch):
    class DummyPreds:
        def __init__(self, n):
            self.predicted_labels = pd.Series(["Tcell"] * (n // 2) + ["Other"] * (n - n // 2))
            # probability matrix with two classes
            self.probability_matrix = pd.DataFrame(
                {
                    "Tcell": np.random.uniform(0.6, 1.0, n),
                    "Other": np.random.uniform(0.0, 0.4, n),
                },
                index=[f"cell{i}" for i in range(n)]
            )

    def dummy_annotate(adata, model, majority_voting=False):
        return DummyPreds(adata.n_obs)

    # patch the function used inside _precompute_celltypist
    import scomnom.cluster_and_annotate as ca
    monkeypatch.setattr("celltypist.annotate", dummy_annotate)
    # also bypass model load
    monkeypatch.setattr("celltypist.models.Model.load", lambda _: None)

    return True


# ----------------------------------------------------------------------
# Mock IO – avoid writing h5ad to disk
# ----------------------------------------------------------------------
@pytest.fixture
def fake_io(monkeypatch):
    import scomnom.io_utils as io

    monkeypatch.setattr(io, "save_adata", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "export_cluster_annotations", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "get_celltypist_model", lambda m: "dummy/path")

    return True


# ----------------------------------------------------------------------
# Test 1 — resolution sweep endpoints
# ----------------------------------------------------------------------
def test_compute_resolutions_basic():
    class DummyCfg:
        res_min = 0.2
        res_max = 1.0
        n_resolutions = 5

    out = _compute_resolutions(DummyCfg)
    assert len(out) == 5
    assert np.isclose(out[0], 0.2)
    assert np.isclose(out[-1], 1.0)


# ----------------------------------------------------------------------
# Test 2 — structural-only clustering
# ----------------------------------------------------------------------
def test_run_clustering_structural_only(tmp_path, fake_io):
    adata = synthetic_adata()
    in_path = tmp_path / "in.h5ad"
    adata.write_h5ad(in_path)

    cfg = ClusterAnnotateConfig(
        input_path=in_path,
        output_path=tmp_path / "out.h5ad",
        make_figures=False,
        bio_guided_clustering=False,
        celltypist_model=None,
        embedding_key="X_pca",
        label_key="leiden",
    )

    out = run_clustering(cfg)

    # must have final clustering
    assert "leiden" in out.obs
    assert "cluster_and_annotate" in out.uns

    info = out.uns["cluster_and_annotate"]

    # structural metadata present
    assert "clustering" in info
    cl = info["clustering"]

    for key in ["silhouette_centroid", "cluster_counts", "composite_scores"]:
        assert key in cl
        assert len(cl[key]) > 0

    # no biological metrics stored
    assert "bio_homogeneity" not in cl
    assert "bio_ari" not in cl
    assert "bio_fragmentation" not in cl


# ----------------------------------------------------------------------
# Test 3 — biological mode with mocked CellTypist
# ----------------------------------------------------------------------
def test_run_clustering_bio_mode(tmp_path, fake_io, mock_celltypist):
    adata = synthetic_adata()
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]  # needed for index alignment

    in_path = tmp_path / "in.h5ad"
    adata.write_h5ad(in_path)

    cfg = ClusterAnnotateConfig(
        input_path=in_path,
        output_path=tmp_path / "out.h5ad",
        make_figures=False,
        bio_guided_clustering=True,
        celltypist_model="dummy_model",
        embedding_key="X_pca",
        label_key="leiden",
    )

    out = run_clustering(cfg)
    info = out.uns["cluster_and_annotate"]
    cl = info["clustering"]

    # Biological metrics must exist
    assert "bio_homogeneity" in cl
    assert "bio_fragmentation" in cl
    assert "bio_ari" in cl
    assert "bio_composite" in cl

    # arrays must have the same resolution count
    n_res = len(cl["tested_resolutions"])
    assert len(cl["bio_homogeneity"]) == n_res
    assert len(cl["bio_ari"]) == n_res


# ----------------------------------------------------------------------
# Test 4 — majority vote annotation in structural mode
# ----------------------------------------------------------------------
def test_annotation_majority_vote(tmp_path, fake_io, mock_celltypist):
    # biological mode disabled → celltypist precompute happens but metrics ignored
    adata = synthetic_adata()
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]

    in_path = tmp_path / "in.h5ad"
    adata.write_h5ad(in_path)

    cfg = ClusterAnnotateConfig(
        input_path=in_path,
        output_path=tmp_path / "out.h5ad",
        make_figures=False,
        bio_guided_clustering=False,
        celltypist_model="dummy_model",
        embedding_key="X_pca",
        label_key="leiden",
    )

    out = run_clustering(cfg)

    # final labels must exist (cluster-collapsed)
    assert cfg.final_label_key in out.obs
    assert cfg.celltypist_cluster_label_key in out.obs

    # cluster-level labels consistent with majority voting
    clust = out.obs[cfg.label_key]
    final = out.obs[cfg.final_label_key]

    for c in clust.cat.categories:
        sub = final[clust == c]
        mode = sub.mode()[0]
        # all values in that cluster must equal the mode
        assert (sub == mode).all()


# ----------------------------------------------------------------------
# Test 5 — quality: silhouette QC stored
# ----------------------------------------------------------------------
def test_final_silhouette_qc(tmp_path, fake_io):
    adata = synthetic_adata()
    in_path = tmp_path / "in.h5ad"
    adata.write_h5ad(in_path)

    cfg = ClusterAnnotateConfig(
        input_path=in_path,
        output_path=tmp_path / "out.h5ad",
        make_figures=False,
        label_key="leiden",
        embedding_key="X_pca",
    )

    out = run_clustering(cfg)

    qc = out.uns["cluster_and_annotate"].get("real_silhouette_summary")
    assert qc is not None
    for k in ["mean", "median", "p10", "p90"]:
        assert k in qc
