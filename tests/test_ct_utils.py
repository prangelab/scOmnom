from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import scomnom.ct_utils as ct_utils
from scomnom.config import ClusterAnnotateConfig


def _synthetic_adata(n_cells: int = 12, n_genes: int = 6) -> ad.AnnData:
    X = np.random.default_rng(0).poisson(1.0, size=(n_cells, n_genes))
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.var_names = [f"gene{i}" for i in range(n_genes)]
    return adata


def test_clusterannotate_celltypist_model_none_string_normalizes(tmp_path):
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "a.h5ad",
        celltypist_model="None",
    )
    assert cfg.celltypist_model is None


def test_ensure_celltypist_skips_reuse_when_model_disabled():
    adata = _synthetic_adata()
    adata.obs["celltypist_label"] = pd.Categorical(["Immune"] * adata.n_obs)
    adata.obsm["celltypist_proba"] = np.ones((adata.n_obs, 2), dtype=float) * 0.5
    adata.uns["celltypist_proba_columns"] = ["Immune", "Other"]
    adata.uns["celltypist_meta"] = {"model_name": "Immune_All_Low.pkl"}

    cfg = ClusterAnnotateConfig(
        input_path=Path("dummy.h5ad"),
        celltypist_model="None",
        make_figures=False,
    )

    labels, proba, meta = ct_utils.ensure_celltypist(adata, cfg, reuse=True, store=True)

    assert labels is None
    assert proba is None
    assert meta["reused"] is False
    assert meta["requested_model"] is None


def test_ensure_celltypist_recomputes_when_requested_model_differs(monkeypatch):
    adata = _synthetic_adata(n_cells=10, n_genes=5)

    stale_labels = np.array(["Immune"] * adata.n_obs, dtype=object)
    stale_proba = pd.DataFrame(
        {"Immune": np.full(adata.n_obs, 0.9), "Other": np.full(adata.n_obs, 0.1)},
        index=adata.obs_names,
    )
    fresh_proba = pd.DataFrame(
        {"Stromal": np.full(adata.n_obs, 0.8), "Other": np.full(adata.n_obs, 0.2)},
        index=adata.obs_names,
    )

    monkeypatch.setattr(
        ct_utils,
        "get_celltypist_outputs",
        lambda adata_in, label_key, **kwargs: (
            stale_labels,
            stale_proba,
            {"labels_ok": True, "proba_ok": True, "model_name": "Immune_All_Low.pkl"},
        ),
    )
    monkeypatch.setattr(ct_utils, "get_celltypist_model", lambda model_name: Path(f"/tmp/{model_name}"))
    monkeypatch.setattr("celltypist.models.Model.load", lambda _: object())

    class DummyPreds:
        def __init__(self, n_obs, index):
            self.predicted_labels = pd.Series(["Stromal"] * n_obs, index=index)
            self.probability_matrix = fresh_proba

    monkeypatch.setattr(
        "celltypist.annotate",
        lambda adata_ct, model, majority_voting=False: DummyPreds(adata_ct.n_obs, adata_ct.obs_names),
    )

    cfg = ClusterAnnotateConfig(
        input_path=Path("dummy.h5ad"),
        celltypist_model="Fibroblast.pkl",
        make_figures=False,
    )

    labels, proba, meta = ct_utils.ensure_celltypist(adata, cfg, reuse=True, store=True)

    assert meta["reused"] is False
    assert meta["requested_model"] == "Fibroblast.pkl"
    assert labels is not None
    assert set(labels.tolist()) == {"Stromal"}
    assert proba is not None
    assert adata.uns["celltypist_meta"]["model_name"] == "Fibroblast.pkl"
