import numpy as np
import pandas as pd
import scanpy as sc
from unittest.mock import Mock

from scomnom.cluster_and_annotate import run_clustering
from scomnom.clustering_utils import _compute_resolutions
from scomnom.config import ClusterAnnotateConfig


def synthetic_adata(n_cells=48, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    adata = sc.AnnData(X=rng.normal(size=(n_cells, n_genes)))
    adata.obs["batch"] = pd.Categorical(np.repeat(["b1", "b2"], repeats=n_cells // 2))
    adata.obs["celltypist_label"] = pd.Categorical(np.tile(["T cell", "B cell"], reps=n_cells // 2))
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 6))
    return adata


def test_compute_resolutions_basic(tmp_path):
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        res_min=0.2,
        res_max=1.0,
        n_resolutions=5,
    )

    out = _compute_resolutions(cfg)

    assert len(out) == 5
    assert np.isclose(out[0], 0.2)
    assert np.isclose(out[-1], 1.0)


def test_run_clustering_uses_current_round_pipeline(tmp_path, monkeypatch):
    import scomnom.cluster_and_annotate as ca

    adata = synthetic_adata()
    save_mock = Mock()
    ensure_mock = Mock(return_value=(adata.obs["celltypist_label"], None, {}))
    run_bisc_mock = Mock()

    def fake_run_bisc(adata_in, cfg, **kwargs):
        adata_in.obs[cfg.label_key] = pd.Categorical(np.tile(["0", "1"], reps=adata_in.n_obs // 2))
        adata_in.uns["active_cluster_round"] = "r1"
        adata_in.uns["cluster_rounds"] = {
            "r1": {
                "cluster_key": cfg.label_key,
                "best_resolution": 0.5,
                "diagnostics": {"tested_resolutions": [0.2, 0.5]},
            }
        }
        run_bisc_mock(adata_in, cfg, **kwargs)

    monkeypatch.setattr(ca.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(ca.io_utils, "save_dataset", save_mock)
    monkeypatch.setattr(ca.io_utils, "infer_batch_key", lambda adata, key: key or "batch")
    monkeypatch.setattr(ca.plot_utils, "setup_scanpy_figs", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.plot_utils, "capture_plot_artifacts", ca.plot_utils.capture_plot_artifacts)
    monkeypatch.setattr(ca.plot_utils, "persist_plot_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_recompute_hvg_and_pca", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_ensure_embedding", lambda adata, embedding_key: embedding_key)
    monkeypatch.setattr(ca.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.sc.tl, "umap", lambda adata: adata.obsm.__setitem__("X_umap", np.zeros((adata.n_obs, 2))))
    monkeypatch.setattr(ca.ct_utils, "ensure_celltypist", ensure_mock)
    monkeypatch.setattr(ca, "run_BISC", fake_run_bisc)
    monkeypatch.setattr(ca, "_plot_round_clustering_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ca,
        "_run_celltypist_annotation",
        lambda *args, **kwargs: {
            "round_id": "r1",
            "celltypist_cell_key": "celltypist_label",
            "celltypist_cluster_key": "celltypist_cluster_label",
            "pretty_cluster_key": "cluster_label__r1",
        },
    )
    monkeypatch.setattr(ca, "_export_round_annotations_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "clear_top_level_decoupler_state", lambda *args, **kwargs: None)

    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        output_dir=tmp_path / "results",
        make_figures=False,
        run_decoupler=False,
        enable_compacting=False,
        embedding_key="X_pca",
        label_key="leiden",
    )

    out = run_clustering(cfg)

    assert out is adata
    assert out.uns["active_cluster_round"] == "r1"
    assert "leiden" in out.obs
    assert run_bisc_mock.call_args.kwargs["embedding_key"] == "X_pca"
    assert ensure_mock.call_args.kwargs["reuse"] is True
    save_mock.assert_called_once()


def test_force_celltypist_recompute_disables_reuse(tmp_path, monkeypatch):
    import scomnom.cluster_and_annotate as ca

    adata = synthetic_adata()
    ensure_mock = Mock(return_value=(None, None, {}))

    monkeypatch.setattr(ca.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(ca.io_utils, "save_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.io_utils, "infer_batch_key", lambda adata, key: key or "batch")
    monkeypatch.setattr(ca.plot_utils, "setup_scanpy_figs", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.plot_utils, "persist_plot_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_recompute_hvg_and_pca", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_ensure_embedding", lambda adata, embedding_key: embedding_key)
    monkeypatch.setattr(ca.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.sc.tl, "umap", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca.ct_utils, "ensure_celltypist", ensure_mock)
    monkeypatch.setattr(
        ca,
        "run_BISC",
        lambda adata, cfg, **kwargs: (
            adata.uns.update({"active_cluster_round": "r1", "cluster_rounds": {"r1": {"cluster_key": cfg.label_key}}}),
            adata.obs.__setitem__(cfg.label_key, pd.Categorical(["0"] * adata.n_obs)),
        ),
    )
    monkeypatch.setattr(ca, "_plot_round_clustering_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_run_celltypist_annotation", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "_export_round_annotations_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(ca, "clear_top_level_decoupler_state", lambda *args, **kwargs: None)

    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        make_figures=False,
        run_decoupler=False,
        enable_compacting=False,
        force_celltypist_recompute=True,
        embedding_key="X_pca",
        label_key="leiden",
    )

    run_clustering(cfg)

    assert ensure_mock.call_args.kwargs["reuse"] is False
    assert ensure_mock.call_args.kwargs["store"] is True
