import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from pathlib import Path
from unittest.mock import Mock

import scomnom.clustering_utils as cu
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


def test_resolution_sweep_stores_fixed_selector_rules(tmp_path, monkeypatch):
    adata = synthetic_adata()
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        res_min=0.1,
        res_max=0.3,
        n_resolutions=3,
        bio_guided_clustering=False,
    )

    def fake_leiden(adata_in, *, key_added, **kwargs):
        adata_in.obs[key_added] = pd.Categorical(
            np.tile(["0", "1"], reps=adata_in.n_obs // 2)
        )

    monkeypatch.setattr(cu.sc.tl, "leiden", fake_leiden)
    monkeypatch.setattr(cu, "_centroid_silhouette", lambda *args, **kwargs: 0.5)

    _, sweep, _ = cu._resolution_sweep(
        adata,
        cfg,
        "X_pca",
        celltypist_labels=None,
    )

    assert sweep["selection_rules"] == cu._bisc_fixed_rule_snapshot()


def test_plateau_probe_subsampling_reuses_one_neighbor_graph_per_repeat(
    tmp_path, monkeypatch
):
    adata = synthetic_adata(n_cells=12)
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        stability_repeats=3,
        subsample_frac=0.75,
        random_state=7,
    )
    labels = {
        0.2: np.tile(np.array(["0", "1"]), 6),
        0.4: np.tile(np.array(["0", "1", "2"]), 4),
    }
    neighbor_calls = []

    def fake_neighbors(adata_in, **kwargs):
        neighbor_calls.append((adata_in.n_obs, kwargs["random_state"]))

    def fake_leiden(adata_in, *, resolution, key_added, **kwargs):
        n_clusters = 2 if np.isclose(resolution, 0.2) else 3
        adata_in.obs[key_added] = pd.Categorical(
            np.arange(adata_in.n_obs) % n_clusters
        )

    monkeypatch.setattr(cu.sc.pp, "neighbors", fake_neighbors)
    monkeypatch.setattr(cu.sc.tl, "leiden", fake_leiden)

    result = cu._subsampling_candidate_stability(
        adata,
        cfg,
        "X_pca",
        labels,
        [0.2, 0.4],
    )

    assert neighbor_calls == [(9, 7), (9, 8), (9, 9)]
    assert set(result) == {0.2, 0.4}
    assert all(len(values) == 3 for values in result.values())


def test_resolution_sweep_uses_probe_subsampling_to_choose_plateau(
    tmp_path, monkeypatch
):
    adata = synthetic_adata()
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "integrated.zarr",
        res_min=0.1,
        res_max=0.6,
        n_resolutions=6,
        bio_guided_clustering=False,
    )

    def fake_leiden(adata_in, *, key_added, **kwargs):
        adata_in.obs[key_added] = pd.Categorical(
            np.tile(["0", "1"], reps=adata_in.n_obs // 2)
        )

    monkeypatch.setattr(cu.sc.tl, "leiden", fake_leiden)
    monkeypatch.setattr(cu, "_centroid_silhouette", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        cu,
        "_detect_plateaus",
        lambda *args, **kwargs: [
            cu.Plateau([0.2, 0.3], mean_stability=0.95),
            cu.Plateau([0.4, 0.5], mean_stability=0.90),
        ],
    )

    def fake_probe_stability(
        adata_in, cfg_in, embedding_key, labels_per_resolution, candidate_resolutions
    ):
        del adata_in, cfg_in, embedding_key, labels_per_resolution
        assert candidate_resolutions == [0.2, 0.4]
        return {0.2: [0.93, 0.93], 0.4: [0.95, 0.95]}

    monkeypatch.setattr(cu, "_subsampling_candidate_stability", fake_probe_stability)

    best, sweep, _ = cu._resolution_sweep(
        adata,
        cfg,
        "X_pca",
        celltypist_labels=None,
    )

    assert best == 0.4
    assert sweep["selection"]["selected_plateau_index"] == 1
    assert sweep["selection"]["alternative_plateau_index"] == 0
    assert sweep["selection"]["mode"] == "plateau_probe_subsampling"
    assert sweep["selection"]["confidence"] == "multiscale"
    assert sweep["selection"]["probe_reproducibility_gap"] == pytest.approx(0.02)
    assert sweep["selection"]["selected_probe_n_clusters"] == 2
    assert sweep["selection"]["alternative_probe_n_clusters"] == 2


def test_clustering_report_captions_distinguish_stability_concepts():
    from scomnom.reporting import _describe_plot

    assert _describe_plot(Path("clustering_stability_ari.png")) == (
        "Post-selection subsampling reproducibility (ARI vs full-data partition)."
    )
    assert _describe_plot(Path("cluster_selection_stability.png")) == (
        "Resolution selection metrics, including adjacent-resolution stability."
    )
    assert _describe_plot(Path("plateau_probe_reproducibility.png")) == (
        "Fixed-resolution subsampling reproducibility used to select among BISC plateaus."
    )


def test_round_diagnostics_ignore_legacy_penalized_scores(tmp_path, monkeypatch):
    import scomnom.cluster_and_annotate as ca

    adata = synthetic_adata()
    legacy_scores = {"0.200": 0.1, "0.500": 0.05}
    adata.uns["active_cluster_round"] = "r0"
    adata.uns["cluster_rounds"] = {
        "r0": {
            "best_resolution": 0.5,
            "sweep": {"resolutions": [0.2, 0.5]},
            "diagnostics": {
                "tested_resolutions": [0.2, 0.5],
                "silhouette_centroid": {"0.200": 0.2, "0.500": 0.3},
                "cluster_counts": {"0.200": 5, "0.500": 8},
                "penalized_scores": legacy_scores.copy(),
            },
        }
    }

    sweep_plot = Mock(return_value=[])
    monkeypatch.setattr(ca.plot_utils, "plot_clustering_resolution_sweep", sweep_plot)
    monkeypatch.setattr(ca.plot_utils, "plot_cluster_umaps", Mock(return_value=[]))
    monkeypatch.setattr(ca.plot_utils, "plot_clustering_stability_ari", Mock(return_value=[]))
    monkeypatch.setattr(ca.plot_utils, "persist_plot_artifacts", Mock())

    cfg = ClusterAnnotateConfig(input_path=tmp_path / "legacy.zarr", label_key="leiden")
    ca._plot_round_clustering_diagnostics(adata, cfg, embedding_key="X_pca", batch_key="batch")

    assert "penalized_scores" not in sweep_plot.call_args.kwargs
    assert adata.uns["cluster_rounds"]["r0"]["diagnostics"]["penalized_scores"] == legacy_scores


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
