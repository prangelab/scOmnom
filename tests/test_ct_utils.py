from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scomnom.ct_utils as ct_utils
from scomnom.config import ClusterAnnotateConfig
from scomnom.clustering_utils import _run_celltypist_annotation


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


def test_entropy_margin_mask_normalizes_independent_celltypist_scores():
    probabilities = pd.DataFrame(
        [
            [0.90, 0.10, 0.10],
            [0.90, 0.80, 0.10],
            [0.55, 0.50, 0.45],
        ],
        columns=["A", "B", "C"],
    )

    mask, stats = ct_utils.build_entropy_margin_mask(
        probabilities,
        entropy_abs_limit=0.0,
        entropy_quantile=2 / 3,
        margin_min=0.05,
    )

    assert mask.tolist() == [True, True, False]
    assert stats["entropy_probability_normalization"] == "row_sum"
    assert stats["entropy_cut_rule"] == "max_baseline_or_quantile"
    assert stats["probability_row_sum_min"] == pytest.approx(1.1)
    assert stats["probability_row_sum_max"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (pd.DataFrame({"A": [0.9, 0.8]}), "at least two"),
        (pd.DataFrame({"A": [0.9, np.nan], "B": [0.1, 0.2]}), "non-finite"),
        (pd.DataFrame({"A": [0.9, -0.1], "B": [0.1, 0.2]}), "lie in"),
        (pd.DataFrame({"A": [0.0, 0.8], "B": [0.0, 0.2]}), "no positive score"),
    ],
)
def test_entropy_margin_mask_rejects_invalid_probability_contract(probabilities, message):
    with pytest.raises(ValueError, match=message):
        ct_utils.build_entropy_margin_mask(
            probabilities,
            entropy_abs_limit=0.5,
            entropy_quantile=0.7,
            margin_min=0.1,
        )


def test_cluster_celltypist_summary_requires_coverage_and_strict_majority():
    index = pd.Index([f"cell{i}" for i in range(20)])
    clusters = pd.Series(["C00"] * 10 + ["C01"] * 10, index=index)
    labels = pd.Series(
        ["A"] * 5 + ["B"] * 5 + ["A"] * 6 + ["B"] * 4,
        index=index,
    )
    confidence = np.array([True] * 20)

    assignments, audit = ct_utils.summarize_cluster_celltypist_labels(
        clusters,
        labels,
        confidence,
        celltypist_ok=True,
        min_confident_cells=5,
        min_confident_fraction=0.5,
        min_label_purity=0.5,
    )

    assert assignments == {"C00": "Unknown", "C01": "A"}
    by_cluster = audit.set_index("cluster")
    assert by_cluster.loc["C00", "status"] == "insufficient_label_purity"
    assert by_cluster.loc["C00", "winning_label"] == "A"
    assert by_cluster.loc["C00", "runner_up_label"] == "B"
    assert by_cluster.loc["C01", "winning_fraction"] == pytest.approx(0.6)
    assert by_cluster.loc["C01", "status"] == "assigned"


def test_cluster_celltypist_summary_records_low_coverage_and_unavailable_states():
    clusters = pd.Series(["C00"] * 5 + ["C01"] * 5)
    labels = pd.Series(["A"] * 10)
    confidence = np.array([True, False, False, False, False] + [True] * 5)

    assignments, audit = ct_utils.summarize_cluster_celltypist_labels(
        clusters,
        labels,
        confidence,
        celltypist_ok=True,
        min_confident_cells=2,
        min_confident_fraction=0.2,
        min_label_purity=0.5,
    )
    assert assignments == {"C00": "Unknown", "C01": "A"}
    assert audit.set_index("cluster").loc["C00", "status"] == "insufficient_confident_cells"

    unavailable, unavailable_audit = ct_utils.summarize_cluster_celltypist_labels(
        clusters,
        labels,
        confidence,
        celltypist_ok=False,
        min_confident_cells=0,
        min_confident_fraction=0.0,
        min_label_purity=0.0,
    )
    assert set(unavailable.values()) == {"Unknown"}
    assert set(unavailable_audit["status"]) == {"celltypist_unavailable"}


def test_round_annotation_stores_cluster_vote_audit_and_rejects_tie(tmp_path):
    adata = _synthetic_adata(n_cells=12, n_genes=4)
    adata.obs["leiden"] = pd.Categorical(["0"] * 6 + ["1"] * 6)
    adata.uns["active_cluster_round"] = "r1"
    adata.uns["cluster_rounds"] = {"r1": {"cluster_key": "leiden"}}
    labels = np.array(["A"] * 3 + ["B"] * 3 + ["A"] * 4 + ["B"] * 2)
    probabilities = pd.DataFrame(
        np.where(labels[:, None] == np.array(["A", "B"])[None, :], 0.9, 0.1),
        index=adata.obs_names,
        columns=["A", "B"],
    )
    cfg = ClusterAnnotateConfig(
        input_path=tmp_path / "input.zarr",
        celltypist_model="Immune_All_Low.pkl",
        pretty_label_min_masked_cells=1,
        pretty_label_min_masked_frac=0.0,
        pretty_label_min_purity=0.5,
        bio_entropy_abs_limit=1.0,
        bio_entropy_quantile=1.0,
        bio_margin_min=0.1,
        make_figures=False,
    )

    _run_celltypist_annotation(
        adata,
        cfg,
        cluster_key="leiden",
        round_id="r1",
        precomputed_labels=labels,
        precomputed_proba=probabilities,
    )

    cluster_labels = adata.obs.groupby("leiden", observed=True)["celltypist_cluster_label__r1"].first()
    assert cluster_labels.astype(str).to_dict() == {"0": "Unknown", "1": "A"}
    annotation = adata.uns["cluster_rounds"]["r1"]["annotation"]
    assert annotation["pretty_label_min_purity"] == pytest.approx(0.5)
    audit = annotation["celltypist_cluster_label_audit"].set_index("cluster")
    assert audit.loc["0", "status"] == "insufficient_label_purity"
    assert audit.loc["1", "status"] == "assigned"
