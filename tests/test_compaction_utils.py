from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scomnom.compaction_utils import (
    _complete_link_components,
    _connected_components,
    compact_clusters_by_multiview_agreement,
    create_compacted_round_from_parent_round,
    export_compaction_audit_tables,
)
from scomnom.plot_utils import plot_compaction_review


CLUSTERS = ["A", "B", "C", "D"]


def _activity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [4.0, 4.1, -3.0, 0.0],
            "f2": [3.0, 3.1, -2.0, 1.0],
            "f3": [2.0, 2.1, 2.0, -1.0],
            "f4": [1.0, 1.1, 3.0, 0.0],
        },
        index=CLUSTERS,
    )


def _make_adata(
    *,
    labels: dict[str, str] | None = None,
    statuses: dict[str, str] | None = None,
    activity: pd.DataFrame | None = None,
    msigdb_activity: pd.DataFrame | None = None,
) -> tuple[ad.AnnData, dict]:
    labels = labels or {"A": "Myeloid", "B": "Myeloid", "C": "Myeloid", "D": "Other"}
    statuses = statuses or {cluster: "assigned" for cluster in CLUSTERS}
    cluster_values = np.repeat(CLUSTERS, 10)
    celltypist_values = np.concatenate(
        [np.repeat(labels[cluster] if statuses[cluster] == "assigned" else "Unknown", 10) for cluster in CLUSTERS]
    )
    obs = pd.DataFrame(
        {
            "clusters__r0": pd.Categorical(cluster_values),
            "ct_cluster__r0": pd.Categorical(celltypist_values),
            "ct_cell": pd.Categorical(celltypist_values),
        },
        index=[f"cell_{index}" for index in range(cluster_values.size)],
    )
    profiles = {
        "A": np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5]),
        "B": np.array([5.1, 4.1, 3.1, 2.1, 1.1, 0.6]),
        "C": np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "D": np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0]),
    }
    matrix = np.vstack([profiles[cluster] for cluster in cluster_values])
    adata = ad.AnnData(X=matrix, obs=obs)
    adata.var_names = [f"gene_{index}" for index in range(matrix.shape[1])]

    audit_rows = []
    for cluster in CLUSTERS:
        status = statuses[cluster]
        assigned = labels[cluster] if status == "assigned" else "Unknown"
        audit_rows.append(
            {
                "cluster": cluster,
                "n_total": 10,
                "n_confident": 8,
                "confident_fraction": 0.8,
                "winning_label": labels[cluster],
                "winning_count": 8,
                "winning_fraction": 1.0,
                "runner_up_label": "",
                "runner_up_count": 0,
                "runner_up_fraction": 0.0,
                "assigned_label": assigned,
                "status": status,
            }
        )

    activity = _activity_frame() if activity is None else activity
    msigdb_activity = activity.copy() if msigdb_activity is None else msigdb_activity
    round_snapshot = {
        "cluster_key": "clusters",
        "labels_obs_key": "clusters__r0",
        "annotation": {
            "celltypist_cluster_key": "ct_cluster__r0",
            "celltypist_cluster_label_audit": pd.DataFrame(audit_rows),
        },
        "decoupler": {
            "progeny": {"activity": activity.copy(), "method_provenance": {"method": "consensus"}},
            "dorothea": {"activity": activity.copy(), "method_provenance": {"method": "consensus"}},
            "msigdb": {
                "activity_by_gmt": {"HALLMARK": msigdb_activity.copy()},
                "method_provenance": {"method": "consensus"},
            },
        },
    }
    return adata, round_snapshot


def _run(adata: ad.AnnData, snapshot: dict, **kwargs):
    return compact_clusters_by_multiview_agreement(
        adata=adata,
        round_snapshot=snapshot,
        celltypist_obs_key="ct_cell",
        **kwargs,
    )


def test_complete_link_prevents_transitive_chain_merge():
    edges = {("A", "B"), ("B", "C")}
    assert _complete_link_components(["A", "B", "C"], edges) == [["A", "B"], ["C"]]
    assert _connected_components(["A", "B", "C"], list(edges)) == [["A", "B", "C"]]


def test_compaction_merges_only_supported_pair():
    adata, snapshot = _make_adata()
    result = _run(adata, snapshot)

    assert any(set(component) == {"A", "B"} for component in result.components)
    assert not any(set(component) >= {"A", "B", "C"} for component in result.components)
    pair = result.edges.set_index(["a", "b"]).loc[("A", "B")]
    assert bool(pair["pass_all"])
    assert bool(pair["pass_transcriptome"])
    assert pair["sim_transcriptome"] >= 0.90
    thresholds = result.thresholds_by_label.query("celltypist_label == 'Myeloid'")
    assert not thresholds["adaptive_used"].any()
    assert set(thresholds["effective_threshold"]) == {0.90, 0.70, 0.60}


def test_transcriptomic_guard_vetoes_activity_supported_pair():
    labels = {"A": "Pair", "B": "Pair", "C": "Other C", "D": "Other D"}
    adata, snapshot = _make_adata(labels=labels)
    cluster_b = adata.obs["clusters__r0"].astype(str) == "B"
    adata.X[cluster_b.to_numpy()] = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

    result = _run(adata, snapshot)

    pair = result.edges.set_index(["a", "b"]).loc[("A", "B")]
    assert bool(pair["pass_progeny"])
    assert bool(pair["pass_dorothea"])
    assert bool(pair["pass_msigdb"])
    assert not bool(pair["pass_transcriptome"])
    assert not bool(pair["pass_all"])
    assert result.cluster_id_map["A"] != result.cluster_id_map["B"]


def test_transcriptomic_auto_source_prefers_cellbender_counts():
    adata, snapshot = _make_adata()
    adata.layers["counts_raw"] = sparse.csr_matrix(np.rint(adata.X * 10.0))
    adata.layers["counts_cb"] = sparse.csr_matrix(np.rint(adata.X * 8.0))

    result = _run(adata, snapshot)

    provenance = result.transcriptomic_provenance
    assert provenance["requested_source"] == "auto"
    assert provenance["resolved_source"] == "counts_cb"
    assert provenance["aggregation"] == "cluster_sum_target_10000_log1p"
    assert provenance["n_selected_features"] == adata.n_vars
    assert len(provenance["selected_feature_sha256"]) == 64


def test_missing_required_activity_row_is_ineligible_without_imputation():
    activity = _activity_frame().drop(index="D")
    adata, snapshot = _make_adata(activity=activity)
    result = _run(adata, snapshot)

    eligibility = result.cluster_eligibility.set_index("cluster")
    assert not bool(eligibility.loc["D", "eligible"])
    assert "missing_progeny_activity" in eligibility.loc["D", "exclusion_reasons"]
    assert result.cluster_id_map["D"] != result.cluster_id_map["A"]


def test_missing_required_view_stops_compaction():
    adata, snapshot = _make_adata()
    snapshot["decoupler"]["progeny"]["activity"] = None
    with pytest.raises(ValueError, match="requires non-empty PROGENy"):
        _run(adata, snapshot)


def test_invalid_configured_msigdb_block_stops_required_compaction():
    adata, snapshot = _make_adata()
    snapshot["decoupler"]["msigdb"]["activity_by_gmt"]["BROKEN"] = pd.DataFrame()
    with pytest.raises(ValueError, match="MSigDB:BROKEN"):
        _run(adata, snapshot)


def test_celltypist_audit_is_required():
    adata, snapshot = _make_adata()
    snapshot["annotation"].pop("celltypist_cluster_label_audit")
    with pytest.raises(KeyError, match="cluster_label_audit"):
        _run(adata, snapshot)


def test_unassigned_and_small_clusters_remain_singletons():
    statuses = {cluster: "assigned" for cluster in CLUSTERS}
    statuses["B"] = "insufficient_label_purity"
    adata, snapshot = _make_adata(statuses=statuses)
    result = _run(adata, snapshot, min_cells=11)

    eligibility = result.cluster_eligibility.set_index("cluster")
    assert not eligibility["eligible"].any()
    assert "celltypist_insufficient_label_purity" in eligibility.loc["B", "exclusion_reasons"]
    assert all(len(component) == 1 for component in result.components)


def test_adaptive_thresholds_are_used_only_for_groups_of_four_or_more():
    labels = {cluster: "Shared" for cluster in CLUSTERS}
    adata, snapshot = _make_adata(labels=labels)
    result = _run(adata, snapshot)

    thresholds = result.thresholds_by_label
    assert thresholds["adaptive_used"].all()
    assert set(thresholds["n_clusters"]) == {4}
    assert np.isfinite(thresholds["adaptive_value"]).all()


def test_optional_msigdb_is_retained_as_diagnostic_evidence():
    msigdb = _activity_frame()
    msigdb.loc["B"] = -msigdb.loc["A"]
    labels = {"A": "Pair", "B": "Pair", "C": "Other C", "D": "Other D"}
    adata, snapshot = _make_adata(labels=labels, msigdb_activity=msigdb)
    result = _run(adata, snapshot, msigdb_required=False)

    pair = result.edges.set_index(["a", "b"]).loc[("A", "B")]
    assert "sim_msigdb__HALLMARK" in result.edges.columns
    assert not bool(pair["pass_msigdb__HALLMARK"])
    assert bool(pair["pass_all"])


def test_msigdb_decision_margin_matches_majority_rule():
    labels = {"A": "Pair", "B": "Pair", "C": "Other C", "D": "Other D"}
    adata, snapshot = _make_adata(labels=labels)
    passing = _activity_frame()
    failing = passing.copy()
    failing.loc["B"] = -failing.loc["A"]
    snapshot["decoupler"]["msigdb"]["activity_by_gmt"] = {
        "BLOCK_1": passing.copy(),
        "BLOCK_2": passing.copy(),
        "BLOCK_3": passing.copy(),
        "BLOCK_4": failing,
    }
    result = _run(adata, snapshot)

    pair = result.edges.set_index(["a", "b"]).loc[("A", "B")]
    assert pair["msigdb_majority_needed"] == 3
    assert pair["msigdb_majority_passed"] == 3
    assert bool(pair["pass_msigdb"])
    assert pair["msigdb_decision_margin"] >= 0.0
    assert pair["decision_margin"] >= 0.0


def test_threshold_caps_cannot_undercut_floors():
    adata, snapshot = _make_adata()
    with pytest.raises(ValueError, match="progeny_threshold_cap"):
        _run(adata, snapshot, progeny_threshold_cap=0.69)
    with pytest.raises(ValueError, match="MSigDB cap"):
        _run(adata, snapshot, msigdb_threshold_cap_by_gmt={"HALLMARK": 0.59})
    with pytest.raises(ValueError, match="msigdb_threshold_cap"):
        _run(adata, snapshot, msigdb_threshold_cap=0.59)
    with pytest.raises(ValueError, match="transcriptomic_threshold_cap"):
        _run(adata, snapshot, transcriptomic_threshold_cap=0.89)


def test_no_op_child_is_active_and_persists_review_tables(tmp_path):
    labels = {cluster: f"Label {cluster}" for cluster in CLUSTERS}
    adata, snapshot = _make_adata(labels=labels)
    adata.uns["cluster_rounds"] = {"r0": snapshot}
    adata.uns["active_cluster_round"] = "r0"
    cfg = SimpleNamespace(model_dump=lambda mode="json": {"compact_grouping": "complete_link"})

    create_compacted_round_from_parent_round(
        adata,
        cfg,
        parent_round_id="r0",
        new_round_id="r1_compacted",
        celltypist_obs_key="ct_cell",
    )

    child = adata.uns["cluster_rounds"]["r1_compacted"]
    assert adata.uns["active_cluster_round"] == "r1_compacted"
    assert child["compacting"]["did_merge"] is False
    assert child["compacting"]["method_identity"] == "multiview_all_pairs_with_transcriptomic_guard"
    assert child["compacting"]["transcriptomic_provenance"]["resolved_source"] == "X"
    assert child["compacting"]["threshold_policy"]["transcriptomic_floor"] == pytest.approx(0.90)
    assert child["compacting"]["components"]
    assert child["cfg"]["compact_grouping"] == "complete_link"

    written = export_compaction_audit_tables(
        adata, round_id="r1_compacted", output_dir=tmp_path / "compaction"
    )
    assert set(written) == {
        "view_audit", "cluster_eligibility", "thresholds_by_label",
        "pairwise_evidence", "group_membership",
    }
    assert all(path.exists() for path in written.values())

    artifacts = plot_compaction_review(
        adata,
        child_round_id="r1_compacted",
        figdir="cluster_and_annotate/r1_compacted/clustering",
    )
    assert len(artifacts) == 1
    assert artifacts[0].stem == "compaction_review"
    assert artifacts[0].fig is not None
