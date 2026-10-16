from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scomnom import load_dataset, save_dataset
from scomnom import clustering_utils as cu
from scomnom.config import ClusterAnnotateConfig
from scomnom.reporting import generate_cluster_and_annotate_report


def _register(adata, round_id, parent_round_id=None):
    labels_key = f"leiden__{round_id}"
    adata.obs[labels_key] = pd.Categorical(["0", "0", "1", "1"])
    cu._register_round(
        adata,
        round_id=round_id,
        cluster_key="leiden",
        labels_obs_key=labels_key,
        kind="test",
        best_resolution=0.5,
        sweep=None,
        cfg_snapshot=None,
        parent_round_id=parent_round_id,
    )


def _adata_with_rounds():
    adata = ad.AnnData(np.arange(12, dtype=np.float32).reshape(4, 3))
    adata.layers["counts_raw"] = adata.X.copy()
    adata.layers["counts_cb"] = adata.X.copy()
    adata.obsm["X_umap"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    _register(adata, "r0_parent")
    _register(adata, "r1_compacted", parent_round_id="r0_parent")
    adata.obs["cluster_label__r0_parent"] = pd.Categorical(["A", "A", "B", "B"])
    adata.uns["cluster_rounds"]["r0_parent"]["annotation"] = {
        "pretty_label_key": "cluster_label__r0_parent"
    }
    return adata


@pytest.mark.parametrize("factory", [list, tuple, np.array, pd.Index])
def test_round_scaffold_preserves_valid_sequence(factory):
    adata = _adata_with_rounds()
    expected = ["r1_compacted", "r0_parent"]
    adata.uns["cluster_round_order"] = factory(expected)
    payloads = deepcopy(adata.uns["cluster_rounds"])

    cu._ensure_cluster_rounds(adata)
    cu._ensure_cluster_rounds(adata)

    assert adata.uns["cluster_round_order"] == expected
    assert isinstance(adata.uns["cluster_round_order"], list)
    assert adata.uns["cluster_rounds"] == payloads
    assert adata.uns["active_cluster_round"] == "r1_compacted"


@pytest.mark.parametrize("order", [[], (), np.array([]), pd.Index([])])
def test_round_scaffold_accepts_empty_sequence(order):
    adata = ad.AnnData(np.eye(2))
    adata.uns["cluster_round_order"] = order
    cu._ensure_cluster_rounds(adata)
    assert adata.uns["cluster_round_order"] == []
    assert adata.uns["cluster_rounds"] == {}
    assert adata.uns["active_cluster_round"] is None


def test_round_scaffold_initializes_missing_metadata():
    adata = ad.AnnData(np.eye(2))
    cu._ensure_cluster_rounds(adata)
    assert adata.uns["cluster_round_order"] == []
    assert adata.uns["cluster_rounds"] == {}
    assert adata.uns["active_cluster_round"] is None


@pytest.mark.parametrize("order", [
    None, "r0_parent", {"r0_parent": 0}, {"r0_parent"}, 1,
    np.array("r0_parent"), np.array([["r0_parent"]]),
    ["r0_parent", None], [0], [""], [["r0_parent"]],
    ["r0_parent", "r0_parent"],
])
def test_invalid_round_order_fails_before_registration_without_erasing_history(order):
    adata = _adata_with_rounds()
    adata.uns["cluster_round_order"] = order
    payloads = deepcopy(adata.uns["cluster_rounds"])
    labels = adata.obs.copy(deep=True)

    with pytest.raises(ValueError, match="cluster_round_order"):
        cu._ensure_cluster_rounds(adata)
    with pytest.raises(ValueError, match="cluster_round_order"):
        cu._register_round(
            adata, round_id="r2_new", cluster_key="leiden",
            labels_obs_key="leiden__r0_parent", kind="test",
            best_resolution=None, sweep=None, cfg_snapshot=None,
        )

    assert adata.uns["cluster_round_order"] is order
    assert adata.uns["cluster_rounds"] == payloads
    assert adata.uns["active_cluster_round"] == "r1_compacted"
    pd.testing.assert_frame_equal(adata.obs, labels)


@pytest.mark.parametrize("fmt,archive", [("zarr", False), ("zarr", True), ("h5ad", False)])
def test_roundtrip_append_and_activation_preserve_prior_history(tmp_path, fmt, archive):
    original = _adata_with_rounds()
    path = tmp_path / f"parent.{fmt}"
    save_dataset(original, path, fmt=fmt, archive=archive)
    loaded = load_dataset(path.with_suffix(path.suffix + ".tar.zst") if archive else path)
    prior_obs = loaded.obs.copy(deep=True)
    payloads = deepcopy(loaded.uns["cluster_rounds"])

    assert cu._next_round_index(loaded) == 2
    _register(loaded, "r2_subset", parent_round_id="r1_compacted")
    _register(loaded, "r3_subset_compacted", parent_round_id="r2_subset")
    expected_order = ["r0_parent", "r1_compacted", "r2_subset", "r3_subset_compacted"]
    assert loaded.uns["cluster_round_order"] == expected_order
    assert cu._next_round_index(loaded) == 4
    assert loaded.uns["active_cluster_round"] == "r3_subset_compacted"

    # Activation must not change historical order or per-round label columns.
    cu.set_active_round(loaded, "r0_parent")
    assert loaded.uns["cluster_round_order"] == expected_order
    assert loaded.uns["active_cluster_round"] == "r0_parent"
    pd.testing.assert_series_equal(loaded.obs["leiden"], loaded.obs["leiden__r0_parent"], check_names=False)
    pd.testing.assert_frame_equal(loaded.obs[prior_obs.columns], prior_obs)
    for rid, payload in payloads.items():
        assert loaded.uns["cluster_rounds"][rid] == payload

    second = tmp_path / f"refined.{fmt}"
    save_dataset(loaded, second, fmt=fmt, archive=archive)
    result = load_dataset(second.with_suffix(second.suffix + ".tar.zst") if archive else second)
    cu.set_active_round(result, "r3_subset_compacted")
    assert result.uns["cluster_round_order"] == expected_order
    assert result.uns["active_cluster_round"] == "r3_subset_compacted"
    pd.testing.assert_frame_equal(result.obs[prior_obs.columns], prior_obs)
    for rid, payload in payloads.items():
        assert result.uns["cluster_rounds"][rid] == payload
    for layer in ("counts_raw", "counts_cb"):
        np.testing.assert_array_equal(result.layers[layer], original.layers[layer])
    np.testing.assert_array_equal(result.X, original.X)
    np.testing.assert_array_equal(result.obsm["X_umap"], original.obsm["X_umap"])


def test_reregister_existing_round_does_not_duplicate_or_reorder():
    adata = _adata_with_rounds()
    adata.uns["cluster_round_order"] = np.array(["r0_parent", "r1_compacted"])
    _register(adata, "r0_parent")
    assert adata.uns["cluster_round_order"] == ["r0_parent", "r1_compacted"]


def test_shallow_round_creation_preserves_reloaded_history(tmp_path):
    path = tmp_path / "parent.zarr"
    save_dataset(_adata_with_rounds(), path, archive=False)
    loaded = load_dataset(path)
    new_round = cu._create_shallow_round_from_parent(
        loaded, parent_round_id="r1_compacted", round_name="reviewed",
        round_type="manual", kind="manual", set_active=False,
    )
    assert new_round == "r2_reviewed"
    assert loaded.uns["cluster_round_order"] == ["r0_parent", "r1_compacted", new_round]
    assert loaded.uns["active_cluster_round"] == "r1_compacted"


def test_report_respects_reloaded_explicit_round_order(tmp_path):
    adata = _adata_with_rounds()
    expected = ["r1_compacted", "r0_parent"]
    adata.uns["cluster_round_order"] = expected
    path = tmp_path / "parent.zarr"
    save_dataset(adata, path, archive=False)
    loaded = load_dataset(path)
    order_before = loaded.uns["cluster_round_order"].copy()
    run_dir = tmp_path / "report"
    run_dir.mkdir()
    generate_cluster_and_annotate_report(
        fig_root=tmp_path, fmt="png", run_dir=run_dir,
        cfg=ClusterAnnotateConfig(input_path=path), version="test", adata=loaded,
    )
    report = (run_dir / "cluster_and_annotate_report.html").read_text()
    assert report.index("Clustering state: r1_compacted") < report.index("Clustering state: r0_parent")
    np.testing.assert_array_equal(loaded.uns["cluster_round_order"], order_before)
