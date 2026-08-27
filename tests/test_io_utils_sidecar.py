from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scomnom.io_utils import _restore_legacy_dataframe_column, load_dataset, save_dataset


def _mixed_dataframe() -> pd.DataFrame:
    index = pd.Index(["r0", "r1", "r2"], dtype=str)
    return pd.DataFrame(
        {
            "label": pd.Series(["A", None, "C"], index=index, dtype="string"),
            "passed": pd.Series([True, False, True], index=index, dtype="bool"),
            "reviewed": pd.Series([True, pd.NA, False], index=index, dtype="boolean"),
            "count": pd.Series([1, pd.NA, 3], index=index, dtype="Int64"),
            "score": pd.Series([1.5, np.nan, -0.25], index=index, dtype="float64"),
            "note": pd.Series(["kept", None, "rejected"], index=index, dtype=object),
        },
        index=index,
    )


def _adata_with_uns_frame(key: str, frame: pd.DataFrame) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata.obs_names = ["cell0", "cell1", "cell2"]
    adata.var_names = ["gene0", "gene1"]
    adata.uns[key] = frame
    return adata


def test_legacy_boolean_string_restore_preserves_false_values() -> None:
    restored = _restore_legacy_dataframe_column(
        pd.Series(["True", "False", "1", "0"]),
        "bool",
    )
    assert restored.dtype == bool
    assert restored.tolist() == [True, False, True, False]


def test_save_and_load_dataset_with_uns_sidecar_roundtrip(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((3, 4), dtype=np.float32))
    adata.obs_names = [f"cell{i}" for i in range(3)]
    adata.var_names = [f"gene{i}" for i in range(4)]
    adata.uns["df_payload"] = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["row1", "row2"],
        columns=["C00", "C01"],
    )
    adata.uns["arr_payload"] = np.array([[1, 2], [3, 4]], dtype=np.int32)
    adata.uns["series_payload"] = pd.Series([0.1, 0.2], index=["a", "b"], name="s")

    out_path = tmp_path / "sidecar_test.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)

    assert (out_path / "__scomnom_payloads__" / "v2").exists()

    loaded = load_dataset(out_path)
    assert isinstance(loaded.uns["df_payload"], pd.DataFrame)
    assert isinstance(loaded.uns["arr_payload"], np.ndarray)
    assert isinstance(loaded.uns["series_payload"], pd.Series)
    assert loaded.uns["df_payload"].shape == (2, 2)
    assert loaded.uns["arr_payload"].shape == (2, 2)
    assert loaded.uns["series_payload"].shape == (2,)


def test_save_and_load_dataset_with_object_dataframe_sidecar(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.obs_names = ["cell0", "cell1"]
    adata.var_names = ["gene0", "gene1"]
    adata.uns["object_df"] = pd.DataFrame(
        {
            "label": ["A", "B"],
            "note": ["foo", None],
        },
        index=["r0", "r1"],
    )

    out_path = tmp_path / "sidecar_object_df.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)

    loaded = load_dataset(out_path)
    assert isinstance(loaded.uns["object_df"], pd.DataFrame)
    assert loaded.uns["object_df"].shape == (2, 2)


def test_mixed_dataframe_sidecar_preserves_values_dtypes_and_missingness(tmp_path: Path) -> None:
    expected = _mixed_dataframe()
    adata = _adata_with_uns_frame("mixed", expected)

    out_path = tmp_path / "mixed.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)

    assert (out_path / "__scomnom_payloads__" / "v2").exists()
    loaded = load_dataset(out_path)
    pd.testing.assert_frame_equal(loaded.uns["mixed"], expected)


def test_mixed_dataframe_roundtrip_in_archived_zarr(tmp_path: Path) -> None:
    expected = _mixed_dataframe()
    adata = _adata_with_uns_frame("mixed", expected)

    out_path = tmp_path / "mixed.zarr"
    archive_path = tmp_path / "mixed.zarr.tar.zst"
    save_dataset(adata, out_path, fmt="zarr", archive=True)

    assert archive_path.exists()
    loaded = load_dataset(archive_path)
    pd.testing.assert_frame_equal(loaded.uns["mixed"], expected)


def test_mixed_dataframe_roundtrip_in_h5ad(tmp_path: Path) -> None:
    expected = _mixed_dataframe()
    adata = _adata_with_uns_frame("mixed", expected)

    out_path = tmp_path / "mixed.h5ad"
    save_dataset(adata, out_path, fmt="h5ad", archive=False)

    loaded = load_dataset(out_path)
    pd.testing.assert_frame_equal(loaded.uns["mixed"], expected)


def test_compaction_edge_audit_booleans_roundtrip_with_mapping(tmp_path: Path) -> None:
    adata = _adata_with_uns_frame(
        "unused",
        pd.DataFrame({"value": [1]}, index=["row"]),
    )
    del adata.uns["unused"]
    pairwise = pd.DataFrame(
        {
            "cluster_a": ["C00", "C00", "C01"],
            "cluster_b": ["C01", "C02", "C02"],
            "progeny_pass": [True, False, True],
            "dorothea_pass": [True, False, False],
            "merge_pass": [True, False, False],
            "similarity": [0.97, 0.42, 0.71],
        }
    )
    adata.uns["cluster_rounds"] = {
        "r1_compacted": {
            "compaction": {
                "pairwise_evidence": pairwise,
                "reverse_map": {"C00": "C00+C01", "C01": "C00+C01", "C02": "C02"},
                "decision_log": ["C00+C01 merged", "C02 retained"],
            }
        }
    }

    out_path = tmp_path / "compaction.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)
    loaded = load_dataset(out_path)
    compaction = loaded.uns["cluster_rounds"]["r1_compacted"]["compaction"]

    pd.testing.assert_frame_equal(compaction["pairwise_evidence"], pairwise)
    assert compaction["pairwise_evidence"]["merge_pass"].tolist() == [True, False, False]
    assert compaction["reverse_map"] == {
        "C00": "C00+C01",
        "C01": "C00+C01",
        "C02": "C02",
    }
    assert list(compaction["decision_log"]) == ["C00+C01 merged", "C02 retained"]


def test_save_and_load_dataset_with_object_ndarray_sidecar(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.obs_names = ["cell0", "cell1"]
    adata.var_names = ["gene0", "gene1"]
    adata.uns["object_arr"] = np.array([["A", None], ["B", "C"]], dtype=object)

    out_path = tmp_path / "sidecar_object_arr.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)

    loaded = load_dataset(out_path)
    assert isinstance(loaded.uns["object_arr"], np.ndarray)
    assert loaded.uns["object_arr"].shape == (2, 2)


def test_save_dataset_zarr_coerces_object_columns_in_obs_var_obsm(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata.obs_names = ["cell0", "cell1", "cell2"]
    adata.var_names = ["gene0", "gene1"]
    adata.obs["mixed_obj"] = pd.Series(["A", 7, None], dtype=object)
    adata.var["mixed_obj"] = pd.Series(["x", None], index=adata.var_names, dtype=object)
    adata.obsm["meta_df"] = pd.DataFrame(
        {"label": pd.Series(["u", None, "v"], dtype=object)},
        index=adata.obs_names,
    )

    out_path = tmp_path / "sidecar_object_obs_var_obsm.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)

    loaded = load_dataset(out_path)
    assert "mixed_obj" in loaded.obs.columns
    assert "mixed_obj" in loaded.var.columns
    assert "meta_df" in loaded.obsm
