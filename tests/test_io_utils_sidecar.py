from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scomnom.io_utils import (
    _decode_structured_array,
    _encode_structured_array,
    _restore_legacy_dataframe_column,
    load_dataset,
    save_dataset,
)


def test_structured_object_scalar_types_and_corrupt_payloads() -> None:
    expected = np.empty(10, dtype=[("value", object)])
    expected["value"] = [True, 2**80, -0.0, float("inf"), "None", "nan",
                         b"\x00\xff", np.int16(-12), np.datetime64("NaT"), None]
    payload = _encode_structured_array(expected)
    actual = _decode_structured_array(payload)
    for before, after in zip(expected["value"], actual["value"]):
        assert type(before) is type(after)
        if isinstance(before, np.datetime64):
            assert np.isnat(after)
        else:
            assert before == after
    assert np.signbit(actual["value"][2])
    payload["fields"][0]["values"][0] = ["unsupported"]
    with pytest.raises(ValueError, match="Unknown structured-array object encoding"):
        _decode_structured_array(payload)


def test_legacy_sidecar_arrays_remain_readable_without_guessing(tmp_path: Path) -> None:
    import zarr

    adata = ad.AnnData(X=np.eye(2, dtype=np.float32))
    scores = np.array([(1.0,), (np.nan,)], dtype=[("0", "f8")])
    adata.uns["scores"] = scores
    path = tmp_path / "legacy.zarr"
    save_dataset(adata, path, fmt="zarr", archive=False)
    root = zarr.open_group(str(path), mode="a", zarr_format=2, use_consolidated=False)
    payloads = root["__scomnom_payloads__"]["v2"]
    node = next(payloads[name] for name in payloads.group_keys() if name != "__manifest__")
    del node["payload_json"]
    del node.attrs["storage_layout"]
    node.create_array("data", data=scores)
    np.testing.assert_array_equal(load_dataset(path).uns["scores"]["0"], scores["0"])
    del node["data"]
    old_names = np.array(["('STAT2',)", "(None,)"])
    node.create_array("data", data=old_names)
    np.testing.assert_array_equal(load_dataset(path).uns["scores"], old_names)


@pytest.mark.parametrize("fmt,archive", [("zarr", False), ("zarr", True), ("h5ad", False)])
def test_marker_structured_arrays_roundtrip(tmp_path: Path, fmt: str, archive: bool) -> None:
    names = np.array(
        [("STAT2", "HOTAIRM1"), (None, np.nan), ("", pd.NA)],
        dtype=[("0", object), ("T/B cells", object)],
    ).view(np.recarray)
    scores = np.array([(3.25, 2.5), (np.nan, np.nan), (-0.0, np.inf)],
                      dtype=[("0", "<f8"), ("T/B cells", ">f8")])
    adata = ad.AnnData(X=np.eye(2, dtype=np.float32))
    adata.uns["rank_genes_groups"] = {"names": names, "scores": scores}
    path = tmp_path / f"markers.{fmt}"
    save_dataset(adata, path, fmt=fmt, archive=archive)
    loaded = load_dataset(Path(str(path) + ".tar.zst") if archive else path)
    result = loaded.uns["rank_genes_groups"]
    assert result["names"].dtype == names.dtype
    assert result["names"].shape == names.shape
    assert result["names"].dtype.names == ("0", "T/B cells")
    assert result["names"]["0"].tolist() == ["STAT2", None, ""]
    assert result["names"]["T/B cells"][0] == "HOTAIRM1"
    assert np.isnan(result["names"]["T/B cells"][1])
    assert result["names"]["T/B cells"][2] is pd.NA
    assert result["scores"].dtype == scores.dtype
    for field in scores.dtype.names:
        np.testing.assert_array_equal(result["scores"][field], scores[field])
    np.testing.assert_array_equal(loaded.X, adata.X)
    assert adata.uns["rank_genes_groups"]["names"] is names
    assert names["0"][1] is None


@pytest.mark.parametrize("fmt", ["zarr", "h5ad"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_structured_layout_and_object_scalars_roundtrip(tmp_path: Path, fmt: str, shape: tuple) -> None:
    dtype = np.dtype([(("label title", "label"), object),
                      ("nested", [("gene", object), ("value", ">f4")]),
                      ("vector", "<i2", (2,))], align=True)
    expected = np.empty(shape, dtype=dtype)
    expected["label"] = b"byte label"
    expected["nested"]["gene"] = np.float32(np.nan)
    expected["nested"]["value"] = 1.25
    expected["vector"] = [1, 2]
    adata = ad.AnnData(X=np.eye(2, dtype=np.float32))
    adata.uns["records"] = expected
    path = tmp_path / f"records.{fmt}"
    save_dataset(adata, path, fmt=fmt, archive=False)
    actual = load_dataset(path).uns["records"]
    assert actual.dtype == expected.dtype
    assert actual.dtype.isalignedstruct
    assert actual.shape == shape
    np.testing.assert_array_equal(actual["label"], expected["label"])
    np.testing.assert_array_equal(actual["nested"]["value"], expected["nested"]["value"])
    np.testing.assert_array_equal(actual["vector"], expected["vector"])
    for value in actual["nested"]["gene"].flat:
        assert type(value) is np.float32
        assert np.isnan(value)


@pytest.mark.parametrize("fmt", ["zarr", "h5ad"])
def test_structured_object_values_fail_closed(tmp_path: Path, fmt: str) -> None:
    adata = ad.AnnData(X=np.eye(2, dtype=np.float32))
    records = np.empty(1, dtype=[("gene", object)])
    records["gene"][0] = {"not": "a supported scalar"}
    adata.uns["records"] = records
    with pytest.raises((TypeError, RuntimeError)) as exc:
        save_dataset(adata, tmp_path / f"invalid.{fmt}", fmt=fmt, archive=False)
    cause = exc.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert "Unsupported structured-array object value: dict" in str(cause)
    assert adata.uns["records"] is records


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


def test_compaction_ragged_sequences_roundtrip(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata.obs_names = ["cell0", "cell1", "cell2"]
    adata.var_names = ["gene0", "gene1"]
    adata.uns["cluster_rounds"] = {
        "r1_compacted": {
            "compacting": {
                "components": [["2", "5"], ["7"], ["3", "4", "8"]],
                "reverse_map": {
                    "C00": ["2", "5"],
                    "C01": ["7"],
                    "C02": ["3", "4", "8"],
                },
                "pairwise": {
                    "adjacency": {
                        "T cells": [("2", "5")],
                        "B cells": [],
                    }
                },
                "decision_log": [
                    {
                        "celltypist_label": "T cells",
                        "members": ["2", "5"],
                        "n_members": 2,
                    },
                    {
                        "celltypist_label": "Myeloid",
                        "members": ["3", "4", "8"],
                        "n_members": 3,
                    },
                ],
            }
        }
    }

    out_path = tmp_path / "compaction_ragged.zarr"
    save_dataset(adata, out_path, fmt="zarr", archive=False)
    loaded = load_dataset(out_path)
    compacting = loaded.uns["cluster_rounds"]["r1_compacted"]["compacting"]

    assert compacting["components"] == [["2", "5"], ["7"], ["3", "4", "8"]]
    reverse_map = {
        key: list(map(str, members))
        for key, members in compacting["reverse_map"].items()
    }
    assert reverse_map == {
        "C00": ["2", "5"],
        "C01": ["7"],
        "C02": ["3", "4", "8"],
    }
    adjacency = {
        key: [tuple(map(str, pair)) for pair in pairs]
        for key, pairs in compacting["pairwise"]["adjacency"].items()
    }
    assert adjacency == {
        "T cells": [("2", "5")],
        "B cells": [],
    }
    assert compacting["decision_log"][1]["members"] == ["3", "4", "8"]


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
