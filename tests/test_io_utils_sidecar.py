from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scomnom.io_utils import load_dataset, save_dataset


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

    assert (out_path / "__scomnom_payloads__" / "v1").exists()

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
