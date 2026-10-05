import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scomnom import io_utils


@pytest.mark.parametrize("helper", [True, False])
@pytest.mark.parametrize("all_empty", [True, False])
@pytest.mark.parametrize("storage", [None, "zarr", "h5ad", "archive"])
def test_marker_exports_omit_padding(tmp_path, monkeypatch, helper, all_empty, storage):
    names = [None, np.nan, ""] if all_empty else ["STAT2", None, np.nan, ""]
    scores = [np.nan] * len(names)
    if not all_empty:
        scores[0] = 3.25
    adata = ad.AnnData(var=pd.DataFrame({"gene_type": ["protein_coding"]}, index=["STAT2"]))
    adata.uns["markers"] = {
        "names": np.rec.fromarrays([names], names=["0"], formats=[object]),
        "scores": np.rec.fromarrays([scores], names=["0"]),
    }
    if storage is not None:
        fmt = "h5ad" if storage == "h5ad" else "zarr"
        path = tmp_path / f"markers.{fmt}"
        io_utils.save_dataset(adata, path, fmt=fmt, archive=storage == "archive")
        if storage == "archive":
            path = path.with_name(path.name + ".tar.zst")
        adata = io_utils.load_dataset(path)
        assert adata.uns["markers"]["names"].dtype.names == ("0",)

    def dataframe(*args, **kwargs):
        if not helper:
            raise KeyError("No Scanpy params in stored payload")
        return pd.DataFrame({"names": names, "scores": scores})

    monkeypatch.setattr("scanpy.get.rank_genes_groups_df", dataframe)
    io_utils.export_rank_genes_groups_tables(adata, key_added="markers", output_dir=tmp_path)
    io_utils.export_rank_genes_groups_excel(adata, key_added="markers", output_dir=tmp_path)
    csv = pd.read_csv(tmp_path / "marker_tables/celllevel_markers__all_groups.csv")
    sheets = pd.read_excel(tmp_path / "marker_tables/celllevel_markers.xlsx", sheet_name=None)
    assert len(csv) == (0 if all_empty else 1)
    assert sum(len(sheet) for sheet in sheets.values()) == len(csv)
    if not all_empty:
        assert csv["gene"].tolist() == ["STAT2"]
        column = "scores" if helper else "score"
        assert csv[column].tolist() == [3.25]
    stored = adata.uns["markers"]["names"]["0"]
    assert len(stored) == len(names)
    assert pd.isna(stored[0 if all_empty else 1])
    assert stored[-1] == ""
