"""Feature identities must survive sample-specific gene filtering and padding."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scomnom import io_utils as io


def sample(genes, values, **columns):
    return ad.AnnData(
        X=sparse.csr_matrix([values], dtype=np.int32),
        obs=pd.DataFrame({"condition": ["ctrl"]}, index=["cell"]),
        var=pd.DataFrame(columns, index=genes),
    )


@pytest.mark.parametrize("layer", ["counts_raw", "counts_cb"])
def test_union_merge_preserves_identity_counts_and_roundtrip(tmp_path, monkeypatch, layer):
    monkeypatch.chdir(tmp_path)
    first = sample(["B", "A"], [3, 2], gene_ids=["idB", "idA"],
                   feature_types=["Gene Expression"] * 2, genome=["GRCh38"] * 2,
                   total_counts=[3, 2])
    second = sample(["C", "B"], [9, 7], gene_ids=["idC", "idB"],
                    feature_types=["Gene Expression"] * 2, genome=["GRCh38"] * 2,
                    total_counts=[9, 7])
    original_vars = [first.var.copy(deep=True), second.var.copy(deep=True)]
    merged = io.merge_samples({"s1": first, "s2": second}, "sample_id", layer)
    assert list(merged.var_names) == ["A", "B", "C"]
    assert merged.var["gene_ids"].tolist() == ["idA", "idB", "idC"]
    assert merged.var["feature_types"].tolist() == ["Gene Expression"] * 3
    assert merged.var["genome"].tolist() == ["GRCh38"] * 3
    assert merged.var["total_counts"].tolist() == [2, 3, 0]
    expected = np.array([[2, 3, 0], [0, 7, 9]], dtype=np.int32)
    np.testing.assert_array_equal(merged.X.toarray(), expected)
    np.testing.assert_array_equal(merged.layers[layer].toarray(), expected)
    assert merged.X.dtype == np.int32
    assert merged.obs_names.tolist() == ["s1_cell", "s2_cell"]
    assert merged.obs["sample_id"].tolist() == ["s1", "s2"]
    for original, source in zip(original_vars, (first, second)):
        pd.testing.assert_frame_equal(original, source.var)
    np.testing.assert_array_equal(first.X.toarray(), [[3, 2]])
    np.testing.assert_array_equal(second.X.toarray(), [[9, 7]])
    io.save_dataset(merged, tmp_path / "merged.zarr")
    reloaded = io.load_dataset(tmp_path / "merged.zarr.tar.zst")
    identity_types = {column: str for column in ("gene_ids", "feature_types", "genome")}
    pd.testing.assert_frame_equal(merged.var.astype(identity_types), reloaded.var.astype(identity_types))
    np.testing.assert_array_equal(reloaded.layers[layer].toarray(), expected)
    assert not (tmp_path / "tmp_merge").exists()


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "  "])
def test_missing_id_is_filled_from_other_sample(tmp_path, monkeypatch, missing):
    monkeypatch.chdir(tmp_path)
    first = sample(["A"], [2], gene_ids=[missing])
    second = sample(["A"], [4], gene_ids=pd.Categorical(["idA"]))
    merged = io.merge_samples({"s1": first, "s2": second}, "sample_id", "counts_raw")
    assert merged.var["gene_ids"].tolist() == ["idA"]


def test_missing_column_and_identity_not_invented(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = sample(["A", "C"], [2, 3])
    second = sample(["A", "B"], [4, 5], gene_ids=["idA", "idB"])
    merged = io.merge_samples({"s1": first, "s2": second}, "sample_id", "counts_raw")
    assert merged.var["gene_ids"].tolist() == ["idA", "idB", ""]
    assert "genome" not in merged.var and "feature_types" not in merged.var


@pytest.mark.parametrize("column", ["gene_ids", "feature_types", "genome"])
def test_conflicting_identity_fails_before_writing(tmp_path, monkeypatch, column):
    monkeypatch.chdir(tmp_path)
    first = sample(["A"], [2], **{column: ["first"]})
    second = sample(["A"], [4], **{column: ["second"]})
    with pytest.raises(ValueError, match=f"Conflicting {column}.*A.*first.*second.*s2"):
        io.merge_samples({"s1": first, "s2": second}, "sample_id", "counts_raw")
    assert not (tmp_path / "tmp_merge").exists()


def test_feature_identity_independent_of_first_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = sample(["A"], [2], gene_ids=["idA"])
    second = sample(["B"], [4], gene_ids=["idB"])
    forward = io.merge_samples({"s1": first, "s2": second}, "sample_id", "counts_raw")
    reverse = io.merge_samples({"s2": first, "s1": second}, "sample_id", "counts_raw")
    pd.testing.assert_series_equal(forward.var["gene_ids"], reverse.var["gene_ids"])


def test_absent_identity_columns_remain_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    merged = io.merge_samples({"s1": sample(["A"], [2]), "s2": sample(["B"], [4])},
                              "sample_id", "counts_raw")
    assert not {"gene_ids", "feature_types", "genome"}.intersection(merged.var.columns)
    np.testing.assert_array_equal(merged.X.toarray(), [[2, 0], [0, 4]])
