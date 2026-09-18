from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scomnom.sample_enrichment import (
    _prepare_pseudobulks,
    _resolve_count_assay,
    _resolve_replicate_metadata,
)


@pytest.fixture
def sample_adata():
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "population": ["0", "0", "1", "0", "0", "1"],
            "condition": ["control"] * 3 + ["treated"] * 3,
            "age": [40.0] * 3 + [60.0] * 3,
            "batch": pd.Categorical(["A"] * 3 + ["B"] * 3),
        },
        index=[f"cell{i}" for i in range(6)],
    )
    counts = sparse.csr_matrix(
        [[1, 3, 0], [2, 0, 0], [4, 1, 0], [0, 2, 0], [1, 1, 0], [2, 3, 0]],
        dtype=np.int64,
    )
    obj = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=["G1", "G2", "G0"]))
    obj.layers["counts_cb"] = counts.copy()
    obj.layers["counts_raw"] = counts * 2
    return obj


def test_assay_priority_and_explicit_selection(sample_adata):
    layer, provenance = _resolve_count_assay(sample_adata)
    assert layer == "counts_cb"
    assert provenance["source"] == "layers/counts_cb"
    assert provenance["integer_like"] is True
    assert provenance["sparse"] is True
    assert provenance["shape"] == [6, 3]
    assert provenance["source_priority"] == ["counts_cb", "counts_raw", "X"]
    assert _resolve_count_assay(sample_adata, "counts_raw")[0] == "counts_raw"
    del sample_adata.layers["counts_cb"]
    assert _resolve_count_assay(sample_adata)[0] == "counts_raw"
    del sample_adata.layers["counts_raw"]
    assert _resolve_count_assay(sample_adata)[0] is None
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_count_assay(sample_adata, "counts_cb")


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 0.25])
@pytest.mark.parametrize("source", ["X", "counts_cb"])
def test_invalid_assay_fails_without_fallback(sample_adata, value, source):
    matrix = sample_adata.X.astype(float)
    matrix.data[0] = value
    if source == "X":
        sample_adata.X = matrix
    else:
        sample_adata.layers[source] = matrix
    with pytest.raises(ValueError, match="counts"):
        _resolve_count_assay(sample_adata, "auto" if source == "counts_cb" else source)


def test_raw_is_not_an_automatic_count_source(sample_adata):
    sample_adata.raw = sample_adata.copy()
    del sample_adata.layers["counts_cb"]
    del sample_adata.layers["counts_raw"]
    sample_adata.X = sample_adata.X.astype(float) * 0.25
    with pytest.raises(ValueError, match="integer"):
        _resolve_count_assay(sample_adata)


def test_replicate_metadata_preserves_numeric_and_categorical_types(sample_adata):
    meta = _resolve_replicate_metadata(
        sample_adata.obs, replicate_key="sample", condition_key="condition", covariates=("age", "batch")
    )
    assert meta.index.tolist() == ["s1", "s2"]
    assert meta["age"].tolist() == [40.0, 60.0]
    assert pd.api.types.is_numeric_dtype(meta["age"])
    assert isinstance(meta["batch"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("column,value", [("condition", "treated"), ("age", 99.0)])
def test_ambiguous_metadata_is_fatal(sample_adata, column, value):
    sample_adata.obs.loc["cell0", column] = value
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_replicate_metadata(
            sample_adata.obs, replicate_key="sample", condition_key="condition", covariates=("age",)
        )


def test_missing_covariate_is_retained_as_missing(sample_adata):
    sample_adata.obs.loc["cell0", "age"] = np.nan
    meta = _resolve_replicate_metadata(
        sample_adata.obs, replicate_key="sample", condition_key="condition", covariates=("age",)
    )
    assert pd.isna(meta.loc["s1", "age"])
    assert meta.loc["s2", "age"] == 60.0


@pytest.mark.parametrize("column", ["sample", "condition"])
def test_missing_required_metadata_is_fatal(sample_adata, column):
    sample_adata.obs.loc["cell0", column] = None
    with pytest.raises(ValueError, match="missing"):
        _resolve_replicate_metadata(sample_adata.obs, replicate_key="sample", condition_key="condition")


def test_unknown_covariate_is_fatal(sample_adata):
    with pytest.raises(ValueError, match="unknown"):
        _resolve_replicate_metadata(
            sample_adata.obs, replicate_key="sample", condition_key="condition", covariates=("unknown",)
        )


def test_pseudobulk_sums_qc_and_hand_calculated_log_cpm(sample_adata):
    before = sample_adata.layers["counts_cb"].copy()
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population",
        condition_key="condition", covariates=("age",), min_cells_per_replicate_group=2,
    )
    assert result.counts.sparse.to_coo().sum() == before.sum()
    assert len(result.qc) == 4
    assert result.qc["included"].sum() == 2
    excluded = result.qc.loc[~result.qc["included"]]
    assert set(excluded["exclusion_reason"]) == {"insufficient_cells"}
    row_id = result.qc.index[(result.qc["replicate_id"] == "s1") & (result.qc["population_id"] == "0")][0]
    np.testing.assert_allclose(result.expression.loc[row_id].to_numpy(), np.log1p([500000, 500000]))
    assert result.qc.loc[row_id, "library_size"] == 6
    assert result.qc.loc[row_id, "detected_genes"] == 2
    assert result.expression.columns.tolist() == ["G1", "G2"]
    assert result.provenance["gene_filter"]["input_genes"] == 3
    assert result.provenance["gene_filter"]["retained_genes"] == 2
    assert (before != sample_adata.layers["counts_cb"]).nnz == 0


def test_zero_libraries_are_retained_in_qc(sample_adata):
    sample_adata.layers["counts_cb"] = sparse.csr_matrix(sample_adata.shape, dtype=np.int64)
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    assert not result.qc["included"].any()
    assert set(result.qc["exclusion_reason"]) == {"zero_library"}
    assert result.expression.shape == (0, 0)


def test_filter_uses_all_eligible_populations(sample_adata):
    matrix = sample_adata.layers["counts_cb"].toarray()
    matrix[2, 2] = 7
    sample_adata.layers["counts_cb"] = sparse.csr_matrix(matrix)
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    assert result.expression.columns.tolist() == ["G1", "G2", "G0"]
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=2,
    )
    assert result.expression.columns.tolist() == ["G1", "G2"]


def test_arbitrary_identifiers_cannot_collide_in_aggregation(sample_adata):
    sample_adata.obs["sample"] = ["a||b"] * 3 + ["a"] * 3
    sample_adata.obs["population"] = ["c", "c", "b||c", "b||c", "b||c", "c"]
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    assert len(result.qc) == 4
    assert set(result.qc["replicate_id"]) == {"a||b", "a"}
    assert set(result.qc["population_id"]) == {"b||c", "c"}


def test_small_integer_dtype_does_not_overflow(sample_adata):
    sample_adata.layers["counts_cb"] = sparse.csr_matrix(np.full(sample_adata.shape, 100, dtype=np.int8))
    result = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    assert result.counts.sparse.to_coo().sum() == 1800
    assert result.qc["library_size"].max() == 600


def test_default_cell_threshold_boundary_and_dense_counts():
    obs = pd.DataFrame(
        {"sample": ["below"] * 19 + ["at"] * 20, "population": ["0"] * 39},
        index=[f"cell{i}" for i in range(39)],
    )
    obj = ad.AnnData(X=np.ones((39, 2), dtype=np.int64), obs=obs)
    result = _prepare_pseudobulks(obj, replicate_key="sample", population_key="population")
    qc = result.qc.set_index("replicate_id")
    assert not qc.loc["below", "included"]
    assert qc.loc["at", "included"]
    assert result.provenance["count_assay"]["source"] == "X"
    assert result.provenance["count_assay"]["sparse"] is False
    np.testing.assert_allclose(result.expression.to_numpy(), np.log1p([[500000, 500000]]))


def test_population_metadata_and_source_are_not_modified(sample_adata):
    original_obs = sample_adata.obs.copy(deep=True)
    original_var = sample_adata.var.copy(deep=True)
    original_x = sample_adata.X.copy()
    _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    pd.testing.assert_frame_equal(sample_adata.obs, original_obs)
    pd.testing.assert_frame_equal(sample_adata.var, original_var)
    assert (sample_adata.X != original_x).nnz == 0


def test_missing_population_is_fatal(sample_adata):
    sample_adata.obs.loc["cell0", "population"] = None
    with pytest.raises(ValueError, match="missing"):
        _prepare_pseudobulks(sample_adata, replicate_key="sample", population_key="population")


def test_duplicate_gene_identifiers_are_fatal(sample_adata):
    sample_adata.var_names = ["G1", "G1", "G0"]
    with pytest.raises(ValueError, match="unique"):
        _prepare_pseudobulks(sample_adata, replicate_key="sample", population_key="population")


def test_string_conversion_cannot_merge_replicates(sample_adata):
    sample_adata.obs["sample"] = [1, 1, 1, "1", "1", "1"]
    with pytest.raises(ValueError, match="collide"):
        _prepare_pseudobulks(sample_adata, replicate_key="sample", population_key="population")


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True])
def test_invalid_cell_threshold_fails(sample_adata, threshold):
    with pytest.raises(ValueError, match="positive integer"):
        _prepare_pseudobulks(
            sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=threshold,
        )
