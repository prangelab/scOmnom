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
    _resolve_populations,
    _load_activity_resources,
    _score_populations,
    _prepare_sample_inputs,
)
from scomnom.config import SampleEnrichmentConfig
import scomnom.annotation_utils as au


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


def _add_rounds(adata):
    adata.obs["pretty_r1"] = adata.obs["population"].map({"0": "C00: Monocytes", "1": "C01: T cells"})
    adata.uns["active_cluster_round"] = "r1"
    adata.uns["cluster_rounds"] = {
        "r1": {"labels_obs_key": "population", "annotation": {"pretty_cluster_key": "pretty_r1"}},
        "old": {"labels_obs_key": "missing_old_labels", "cluster_key": "population"},
    }


@pytest.mark.parametrize("selector", ["0", "C00", "C00: Monocytes"])
def test_round_population_selection_preserves_ids_and_labels(sample_adata, selector):
    _add_rounds(sample_adata)
    selection = _resolve_populations(sample_adata, target_groups=(selector,))
    assert selection.round_id == "r1"
    assert selection.population_key == "population"
    assert selection.mapping["population_id"].tolist() == ["0", "1"]
    assert selection.selected_ids == ("0",)
    assert selection.mapping.iloc[0]["population_label"] == "C00: Monocytes"


@pytest.mark.parametrize("round_id", ["unknown", "old"])
def test_round_resolution_never_uses_another_round(sample_adata, round_id):
    _add_rounds(sample_adata)
    with pytest.raises(ValueError, match="round|Round"):
        _resolve_populations(sample_adata, round_id=round_id)


def test_unavailable_and_ambiguous_population_selectors_fail(sample_adata):
    _add_rounds(sample_adata)
    with pytest.raises(ValueError, match="population"):
        _resolve_populations(sample_adata, target_groups=("C99",))
    sample_adata.obs["pretty_r1"] = "C00: Shared"
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_populations(sample_adata, target_groups=("C00",))


def test_inconsistent_display_labels_fail(sample_adata):
    _add_rounds(sample_adata)
    sample_adata.obs.loc["cell0", "pretty_r1"] = "C09: Other"
    with pytest.raises(ValueError, match="multiple display"):
        _resolve_populations(sample_adata)


def test_cells_outside_selected_round_are_excluded(sample_adata):
    _add_rounds(sample_adata)
    sample_adata.obs.loc["cell0", "population"] = None
    selected = _resolve_populations(sample_adata)
    assert selected.round_cell_mask.sum() == 5


def test_sample_config_defaults_and_method_inheritance():
    cfg = SampleEnrichmentConfig(input_path="input.zarr", replicate_key="sample", decoupler_method="ulm")
    assert cfg.min_cells_per_replicate_group == 20
    assert cfg.min_replicates_per_level == 3
    assert cfg.min_replicates_total == 6
    assert cfg.min_complete_subjects == 3
    assert cfg.msigdb_gene_sets == ["HALLMARK", "REACTOME"]
    assert cfg.dorothea_confidence == ["A", "B", "C"]
    assert cfg.dorothea_method is None
    assert cfg.decoupler_method == "ulm"


def test_sample_config_parses_repeated_csv_fields():
    cfg = SampleEnrichmentConfig(
        input_path="input.zarr", replicate_key="sample", condition_key="condition",
        contrasts=["stim:ctrl,other:ctrl"], covariates=["donor,age", "age"],
        subject_key="donor", target_groups=["C00,C01"], plot_activity=["STAT2,IRF9"],
    )
    assert cfg.contrasts == ("stim:ctrl", "other:ctrl")
    assert cfg.covariates == ("donor", "age")
    assert cfg.plot_activity == ("STAT2", "IRF9")


@pytest.mark.parametrize("kwargs", [
    {"subject_key": "donor"}, {"contrasts": ["stim:ctrl"]},
    {"condition_key": "condition", "contrasts": ["bad"]},
    {"condition_key": "condition", "contrasts": ["ctrl:ctrl"]},
    {"min_replicates_per_level": 0}, {"min_complete_subjects": True},
    {"counts_layer": "raw"}, {"dorothea_confidence": ["Z"]},
    {"decoupler_consensus_methods": ["ulm"]},
])
def test_sample_config_rejects_invalid_designs(kwargs):
    with pytest.raises(ValueError):
        SampleEnrichmentConfig(input_path="input.zarr", replicate_key="sample", **kwargs)


def _fake_resources(monkeypatch):
    net = pd.DataFrame({
        "source": ["TF1", "TF1", "TF2"], "target": ["G1", "G2", "absent"], "weight": [1., -1., 1.],
    })
    monkeypatch.setattr(au, "_load_dorothea_net", lambda *args, **kwargs: net.copy())
    return net


def test_scoring_is_population_local_and_audits_target_overlap(sample_adata, monkeypatch):
    _add_rounds(sample_adata)
    _fake_resources(monkeypatch)
    cfg = SampleEnrichmentConfig(
        input_path="input.zarr", replicate_key="sample", run_msigdb=False, run_progeny=False,
        dorothea_method="ulm", decoupler_min_n_targets=2,
    )
    resources = _load_activity_resources(cfg)
    assert resources[0].method == "ulm"
    assert len(resources[0].provenance["network_sha256"]) == 64
    prepared = _prepare_pseudobulks(
        sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1,
    )
    calls = []

    def score(**kwargs):
        calls.append(kwargs["mat"].copy())
        result = pd.DataFrame([np.arange(kwargs["mat"].shape[1], dtype=float)], index=["TF1"], columns=kwargs["mat"].columns)
        result.attrs["method_provenance"] = {"requested_method": "ulm", "successful_constituents": ["ulm"]}
        return result

    monkeypatch.setattr(au, "_dc_run_method", score)
    result = _score_populations(prepared, _resolve_populations(sample_adata), resources, cfg)
    assert len(calls) == 2
    assert all(matrix.shape == (2, 2) for matrix in calls)
    assert set(result.scores["population_id"]) == {"0", "1"}
    assert len(result.scores) == 4
    excluded = result.audit.loc[result.audit["activity"] == "TF2"]
    assert set(excluded["status"]) == {"insufficient_target_overlap"}
    assert set(excluded["target_overlap"]) == {0}
    assert all("ulm" in value for value in result.audit["method_provenance"])
    assert "dorothea" not in sample_adata.uns


def test_failed_requested_resource_is_fatal(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("resource unavailable")
    monkeypatch.setattr(au, "_load_dorothea_net", fail)
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample", run_msigdb=False, run_progeny=False)
    with pytest.raises(RuntimeError, match="resource unavailable"):
        _load_activity_resources(cfg)


def test_failed_scoring_is_fatal(sample_adata, monkeypatch):
    _add_rounds(sample_adata)
    _fake_resources(monkeypatch)
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample", run_msigdb=False, run_progeny=False, decoupler_min_n_targets=2)
    resources = _load_activity_resources(cfg)
    prepared = _prepare_pseudobulks(sample_adata, replicate_key="sample", population_key="population", min_cells_per_replicate_group=1)
    def fail(**kwargs):
        raise RuntimeError("method failed")
    monkeypatch.setattr(au, "_dc_run_method", fail)
    with pytest.raises(RuntimeError, match="method failed"):
        _score_populations(prepared, _resolve_populations(sample_adata), resources, cfg)


def test_one_failed_msigdb_collection_cannot_be_silently_omitted(monkeypatch):
    monkeypatch.setattr(au, "_resolve_msigdb_gene_sets_cached", lambda spec: (["ok.gmt", "bad.gmt"], list(spec), "test-release"))
    monkeypatch.setattr(au, "_load_msigdb_decoupler_net_cached", lambda paths: None if paths == ["bad.gmt"] else pd.DataFrame({"source": ["P"], "target": ["G1"]}))
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample", run_dorothea=False, run_progeny=False)
    with pytest.raises(RuntimeError, match="bad.gmt"):
        _load_activity_resources(cfg)


def test_unresolved_msigdb_request_is_fatal(monkeypatch):
    monkeypatch.setattr(au, "_resolve_msigdb_gene_sets_cached", lambda spec: (["ok.gmt"], ["HALLMARK"], "test"))
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample", run_dorothea=False, run_progeny=False)
    with pytest.raises(RuntimeError, match="REACTOME"):
        _load_activity_resources(cfg)


def test_requested_gene_filter_preserves_library_size(sample_adata):
    _add_rounds(sample_adata)
    cfg = SampleEnrichmentConfig(
        input_path="x", replicate_key="sample", min_cells_per_replicate_group=1,
        gene_filter=("gene != 'G2'",), target_groups=("C00",),
    )
    selection, prepared = _prepare_sample_inputs(sample_adata, cfg)
    assert selection.selected_ids == ("0",)
    assert set(prepared.qc["population_id"]) == {"0", "1"}
    assert prepared.expression.columns.tolist() == ["G1"]
    row = prepared.qc.index[(prepared.qc["replicate_id"] == "s1") & (prepared.qc["population_id"] == "0")][0]
    assert prepared.qc.loc[row, "library_size"] == 6
    assert prepared.expression.loc[row, "G1"] == pytest.approx(np.log1p(500000))


def test_installed_ulm_scoring_and_population_selection_invariance(monkeypatch):
    rng = np.random.default_rng(173)
    genes = [f"G{i}" for i in range(12)]
    obs = pd.DataFrame({
        "sample": [f"s{i}" for i in range(6)] * 2,
        "population": ["0"] * 6 + ["1"] * 6,
    }, index=[f"cell{i}" for i in range(12)])
    obj = ad.AnnData(X=sparse.csr_matrix(rng.poisson(30, size=(12, 12))), obs=obs, var=pd.DataFrame(index=genes))
    _add_rounds(obj)
    net = pd.DataFrame({"source": np.repeat(["TF1", "TF2", "TF3"], 4), "target": genes, "weight": [1., -1., .5, 1.] * 3})
    monkeypatch.setattr(au, "_load_dorothea_net", lambda *args, **kwargs: net.copy())
    cfg = SampleEnrichmentConfig(
        input_path="x", replicate_key="sample", min_cells_per_replicate_group=1,
        run_msigdb=False, run_progeny=False, dorothea_method="ulm", decoupler_min_n_targets=3,
    )
    selection, prepared = _prepare_sample_inputs(obj, cfg)
    resources = _load_activity_resources(cfg)
    full = _score_populations(prepared, selection, resources, cfg)
    assert len(full.scores) == 36
    assert np.isfinite(full.scores["score"]).all()
    assert set(full.audit["status"]) == {"ok"}
    restricted_cfg = cfg.model_copy(update={"target_groups": ("C00",), "plot_activity": ("TF1",)})
    restricted_selection, restricted_prepared = _prepare_sample_inputs(obj, restricted_cfg)
    restricted = _score_populations(restricted_prepared, restricted_selection, resources, restricted_cfg)
    expected = full.scores.loc[full.scores["population_id"] == "0"].reset_index(drop=True)
    pd.testing.assert_frame_equal(restricted.scores, expected)
