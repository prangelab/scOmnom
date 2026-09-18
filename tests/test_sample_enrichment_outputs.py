from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scomnom.annotation_utils as au
import scomnom.sample_enrichment as se
from scomnom import io_utils
from scomnom.config import SampleEnrichmentConfig


@pytest.fixture
def output_case(tmp_path, monkeypatch):
    rng = np.random.default_rng(801)
    genes = [f"G{i}" for i in range(12)]
    obs = pd.DataFrame({
        "sample": [f"s{i}" for i in range(12)], "condition": ["ctrl"] * 6 + ["stim"] * 6,
        "round_labels": ["3"] * 12, "pretty": ["C03: Macrophages"] * 12,
        "age": [np.nan] + list(range(31, 42)),
    }, index=[f"cell{i}" for i in range(12)])
    obj = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(30, size=(12, 12))), obs=obs,
        var=pd.DataFrame(index=genes),
    )
    obj.layers["counts_cb"] = obj.X.copy()
    obj.uns["active_cluster_round"] = "r1"
    obj.uns["cluster_rounds"] = {"r1": {
        "labels_obs_key": "round_labels", "annotation": {"pretty_cluster_key": "pretty"},
        "decoupler": {"sentinel": "existing cluster result"},
    }}
    for key in ("dorothea", "msigdb", "progeny", "pseudobulk"):
        obj.uns[key] = {"sentinel": "existing top-level result"}
    net = pd.DataFrame({"source": np.repeat(["TF1", "TF2", "TF3"], 4), "target": genes, "weight": [1., -.5, .5, 1.] * 3})
    monkeypatch.setattr(au, "_load_dorothea_net", lambda *args, **kwargs: net.copy())
    input_path = tmp_path / "results" / "nested" / "input.zarr"
    input_path.parent.mkdir(parents=True)
    io_utils.save_dataset(obj, input_path, archive=False)
    cfg = SampleEnrichmentConfig(
        input_path=input_path, replicate_key="sample", condition_key="condition",
        contrasts=("stim:ctrl",), covariates=("age",), min_cells_per_replicate_group=1,
        run_msigdb=False, run_progeny=False, dorothea_method="ulm", decoupler_min_n_targets=3,
    )
    return obj, cfg


def test_default_output_routing_and_name(output_case):
    _, cfg = output_case
    root = cfg.input_path.parent.parent
    assert se._sample_output_dir(cfg) == root
    assert se._sample_output_stem(cfg, "r1") == "adata.enrichment_sample_r1"
    assert se._sample_output_stem(cfg.model_copy(update={"output_name": "custom.zarr.tar.zst"}), "r1") == "custom"
    explicit = cfg.model_copy(update={"output_dir": root / "elsewhere"})
    assert se._sample_output_dir(explicit) == root / "elsewhere"


@pytest.mark.parametrize("fmt", ["zarr", "h5ad"])
def test_round_payload_roundtrips_and_preserves_existing_results(output_case, fmt):
    original, cfg = output_case
    cfg = cfg.model_copy(update={"save_h5ad": fmt == "h5ad"})
    command = ["scomnom", "enrichment", "sample", "--input-path", str(cfg.input_path)]
    result = se.run_sample_enrichment(cfg, command=command)
    root = cfg.input_path.parent.parent
    output_path = root / ("adata.enrichment_sample_r1.h5ad" if fmt == "h5ad" else "adata.enrichment_sample_r1.zarr.tar.zst")
    loaded = io_utils.load_dataset(output_path)
    analysis_id = "enrichment_sample_r1_round1"
    payload = loaded.uns["cluster_rounds"]["r1"]["sample_enrichment"][analysis_id]
    expected = result.uns["cluster_rounds"]["r1"]["sample_enrichment"][analysis_id]
    for table in se._TABLE_COLUMNS:
        pd.testing.assert_frame_equal(payload["tables"][table], expected["tables"][table])
    pd.testing.assert_frame_equal(payload["population_mapping"], expected["population_mapping"])
    assert payload["schema_version"] == 1
    assert payload["normalization"] == "log1p(counts / library_size * 1000000)"
    assert payload["input_provenance"]["count_assay"]["source"] == "layers/counts_cb"
    assert payload["population_mapping"].iloc[0]["population_id"] == "3"
    assert payload["software_versions"]["statsmodels"] != "unavailable"
    for key in ("dorothea", "msigdb", "progeny", "pseudobulk"):
        assert loaded.uns[key] == original.uns[key]
    assert loaded.uns["cluster_rounds"]["r1"]["decoupler"] == original.uns["cluster_rounds"]["r1"]["decoupler"]
    assert (loaded.layers["counts_cb"] != original.layers["counts_cb"]).nnz == 0
    folder = root / "tables" / analysis_id
    manifest = json.loads((folder / "settings.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["command"] == command
    assert manifest["resolved_config"]["round_id"] == "r1"
    assert manifest["table_schema"]["activity_contrasts"]["units"]["effect_standardized"] == "outcome SD"
    scores = pd.read_csv(folder / "activity_scores.tsv", sep="\t", dtype={"population_id": str})
    assert {"sample", "condition", "age"}.issubset(scores.columns)
    assert scores["population_id"].unique().tolist() == ["3"]
    assert len(scores) == 36
    excluded = pd.read_csv(folder / "model_exclusions.tsv", sep="\t")
    assert excluded.loc[~excluded["included"], "exclusion_reason"].tolist() == ["missing_covariate:age"]


def test_repeated_analysis_keeps_prior_round_payload(output_case):
    _, cfg = output_case
    first = se.run_sample_enrichment(cfg)
    root = cfg.input_path.parent.parent
    cfg2 = cfg.model_copy(update={
        "input_path": root / "adata.enrichment_sample_r1.zarr.tar.zst", "output_name": "second",
    })
    second = se.run_sample_enrichment(cfg2)
    payloads = second.uns["cluster_rounds"]["r1"]["sample_enrichment"]
    assert set(payloads) == {"enrichment_sample_r1_round1", "enrichment_sample_r1_round2"}
    pd.testing.assert_frame_equal(
        first.uns["cluster_rounds"]["r1"]["sample_enrichment"]["enrichment_sample_r1_round1"]["tables"]["activity_contrasts"],
        payloads["enrichment_sample_r1_round1"]["tables"]["activity_contrasts"],
    )


def test_scoring_only_outputs_have_readable_empty_model_tables(output_case):
    _, cfg = output_case
    cfg = cfg.model_copy(update={"condition_key": None, "contrasts": (), "covariates": ()})
    se.run_sample_enrichment(cfg)
    folder = cfg.input_path.parent.parent / "tables" / "enrichment_sample_r1_round1"
    for table in ("activity_contrasts", "model_audit", "model_exclusions"):
        frame = pd.read_csv(folder / f"{table}.tsv", sep="\t")
        assert frame.empty
        assert frame.columns.tolist() == se._TABLE_COLUMNS[table]


def test_serialization_failure_is_fatal_and_manifest_records_failure(output_case, monkeypatch):
    _, cfg = output_case
    def fail(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(io_utils, "save_dataset", fail)
    with pytest.raises(OSError, match="synthetic disk failure"):
        se.run_sample_enrichment(cfg)
    folder = cfg.input_path.parent.parent / "tables" / "enrichment_sample_r1_round1"
    manifest = json.loads((folder / "settings.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "OSError"
    assert not (cfg.input_path.parent.parent / "adata.enrichment_sample_r1.zarr.tar.zst").exists()


def test_resource_failure_does_not_write_successful_manifest(output_case, monkeypatch):
    _, cfg = output_case
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic resource failure")
    monkeypatch.setattr(au, "_load_dorothea_net", fail)
    with pytest.raises(RuntimeError, match="synthetic resource failure"):
        se.run_sample_enrichment(cfg)
    manifest = json.loads((cfg.input_path.parent.parent / "tables" / "enrichment_sample_r1_round1" / "settings.json").read_text())
    assert manifest["status"] == "failed"


def test_default_output_does_not_replace_input(output_case):
    _, cfg = output_case
    cfg = cfg.model_copy(update={"output_dir": cfg.input_path.parent, "output_name": "input"})
    with pytest.raises(ValueError, match="input dataset"):
        se.run_sample_enrichment(cfg)
