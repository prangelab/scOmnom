from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from scipy.sparse import csr_matrix

from scomnom.composition_utils import (
    _annotate_sccoda_contrasts,
    _build_composition_consensus_summary,
    _standardize_composition_results,
    run_graph_da,
    run_milo_da,
)


def _make_milo_adata(*, levels: tuple[str, ...] = ("control", "treated")) -> ad.AnnData:
    rng = np.random.default_rng(12)
    samples_per_level = 4
    cells_per_sample = 24
    sample_names = [
        f"{level}_s{sample_idx}"
        for level in levels
        for sample_idx in range(samples_per_level)
    ]
    conditions = [
        level
        for level in levels
        for _ in range(samples_per_level)
    ]
    sample = np.repeat(sample_names, cells_per_sample)
    condition = np.repeat(conditions, cells_per_sample)
    cluster = np.tile(np.repeat(["C00", "C01"], cells_per_sample // 2), len(sample_names))
    cluster_shift = np.where(cluster == "C00", -1.0, 1.0)
    embedding = np.column_stack(
        [
            cluster_shift + rng.normal(0, 0.35, len(sample)),
            rng.normal(0, 0.5, len(sample)),
            rng.normal(0, 0.5, len(sample)),
        ]
    )
    obs = pd.DataFrame(
        {
            "sample_id": sample,
            "condition": pd.Categorical(condition, categories=list(levels), ordered=True),
            "cluster": pd.Categorical(cluster),
        },
        index=[f"cell_{idx}" for idx in range(len(sample))],
    )
    adata = ad.AnnData(X=csr_matrix((len(sample), 2), dtype=np.float32), obs=obs)
    adata.obsm["X_integrated"] = embedding
    return adata


def _install_fake_milo_fit(monkeypatch, *, fail: bool = False) -> list[dict]:
    calls: list[dict] = []

    def fake_da_nhoods(
        self,
        mdata,
        design,
        model_contrasts=None,
        subset_samples=None,
        add_intercept=True,
        feature_key="rna",
        solver="edger",
    ):
        calls.append(
            {
                "design": design,
                "contrast": model_contrasts,
                "samples": tuple(subset_samples or ()),
                "solver": solver,
                "n_neighborhoods": mdata["milo"].n_vars,
            }
        )
        if fail:
            raise ValueError("synthetic fit failure")
        n_neighborhoods = mdata["milo"].n_vars
        mdata["milo"].var["logFC"] = np.linspace(-1.0, 1.0, n_neighborhoods)
        mdata["milo"].var["PValue"] = np.linspace(0.001, 0.2, n_neighborhoods)
        mdata["milo"].var["FDR"] = np.linspace(0.01, 0.3, n_neighborhoods)
        mdata["milo"].var["SpatialFDR"] = np.linspace(0.02, 0.4, n_neighborhoods)

    monkeypatch.setattr(pt.tl.Milo, "da_nhoods", fake_da_nhoods)
    return calls


def test_milo_uses_refined_neighborhoods_and_maps_results(monkeypatch) -> None:
    calls = _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata()

    results, neighborhoods, regions, region_samples, coverage = run_milo_da(
        adata,
        cluster_key="cluster",
        sample_key="sample_id",
        condition_key="condition",
        covariates=[],
        n_seeds=32,
        k_ref=12,
        min_size=5,
        min_nonzero_samples_per_level=2,
        random_state=9,
    )

    assert len(calls) == 1
    assert calls[0]["solver"] == "pydeseq2"
    assert calls[0]["contrast"] == "__milo_conditionL001-__milo_conditionL000"
    assert not results.empty
    assert set(results["pair"]) == {"control_vs_treated"}
    assert set(results["level_ref"]) == {"control"}
    assert set(results["level_test"]) == {"treated"}
    assert np.allclose(results["effect"], results["effect_raw"], equal_nan=True)
    assert "effect_shrunk" not in results
    assert np.allclose(results["fdr"], results["fdr_spatial"], equal_nan=True)
    assert set(results["engine"]) == {"pertpy_milo"}
    assert "effect_requires_review" in results
    assert results["region_id"].notna().any()
    assert not regions.empty
    assert not region_samples.empty
    assert not coverage.empty
    assert coverage["fraction_unique_significant_cells"].between(0, 1).all()
    assert "coverage_requires_review" in coverage

    retained = neighborhoods[neighborhoods["passes_min_size"]]
    assert retained["index_cell"].is_unique
    assert retained["neighborhood_size"].ge(5).all()
    assert retained["refined_neighborhoods"].iloc[0] <= 32
    assert retained["retained_neighborhoods"].iloc[0] == len(retained)
    assert retained["graph_neighbors_actual"].eq(12).all()
    assert retained["cluster_label"].isin({"C00", "C01"}).all()
    assert retained["tested"].any()


def test_milo_runs_all_pairwise_contrasts(monkeypatch) -> None:
    calls = _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata(levels=("control", "mild", "strong"))

    results, neighborhoods, regions, region_samples, coverage = run_milo_da(
        adata,
        cluster_key="cluster",
        sample_key="sample_id",
        condition_key="condition",
        covariates=[],
        n_seeds=24,
        k_ref=10,
        min_size=4,
        min_nonzero_samples_per_level=1,
        random_state=5,
        solver="edger",
    )

    assert len(calls) == 3
    assert {call["solver"] for call in calls} == {"edger"}
    assert {call["samples"] for call in calls} == {()}
    assert set(results["pair"]) == {
        "control_vs_mild",
        "control_vs_strong",
        "mild_vs_strong",
    }
    assert neighborhoods["n_pairs_total"].eq(3).all()
    assert neighborhoods.loc[neighborhoods["passes_min_size"], "tested_pair_count"].le(3).all()
    assert set(coverage["pair"]) == set(results["pair"])


def test_milo_real_pydeseq2_handles_three_level_design() -> None:
    adata = _make_milo_adata(levels=("control", "mild", "strong"))

    results, neighborhoods, regions, region_samples, coverage = run_milo_da(
        adata,
        cluster_key="cluster",
        sample_key="sample_id",
        condition_key="condition",
        covariates=[],
        n_seeds=12,
        k_ref=10,
        min_size=4,
        min_nonzero_samples_per_level=1,
        random_state=5,
        solver="pydeseq2",
        group_regions=False,
    )

    assert set(results["pair"]) == {
        "control_vs_mild",
        "control_vs_strong",
        "mild_vs_strong",
    }
    assert neighborhoods["tested_pair_count"].max() == 3
    assert regions.empty
    assert region_samples.empty
    assert set(coverage["pair"]) == set(results["pair"])


def test_milo_model_failure_is_not_replaced_by_another_backend(monkeypatch) -> None:
    calls = _install_fake_milo_fit(monkeypatch, fail=True)
    adata = _make_milo_adata()

    with pytest.raises(
        RuntimeError,
        match="Milo inference failed.*solver=pydeseq2.*treated vs control",
    ):
        run_milo_da(
            adata,
            cluster_key="cluster",
            sample_key="sample_id",
            condition_key="condition",
            covariates=[],
            n_seeds=20,
            k_ref=10,
            min_size=4,
            min_nonzero_samples_per_level=1,
        )

    assert len(calls) == 1


def test_milo_refinement_is_deterministic_and_applies_min_size(monkeypatch) -> None:
    calls = _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata()
    kwargs = {
        "cluster_key": "cluster",
        "sample_key": "sample_id",
        "condition_key": "condition",
        "covariates": [],
        "n_seeds": 18,
        "k_ref": 8,
        "min_size": 100,
        "min_nonzero_samples_per_level": 99,
        "random_state": 17,
    }

    first_results, first, first_regions, first_samples, first_coverage = run_milo_da(adata, **kwargs)
    second_results, second, second_regions, second_samples, second_coverage = run_milo_da(adata, **kwargs)

    assert first_results.empty and second_results.empty
    assert not calls
    assert not first["passes_min_size"].any()
    pd.testing.assert_frame_equal(
        first[["index_cell", "neighborhood_size", "kth_distance"]].reset_index(drop=True),
        second[["index_cell", "neighborhood_size", "kth_distance"]].reset_index(drop=True),
    )


def test_milo_bounds_neighbors_and_records_unsupported_neighborhoods(monkeypatch) -> None:
    calls = _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata()[:8].copy()
    adata.obs["condition"] = pd.Categorical(
        ["control"] * 4 + ["treated"] * 4,
        categories=["control", "treated"],
        ordered=True,
    )
    adata.obs["sample_id"] = [f"sample_{idx}" for idx in range(8)]

    results, neighborhoods, regions, region_samples, coverage = run_milo_da(
        adata,
        cluster_key="cluster",
        sample_key="sample_id",
        condition_key="condition",
        covariates=[],
        n_seeds=5,
        k_ref=30,
        min_size=1,
        min_nonzero_samples_per_level=99,
    )

    assert results.empty
    assert not calls
    assert not neighborhoods.empty
    assert neighborhoods["graph_neighbors_actual"].eq(7).all()
    assert not neighborhoods["tested"].any()


def test_run_graph_da_is_a_deprecated_milo_alias(monkeypatch) -> None:
    _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata()

    with pytest.deprecated_call(match="use run_milo_da"):
        results, neighborhoods, regions, region_samples, coverage = run_graph_da(
            adata,
            cluster_key="cluster",
            sample_key="sample_id",
            condition_key="condition",
            covariates=[],
            n_seeds=12,
            k_ref=8,
            min_size=4,
            min_nonzero_samples_per_level=1,
        )

    assert not results.empty
    assert not neighborhoods.empty


def test_composition_consensus_uses_canonical_milo_evidence() -> None:
    global_result = {
        "cluster": ["C01"],
        "term": ["condition_treated"],
        "pair": ["control_vs_treated"],
        "effect": [1.0],
        "fdr": [0.01],
    }
    summary = _build_composition_consensus_summary(
        {
            "milo": pd.DataFrame(
                {
                    "cluster": ["nh_000001", "nh_000002"],
                    "cluster_label": ["C01", "C01"],
                    "region_id": ["milo_region_control_vs_treated_001"] * 2,
                    "region_cluster_label": ["C01", "C01"],
                    "region_effect_median": [1.2, 1.2],
                    "region_min_fdr": [0.01, 0.01],
                    "term": ["condition_treated"] * 2,
                    "pair": ["control_vs_treated"] * 2,
                    "effect": [1.0, 1.4],
                    "fdr": [0.01, 0.02],
                }
            ),
            "clr": pd.DataFrame(global_result),
        },
        alpha=0.05,
        condition_key="condition",
    )

    assert summary.loc[0, "method_sig_milo"]
    assert summary.loc[0, "method_sign_milo"] == 1
    assert summary.loc[0, "high_confidence_da"]
    assert summary.loc[0, "da_evidence_tier"] == "cross_scale_supported"


def test_milo_region_grouping_can_be_disabled(monkeypatch) -> None:
    _install_fake_milo_fit(monkeypatch)
    adata = _make_milo_adata()

    results, neighborhoods, regions, region_samples, coverage = run_milo_da(
        adata,
        cluster_key="cluster",
        sample_key="sample_id",
        condition_key="condition",
        covariates=[],
        n_seeds=20,
        k_ref=10,
        min_size=4,
        min_nonzero_samples_per_level=1,
        group_regions=False,
    )

    assert not results.empty
    assert results["region_id"].isna().all()
    assert regions.empty
    assert region_samples.empty
    assert not coverage.empty
    assert coverage["n_regions"].isna().all()


def test_composition_consensus_accepts_selected_sccoda_support() -> None:
    summary = _build_composition_consensus_summary(
        {
            "milo": pd.DataFrame(
                {
                    "cluster": ["nh_000001"],
                    "region_id": ["milo_region_control_vs_treated_001"],
                    "region_cluster_label": ["C01"],
                    "region_effect_median": [0.8],
                    "region_min_fdr": [0.01],
                    "term": ["condition_treated"],
                    "pair": ["control_vs_treated"],
                    "effect": [0.8],
                    "fdr": [0.01],
                }
            ),
            "sccoda": pd.DataFrame(
                {
                    "cluster": ["C01"],
                    "term": ["conditionT.treated"],
                    "pair": ["control_vs_treated"],
                    "effect": [0.5],
                    "is_significant": [True],
                }
            ),
        },
        alpha=0.05,
        condition_key="condition",
    )

    assert summary.loc[0, "method_sig_sccoda"]
    assert summary.loc[0, "high_confidence_da"]


def test_sccoda_selected_effect_and_contrast_are_preserved() -> None:
    raw = pd.DataFrame(
        {
            "Final Parameter": [0.7, 0.0],
            "Inclusion probability": [0.98, 0.2],
            "term": ["conditionT.mild-moderate", "conditionT.mild-moderate"],
            "cluster": ["C01", "C02"],
        }
    )
    standardized = _standardize_composition_results(
        raw,
        backend="sccoda",
        condition_key="condition",
    )
    annotated = _annotate_sccoda_contrasts(
        standardized,
        condition_levels=["control", "mild-moderate"],
    )

    assert annotated["is_significant"].tolist() == [True, False]
    assert annotated["pair"].tolist() == ["control_vs_mild-moderate"] * 2
