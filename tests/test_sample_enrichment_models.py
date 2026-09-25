from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from scomnom.config import SampleEnrichmentConfig
from scomnom.sample_enrichment import (
    _PreparedPseudobulks,
    _ScoredActivities,
    _fit_activity_contrasts,
    _resolve_contrasts,
)


def _inputs(values=None, *, paired=False, extra_metadata=None):
    if paired:
        condition = ["ctrl", "stim"] * 4
        values = np.array([0, 1, 10, 12, 20, 23, 30, 32], dtype=float) if values is None else values
    else:
        condition = ["ctrl"] * 6 + ["stim"] * 6
        values = np.tile([-1., 0., 1.], 4) + np.repeat([0., 2.], 6) if values is None else values
    qc = pd.DataFrame({
        "replicate_id": [f"s{i}" for i in range(len(condition))],
        "population_id": "0", "condition": condition, "included": True,
        "exclusion_reason": "", "n_cells": 20,
    }, index=pd.Index([f"pb{i}" for i in range(len(condition))], name="pb_id"))
    if paired:
        qc["donor"] = np.repeat([1, 2, 3, 4], 2)
    for key, val in (extra_metadata or {}).items():
        qc[key] = val
    scores = pd.DataFrame({
        "pb_id": qc.index, "replicate_id": qc["replicate_id"].to_numpy(),
        "population_id": "0", "population_label": "C00", "resource": "dorothea",
        "activity": "TF1", "score": values,
    })
    audit = pd.DataFrame({"population_id": ["0"], "population_label": ["C00"], "resource": ["dorothea"], "activity": ["TF1"], "status": ["ok"]})
    prepared = _PreparedPseudobulks(pd.DataFrame(), pd.DataFrame(), qc, {})
    scored = _ScoredActivities(scores, audit)
    kwargs = {"subject_key": "donor", "covariates": ("donor",)} if paired else {}
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample", condition_key="condition", contrasts=("stim:ctrl",), **kwargs)
    return prepared, scored, cfg


def test_independent_effect_hc3_t_intervals_and_standardization():
    prepared, scored, cfg = _inputs()
    result = _fit_activity_contrasts(prepared, scored, cfg)
    row = result.contrasts.iloc[0]
    se = np.sqrt(8 / 25)
    margin = t.ppf(.975, df=10) * se
    sd = scored.scores["score"].std(ddof=1)
    assert row["status"] == "ok"
    assert row["effect_raw"] == pytest.approx(2)
    assert row["se_raw"] == pytest.approx(se)
    assert row["ci_low_raw"] == pytest.approx(2 - margin)
    assert row["ci_high_raw"] == pytest.approx(2 + margin)
    assert row["effect_standardized"] == pytest.approx(2 / sd)
    assert row["ci_low_standardized"] == pytest.approx((2 - margin) / sd)
    assert row["ci_high_standardized"] == pytest.approx((2 + margin) / sd)
    assert row["pvalue"] == pytest.approx(2 * t.sf(2 / se, df=10))
    assert row["fdr"] == row["pvalue"]
    assert row["n_test"] == row["n_reference"] == 6
    assert result.audit.iloc[0]["covariance"] == "HC3"
    assert result.audit.iloc[0]["inference_distribution"] == "t"
    assert result.audit.iloc[0]["df_resid"] == 10


def test_null_effect_and_reverse_contrast():
    prepared, scored, cfg = _inputs(values=np.tile([-1., 0., 1.], 4))
    null = _fit_activity_contrasts(prepared, scored, cfg).contrasts.iloc[0]
    assert null["effect_raw"] == pytest.approx(0, abs=1e-12)
    assert null["pvalue"] == pytest.approx(1)
    prepared, scored, cfg = _inputs()
    forward = _fit_activity_contrasts(prepared, scored, cfg).contrasts.iloc[0]
    reverse = _fit_activity_contrasts(prepared, scored, cfg.model_copy(update={"contrasts": ("ctrl:stim",)})).contrasts.iloc[0]
    assert reverse["effect_raw"] == pytest.approx(-forward["effect_raw"])
    assert reverse["ci_low_raw"] == pytest.approx(-forward["ci_high_raw"])
    assert reverse["pvalue"] == pytest.approx(forward["pvalue"])


def test_numeric_and_categorical_covariates_recover_adjusted_effect():
    age = np.tile(np.arange(6), 2)
    batch = np.tile([0, 1], 6)
    condition = np.repeat([0, 1], 6)
    x = np.column_stack([np.ones(12), age, batch, condition])
    residual = np.array([1, -1, 2, -2, 3, -3] * 2, dtype=float)
    residual -= x @ np.linalg.lstsq(x, residual, rcond=None)[0]
    y = 5 + 3 * age + 4 * batch + 2 * condition + residual
    prepared, scored, cfg = _inputs(y, extra_metadata={"age": age, "batch": pd.Categorical(np.where(batch, "B", "A"))})
    cfg = cfg.model_copy(update={"covariates": ("age", "batch")})
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["effect_raw"] == pytest.approx(2)
    terms = json.loads(result.audit.iloc[0]["design_terms"])
    assert any(term["field"] == "age" and term["kind"] == "numeric" for term in terms)
    assert any(term["field"] == "batch" and term["kind"] == "categorical" for term in terms)


def test_numeric_subject_id_is_a_fixed_categorical_effect():
    prepared, scored, cfg = _inputs(paired=True)
    result = _fit_activity_contrasts(prepared, scored, cfg)
    row = result.contrasts.iloc[0]
    assert row["effect_raw"] == pytest.approx(2)
    assert row["n_subjects"] == 4
    assert result.audit.iloc[0]["df_resid"] == 3
    terms = json.loads(result.audit.iloc[0]["design_terms"])
    assert sum(term["field"] == "donor" for term in terms) == 3
    assert all(term["kind"] == "categorical" for term in terms if term["field"] == "donor")


def test_missing_covariate_excludes_the_entire_pair_and_records_each_library():
    prepared, scored, cfg = _inputs(paired=True)
    prepared.qc["time"] = np.tile([0., 2.], 4)
    prepared.qc.loc["pb0", "time"] = np.nan
    prepared.qc.loc["pb3", "time"] = 3
    prepared.qc.loc["pb5", "time"] = 1
    cfg = cfg.model_copy(update={"covariates": ("donor", "time")})
    result = _fit_activity_contrasts(prepared, scored, cfg)
    excluded = result.exclusions.set_index("pb_id")
    assert "missing_covariate:time" in excluded.loc["pb0", "exclusion_reason"]
    assert "incomplete_pair" in excluded.loc["pb1", "exclusion_reason"]
    assert result.contrasts.iloc[0]["n_included"] == 6
    assert result.contrasts.iloc[0]["n_excluded"] == 2


def test_qc_exclusion_does_not_turn_a_paired_model_into_an_unpaired_model():
    prepared, scored, cfg = _inputs(paired=True)
    prepared.qc.loc[["pb0", "pb2"], "included"] = False
    prepared.qc.loc[["pb0", "pb2"], "exclusion_reason"] = "insufficient_cells"
    scored.scores.drop(index=[0, 2], inplace=True)
    result = _fit_activity_contrasts(prepared, scored, cfg)
    row = result.contrasts.iloc[0]
    assert row["status"] == "insufficient_complete_pairs"
    assert row["n_subjects"] == 2
    assert np.isnan(row["pvalue"])
    assert set(result.exclusions.loc[~result.exclusions["included"], "pb_id"]) == {"pb0", "pb1", "pb2", "pb3"}


def test_duplicate_subject_condition_libraries_are_fatal_before_qc():
    prepared, scored, cfg = _inputs(paired=True)
    prepared.qc.loc["pb2", "donor"] = 1
    prepared.qc.loc["pb2", "included"] = False
    with pytest.raises(ValueError, match="multiple replicate libraries"):
        _fit_activity_contrasts(prepared, scored, cfg)


@pytest.mark.parametrize("kind,status", [
    ("constant", "covariate_no_variation"),
    ("collinear", "rank_deficient"),
    ("saturated", "zero_residual_degrees_of_freedom"),
])
def test_nonestimable_designs_keep_exact_statuses(kind, status):
    prepared, scored, cfg = _inputs()
    if kind == "constant":
        prepared.qc["cov"] = 1.
    elif kind == "collinear":
        prepared.qc["cov"] = np.repeat([0., 1.], 6)
    else:
        prepared.qc["cov"] = pd.Categorical(["shared"] + [f"id{i}" for i in range(1, 6)] + ["shared"] + [f"id{i}" for i in range(7, 12)])
    result = _fit_activity_contrasts(prepared, scored, cfg.model_copy(update={"covariates": ("cov",)}))
    assert result.contrasts.iloc[0]["status"] == status
    assert np.isnan(result.contrasts.iloc[0]["pvalue"])
    assert np.isnan(result.contrasts.iloc[0]["fdr"])


@pytest.mark.parametrize("values,status", [
    (np.ones(12), "zero_variance_activity"),
    (np.array([np.nan] + [1.] * 11), "nonfinite_activity"),
    (np.repeat([0., 2.], 6), "zero_residual_variance"),
])
def test_invalid_activity_is_not_silently_filtered(values, status):
    prepared, scored, cfg = _inputs(values)
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["status"] == status
    assert np.isnan(result.contrasts.iloc[0]["pvalue"])
    assert result.contrasts.iloc[0]["n_included"] == 12


def test_replicate_threshold_boundary_and_explicit_configuration():
    prepared, scored, cfg = _inputs()
    prepared.qc.loc[["pb0", "pb1", "pb2", "pb6", "pb7", "pb8"], "included"] = False
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["status"] == "ok"
    cfg = cfg.model_copy(update={"min_replicates_per_level": 4})
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["status"] == "insufficient_replicates"


def test_explicit_reference_and_absent_global_level_rules():
    prepared, _, cfg = _inputs()
    assert _resolve_contrasts(prepared.qc, cfg.model_copy(update={"contrasts": (), "reference": "ctrl"})) == (("stim", "ctrl"),)
    with pytest.raises(ValueError, match="explicit"):
        _resolve_contrasts(prepared.qc, cfg.model_copy(update={"contrasts": ()}))
    with pytest.raises(ValueError, match="absent"):
        _resolve_contrasts(prepared.qc, cfg.model_copy(update={"contrasts": ("missing:ctrl",)}))
    prepared.qc.loc["pb0", "condition"] = "third"
    with pytest.raises(ValueError, match="explicit"):
        _resolve_contrasts(prepared.qc, cfg.model_copy(update={"contrasts": (), "reference": "ctrl"}))


def test_bh_family_includes_all_tested_activities_and_ignores_plot_selection():
    prepared, scored, cfg = _inputs()
    for name, multiplier, resource in [("TF2", .4, "dorothea"), ("TF3", 0., "dorothea"), ("Pathway", .2, "progeny")]:
        extra = scored.scores.iloc[:12].copy()
        extra["activity"] = name
        extra["resource"] = resource
        extra["score"] = np.tile([-1., 0., 1.], 4) + np.repeat([0., 2 * multiplier], 6)
        scored = _ScoredActivities(pd.concat([scored.scores, extra], ignore_index=True), pd.concat([
            scored.audit, pd.DataFrame({"population_id": ["0"], "population_label": ["C00"], "resource": [resource], "activity": [name], "status": ["ok"]}),
        ], ignore_index=True))
    result = _fit_activity_contrasts(prepared, scored, cfg)
    tf = result.contrasts.query("resource == 'dorothea'").sort_values("pvalue")
    expected = np.minimum.accumulate((tf["pvalue"].to_numpy() * 3 / np.arange(1, 4))[::-1])[::-1].clip(0, 1)
    np.testing.assert_allclose(tf["fdr"], expected)
    assert set(tf["fdr_family_size"]) == {3}
    assert result.contrasts.query("resource == 'progeny'")["fdr_family_size"].iloc[0] == 1
    filtered = _fit_activity_contrasts(prepared, scored, cfg.model_copy(update={"plot_activity": ("TF1",)}))
    pd.testing.assert_frame_equal(filtered.contrasts, result.contrasts)


def test_outcome_rescaling_preserves_standardized_estimates_and_pvalues():
    prepared, scored, cfg = _inputs()
    baseline = _fit_activity_contrasts(prepared, scored, cfg).contrasts.iloc[0]
    scored.scores["score"] = 13 * scored.scores["score"] + 5
    scaled = _fit_activity_contrasts(prepared, scored, cfg).contrasts.iloc[0]
    assert scaled["effect_raw"] == pytest.approx(13 * baseline["effect_raw"])
    for key in ["effect_standardized", "ci_low_standardized", "ci_high_standardized", "pvalue"]:
        assert scaled[key] == pytest.approx(baseline[key])


def test_model_does_not_weight_replicates_by_cell_count():
    prepared, scored, cfg = _inputs()
    baseline = _fit_activity_contrasts(prepared, scored, cfg).contrasts
    prepared.qc["n_cells"] = [20, 30, 50, 100, 500, 1000] * 2
    imbalanced = _fit_activity_contrasts(prepared, scored, cfg).contrasts
    pd.testing.assert_frame_equal(imbalanced, baseline)


def test_population_specific_absent_level_is_a_status():
    prepared, scored, cfg = _inputs()
    prepared.qc.loc["pb0":"pb5", "population_id"] = "1"
    scored.scores.loc[0:5, "population_id"] = "1"
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["status"] == "absent_contrast_level"


def test_high_leverage_covariate_receives_explicit_hc3_status():
    prepared, scored, cfg = _inputs(extra_metadata={"singleton": [1.] + [0.] * 11})
    result = _fit_activity_contrasts(prepared, scored, cfg.model_copy(update={"covariates": ("singleton",)}))
    assert result.contrasts.iloc[0]["status"] == "hc3_undefined_leverage"
    assert np.isnan(result.contrasts.iloc[0]["pvalue"])


def test_untested_activity_is_excluded_from_bh_family_size():
    prepared, scored, cfg = _inputs()
    audit = pd.concat([scored.audit, scored.audit.assign(activity="TF_missing", status="insufficient_target_overlap")], ignore_index=True)
    result = _fit_activity_contrasts(prepared, _ScoredActivities(scored.scores, audit), cfg)
    assert set(result.contrasts["fdr_family_size"]) == {1}
    assert np.isnan(result.contrasts.set_index("activity").loc["TF_missing", "fdr"])


def test_pairing_collinearity_does_not_drop_subject_or_covariate():
    prepared, scored, cfg = _inputs(paired=True, extra_metadata={"age": np.repeat([40., 50., 60., 70.], 2)})
    cfg = cfg.model_copy(update={"covariates": ("donor", "age")})
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.iloc[0]["status"] == "rank_deficient"
    assert "age" in result.audit.iloc[0]["formula"]
    assert "donor" in result.audit.iloc[0]["formula"]


def test_bh_families_are_separate_for_each_population_and_contrast():
    prepared, scored, cfg = _inputs()
    other_qc = prepared.qc.copy()
    other_qc.index = other_qc.index + "_other"
    other_qc["population_id"] = "1"
    prepared = _PreparedPseudobulks(pd.DataFrame(), pd.DataFrame(), pd.concat([prepared.qc, other_qc]), {})
    other_scores = scored.scores.assign(population_id="1", population_label="C01", pb_id=scored.scores["pb_id"] + "_other")
    scored = _ScoredActivities(pd.concat([scored.scores, other_scores], ignore_index=True), pd.concat([
        scored.audit, scored.audit.assign(population_id="1", population_label="C01"),
    ], ignore_index=True))
    cfg = cfg.model_copy(update={"contrasts": ("stim:ctrl", "ctrl:stim")})
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts["fdr_family_id"].nunique() == 4
    assert set(result.contrasts["fdr_family_size"]) == {1}


def test_synthetic_null_panel_has_no_widespread_significance():
    prepared, scored, cfg = _inputs()
    rng = np.random.default_rng(5713)
    scores, audits = [], []
    for number in range(100):
        name = f"null_{number}"
        scores.append(scored.scores.assign(activity=name, score=rng.normal(size=12)))
        audits.append(scored.audit.assign(activity=name))
    result = _fit_activity_contrasts(prepared, _ScoredActivities(pd.concat(scores), pd.concat(audits)), cfg)
    assert set(result.contrasts["status"]) == {"ok"}
    assert .4 < result.contrasts["pvalue"].mean() < .7
    assert (result.contrasts["pvalue"] < .05).sum() <= 15
    assert (result.contrasts["fdr"] < .05).sum() <= 5


def test_score_only_configuration_produces_no_inferential_results():
    prepared, scored, cfg = _inputs()
    cfg = SampleEnrichmentConfig(input_path="x", replicate_key="sample")
    result = _fit_activity_contrasts(prepared, scored, cfg)
    assert result.contrasts.empty
    assert result.audit.empty
    assert result.exclusions.empty
