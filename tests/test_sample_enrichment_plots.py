from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scomnom.sample_enrichment_plot_utils import sample_enrichment_artifacts


@pytest.fixture
def plot_payload():
    qc = pd.DataFrame({
        "pb_id": [f"p{i}" for i in range(8)], "replicate_id": [f"s{i}" for i in range(8)],
        "population_id": "3", "population_label": "C03: Macrophages",
        "n_cells": [100, 80, 90, 70, 120, 95, 110, 10],
        "library_size": [1000, 800, 900, 700, 1200, 950, 1100, 100],
        "included": [True] * 7 + [False], "selected_population": True,
    })
    scores = pd.concat([qc.iloc[:7].assign(
        resource="dorothea", activity=activity, score=np.arange(7) * .4 + k,
        condition=["ctrl"] * 4 + ["stim"] * 3, donor=["a", "b", "c", "d", "a", "b", "c"],
    ) for k, activity in enumerate(["TF1", "TF2", "TF3"])], ignore_index=True)
    contrasts = pd.DataFrame({
        "population_id": "3", "population_label": "C03: Macrophages", "resource": "dorothea",
        "activity": ["TF1", "TF2", "TF3"], "contrast": "stim:ctrl", "test": "stim", "reference": "ctrl",
        "effect_standardized": [1., -.5, .1], "ci_low_standardized": [.3, -1., -.2],
        "ci_high_standardized": [1.7, 0., .4], "fdr": [.01, .09, .8], "status": "ok",
    })
    exclusions = qc.assign(contrast="stim:ctrl", included=[True, True, True, False, True, True, True, False])
    return {"resolved_config": {"condition_key": "condition", "subject_key": "donor", "min_cells_per_replicate_group": 20},
            "tables": {"pseudobulk_qc": qc, "activity_scores": scores, "activity_contrasts": contrasts,
                       "model_exclusions": exclusions}}


def test_artifacts_do_not_save_or_mutate_and_selectors_leave_overview_complete(plot_payload, tmp_path):
    before = deepcopy(plot_payload)
    artifacts = list(sample_enrichment_artifacts(plot_payload, figdir=tmp_path, activities=("TF2",)))
    assert {a.stem.split("_")[0] for a in artifacts} == {"qc", "overview", "forest", "samples"}
    assert not list(tmp_path.iterdir())
    overview = next(a.fig for a in artifacts if a.stem.startswith("overview"))
    assert len(overview.axes[0].collections[0].get_offsets()) == 3
    sample = next(a.fig for a in artifacts if a.stem.startswith("samples"))
    assert "TF2" in sample.axes[0].get_title()
    assert len(sample.axes[0].lines) == 3
    assert "Adjusted" in sample.axes[0].get_xlabel()
    for key, frame in before["tables"].items():
        pd.testing.assert_frame_equal(plot_payload["tables"][key], frame)
    for artifact in artifacts:
        plt.close(artifact.fig)


def test_independent_plots_do_not_connect_libraries(plot_payload):
    plot_payload["resolved_config"]["subject_key"] = None
    artifacts = list(sample_enrichment_artifacts(plot_payload, figdir=Path("sample"), activities=("TF1",)))
    sample = next(a.fig for a in artifacts if a.stem.startswith("samples"))
    assert not sample.axes[0].lines
    for artifact in artifacts:
        plt.close(artifact.fig)


def test_score_only_plots_and_unknown_selectors(plot_payload):
    plot_payload["tables"]["activity_contrasts"] = plot_payload["tables"]["activity_contrasts"].iloc[:0]
    artifacts = list(sample_enrichment_artifacts(plot_payload, figdir=Path("sample")))
    assert not any(a.stem.startswith(("forest", "overview")) for a in artifacts)
    assert all("Adjusted" not in a.fig.axes[0].get_xlabel() for a in artifacts)
    for artifact in artifacts:
        plt.close(artifact.fig)
    with pytest.raises(ValueError, match="Unknown plot activities"):
        list(sample_enrichment_artifacts(plot_payload, figdir=Path("sample"), activities=("missing",)))


def test_regeneration_selects_explicit_analysis_and_reserves_new_folders(plot_payload, tmp_path, monkeypatch):
    import anndata as ad
    from scomnom.config import SampleEnrichmentConfig
    from scomnom import sample_enrichment as se

    obj = ad.AnnData()
    obj.uns = {"active_cluster_round": "r1", "cluster_rounds": {"r1": {
        "sample_enrichment": {"first": plot_payload, "second": deepcopy(plot_payload)},
    }}}
    for payload in obj.uns["cluster_rounds"]["r1"]["sample_enrichment"].values():
        payload["schema_version"] = 1
    cfg = SampleEnrichmentConfig(input_path=tmp_path / "input.zarr", regenerate_figures=True)
    with pytest.raises(ValueError, match="Choose --analysis-id"):
        se._regenerate_sample_figures(obj, cfg, tmp_path, None)
    chosen = []
    monkeypatch.setattr(se, "_persist_sample_figures", lambda payload, *args: chosen.append(payload) or [])
    cfg = cfg.model_copy(update={"analysis_id": "second"})
    for _ in range(2):
        se._regenerate_sample_figures(obj, cfg, tmp_path, None)
    assert all(p is obj.uns["cluster_rounds"]["r1"]["sample_enrichment"]["second"] for p in chosen)
    assert len(list((tmp_path / "figures" / "regeneration").glob("*/settings.json"))) == 2
    assert len(obj.uns["cluster_rounds"]["r1"]["sample_enrichment"]) == 2


def test_plot_failure_restores_figure_routing_and_closes_figure(plot_payload, tmp_path, monkeypatch):
    from scomnom import plot_utils as pu
    from scomnom import sample_enrichment as se

    prior = (pu.ROOT_FIGDIR, pu.RUN_FIG_SUBDIR, pu.RUN_KEY, pu.FIGURE_FORMATS)
    figures = plt.get_fignums()
    def fail(*args):
        raise OSError("synthetic plot disk failure")
    monkeypatch.setattr(pu, "persist_plot_artifacts", fail)
    with pytest.raises(OSError, match="disk failure"):
        se._persist_sample_figures(plot_payload, tmp_path, "new_run", ["png"], ())
    assert prior == (pu.ROOT_FIGDIR, pu.RUN_FIG_SUBDIR, pu.RUN_KEY, pu.FIGURE_FORMATS)
    assert plt.get_fignums() == figures
