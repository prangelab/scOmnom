import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from unittest.mock import Mock

from scomnom.config import IntegrateConfig
from scomnom.integrate import (
    _run_integrations,
    _select_best_embedding,
    _select_embedding_from_scib_metrics,
    _store_scib_selection_decision,
    run_integrate,
)
from scomnom.io_utils import load_dataset, save_dataset


def synthetic_adata(n_cells=64, n_genes=24, n_batches=2, seed=0):
    rng = np.random.default_rng(seed)
    adata = sc.AnnData(X=rng.normal(size=(n_cells, n_genes)))
    adata.obs["batch"] = pd.Categorical(
        rng.integers(0, n_batches, size=n_cells).astype(str)
    )
    adata.obs["label"] = pd.Categorical(
        rng.integers(0, 4, size=n_cells).astype(str)
    )
    adata.var["highly_variable"] = True
    return adata


def scib_metrics(rows):
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["Bio conservation", "Batch correction", "Total"],
    )


def test_scib_selector_preserves_pareto_tier_over_higher_total_tradeoff():
    metrics = scib_metrics(
        {
            "Unintegrated": (0.50, 0.50, 0.50),
            "Harmony": (0.51, 0.51, 0.51),
            "scANVI": (0.90, 0.40, 0.70),
        }
    )

    decision = _select_embedding_from_scib_metrics(metrics)

    assert decision.selected_embedding == "Harmony"
    assert decision.tier == "tier1_bio_and_batch"


def test_scib_selector_skips_net_harmful_higher_tier_and_falls_through():
    metrics = scib_metrics(
        {
            "Unintegrated": (0.50, 0.50, 0.50),
            "BioOnlyNetLoss": (0.51, 0.10, 0.346),
            "BatchTradeoffGain": (0.40, 0.80, 0.56),
        }
    )

    decision = _select_embedding_from_scib_metrics(metrics)

    assert decision.selected_embedding == "BatchTradeoffGain"
    assert decision.tier == "tier3_batch"


@pytest.mark.parametrize(
    "candidate",
    [
        (0.40, 0.40, 0.40),
        (0.70, 0.20, 0.50),
        (0.51, 0.10, 0.346),
    ],
)
def test_scib_selector_retains_baseline_without_total_improvement(candidate):
    metrics = scib_metrics(
        {
            "Unintegrated": (0.50, 0.50, 0.50),
            "Candidate": candidate,
        }
    )

    decision = _select_embedding_from_scib_metrics(metrics)

    assert decision.selected_embedding == "Unintegrated"
    assert decision.tier == "baseline"
    selected = decision.decision_table.loc[decision.decision_table["selected"]]
    assert selected["embedding"].tolist() == ["Unintegrated"]


def test_scib_selector_breaks_candidate_ties_deterministically():
    metrics = scib_metrics(
        {
            "Unintegrated": (0.50, 0.50, 0.50),
            "Zeta": (0.70, 0.70, 0.70),
            "Alpha": (0.70, 0.70, 0.70),
        }
    )

    decision = _select_embedding_from_scib_metrics(metrics)

    assert decision.selected_embedding == "Alpha"


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (scib_metrics({"Harmony": (0.5, 0.5, 0.5)}), "Unintegrated baseline missing"),
        (
            pd.DataFrame(
                {"Bio conservation": [0.5], "Batch correction": [0.5]},
                index=["Unintegrated"],
            ),
            "missing required selector column",
        ),
        (scib_metrics({"Unintegrated": (0.5, 0.5, 0.5), "Harmony": (np.nan, 0.6, 0.6)}), "non-finite"),
    ],
)
def test_scib_selector_fails_closed_on_invalid_tables(metrics, message):
    with pytest.raises(RuntimeError, match=message):
        _select_embedding_from_scib_metrics(metrics)


def test_scib_selector_rejects_duplicate_embedding_names():
    metrics = scib_metrics(
        {
            "Unintegrated": (0.5, 0.5, 0.5),
            "Harmony": (0.6, 0.6, 0.6),
        }
    )
    metrics = pd.concat([metrics, metrics.loc[["Harmony"]]])

    with pytest.raises(RuntimeError, match="duplicate embedding names"):
        _select_embedding_from_scib_metrics(metrics)


def test_scib_selector_persists_machine_readable_decision():
    adata = synthetic_adata()
    decision = _select_embedding_from_scib_metrics(
        scib_metrics(
            {
                "Unintegrated": (0.50, 0.50, 0.50),
                "Harmony": (0.70, 0.70, 0.70),
            }
        )
    )

    _store_scib_selection_decision(adata, decision)

    stored = adata.uns["integration"]
    assert stored["selection_policy"] == "pareto_tiers_with_total_guard"
    assert stored["selection_tier"] == "tier1_bio_and_batch"
    assert stored["selection_delta_total"] == pytest.approx(0.20)
    assert isinstance(stored["selection_decision_table"], pd.DataFrame)


def test_scib_selection_decision_survives_dataset_roundtrip(tmp_path):
    adata = synthetic_adata()
    decision = _select_embedding_from_scib_metrics(
        scib_metrics(
            {
                "Unintegrated": (0.50, 0.50, 0.50),
                "Harmony": (0.70, 0.70, 0.70),
                "scANVI": (0.80, 0.40, 0.64),
            }
        )
    )
    _store_scib_selection_decision(adata, decision)
    output_path = tmp_path / "integration_selection.zarr"

    save_dataset(adata, output_path, fmt="zarr", archive=False)
    loaded = load_dataset(output_path)

    stored = loaded.uns["integration"]
    assert stored["selection_policy"] == "pareto_tiers_with_total_guard"
    assert stored["selection_tier"] == "tier1_bio_and_batch"
    pd.testing.assert_frame_equal(
        stored["selection_decision_table"],
        decision.decision_table,
    )


def test_select_best_embedding_writes_and_stores_guarded_decision(
    monkeypatch,
    tmp_path,
):
    import scib_metrics.benchmark as benchmark_module
    import scomnom.integrate as integ

    adata = synthetic_adata()
    adata.obsm["Unintegrated"] = np.ones((adata.n_obs, 5))
    adata.obsm["Harmony"] = np.full((adata.n_obs, 5), 2.0)
    scaled = pd.DataFrame(
        {
            "Bio conservation": [0.80, 0.00, "Aggregate score"],
            "Batch correction": [0.20, 0.80, "Aggregate score"],
            "Total": [0.56, 0.32, "Aggregate score"],
        },
        index=["Unintegrated", "Harmony", "Metric Type"],
    )

    class FakeBenchmarker:
        def __init__(self, *args, **kwargs):
            pass

        def benchmark(self):
            pass

        def get_results(self, *, min_max_scale):
            return scaled.copy()

    monkeypatch.setattr(benchmark_module, "Benchmarker", FakeBenchmarker)
    monkeypatch.setattr(benchmark_module, "BioConservation", lambda: object())
    monkeypatch.setattr(benchmark_module, "BatchCorrection", lambda: object())
    monkeypatch.setattr(integ.plot_utils, "plot_scib_results_table", lambda *args, **kwargs: [])
    monkeypatch.setattr(integ.plot_utils, "persist_plot_artifacts", lambda *args, **kwargs: None)

    selected = _select_best_embedding(
        adata,
        ["Unintegrated", "Harmony"],
        "batch",
        "label",
        1,
        tmp_path,
    )

    assert selected == "Unintegrated"
    decision_path = tmp_path / "integration_metrics" / "integration_selection_decision.tsv"
    assert decision_path.exists()
    written = pd.read_csv(decision_path, sep="\t")
    assert written.loc[written["selected"], "embedding"].tolist() == ["Unintegrated"]
    stored = adata.uns["integration"]
    assert stored["selection_tier"] == "baseline"
    assert stored["selection_delta_total"] == pytest.approx(0.0)


def test_run_integrations_requires_hvgs(tmp_path):
    adata = synthetic_adata()
    del adata.var["highly_variable"]
    cfg = IntegrateConfig(input_path=tmp_path / "input.zarr", output_dir=tmp_path)

    try:
        _run_integrations(adata, cfg, methods=["Harmony"], batch_key="batch")
    except RuntimeError as exc:
        assert "Expected HVGs" in str(exc)
    else:
        raise AssertionError("_run_integrations should require highly_variable genes")


def test_run_integrations_harmony_embedding(monkeypatch, tmp_path):
    import scomnom.integrate as integ

    adata = synthetic_adata()
    cfg = IntegrateConfig(input_path=tmp_path / "input.zarr", output_dir=tmp_path)

    monkeypatch.setattr(integ.sc.tl, "pca", lambda adata, **kwargs: adata.obsm.__setitem__("X_pca", np.ones((adata.n_obs, 5))))
    monkeypatch.setattr(integ, "_run_harmony", lambda adata, batch_key, use_rep: np.full((adata.n_obs, 5), 2.0))

    out, created = _run_integrations(adata, cfg, methods=["Harmony"], batch_key="batch")

    assert out is adata
    assert created == ["Unintegrated", "Harmony"]
    assert "Unintegrated" in adata.obsm
    assert "Harmony" in adata.obsm
    assert adata.obsm["Harmony"].shape == (adata.n_obs, 5)


def test_run_integrate_standard_path(monkeypatch, tmp_path):
    import scomnom.integrate as integ

    adata = synthetic_adata()
    save_mock = Mock()

    def fake_run_integrations(adata_in, cfg, *, methods, batch_key):
        adata_in.obsm["Unintegrated"] = np.ones((adata_in.n_obs, 5))
        adata_in.obsm["Harmony"] = np.full((adata_in.n_obs, 5), 2.0)
        return adata_in, ["Unintegrated", "Harmony"]

    monkeypatch.setattr(integ.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(integ.io_utils, "save_dataset", save_mock)
    monkeypatch.setattr(integ.plot_utils, "setup_scanpy_figs", lambda *args, **kwargs: None)
    monkeypatch.setattr(integ.plot_utils, "plot_integration_umaps", lambda *args, **kwargs: [])
    monkeypatch.setattr(integ.plot_utils, "persist_plot_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(integ.reporting, "generate_integration_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(integ, "_run_integrations", fake_run_integrations)
    monkeypatch.setattr(integ, "_resolve_scib_truth", lambda adata, cfg, round_id: ("label", "truth-label", None))
    monkeypatch.setattr(integ, "_select_best_embedding", lambda *args, **kwargs: "Harmony")
    monkeypatch.setattr(integ.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(integ.sc.tl, "umap", lambda adata: adata.obsm.__setitem__("X_umap", np.zeros((adata.n_obs, 2))))

    cfg = IntegrateConfig(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "results",
        output_name="adata.integrated.test",
        batch_key="batch",
        label_key="label",
        methods=["Harmony"],
        figure_formats=["png"],
    )

    out = run_integrate(cfg)

    assert out is adata
    assert "X_integrated" in adata.obsm
    assert np.array_equal(adata.obsm["X_integrated"], adata.obsm["Harmony"])
    assert adata.uns["integration"]["best_embedding"] == "Harmony"
    assert adata.uns["integration"]["available_embeddings"] == ["Unintegrated", "Harmony"]
    save_mock.assert_called_once()
    assert save_mock.call_args.args[1] == tmp_path / "results" / "adata.integrated.test.zarr"
