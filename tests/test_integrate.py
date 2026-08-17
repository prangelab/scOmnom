import numpy as np
import pandas as pd
import scanpy as sc
from unittest.mock import Mock

from scomnom.config import IntegrateConfig
from scomnom.integrate import _run_integrations, run_integrate


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
