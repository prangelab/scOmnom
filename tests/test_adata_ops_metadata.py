from __future__ import annotations

from pathlib import Path
import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scomnom as om

from scomnom.adata_ops import add_obs_metadata, import_dataset_obs_metadata


def _make_metadata_test_adata() -> ad.AnnData:
    X = np.random.default_rng(0).normal(size=(6, 4))
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell{i}" for i in range(6)]
    adata.var_names = [f"g{i}" for i in range(4)]
    adata.obs["sample_id"] = pd.Categorical(["S1", "S1", "S2", "S2", "S3", "S3"])
    return adata


def test_add_obs_metadata_by_obs_names_dataframe_index() -> None:
    adata = _make_metadata_test_adata()
    metadata = pd.DataFrame(
        {"score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        index=adata.obs_names,
    )

    summary = add_obs_metadata(adata, metadata, metadata_key=None, obs_key=None, columns=("score",))

    assert "score" in adata.obs
    assert np.allclose(adata.obs["score"].to_numpy(dtype=float), metadata["score"].to_numpy(dtype=float))
    assert summary.loc[0, "obs_key"] == "obs_names"
    assert summary.loc[0, "metadata_key"] == "index"


def test_add_obs_metadata_by_sample_id_uses_safe_categories() -> None:
    adata = _make_metadata_test_adata()
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "condition": ["ctrl", "case", "case"],
            "patient": ["P1", "P2", "P3"],
        }
    )

    summary = add_obs_metadata(
        adata,
        metadata,
        metadata_key="sample_id",
        obs_key="sample_id",
        columns=("condition", "patient"),
    )

    assert "condition" in adata.obs
    assert "patient" in adata.obs
    assert set(adata.obs["condition"].astype(str)) == {"ctrl", "case"}
    assert set(adata.obs["patient"].astype(str)) == {"P1", "P2", "P3"}
    assert summary.loc[0, "n_imported_columns"] == 2


def test_add_obs_metadata_rejects_duplicate_metadata_keys() -> None:
    adata = _make_metadata_test_adata()
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "condition": ["a", "b", "c"],
        }
    )

    with pytest.raises(ValueError, match="must be unique"):
        add_obs_metadata(
            adata,
            metadata,
            metadata_key="sample_id",
            obs_key="sample_id",
            columns=("condition",),
        )


def test_import_dataset_obs_metadata_saves_output(tmp_path: Path, monkeypatch) -> None:
    adata = _make_metadata_test_adata()
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "condition": ["ctrl", "case", "case"],
        }
    )
    metadata_path = tmp_path / "meta.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)

    saved: list[tuple[str, str]] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr", archive=True):
        saved.append((str(out_path), str(fmt), bool(archive)))

    internal_ops = importlib.import_module("scomnom.adata_ops")
    monkeypatch.setattr(internal_ops, "save_dataset", _fake_save_dataset)

    out_paths, summary = import_dataset_obs_metadata(
        adata,
        metadata_path,
        output_root=tmp_path / "results",
        metadata_key="sample_id",
        obs_key="sample_id",
        columns=("condition",),
    )

    assert out_paths["metadata_imported"] == tmp_path / "results" / "adata.metadata_imported.zarr.tar.zst"
    assert saved == [(str(tmp_path / "results" / "adata.metadata_imported.zarr.tar.zst"), "zarr", True)]
    assert (tmp_path / "results" / "tables" / "adata.metadata_imported__metadata_import_summary.tsv").exists()
    assert summary.loc[0, "imported_columns"] == "condition"
    assert "condition" in adata.obs


def test_import_dataset_obs_metadata_infers_results_output_dir_for_metadata_import() -> None:
    from scomnom.config import AdataOpsConfig

    cfg = AdataOpsConfig(
        input_path=Path("/tmp/project/adata.clustered.annotated.zarr.tar.zst"),
        operation="metadata_import",
        metadata_file=Path("/tmp/project/meta.tsv"),
        metadata_key="sample_id",
    )

    assert cfg.resolved_output_dir == Path("/tmp/project/results").resolve()


def test_public_api_add_obs_metadata_wrapper_updates_alias() -> None:
    assert om.adata_ops.add_obs_metadata is not None
