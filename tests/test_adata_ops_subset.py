from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scomnom.adata_ops import (
    load_subset_mapping_tsv,
    subset_adata_by_cluster_mapping,
    subset_dataset_from_tsv,
)


def _make_test_adata() -> ad.AnnData:
    X = np.random.default_rng(0).normal(size=(6, 4))
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell{i}" for i in range(6)]
    adata.var_names = [f"g{i}" for i in range(4)]

    adata.obs["leiden__r0"] = pd.Categorical(["0", "0", "1", "1", "2", "2"])
    adata.obs["cluster_label__r0"] = pd.Categorical(
        [
            "C00: Immune",
            "C00: Immune",
            "C01: Stromal",
            "C01: Stromal",
            "C02: Endothelial",
            "C02: Endothelial",
        ]
    )
    adata.obs["cluster_label"] = adata.obs["cluster_label__r0"]

    adata.uns["cluster_rounds"] = {
        "r0": {
            "labels_obs_key": "leiden__r0",
            "cluster_key": "leiden",
            "cluster_sizes": {"0": 2, "1": 2, "2": 2},
            "cluster_order": ["0", "1", "2"],
            "cluster_display_map": {
                "0": "C00: Immune",
                "1": "C01: Stromal",
                "2": "C02: Endothelial",
            },
            "annotation": {"pretty_cluster_key": "cluster_label__r0"},
            "decoupler": {},
        }
    }
    adata.uns["active_cluster_round"] = "r0"
    return adata


def test_load_subset_mapping_tsv(tmp_path: Path) -> None:
    path = tmp_path / "mapping.tsv"
    path.write_text("C00\timmune\nC01\tstromal\nC02\timmune\n")

    out = load_subset_mapping_tsv(path)
    assert out == {"immune": ["C00", "C02"], "stromal": ["C01"]}


def test_load_subset_mapping_tsv_rejects_duplicate_cluster_assignment(tmp_path: Path) -> None:
    path = tmp_path / "mapping.tsv"
    path.write_text("C00\timmune\nC00\tstromal\n")

    with pytest.raises(ValueError, match="mapped to multiple subsets"):
        _ = load_subset_mapping_tsv(path)


def test_subset_adata_by_cluster_mapping_updates_round_sizes() -> None:
    adata = _make_test_adata()
    subset_map = {"immune": ["C00", "C02"], "stromal": ["C01"]}

    outputs, summary = subset_adata_by_cluster_mapping(adata, subset_map)
    assert set(outputs.keys()) == {"immune", "stromal"}
    assert outputs["immune"].n_obs == 4
    assert outputs["stromal"].n_obs == 2

    immune_round = outputs["immune"].uns["cluster_rounds"]["r0"]
    assert immune_round["cluster_sizes"] == {"0": 2, "2": 2}
    assert immune_round["cluster_order"] == ["0", "2"]
    assert immune_round["cluster_display_map"] == {"0": "C00: Immune", "2": "C02: Endothelial"}

    got = summary.set_index("subset_name")["n_cells"].to_dict()
    assert got == {"immune": 4, "stromal": 2}


def test_subset_dataset_from_tsv_saves_outputs(tmp_path: Path, monkeypatch) -> None:
    adata = _make_test_adata()
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("C00\timmune\nC01\tstromal\nC02\timmune\n")

    calls: list[tuple[str, str]] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        calls.append((str(out_path), str(fmt)))

    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)

    out_paths, summary = subset_dataset_from_tsv(
        adata,
        mapping,
        output_root=tmp_path / "results",
        output_format="zarr",
    )

    assert set(out_paths.keys()) == {"immune", "stromal"}
    assert len(calls) == 2
    assert all(path.endswith(".zarr") for path, _ in calls)
    assert summary["n_cells"].sum() == 6
    assert (tmp_path / "results" / "tables" / "adata__subset_summary.tsv").exists()
