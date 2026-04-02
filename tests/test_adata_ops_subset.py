from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scomnom.adata_ops import (
    _dataset_stem_for_outputs,
    annotation_merge_datasets,
    load_subset_mapping_tsv,
    rename_dataset_idents,
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


def _make_child_subset(
    obs_names: list[str],
    labels_obs_key: str,
    pretty_key: str,
    labels: list[str],
    pretty_labels: list[str],
    round_id: str,
) -> ad.AnnData:
    X = np.random.default_rng(1).normal(size=(len(obs_names), 4))
    adata = ad.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = [f"g{i}" for i in range(4)]
    adata.obs[labels_obs_key] = pd.Categorical(labels)
    adata.obs[pretty_key] = pd.Categorical(pretty_labels)
    adata.obs["cluster_label"] = adata.obs[pretty_key]
    adata.uns["cluster_rounds"] = {
        round_id: {
            "labels_obs_key": labels_obs_key,
            "cluster_key": "leiden",
            "cluster_sizes": dict(pd.Series(labels).value_counts()),
            "cluster_order": sorted(set(labels)),
            "cluster_display_map": {
                str(lbl): str(pretty)
                for lbl, pretty in zip(labels, pretty_labels)
            },
            "annotation": {"pretty_cluster_key": pretty_key},
            "round_type": "manual_rename",
            "decoupler": {},
        }
    }
    adata.uns["active_cluster_round"] = round_id
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


def test_subset_adata_by_cluster_mapping_rejects_missing_requested_clusters() -> None:
    adata = _make_test_adata()
    subset_map = {"immune": ["C00", "C99"]}

    with pytest.raises(ValueError, match="Requested Cnn clusters were not found"):
        _ = subset_adata_by_cluster_mapping(adata, subset_map)


def test_dataset_stem_for_outputs_normalizes_archives() -> None:
    assert _dataset_stem_for_outputs(Path("foo.zarr.tar.zst")) == "foo"
    assert _dataset_stem_for_outputs(Path("foo.zarr")) == "foo"
    assert _dataset_stem_for_outputs(Path("foo.h5ad")) == "foo"
    assert _dataset_stem_for_outputs(Path("foo.bar")) == "foo"


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


def test_subset_dataset_from_tsv_uses_archive_stem_for_output_names(tmp_path: Path, monkeypatch) -> None:
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("C00\timmune\n")

    adata = _make_test_adata()

    def _fake_load_dataset(_path):
        return adata

    calls: list[tuple[str, str]] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        calls.append((str(out_path), str(fmt)))

    monkeypatch.setattr("scomnom.adata_ops.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)

    _ = subset_dataset_from_tsv(
        tmp_path / "input.zarr.tar.zst",
        mapping,
        output_root=tmp_path / "results",
        output_format="zarr",
    )

    assert any("input__subset_immune.zarr" in path for path, _ in calls)


def test_rename_dataset_idents_creates_new_round_and_saves_output(tmp_path: Path, monkeypatch) -> None:
    adata = _make_test_adata()
    mapping = tmp_path / "rename.tsv"
    mapping.write_text("C00\tT cells\nC01\tStromal refined\n")

    saved: list[tuple[str, str]] = []
    plotted: list[str] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        saved.append((str(out_path), str(fmt)))

    def _fake_emit_rename_round_plots(in_adata, *, output_root, round_id):
        plotted.append(str(round_id))

    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)
    monkeypatch.setattr("scomnom.adata_ops._emit_rename_round_plots", _fake_emit_rename_round_plots)

    out_paths, summary = rename_dataset_idents(
        adata,
        mapping,
        output_root=tmp_path / "results",
        output_format="zarr",
        round_name="refined_idents",
    )

    assert set(out_paths.keys()) == {"renamed"}
    assert saved == [(str(tmp_path / "results" / "adata.renamed.zarr"), "zarr")]
    assert len(plotted) == 1

    new_round_id = plotted[0]
    assert new_round_id.endswith("_refined_idents")
    assert adata.uns["active_cluster_round"] == new_round_id

    pretty_key = adata.uns["cluster_rounds"][new_round_id]["annotation"]["pretty_cluster_key"]
    assert pretty_key == f"cluster_label__{new_round_id}"
    renamed = adata.obs[pretty_key].astype(str).tolist()
    assert renamed[:2] == ["C00: T cells", "C00: T cells"]
    assert renamed[2:4] == ["C01: Stromal refined", "C01: Stromal refined"]

    assert summary.loc[0, "new_round_id"] == new_round_id
    assert summary.loc[0, "parent_round_id"] == "r0"
    assert summary.loc[0, "n_renamed_clusters"] == 2


def test_rename_dataset_idents_can_collapse_same_labels(tmp_path: Path, monkeypatch) -> None:
    adata = _make_test_adata()
    mapping = tmp_path / "rename.tsv"
    mapping.write_text("C00\tImmune archetype\nC01\tStromal archetype\nC02\tImmune archetype\n")

    saved: list[tuple[str, str]] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        saved.append((str(out_path), str(fmt)))

    def _fake_emit_rename_round_plots(in_adata, *, output_root, round_id):
        return None

    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)
    monkeypatch.setattr("scomnom.adata_ops._emit_rename_round_plots", _fake_emit_rename_round_plots)

    out_paths, summary = rename_dataset_idents(
        adata,
        mapping,
        output_root=tmp_path / "results",
        output_format="zarr",
        round_name="archetypes",
        collapse_same_labels=True,
    )

    assert set(out_paths.keys()) == {"renamed"}
    assert saved == [(str(tmp_path / "results" / "adata.renamed.zarr"), "zarr")]

    new_round_id = str(summary.loc[0, "new_round_id"])
    round_info = adata.uns["cluster_rounds"][new_round_id]
    labels_obs_key = round_info["labels_obs_key"]
    pretty_key = round_info["annotation"]["pretty_cluster_key"]

    assert labels_obs_key == f"leiden__{new_round_id}"
    assert bool(round_info["manual_rename"]["collapse_same_labels"]) is True
    assert set(adata.obs[labels_obs_key].astype(str).unique()) == {"0", "1"}
    assert list(adata.obs[pretty_key].astype(str).cat.categories) == [
        "C00: Immune archetype",
        "C01: Stromal archetype",
    ]
    assert round_info["cluster_display_map"] == {
        "0": "C00: Immune archetype",
        "1": "C01: Stromal archetype",
    }
    assert bool(summary.loc[0, "collapse_same_labels"]) is True


def test_rename_dataset_idents_can_leave_active_round_unchanged(tmp_path: Path, monkeypatch) -> None:
    adata = _make_test_adata()
    mapping = tmp_path / "rename.tsv"
    mapping.write_text("C00\tT cells\n")

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        return None

    def _fake_emit_rename_round_plots(in_adata, *, output_root, round_id):
        return None

    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)
    monkeypatch.setattr("scomnom.adata_ops._emit_rename_round_plots", _fake_emit_rename_round_plots)

    _, summary = rename_dataset_idents(
        adata,
        mapping,
        output_root=tmp_path / "results",
        output_format="zarr",
        round_name="inactive_labels",
        set_active=False,
    )

    new_round_id = str(summary.loc[0, "new_round_id"])
    assert adata.uns["active_cluster_round"] == "r0"
    assert new_round_id != "r0"
    assert adata.obs["cluster_label"].astype(str).tolist() == adata.obs["cluster_label__r0"].astype(str).tolist()


def test_annotation_merge_creates_new_subset_annotation_round(tmp_path: Path, monkeypatch) -> None:
    parent = _make_test_adata()
    child = _make_child_subset(
        ["cell0", "cell1"],
        "leiden__r1",
        "cluster_label__r1",
        ["0", "1"],
        ["C00: T cell", "C01: B cell"],
        "r1_manual_rename",
    )

    def _fake_load_dataset(path):
        path = Path(path)
        if path.name == "parent.zarr":
            return parent
        if path.name == "child1.zarr":
            return child
        raise AssertionError(f"Unexpected path {path}")

    calls: list[tuple[str, str]] = []

    def _fake_save_dataset(in_adata, out_path, fmt="zarr"):
        calls.append((str(out_path), str(fmt)))

    monkeypatch.setattr("scomnom.adata_ops.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("scomnom.adata_ops.save_dataset", _fake_save_dataset)

    out_paths, summary = annotation_merge_datasets(
        tmp_path / "parent.zarr",
        child_paths=[tmp_path / "child1.zarr"],
        output_root=tmp_path / "results",
        output_format="zarr",
    )

    assert set(out_paths.keys()) == {"merged"}
    assert len(calls) == 1
    assert summary.loc[0, "n_children"] == 1

    new_round_id = parent.uns["active_cluster_round"]
    assert new_round_id.endswith("_subset_annotation")
    round_info = parent.uns["cluster_rounds"][new_round_id]
    assert round_info["round_type"] == "subset_annotation"
    assert round_info["subset_annotation"]["base_round_id"] == "r0"
    assert round_info["annotation"]["pretty_cluster_key"] == f"cluster_label__{new_round_id}"

    merged_labels = parent.obs[f"cluster_label__{new_round_id}"].astype(str).to_dict()
    assert merged_labels["cell2"] == "C00: Stromal"
    assert merged_labels["cell1"] == "C01: B cell"
    assert merged_labels["cell0"] == "C02: T cell"


def test_annotation_merge_updates_existing_subset_annotation_round(tmp_path: Path, monkeypatch) -> None:
    parent = _make_test_adata()
    child1 = _make_child_subset(
        ["cell0", "cell1"],
        "leiden__r1",
        "cluster_label__r1",
        ["0", "1"],
        ["C00: T cell", "C01: B cell"],
        "r1_manual_rename",
    )
    child2 = _make_child_subset(
        ["cell4", "cell5"],
        "leiden__r2",
        "cluster_label__r2",
        ["0", "0"],
        ["C00: Endothelial refined", "C00: Endothelial refined"],
        "r2_manual_rename",
    )

    datasets = {
        "parent.zarr": parent,
        "child1.zarr": child1,
        "child2.zarr": child2,
    }

    def _fake_load_dataset(path):
        return datasets[Path(path).name]

    monkeypatch.setattr("scomnom.adata_ops.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("scomnom.adata_ops.save_dataset", lambda *args, **kwargs: None)

    annotation_merge_datasets(
        tmp_path / "parent.zarr",
        child_paths=[tmp_path / "child1.zarr"],
        output_root=tmp_path / "results",
        output_format="zarr",
    )
    created_round_id = str(parent.uns["active_cluster_round"])

    annotation_merge_datasets(
        tmp_path / "parent.zarr",
        child_paths=[tmp_path / "child2.zarr"],
        output_root=tmp_path / "results",
        output_format="zarr",
        target_round_id=created_round_id,
        update_existing_round=True,
    )

    updated = parent.obs[f"cluster_label__{created_round_id}"].astype(str).to_dict()
    assert updated["cell1"].endswith("B cell")
    assert updated["cell5"].endswith("Endothelial refined")
    assert parent.uns["cluster_rounds"][created_round_id]["round_type"] == "subset_annotation"


def test_annotation_merge_rejects_overlapping_children(tmp_path: Path, monkeypatch) -> None:
    parent = _make_test_adata()
    child1 = _make_child_subset(
        ["cell0", "cell1"],
        "leiden__r1",
        "cluster_label__r1",
        ["0", "1"],
        ["C00: T cell", "C01: B cell"],
        "r1_manual_rename",
    )
    child2 = _make_child_subset(
        ["cell1", "cell4"],
        "leiden__r2",
        "cluster_label__r2",
        ["0", "1"],
        ["C00: X", "C01: Y"],
        "r2_manual_rename",
    )

    def _fake_load_dataset(path):
        return {
            "parent.zarr": parent,
            "child1.zarr": child1,
            "child2.zarr": child2,
        }[Path(path).name]

    monkeypatch.setattr("scomnom.adata_ops.load_dataset", _fake_load_dataset)

    with pytest.raises(ValueError, match="overlap"):
        annotation_merge_datasets(
            tmp_path / "parent.zarr",
            child_paths=[tmp_path / "child1.zarr", tmp_path / "child2.zarr"],
            output_root=tmp_path / "results",
            output_format="zarr",
        )


def test_annotation_merge_emits_plots_in_merge_annotation_subfolder(tmp_path: Path, monkeypatch) -> None:
    parent = _make_test_adata()
    parent.obsm["X_umap"] = np.random.default_rng(2).normal(size=(parent.n_obs, 2))
    parent.obsm["X_pca"] = np.random.default_rng(3).normal(size=(parent.n_obs, 3))
    parent.obs["sample"] = pd.Categorical(["s1", "s1", "s1", "s2", "s2", "s2"])
    parent.obs["n_genes_by_counts"] = [100, 120, 110, 90, 95, 105]
    parent.obs["total_counts"] = [1000, 1100, 1050, 900, 950, 980]
    parent.obs["pct_counts_mt"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8]
    child = _make_child_subset(
        ["cell0", "cell1"],
        "leiden__r1",
        "cluster_label__r1",
        ["0", "1"],
        ["C00: T cell", "C01: B cell"],
        "r1_manual_rename",
    )

    def _fake_load_dataset(path):
        return {"parent.zarr": parent, "child1.zarr": child}[Path(path).name]

    monkeypatch.setattr("scomnom.adata_ops.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("scomnom.adata_ops.save_dataset", lambda *args, **kwargs: None)

    setup_calls: list[Path] = []
    persisted: list[object] = []
    plot_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "scomnom.adata_ops.plot_utils.setup_scanpy_figs",
        lambda figdir, formats=None: setup_calls.append(Path(figdir)),
    )
    monkeypatch.setattr(
        "scomnom.adata_ops.plot_utils.persist_plot_artifacts",
        lambda artifacts: persisted.extend(list(artifacts)),
    )

    def _record(name):
        def _inner(*args, **kwargs):
            plot_calls.append((name, str(kwargs.get("figdir", args[-1]))))
            return [name]
        return _inner

    monkeypatch.setattr("scomnom.adata_ops.plot_utils.plot_cluster_umaps", _record("umap"))
    monkeypatch.setattr("scomnom.adata_ops.plot_utils.umap_by_two_legend_styles", _record("pretty_umap"))
    monkeypatch.setattr("scomnom.adata_ops.plot_utils.plot_cluster_sizes", _record("sizes"))
    monkeypatch.setattr("scomnom.adata_ops.plot_utils.plot_cluster_qc_summary", _record("qc"))
    monkeypatch.setattr("scomnom.adata_ops.plot_utils.plot_cluster_silhouette_by_cluster", _record("silhouette"))
    monkeypatch.setattr("scomnom.adata_ops.plot_utils.plot_cluster_batch_composition", _record("batch"))

    annotation_merge_datasets(
        tmp_path / "parent.zarr",
        child_paths=[tmp_path / "child1.zarr"],
        output_root=tmp_path / "results",
        output_format="zarr",
    )

    round_id = str(parent.uns["active_cluster_round"])
    assert setup_calls == [tmp_path / "results" / "figures"]
    assert len(persisted) == 6
    assert all(figdir == f"merge-annotation/{round_id}/clustering" for _, figdir in plot_calls)
