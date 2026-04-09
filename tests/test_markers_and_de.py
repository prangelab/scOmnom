from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scomnom.composition_utils import _resolve_active_cluster_key
from scomnom.markers_and_de import (
    _run_namespace_for_round,
    _collect_pseudobulk_de_tables_from_dir,
    _collect_cell_contrast_tables_from_dir,
    run_enrichment_cluster,
)
from scomnom.annotation_utils import (
    _apply_gene_filters_to_expr,
    _apply_gene_filters_to_var_names,
    _prepare_decoupler_grouping,
    run_decoupler_for_round,
)
from scomnom.reporting import _de_report_summary_rows


def _make_adata_with_round(round_id: str | None) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    if round_id is not None:
        adata.uns["active_cluster_round"] = round_id
    return adata


def test_run_namespace_for_round_uses_explicit_round_id() -> None:
    adata = _make_adata_with_round("r3_refined_idents")
    got = _run_namespace_for_round(adata, prefix="de", round_id="r5_archetypes")
    assert got == "de_r5_archetypes"


def test_run_namespace_for_round_falls_back_to_active_round() -> None:
    adata = _make_adata_with_round("r4_subset_annotation")
    got = _run_namespace_for_round(adata, prefix="markers", round_id=None)
    assert got == "markers_r4_subset_annotation"


def test_run_namespace_for_round_without_round_uses_prefix_only() -> None:
    adata = _make_adata_with_round(None)
    got = _run_namespace_for_round(adata, prefix="da", round_id=None)
    assert got == "da"


def test_resolve_active_cluster_key_prefers_round_labels_obs_key() -> None:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.obs["leiden"] = ["0", "1"]
    adata.obs["leiden__r5_archetypes"] = ["0", "0"]
    adata.uns["active_cluster_round"] = "r3_refined_idents"
    adata.uns["cluster_rounds"] = {
        "r3_refined_idents": {
            "cluster_key": "leiden",
            "labels_obs_key": "leiden",
        },
        "r5_archetypes": {
            "cluster_key": "leiden",
            "labels_obs_key": "leiden__r5_archetypes",
        },
    }

    got = _resolve_active_cluster_key(adata, round_id="r5_archetypes")
    assert got == "leiden__r5_archetypes"


@patch("scomnom.markers_and_de.io_utils.save_dataset")
@patch("scomnom.markers_and_de.plot_utils.persist_plot_artifacts")
@patch("scomnom.markers_and_de.plot_utils.plot_decoupler_all_styles", return_value=[])
@patch("scomnom.markers_and_de.plot_utils.get_run_subdir", return_value="enrichment_r5_archetypes_round1")
@patch("scomnom.markers_and_de.plot_utils.setup_scanpy_figs")
@patch("scomnom.markers_and_de.run_decoupler_for_round")
@patch("scomnom.markers_and_de.io_utils.load_dataset")
def test_run_enrichment_cluster_uses_requested_round_id(
    mock_load_dataset,
    mock_run_decoupler_for_round,
    _mock_setup_scanpy_figs,
    _mock_get_run_subdir,
    _mock_plot_decoupler_all_styles,
    _mock_persist_plot_artifacts,
    mock_save_dataset,
) -> None:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.uns["active_cluster_round"] = "r3_refined_idents"
    adata.uns["cluster_rounds"] = {
        "r3_refined_idents": {"decoupler": {}},
        "r5_archetypes": {"decoupler": {}},
    }
    mock_load_dataset.return_value = adata

    cfg = SimpleNamespace(
        logfile=None,
        output_dir=Path("/tmp/enrichment-out"),
        figdir_name="figures",
        figure_formats=["png"],
        input_path=Path("/tmp/input.zarr.tar.zst"),
        output_name="adata.enrichment_r5_archetypes",
        save_h5ad=False,
        make_figures=True,
        regenerate_figures=False,
        round_id="r5_archetypes",
        decoupler_bar_top_n_up=None,
        decoupler_bar_top_n_down=None,
        decoupler_bar_split_signed=True,
    )

    got = run_enrichment_cluster(cfg)

    assert got is adata
    mock_run_decoupler_for_round.assert_called_once_with(adata, cfg, round_id="r5_archetypes")
    mock_save_dataset.assert_called_once()
    assert mock_save_dataset.call_args.args[0] is adata
    assert mock_save_dataset.call_args.args[1] == Path("/tmp/enrichment-out/adata.enrichment_r5_archetypes.zarr")
    assert mock_save_dataset.call_args.kwargs["fmt"] == "zarr"


def test_collect_pseudobulk_de_tables_from_dir_reads_exported_csvs(tmp_path: Path) -> None:
    cond_dir = tmp_path / "condition_within_cluster__sex"
    cond_dir.mkdir(parents=True)
    pd.DataFrame({"gene": ["CXCL8"], "stat": [3.0]}).to_csv(
        cond_dir / "condition_within_cluster__C00__female_vs_male.csv",
        index=False,
    )

    got = _collect_pseudobulk_de_tables_from_dir(tmp_path)

    assert sorted(got.keys()) == ["sex"]
    assert sorted(got["sex"].keys()) == ["female_vs_male"]
    assert sorted(got["sex"]["female_vs_male"].keys()) == ["C00"]


def test_collect_cell_contrast_tables_from_dir_prefers_combined_csv(tmp_path: Path) -> None:
    pair_dir = tmp_path / "sex_female_vs_male_DE" / "cluster__C00__female_vs_male"
    pair_dir.mkdir(parents=True)
    pd.DataFrame({"gene": ["CXCL8"], "stat": [1.0]}).to_csv(
        pair_dir / "cluster__C00__female_vs_male__combined.csv",
        index=False,
    )
    pd.DataFrame({"gene": ["CXCL8"], "cell_wilcoxon_logfc": [2.0]}).to_csv(
        pair_dir / "cluster__C00__female_vs_male__wilcoxon.csv",
        index=False,
    )

    got = _collect_cell_contrast_tables_from_dir(tmp_path)

    assert sorted(got.keys()) == ["sex"]
    assert sorted(got["sex"].keys()) == ["female_vs_male"]
    df = got["sex"]["female_vs_male"]["C00"]
    assert list(df.columns) == ["gene", "stat"]


def test_apply_gene_filters_to_expr_filters_pseudobulk_genes() -> None:
    import pandas as pd

    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.var_names = ["MT-CO1", "RPL13", "CXCL8"]
    expr = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    expr_frame = pd.DataFrame(
        expr,
        index=adata.var_names,
        columns=["C00", "C01"],
    )

    got, info = _apply_gene_filters_to_expr(
        adata,
        expr_frame,
        gene_filter=(
            "not gene.str.startswith('MT-')",
            "expr=not gene.str.startswith('RPL')",
        ),
        resource_name="test",
    )

    assert list(got.index) == ["CXCL8"]
    assert info == {
        "gene_filter": (
            "not gene.str.startswith('MT-')",
            "not gene.str.startswith('RPL')",
        ),
        "n_genes_input": 3,
        "n_genes_retained": 1,
    }


def test_apply_gene_filters_to_var_names_filters_de_genes() -> None:
    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.var_names = ["MT-CO1", "RPL13", "CXCL8", "LST1"]
    adata.var["gene_biotype"] = ["protein_coding", "protein_coding", "protein_coding", "lncRNA"]

    keep_mask, info = _apply_gene_filters_to_var_names(
        adata,
        gene_filter=(
            "not gene.str.startswith('MT-')",
            "expr=gene_biotype == 'protein_coding'",
        ),
        resource_name="test",
    )

    assert keep_mask.tolist() == [False, True, True, False]
    assert info == {
        "gene_filter": (
            "not gene.str.startswith('MT-')",
            "gene_biotype == 'protein_coding'",
        ),
        "n_genes_input": 4,
        "n_genes_retained": 2,
    }


def test_de_report_summary_rows_include_gene_filter_metadata() -> None:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.uns["markers_and_de"] = {
        "gene_filter": ["not gene.str.startswith('MT-')"],
        "gene_filter_n_genes_input": 20000,
        "gene_filter_n_genes_retained": 18321,
    }
    cfg = SimpleNamespace(gene_filter=("not gene.str.startswith('MT-')",))

    rows = _de_report_summary_rows(rel_imgs=[Path("a.png"), Path("b.png")], cfg=cfg, adata=adata)

    assert rows == [
        ("n_plots_total", "2"),
        ("gene_filter", "not gene.str.startswith('MT-')"),
        ("gene_filter_n_genes_input", "20000"),
        ("gene_filter_n_genes_retained", "18321"),
    ]


def test_prepare_decoupler_grouping_builds_cluster_condition_groups() -> None:
    adata = ad.AnnData(X=np.zeros((4, 2)))
    adata.obs["leiden__r5_archetypes"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5_archetypes"] = [
        "C00: Macrophages",
        "C00: Macrophages",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.obs["sex"] = pd.Categorical(["female", "male", "female", "male"], categories=["female", "male"])
    adata.uns["cluster_rounds"] = {
        "r5_archetypes": {
            "labels_obs_key": "leiden__r5_archetypes",
            "cluster_order": ["C00", "C01"],
        }
    }

    got = _prepare_decoupler_grouping(
        adata,
        round_id="r5_archetypes",
        labels_obs_key="leiden__r5_archetypes",
        condition_key="sex",
    )

    assert got["group_key"] == "__decoupler_group__r5_archetypes_sex"
    assert got["cleanup_key"] == "__decoupler_group__r5_archetypes_sex"
    assert got["condition_key"] == "sex"
    assert got["display_map"] == {
        "C00__female": "C00: Macrophages | female",
        "C00__male": "C00: Macrophages | male",
        "C01__female": "C01: T cells | female",
        "C01__male": "C01: T cells | male",
    }
    assert got["display_order"] == [
        "C00: Macrophages | female",
        "C00: Macrophages | male",
        "C01: T cells | female",
        "C01: T cells | male",
    ]


@patch("scomnom.annotation_utils._store_cluster_pseudobulk")
def test_run_decoupler_for_round_drops_temporary_condition_group_key(mock_store_cluster_pseudobulk) -> None:
    adata = ad.AnnData(X=np.zeros((4, 2)))
    adata.var_names = ["CXCL8", "LST1"]
    adata.obs["leiden__r4_subset_annotation"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r4_subset_annotation"] = [
        "C00: Macrophages",
        "C00: Macrophages",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.obs["sex"] = ["female", "male", "female", "male"]
    adata.obs["MASLD"] = ["yes", "yes", "no", "no"]
    adata.uns["cluster_rounds"] = {
        "r4_subset_annotation": {
            "labels_obs_key": "leiden__r4_subset_annotation",
            "cluster_order": ["C00", "C01"],
            "decoupler": {},
        }
    }

    cfg = SimpleNamespace(
        condition_key="sex:MASLD",
        decoupler_pseudobulk_agg="mean",
        decoupler_use_raw=False,
        msigdb_gene_sets=[],
        run_progeny=False,
        run_dorothea=False,
    )

    run_decoupler_for_round(adata, cfg, round_id="r4_subset_annotation")

    assert "__decoupler_group__r4_subset_annotation_sex.MASLD" not in adata.obs.columns
