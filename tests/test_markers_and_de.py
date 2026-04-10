from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scomnom.composition_utils import _resolve_active_cluster_key
from scomnom.markers_and_de import (
    _run_namespace_for_round,
    _collect_pseudobulk_de_tables_from_dir,
    _collect_cell_contrast_tables_from_dir,
    _load_module_definitions,
    _compute_module_score_on_adata,
    run_enrichment_cluster,
    run_module_score,
)
from scomnom.annotation_utils import (
    _apply_gene_filters_to_expr,
    _apply_gene_filters_to_var_names,
    _prepare_decoupler_grouping,
    run_decoupler_for_round,
)
from scomnom.reporting import _de_report_summary_rows
from scomnom import reporting


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


@patch("scomnom.annotation_utils.io_utils.get_gene_type_map")
def test_apply_gene_filters_to_var_names_annotates_required_gene_metadata(mock_get_gene_type_map) -> None:
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.var_names = ["XIST", "RPL13", "CXCL8"]
    mock_get_gene_type_map.return_value = (
        {"XIST": "lncRNA", "RPL13": "protein_coding", "CXCL8": "protein_coding"},
        "biomart",
        {"XIST": "X", "RPL13": "19", "CXCL8": "4"},
        "biomart",
        {"XIST": "ENSG1", "RPL13": "ENSG2", "CXCL8": "ENSG3"},
    )

    keep_mask, info = _apply_gene_filters_to_var_names(
        adata,
        gene_filter=(
            "gene_chrom not in ['X','Y']",
            "gene_type != 'lncRNA'",
        ),
        resource_name="test",
    )

    assert keep_mask.tolist() == [False, True, True]
    assert adata.var["gene_type"].tolist() == ["lncRNA", "protein_coding", "protein_coding"]
    assert adata.var["gene_chrom"].tolist() == ["X", "19", "4"]
    assert info == {
        "gene_filter": (
            "gene_chrom not in ['X','Y']",
            "gene_type != 'lncRNA'",
        ),
        "n_genes_input": 3,
        "n_genes_retained": 2,
    }


@patch("scomnom.annotation_utils.io_utils.get_gene_type_map", side_effect=RuntimeError("biomart down"))
def test_apply_gene_filters_to_expr_raises_when_required_gene_metadata_unavailable(mock_get_gene_type_map) -> None:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.var_names = ["XIST", "CXCL8"]
    expr = pd.DataFrame(
        [[1.0], [2.0]],
        index=adata.var_names,
        columns=["C00"],
    )

    with pytest.raises(RuntimeError, match="unable to annotate gene metadata required for gene_filter"):
        _apply_gene_filters_to_expr(
            adata,
            expr,
            gene_filter=("gene_chrom not in ['X','Y']",),
            resource_name="test",
        )


def test_apply_gene_filters_to_de_stats_raises_on_invalid_query() -> None:
    from scomnom.markers_and_de import _apply_gene_filters_to_de_stats

    stats = pd.DataFrame({"stat": [1.0, 2.0]}, index=pd.Index(["A", "B"], name="gene"))
    gene_meta = pd.DataFrame(
        {"gene": ["A", "B"], "gene_type": ["protein_coding", "protein_coding"]}
    ).set_index("gene", drop=False)

    with pytest.raises(RuntimeError, match="gene_filter failed"):
        _apply_gene_filters_to_de_stats(
            stats,
            gene_meta=gene_meta,
            gene_filter=("gene_type ==",),
            resource_name="test",
        )


def test_generate_enrichment_cluster_report_writes_html(tmp_path: Path) -> None:
    fig_root = tmp_path / "figures"
    run_dir = fig_root / "png" / "enrichment_r5_archetypes_round1"
    run_dir.mkdir(parents=True)
    (run_dir / "r5_heatmap_top_hallmark_.png").write_bytes(b"")
    (run_dir / "r5_dotplot_top_hallmark_.png").write_bytes(b"")

    cfg = SimpleNamespace(round_id="r5_archetypes", condition_key="sex", gene_filter=("not gene.str.startswith('MT-')",))

    reporting.generate_enrichment_cluster_report(
        fig_root=fig_root,
        fmt="png",
        cfg=cfg,
        version="0.3.3",
        run_dir=run_dir,
    )

    html = (run_dir / "enrichment_report.html").read_text(encoding="utf-8")
    assert "scOmnom enrichment report" in html
    assert "r5_archetypes" in html


def test_generate_enrichment_de_report_writes_html(tmp_path: Path) -> None:
    fig_root = tmp_path / "figures"
    run_dir = fig_root / "png" / "enrichment_de_demo_round1"
    plot_dir = run_dir / "pseudobulk_DE" / "sex" / "female_vs_male"
    plot_dir.mkdir(parents=True)
    (plot_dir / "heatmap_top_up_.png").write_bytes(b"")

    cfg = SimpleNamespace(input_dir="/tmp/de_tables", de_decoupler_source="auto", gene_filter=())

    reporting.generate_enrichment_de_report(
        fig_root=fig_root,
        fmt="png",
        cfg=cfg,
        version="0.3.3",
        run_dir=run_dir,
    )

    html = (run_dir / "enrichment_de_report.html").read_text(encoding="utf-8")
    assert "scOmnom enrichment-from-DE report" in html
    assert "Pseudobulk De" in html


def test_generate_module_score_report_writes_html(tmp_path: Path) -> None:
    fig_root = tmp_path / "figures"
    run_dir = fig_root / "png" / "module_score_curated_programs_r5_archetypes_round1"
    (run_dir / "umaps").mkdir(parents=True)
    (run_dir / "module_score_summary_mean_z.png").write_bytes(b"")
    (run_dir / "umaps" / "module_score_module_a.png").write_bytes(b"")

    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.uns["active_cluster_round"] = "r5_archetypes"
    adata.uns["cluster_rounds"] = {
        "r5_archetypes": {
            "module_scores": {
                "curated_programs": {
                    "module_meta": pd.DataFrame(
                        {
                            "module": ["A", "B"],
                            "n_genes_input": [2, 3],
                            "n_genes_retained": [2, 2],
                        }
                    )
                }
            }
        }
    }
    cfg = SimpleNamespace(round_id="r5_archetypes", module_set_name="curated_programs", condition_key=None, module_score_method="scanpy")

    reporting.generate_module_score_report(
        fig_root=fig_root,
        fmt="png",
        cfg=cfg,
        version="0.3.3",
        adata=adata,
        run_dir=run_dir,
    )

    html = (run_dir / "module_score_report.html").read_text(encoding="utf-8")
    assert "scOmnom module-score report" in html
    assert "curated_programs" in html


def test_load_module_definitions_reads_gmt_and_txt(tmp_path: Path) -> None:
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("Inflammation\tna\tCXCL8\tIL1B\nFibrosis\tna\tCOL1A1\tCOL3A1\n")
    txt = tmp_path / "stress.txt"
    txt.write_text("ATF3\nDDIT3\n")

    got = _load_module_definitions([gmt, txt])

    assert got == {
        "Inflammation": ["CXCL8", "IL1B"],
        "Fibrosis": ["COL1A1", "COL3A1"],
        "stress": ["ATF3", "DDIT3"],
    }


def test_compute_module_score_on_adata_aucell_backend(monkeypatch, tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.zeros((4, 4)))
    adata.var_names = ["CXCL8", "IL1B", "LST1", "FCN1"]
    adata.obs_names = [f"cell{i}" for i in range(4)]
    adata.obs["leiden__r5_archetypes"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5_archetypes"] = [
        "C00: Macrophages",
        "C00: Macrophages",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.uns["active_cluster_round"] = "r5_archetypes"
    adata.uns["cluster_rounds"] = {
        "r5_archetypes": {
            "labels_obs_key": "leiden__r5_archetypes",
            "cluster_order": ["C00", "C01"],
        },
    }

    module_file = tmp_path / "mini.txt"
    module_file.write_text("CXCL8\nIL1B\n")

    class _FakeAucell:
        def __call__(self, data, net, tmin=1, raw=False, layer=None, verbose=False):
            data.obsm["score_aucell"] = pd.DataFrame(
                {"mini": [0.9, 0.7, 0.1, 0.2]},
                index=data.obs_names,
            )
            return None

    class _FakeMt:
        aucell = _FakeAucell()

    class _FakeDc:
        mt = _FakeMt()

    import sys
    monkeypatch.setitem(sys.modules, "decoupler", _FakeDc())

    cfg = SimpleNamespace(
        round_id="r5_archetypes",
        condition_key=None,
        module_files=(str(module_file),),
        module_set_name="mini",
        module_score_method="aucell",
        module_score_use_raw=False,
        module_score_layer=None,
        module_score_ctrl_size=50,
        module_score_n_bins=25,
        module_score_random_state=0,
        module_score_max_umaps=4,
    )

    payload, score_keys, rid = _compute_module_score_on_adata(adata, cfg)

    assert rid == "r5_archetypes"
    assert score_keys == ["module_score__r5_archetypes__mini__mini"]
    assert payload["method"] == "aucell"
    assert "module_score__r5_archetypes__mini__mini" in adata.obs.columns
    assert adata.obsm.get("score_aucell", None) is None


@patch("scomnom.markers_and_de.io_utils.save_dataset")
@patch("scomnom.markers_and_de.plot_utils.persist_plot_artifacts")
@patch("scomnom.markers_and_de.plot_utils.umap_by", return_value=[])
@patch("scomnom.markers_and_de.plot_utils.plot_module_score_summary_heatmap", return_value=[])
@patch("scomnom.markers_and_de.plot_utils.get_run_subdir", return_value="module_score_mini_r5_archetypes_round1")
@patch("scomnom.markers_and_de.plot_utils.setup_scanpy_figs")
@patch("scomnom.markers_and_de.io_utils.load_dataset")
def test_run_module_score_uses_requested_round_and_stores_payload(
    mock_load_dataset,
    _mock_setup_scanpy_figs,
    _mock_get_run_subdir,
    _mock_plot_heatmap,
    _mock_umap_by,
    _mock_persist_plot_artifacts,
    mock_save_dataset,
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(X=np.random.RandomState(0).rand(4, 4))
    adata.var_names = ["CXCL8", "IL1B", "LST1", "FCN1"]
    adata.obs["leiden__r5_archetypes"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5_archetypes"] = [
        "C00: Macrophages",
        "C00: Macrophages",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.uns["active_cluster_round"] = "r3_refined_idents"
    adata.uns["cluster_rounds"] = {
        "r3_refined_idents": {},
        "r5_archetypes": {
            "labels_obs_key": "leiden__r5_archetypes",
            "cluster_order": ["C00", "C01"],
        },
    }
    mock_load_dataset.return_value = adata

    module_file = tmp_path / "mini.txt"
    module_file.write_text("CXCL8\nIL1B\n")

    cfg = SimpleNamespace(
        logfile=None,
        output_dir=tmp_path / "out",
        figdir_name="figures",
        figure_formats=["png"],
        input_path=tmp_path / "input.zarr.tar.zst",
        output_name="adata.module_score_mini_r5_archetypes",
        save_h5ad=False,
        make_figures=True,
        round_id="r5_archetypes",
        condition_key=None,
        module_files=(str(module_file),),
        module_set_name="mini",
        module_score_method="scanpy",
        module_score_use_raw=False,
        module_score_layer=None,
        module_score_ctrl_size=2,
        module_score_n_bins=2,
        module_score_random_state=0,
        module_score_max_umaps=4,
    )

    got = run_module_score(cfg)

    assert got is adata
    round_payload = adata.uns["cluster_rounds"]["r5_archetypes"]["module_scores"]["mini"]
    assert round_payload["module_set_name"] == "mini"
    assert round_payload["round_id"] == "r5_archetypes"
    assert "module_score__r5_archetypes__mini__mini" in adata.obs.columns
    assert list(round_payload["summary_mean"].columns) == ["mini"]
    mock_save_dataset.assert_called_once()
    assert mock_save_dataset.call_args.args[1] == tmp_path / "out" / "adata.module_score_mini_r5_archetypes.zarr"


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


@patch("scomnom.annotation_utils._run_dorothea")
@patch("scomnom.annotation_utils._run_progeny")
@patch("scomnom.annotation_utils._run_msigdb")
@patch("scomnom.annotation_utils._store_cluster_pseudobulk")
def test_run_decoupler_for_round_drops_temporary_condition_group_key(
    mock_store_cluster_pseudobulk,
    _mock_run_msigdb,
    _mock_run_progeny,
    _mock_run_dorothea,
) -> None:
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
        run_decoupler=True,
        condition_key="sex:MASLD",
        decoupler_pseudobulk_agg="mean",
        decoupler_use_raw=False,
        msigdb_gene_sets=[],
        run_progeny=False,
        run_dorothea=False,
    )

    run_decoupler_for_round(adata, cfg, round_id="r4_subset_annotation")

    assert "__decoupler_group__r4_subset_annotation_sex.MASLD" not in adata.obs.columns


@patch("scomnom.annotation_utils._run_dorothea")
@patch("scomnom.annotation_utils._run_progeny")
@patch("scomnom.annotation_utils._run_msigdb")
@patch("scomnom.annotation_utils._store_cluster_pseudobulk")
def test_run_decoupler_for_round_drops_persisted_pseudobulk_store(
    mock_store_cluster_pseudobulk,
    _mock_run_msigdb,
    _mock_run_progeny,
    _mock_run_dorothea,
) -> None:
    adata = ad.AnnData(X=np.zeros((4, 2)))
    adata.var_names = ["CXCL8", "LST1"]
    adata.obs["leiden__r4_subset_annotation"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r4_subset_annotation"] = [
        "C00: Macrophages",
        "C00: Macrophages",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.uns["cluster_rounds"] = {
        "r4_subset_annotation": {
            "labels_obs_key": "leiden__r4_subset_annotation",
            "cluster_order": ["C00", "C01"],
            "decoupler": {},
        }
    }

    def _fake_store(in_adata, *, store_key, **kwargs):
        in_adata.uns[store_key] = {
            "genes": np.array(["CXCL8", "LST1"], dtype=object),
            "clusters": np.array(["C00", "C01"], dtype=object),
            "expr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }

    mock_store_cluster_pseudobulk.side_effect = _fake_store

    cfg = SimpleNamespace(
        run_decoupler=True,
        condition_key=None,
        decoupler_pseudobulk_agg="mean",
        decoupler_use_raw=False,
        msigdb_gene_sets=[],
        run_progeny=False,
        run_dorothea=False,
    )

    run_decoupler_for_round(adata, cfg, round_id="r4_subset_annotation")

    assert "pseudobulk__r4_subset_annotation" not in adata.uns
    assert "pseudobulk_store_key" not in adata.uns["cluster_rounds"]["r4_subset_annotation"]["decoupler"]
