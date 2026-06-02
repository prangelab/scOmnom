from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import scomnom.annotation_utils as au

md_mod = importlib.import_module("scomnom.markers_and_de")

from scomnom.composition_utils import _resolve_active_cluster_key
from scomnom.markers_and_de import (
    _run_namespace_for_round,
    _collect_pseudobulk_de_tables_from_dir,
    _collect_cell_contrast_tables_from_dir,
    _build_stats_matrix_from_tables,
    _liana_family_label,
    _load_de_enrichment_payload_from_tables,
    _load_module_definitions,
    _liana_plot_color_map,
    _prepare_liana_plot_df,
    _prune_uns_de,
    _compute_module_score_on_adata,
    _write_de_enrichment_tables,
    run_enrichment_cluster,
    run_module_score,
    run_liana_ccc,
    run_mebocost_ccc,
    run_nichenet_ccc,
)
from scomnom.annotation_utils import (
    _apply_gene_filters_to_expr,
    _apply_gene_filters_to_var_names,
    _merge_msigdb_decoupler_and_gsea,
    _prepare_decoupler_grouping,
    _run_msigdb_gsea_from_stats,
    run_decoupler_for_round,
)
from scomnom.reporting import _de_report_summary_rows
from scomnom import reporting


def test_run_msigdb_gsea_from_stats_returns_long_results(monkeypatch, tmp_path: Path) -> None:
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_TEST\tna\tG1\tG2\tG3\n", encoding="utf-8")

    class _FakePrerankRes:
        def __init__(self):
            self.res2d = pd.DataFrame(
                {
                    "Term": ["HALLMARK_TEST"],
                    "ES": [0.6],
                    "NES": [1.8],
                    "NOM p-val": [0.01],
                    "FDR q-val": [0.03],
                    "Lead_genes": ["G1;G2"],
                }
            )

    class _FakeGseapy:
        @staticmethod
        def prerank(**kwargs):
            return _FakePrerankRes()

    import sys

    monkeypatch.setitem(sys.modules, "gseapy", _FakeGseapy)
    monkeypatch.setattr(
        "scomnom.annotation_utils.resolve_msigdb_gene_sets",
        lambda gene_sets: ([str(gmt)], ["HALLMARK"], "vX"),
    )

    stats = pd.DataFrame({"C00": [3.0, 1.0, -1.0]}, index=["G1", "G2", "G3"])
    cfg = SimpleNamespace(
        msigdb_gene_sets=["HALLMARK"],
        gsea_min_size=1,
        gsea_max_size=500,
        gsea_eps=1e-10,
        random_state=42,
        joint_enrichment_leading_edge_top_n=5,
    )

    payload = _run_msigdb_gsea_from_stats(stats, cfg, input_label="demo")

    assert payload is not None
    results = payload["results"]
    assert list(results.columns[:7]) == ["cluster", "pathway", "gmt", "NES", "ES", "pval", "padj"]
    assert results.loc[0, "cluster"] == "C00"
    assert results.loc[0, "pathway"] == "HALLMARK_TEST"
    assert results.loc[0, "gmt"] == "HALLMARK"
    assert results.loc[0, "leading_edge_n"] == 2
    assert "G1" in results.loc[0, "leading_edge_preview"]


def test_run_msigdb_gsea_from_stats_runs_each_selected_set_separately(monkeypatch, tmp_path: Path) -> None:
    hallmark = tmp_path / "hallmark.gmt"
    hallmark.write_text("HALLMARK_TEST\tna\tG1\tG2\tG3\n", encoding="utf-8")
    reactome = tmp_path / "reactome.gmt"
    reactome.write_text("REACTOME_TEST\tna\tG1\tG2\tG3\n", encoding="utf-8")

    calls: list[list[str]] = []

    class _FakePrerankRes:
        def __init__(self, term: str):
            self.res2d = pd.DataFrame(
                {
                    "Term": [term],
                    "ES": [0.6],
                    "NES": [1.8],
                    "NOM p-val": [0.01],
                    "FDR q-val": [0.03],
                    "Lead_genes": ["G1;G2"],
                }
            )

    class _FakeGseapy:
        @staticmethod
        def prerank(**kwargs):
            gene_sets = kwargs["gene_sets"]
            calls.append(sorted(gene_sets.keys()))
            term = sorted(gene_sets.keys())[0]
            return _FakePrerankRes(term)

    monkeypatch.setitem(sys.modules, "gseapy", _FakeGseapy)
    monkeypatch.setattr(au, "_MSIGDB_RESOLUTION_CACHE", {}, raising=False)
    monkeypatch.setattr(au, "_MSIGDB_GENE_SET_CACHE", {}, raising=False)
    monkeypatch.setattr(
        "scomnom.annotation_utils.resolve_msigdb_gene_sets",
        lambda gene_sets: ([str(hallmark), str(reactome)], ["HALLMARK", "REACTOME"], "vX"),
    )

    stats = pd.DataFrame({"C00": [3.0, 1.0, -1.0]}, index=["G1", "G2", "G3"])
    cfg = SimpleNamespace(
        msigdb_gene_sets=["HALLMARK", "REACTOME"],
        gsea_min_size=1,
        gsea_max_size=500,
        gsea_eps=1e-10,
        random_state=42,
        joint_enrichment_leading_edge_top_n=5,
        n_jobs=1,
    )

    payload = _run_msigdb_gsea_from_stats(stats, cfg, input_label="demo")

    assert payload is not None
    assert sorted(calls) == [["HALLMARK_TEST"], ["REACTOME_TEST"]]
    assert sorted(payload["results"]["gmt"].unique().tolist()) == ["HALLMARK", "REACTOME"]


def test_run_msigdb_gsea_from_stats_caches_msigdb_resolution(monkeypatch, tmp_path: Path) -> None:
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_TEST\tna\tG1\tG2\tG3\n", encoding="utf-8")

    class _FakePrerankRes:
        def __init__(self):
            self.res2d = pd.DataFrame(
                {
                    "Term": ["HALLMARK_TEST"],
                    "ES": [0.6],
                    "NES": [1.8],
                    "NOM p-val": [0.01],
                    "FDR q-val": [0.03],
                    "Lead_genes": ["G1;G2"],
                }
            )

    class _FakeGseapy:
        @staticmethod
        def prerank(**kwargs):
            return _FakePrerankRes()

    import sys

    monkeypatch.setitem(sys.modules, "gseapy", _FakeGseapy)
    monkeypatch.setattr(au, "_MSIGDB_RESOLUTION_CACHE", {}, raising=False)
    monkeypatch.setattr(au, "_MSIGDB_GENE_SET_CACHE", {}, raising=False)

    calls = {"n": 0}

    def _resolve(gene_sets):
        calls["n"] += 1
        return [str(gmt)], ["HALLMARK"], "vX"

    monkeypatch.setattr("scomnom.annotation_utils.resolve_msigdb_gene_sets", _resolve)

    stats = pd.DataFrame({"C00": [3.0, 1.0, -1.0]}, index=["G1", "G2", "G3"])
    cfg = SimpleNamespace(
        msigdb_gene_sets=["HALLMARK"],
        gsea_min_size=1,
        gsea_max_size=500,
        gsea_eps=1e-10,
        random_state=42,
        joint_enrichment_leading_edge_top_n=5,
        n_jobs=1,
    )

    payload1 = _run_msigdb_gsea_from_stats(stats, cfg, input_label="demo1")
    payload2 = _run_msigdb_gsea_from_stats(stats, cfg, input_label="demo2")

    assert payload1 is not None
    assert payload2 is not None
    assert calls["n"] == 1


def test_build_stats_matrix_from_tables_prefers_cell_wilcoxon_score_over_logfc() -> None:
    tables = {
        "C00": pd.DataFrame(
            {
                "gene": ["G1", "G2"],
                "cell_wilcoxon_logfc": [0.2, 0.1],
                "cell_wilcoxon_score": [5.0, -3.0],
            }
        )
    }

    got = _build_stats_matrix_from_tables(
        tables,
        preferred_col="stat",
        fallback_cols=("log2FoldChange", "cell_wilcoxon_score", "cell_wilcoxon_logfc"),
    )

    assert got.loc["G1", "C00"] == 5.0
    assert got.loc["G2", "C00"] == -3.0


def test_merge_msigdb_decoupler_and_gsea_marks_concordance() -> None:
    decoupler_payload = {
        "activity": pd.DataFrame(
            [[1.5, -2.0]],
            index=["C00"],
            columns=["HALLMARK_UP", "HALLMARK_DOWN"],
        )
    }
    gsea_payload = {
        "results": pd.DataFrame(
            {
                "cluster": ["C00", "C00"],
                "pathway": ["HALLMARK_UP", "HALLMARK_DOWN"],
                "NES": [2.1, -1.7],
                "ES": [0.5, -0.4],
                "pval": [0.001, 0.02],
                "padj": [0.01, 0.08],
                "leading_edge": ["G1,G2", "G3,G4"],
                "leading_edge_n": [2, 2],
                "leading_edge_preview": ["G1, G2", "G3, G4"],
                "direction": ["up", "down"],
            }
        )
    }

    payload = _merge_msigdb_decoupler_and_gsea(
        decoupler_payload=decoupler_payload,
        gsea_payload=gsea_payload,
        alpha=0.05,
        leading_edge_top_n=5,
    )

    assert payload is not None
    results = payload["results"]
    assert set(results["sign_concordant"].tolist()) == {True}
    got = results.set_index("pathway")["supported_by_both"].to_dict()
    assert got["HALLMARK_UP"] is True
    assert got["HALLMARK_DOWN"] is False


def test_write_de_enrichment_tables_writes_results_and_activity(tmp_path: Path) -> None:
    payloads = {
        "msigdb": {"activity": pd.DataFrame([[1.0]], index=["C00"], columns=["TERM"])},
        "msigdb_gsea": {
            "results": pd.DataFrame(
                {
                    "cluster": ["C00"],
                    "pathway": ["TERM"],
                    "NES": [1.9],
                    "padj": [0.02],
                }
            )
        },
    }

    _write_de_enrichment_tables(payloads, tables_root=tmp_path)

    assert (tmp_path / "msigdb" / "activity.tsv").exists()
    assert (tmp_path / "msigdb_gsea" / "results.tsv").exists()


def test_load_de_enrichment_payload_from_tables_rehydrates_summary_payload(tmp_path: Path) -> None:
    payloads = {
        "msigdb_gsea": {
            "results": pd.DataFrame(
                {
                    "cluster": ["C00"],
                    "pathway": ["TERM"],
                    "NES": [1.9],
                    "padj": [0.02],
                }
            )
        },
        "msigdb": {"activity": pd.DataFrame([[1.0]], index=["C00"], columns=["TERM"])},
    }
    _write_de_enrichment_tables(payloads, tables_root=tmp_path)

    got = _load_de_enrichment_payload_from_tables({}, net_name="msigdb_gsea", tables_root=tmp_path)
    assert "results" in got
    assert isinstance(got["results"], pd.DataFrame)
    assert got["results"].loc[0, "pathway"] == "TERM"

    got_msigdb = _load_de_enrichment_payload_from_tables({}, net_name="msigdb", tables_root=tmp_path)
    assert "activity" in got_msigdb
    assert isinstance(got_msigdb["activity"], pd.DataFrame)
    assert got_msigdb["activity"].loc["C00", "TERM"] == 1.0


def test_prune_uns_de_replaces_de_enrichment_tables_with_summaries() -> None:
    adata = ad.AnnData(X=np.zeros((1, 1)))
    adata.uns["scomnom_de"] = {
        "de_decoupler": {
            "sex": {
                "female_vs_male": {
                    "cell": {
                        "nets": {
                            "msigdb_gsea": {
                                "results": pd.DataFrame(
                                    {
                                        "cluster": ["C00", "C00"],
                                        "pathway": ["P1", "P2"],
                                        "NES": [1.8, -1.2],
                                        "padj": [0.01, 0.03],
                                        "leading_edge_n": [3, 2],
                                        "leading_edge_preview": ["G1, G2", "G3, G4"],
                                    }
                                ),
                                "config": {"resource": "msigdb_gsea"},
                            }
                        }
                    }
                }
            }
        }
    }

    _prune_uns_de(adata, "scomnom_de", top_n=10, decoupler_top_n=5)

    payload = adata.uns["scomnom_de"]["de_decoupler"]["sex"]["female_vs_male"]["cell"]["nets"]["msigdb_gsea"]
    assert "results" not in payload
    assert "results_top" in payload
    assert "summary" in payload
    assert int(payload["summary"]["n_rows"]) == 2


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


def test_apply_gene_filters_to_var_names_accepts_protein_coding_hyphen_alias() -> None:
    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.var_names = ["XIST", "RPL13", "CXCL8", "LST1"]
    adata.var["gene_type"] = ["lncrna", "protein_coding", "protein_coding", "protein_coding"]

    keep_mask, info = _apply_gene_filters_to_var_names(
        adata,
        gene_filter=("gene_type == 'protein-coding'",),
        resource_name="test",
    )

    assert keep_mask.tolist() == [False, True, True, True]
    assert info == {
        "gene_filter": ("gene_type == 'protein-coding'",),
        "n_genes_input": 4,
        "n_genes_retained": 3,
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
    assert adata.var["gene_type"].tolist() == ["lncrna", "protein_coding", "protein_coding"]
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


def test_apply_gene_filters_to_de_stats_accepts_protein_coding_hyphen_alias() -> None:
    from scomnom.markers_and_de import _apply_gene_filters_to_de_stats

    stats = pd.DataFrame({"stat": [1.0, 2.0, 3.0]}, index=pd.Index(["A", "B", "C"], name="gene"))
    gene_meta = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "gene_type": ["protein_coding", "lncrna", "protein_coding"],
        }
    ).set_index("gene", drop=False)

    filtered, info = _apply_gene_filters_to_de_stats(
        stats,
        gene_meta=gene_meta,
        gene_filter=("gene_type == 'protein-coding'",),
        resource_name="test",
    )

    assert filtered.index.tolist() == ["A", "C"]
    assert info == {
        "gene_filter": ("gene_type == 'protein-coding'",),
        "n_genes_input": 3,
        "n_genes_retained": 2,
    }


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
    gsea_dir = run_dir / "msigdb_gsea" / "sex" / "female_vs_male"
    gsea_dir.mkdir(parents=True)
    (gsea_dir / "msigdb_gsea_summary.png").write_bytes(b"")
    joint_dir = run_dir / "msigdb_joint" / "sex" / "female_vs_male"
    joint_dir.mkdir(parents=True)
    (joint_dir / "msigdb_joint_concordant.png").write_bytes(b"")

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
    assert "MSigDB GSEA" in html
    assert "MSigDB concordance" in html


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


def test_run_liana_ccc_writes_tables_and_uns(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class _FakeMethod:
        def __init__(self, score_col: str):
            self.score_col = score_col

        def __call__(self, adata, **kwargs):
            calls.append({"kind": self.score_col, **kwargs})
            return pd.DataFrame(
                {
                    "source": ["C00", "C01"],
                    "target": ["C01", "C00"],
                    "ligand_complex": ["LIG1", "LIG2"],
                    "receptor_complex": ["REC1", "REC2"],
                    self.score_col: [0.9, 0.7],
                }
            )

    class _FakeAggregate:
        def __init__(self, methods=None):
            self.methods = methods or []

        def __call__(self, adata, **kwargs):
            calls.append({"kind": "rank_aggregate", **kwargs})
            return pd.DataFrame(
                {
                    "source": ["C00", "C01"],
                    "target": ["C01", "C00"],
                    "ligand_complex": ["LIG1", "LIG2"],
                    "receptor_complex": ["REC1", "REC2"],
                    "magnitude_rank": [0.001, 0.01],
                    "specificity_rank": [0.002, 0.02],
                }
            )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(
            rank_aggregate=_FakeAggregate(),
            AggregateClass=lambda meta, methods: _FakeAggregate(methods=methods),
            aggregate_meta=object(),
        ),
        method=SimpleNamespace(
            cellphonedb=_FakeMethod("lr_means"),
            connectome=_FakeMethod("scaled_weight"),
            natmi=_FakeMethod("spec_weight"),
            singlecellsignalr=_FakeMethod("lrscore"),
            logfc=_FakeMethod("lr_logfc"),
        ),
    )
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    adata = ad.AnnData(X=np.ones((4, 3)))
    adata.var_names = ["G1", "G2", "G3"]
    adata.layers["counts_cb"] = adata.X.copy()
    adata.layers["counts_raw"] = adata.X.copy() * 2
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5"] = [
        "C00: Kupffer",
        "C00: Kupffer",
        "C01: T cells",
        "C01: T cells",
    ]
    adata.obs["sex"] = ["female", "female", "male", "male"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {"C00": "C00: Kupffer", "C01": "C01: T cells"},
        }
    }

    saved: list[tuple[str, str]] = []

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(
        md_mod.io_utils,
        "save_dataset",
        lambda _adata, path, fmt: saved.append((str(path), str(fmt))),
    )

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.log",
        make_figures=False,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_condition_key="sex",
        ccc_condition_keys=("sex",),
        ccc_condition_values=("female", "male"),
        ccc_compare_levels=(),
        liana_resource="consensus",
        liana_methods=("rank_aggregate", "cellphonedb"),
        liana_expr_prop=0.1,
        liana_use_raw=False,
        liana_layer=None,
        liana_n_perms=None,
        liana_seed=42,
        liana_return_all_lrs=False,
        liana_top_n=25,
        liana_plot_top_n=10,
    )

    out = run_liana_ccc(cfg)

    assert out is adata
    assert saved
    assert saved[0][0].endswith("adata.ccc_liana.zarr")
    runs = adata.uns["markers_and_de"]["ccc"]["liana"]["runs"]
    assert set(runs.keys()) == {"sex::female", "sex::male"}
    assert runs["sex::female"]["primary_method"] == "rank_aggregate"
    assert runs["sex::female"]["aggregated_methods"] == ["cellphonedb"]
    assert isinstance(runs["sex::female"]["top_interactions"], pd.DataFrame)
    assert isinstance(runs["sex::female"]["source_target_summary"], pd.DataFrame)
    assert isinstance(runs["sex::female"]["route_family_summary"], pd.DataFrame)
    assert (tmp_path / "out" / "tables").exists()
    assert list((tmp_path / "out" / "tables").glob("**/route_family_summary.tsv"))
    assert all(call["use_raw"] is False for call in calls)
    assert all(call["layer"] == "counts_cb" for call in calls)


def test_run_liana_ccc_lognorm_mode_builds_cached_layer(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class _FakeAggregate:
        def __call__(self, adata, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {
                    "source": ["C00", "C01"],
                    "target": ["C01", "C00"],
                    "ligand_complex": ["LIG1", "LIG2"],
                    "receptor_complex": ["REC1", "REC2"],
                    "magnitude_rank": [0.001, 0.01],
                    "specificity_rank": [0.002, 0.02],
                }
            )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(
            rank_aggregate=_FakeAggregate(),
            AggregateClass=lambda meta, methods: _FakeAggregate(),
            aggregate_meta=object(),
        ),
        method=SimpleNamespace(
            cellphonedb=lambda *args, **kwargs: pd.DataFrame(),
            connectome=lambda *args, **kwargs: pd.DataFrame(),
            natmi=lambda *args, **kwargs: pd.DataFrame(),
            singlecellsignalr=lambda *args, **kwargs: pd.DataFrame(),
            logfc=lambda *args, **kwargs: pd.DataFrame(),
        ),
    )
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    adata = ad.AnnData(X=np.zeros((4, 3)))
    adata.var_names = ["G1", "G2", "G3"]
    adata.layers["counts_cb"] = np.array(
        [
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5"] = ["C00: Kupffer", "C00: Kupffer", "C01: T cells", "C01: T cells"]
    adata.obs["sex"] = ["female", "female", "male", "male"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {"C00": "C00: Kupffer", "C01": "C01: T cells"},
        }
    }

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.log",
        make_figures=False,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_condition_key="sex",
        ccc_condition_keys=("sex",),
        ccc_condition_values=("female", "male"),
        ccc_compare_levels=(),
        liana_resource="consensus",
        liana_methods=("rank_aggregate",),
        liana_expr_prop=0.1,
        liana_input_mode="lognorm",
        liana_lognorm_target_sum=10000.0,
        liana_use_raw=False,
        liana_layer=None,
        liana_n_perms=None,
        liana_seed=42,
        liana_return_all_lrs=False,
        liana_top_n=25,
        liana_plot_top_n=10,
    )

    run_liana_ccc(cfg)

    assert "lognorm_counts_cb" in adata.layers
    assert all(call["use_raw"] is False for call in calls)
    assert all(call["layer"] == "lognorm_counts_cb" for call in calls)
    row_sums = np.asarray(np.expm1(adata.layers["lognorm_counts_cb"]).sum(axis=1)).ravel()
    assert np.allclose(row_sums, np.full(adata.n_obs, 10000.0))


def test_liana_route_family_prefers_cellchat_pathway_lookup() -> None:
    assert md_mod._liana_route_family("BMP2", "BMPR1A_BMPR2") == "BMP"
    assert md_mod._liana_route_family("TGFB1", "TGFBR1_TGFBR2") == "TGFb"


def test_run_liana_ccc_expands_a_within_b_condition_spec(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class _FakeAggregate:
        def __call__(self, adata, **kwargs):
            calls.append({"n_obs": int(adata.n_obs), **kwargs})
            return pd.DataFrame(
                {
                    "source": ["C00"],
                    "target": ["C01"],
                    "ligand_complex": ["LIG1"],
                    "receptor_complex": ["REC1"],
                    "magnitude_rank": [0.001],
                    "specificity_rank": [0.002],
                }
            )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(
            rank_aggregate=_FakeAggregate(),
            AggregateClass=lambda meta, methods: _FakeAggregate(),
            aggregate_meta=object(),
        ),
        method=SimpleNamespace(
            cellphonedb=lambda *args, **kwargs: pd.DataFrame(),
            connectome=lambda *args, **kwargs: pd.DataFrame(),
            natmi=lambda *args, **kwargs: pd.DataFrame(),
            singlecellsignalr=lambda *args, **kwargs: pd.DataFrame(),
            logfc=lambda *args, **kwargs: pd.DataFrame(),
        ),
    )
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    adata = ad.AnnData(X=np.ones((6, 2)))
    adata.var_names = ["G1", "G2"]
    adata.layers["counts_cb"] = adata.X.copy()
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01", "C00", "C01"]
    adata.obs["cluster_label__r5"] = ["C00: A", "C00: A", "C01: B", "C01: B", "C00: A", "C01: B"]
    adata.obs["sex"] = ["female", "male", "female", "male", "female", "male"]
    adata.obs["MASLD"] = ["yes", "yes", "yes", "yes", "no", "no"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {"C00": "C00: A", "C01": "C01: B"},
        }
    }

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.log",
        make_figures=False,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_condition_key="sex@MASLD",
        ccc_condition_keys=("sex@MASLD",),
        ccc_condition_values=(),
        ccc_compare_levels=(),
        liana_resource="consensus",
        liana_methods=("rank_aggregate",),
        liana_expr_prop=0.1,
        liana_use_raw=False,
        liana_layer=None,
        liana_n_perms=None,
        liana_seed=42,
        liana_return_all_lrs=False,
        liana_top_n=25,
        liana_plot_top_n=10,
    )

    run_liana_ccc(cfg)

    runs = adata.uns["markers_and_de"]["ccc"]["liana"]["runs"]
    assert set(runs.keys()) == {
        "sex@MASLD::MASLD=no::female",
        "sex@MASLD::MASLD=no::male",
        "sex@MASLD::MASLD=yes::female",
        "sex@MASLD::MASLD=yes::male",
    }
    assert runs["sex@MASLD::MASLD=yes::female"]["condition_key"] == "sex"
    assert runs["sex@MASLD::MASLD=yes::female"]["condition_value"] == "female"
    assert runs["sex@MASLD::MASLD=yes::female"]["context_key"] == "MASLD"
    assert runs["sex@MASLD::MASLD=yes::female"]["context_value"] == "yes"
    assert sorted(call["n_obs"] for call in calls) == [1, 1, 2, 2]


def test_run_liana_ccc_cross_tissue_secreted_filter(monkeypatch, tmp_path: Path) -> None:
    class _FakeAggregate:
        def __call__(self, adata, **kwargs):
            return pd.DataFrame(
                {
                    "source": ["C00", "C10", "C00", "C01"],
                    "target": ["C10", "C00", "C11", "C10"],
                    "ligand_complex": ["BMP2", "BMP2", "LIGX", "TGFB1"],
                    "receptor_complex": ["BMPR1A_BMPR2", "BMPR1A_BMPR2", "RECX", "TGFBR1_TGFBR2"],
                    "magnitude_rank": [0.001, 0.002, 0.003, 0.004],
                    "specificity_rank": [0.001, 0.002, 0.003, 0.004],
                }
            )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(
            rank_aggregate=_FakeAggregate(),
            AggregateClass=lambda meta, methods: _FakeAggregate(),
            aggregate_meta=object(),
        ),
        method=SimpleNamespace(
            cellphonedb=lambda *args, **kwargs: pd.DataFrame(),
            connectome=lambda *args, **kwargs: pd.DataFrame(),
            natmi=lambda *args, **kwargs: pd.DataFrame(),
            singlecellsignalr=lambda *args, **kwargs: pd.DataFrame(),
            logfc=lambda *args, **kwargs: pd.DataFrame(),
        ),
    )
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    adata = ad.AnnData(X=np.ones((6, 2)))
    adata.var_names = ["G1", "G2"]
    adata.layers["counts_cb"] = adata.X.copy()
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C10", "C10", "C11"]
    adata.obs["cluster_label__r5"] = ["C00: L1", "C00: L1", "C01: L2", "C10: F1", "C10: F1", "C11: F2"]
    adata.obs["tissue"] = ["liver", "liver", "liver", "fat", "fat", "fat"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {
                "C00": "C00: L1",
                "C01": "C01: L2",
                "C10": "C10: F1",
                "C11": "C11: F2",
            },
        }
    }

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.log",
        make_figures=False,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_condition_key=None,
        ccc_condition_keys=(),
        ccc_condition_values=(),
        ccc_compare_levels=(),
        ccc_dataset_key="tissue",
        ccc_source_levels=("liver",),
        ccc_target_levels=("fat",),
        ccc_signal_scope="secreted",
        liana_resource="consensus",
        liana_methods=("rank_aggregate",),
        liana_expr_prop=0.1,
        liana_use_raw=False,
        liana_layer=None,
        liana_n_perms=None,
        liana_seed=42,
        liana_return_all_lrs=False,
        liana_top_n=25,
        liana_plot_top_n=10,
    )

    run_liana_ccc(cfg)

    runs = adata.uns["markers_and_de"]["ccc"]["liana"]["runs"]
    top = runs["global"]["top_interactions"]
    assert top[["source", "target"]].astype(str).values.tolist() == [["C00", "C10"], ["C01", "C10"]]
    assert set(top["ligand_complex"].astype(str)) == {"BMP2", "TGFB1"}
    assert runs["global"]["cross_tissue_mode"] is True
    assert runs["global"]["signal_scope"] == "secreted"


@pytest.mark.parametrize(
    ("input_mode", "expect_layer"),
    [
        ("counts", "counts_cb"),
        ("lognorm", "lognorm_counts_cb"),
    ],
)
def test_run_nichenet_ccc_cross_tissue_sender_focused(
    monkeypatch, tmp_path: Path, input_mode: str, expect_layer: str
) -> None:
    project_r_lib = tmp_path / "Library" / "Caches" / "scOmnom" / "r-libs" / "nichenet"
    adata = ad.AnnData(X=np.ones((8, 4)))
    adata.var_names = ["BMP2", "TGFB1", "COL1A1", "SOX9"]
    adata.layers["counts_cb"] = adata.X.copy()
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01", "C03", "C03", "C03", "C10"]
    adata.obs["cluster_label__r5"] = [
        "C00: Hep 1",
        "C00: Hep 1",
        "C01: Hep 2",
        "C01: Hep 2",
        "C03: Adipo",
        "C03: Adipo",
        "C03: Adipo",
        "C10: Other",
    ]
    adata.obs["tissue"] = ["liver", "liver", "liver", "liver", "fat", "fat", "fat", "fat"]
    adata.obs["sex"] = ["female", "male", "female", "male", "female", "female", "male", "male"]
    adata.obs["MASLD"] = ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {
                "C00": "C00: Hep 1",
                "C01": "C01: Hep 2",
                "C03": "C03: Adipo",
                "C10": "C10: Other",
            },
        }
    }

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda a, p, fmt: saved.append((str(p), str(fmt))))
    monkeypatch.setattr(md_mod, "_ensure_nichenet_r_runtime", lambda install_missing: project_r_lib)
    captured_layers: list[tuple[Optional[str], bool]] = []

    def _fake_compute_expressed_genes(adata_in, *, mask, layer, use_raw, min_fraction):
        captured_layers.append((layer, use_raw))
        return ["BMP2", "TGFB1", "COL1A1", "SOX9"]

    monkeypatch.setattr(md_mod, "_compute_expressed_genes", _fake_compute_expressed_genes)

    monkeypatch.setattr(
        md_mod,
        "_extract_receiver_de_geneset",
        lambda *args, **kwargs: (
            ["SOX9", "COL1A1"],
            ["BMP2", "TGFB1", "COL1A1", "SOX9"],
            pd.DataFrame({"gene": ["SOX9", "COL1A1"], "logfoldchanges": [1.2, 0.8], "pvals_adj": [0.01, 0.02]}),
        ),
    )
    monkeypatch.setattr(
        md_mod,
        "_run_nichenet_sender_focused",
        lambda **kwargs: {
            "ligand_activity": pd.DataFrame({"test_ligand": ["BMP2", "TGFB1"], "pearson": [0.21, 0.17]}),
            "ligand_target_links": pd.DataFrame(
                {
                    "ligand": ["BMP2", "BMP2", "TGFB1"],
                    "target": ["SOX9", "COL1A1", "SOX9"],
                    "weight": [0.8, 0.5, 0.6],
                }
            ),
            "ligand_receptor_links": pd.DataFrame({"from": ["BMP2"], "to": ["BMPR1A_BMPR2"], "weight": [0.9]}),
            "potential_ligands": pd.DataFrame({"ligand": ["BMP2", "TGFB1"]}),
        },
    )

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_nichenet",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.nichenet.log",
        make_figures=False,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_backend="nichenet",
        ccc_condition_key="sex@MASLD",
        ccc_condition_keys=("sex@MASLD",),
        ccc_condition_values=("yes",),
        ccc_compare_levels=("female", "male"),
        ccc_dataset_key="tissue",
        ccc_source_levels=("liver",),
        ccc_target_levels=("fat",),
        ccc_signal_scope="secreted",
        nichenet_receiver_cluster="all",
        nichenet_sender_clusters=("C00", "C01"),
        nichenet_gene_list_file=None,
        nichenet_expression_pct=0.1,
        nichenet_input_mode=input_mode,
        nichenet_lognorm_target_sum=5000.0,
        nichenet_top_n_ligands=10,
        nichenet_top_n_targets=50,
        nichenet_min_logfc=0.25,
        nichenet_padj_threshold=0.05,
        nichenet_organism="human",
        nichenet_install_missing_r_deps=False,
    )

    out = run_nichenet_ccc(cfg)

    assert out is adata
    assert saved
    assert captured_layers
    assert all(layer == expect_layer for layer, use_raw in captured_layers)
    assert all(use_raw is False for layer, use_raw in captured_layers)
    if input_mode == "lognorm":
        assert "lognorm_counts_cb" in adata.layers
        assert adata.layers["lognorm_counts_cb"].dtype == np.float32
    else:
        assert "lognorm_counts_cb" not in adata.layers
    runs = adata.uns["markers_and_de"]["ccc"]["nichenet"]["runs"]
    run_key = "sex@MASLD::MASLD=yes::female_vs_male::receiver=C03"
    assert run_key in runs
    assert set(runs[run_key]["ligand_activity"]["test_ligand"].astype(str)) == {"BMP2", "TGFB1"}
    assert runs[run_key]["cross_tissue_mode"] is True
    assert runs[run_key]["receiver_cluster_mode"] == "all"
    assert str(runs[run_key]["project_r_lib"]) == str(project_r_lib)
    assert list((tmp_path / "out" / "tables").glob("**/nichenet_ligand_activity.tsv"))


def test_run_mebocost_ccc_cross_tissue(monkeypatch, tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.ones((8, 4)))
    adata.layers["counts_cb"] = np.full((8, 4), 2.0)
    adata.var_names = ["BMP2", "TGFB1", "COL1A1", "SOX9"]
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01", "C03", "C03", "C03", "C10"]
    adata.obs["cluster_label__r5"] = [
        "C00: Hep 1",
        "C00: Hep 1",
        "C01: Hep 2",
        "C01: Hep 2",
        "C03: Adipo",
        "C03: Adipo",
        "C03: Adipo",
        "C10: Other",
    ]
    adata.obs["tissue"] = ["liver", "liver", "liver", "liver", "fat", "fat", "fat", "fat"]
    adata.obs["timepoint"] = ["5_years_post"] * 8
    adata.obs["masld_status"] = ["better", "worse", "better", "worse", "better", "better", "worse", "worse"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {
                "C00": "C00: Hep 1",
                "C01": "C01: Hep 2",
                "C03": "C03: Adipo",
                "C10": "C10: Other",
            },
        }
    }

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda a, p, fmt: saved.append((str(p), str(fmt))))
    artifact_stems: list[str] = []
    monkeypatch.setattr(
        md_mod.plot_utils,
        "persist_plot_artifacts",
        lambda artifacts: artifact_stems.extend(str(a.stem) for a in artifacts),
    )

    class _FakeMeboObj:
        def __init__(self):
            self.commu_res = pd.DataFrame(
                {
                    "sender": ["C00", "C01", "C03"],
                    "receiver": ["C03", "C03", "C10"],
                    "metabolite": ["HMDB0001", "HMDB0002", "HMDB0003"],
                    "Metabolite_Name": ["glutamine", "lactate", "palmitate"],
                    "sensor": ["SLC1A5", "HCAR1", "CD36"],
                    "Annotation": ["Transporter", "Receptor", "Transporter"],
                    "commu_score": [0.8, 0.6, 0.2],
                    "permutation_test_fdr": [0.01, 0.03, 0.2],
                }
            )

        def infer_commu(self, **kwargs):
            return self.commu_res

    create_obj_calls: list[dict[str, object]] = []

    class _FakeMeboApi:
        @staticmethod
        def create_obj(**kwargs):
            create_obj_calls.append(kwargs)
            return _FakeMeboObj()

    monkeypatch.setattr(md_mod, "_import_mebocost_api", lambda install_missing: _FakeMeboApi())
    monkeypatch.setattr(md_mod, "_ensure_mebocost_resource_config", lambda install_missing: tmp_path / "mebocost.conf")
    monkeypatch.setattr(
        md_mod,
        "_load_mebocost_annotation_tables",
        lambda config_path, species: (
            pd.DataFrame(
                {
                    "HMDB_ID": ["HMDB0001", "HMDB0002", "HMDB0003"],
                    "metabolite": ["glutamine", "lactate", "palmitate"],
                    "kingdom": ["Organic compounds", "Organic compounds", "Lipids"],
                    "super_class": ["Organic acids", "Organic acids", "Lipids and lipid-like molecules"],
                    "class": ["Carboxylic acids", "Alpha hydroxy acids", "Fatty acyls"],
                    "sub_class": ["Amino acids", "Hydroxy acids", "Fatty acids"],
                    "BioLocation_Summary": ["Blood", "Blood", "Blood"],
                    "Subcellular": ["Cytoplasm", "Cytoplasm", "Membrane"],
                    "Kegg_ID": ["C00064", "C00186", "C00249"],
                    "associated_gene": ["GLS", "LDHA", "CD36"],
                }
            ),
            pd.DataFrame(
                {
                    "HMDB_ID": ["HMDB0001", "HMDB0002", "HMDB0003"],
                    "standard_metName": ["glutamine", "lactate", "palmitate"],
                    "Gene_name": ["SLC1A5", "HCAR1", "CD36"],
                    "Protein_name": ["Neutral amino acid transporter B(0)", "Hydroxycarboxylic acid receptor 1", "Platelet glycoprotein 4"],
                    "Annotation": ["Transporter", "Receptor", "Transporter"],
                    "Evidence": ["PMID:1", "PMID:2", "PMID:3"],
                }
            ),
        ),
    )

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_mebocost",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.mebocost.log",
        make_figures=True,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_backend="mebocost",
        ccc_condition_key="masld_status@timepoint",
        ccc_condition_keys=("masld_status@timepoint",),
        ccc_condition_values=("5_years_post",),
        ccc_compare_levels=("better", "worse"),
        ccc_dataset_key="tissue",
        ccc_source_levels=("liver",),
        ccc_target_levels=("fat",),
        mebocost_organism="human",
        mebocost_n_shuffle=100,
        mebocost_seed=42,
        mebocost_min_cell_number=10,
        mebocost_pval_cutoff=0.05,
        mebocost_plot_top_n=20,
        mebocost_install_missing_python_deps=False,
    )

    out = run_mebocost_ccc(cfg)

    assert out is adata
    assert saved
    runs = adata.uns["markers_and_de"]["ccc"]["mebocost"]["runs"]
    run_key = "masld_status@timepoint::timepoint=5_years_post::better"
    assert run_key in runs
    got = runs[run_key]["commu_res"]
    assert set(got["source"].astype(str)) == {"C00", "C01"}
    assert set(got["target"].astype(str)) == {"C03"}
    assert "source_label" in got.columns
    assert "target_label" in got.columns
    assert "super_class" in got.columns
    assert "sensor_annotation" in got.columns
    assert "sensor_protein_name" in got.columns
    assert "metabolite_label" in got.columns
    assert got["HMDB_ID"].notna().all()
    assert set(got["source_label"].astype(str)) == {"C00: Hep 1", "C01: Hep 2"}
    assert set(got["target_label"].astype(str)) == {"C03: Adipo"}
    assert set(got["sensor_annotation"].dropna().astype(str)) == {"Transporter", "Receptor"}
    assert any("glutamine [HMDB0001]" == x for x in got["metabolite_label"].astype(str))
    assert runs[run_key]["cross_tissue_mode"] is True
    assert create_obj_calls
    assert create_obj_calls[0]["config_path"] == str(tmp_path / "mebocost.conf")
    np.testing.assert_array_equal(np.asarray(create_obj_calls[0]["adata"].X), np.full((4, 4), 2.0))
    assert "mebocost_top_events" in artifact_stems
    assert "mebocost_metabolite_superclass_summary" in artifact_stems
    assert "mebocost_sensor_annotation_summary" in artifact_stems
    assert "mebocost_metabolite_superclass_by_source__C00" in artifact_stems
    assert "mebocost_metabolite_superclass_by_target__C03" in artifact_stems
    assert "mebocost_sensor_annotation_by_source__C00" in artifact_stems
    assert "mebocost_sensor_annotation_by_target__C03" in artifact_stems
    assert "mebocost_source_target_heatmap" in artifact_stems
    assert "mebocost_top_target_clusters_by_source__C00" in artifact_stems
    assert "mebocost_top_source_clusters_by_target__C03" in artifact_stems
    assert "mebocost_top_events_by_source__C00" in artifact_stems
    assert "mebocost_top_events_by_target__C03" in artifact_stems
    assert "mebocost_source_target_compare_heatmap" in artifact_stems
    assert "mebocost_source_target_compare_mean_score_heatmap" in artifact_stems
    assert "mebocost_top_events_by_source_by_condition__C00" in artifact_stems
    assert "mebocost_top_events_by_target_by_condition__C03" in artifact_stems
    assert "mebocost_metabolite_superclass_by_source_by_condition__C00" in artifact_stems
    assert "mebocost_metabolite_superclass_by_target_by_condition__C03" in artifact_stems
    assert "mebocost_sensor_annotation_by_source_by_condition__C00" in artifact_stems
    assert "mebocost_sensor_annotation_by_target_by_condition__C03" in artifact_stems
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_commu_res.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_metabolite_superclass_by_source.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_metabolite_superclass_by_target.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_sensor_annotation_by_source.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_sensor_annotation_by_target.tsv"))


def test_make_mebocost_object_for_sample_uses_requested_layer() -> None:
    adata = ad.AnnData(X=np.ones((3, 2)))
    adata.layers["counts_cb"] = np.full((3, 2), 5.0)
    adata.layers["lognorm_counts_cb"] = np.full((3, 2), 9.0)
    adata.obs["cluster"] = ["C00", "C00", "C01"]

    captured: dict[str, np.ndarray] = {}

    class _FakeMeboObj:
        def _load_config_(self):
            return None

        def _avg_by_group_(self):
            return None

        def _get_gene_exp_(self):
            return None

        def estimator(self):
            return None

        def _avg_met_group_(self):
            return None

        def _check_aboundance_(self):
            return None

    class _FakeMeboApi:
        @staticmethod
        def create_obj(**kwargs):
            captured["X"] = np.asarray(kwargs["adata"].X)
            return _FakeMeboObj()

    md_mod._make_mebocost_object_for_sample(
        adata,
        groupby="cluster",
        organism="human",
        config_path=Path("mebocost.conf"),
        mebocost_api=_FakeMeboApi(),
        layer="lognorm_counts_cb",
    )

    np.testing.assert_array_equal(captured["X"], np.full((3, 2), 9.0))


def test_run_mebocost_paired_rescore_lognorm(monkeypatch, tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.ones((6, 3)))
    adata.layers["counts_cb"] = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
        ]
    )
    adata.var_names = ["CD36", "SLC1A5", "HCAR1"]
    adata.obs["leiden__r5"] = ["C05", "C05", "C00", "C00", "C03", "C03"]
    adata.obs["cluster_label__r5"] = [
        "C05: Adipo A",
        "C05: Adipo A",
        "C00: Hep 1",
        "C00: Hep 1",
        "C03: Mac 1",
        "C03: Mac 1",
    ]
    adata.obs["sample_id"] = ["S1", "S1", "S1", "S1", "S2", "S2"]
    adata.obs["sex"] = ["female", "female", "female", "female", "male", "male"]
    adata.obs["MASLD"] = ["yes", "yes", "yes", "yes", "yes", "yes"]
    adata.obs["tissue"] = ["vfat", "vfat", "liv", "liv", "vfat", "liv"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {
                "C05": "C05: Adipo A",
                "C00": "C00: Hep 1",
                "C03": "C03: Mac 1",
            },
        }
    }

    candidate_path = tmp_path / "candidate_events.tsv"
    pd.DataFrame(
        {
            "source": ["C05"],
            "target": ["C00"],
            "HMDB_ID": ["HMDB0001"],
            "metabolite": ["HMDB0001"],
            "Metabolite_Name": ["palmitate"],
            "sensor_gene": ["CD36"],
            "Norm_Commu_Score": [0.9],
            "super_class": ["Lipids and lipid-like molecules"],
            "sensor_annotation": ["Transporter"],
        }
    ).to_csv(candidate_path, sep="\t", index=False)

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda a, p, fmt: saved.append((str(p), str(fmt))))
    monkeypatch.setattr(md_mod, "_import_mebocost_api", lambda install_missing: object())
    monkeypatch.setattr(md_mod, "_ensure_mebocost_resource_config", lambda install_missing: tmp_path / "mebocost.conf")
    monkeypatch.setattr(md_mod, "_annotate_mebocost_commu_table", lambda df, config_path, species: df.copy())
    monkeypatch.setattr(md_mod, "_prepare_mebocost_plot_df", lambda df, display_map, valid_group_tokens=None: df.copy())
    artifact_stems: list[str] = []
    monkeypatch.setattr(
        md_mod.plot_utils,
        "persist_plot_artifacts",
        lambda artifacts: artifact_stems.extend(str(a.stem) for a in artifacts),
    )

    captured: dict[str, object] = {}

    def _fake_score_events(
        adata_in,
        *,
        candidate_df,
        groupby,
        pairing_key,
        condition_cols,
        dataset_key,
        source_levels,
        target_levels,
        organism,
        config_path,
        mebocost_api,
        layer,
        score_method,
        min_sender_cells,
        min_receiver_cells,
    ):
        captured["layer"] = layer
        return pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "sex": ["female", "male"],
                "MASLD": ["yes", "yes"],
                "source_token": ["C05", "C05"],
                "target_token": ["C00", "C00"],
                "source_label": ["C05: Adipo A", "C05: Adipo A"],
                "target_label": ["C00: Hep 1", "C00: Hep 1"],
                "HMDB_ID": ["HMDB0001", "HMDB0001"],
                "Metabolite_Name": ["palmitate", "palmitate"],
                "super_class": ["Lipids and lipid-like molecules", "Lipids and lipid-like molecules"],
                "sensor_gene": ["CD36", "CD36"],
                "sensor_annotation": ["Transporter", "Transporter"],
                "paired_commu_score": [1.2, 0.7],
                "paired_commu_score_log1p": [np.log1p(1.2), np.log1p(0.7)],
            }
        )

    monkeypatch.setattr(md_mod, "_score_mebocost_paired_events", _fake_score_events)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_mebocost_paired",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.mebocost.paired.log",
        make_figures=True,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_backend="mebocost_paired_rescore",
        ccc_condition_key="sex",
        ccc_condition_keys=("sex",),
        ccc_condition_values=(),
        ccc_compare_levels=(),
        ccc_dataset_key="tissue",
        ccc_source_levels=("vfat",),
        ccc_target_levels=("liv",),
        mebocost_candidate_events=str(candidate_path),
        mebocost_pairing_key="sample_id",
        mebocost_organism="human",
        mebocost_input_mode="lognorm",
        mebocost_lognorm_target_sum=5000.0,
        mebocost_source_filter=(),
        mebocost_target_filter=(),
        mebocost_metabolite_filter=(),
        mebocost_sensor_filter=(),
        mebocost_superclass_filter=(),
        mebocost_class_filter=(),
        mebocost_subclass_filter=(),
        mebocost_max_events=50,
        mebocost_score_method="mebocost-metabolite-sensor",
        mebocost_min_sender_cells=2,
        mebocost_min_receiver_cells=2,
        mebocost_min_scored_donors_per_group=1,
        mebocost_plot_top_n=10,
        mebocost_install_missing_python_deps=False,
    )

    out = md_mod.run_mebocost_paired_rescore(cfg)

    assert out is adata
    assert captured["layer"] == "lognorm_counts_cb"
    assert "lognorm_counts_cb" in adata.layers
    assert adata.layers["lognorm_counts_cb"].dtype == np.float32
    np.testing.assert_allclose(
        np.expm1(np.asarray(adata.layers["lognorm_counts_cb"])).sum(axis=1),
        np.full(adata.n_obs, 5000.0),
        rtol=1e-5,
        atol=1e-2,
    )
    assert saved
    assert saved[0][0].endswith(".zarr")
    paired_runs = adata.uns["markers_and_de"]["ccc"]["mebocost_paired_rescore"]
    assert len(paired_runs) == 1
    paired = next(iter(paired_runs.values()))
    assert "event_missingness" in paired
    assert "route_missingness" in paired
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_paired_event_missingness.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/mebocost_paired_route_missingness.tsv"))
    settings_paths = list((tmp_path / "out" / "tables").glob("**/mebocost_paired_settings.tsv"))
    assert settings_paths
    assert "min_scored_donors_per_group\t1" in settings_paths[0].read_text()


def test_run_liana_paired_rescore_lognorm(monkeypatch, tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.ones((6, 4)))
    adata.layers["counts_cb"] = np.array(
        [
            [6.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )
    adata.var_names = ["ADIPOQ", "ADIPOR2", "LEP", "LEPR"]
    adata.obs["leiden__r5"] = ["C05", "C05", "C00", "C00", "C05", "C00"]
    adata.obs["cluster_label__r5"] = [
        "C05: Adipo A",
        "C05: Adipo A",
        "C00: Hep 1",
        "C00: Hep 1",
        "C05: Adipo A",
        "C00: Hep 1",
    ]
    adata.obs["sample_id"] = ["S1", "S1", "S1", "S1", "S2", "S2"]
    adata.obs["sex"] = ["female", "female", "female", "female", "male", "male"]
    adata.obs["MASLD"] = ["yes", "yes", "yes", "yes", "yes", "yes"]
    adata.obs["tissue"] = ["vfat", "vfat", "liv", "liv", "vfat", "liv"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {
                "C05": "C05: Adipo A",
                "C00": "C00: Hep 1",
            },
        }
    }

    candidate_path = tmp_path / "liana_candidates.tsv"
    pd.DataFrame(
        {
            "source": ["C05"],
            "target": ["C00"],
            "ligand_complex": ["ADIPOQ"],
            "receptor_complex": ["ADIPOR2"],
            "magnitude_rank": [0.02],
        }
    ).to_csv(candidate_path, sep="\t", index=False)

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda a, p, fmt: saved.append((str(p), str(fmt))))
    artifact_stems: list[str] = []
    monkeypatch.setattr(
        md_mod.plot_utils,
        "persist_plot_artifacts",
        lambda artifacts: artifact_stems.extend(str(a.stem) for a in artifacts),
    )

    captured: dict[str, object] = {}

    def _fake_score_edges(
        adata_in,
        *,
        candidate_df,
        groupby,
        pairing_key,
        condition_cols,
        dataset_key,
        source_levels,
        target_levels,
        layer,
        values_logged,
        min_sender_cells,
        min_receiver_cells,
    ):
        captured["layer"] = layer
        captured["values_logged"] = values_logged
        return pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "sex": ["female", "male"],
                "MASLD": ["yes", "yes"],
                "source_token": ["C05", "C05"],
                "target_token": ["C00", "C00"],
                "source_label": ["C05: Adipo A", "C05: Adipo A"],
                "target_label": ["C00: Hep 1", "C00: Hep 1"],
                "branch_pair": ["C05: Adipo A -> C00: Hep 1", "C05: Adipo A -> C00: Hep 1"],
                "route_family": ["Scavenger / metabolic handling", "Scavenger / metabolic handling"],
                "ligand_complex": ["ADIPOQ", "ADIPOQ"],
                "receptor_complex": ["ADIPOR2", "ADIPOR2"],
                "ligand_expr": [2.0, 1.1],
                "receptor_expr": [1.4, 0.8],
                "edge_score": [np.sqrt(2.8), np.sqrt(0.88)],
                "edge_score_log1p": [np.log1p(np.sqrt(2.8)), np.log1p(np.sqrt(0.88))],
                "sender_n_cells": [2, 1],
                "receiver_n_cells": [2, 1],
            }
        )

    monkeypatch.setattr(md_mod, "_score_liana_paired_edges", _fake_score_edges)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana_paired",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.paired.log",
        make_figures=True,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_backend="liana_paired_rescore",
        ccc_condition_key="sex@MASLD",
        ccc_condition_keys=("sex@MASLD",),
        ccc_condition_values=("yes",),
        ccc_compare_levels=("female", "male"),
        ccc_dataset_key="tissue",
        ccc_source_levels=("vfat",),
        ccc_target_levels=("liv",),
        liana_candidate_events=str(candidate_path),
        liana_pairing_key="sample_id",
        liana_input_mode="lognorm",
        liana_lognorm_target_sum=5000.0,
        liana_source_filter=(),
        liana_target_filter=(),
        liana_ligand_filter=(),
        liana_receptor_filter=(),
        liana_route_family_filter=(),
        liana_max_edges=25,
        liana_min_sender_cells=1,
        liana_min_receiver_cells=1,
        liana_min_scored_donors_per_group=1,
        liana_plot_top_n=10,
    )

    out = md_mod.run_liana_paired_rescore(cfg)

    assert out is adata
    assert captured["layer"] == "lognorm_counts_cb"
    assert captured["values_logged"] is True
    assert "lognorm_counts_cb" in adata.layers
    assert adata.layers["lognorm_counts_cb"].dtype == np.float32
    np.testing.assert_allclose(
        np.expm1(np.asarray(adata.layers["lognorm_counts_cb"])).sum(axis=1),
        np.full(adata.n_obs, 5000.0),
        rtol=1e-5,
        atol=1e-2,
    )
    assert saved
    assert saved[0][0].endswith(".zarr")
    paired = adata.uns["markers_and_de"]["ccc"]["liana_paired_rescore"]["sex@MASLD=yes"]
    assert "edge_scores" in paired
    assert "route_scores" in paired
    assert "edge_missingness" in paired
    assert "route_missingness" in paired
    assert "edge_effects" in paired
    assert "route_effects" in paired
    assert "liana_paired_route_dotplot__female_vs_male__MASLD_yes" in artifact_stems
    assert "liana_paired_lr_edge_strip__female_vs_male__MASLD_yes" in artifact_stems
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_lr_edge_scores.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_route_scores.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_lr_edge_missingness.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_route_missingness.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_lr_edge_effects.tsv"))
    assert list((tmp_path / "out" / "tables").glob("**/liana_paired_route_effects.tsv"))
    settings_paths = list((tmp_path / "out" / "tables").glob("**/liana_paired_settings.tsv"))
    assert settings_paths
    assert "min_scored_donors_per_group\t1" in settings_paths[0].read_text()


def test_prepare_liana_plot_df_uses_cnn_tokens_from_display_map() -> None:
    df = pd.DataFrame(
        {
            "source": ["0", "1"],
            "target": ["1", "0"],
            "value": [1, 2],
        }
    )
    display_map = {"0": "C00: Kupffer cells", "1": "C01: T cells"}

    got = _prepare_liana_plot_df(df, display_map=display_map)

    assert got["source"].tolist() == ["C00", "C01"]
    assert got["target"].tolist() == ["C01", "C00"]


def test_normalize_liana_candidate_events_prefers_tokens_and_composite_labels() -> None:
    candidate_df = pd.DataFrame(
        {
            "source": ["vfat C05: Adipo A"],
            "target": ["liv C00: Hep 1"],
            "source_token": ["C05"],
            "target_token": ["C00"],
            "ligand_complex": ["ADIPOQ"],
            "receptor_complex": ["ADIPOR2"],
        }
    )
    display_map = {"C05": "C05: Adipo A", "C00": "C00: Hep 1"}

    got = md_mod._normalize_liana_candidate_events(
        candidate_df,
        display_map=display_map,
        valid_group_tokens=("C05", "C00"),
    )

    assert got["source_token"].tolist() == ["C05"]
    assert got["target_token"].tolist() == ["C00"]
    assert got["source_label"].tolist() == ["C05: Adipo A"]
    assert got["target_label"].tolist() == ["C00: Hep 1"]


def test_normalize_mebocost_candidate_events_prefers_tokens_and_composite_labels() -> None:
    candidate_df = pd.DataFrame(
        {
            "source": ["vfat C05: Adipo A"],
            "target": ["liv C00: Hep 1"],
            "source_token": ["C05"],
            "target_token": ["C00"],
            "HMDB_ID": ["HMDB0001"],
            "metabolite": ["HMDB0001"],
            "sensor_gene": ["CD36"],
            "sensor": ["CD36"],
        }
    )
    display_map = {"C05": "C05: Adipo A", "C00": "C00: Hep 1"}

    got = md_mod._normalize_mebocost_candidate_events(
        candidate_df,
        display_map=display_map,
        valid_group_tokens=("C05", "C00"),
    )

    assert got["source_token"].tolist() == ["C05"]
    assert got["target_token"].tolist() == ["C00"]
    assert got["source_label"].tolist() == ["C05: Adipo A"]
    assert got["target_label"].tolist() == ["C00: Hep 1"]


def test_liana_plot_color_map_uses_cnn_tokens() -> None:
    adata = ad.AnnData(X=np.zeros((2, 1)))
    adata.obs["leiden__r5"] = ["0", "1"]
    adata.uns["leiden__r5_colors"] = ["#112233", "#445566"]
    display_map = {"0": "C00: Kupffer cells", "1": "C01: T cells"}

    got = _liana_plot_color_map(
        adata,
        cluster_key="leiden__r5",
        display_map=display_map,
        round_id=None,
        raw_labels=["0", "1"],
    )

    assert got == {"C00": "#112233", "C01": "#445566"}


def test_liana_family_label_uses_pretty_label_tail() -> None:
    display_map = {"0": "C00: Kupffer cells", "1": "C01 - T cells"}

    assert _liana_family_label("0", display_map) == "Kupffer cells"
    assert _liana_family_label("1", display_map) == "T cells"


def test_run_liana_ccc_emits_comparison_heatmap_for_multiple_runs(monkeypatch, tmp_path: Path) -> None:
    class _FakeAggregate:
        def __call__(self, adata, **kwargs):
            return pd.DataFrame(
                {
                    "source": ["C00", "C01"],
                    "target": ["C01", "C00"],
                    "ligand_complex": ["LIG1", "LIG2"],
                    "receptor_complex": ["REC1", "REC2"],
                    "magnitude_rank": [0.001, 0.01],
                    "specificity_rank": [0.002, 0.02],
                }
            )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(
            rank_aggregate=_FakeAggregate(),
            AggregateClass=lambda meta, methods: _FakeAggregate(),
            aggregate_meta=object(),
        ),
        method=SimpleNamespace(
            cellphonedb=lambda *args, **kwargs: pd.DataFrame(),
            connectome=lambda *args, **kwargs: pd.DataFrame(),
            natmi=lambda *args, **kwargs: pd.DataFrame(),
            singlecellsignalr=lambda *args, **kwargs: pd.DataFrame(),
            logfc=lambda *args, **kwargs: pd.DataFrame(),
        ),
    )
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    adata = ad.AnnData(X=np.ones((4, 2)))
    adata.var_names = ["G1", "G2"]
    adata.layers["counts_cb"] = adata.X.copy()
    adata.obs["leiden__r5"] = ["C00", "C00", "C01", "C01"]
    adata.obs["cluster_label__r5"] = ["C00: Kupffer", "C00: Kupffer", "C01: T cells", "C01: T cells"]
    adata.obs["sex"] = ["female", "female", "male", "male"]
    adata.uns["cluster_rounds"] = {
        "r5": {
            "labels_obs_key": "leiden__r5",
            "cluster_display_map": {"C00": "C00: Kupffer", "C01": "C01: T cells"},
        }
    }

    calls: list[str] = []

    def _fake_plot(*args, **kwargs):
        stem = kwargs.get("stem", "unknown")
        calls.append(str(stem))
        return []

    monkeypatch.setattr(md_mod.io_utils, "load_dataset", lambda path: adata)
    monkeypatch.setattr(md_mod.io_utils, "save_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(md_mod.plot_utils, "persist_plot_artifacts", lambda artifacts: None)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_source_target_heatmap", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_send_receive_summary", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_circos", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_top_interactions", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_top_interactions_by_family", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_top_interactions_by_target_cluster", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_heatmap_grid", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_circos_grid", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_alluvial_grid", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_split_top_interactions", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_split_family_counts", _fake_plot)
    monkeypatch.setattr(md_mod.plot_utils, "plot_liana_condition_split_target_clusters", _fake_plot)

    cfg = SimpleNamespace(
        input_path=tmp_path / "input.zarr",
        output_dir=tmp_path / "out",
        output_name="adata.ccc_liana",
        save_h5ad=False,
        n_jobs=1,
        logfile=tmp_path / "out" / "logs" / "markers-and-de.ccc.liana.log",
        make_figures=True,
        figdir_name="figures",
        figure_formats=["png"],
        groupby=None,
        label_source="pretty",
        round_id="r5",
        ccc_condition_key="sex",
        ccc_condition_keys=("sex",),
        ccc_condition_values=("female", "male"),
        ccc_compare_levels=(),
        liana_resource="consensus",
        liana_methods=("rank_aggregate",),
        liana_expr_prop=0.1,
        liana_use_raw=False,
        liana_layer=None,
        liana_n_perms=None,
        liana_seed=42,
        liana_return_all_lrs=False,
        liana_top_n=25,
        liana_plot_top_n=10,
    )

    run_liana_ccc(cfg)

    assert "liana_source_target_compare_heatmap" in calls
    assert "liana_source_target_compare_mean_score_heatmap" in calls
    assert "liana_circos_by_condition" in calls
    assert "liana_circos_mean_score_by_condition" in calls
    assert "liana_source_target_alluvial_by_condition" in calls
    assert "liana_source_target_mean_score_alluvial_by_condition" in calls
    assert "liana_top_rank_aggregate_route_families" in calls
    assert "liana_top_rank_aggregate_by_condition" in calls
    assert "liana_top_rank_aggregate_route_families_by_condition" in calls
    assert "liana_top_rank_aggregate_target_clusters_by_condition" in calls
    assert "liana_top_rank_aggregate_target_cluster_share_by_condition" in calls
    assert "liana_top_rank_aggregate_target_cluster_mean_score_by_condition" in calls
