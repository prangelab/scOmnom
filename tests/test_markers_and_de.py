from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import scomnom.annotation_utils as au

from scomnom.composition_utils import _resolve_active_cluster_key
from scomnom.markers_and_de import (
    _run_namespace_for_round,
    _collect_pseudobulk_de_tables_from_dir,
    _collect_cell_contrast_tables_from_dir,
    _build_stats_matrix_from_tables,
    _load_de_enrichment_payload_from_tables,
    _load_module_definitions,
    _prune_uns_de,
    _compute_module_score_on_adata,
    _write_de_enrichment_tables,
    run_enrichment_cluster,
    run_module_score,
    run_liana_ccc,
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
    class _FakeMethod:
        def __init__(self, score_col: str):
            self.score_col = score_col

        def __call__(self, adata, **kwargs):
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

    monkeypatch.setattr("scomnom.markers_and_de.io_utils.load_dataset", lambda path: adata)
    monkeypatch.setattr(
        "scomnom.markers_and_de.io_utils.save_dataset",
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
        ccc_condition_values=("female", "male"),
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
    assert set(runs.keys()) == {"female", "male"}
    assert runs["female"]["primary_method"] == "rank_aggregate"
    assert isinstance(runs["female"]["top_interactions"], pd.DataFrame)
    assert isinstance(runs["female"]["source_target_summary"], pd.DataFrame)
    assert (tmp_path / "out" / "tables").exists()
