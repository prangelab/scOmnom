# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .logging_utils import init_logging

from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
    compute_markers_celllevel,
    de_cluster_vs_rest_pseudobulk,
    de_condition_within_group_pseudobulk,
    resolve_group_key,
)


LOGGER = logging.getLogger(__name__)


def run_markers_and_de(cfg) -> ad.AnnData:
    """
    Orchestrator wrapper for CLI / pipeline runs.

    IMPORTANT:
      - Notebook users should call functions in scomnom.de directly.
      - This orchestrator is the *only* place that should load/save AnnData.

    Expected cfg fields (minimal):
      - input_path: Path
      - output_dir: Path
      - logfile: Optional[Path]
      - figdir_name: str, figure_formats: list[str], make_figures: bool (optional)
      - batch_key/sample_key, groupby/round selection knobs (optional)
      - de settings: min_cells, alpha, condition_key, etc. (optional)
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers_and_de orchestrator")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # Resolve groupby key (round-aware by default)
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
    )

    sample_key = getattr(cfg, "sample_key", None) or getattr(cfg, "batch_key", None) or adata.uns.get("batch_key")
    if sample_key is None:
        raise RuntimeError("markers_and_de: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")

    LOGGER.info("markers_and_de: groupby=%r, sample_key=%r", groupby, sample_key)

    # Markers
    markers_spec = CellLevelMarkerSpec(
        method=str(getattr(cfg, "markers_method", "wilcoxon")),
        n_genes=int(getattr(cfg, "markers_n_genes", 100)),
        use_raw=bool(getattr(cfg, "markers_use_raw", False)),
        layer=getattr(cfg, "markers_layer", None),
        rankby_abs=bool(getattr(cfg, "markers_rankby_abs", True)),
        max_cells_per_group=int(getattr(cfg, "markers_downsample_max_per_group", 2000)),
        random_state=int(getattr(cfg, "random_state", 42)),
    )

    markers_key = str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon"))
    compute_markers_celllevel(
        adata,
        groupby=groupby,
        round_id=getattr(cfg, "round_id", None),
        spec=markers_spec,
        key_added=markers_key,
        store=True,
    )

    # DE: cluster vs rest
    # Choose first available layer from cfg.counts_layers; else fall back to X if allowed.
    layer_candidates = list(getattr(cfg, "counts_layers", ["counts_cb", "counts_raw"]))
    counts_layer = None
    for layer in layer_candidates:
        if layer in adata.layers:
            counts_layer = layer
            break

    if counts_layer is None and not bool(getattr(cfg, "allow_X_counts", True)):
        raise RuntimeError(
            f"markers_and_de: none of counts_layers found in adata.layers: {layer_candidates}, and allow_X_counts=False"
        )

    pb_spec = PseudobulkSpec(
        sample_key=str(sample_key),
        counts_layer=counts_layer,  # can be None -> uses adata.X
    )

    pb_opts = PseudobulkDEOptions(
        min_cells_per_sample_group=int(getattr(cfg, "min_cells_target", 20)),
        min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
        alpha=float(getattr(cfg, "alpha", 0.05)),
        shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
    )

    marker_genes_all_clusters = de_cluster_vs_rest_pseudobulk(
        adata,
        groupby=groupby,
        round_id=getattr(cfg, "round_id", None),
        spec=pb_spec,
        opts=pb_opts,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        store=True,
    )

    # Optional condition DE
    condition_key = getattr(cfg, "condition_key", None)
    if condition_key:
        reference = getattr(cfg, "condition_reference", None)
        if reference is None:
            raise RuntimeError("markers_and_de: condition_reference must be set when condition_key is provided.")

        # per-cluster condition DE
        groups = pd.Index(pd.unique(adata.obs[groupby].astype(str))).sort_values()
        conditional_de_genes_all_clusters = {}

        cond_opts = PseudobulkDEOptions(
            min_cells_per_sample_group=int(getattr(cfg, "min_cells_condition", 20)),
            min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
            alpha=float(getattr(cfg, "alpha", 0.05)),
            shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
        )

        for g in groups:
            res = de_condition_within_group_pseudobulk(
                adata,
                group_value=str(g),
                groupby=groupby,
                round_id=getattr(cfg, "round_id", None),
                condition_key=str(condition_key),
                reference=str(reference),
                spec=pb_spec,
                opts=cond_opts,
                store_key=str(getattr(cfg, "store_key", "scomnom_de")),
                store=True,
            )
            conditional_de_genes_all_clusters[str(g)] = res
    else:
        conditional_de_genes_all_clusters = None

    # Minimal provenance
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "groupby": str(groupby),
            "sample_key": str(sample_key),
            "markers_key": str(markers_key),
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer,
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "tables": {
                "cluster_vs_rest_clusters": sorted(list(marker_genes_all_clusters.keys()))
                if hasattr(marker_genes_all_clusters, "to_dict")
                else None,
                "condition_within_cluster_clusters": sorted(
                list(conditional_de_genes_all_clusters.keys())) if conditional_de_genes_all_clusters else None
                if (conditional_de_genes_all_clusters is not None and hasattr(conditional_de_genes_all_clusters, "to_dict"))
                else None,
            },
        }
    )

    # ------------------------------------------------------------------
    # Write tables
    # ------------------------------------------------------------------
    # Markers: cell-level rank_genes_groups -> CSVs
    io_utils.export_rank_genes_groups_tables(
        adata,
        key_added=markers_key,
        output_dir=output_dir,
        groupby=str(groupby),
        prefix="celllevel_markers",
    )

    # Markers: cell-level rank_genes_groups -> XLSX
    io_utils.export_rank_genes_groups_excel(
        adata,
        key_added=markers_key,
        output_dir=output_dir,
        groupby=str(groupby),
        filename="celllevel_markers.xlsx",
        max_genes=int(getattr(cfg, "markers_n_genes", 100)),
    )

    # Pseudobulk cluster-vs-rest: CSV folder
    io_utils.export_pseudobulk_de_tables(
        adata,
        output_dir=output_dir,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        groupby=str(groupby),
        condition_key=None,
    )

    # Pseudobulk cluster-vs-rest: single XLSX
    io_utils.export_pseudobulk_cluster_vs_rest_excel(
        adata,
        output_dir=output_dir,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
    )

    if condition_key:
        # Condition within cluster: CSV folder
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            groupby=str(groupby),
            condition_key=str(condition_key),
        )

        # Condition within cluster: single XLSX
        io_utils.export_pseudobulk_condition_within_cluster_excel(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            condition_key=str(condition_key),
        )

    # ------------------------------------------------------------------
    # Plotting (CLI behavior: create figs + save via save_multi)
    # ------------------------------------------------------------------
    from . import de_plot_utils

    if bool(getattr(cfg, "make_figures", True)):
        store_key = str(getattr(cfg, "store_key", "scomnom_de"))
        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))

        # how many genes to include in expression plots
        top_n_per_cluster = int(getattr(cfg, "plot_top_n_per_cluster", 10))
        max_genes_total = int(getattr(cfg, "plot_max_genes_total", 80))

        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)
        ncols = int(getattr(cfg, "plot_umap_ncols", 3))

        figroot = figdir / "markers_and_de"

        # ---------------------------
        # helpers
        # ---------------------------
        def _select_top_genes(df, *, padj_thresh: float, top_n: int) -> list[str]:
            if df is None or getattr(df, "empty", True):
                return []
            if "gene" not in df.columns or "padj" not in df.columns or "log2FoldChange" not in df.columns:
                return []
            tmp = df[["gene", "padj", "log2FoldChange"]].copy()
            tmp["gene"] = tmp["gene"].astype(str)
            tmp["padj"] = pd.to_numeric(tmp["padj"], errors="coerce")
            tmp["log2FoldChange"] = pd.to_numeric(tmp["log2FoldChange"], errors="coerce")
            tmp = tmp.dropna(subset=["gene", "padj", "log2FoldChange"])
            tmp = tmp[tmp["padj"] < float(padj_thresh)]
            if tmp.empty:
                return []
            tmp["__abs_lfc"] = tmp["log2FoldChange"].abs()
            tmp = tmp.sort_values(["padj", "__abs_lfc"], ascending=[True, False])
            genes = tmp["gene"].head(int(top_n)).tolist()
            # unique preserve order
            out = []
            seen = set()
            for g in genes:
                if g and g not in seen:
                    out.append(g)
                    seen.add(g)
            return out

        def _append_unique(acc: list[str], genes: list[str], limit: int) -> list[str]:
            seen = set(acc)
            for g in genes:
                if g not in seen:
                    acc.append(g)
                    seen.add(g)
                if len(acc) >= limit:
                    break
            return acc

        # ---------------------------
        # 1) Cluster-vs-rest volcanoes + gene list
        # ---------------------------
        pb = adata.uns.get(store_key, {}).get("pseudobulk_cluster_vs_rest", {})
        results_by_cluster = pb.get("results", {}) if isinstance(pb, dict) else {}
        if not isinstance(results_by_cluster, dict):
            results_by_cluster = {}

        genes_for_expression_plots: list[str] = []

        if results_by_cluster:
            figdir_pb = figroot / "cluster_vs_rest"

            for cl, df_de in results_by_cluster.items():
                title = f"Cluster vs rest: {cl}"
                fig = de_plot_utils.volcano(
                    df_de,
                    padj_thresh=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    title=title,
                    show=False,
                )
                plot_utils.save_multi(
                    stem=f"volcano__cluster_vs_rest__{cl}",
                    figdir=figdir_pb,
                    fig=fig,
                )

                top_genes = _select_top_genes(df_de, padj_thresh=alpha, top_n=top_n_per_cluster)
                genes_for_expression_plots = _append_unique(
                    genes_for_expression_plots,
                    top_genes,
                    limit=max_genes_total,
                )
        else:
            LOGGER.warning("markers_and_de: no pseudobulk_cluster_vs_rest results found under adata.uns[%r].", store_key)

        # If we collected genes, make scanpy expression plots once
        if genes_for_expression_plots:
            figdir_expr = figroot / "cluster_vs_rest" / "expression"

            fig = de_plot_utils.dotplot_top_genes(
                adata,
                genes=genes_for_expression_plots,
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"dotplot__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            fig = de_plot_utils.heatmap_top_genes(
                adata,
                genes=genes_for_expression_plots,
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"heatmap__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            fig = de_plot_utils.umap_features_grid(
                adata,
                genes=genes_for_expression_plots,
                use_raw=use_raw,
                layer=layer,
                ncols=ncols,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"umap_features__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            fig = de_plot_utils.violin_genes(
                adata,
                genes=genes_for_expression_plots[: min(len(genes_for_expression_plots), 24)],
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"violin__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

        # ---------------------------
        # 2) Condition-within-cluster plots (if present)
        # ---------------------------
        cond_block = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
        if isinstance(cond_block, dict) and cond_block and condition_key:
            figdir_cond = figroot / f"condition_within_cluster__{condition_key}"

            # One volcano per stored contrast key
            for k, payload in cond_block.items():
                if not isinstance(payload, dict):
                    continue
                df_de = payload.get("results", None)
                if df_de is None:
                    continue

                title = str(k)
                fig = de_plot_utils.volcano(
                    df_de,
                    padj_thresh=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    title=title,
                    show=False,
                )
                plot_utils.save_multi(
                    stem=f"volcano__condition_within_cluster__{k}",
                    figdir=figdir_cond,
                    fig=fig,
                )
        elif condition_key:
            LOGGER.info("markers_and_de: no pseudobulk_condition_within_group results found for condition_key=%r.", condition_key)


    # Save
    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers_and_de orchestrator")
    return adata
