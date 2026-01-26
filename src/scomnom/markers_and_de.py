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
    resolve_group_key, de_condition_within_group_pseudobulk_multi,
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
    LOGGER.info("Starting markers-and-de...")

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
        min_pct=float(getattr(cfg, "min_pct", 0.25)),
        min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
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
        min_pct=float(getattr(cfg, "min_pct", 0.25)),
        min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
    )

    LOGGER.info(
        "Pseudobulk Markers: cluster-vs-rest "
        "(sample_key=%r, counts_layer=%r, min_cells_per_sample_group=%d, min_samples_per_level=%d)",
        pb_spec.sample_key,
        pb_spec.counts_layer,
        pb_opts.min_cells_per_sample_group,
        pb_opts.min_samples_per_level,
    )
    marker_genes_all_clusters = de_cluster_vs_rest_pseudobulk(
        adata,
        groupby=groupby,
        round_id=getattr(cfg, "round_id", None),
        spec=pb_spec,
        opts=pb_opts,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        store=True,
        n_cpus=int(getattr(cfg, "n_jobs", 1)),
    )

    # Optional condition DE
    condition_key = getattr(cfg, "condition_key", None)
    if condition_key:

        # per-cluster condition DE
        groups = pd.Index(pd.unique(adata.obs[groupby].astype(str))).sort_values()
        conditional_de_genes_all_clusters = {}

        cond_opts = PseudobulkDEOptions(
            min_cells_per_sample_group=int(getattr(cfg, "min_cells_condition", 20)),
            min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
            alpha=float(getattr(cfg, "alpha", 0.05)),
            shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
            min_pct=float(getattr(cfg, "min_pct", 0.25)),
            min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
        )

        condition_contrasts = list(getattr(cfg, "condition_contrasts", [])) or None

        for g in groups:
            _ = de_condition_within_group_pseudobulk_multi(
                adata,
                group_value=str(g),
                groupby=groupby,
                round_id=getattr(cfg, "round_id", None),
                condition_key=str(condition_key),
                spec=pb_spec,
                opts=cond_opts,
                contrasts=condition_contrasts,
                store_key=str(getattr(cfg, "store_key", "scomnom_de")),
                store=True,
                n_cpus=int(getattr(cfg, "n_jobs", 1)),
            )
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
    # Contrast-conditional mode (pairwise A vs B within each cluster)
    # ------------------------------------------------------------------
    if bool(getattr(cfg, "contrast_conditional_de", False)):
        from .de_utils import ContrastConditionalSpec, contrast_conditional_markers

        contrast_key = getattr(cfg, "contrast_key", None) or str(sample_key)
        cc_spec = ContrastConditionalSpec(
            contrast_key=str(contrast_key),
            methods=tuple(getattr(cfg, "contrast_methods", ("wilcoxon", "logreg"))),
            min_cells_per_level_in_cluster=int(getattr(cfg, "contrast_min_cells_per_level", 50)),
            max_cells_per_level_in_cluster=int(getattr(cfg, "contrast_max_cells_per_level", 2000)),
            min_total_counts=int(getattr(cfg, "contrast_min_total_counts", 10)),
            pseudocount=float(getattr(cfg, "contrast_pseudocount", 1.0)),
            cl_alpha=float(getattr(cfg, "contrast_cl_alpha", 0.05)),
            cl_min_abs_logfc=float(getattr(cfg, "contrast_cl_min_abs_logfc", 0.25)),
            lr_min_abs_coef=float(getattr(cfg, "contrast_lr_min_abs_coef", 0.25)),
            pb_min_abs_log2fc=float(getattr(cfg, "contrast_pb_min_abs_log2fc", 0.5)),
            random_state=int(getattr(cfg, "random_state", 42)),
            min_pct=float(getattr(cfg, "min_pct", 0.25)),
            min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
        )

        _ = contrast_conditional_markers(
            adata,
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            spec=cc_spec,
            pb_spec=pb_spec,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            store=True,
        )

        # write tables (CSV + XLSX + summary)
        io_utils.export_contrast_conditional_markers_tables(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        )

    # ------------------------------------------------------------------
    # Plotting (CLI behavior: create figs + save via save_multi)
    # ------------------------------------------------------------------
    if bool(getattr(cfg, "make_figures", True)):
        from . import de_plot_utils

        store_key = str(getattr(cfg, "store_key", "scomnom_de"))
        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
        top_n_genes = int(getattr(cfg, "plot_top_n_genes", 9))
        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)
        ncols = int(getattr(cfg, "plot_umap_ncols", 3))

        de_plot_utils.plot_marker_genes_pseudobulk(
            adata,
            groupby=str(groupby),
            store_key=store_key,
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            top_label_n=top_label_n,
            top_n_genes=top_n_genes,
            use_raw=use_raw,
            layer=layer,
            umap_ncols=ncols,
        )

        de_plot_utils.plot_marker_genes_ranksum(
            adata,
            groupby=str(groupby),
            markers_key=str(markers_key),
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            top_label_n=top_label_n,
            top_n_genes=top_n_genes,
            use_raw=use_raw,
            layer=layer,
            umap_ncols=ncols,
        )

        if condition_key:
            de_plot_utils.plot_condition_within_cluster_all(
                adata,
                cluster_key=str(groupby),
                condition_key=str(condition_key),
                store_key=store_key,
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                dotplot_top_n=9,
                violin_top_n=9,
                heatmap_top_n=25,
                use_raw=use_raw,
                layer=layer,
            )

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
