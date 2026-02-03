# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import pandas as pd

from scomnom import __version__
from . import io_utils, plot_utils
from .logging_utils import init_logging
from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
    compute_markers_celllevel,
    de_cluster_vs_rest_pseudobulk,
    resolve_group_key,
)

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal policy guard
# -----------------------------------------------------------------------------
_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK = 6


def _prune_uns_de(adata: ad.AnnData, store_key: str) -> None:
    """
    Reduce memory footprint of adata.uns[store_key] by keeping summaries
    and dropping large per-gene result tables.
    """
    if store_key not in adata.uns or not isinstance(adata.uns.get(store_key), dict):
        return

    block = adata.uns.get(store_key, {})

    # Cluster-vs-rest pseudobulk
    pb_cvr = block.get("pseudobulk_cluster_vs_rest", None)
    if isinstance(pb_cvr, dict):
        if "results" in pb_cvr:
            pb_cvr.pop("results", None)

    # Within-cluster pseudobulk multi
    pb_within = block.get("pseudobulk_condition_within_group_multi", None)
    if isinstance(pb_within, dict):
        # keep metadata, drop per-contrast results tables
        keys = list(pb_within.keys())
        for k in keys:
            if isinstance(pb_within.get(k), dict) and "results" in pb_within[k]:
                pb_within[k].pop("results", None)

    # Contrast-conditional (cell-level within-cluster)
    cc = block.get("contrast_conditional", None)
    if isinstance(cc, dict):
        if "results" in cc:
            cc.pop("results", None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _choose_counts_layer(
    adata: ad.AnnData,
    *,
    candidates: Sequence[str],
    allow_X_counts: bool,
) -> Optional[str]:
    """
    Choose counts layer by priority. Returns:
      - layer name if found
      - None if falling back to adata.X is allowed
    Raises if nothing found and allow_X_counts=False.
    """
    for layer in candidates:
        if layer in adata.layers:
            return str(layer)

    if allow_X_counts:
        return None

    raise RuntimeError(
        "markers_and_de (pseudobulk): no candidate counts layer found in adata.layers "
        f"(candidates={list(candidates)!r}) and allow_X_counts=False"
    )


def _n_unique_samples(adata: ad.AnnData, sample_key: str) -> int:
    return int(adata.obs[str(sample_key)].astype(str).nunique(dropna=True))


# -----------------------------------------------------------------------------
# Orchestrator 1: cluster-vs-rest
# -----------------------------------------------------------------------------
def run_cluster_vs_rest(cfg) -> ad.AnnData:
    """
    Cluster-vs-rest orchestrator.

    Runs:
      - cell-level markers (scanpy rank_genes_groups) if cfg.run in {"cell","both"}
      - pseudobulk cluster-vs-rest DE (DESeq2) if cfg.run in {"pseudobulk","both"}
        and guard passes (>= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK unique samples)

    Always:
      - saves dataset
      - writes only the tables that correspond to what actually ran
      - plots only what actually ran
      - records provenance without referencing undefined locals

    Notes:
      - This orchestrator does NOT handle within-cluster contrasts; that is split out.
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (cluster-vs-rest)...")

    # ----------------------------
    # I/O + figure setup
    # ----------------------------
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # ----------------------------
    # Resolve groupby + sample_key
    # ----------------------------
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
    )

    sample_key = (
        getattr(cfg, "sample_key", None)
        or getattr(cfg, "batch_key", None)
        or adata.uns.get("batch_key")
    )
    if sample_key is None:
        raise RuntimeError("cluster-vs-rest: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")
    if str(sample_key) not in adata.obs:
        raise RuntimeError(f"cluster-vs-rest: sample_key={sample_key!r} not found in adata.obs")

    LOGGER.info("cluster-vs-rest: groupby=%r, sample_key=%r", str(groupby), str(sample_key))

    # ----------------------------
    # Mode decoding
    # ----------------------------
    mode = str(getattr(cfg, "run", "both")).lower()
    run_cell = mode in ("cell", "both")
    run_pb_requested = mode in ("pseudobulk", "both")

    # ----------------------------
    # Predefine locals (avoid NameErrors)
    # ----------------------------
    positive_only = bool(getattr(cfg, "positive_only", True))

    markers_key: Optional[str] = None
    layer_candidates = list(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw")))
    counts_layer_used: Optional[str] = None

    n_samples_total = _n_unique_samples(adata, str(sample_key))
    run_pseudobulk = False

    pb_spec: Optional[PseudobulkSpec] = None
    pb_opts: Optional[PseudobulkDEOptions] = None

    # ----------------------------
    # 1) Cell-level markers (cluster-vs-rest)
    # ----------------------------
    if run_cell:
        markers_spec = CellLevelMarkerSpec(
            method=str(getattr(cfg, "markers_method", "wilcoxon")).lower(),
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
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            spec=markers_spec,
            key_added=markers_key,
            store=True,
        )
    else:
        LOGGER.info("cluster-vs-rest: skipping cell-level markers (run=%r).", mode)

    # ----------------------------
    # 2) Pseudobulk cluster-vs-rest DE (guarded)
    # ----------------------------
    if run_pb_requested:
        run_pseudobulk = n_samples_total >= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK
        if not run_pseudobulk:
            LOGGER.warning(
                "cluster-vs-rest: pseudobulk requested but disabled: only %d unique samples in %r (< %d).",
                n_samples_total,
                str(sample_key),
                _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK,
            )
        else:
            counts_layer_used = _choose_counts_layer(
                adata,
                candidates=layer_candidates,
                allow_X_counts=bool(getattr(cfg, "allow_X_counts", True)),
            )

            pb_spec = PseudobulkSpec(
                sample_key=str(sample_key),
                counts_layer=counts_layer_used,  # None -> uses adata.X
            )

            pb_opts = PseudobulkDEOptions(
                min_cells_per_sample_group=int(getattr(cfg, "min_cells_target", 20)),
                min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
                alpha=float(getattr(cfg, "alpha", 0.05)),
                shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
            )

            LOGGER.info(
                "cluster-vs-rest: pseudobulk DE (sample_key=%r, counts_layer=%r, min_cells_per_sample_group=%d).",
                pb_spec.sample_key,
                pb_spec.counts_layer,
                pb_opts.min_cells_per_sample_group,
            )

            _ = de_cluster_vs_rest_pseudobulk(
                adata,
                groupby=str(groupby),
                round_id=getattr(cfg, "round_id", None),
                spec=pb_spec,
                opts=pb_opts,
                store_key=str(getattr(cfg, "store_key", "scomnom_de")),
                store=True,
                n_jobs=int(getattr(cfg, "n_jobs", 1)),
            )
    else:
        LOGGER.info("cluster-vs-rest: skipping pseudobulk (run=%r).", mode)

    # ----------------------------
    # Provenance (safe)
    # ----------------------------
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "mode": "cluster-vs-rest",
            "groupby": str(groupby),
            "sample_key": str(sample_key),
            "run": mode,
            "cell_ran": bool(run_cell),
            "pseudobulk_requested": bool(run_pb_requested),
            "pseudobulk_enabled": bool(run_pseudobulk),
            "pseudobulk_guard_min_total_samples": int(_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK),
            "pseudobulk_n_unique_samples": int(n_samples_total),
            "markers_key": str(markers_key) if markers_key else None,
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer_used,
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "positive_only_markers": bool(positive_only),
        }
    )

    # ----------------------------
    # Exports (only what ran)
    # ----------------------------
    if run_cell and markers_key:
        io_utils.export_rank_genes_groups_tables(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
            prefix="celllevel_markers",
        )
        io_utils.export_rank_genes_groups_excel(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
            filename="celllevel_markers.xlsx",
            max_genes=int(getattr(cfg, "markers_n_genes", 100)),
        )
    else:
        LOGGER.info("cluster-vs-rest: skipping cell-level exports (cell markers did not run).")

    store_key = str(getattr(cfg, "store_key", "scomnom_de"))

    if run_pseudobulk:
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            groupby=str(groupby),
            condition_key=None,
        )
        io_utils.export_pseudobulk_cluster_vs_rest_excel(
            adata,
            output_dir=output_dir,
            store_key=store_key,
        )
    else:
        LOGGER.info("cluster-vs-rest: skipping pseudobulk exports (pseudobulk disabled or not requested).")

    # ----------------------------
    # Plotting (only what ran)
    # ----------------------------
    if bool(getattr(cfg, "make_figures", True)):
        from . import de_plot_utils

        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
        top_n_genes = int(getattr(cfg, "plot_top_n_genes", 9))
        dotplot_top_n_genes = int(getattr(cfg, "plot_dotplot_top_n_genes", 15))
        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)
        ncols = int(getattr(cfg, "plot_umap_ncols", 3))

        if run_cell and markers_key:
            de_plot_utils.plot_marker_genes_ranksum(
                adata,
                groupby=str(groupby),
                markers_key=str(markers_key),
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                top_n_genes=top_n_genes,
                dotplot_top_n_genes=dotplot_top_n_genes,
                use_raw=use_raw,
                layer=layer,
                umap_ncols=ncols,
            )
        else:
            LOGGER.info("cluster-vs-rest: skipping cell-level marker plots (no markers computed).")

        if run_pseudobulk:
            de_plot_utils.plot_marker_genes_pseudobulk(
                adata,
                groupby=str(groupby),
                store_key=store_key,
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                top_n_genes=top_n_genes,
                dotplot_top_n_genes=dotplot_top_n_genes,
                use_raw=use_raw,
                layer=layer,
                umap_ncols=ncols,
            )
        else:
            LOGGER.info("cluster-vs-rest: skipping pseudobulk plots (pseudobulk disabled or not requested).")

    # ----------------------------
    # Save dataset
    # ----------------------------
    if bool(getattr(cfg, "prune_uns_de", False)):
        _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (cluster-vs-rest).")
    return adata


def run_within_cluster(cfg) -> ad.AnnData:
    """
    Within-cluster contrasts orchestrator.

    Runs (depending on cfg.run):
      - cell-level within-cluster contrasts via contrast_conditional_markers
        if cfg.run in {"cell","both"}  (NO pseudobulk guard)
      - pseudobulk within-cluster DE (DESeq2) if cfg.run in {"pseudobulk","both"}
        and guard passes (>= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK unique samples)

    Always:
      - saves dataset
      - writes only the tables that correspond to what actually ran
      - plots only what actually ran
      - records provenance without referencing undefined locals
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (within-cluster)...")

    # ----------------------------
    # I/O + figure setup
    # ----------------------------
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # ----------------------------
    # Resolve groupby + sample_key
    # ----------------------------
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
    )

    sample_key = (
        getattr(cfg, "sample_key", None)
        or getattr(cfg, "batch_key", None)
        or adata.uns.get("batch_key")
    )
    if sample_key is None:
        raise RuntimeError("within-cluster: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")
    if str(sample_key) not in adata.obs:
        raise RuntimeError(f"within-cluster: sample_key={sample_key!r} not found in adata.obs")

    # ----------------------------
    # Condition key (required)
    # ----------------------------
    condition_key = getattr(cfg, "condition_key", None)
    if not condition_key:
        raise RuntimeError("within-cluster: condition_key is required.")
    if str(condition_key) not in adata.obs:
        raise RuntimeError(f"within-cluster: condition_key={condition_key!r} not found in adata.obs")

    # Optional explicit contrasts (e.g. ["M_vs_F"])
    condition_contrasts = list(getattr(cfg, "condition_contrasts", [])) or None

    LOGGER.info(
        "within-cluster: groupby=%r, condition_key=%r, sample_key=%r, contrasts=%r",
        str(groupby),
        str(condition_key),
        str(sample_key),
        condition_contrasts,
    )

    # ----------------------------
    # Mode decoding
    # ----------------------------
    mode = str(getattr(cfg, "run", "both")).lower()
    run_cell_requested = mode in ("cell", "both")
    run_pb_requested = mode in ("pseudobulk", "both")

    # ----------------------------
    # Predefine locals (avoid NameErrors)
    # ----------------------------
    positive_only = bool(getattr(cfg, "positive_only", True))
    layer_candidates = list(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw")))
    counts_layer_used: Optional[str] = None

    n_samples_total = _n_unique_samples(adata, str(sample_key))
    pseudobulk_enabled = n_samples_total >= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK

    pb_spec: Optional[PseudobulkSpec] = None
    pb_opts: Optional[PseudobulkDEOptions] = None

    store_key = str(getattr(cfg, "store_key", "scomnom_de"))

    # ----------------------------
    # 1) Pseudobulk within-cluster DE (guarded)
    # ----------------------------
    ran_pseudobulk = False
    if run_pb_requested:
        if not pseudobulk_enabled:
            LOGGER.warning(
                "within-cluster: pseudobulk requested but disabled: only %d unique samples in %r (< %d).",
                n_samples_total,
                str(sample_key),
                _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK,
            )
        else:
            counts_layer_used = _choose_counts_layer(
                adata,
                candidates=layer_candidates,
                allow_X_counts=bool(getattr(cfg, "allow_X_counts", True)),
            )

            pb_spec = PseudobulkSpec(
                sample_key=str(sample_key),
                counts_layer=counts_layer_used,  # None -> uses adata.X
            )

            pb_opts = PseudobulkDEOptions(
                min_cells_per_sample_group=int(getattr(cfg, "min_cells_condition", 20)),
                min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
                alpha=float(getattr(cfg, "alpha", 0.05)),
                shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
            )

            groups = pd.Index(pd.unique(adata.obs[str(groupby)].astype(str))).sort_values()
            LOGGER.info(
                "within-cluster: pseudobulk DE across %d groups (groupby=%r) for condition_key=%r",
                len(groups),
                str(groupby),
                str(condition_key),
            )

            from .de_utils import de_condition_within_group_pseudobulk_multi

            for g in groups:
                _ = de_condition_within_group_pseudobulk_multi(
                    adata,
                    group_value=str(g),
                    groupby=str(groupby),
                    round_id=getattr(cfg, "round_id", None),
                    condition_key=str(condition_key),
                    spec=pb_spec,
                    opts=pb_opts,
                    contrasts=condition_contrasts,
                    store_key=store_key,
                    store=True,
                    n_cpus=int(getattr(cfg, "n_jobs", 1)),
                )

            ran_pseudobulk = True
    else:
        LOGGER.info("within-cluster: skipping pseudobulk (run=%r).", mode)

    # ----------------------------
    # 2) Cell-level within-cluster contrasts (NO pseudobulk guard)
    # ----------------------------
    ran_cell_contrast = False
    if run_cell_requested:
        from .de_utils import ContrastConditionalSpec, contrast_conditional_markers

        # What defines the levels to contrast at cell-level:
        # prefer explicit cfg.contrast_key; else condition_key.
        contrast_key = getattr(cfg, "contrast_key", None) or str(condition_key)

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
            pb_spec=pb_spec,  # may be None; function should tolerate (cell-only mode)
            store_key=store_key,
            store=True,
        )

        # tables for contrast-conditional markers
        io_utils.export_contrast_conditional_markers_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
        )
        ran_cell_contrast = True
    else:
        LOGGER.info("within-cluster: skipping cell-level within-cluster contrasts (run=%r).", mode)

    # ----------------------------
    # Provenance (safe)
    # ----------------------------
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "mode": "within-cluster",
            "groupby": str(groupby),
            "condition_key": str(condition_key),
            "condition_contrasts": list(condition_contrasts) if condition_contrasts else None,
            "sample_key": str(sample_key),
            "run": mode,
            "cell_requested": bool(run_cell_requested),
            "cell_ran": bool(ran_cell_contrast),
            "pseudobulk_requested": bool(run_pb_requested),
            "pseudobulk_enabled": bool(pseudobulk_enabled and run_pb_requested),
            "pseudobulk_ran": bool(ran_pseudobulk),
            "pseudobulk_guard_min_total_samples": int(_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK),
            "pseudobulk_n_unique_samples": int(n_samples_total),
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer_used,
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "positive_only_markers": bool(positive_only),
        }
    )

    # ----------------------------
    # Pseudobulk exports (only if actually ran)
    # ----------------------------
    if ran_pseudobulk:
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            groupby=str(groupby),
            condition_key=str(condition_key),
        )
        io_utils.export_pseudobulk_condition_within_cluster_excel(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            condition_key=str(condition_key),
        )
    else:
        LOGGER.info("within-cluster: skipping pseudobulk exports (not run).")

    # ----------------------------
    # Plotting (only what ran)
    # ----------------------------
    if bool(getattr(cfg, "make_figures", True)):
        from . import de_plot_utils

        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)

        # Condition plots only if pseudobulk ran
        if ran_pseudobulk:
            de_plot_utils.plot_condition_within_cluster_all(
                adata,
                cluster_key=str(groupby),
                condition_key=str(condition_key),
                store_key=store_key,
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                dotplot_top_n=15,
                violin_top_n=9,
                heatmap_top_n=25,
                use_raw=use_raw,
                layer=layer,
            )
        else:
            LOGGER.info("within-cluster: skipping condition plots (pseudobulk not run).")

        # (Optional) If you add dedicated plots for contrast_conditional_markers later, hook here.
        if ran_cell_contrast:
            LOGGER.info("within-cluster: cell-level contrast-conditional markers ran (tables exported).")

    # ----------------------------
    # Save dataset
    # ----------------------------
    if bool(getattr(cfg, "prune_uns_de", False)):
        _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (within-cluster).")
    return adata
