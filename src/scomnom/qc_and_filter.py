# src/scomnom/qc_and_filter.py

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from .config import QCFilterConfig
from . import io_utils
from . import plot_utils
from .logging_utils import init_logging


LOGGER = logging.getLogger(__name__)





# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------
def run_qc_and_filter(cfg: QCFilterConfig) -> ad.AnnData:
    init_logging(cfg.logfile)
    LOGGER.info("Starting qc_and_filter")

    # Set up figure saving
    if cfg.make_figures:
        plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    # ------------------------------------------------------------------
    # Load dataset using io_utils.load_dataset()
    # ------------------------------------------------------------------
    adata = io_utils.load_dataset(cfg.input_path)
    LOGGER.info("Loaded dataset for QC: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # ------------------------------------------------------------------
    # Infer batch key correctly
    # ------------------------------------------------------------------
    batch_key = io_utils.infer_batch_key(adata, cfg.batch_key)
    cfg.batch_key = batch_key
    LOGGER.info("Using batch_key='%s'", batch_key)

    # ------------------------------------------------------------------
    # Pre-filter QC
    # ------------------------------------------------------------------
    adata = compute_qc_metrics(adata, cfg)

    qc_df = pd.DataFrame(
        {
            "sample": adata.obs[batch_key].astype(str),
            "total_counts": adata.obs["total_counts"],
            "n_genes_by_counts": adata.obs["n_genes_by_counts"],
            "pct_counts_mt": adata.obs["pct_counts_mt"],
        }
    )

    counts_before = adata.obs[batch_key].value_counts().sort_index()

    if cfg.make_figures:
        plot_utils.run_qc_plots_pre_filter_df(qc_df, cfg)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    adata = sparse_filter_cells_and_genes(
        adata,
        min_genes=cfg.min_genes,
        min_cells=cfg.min_cells,
    )

    # MT filter
    if "pct_counts_mt" in adata.obs:
        mask = adata.obs["pct_counts_mt"] <= cfg.max_pct_mt
        adata = adata[mask].copy()
    else:
        LOGGER.warning("Missing pct_counts_mt; skipping mitochondrial filtering.")

    # Per-sample minimum filter
    if cfg.min_cells_per_sample > 0 and batch_key in adata.obs:
        counts = adata.obs[batch_key].value_counts()
        small = counts[counts < cfg.min_cells_per_sample].index
        if len(small) > 0:
            adata = adata[~adata.obs[batch_key].isin(small)].copy()

    counts_after = (
        adata.obs[batch_key].value_counts().sort_index()
        if batch_key in adata.obs
        else pd.Series(dtype=int)
    )

    # ------------------------------------------------------------------
    # Before/after plot
    # ------------------------------------------------------------------
    if cfg.make_figures and len(counts_before) > 0:
        figdir_qc = cfg.figdir / "QC_plots"
        figdir_qc.mkdir(parents=True, exist_ok=True)

        after = counts_after.reindex(counts_before.index, fill_value=0).astype(int)

        df_counts = pd.DataFrame(
            {
                "sample": counts_before.index.astype(str),
                "before": counts_before.values,
                "after": after.values,
            }
        )
        df_counts["retained_pct"] = np.where(
            df_counts["before"] > 0,
            100 * df_counts["after"] / df_counts["before"],
            0.0,
        )

        plot_utils.barplot_before_after(
            df_counts,
            figpath=figdir_qc / "cell_counts_before_after.png",
            min_cells_per_sample=cfg.min_cells_per_sample,
        )

    # ------------------------------------------------------------------
    # Post-filter QC
    # ------------------------------------------------------------------
    adata = compute_qc_metrics(adata, cfg)

    if cfg.make_figures:
        plot_utils.run_qc_plots_postfilter(adata, cfg)
        plot_utils.plot_final_cell_counts(adata, cfg)

    adata.uns["batch_key"] = batch_key

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    out_zarr = cfg.output_dir / f"{cfg.output_name}.zarr"
    LOGGER.info("Saving filtered dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        out_h5ad = cfg.output_dir / f"{cfg.output_name}.h5ad"
        LOGGER.warning("Writing H5AD copy (loads data into RAM)")
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished qc_and_filter")
    return adata
