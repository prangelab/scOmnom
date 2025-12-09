# src/scomnom/qc_and_filter.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from .config import QCFilterConfig
from . import io_utils
from . import plot_utils

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# QC metric computation (sparse-safe)
# ---------------------------------------------------------------------
def compute_qc_metrics(adata: ad.AnnData, cfg: QCFilterConfig) -> ad.AnnData:
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        LOGGER.warning("X is dense — QC may be slow and memory intensive.")

    n_cells, n_genes = X.shape

    # Gene categories
    mt_prefix = getattr(cfg, "mt_prefix", "MT-")
    ribo_prefixes = getattr(cfg, "ribo_prefixes", ["RPL", "RPS"])
    hb_regex = getattr(cfg, "hb_regex", "^(HB[AB]|HBA|HBB)")

    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(ribo_prefixes))
    adata.var["hb"] = adata.var_names.str.contains(hb_regex, regex=True)

    mt_idx = np.where(adata.var["mt"].values)[0]
    ribo_idx = np.where(adata.var["ribo"].values)[0]
    hb_idx = np.where(adata.var["hb"].values)[0]

    LOGGER.info("Computing sparse per-cell QC metrics...")

    total_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes_by_counts = np.diff(X.indptr)

    def pct_from_idx(idx):
        if len(idx) == 0:
            return np.zeros(n_cells)
        vals = np.asarray(X[:, idx].sum(axis=1)).ravel()
        return vals / np.maximum(total_counts, 1)

    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    adata.obs["pct_counts_mt"] = pct_from_idx(mt_idx) * 100
    adata.obs["pct_counts_ribo"] = pct_from_idx(ribo_idx) * 100
    adata.obs["pct_counts_hb"] = pct_from_idx(hb_idx) * 100

    LOGGER.info("Computing sparse per-gene QC metrics...")

    n_cells_by_counts = np.diff(X.tocsc().indptr)
    total_counts_gene = np.asarray(X.sum(axis=0)).ravel()
    mean_counts = total_counts_gene / max(n_cells, 1)
    pct_dropout = 100 * (1 - (n_cells_by_counts / max(n_cells, 1)))

    adata.var["n_cells_by_counts"] = n_cells_by_counts
    adata.var["mean_counts"] = mean_counts
    adata.var["total_counts"] = total_counts_gene
    adata.var["pct_dropout_by_counts"] = pct_dropout

    adata.uns["qc_metrics"] = {
        "qc_vars": ["mt", "ribo", "hb"],
        "percent_top": {},
        "log1p": False,
        "raw_qc_metrics": {},
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    }

    return adata


# ---------------------------------------------------------------------
# Sparse filtering
# ---------------------------------------------------------------------
def sparse_filter_cells_and_genes(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
) -> ad.AnnData:
    import psutil
    import scipy.sparse as sp

    LOGGER.info(
        "[Filtering] Start: %d cells × %d genes (RSS=%.2f GB)",
        adata.n_obs,
        adata.n_vars,
        psutil.Process().memory_info().rss / 1024**3,
    )

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    X = adata.X

    # Cell filtering
    gene_counts = np.diff(X.indptr)
    cell_mask = gene_counts >= min_genes
    if cell_mask.sum() == 0:
        raise ValueError(f"All cells removed by min_genes={min_genes}.")
    adata = adata[cell_mask].copy()
    X = adata.X

    # Gene filtering
    gene_nnz = np.bincount(X.indices, minlength=adata.n_vars)
    gene_mask = gene_nnz >= min_cells
    if gene_mask.sum() == 0:
        raise ValueError(f"All genes removed by min_cells={min_cells}.")
    adata = adata[:, gene_mask].copy()

    return adata


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------
def run_qc_and_filter(cfg: QCFilterConfig, logfile: Optional[Path] = None) -> ad.AnnData:
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
