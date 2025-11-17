from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad

from . import io_utils, plot_utils
from .config import CellQCConfig

LOGGER = logging.getLogger(__name__)


def _compute_cell_counts(sample_map: Dict[str, ad.AnnData]) -> Dict[str, int]:
    """Return per-sample cell counts from a sample -> AnnData mapping."""
    return {sample: int(adata.n_obs) for sample, adata in sample_map.items()}

def run_cell_qc(cfg: CellQCConfig) -> None:
    LOGGER.info("Starting cell_qc")

    # Prepare figure root
    plot_utils.setup_scanpy_figs(
        cfg.output_dir / cfg.figdir_name,
        cfg.figure_formats
    )
    figdir = (cfg.output_dir / cfg.figdir_name / "cell_qc").resolve()
    figdir.mkdir(parents=True, exist_ok=True)

    # Containers
    raw_filtered_reads = None       # after knee+GMM
    raw_unfiltered_reads = None     # before any filtering (true raw)
    cr_filtered_reads = None
    cb_reads = None
    cell_counts_per_dataset = {}

    # RAW
    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading RAW dataset...")
        raw_map, raw_filtered_reads, raw_unfiltered_reads = io_utils.load_raw_data(
            cfg,
            record_pre_filter_counts=True,
        )
        cell_counts_per_dataset["raw-filtered"] = _compute_cell_counts(raw_map)

    # CellRanger-filtered
    if cfg.filtered_sample_dir is not None:
        LOGGER.info("Loading CellRanger-filtered dataset...")
        cr_map, cr_filtered_reads = io_utils.load_filtered_data(cfg)
        cell_counts_per_dataset["cellranger-filtered"] = _compute_cell_counts(cr_map)

    # CellBender
    if cfg.cellbender_dir is not None:
        LOGGER.info("Loading CellBender dataset...")
        cb_map, cb_reads = io_utils.load_cellbender_data(cfg)
        cell_counts_per_dataset["cellbender"] = _compute_cell_counts(cb_map)

    # READ-COUNT COMPARISONS
    if raw_unfiltered_reads is not None:
        # Raw-unfiltered vs raw-filtered (knee+GMM)
        if raw_filtered_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=raw_unfiltered_reads,
                other_counts=raw_filtered_reads,
                ref_label="raw-unfiltered",
                other_label="raw-filtered",
                figdir=figdir,
                stem="reads_raw_unfiltered_vs_raw_filtered",
            )

        # Raw-unfiltered vs CellRanger-filtered
        if cr_filtered_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=raw_unfiltered_reads,
                other_counts=cr_filtered_reads,
                ref_label="raw-unfiltered",
                other_label="cellranger-filtered",
                figdir=figdir,
                stem="reads_raw_unfiltered_vs_cellranger_filtered",
            )

        # Raw-unfiltered vs CellBender
        if cb_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=raw_unfiltered_reads,
                other_counts=cb_reads,
                ref_label="raw-unfiltered",
                other_label="cellbender",
                figdir=figdir,
                stem="reads_raw_unfiltered_vs_cellbender",
            )
        if cr_filtered_reads is not None and cb_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=cr_filtered_reads,
                other_counts=cb_reads,
                ref_label="cellranger-filtered",
                other_label="cellbender",
                figdir=figdir,
                stem="reads_cellranger_filtered_vs_cellbender",
            )

    # CELL-COUNT PLOTS
    if cell_counts_per_dataset:
        plot_utils.plot_cell_counts_multi(
            cell_counts_per_dataset=cell_counts_per_dataset,
            figdir=figdir,
            stem="cell_counts_per_sample",
        )

    # MEDIAN + PERCENTILE (90th) READ COMPARISONS
    datasets_for_stats = {}

    # Build dict of sample → reads for each dataset
    if raw_filtered_reads is not None and raw_unfiltered_reads is not None:
        datasets_for_stats["raw-filtered"] = raw_filtered_reads

    if cr_filtered_reads is not None:
        datasets_for_stats["cellranger-filtered"] = cr_filtered_reads

    if cb_reads is not None:
        datasets_for_stats["cellbender"] = cb_reads

    # Need at least 2 datasets + raw-unfiltered as reference
    if raw_unfiltered_reads is not None and len(datasets_for_stats) >= 1:
        # Median read comparison
        plot_utils.plot_read_medians(
            ref_counts=raw_unfiltered_reads,
            datasets=datasets_for_stats,
            figdir=figdir,
            stem="reads_median_comparison",
        )

        # 90th percentile comparison
        plot_utils.plot_read_percentiles(
            ref_counts=raw_unfiltered_reads,
            datasets=datasets_for_stats,
            figdir=figdir,
            stem="reads_p90_comparison",
            pct=90.0,
        )

    LOGGER.info("Finished cell_qc")
