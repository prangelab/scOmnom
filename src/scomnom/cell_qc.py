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
    raw_map = cr_map = cb_map = None
    raw_unfiltered_reads = raw_filtered_reads = None
    cr_filtered_reads = cb_reads = None

    # RAW (raw-unfiltered + raw-filtered reads)
    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading RAW matrices (raw + knee+GMM filtered)...")
        raw_map, raw_filtered_reads, raw_unfiltered_reads = io_utils.load_raw_data(
            cfg,
            record_pre_filter_counts=True,  # ensures raw_unfiltered_reads is populated
        )

    # CellRanger filtered
    if cfg.filtered_sample_dir is not None:
        LOGGER.info("Loading CellRanger filtered matrices...")
        cr_map, cr_filtered_reads = io_utils.load_filtered_data(cfg)

    # CellBender
    if cfg.cellbender_dir is not None:
        LOGGER.info("Loading CellBender matrices...")
        cb_map, cb_reads = io_utils.load_cellbender_data(cfg)

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

    LOGGER.info("Finished cell_qc")
