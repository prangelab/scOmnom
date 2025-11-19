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

    # Containers
    raw_map = cr_map = cb_map = None
    raw_unfiltered_reads = raw_filtered_reads = None
    cr_filtered_reads = cb_reads = None

    # RAW (raw-unfiltered + raw-filtered reads)
    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading RAW matrices (raw + knee+GMM filtered)...")
        raw_map, raw_filtered_reads, raw_unfiltered_reads = io_utils.load_raw_data(
            cfg,
            record_pre_filter_counts=True,
            plot_dir=Path("cell_qc"),
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
    # ----------------------

    if raw_unfiltered_reads is not None:
        # ====================================================
        # Case 1 — RAW is present → full set of comparisons
        # ====================================================
        if raw_filtered_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=raw_unfiltered_reads,
                other_counts=raw_filtered_reads,
                ref_label="raw-unfiltered",
                other_label="raw-filtered",
                figdir=figdir,
                stem="reads_raw_unfiltered_vs_raw_filtered",
            )

        if cr_filtered_reads is not None:
            plot_utils.plot_read_comparison(
                ref_counts=raw_unfiltered_reads,
                other_counts=cr_filtered_reads,
                ref_label="raw-unfiltered",
                other_label="cellranger-filtered",
                figdir=figdir,
                stem="reads_raw_unfiltered_vs_cellranger_filtered",
            )

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

    else:
        # ================================================
        # Case 2 — No RAW present
        # ================================================
        if cr_filtered_reads is not None and cb_reads is not None:
            # CellRanger vs CellBender only
            plot_utils.plot_read_comparison(
                ref_counts=cr_filtered_reads,
                other_counts=cb_reads,
                ref_label="cellranger-filtered",
                other_label="cellbender",
                figdir=figdir,
                stem="reads_cellranger_filtered_vs_cellbender",
            )

        else:
            # ==================================================
            # Case 3 — Only one data source available
            # → fallback: unfiltered_cell_counts
            # ==================================================
            LOGGER.info("Only one data source available; plotting unfiltered cell counts.")

            # Pick whichever map exists
            sample_map_nonnull = raw_map or cr_map or cb_map
            if sample_map_nonnull is not None:
                import matplotlib.pyplot as plt
                adata_combined = ad.concat(
                    list(sample_map_nonnull.values()),
                    label="sample_id",
                    keys=list(sample_map_nonnull.keys()),
                    index_unique="_",
                    join="outer",
                )

                # Reuse final cell count plot but rename output
                counts = adata_combined.obs["sample_id"].value_counts().sort_index()
                mean_cells = counts.mean()
                total_cells = counts.sum()

                fig, ax = plt.subplots(figsize=(8, 4))
                counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")

                ax.axhline(mean_cells, linestyle="--", color="#1f4e79", linewidth=1.0)
                ax.grid(False)

                ax.set_ylabel("Cell count")
                ax.set_title("Unfiltered cell counts per sample")

                summary_text = f"Total cells: {total_cells:,}\nMean per sample: {mean_cells:,.0f}"
                ax.text(
                    0.02, 0.98,
                    summary_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
                )

                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()

                plot_utils.save_multi(
                    stem="unfiltered_cell_counts",
                    figdir=figdir,
                )

                plt.close(fig)

    LOGGER.info("Finished cell_qc")
