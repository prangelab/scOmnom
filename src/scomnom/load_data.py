# src/scomnom/load_data.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import pandas as pd

from .config import LoadDataConfig
from . import io_utils, plot_utils

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------
def add_metadata(
    adata: ad.AnnData,
    metadata_tsv: Path,
    sample_id_col: str,
) -> ad.AnnData:
    """
    Attach per-sample metadata from a TSV to adata.obs.

    Parameters
    ----------
    adata
        Merged AnnData; must have `sample_id_col` in .obs.
    metadata_tsv
        TSV with one row per sample and a column matching sample_id_col.
    sample_id_col
        Column in metadata_tsv and adata.obs used to join metadata.

    Notes
    -----
    - Does not overwrite the sample_id_col in obs.
    - Casts low-cardinality string columns to 'category'.
    """
    if metadata_tsv is None:
        raise RuntimeError("metadata_tsv is required but was None")

    if not metadata_tsv.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {metadata_tsv}")

    df = pd.read_csv(metadata_tsv, sep="\t")
    if sample_id_col not in df.columns:
        raise KeyError(
            f"Metadata TSV does not contain required column '{sample_id_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    df[sample_id_col] = df[sample_id_col].astype(str)

    obs_sample_ids = pd.Index(adata.obs[sample_id_col].astype(str))
    meta_sample_ids = pd.Index(df[sample_id_col])

    missing = obs_sample_ids.unique().difference(meta_sample_ids)
    if len(missing) > 0:
        raise ValueError(
            "Some sample IDs in adata.obs are missing in metadata_tsv:\n"
            f"  {list(missing)}"
        )

    # Join metadata onto obs using sample_id_col
    obs_col = adata.obs[sample_id_col].astype(str)
    temp = pd.DataFrame({sample_id_col: obs_col}, index=adata.obs_names)
    merged = temp.merge(df, on=sample_id_col, how="left")

    for col in df.columns:
        if col == sample_id_col:
            # never overwrite the sample_id/batch_key column itself
            continue

        adata.obs[col] = merged[col].values

        # Optional: cast low-cardinality strings to category
        if (
            adata.obs[col].dtype == object
            and adata.obs[col].nunique() < 0.1 * len(adata.obs)
        ):
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


def _validate_metadata_samples(
    metadata_tsv: Path,
    batch_key: str,
    loaded_samples: Dict[str, ad.AnnData],
) -> None:
    """
    Ensure metadata_tsv contains exactly one row per sample and matches loaded sample IDs.
    """
    df_meta = pd.read_csv(metadata_tsv, sep="\t")

    if batch_key not in df_meta.columns:
        raise KeyError(
            f"Metadata TSV must contain the batch key column '{batch_key}'. "
            f"Found columns: {list(df_meta.columns)}"
        )

    meta_samples = pd.Index(df_meta[batch_key].astype(str))
    loaded = pd.Index(list(loaded_samples.keys())).astype(str)

    missing_rows = loaded.difference(meta_samples)
    extra_rows = meta_samples.difference(loaded)

    if len(missing_rows) > 0:
        raise ValueError(
            "The following samples were found in the input data but "
            "are missing from metadata_tsv:\n"
            f"  {list(missing_rows)}"
        )

    if len(extra_rows) > 0:
        raise ValueError(
            "The following samples exist in metadata_tsv but were not found "
            "in the input data folders:\n"
            f"  {list(extra_rows)}"
        )

    if len(meta_samples) != len(loaded):
        raise ValueError(
            f"metadata_tsv has {len(meta_samples)} rows for batch_key='{batch_key}', "
            f"but {len(loaded)} samples were loaded."
        )


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------
def run_load_data(cfg: LoadDataConfig, logfile: Optional[Path] = None) -> ad.AnnData:
    """
    Load stage for the new pipeline:

      1. Load input matrices (raw, filtered, or CellBender).
         * For raw: perform CellRanger-like cell-calling (knee+GMM) via filter_raw_barcodes.
      2. (If raw) Compute per-sample fraction of reads in cells and readcount comparison plot.
      3. Infer batch_key from metadata TSV and validate sample set.
      4. Merge all samples into a single AnnData using union gene space (io_utils.merge_samples).
      5. Attach metadata to per-cell obs.
      6. Save merged dataset as Zarr (always) and optional H5AD.

    No QC filtering, doublet detection, HVG, PCA, or clustering happens here.
    Those are handled in downstream qc-and-filter / integrate modules.
    """
    LOGGER.info("Starting load_data")

    # ---------------------------------------------------------
    # Figure root
    # ---------------------------------------------------------
    if cfg.make_figures:
        plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Infer batch_key from metadata header
    # ---------------------------------------------------------
    if cfg.metadata_tsv is None:
        raise RuntimeError("metadata_tsv is required for load_data but was None")

    batch_key = io_utils.infer_batch_key_from_metadata_tsv(
        cfg.metadata_tsv,
        cfg.batch_key,
    )
    cfg.batch_key = batch_key
    LOGGER.info("Using batch_key='%s'", batch_key)

    # ---------------------------------------------------------
    # Load input data (exactly one source as enforced by config)
    # ---------------------------------------------------------
    sample_map: Dict[str, ad.AnnData] = {}
    raw_unfiltered_reads: Optional[Dict[str, float]] = None
    raw_filtered_reads: Optional[Dict[str, float]] = None

    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading RAW 10x matrices with CellRanger-like cell calling...")
        # Returns: raw_map (filtered by knee+GMM), filtered_read_counts, unfiltered_read_counts
        sample_map, raw_filtered_reads, raw_unfiltered_reads = io_utils.load_raw_data(
            cfg,
            record_pre_filter_counts=True,
        )

    elif cfg.filtered_sample_dir is not None:
        LOGGER.info("Loading CellRanger filtered matrices...")
        sample_map, cr_filtered_reads = io_utils.load_filtered_data(cfg)

    else:
        LOGGER.info("Loading CellBender matrices...")
        sample_map, cb_reads = io_utils.load_cellbender_data(cfg)

    if not sample_map:
        raise RuntimeError("load_data: no samples were loaded.")

    LOGGER.info("Loaded %d samples.", len(sample_map))

    # ---------------------------------------------------------
    # Readcount QC: fraction of reads in cells (RAW case only)
    # ---------------------------------------------------------
    if (
        cfg.make_figures
        and raw_unfiltered_reads is not None
        and raw_filtered_reads is not None
    ):
        figdir = cfg.figdir / "cell_qc"
        figdir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Plotting raw-unfiltered vs raw-filtered read counts "
            "(proxy for 'fraction of reads in cells')."
        )
        plot_utils.plot_read_comparison(
            ref_counts=raw_unfiltered_reads,
            other_counts=raw_filtered_reads,
            ref_label="raw-unfiltered",
            other_label="raw-filtered",
            figdir=figdir,
            stem="reads_raw_unfiltered_vs_raw_filtered",
        )

        # Log per-sample fractions
        LOGGER.info("Per-sample fraction of reads in cells (raw-filtered / raw-unfiltered):")
        for sample in sorted(raw_unfiltered_reads.keys()):
            total = float(raw_unfiltered_reads.get(sample, 0.0))
            in_cells = float(raw_filtered_reads.get(sample, 0.0))
            if total <= 0:
                frac = 0.0
            else:
                frac = 100.0 * in_cells / total
            LOGGER.info(
                "  %s: %.2f%% (%0.2e / %0.2e)",
                sample, frac, in_cells, total
            )

    # ---------------------------------------------------------
    # Validate metadata ↔ sample consistency
    # ---------------------------------------------------------
    LOGGER.info("Validating metadata vs loaded samples...")
    _validate_metadata_samples(cfg.metadata_tsv, batch_key, sample_map)

    # ---------------------------------------------------------
    # Merge samples using union gene space (disk-backed padding)
    # ---------------------------------------------------------
    tmp_dir = cfg.output_dir / "tmp_merge_load"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    merge_out = tmp_dir / "merged.zarr"

    LOGGER.info("Merging samples into a single AnnData (union gene space)...")
    adata = io_utils.merge_samples(
        sample_map,
        batch_key=cfg.batch_key,
        out_path=merge_out,
    )
    LOGGER.info(
        "Merged dataset: %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )

    # ---------------------------------------------------------
    # Attach metadata to each cell
    # ---------------------------------------------------------
    LOGGER.info("Adding per-sample metadata from %s", cfg.metadata_tsv)
    adata = add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    adata.uns["batch_key"] = cfg.batch_key

    # Optionally stash load-stage readcount metrics in .uns
    if raw_unfiltered_reads is not None:
        metrics = adata.uns.setdefault("load_data_metrics", {})
        metrics["raw_unfiltered_reads"] = raw_unfiltered_reads
        metrics["raw_filtered_reads"] = raw_filtered_reads

    # ---------------------------------------------------------
    # Save merged dataset
    # ---------------------------------------------------------
    out_zarr = cfg.output_dir / (cfg.output_name + ".zarr")
    LOGGER.info("Saving merged dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        h5ad_out = cfg.output_dir / (cfg.output_name + ".h5ad")
        LOGGER.warning(
            "Writing H5AD copy of merged dataset (this loads the full matrix into RAM)."
        )
        io_utils.save_dataset(adata, h5ad_out, fmt="h5ad")
        LOGGER.info("Saved H5AD dataset → %s", h5ad_out)

    LOGGER.info("Finished load_data")
    return adata
