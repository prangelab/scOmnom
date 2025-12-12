# src/scomnom/load_data.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import shutil
import anndata as ad
import pandas as pd

from .config import LoadDataConfig
from . import io_utils
from .logging_utils import init_logging

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
def run_load_data(cfg: LoadDataConfig) -> ad.AnnData:
    """
    Load stage for the new pipeline (load-only, no QC):

      1. Load input matrices (raw, filtered, or CellBender).
         * For raw: perform CellRanger-like cell-calling (knee+GMM)
           via io_utils.load_raw_data (as before).
      2. Infer batch_key from metadata TSV and validate sample set.
      3. Merge all samples into a single AnnData using union gene space
         (io_utils.merge_samples; disk-backed padding).
      4. Attach per-sample metadata to per-cell obs.
      5. Save merged dataset as Zarr (always) and optional H5AD.

    No QC filtering, doublet detection, HVG, PCA, or clustering happens here.
    Those are handled in downstream qc-and-filter / integrate modules.
    """
    LOGGER.info("Starting load_data")

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

    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading RAW 10x matrices with CellRanger-like cell calling...")
        # Returns: sample_map, raw_filtered_reads, raw_unfiltered_reads
        sample_map, _, _ = io_utils.load_raw_data(
            cfg,
            record_pre_filter_counts=False,
        )

    elif cfg.filtered_sample_dir is not None:
        LOGGER.info("Loading CellRanger filtered matrices...")
        sample_map, _ = io_utils.load_filtered_data(cfg)

    else:
        LOGGER.info("Loading CellBender matrices...")
        sample_map, _ = io_utils.load_cellbender_data(cfg)

    if not sample_map:
        raise RuntimeError("load_data: no samples were loaded.")

    LOGGER.info("Loaded %d samples.", len(sample_map))

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

    # ---------------------------------------------------------
    # Save merged dataset
    # ---------------------------------------------------------
    out_zarr = cfg.output_dir / cfg.output_name
    LOGGER.info("Saving merged dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        h5ad_out = cfg.output_dir / cfg.output_name
        LOGGER.warning(
            "Writing H5AD copy of merged dataset (this loads the full matrix into RAM)."
        )
        io_utils.save_dataset(adata, h5ad_out, fmt="h5ad")
        LOGGER.info("Saved H5AD dataset → %s", h5ad_out)
    import shutil

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
        LOGGER.info("Removed temporary merge directory: %s", temp_dir)
    except Exception as e:
        LOGGER.warning("Could not remove temp merge directory %s: %s", temp_dir, e)


    LOGGER.info("Finished load_data")
    return adata
