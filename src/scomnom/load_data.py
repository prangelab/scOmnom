from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import anndata as ad
import pandas as pd

from .config import LoadAndQCConfig
from . import io_utils

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------
def setup_logging(logfile: Optional[Path]):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove inherited handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(logfile), mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------
def add_metadata(adata: ad.AnnData, metadata_tsv: Path, sample_id_col: str) -> ad.AnnData:
    df = pd.read_csv(metadata_tsv, sep="\t")

    if sample_id_col not in df.columns:
        raise KeyError(
            f"Metadata TSV missing required column '{sample_id_col}'. "
            f"Found: {list(df.columns)}"
        )

    df[sample_id_col] = df[sample_id_col].astype(str)

    obs_ids = adata.obs[sample_id_col].astype(str)
    meta_ids = pd.Index(df[sample_id_col])

    missing = obs_ids.unique().difference(meta_ids)
    if len(missing) > 0:
        raise ValueError(f"Missing metadata entries for: {list(missing)}")

    # Merge metadata for each cell
    temp = pd.DataFrame({sample_id_col: obs_ids}, index=adata.obs_names)
    merged = temp.merge(df, on=sample_id_col, how="left")

    for col in df.columns:
        if col == sample_id_col:
            continue
        adata.obs[col] = merged[col].values

        # Cast low-cardinality strings to categories
        if (
            adata.obs[col].dtype == object
            and adata.obs[col].nunique() < 0.1 * len(adata.obs)
        ):
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


# ------------------------------------------------------------
# LOAD DATA MODULE 1 — PURE INGEST + MERGE
# ------------------------------------------------------------
def run_load_data(cfg: LoadAndQCConfig, logfile: Optional[Path] = None) -> ad.AnnData:
    """
    Module 1 — LOAD DATA
    ------------------------------------
    Responsibilities:
      • Load samples (raw / filtered / cellbender)
      • Validate metadata
      • Merge samples (OOM-safe)
      • Add metadata
      • Save merged dataset

    Output: <output_name>.zarr containing the merged raw matrix.
    """

    setup_logging(logfile)
    LOGGER.info("Starting load_data (pure ingestion)")

    # ------------------------------------------------------------
    # 1) Infer batch key from metadata
    # ------------------------------------------------------------
    batch_key = io_utils.infer_batch_key_from_metadata_tsv(
        cfg.metadata_tsv, cfg.batch_key
    )
    cfg.batch_key = batch_key
    LOGGER.info(f"Batch key inferred: {batch_key}")

    # ------------------------------------------------------------
    # 2) Load samples from exactly one source
    # ------------------------------------------------------------
    n_sources = sum([
        cfg.raw_sample_dir is not None,
        cfg.filtered_sample_dir is not None,
        cfg.cellbender_dir is not None,
    ])
    if n_sources != 1:
        raise RuntimeError(
            "Exactly one input source must be provided: "
            "--raw-sample-dir OR --filtered-sample-dir OR --cellbender-dir"
        )

    if cfg.raw_sample_dir:
        LOGGER.info("Loading raw 10x matrices...")
        sample_map, read_counts, _ = io_utils.load_raw_data(cfg)

    elif cfg.filtered_sample_dir:
        LOGGER.info("Loading filtered CellRanger matrices...")
        sample_map, read_counts = io_utils.load_filtered_data(cfg)

    else:  # CellBender
        LOGGER.info("Loading CellBender outputs...")
        sample_map, read_counts = io_utils.load_cellbender_data(cfg)

    loaded_samples = pd.Index(map(str, sample_map.keys()))
    LOGGER.info(f"Loaded {len(sample_map)} samples")

    # ------------------------------------------------------------
    # 3) Metadata validation
    # ------------------------------------------------------------
    df_meta = pd.read_csv(cfg.metadata_tsv, sep="\t")

    if cfg.batch_key not in df_meta.columns:
        raise KeyError(
            f"Metadata TSV must contain column '{cfg.batch_key}'. "
            f"Found: {list(df_meta.columns)}"
        )

    meta_samples = pd.Index(df_meta[cfg.batch_key].astype(str))

    missing = loaded_samples.difference(meta_samples)
    extra = meta_samples.difference(loaded_samples)

    if len(missing) > 0:
        raise ValueError(f"Samples loaded but missing in metadata_tsv: {list(missing)}")
    if len(extra) > 0:
        raise ValueError(f"Samples in metadata_tsv but not loaded: {list(extra)}")

    # ------------------------------------------------------------
    # 4) MERGE samples — NO FILTERING
    # ------------------------------------------------------------
    LOGGER.info("Merging samples (OOM-safe)...")
    tmp_dir = Path(cfg.output_dir) / "tmp_merge"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # merge_samples writes a temporary zarr and returns in-memory AnnData
    merged_tmp = tmp_dir / "merged.zarr"
    adata = io_utils.merge_samples(sample_map, cfg.batch_key, out_path=merged_tmp)

    LOGGER.info(f"Merged dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ------------------------------------------------------------
    # 5) Add metadata
    # ------------------------------------------------------------
    LOGGER.info("Adding metadata to merged dataset...")
    adata = add_metadata(adata, cfg.metadata_tsv, cfg.batch_key)
    adata.uns["batch_key"] = batch_key

    # ------------------------------------------------------------
    # 6) Save merged dataset
    # ------------------------------------------------------------
    out_zarr = Path(cfg.output_dir) / f"{cfg.output_name}.zarr"
    LOGGER.info(f"Saving merged dataset → {out_zarr}")
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        out_h5ad = Path(cfg.output_dir) / f"{cfg.output_name}.h5ad"
        LOGGER.warning("Saving H5AD — may be large and memory-heavy.")
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Completed load_data (pure ingestion).")
    return adata
