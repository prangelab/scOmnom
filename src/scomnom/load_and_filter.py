# src/scomnom/load_and_filter.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import pandas as pd

from .config import LoadAndQCConfig
from . import io_utils
from . import plot_utils
from .qc_and_filter import compute_qc_metrics, sparse_filter_cells_and_genes

LOGGER = logging.getLogger(__name__)


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


def _add_metadata(adata: ad.AnnData, metadata_tsv: Path, sample_id_col: str) -> ad.AnnData:
    """
    Attach per-sample metadata from TSV to adata.obs, mirroring load_data.add_metadata.
    """
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

    obs_col = adata.obs[sample_id_col].astype(str)
    temp = pd.DataFrame({sample_id_col: obs_col}, index=adata.obs_names)
    merged = temp.merge(df, on=sample_id_col, how="left")

    for col in df.columns:
        if col == sample_id_col:
            continue
        adata.obs[col] = merged[col].values
        if (
            adata.obs[col].dtype == object
            and adata.obs[col].nunique() < 0.1 * len(adata.obs)
        ):
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


def _per_sample_qc_and_filter(
    sample_map: Dict[str, ad.AnnData],
    cfg: LoadAndQCConfig,
) -> tuple[Dict[str, ad.AnnData], pd.DataFrame]:
    """
    Run QC + min_genes/min_cells filtering per sample (OOM-safe) and
    collect a lightweight QC dataframe for pre-filter plots.
    """
    import pandas as pd

    filtered_samples: Dict[str, ad.AnnData] = {}
    qc_rows = []

    for sample, a in sample_map.items():
        LOGGER.info(
            "[Per-sample QC] %s: %d cells × %d genes",
            sample,
            a.n_obs,
            a.n_vars,
        )

        # QC metrics (sparse-safe)
        a = compute_qc_metrics(a, cfg)  # uses mt_prefix/ribo_prefixes/hb_regex from cfg

        # small QC df for plotting
        qc_rows.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "total_counts": a.obs["total_counts"].to_numpy(),
                    "n_genes_by_counts": a.obs["n_genes_by_counts"].to_numpy(),
                    "pct_counts_mt": a.obs["pct_counts_mt"].to_numpy(),
                }
            )
        )

        # filtering
        a = sparse_filter_cells_and_genes(
            a,
            min_genes=cfg.min_genes,
            min_cells=cfg.min_cells,
        )

        LOGGER.info(
            "[Per-sample QC] %s: %d cells × %d genes after filtering",
            sample,
            a.n_obs,
            a.n_vars,
        )

        filtered_samples[sample] = a

    qc_df = (
        pd.concat(qc_rows, axis=0, ignore_index=True)
        if qc_rows
        else pd.DataFrame(
            columns=["sample", "total_counts", "n_genes_by_counts", "pct_counts_mt"]
        )
    )

    return filtered_samples, qc_df


def run_load_and_filter(
    cfg: LoadAndFilterConfig,
    logfile: Optional[Path] = None,  # kept for CLI compatibility; logging is already set up there
) -> ad.AnnData:

    LOGGER.info("Starting load-and-filter")
    # Configure Scanpy/Matplotlib figure behavior + formats
    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    # ---------------------------------------------------------
    # Infer batch key from metadata if needed
    # ---------------------------------------------------------
    batch_key = io_utils.infer_batch_key_from_metadata_tsv(
        cfg.metadata_tsv, cfg.batch_key
    )
    if cfg.batch_key is None:
        LOGGER.warning(
            "No batch_key provided. Inferred batch_key='%s' from metadata.tsv. "
            "Please verify this is correct.",
            batch_key,
        )
    LOGGER.info("Using batch_key='%s'", batch_key)
    cfg.batch_key = batch_key

    # ---------------------------------------------------------
    # Select input source and load samples
    # ---------------------------------------------------------
    n_sources = sum(
        [
            cfg.raw_sample_dir is not None,
            cfg.filtered_sample_dir is not None,
            cfg.cellbender_dir is not None,
        ]
    )
    if n_sources != 1:
        raise RuntimeError(
            "Exactly one input source must be provided: "
            "--raw-sample-dir OR --filtered-sample-dir OR --cellbender-dir"
        )

    if cfg.raw_sample_dir is not None:
        LOGGER.info("Loading raw 10x matrices with CellRanger-like cell calling...")
        qc_plot_dir = cfg.figdir / "QC_plots"
        sample_map, read_counts, _ = io_utils.load_raw_data(
            cfg,
            plot_dir=qc_plot_dir,
        )
    elif cfg.filtered_sample_dir is not None:
        LOGGER.info("Loading CellRanger filtered matrices...")
        sample_map, read_counts = io_utils.load_filtered_data(cfg)
    else:
        LOGGER.info("Loading CellBender matrices...")
        sample_map, read_counts = io_utils.load_cellbender_data(cfg)

    # ---------------------------------------------------------
    # Validate metadata vs loaded samples
    # ---------------------------------------------------------
    LOGGER.info("Validating samples and metadata...")

    if cfg.metadata_tsv is None:
        raise RuntimeError("metadata_tsv is required but was None")

    df_meta = pd.read_csv(cfg.metadata_tsv, sep="\t")
    if cfg.batch_key not in df_meta.columns:
        raise KeyError(
            f"Metadata TSV must contain the batch key column '{cfg.batch_key}'. "
            f"Found columns: {list(df_meta.columns)}"
        )

    meta_samples = pd.Index(df_meta[cfg.batch_key].astype(str))
    loaded_samples = pd.Index(sample_map.keys()).astype(str)

    missing_rows = loaded_samples.difference(meta_samples)
    extra_rows = meta_samples.difference(loaded_samples)

    if len(missing_rows) > 0:
        raise ValueError(
            "The following samples were found in the input directories but "
            "are missing from metadata_tsv:\n"
            f"  {list(missing_rows)}"
        )

    if len(extra_rows) > 0:
        raise ValueError(
            "The following samples exist in metadata_tsv but were not found "
            "in the input data folders:\n"
            f"  {list(extra_rows)}"
        )

    if len(meta_samples) != len(loaded_samples):
        raise ValueError(
            f"metadata_tsv has {len(meta_samples)} rows for batch_key='{cfg.batch_key}', "
            f"but {len(loaded_samples)} samples were loaded."
        )

    LOGGER.info("Loaded %d samples.", len(sample_map))

    # ---------------------------------------------------------
    # Per-sample QC + sparse filtering (OOM-safe)
    # ---------------------------------------------------------
    LOGGER.info("Running per-sample QC and filtering...")
    filtered_sample_map, qc_df = _per_sample_qc_and_filter(sample_map, cfg)

    # ---------------------------------------------------------
    # Pre-filter QC plots (lightweight, from qc_df only)
    # ---------------------------------------------------------
    if cfg.make_figures:
        LOGGER.info("Plotting pre-filter QC...")
        plot_utils.run_qc_plots_pre_filter_df(qc_df, cfg)

    # ---------------------------------------------------------
    # Merge filtered samples into a single AnnData
    # ---------------------------------------------------------
    tmp_dir = cfg.output_dir / "tmp_merge_load_and_filter"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    merge_out = tmp_dir / "merged.filtered.zarr"

    LOGGER.info("Merging filtered samples into a single AnnData...")
    adata = io_utils.merge_samples(
        filtered_sample_map,
        batch_key=cfg.batch_key,
        out_path=merge_out,
    )
    LOGGER.info(
        "Merged filtered dataset: %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )

    # ---------------------------------------------------------
    # Attach per-sample metadata
    # ---------------------------------------------------------
    LOGGER.info("Adding metadata...")
    adata = _add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    adata.uns["batch_key"] = cfg.batch_key

    # ---------------------------------------------------------
    # Global QC on merged filtered data (for post-filter plots ONLY)
    # ---------------------------------------------------------
    LOGGER.info("Computing QC metrics on merged filtered data...")
    adata = compute_qc_metrics(adata, cfg)

    # ---------------------------------------------------------
    # Post-filter QC plots (NO additional filtering here)
    # ---------------------------------------------------------
    if cfg.make_figures:
        LOGGER.info("Plotting post-filter QC...")
        plot_utils.run_qc_plots_postfilter(adata, cfg)
        plot_utils.plot_final_cell_counts(adata, cfg)

    # ---------------------------------------------------------
    # Save final filtered dataset
    # ---------------------------------------------------------
    out_zarr = cfg.output_dir / adata.filtered
    LOGGER.info("Saving filtered dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        out_h5ad = cfg.output_dir / adata.filtered
        LOGGER.warning("Writing H5AD copy (loads data into RAM).")
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    # Cleanup tmp merge dir
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception as e:
        LOGGER.warning("Could not remove temp merge directory %s: %s", tmp_dir, e)

    LOGGER.info("Finished load-and-filter")
    return adata
