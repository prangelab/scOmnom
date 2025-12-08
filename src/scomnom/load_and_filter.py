from __future__ import annotations
import logging
import anndata as ad
import scanpy as sc
from joblib import Parallel, delayed
from kneed import KneeLocator
from pathlib import Path
from typing import Dict, List, Optional

from .config import LoadAndQCConfig
from . import io_utils
from . import plot_utils

LOGGER = logging.getLogger(__name__)


# ---- logging helper ----
def setup_logging(logfile: Optional[Path]):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all existing handlers to avoid double-formatting and Rich interception
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(logfile), mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ---- step functions ----
def add_metadata(adata: ad.AnnData, metadata_tsv: Optional[Path], sample_id_col: str) -> ad.AnnData:
    import pandas as pd

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

    obs_col = adata.obs[sample_id_col].astype(str)
    temp = pd.DataFrame({sample_id_col: obs_col}, index=adata.obs_names)
    merged = temp.merge(df, on=sample_id_col, how="left")
    for col in df.columns:
        if col == sample_id_col:
            # Protect the batch key from being overwritten
            continue

        # Assign per-cell metadata
        adata.obs[col] = merged[col].values

        # Optional: cast small-cardinality strings to category
        if (
                adata.obs[col].nunique() < 0.1 * len(adata.obs)
                and adata.obs[col].dtype == object
        ):
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


def compute_qc_metrics(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    """
    Memory-safe QC computation that mimics Scanpy's calculate_qc_metrics() API,
    including per-cell metrics, per-gene metrics, and a lightweight qc_metrics
    entry in adata.uns — but without densifying sparse matrices or causing OOM.
    """

    import numpy as np
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        LOGGER.warning("X is dense — QC may be slow and memory intensive.")

    n_cells, n_genes = X.shape

    # ---------------------------------------------------------
    # Annotate gene categories (mt, ribo, hb)
    # ---------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith(cfg.mt_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(cfg.ribo_prefixes))
    adata.var["hb"] = adata.var_names.str.contains(cfg.hb_regex)

    mt_idx = np.where(adata.var["mt"].values)[0]
    ribo_idx = np.where(adata.var["ribo"].values)[0]
    hb_idx = np.where(adata.var["hb"].values)[0]

    # ---------------------------------------------------------
    # Per-cell QC metrics
    # ---------------------------------------------------------
    LOGGER.info("Computing sparse per-cell QC metrics...")

    total_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes_by_counts = np.diff(X.indptr)

    def pct_from_idx(idx):
        if len(idx) == 0:
            return np.zeros(n_cells, dtype=float)
        sub = X[:, idx]
        vals = np.asarray(sub.sum(axis=1)).ravel()
        return vals / np.maximum(total_counts, 1)

    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    adata.obs["pct_counts_mt"] = pct_from_idx(mt_idx) * 100
    adata.obs["pct_counts_ribo"] = pct_from_idx(ribo_idx) * 100
    adata.obs["pct_counts_hb"] = pct_from_idx(hb_idx) * 100

    # ---------------------------------------------------------
    # Per-gene QC metrics (Scanpy-compatible)
    # ---------------------------------------------------------
    LOGGER.info("Computing sparse per-gene QC metrics (Scanpy-compatible)...")

    # How many cells have nonzero for each gene
    n_cells_by_counts = np.diff(X.tocsc().indptr)

    # Sum expression for each gene
    total_counts_gene = np.asarray(X.sum(axis=0)).ravel()

    # Mean expression per gene
    mean_counts = total_counts_gene / max(n_cells, 1)

    # Dropout fraction
    pct_dropout = 100 * (1 - (n_cells_by_counts / max(n_cells, 1)))

    # Attach to adata.var
    adata.var["n_cells_by_counts"] = n_cells_by_counts
    adata.var["mean_counts"] = mean_counts
    adata.var["total_counts"] = total_counts_gene
    adata.var["pct_dropout_by_counts"] = pct_dropout

    # ---------------------------------------------------------
    # Minimal Scanpy-compatible qc_metrics dictionary in .uns
    # ---------------------------------------------------------
    qc_metrics = {
        "qc_vars": ["mt", "ribo", "hb"],
        "percent_top": {},  # Scanpy uses this only for violin plotting
        "log1p": False,
        "raw_qc_metrics": {},  # We skip computing raw because it's unused
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    }

    adata.uns["qc_metrics"] = qc_metrics

    LOGGER.info("QC metrics computed (memory-safe, Scanpy-compatible).")
    return adata


def qc_and_filter_samples(
    sample_map: Dict[str, ad.AnnData],
    cfg: LoadAndQCConfig,
):
    """
    Run QC + basic min_genes/min_cells filtering per sample.
    """

    import pandas as pd

    filtered_samples = {}
    qc_rows = []

    for sample, a in sample_map.items():
        LOGGER.info(
            f"[Per-sample QC] {sample}: {a.n_obs:,} cells × {a.n_vars:,} genes"
        )

        # QC metrics
        a = compute_qc_metrics(a, cfg)

        # lightweight QC df
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
            f"[Per-sample QC] {sample}: {a.n_obs:,} cells × {a.n_vars:,} genes after filtering"
        )

        filtered_samples[sample] = a

    # combine QC rows
    qc_df = (
        pd.concat(qc_rows, axis=0, ignore_index=True)
        if qc_rows
        else pd.DataFrame(
            columns=["sample", "total_counts", "n_genes_by_counts", "pct_counts_mt"]
        )
    )

    return filtered_samples, qc_df


def sparse_filter_cells_and_genes(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
) -> ad.AnnData:
    """
    Memory-efficient filtering of cells and genes for massive sparse matrices.
    - Computes nnz-per-cell from CSR.indptr
    - Computes nnz-per-gene directly from CSR.indices without CSC conversion
    """

    import numpy as np
    import scipy.sparse as sp
    import psutil
    import logging
    LOGGER = logging.getLogger(__name__)

    # ----------------------------------------------------
    # Ensure CSR
    # ----------------------------------------------------
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    # Log initial memory state
    rss0 = psutil.Process().memory_info().rss / 1024**3
    LOGGER.info(
        f"[Filtering] Start: {adata.n_obs:,} cells × {adata.n_vars:,} genes "
        f"(RSS={rss0:.2f} GB)"
    )

    X = adata.X  # CSR matrix

    # ----------------------------------------------------
    # 1) Filter cells by gene count (nnz per row)
    # ----------------------------------------------------
    gene_counts = np.diff(X.indptr)  # nnz per row
    cell_mask = gene_counts >= min_genes

    n_before = adata.n_obs
    n_after = int(cell_mask.sum())

    if n_after == 0:
        raise ValueError(
            f"All cells removed by min_genes={min_genes}."
        )

    # Subset cells
    adata = adata[cell_mask].copy()
    X = adata.X  # Update reference

    LOGGER.info(
        f"[Filtering] Cells kept: {n_after:,} / {n_before:,} "
        f"({100*n_after/n_before:.1f}%)"
    )

    # ----------------------------------------------------
    # 2) Filter genes by occurrence across cells
    # ----------------------------------------------------
    # We need nnz per column.
    # For CSR, column indices live in X.indices.
    # Counting occurrences: bincount(indices)
    gene_nnz = np.bincount(X.indices, minlength=adata.n_vars)

    gene_mask = gene_nnz >= min_cells

    g_before = adata.n_vars
    g_after = int(gene_mask.sum())

    if g_after == 0:
        raise ValueError(
            f"All genes removed by min_cells={min_cells}."
        )

    # Subset genes
    adata = adata[:, gene_mask].copy()

    rss1 = psutil.Process().memory_info().rss / 1024**3
    LOGGER.info(
        f"[Filtering] Genes kept: {g_after:,} / {g_before:,} "
        f"({100*g_after/g_before:.1f}%), final RSS={rss1:.2f} GB"
    )

    return adata


def doublets_detection(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    """
    Memory-safe doublet detection using Scrublet in chunked mode.
    Avoids OOM on very large datasets by processing cells in batches.

    Adds:
        - adata.obs["doublet_score"]
        - adata.obs["predicted_doublet"]
    """
    import numpy as np
    import scrublet as scr
    from scipy import sparse

    LOGGER.info("Running Scrublet (chunked mode) for doublet detection...")

    # Ensure sparse CSR for safe slicing
    if not sparse.issparse(adata.X):
        LOGGER.warning("Converting dense matrix to CSR sparse matrix for Scrublet.")
        adata.X = sparse.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    X = adata.X
    n_cells = adata.n_obs

    # Choose chunk size (auto-tuned)
    if n_cells < 80_000:
        chunk_size = 40_000
    elif n_cells < 200_000:
        chunk_size = 50_000
    elif n_cells < 400_000:
        chunk_size = 60_000
    else:
        chunk_size = 80_000

    LOGGER.info(
        f"Scrublet: {n_cells:,} cells will be processed in chunks of {chunk_size:,}."
    )

    all_scores = np.zeros(n_cells, dtype=np.float32)
    all_preds = np.zeros(n_cells, dtype=bool)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        LOGGER.info(f"  → processing chunk {start:,}–{end:,} ({end-start:,} cells)")

        X_chunk = X[start:end]

        scrub = scr.Scrublet(
            X_chunk,
            expected_doublet_rate=getattr(cfg, "expected_doublet_rate", 0.06),
            verbose=False,
        )

        try:
            scores, preds = scrub.scrub_doublets()
        except Exception as e:
            raise RuntimeError(
                f"Scrublet failed on chunk {start:,}–{end:,}: {e}"
            )

        all_scores[start:end] = scores
        all_preds[start:end] = preds

    # Attach to AnnData
    adata.obs["doublet_score"] = all_scores
    adata.obs["predicted_doublet"] = all_preds

    # Apply configured threshold (Scrublet uses auto-threshold normally)
    thresh = getattr(cfg, "doublet_score_threshold", None)
    if thresh is not None:
        LOGGER.info(f"Applying manual doublet threshold: score > {thresh}")
        adata.obs["predicted_doublet"] = adata.obs["doublet_score"].values > thresh

    LOGGER.info(
        f"Doublets detected: {adata.obs['predicted_doublet'].sum():,} "
        f"({adata.obs['predicted_doublet'].mean()*100:.2f}%)"
    )

    return adata



def normalize_and_hvg(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, batch_key=cfg.batch_key)
    return adata


def pca_neighbors_umap(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    sc.tl.pca(adata)
    pvar = adata.uns['pca']['variance_ratio']
    kl = KneeLocator(range(1, len(pvar)+1), pvar, curve='convex', direction='decreasing')
    n_pcs_elbow = kl.elbow or cfg.max_pcs_plot
    sc.pp.neighbors(adata, n_pcs=n_pcs_elbow)
    sc.tl.umap(adata)
    adata.uns['n_pcs_elbow'] = int(n_pcs_elbow)
    return adata

def cluster_and_cleanup_qc(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    sc.tl.leiden(adata, resolution=1.0, flavor="igraph", random_state=42)
    # remove doublets and high-mt post clustering
    if 'predicted_doublet' in adata.obs:
        adata.obs['predicted_doublet'] = adata.obs['predicted_doublet'].astype('category')
        adata = adata[~adata.obs['predicted_doublet'].to_numpy()].copy()
    if 'pct_counts_mt' in adata.obs:
        adata = adata[adata.obs['pct_counts_mt'] < cfg.max_pct_mt].copy()
    # drop tiny samples
    if cfg.min_cells_per_sample > 0 and cfg.batch_key in adata.obs:
        small = adata.obs[cfg.batch_key].value_counts()[lambda x: x < cfg.min_cells_per_sample].index.tolist()
    if small:
        adata = adata[~adata.obs[cfg.batch_key].isin(small)].copy()
    return adata


# ---- orchestrator ----
def run_load_and_filter(cfg: LoadAndQCConfig, logfile: Optional[Path] = None) -> ad.AnnData:
    setup_logging(logfile)
    LOGGER.info("Starting load_and_filter")
    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    # ================================================================
    # DEV HACK: Restore merged AnnData from a stored Zarr
    # ================================================================
    DEV_RESTORE_ZARR = Path("/home/kprange/data/baria/vfat/results/adata.preprocessed.h5ad.zarr")  # <-- set this

    if DEV_RESTORE_ZARR.exists():
        LOGGER.warning(
            f"[DEV MODE] Restoring merged AnnData from {DEV_RESTORE_ZARR}.\n"
            "Skipping LOAD → QC → FILTER → MERGE → METADATA.\n"
            "Remove DEV_RESTORE_ZARR to run the full preprocessing."
        )

        # Load Zarr fully into RAM (expected for downstream steps)
        adata = io_utils.load_dataset(DEV_RESTORE_ZARR)
        LOGGER.info(
            f"[DEV MODE] Loaded restored AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes"
        )

        # -----------------------------------------------------------
        # Resume pipeline EXACTLY as it is after metadata assignment
        # -----------------------------------------------------------

        LOGGER.info("[DEV MODE] Running QC on merged filtered dataset...")
        adata = compute_qc_metrics(adata, cfg)

        LOGGER.info("[DEV MODE] Running doublet detection...")
        adata = doublets_detection(adata, cfg)

        LOGGER.info("[DEV MODE] Normalising + HVG selection...")
        adata = normalize_and_hvg(adata, cfg)

        LOGGER.info("[DEV MODE] Running PCA / neighbors / UMAP...")
        adata = pca_neighbors_umap(adata, cfg)

        LOGGER.info("[DEV MODE] Clustering + QC cleanup...")
        adata = cluster_and_cleanup_qc(adata, cfg)

        # Save final zarr
        out_zarr = Path(cfg.output_dir) / (cfg.output_name + ".zarr")
        LOGGER.info(f"[DEV MODE] Saving dataset to → {out_zarr}")
        io_utils.save_dataset(adata, out_zarr, fmt="zarr")

        # Optional H5AD
        if getattr(cfg, "save_h5ad", False):
            h5ad_out = cfg.output_dir / (cfg.output_name + ".h5ad")
            LOGGER.warning("[DEV MODE] Writing H5AD (RAM heavy)")
            io_utils.save_dataset(adata, h5ad_out, fmt="h5ad")

        LOGGER.info("[DEV MODE] Completed load-and-filter via fast-forward restore.")
        return adata

    # ================================================================
    # END DEV FAST-FORWARD HACK — full pipeline continues normally
    # ================================================================
    # infer batch key
    batch_key = io_utils.infer_batch_key_from_metadata_tsv(
        cfg.metadata_tsv, cfg.batch_key
    )
    cfg.batch_key = batch_key

    # get input
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

    # Load depending on which is set
    if cfg.raw_sample_dir:
        LOGGER.info("Loading raw 10x matrices with CellRanger-like cell calling...")
        qc_plot_dir = cfg.output_dir / "figures" / "QC_plots"
        sample_map, read_counts, _ = io_utils.load_raw_data(
            cfg,
            plot_dir=qc_plot_dir,
        )

    elif cfg.filtered_sample_dir:
        LOGGER.info("Loading CellRanger filtered matrices...")
        sample_map, read_counts = io_utils.load_filtered_data(cfg)

    else:  # cfg.cellbender_dir
        LOGGER.info("Loading CellBender matrices...")
        sample_map, read_counts = io_utils.load_cellbender_data(cfg)

    # --- validate metadata has 1 row per sample ---
    LOGGER.info("Validating samples and metadata...")
    import pandas as pd

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

    # ------------------------------------------------------------------
    # Per-sample QC + filtering (avoid global OOM)
    # ------------------------------------------------------------------
    LOGGER.info("Running per-sample QC and filtering...")
    filtered_sample_map, qc_df = qc_and_filter_samples(sample_map, cfg)

    # Pre-filter QC plots from lightweight QC dataframe
    LOGGER.info("Plotting pre-filter QC...")
    plot_utils.run_qc_plots_pre_filter_df(qc_df, cfg)

    # Merge samples
    tmp_dir = Path.cwd() / "tmp_merge"
    tmp_dir.mkdir(exist_ok=True)
    merge_out = tmp_dir / "merged.zarr"

    adata = io_utils.merge_samples(
        filtered_sample_map,
        cfg.batch_key,
        out_path=merge_out,
    )

    # metadata
    LOGGER.info("Adding metadata...")
    adata = add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    adata.uns["batch_key"] = batch_key

    # QC metrics on merged, already-filtered data (for post-filter plots)
    LOGGER.info("Running QC on merged filtered data...")
    adata = compute_qc_metrics(adata, cfg)

    # filtering + normalization + reduction + clustering
    LOGGER.info("Running doublet detection...")
    adata = doublets_detection(adata, cfg)

    LOGGER.info("Normalising...")
    adata = normalize_and_hvg(adata, cfg)

    LOGGER.info("Reducing dimensions...")
    adata = pca_neighbors_umap(adata, cfg)

    LOGGER.info("Clustering and cleaning up...")
    adata = cluster_and_cleanup_qc(adata, cfg)

    LOGGER.info("Plotting post-filter QC...")
    plot_utils.run_qc_plots_postfilter(adata, cfg)
    plot_utils.plot_final_cell_counts(adata, cfg)
    plot_utils.plot_elbow_knee(
        adata,
        figpath_stem="QC_elbow_knee_postfilter",
        figdir=cfg.figdir / "QC_plots",
    )

    # Save to zarr
    out_zarr = Path(cfg.output_dir) / (cfg.output_name + ".zarr")
    LOGGER.info(f"Saving dataset to → {out_zarr}")
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    # Optional H5AD output
    if getattr(cfg, "save_h5ad", False):
        LOGGER.warning(
            "User requested H5AD output. This is NOT recommended for large datasets."
        )
        h5ad_out = cfg.output_dir / (cfg.output_name + ".h5ad")
        io_utils.save_dataset(adata, h5ad_out, fmt="h5ad")
        LOGGER.info(f"Saved H5AD dataset → {h5ad_out}")

    LOGGER.info("Finished load_and_filter")
    return adata

