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
    handlers = [logging.StreamHandler()]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile), mode="w"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)


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
    Memory-safe QC metric computation that avoids Scanpy's dense
    calculate_qc_metrics() call, which OOMs on large datasets.
    Produces the same metrics: total_counts, n_genes_by_counts,
    pct_counts_mt, pct_counts_ribo, pct_counts_hb.
    """

    import numpy as np
    from scipy import sparse

    # ---------------------------------------------------------
    # Annotate mitochondrial / ribosomal / hemoglobin genes
    # ---------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith(cfg.mt_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(cfg.ribo_prefixes))
    adata.var["hb"] = adata.var_names.str.contains(cfg.hb_regex)

    # Ensure CSR for efficient row ops
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        LOGGER.warning("X is dense — QC may be very slow and large memory-consuming.")

    # ---------------------------------------------------------
    # Per-cell QC metrics (sparse safe)
    # ---------------------------------------------------------
    LOGGER.info("Computing QC metrics (memory-safe sparse mode)...")

    # Total counts per cell
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    # Number of detected genes per cell
    n_genes_by_counts = np.diff(X.indptr)

    # Mask indices
    mt_idx = np.where(adata.var["mt"].values)[0]
    ribo_idx = np.where(adata.var["ribo"].values)[0]
    hb_idx = np.where(adata.var["hb"].values)[0]

    # Function for computing percentages safely
    def pct_from_idx(idx):
        if len(idx) == 0:
            return np.zeros_like(total_counts, dtype=float)
        sub = X[:, idx]
        vals = np.asarray(sub.sum(axis=1)).ravel()
        return vals / np.maximum(total_counts, 1)

    pct_mt = pct_from_idx(mt_idx)
    pct_ribo = pct_from_idx(ribo_idx)
    pct_hb = pct_from_idx(hb_idx)

    # ---------------------------------------------------------
    # Write to adata.obs (same API as Scanpy)
    # ---------------------------------------------------------
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    adata.obs["pct_counts_mt"] = pct_mt * 100
    adata.obs["pct_counts_ribo"] = pct_ribo * 100
    adata.obs["pct_counts_hb"] = pct_hb * 100

    return adata



def filter_and_doublets(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    # preserve prefilter counts if already set
    pre_counts = adata.uns.get("pre_filter_counts", None)

    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)

    # Log remaining cells per sample
    for sample, n in adata.obs[cfg.batch_key].value_counts().items():
        LOGGER.info(f"Remaining cells in {sample}: {n}")

    # ---- SOLO DOUBLETS----
    from scvi.external import SOLO
    import torch

    # Determine device (CPU or CUDA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f"Running Solo doublet detection on device: {device}")

    SOLO.setup_anndata(adata, layer=None)

    # Instantiate the SOLO model
    solo_model = SOLO(adata, device=device)  # Use SOLO (uppercase)

    # Conservative CPU settings, faster GPU settings
    if device == "cpu":
        solo_model.train(batch_size=256, max_epochs=40)
    else:
        solo_model.train(batch_size=1024, max_epochs=20)

    # Predict doublet scores
    doublet_scores = solo_model.predict()

    # Apply results to AnnData
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = (doublet_scores > cfg.doublet_score_threshold).astype(bool)

    # Reattach uns if Solo rewrote AnnData
    if pre_counts is not None:
        adata.uns["pre_filter_counts"] = pre_counts

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

    # infer batch key
    batch_key = io_utils.infer_batch_key_from_metadata_tsv(cfg.metadata_tsv, cfg.batch_key)
    cfg.batch_key = batch_key

    # get input
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

    # Load depending on which is set
    if cfg.raw_sample_dir:
        LOGGER.info("Loading raw 10x matrices with CellRanger-like cell calling...")
        qc_plot_dir = cfg.output_dir / "figures" / "QC_plots"
        sample_map, read_counts, _ = io_utils.load_raw_data(cfg, plot_dir=qc_plot_dir,)

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

    # merge
    LOGGER.info("Merging samples...")
    adata = io_utils.merge_samples(sample_map, batch_key=cfg.batch_key)

    # metadata
    LOGGER.info("Adding metadata...")
    adata = add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    adata.uns["batch_key"] = batch_key

    # QC metrics + plots
    LOGGER.info("Running QC...")
    adata = compute_qc_metrics(adata, cfg)
    LOGGER.info(" Plotting pre-filter QC...")
    plot_utils.run_qc_plots_pre_filter(adata, cfg)
    plot_utils.plot_elbow_knee(
        adata,
        figpath_stem="QC_elbow_knee_prefilter",
        figdir=cfg.figdir / "QC_plots"
    )

    # filtering + normalization + reduction + clustering
    LOGGER.info("Filtering...")
    adata = filter_and_doublets(adata, cfg)
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
        figdir=cfg.figdir / "QC_plots"
    )

    LOGGER.info("Saving...")
    io_utils.save_adata(adata, cfg.output_dir / cfg.output_name)
    LOGGER.info("Finished load_and_filter")
    return adata

