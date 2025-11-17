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
    if metadata_tsv is None:
        return adata
    import pandas as pd
    if not metadata_tsv.exists():
        LOGGER.error("Metadata TSV not found: %s", metadata_tsv)
        return adata
    df = pd.read_csv(metadata_tsv, sep='\t')
    df.index = df.index.astype(str)
    if sample_id_col not in adata.obs.columns:
        LOGGER.error("Column '%s' not in adata.obs", sample_id_col)
        return adata
    obs_col = adata.obs[sample_id_col].astype(str)
    temp = pd.DataFrame({sample_id_col: obs_col}, index=adata.obs_names)
    merged = temp.merge(df, on=sample_id_col, how="left")
    for col in df.columns:
        adata.obs[col] = merged[col]
        if (adata.obs[col].nunique() / max(1, len(adata.obs[col]))) < 0.1 and not pd.api.types.is_numeric_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype('category')
    return adata


def compute_qc_metrics(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    adata.var["mt"] = adata.var_names.str.startswith(cfg.mt_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(cfg.ribo_prefixes))
    adata.var["hb"] = adata.var_names.str.contains(cfg.hb_regex)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=False)
    if 'counts_raw' in adata.layers:
        adata_raw_view = ad.AnnData(adata.layers['counts_raw'], obs=adata.obs, var=adata.var)
        sc.pp.calculate_qc_metrics(adata_raw_view, qc_vars=["mt"], inplace=True, log1p=False)
        for k in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
            adata.obs[f"{k}_raw"] = adata_raw_view.obs[k]
    return adata


def run_scrublet_parallel(adata: ad.AnnData, batch_key: str, n_jobs: int, **kwargs) -> ad.AnnData:
    batches = [adata[adata.obs[batch_key] == b].copy() for b in adata.obs[batch_key].unique()]

    def _run(batch):
        sc.pp.scrublet(batch, **kwargs)
        return batch

    processed = Parallel(n_jobs=n_jobs)(delayed(_run)(b) for b in batches)
    out = ad.concat(processed, join='outer', merge='same')
    return out


def filter_and_doublets(adata: ad.AnnData, cfg: LoadAndQCConfig) -> ad.AnnData:
    # preserve prefilter counts if already set
    pre_counts = adata.uns.get("pre_filter_counts", None)

    # apply filters in place
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)

    # doublet detection may return a new AnnData
    adata = run_scrublet_parallel(adata, batch_key=cfg.batch_key, n_jobs=cfg.n_jobs)

    # reattach if lost
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
        sample_map, read_counts = io_utils.load_raw_data(cfg)

    elif cfg.filtered_sample_dir:
        LOGGER.info("Loading CellRanger filtered matrices...")
        sample_map, read_counts = io_utils.load_filtered_data(cfg)

    else:  # cfg.cellbender_dir
        LOGGER.info("Loading CellBender matrices...")
        sample_map, read_counts = io_utils.load_cellbender_data(cfg)

    # merge
    adata = io_utils.merge_samples(sample_map, batch_key=cfg.batch_key)

    # metadata
    adata = add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    adata.uns["batch_key"] = batch_key

    # QC metrics + plots
    adata = compute_qc_metrics(adata, cfg)
    plot_utils.run_qc_plots_pre_filter(adata, cfg)
    plot_utils.plot_elbow_knee(adata, cfg, suffix="prefilter")

    # filtering + normalization + reduction + clustering
    adata = filter_and_doublets(adata, cfg)
    adata = normalize_and_hvg(adata, cfg)
    adata = pca_neighbors_umap(adata, cfg)
    adata = cluster_and_cleanup_qc(adata, cfg)

    plot_utils.run_qc_plots_postfilter(adata, cfg)
    plot_utils.plot_final_cell_counts(adata, cfg)
    plot_utils.plot_elbow_knee(adata, cfg, suffix="postfilter")

    io_utils.save_adata(adata, cfg.output_dir / cfg.output_name)
    LOGGER.info("Finished load_and_filter")
    return adata

