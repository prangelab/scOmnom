from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
import scipy.sparse
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

import warnings
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique",
    category=UserWarning,
    module="anndata"
)

# ---- logging helper ----
def setup_logging(logfile: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile), mode="w"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)

# ---- step functions ----
def load_raw_data(cfg: LoadAndQCConfig) -> Dict[str, ad.AnnData]:
    raw_dirs = io_utils.find_raw_dirs(cfg.sample_dir, cfg.raw_pattern)
    out: Dict[str, ad.AnnData] = {}
    for raw in raw_dirs:
        sample = raw.name.split(".raw_feature_bc_matrix")[0]
        adata = io_utils.read_raw_10x(raw)
        out[sample] = adata
        LOGGER.info("Loaded raw %s: %d cells, %d genes", sample, adata.n_obs, adata.n_vars)
    return out


def load_cellbender_data(cfg: LoadAndQCConfig) -> Dict[str, ad.AnnData]:
    if cfg.cellbender_dir is None:
        return {}
    cb_dirs = io_utils.find_cellbender_dirs(cfg.cellbender_dir, cfg.cellbender_pattern)
    out: Dict[str, ad.AnnData] = {}
    for cb in cb_dirs:
        sample = cb.name.split(".cellbender_filtered.output")[0]
        adata = io_utils.read_cellbender_h5(cb, sample, cfg.cellbender_h5_suffix)
        if adata is not None:
            out[sample] = adata
            LOGGER.info("Loaded CellBender %s: %d cells, %d genes", sample, adata.n_obs, adata.n_vars)
    return out


def merge_samples(raw_map: Dict[str, ad.AnnData], cb_map: Dict[str, ad.AnnData], batch_key: str) -> ad.AnnData:
    adatas: List[ad.AnnData] = []

    for sample, raw in raw_map.items():
        if sample in cb_map:
            cb = cb_map[sample].copy()
            common_obs = cb.obs_names.intersection(raw.obs_names)
            common_var = cb.var_names.intersection(raw.var_names)
            if len(common_obs) == 0 or len(common_var) == 0:
                LOGGER.warning("No common cells or genes for %s. Skipping.", sample)
                continue
            cb = cb[common_obs, common_var].copy()
            raw = raw[common_obs, common_var].copy()
            cb.layers["counts_raw"] = raw.X.copy()
            adata = cb
        else:
            # raw-only mode
            raw = raw.copy()
            if not scipy.sparse.issparse(raw.X):
                raw.X = scipy.sparse.csr_matrix(raw.X)
            raw.layers["counts_raw"] = raw.X.copy()
            adata = raw

        adata.obs[batch_key] = sample
        adata.obs_names = [f"{sample}_{bc}" for bc in adata.obs_names]
        adatas.append(adata)

    if len(adatas) == 0:
        raise RuntimeError("No samples loaded after matching.")

    LOGGER.info("Concatenating %d AnnData objects...", len(adatas))
    adata_all = sc.concat(adatas, axis=0, join="outer", merge="first")
    adata_all.obs_names_make_unique()

    # Store counts summary in merged object
    adata_all.uns["before_cellbender_counts"] = {k: v.n_obs for k, v in raw_map.items()}
    adata_all.uns["after_cellbender_counts"] = {k: v.n_obs for k, v in cb_map.items()}

    if "counts_raw" in adata_all.layers:
        raw_counts = adata_all.layers["counts_raw"]
        if not scipy.sparse.issparse(raw_counts):
            raw_counts = scipy.sparse.csr_matrix(raw_counts)
        adata_all.raw = ad.AnnData(X=raw_counts, var=adata_all.var.copy(), obs=adata_all.obs.copy())
        LOGGER.info("adata.raw set with counts_raw layer.")
    else:
        LOGGER.warning("'counts_raw' layer not found; adata.raw not set.")

    return adata_all



def add_metadata(adata: ad.AnnData, metadata_tsv: Optional[Path], sample_id_col: str) -> ad.AnnData:
    if metadata_tsv is None:
        return adata
    import pandas as pd
    if not metadata_tsv.exists():
        LOGGER.error("Metadata TSV not found: %s", metadata_tsv)
        return adata
    df = pd.read_csv(metadata_tsv, sep=' ', index_col=0)
    df.index = df.index.astype(str)
    if sample_id_col not in adata.obs.columns:
        LOGGER.error("Column '%s' not in adata.obs", sample_id_col)
        return adata
    obs_col = adata.obs[sample_id_col].astype(str)
    temp = pd.DataFrame({sample_id_col: obs_col}, index=adata.obs_names)
    merged = temp.join(df, on=sample_id_col, how='left')
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
    # record counts before filtering
    adata.uns["pre_filter_counts"] = adata.obs[cfg.batch_key].value_counts().sort_index().to_dict()

    # apply filters in place
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)

    # preserve uns across reassignments
    pre_counts = adata.uns["pre_filter_counts"]

    # doublet detection may return a new AnnData
    adata = run_scrublet_parallel(adata, batch_key=cfg.batch_key, n_jobs=cfg.n_jobs)

    # reattach if lost
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

# ---- plotting wrappers ----
def run_qc_plots_pre(adata: ad.AnnData, cfg: LoadAndQCConfig) -> None:
    if not cfg.make_figures:
        return
    plot_utils.setup_scanpy_figs(cfg.figdir)
    # Redirect to QC plot folder
    qc_dir = Path(cfg.output_dir) / "figures" / "QC_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = qc_dir
    plot_utils.qc_scatter(adata, groupby=cfg.batch_key)


def run_qc_plots_dimred(adata: ad.AnnData, cfg: LoadAndQCConfig):
    """Generate all QC plots (pre/post-filter, PCA, UMAP) inside QC_plots."""
    import os
    figdir_qc = cfg.figdir / "QC_plots"
    os.makedirs(figdir_qc, exist_ok=True)

    # Set Scanpy’s figure directory temporarily to QC_plots
    from scanpy import settings as sc_settings
    old_figdir = sc_settings.figdir
    sc_settings.figdir = figdir_qc

    try:
        # Pre-filter QC
        if "counts_raw" in adata.layers:
            sc.pl.violin(
                adata,
                ["n_genes_by_counts_raw", "total_counts_raw", "pct_counts_mt_raw"],
                jitter=0.4,
                groupby=cfg.batch_key,
                save="_QC_violin_mt_counts_prefilter.png",
                show=False
            )

        # Post-filter QC
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            groupby=cfg.batch_key,
            save="_QC_violin_mt_counts_postfilter.png",
            show=False
        )

        # PCA, variance, and UMAP
        sc.pl.pca_variance_ratio(
            adata,
            log=True,
            n_pcs=cfg.max_pcs_plot,
            save="_QC_pca_variance_ratio.png",
            show=False
        )
        sc.pl.umap(adata, color=[cfg.batch_key], save="_QC_umap_sample.png", show=False)
        sc.pl.umap(adata, color=[cfg.batch_key, "leiden"], save="_QC_umap_per_sample_and_leiden.png", show=False)
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", save="_QC_scatter_mt.png", show=False)
    finally:
        sc_settings.figdir = old_figdir



def run_qc_plots_counts(adata: ad.AnnData, cfg: LoadAndQCConfig) -> None:
    if not cfg.make_figures:
        return
    import pandas as pd
    import numpy as np
    import os

    # updated folder structure
    figdir_qc = cfg.figdir / "QC_plots"
    figdir_ps = figdir_qc / "per_sample"
    os.makedirs(figdir_qc, exist_ok=True)
    os.makedirs(figdir_ps, exist_ok=True)

    # ---------- before vs after filtering ----------
    raw_pre = adata.uns.get("pre_filter_counts", None)
    if isinstance(raw_pre, dict):
        before_counts = pd.Series(raw_pre).sort_index()
    else:
        before_counts = raw_pre

    after_counts = adata.obs[cfg.batch_key].value_counts().sort_index()
    all_samples = sorted(set(before_counts.index) | set(after_counts.index)) if before_counts is not None else after_counts.index

    before_counts = (
        before_counts.reindex(all_samples, fill_value=0)
        if before_counts is not None
        else pd.Series([0] * len(all_samples), index=all_samples)
    )
    after_counts = after_counts.reindex(all_samples, fill_value=0)

    df_counts = pd.DataFrame({
        "sample": all_samples,
        "before_filter": before_counts.values,
        "after_filter": after_counts.values,
    })
    df_counts["pct_retained_filter"] = np.where(
        df_counts["before_filter"] > 0,
        100 * df_counts["after_filter"] / df_counts["before_filter"],
        0,
    )
    df_counts.to_csv(figdir_qc / "QC_cells_per_sample_filter.tsv", sep="\t", index=False)

    # Per-sample plots (now in subfolder)
    for _, row in df_counts.iterrows():
        s = row["sample"]
        df_s = df_counts[df_counts["sample"] == s].copy()
        df_s = df_s.rename(
            columns={
                "before_filter": "before",
                "after_filter": "after",
                "pct_retained_filter": "retained_pct",
            }
        ).reset_index(drop=True)
        plot_utils.barplot_before_after(
            df_s, figdir_ps / f"{s}_QC_cells_before_after.png", cfg.min_cells_per_sample
        )

    # Aggregate overview (each sample = one bar)
    plot_utils.barplot_before_after(
        df_counts.rename(
            columns={
                "before_filter": "before",
                "after_filter": "after",
                "pct_retained_filter": "retained_pct",
            }
        ).reset_index(drop=True),
        figdir_qc / "QC_cells_before_after_AGGREGATE.png",
        cfg.min_cells_per_sample,
    )


    # ---------- full pipeline ----------
    if "before_cellbender_counts" in adata.uns:
        samples = sorted(adata.uns["before_cellbender_counts"].keys())
        raw_10x = [adata.uns["before_cellbender_counts"][s] for s in samples]
        after_cb = [adata.uns.get("after_cellbender_counts", {}).get(s, 0) for s in samples]
        final_filtered = [adata.obs.query(f"{cfg.batch_key} == @s").shape[0] for s in samples]

        df_full = pd.DataFrame({
            "sample": samples,
            "raw_10x": raw_10x,
            "after_cellbender": after_cb,
            "final_filtered": final_filtered,
        })
        df_full["pct_retained_cb"] = np.where(
            df_full["raw_10x"] > 0, 100 * df_full["after_cellbender"] / df_full["raw_10x"], 0
        )
        df_full["pct_retained_final"] = np.where(
            df_full["raw_10x"] > 0, 100 * df_full["final_filtered"] / df_full["raw_10x"], 0
        )
        df_full.to_csv(figdir_qc / "QC_cells_per_sample_full_pipeline.tsv", sep="\t", index=False)

        # Per-sample plots
        for _, row in df_full.iterrows():
            s = row["sample"]
            plot_utils.barplot_full_pipeline(
                df_full[df_full["sample"] == s],
                figdir_qc / f"{s}_QC_cells_full_pipeline.png",
            )

        # Aggregate overview
        plot_utils.barplot_full_pipeline(
            df_full, figdir_qc / "QC_cells_full_pipeline_AGGREGATE.png"
        )



# ---- orchestrator ----
def run_load_and_qc(cfg: LoadAndQCConfig, logfile: Optional[Path] = None) -> ad.AnnData:
    setup_logging(logfile)
    LOGGER.info("Starting load_and_qc")
    # load
    raw_map = load_raw_data(cfg)
    cb_map = load_cellbender_data(cfg) if cfg.cellbender_dir else {}
    # merge
    adata = merge_samples(raw_map, cb_map, batch_key=cfg.batch_key)
    # metadata
    adata = add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
    # qc metrics and pre-filter plots
    adata = compute_qc_metrics(adata, cfg)
    run_qc_plots_pre(adata, cfg)
    # capture counts per sample before filtering
    adata.uns["pre_filter_counts"] = adata.obs[cfg.batch_key].value_counts().sort_index()
    # filtering + doublets tagging
    adata = filter_and_doublets(adata, cfg)
    # normalize + HVG
    adata = normalize_and_hvg(adata, cfg)
    # PCA/UMAP # cluster + cleanup
    adata = pca_neighbors_umap(adata, cfg)
    # Cluster + cleanup
    adata = cluster_and_cleanup_qc(adata, cfg)
    # Make qc plots
    run_qc_plots_dimred(adata, cfg)
    qc_dir = Path(cfg.output_dir) / "figures" / "QC_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)
    sc.pl.umap(adata, color=[cfg.batch_key, "leiden"], save=qc_dir/"_QC_umap_per_sample_and_leiden.png") if cfg.make_figures else None
    # counts plots
    run_qc_plots_counts(adata, cfg)
    # write
    io_utils.save_adata(adata, cfg.output_dir / cfg.output_name)
    LOGGER.info("Finished load_and_qc")
    return adata
