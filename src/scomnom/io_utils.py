from __future__ import annotations
import scipy.sparse
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import anndata as ad
import scanpy as sc
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
from .config import LoadAndQCConfig

LOGGER = logging.getLogger(__name__)

def filter_raw_barcodes(adata: ad.AnnData, plot: bool = False, plot_path: Optional[Path] = None) -> ad.AnnData:
    """
    Approximate Cell Ranger 'cell calling' on a raw_feature_bc_matrix.
    Keeps barcodes above the knee point in the rank–UMI curve.
    """
    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    sorted_idx = np.argsort(total_counts)[::-1]
    sorted_counts = total_counts[sorted_idx]
    ranks = np.arange(1, len(sorted_counts) + 1)

    kl = KneeLocator(ranks, sorted_counts, curve="convex", direction="decreasing")
    cutoff_rank = kl.elbow or int(0.1 * len(sorted_counts))  # fallback
    cutoff_value = sorted_counts[cutoff_rank - 1]
    keep_mask = total_counts >= cutoff_value
    adata_filtered = adata[keep_mask].copy()

    if plot and plot_path:
        plt.figure(figsize=(5, 4))
        plt.plot(ranks, sorted_counts, lw=1)
        plt.axvline(cutoff_rank, color="red", linestyle="--")
        plt.xlabel("Barcode rank")
        plt.ylabel("Total UMI counts")
        plt.title("Barcode rank vs UMI counts")
        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path.with_suffix(".png"), dpi=300)
        plt.savefig(plot_path.with_suffix(".pdf"))
        plt.close()

    return adata_filtered

def find_raw_dirs(sample_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in glob.glob(str(sample_dir / pattern))]


def find_cellbender_dirs(cb_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in glob.glob(str(cb_dir / pattern))]


def read_raw_10x(raw_dir: Path) -> ad.AnnData:
    adata = sc.read_10x_mtx(str(raw_dir), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    return adata


def read_cellbender_h5(cb_folder: Path, sample: str, h5_suffix: str) -> Optional[ad.AnnData]:
    h5_path = cb_folder / f"{sample}{h5_suffix}"
    if not h5_path.exists():
        LOGGER.warning("CellBender output file missing for %s", sample)
        return None
    adata = sc.read_10x_h5(str(h5_path), gex_only=True)
    adata.var_names_make_unique()
    return adata


def load_raw_data(cfg: LoadAndQCConfig) -> tuple[Dict[str, ad.AnnData], Dict[str, float]]:
    raw_dirs = find_raw_dirs(cfg.sample_dir, cfg.raw_pattern)
    out: Dict[str, ad.AnnData] = {}
    read_counts: Dict[str, float] = {}

    for raw in raw_dirs:
        sample = raw.name.split(".raw_feature_bc_matrix")[0]
        adata = read_raw_10x(raw)
        if cfg.cellbender_dir is None:
            qc_plot_dir = Path(cfg.output_dir) / "figures" / "QC_plots"
            plot_path = qc_plot_dir / f"{sample}_barcode_knee"
            adata = filter_raw_barcodes(
                adata,
                plot=cfg.make_figures,
                plot_path=plot_path,
            )

        out[sample] = adata
        total_reads = float(adata.X.sum())
        read_counts[sample] = total_reads
        LOGGER.info(
            "Loaded raw %s: %d cells, %d genes, %.2e total reads",
            sample, adata.n_obs, adata.n_vars, total_reads,
        )
    return out, read_counts

def load_cellbender_data(cfg: LoadAndQCConfig) -> tuple[Dict[str, ad.AnnData], Dict[str, float]]:
    if cfg.cellbender_dir is None:
        return {}, {}

    cb_dirs = find_cellbender_dirs(cfg.cellbender_dir, cfg.cellbender_pattern)
    if not cb_dirs:
        raise FileNotFoundError(
            f"No CellBender outputs found in {cfg.cellbender_dir} matching pattern {cfg.cellbender_pattern}"
        )

    out: Dict[str, ad.AnnData] = {}
    read_counts: Dict[str, float] = {}

    for cb in cb_dirs:
        sample = cb.name.split(".cellbender_filtered.output")[0]
        adata = read_cellbender_h5(cb, sample, cfg.cellbender_h5_suffix)
        if adata is None:
            raise FileNotFoundError(f"Missing CellBender .h5 file for sample {sample} in {cb}")
        out[sample] = adata
        total_reads = float(adata.X.sum())
        read_counts[sample] = total_reads
        LOGGER.info(
            "Loaded CellBender %s: %d cells, %d genes, %.2e total reads",
            sample, adata.n_obs, adata.n_vars, total_reads,
        )
    return out, read_counts


def prefilter_low_content_cells(
    adata: ad.AnnData,
    min_genes: int,
    min_umis: int,
) -> ad.AnnData:
    """Remove barcodes with extremely low total UMIs or detected genes."""
    import scanpy as sc
    LOGGER.info("Running light prefilter: min_genes=%d, min_umis=%d", min_genes, min_umis)
    n0 = adata.n_obs
    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)
    mask = (adata.obs["n_genes_by_counts"] >= min_genes) & (
        adata.obs["total_counts"] >= min_umis
    )
    adata = adata[mask].copy()
    LOGGER.info("Prefiltered %d → %d cells (%.1f%% retained)", n0, adata.n_obs, 100 * adata.n_obs / n0)
    return adata


def merge_samples(raw_map: Dict[str, ad.AnnData], cb_map: Dict[str, ad.AnnData], batch_key: str) -> ad.AnnData:
    adatas: list[ad.AnnData] = []

    # When CellBender is used, restrict to intersection of sample sets
    if cb_map:
        shared_samples = sorted(set(raw_map) & set(cb_map))
        raw_map = {k: raw_map[k] for k in shared_samples}
    else:
        shared_samples = list(raw_map)

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
            raw = raw.copy()
            if not scipy.sparse.issparse(raw.X):
                raw.X = scipy.sparse.csr_matrix(raw.X)
            raw.layers["counts_raw"] = raw.X.copy()
            adata = raw

        adata.obs[batch_key] = sample
        adata.obs_names = [f"{sample}_{bc}" for bc in adata.obs_names]
        adatas.append(adata)

    if not adatas:
        raise RuntimeError("No samples loaded after matching.")

    LOGGER.info("Concatenating %d AnnData objects...", len(adatas))
    adata_all = sc.concat(adatas, axis=0, join="outer", merge="first")
    adata_all.obs_names_make_unique()

    if "counts_raw" in adata_all.layers:
        raw_counts = adata_all.layers["counts_raw"]
        if not scipy.sparse.issparse(raw_counts):
            raw_counts = scipy.sparse.csr_matrix(raw_counts)
        adata_all.raw = ad.AnnData(X=raw_counts, var=adata_all.var.copy(), obs=adata_all.obs.copy())
        LOGGER.info("adata.raw set with counts_raw layer.")
    else:
        LOGGER.warning("'counts_raw' layer not found; adata.raw not set.")

    return adata_all



def save_adata(adata: ad.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(out_path), compression="gzip")
    LOGGER.info("Wrote %s", out_path)