from __future__ import annotations
import scipy.sparse
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional
import anndata as ad
import scanpy as sc
from .config import LoadAndQCConfig

LOGGER = logging.getLogger(__name__)


def detect_sample_dirs(base: Path, patterns: list[str]) -> List[Path]:
    """
    Detect sample folders based on user-specified patterns in config.
    patterns is a list of glob patterns: e.g.
        ["*.raw_feature_bc_matrix", "*.filtered_feature_bc_matrix", "*.cellbender_filtered.output"]
    """
    out = []
    for pat in patterns:
        out.extend(Path(p) for p in glob.glob(str(base / pat)))
    return sorted(out)


def filter_raw_barcodes(adata: ad.AnnData, plot: bool = False, plot_path: Optional[Path] = None) -> ad.AnnData:
    """
    Approximate Cell Ranger 'cell calling' on a raw_feature_bc_matrix.
    Uses a hybrid of Gaussian mixture modeling and knee detection to
    determine a permissive UMI cutoff separating background from cells.
    Keeps all barcodes above the selected cutoff in total UMI counts.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    from sklearn.mixture import GaussianMixture

    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    sorted_idx = np.argsort(total_counts)[::-1]
    sorted_counts = total_counts[sorted_idx]
    ranks = np.arange(1, len(sorted_counts) + 1)

    # --- 1. Fit two-component GMM in log10(UMIs) space ---
    log_counts = np.log10(sorted_counts + 1).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0)
    gm.fit(log_counts)

    means = gm.means_.flatten()
    bg_comp = np.argmin(means)
    cell_comp = np.argmax(means)

    w_bg, w_cell = gm.weights_[bg_comp], gm.weights_[cell_comp]
    mu_bg, mu_cell = means[bg_comp], means[cell_comp]
    sd_bg = np.sqrt(gm.covariances_[bg_comp]).item()
    sd_cell = np.sqrt(gm.covariances_[cell_comp]).item()

    # --- 2. Analytical intersection of the two Gaussians ---
    a = 1/(2*sd_bg**2) - 1/(2*sd_cell**2)
    b = mu_cell/(sd_cell**2) - mu_bg/(sd_bg**2)
    c = (mu_bg**2)/(2*sd_bg**2) - (mu_cell**2)/(2*sd_cell**2) - np.log((sd_cell*w_bg)/(sd_bg*w_cell))
    disc = b**2 - 4*a*c
    if disc < 0:
        log_thresh = mu_cell  # fallback if intersection fails
    else:
        log_thresh = (-b + np.sqrt(disc)) / (2*a)
    umi_thresh = 10 ** log_thresh

    # --- 3. Knee (inflection) detection as fallback/upper bound ---
    kl = KneeLocator(ranks, sorted_counts, curve="convex", direction="decreasing")
    knee_rank = kl.elbow or len(sorted_counts)
    knee_value = sorted_counts[knee_rank - 1]

    # --- 4. Choose the *more permissive* cutoff ---
    cutoff_value = min(knee_value, umi_thresh)
    keep_mask = total_counts >= cutoff_value
    adata_filtered = adata[keep_mask].copy()

    # --- 5. Optional plot ---
    if plot and plot_path:
        plt.figure(figsize=(5, 4))
        plt.plot(ranks, sorted_counts, lw=1, label="All barcodes")
        plt.axhline(cutoff_value, color="red", linestyle="--", label="Cutoff")
        plt.axvline(knee_rank, color="orange", linestyle=":", label="Knee")
        plt.xlabel("Barcode rank")
        plt.ylabel("Total UMI counts")
        plt.title("Barcode rank vs UMI counts (Cell Ranger-like cutoff)")
        plt.legend()
        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path.with_suffix(".png"), dpi=300)
        plt.savefig(plot_path.with_suffix(".pdf"))
        plt.close()

    LOGGER.info(
        "Cell-calling (Cell Ranger-like): retained %d / %d barcodes (%.1f%%)",
        adata_filtered.n_obs, adata.n_obs, 100 * adata_filtered.n_obs / adata.n_obs
    )
    return adata_filtered

import logging
LOGGER = logging.getLogger(__name__)

def infer_batch_key_from_metadata_tsv(metadata_tsv: Path, user_batch_key: Optional[str]) -> str:
    """Infer or validate batch_key using only the metadata TSV header."""
    import pandas as pd
    import logging

    LOGGER = logging.getLogger(__name__)

    meta = pd.read_csv(metadata_tsv, sep="\t")
    cols = set(meta.columns)

    # User provided a batch key → validate
    if user_batch_key is not None:
        if user_batch_key not in cols:
            raise KeyError(
                f"batch_key '{user_batch_key}' not found in metadata columns: {sorted(cols)}"
            )
        return user_batch_key

    # Try standard candidates
    for cand in ("sample", "sample_id", "batch"):
        if cand in cols:
            LOGGER.warning(
                "No batch_key provided. Inferred batch_key='%s' from metadata.tsv. "
                "Please verify this is correct.", cand
            )
            return cand

    raise KeyError(
        f"Could not infer batch_key. metadata.tsv does not contain any of: "
        f"'sample', 'sample_id', 'batch'. Metadata columns: {sorted(cols)}"
    )

def infer_batch_key(adata, explicit_batch_key=None):
    # User explicitly provided key
    if explicit_batch_key is not None:
        if explicit_batch_key not in adata.obs:
            raise KeyError(
                f"batch_key '{explicit_batch_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return explicit_batch_key

    # Automatic inference
    for cand in ("sample_id", "sample", "batch"):
        if cand in adata.obs:
            LOGGER.warning(
                "Inferring batch_key='%s'. If this is incorrect, specify --batch-key explicitly.",
                cand,
            )
            return cand

    raise KeyError(
        "Could not infer batch_key automatically. None of ['sample_id', 'sample', 'batch'] "
        "found in adata.obs. Specify --batch-key explicitly."
    )


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


def load_raw_data(
    cfg: LoadAndQCConfig | CellQCConfig,
    record_pre_filter_counts: bool = False,
) -> tuple[
    Dict[str, ad.AnnData],
    Dict[str, float],   # filtered reads
    Optional[Dict[str, float]]  # unfiltered reads before knee+GMM
]:
    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)
    out: Dict[str, ad.AnnData] = {}
    read_counts_filtered: Dict[str, float] = {}
    read_counts_unfiltered: Dict[str, float] = {} if record_pre_filter_counts else None

    for raw in raw_dirs:
        sample = raw.name.split(".raw_feature_bc_matrix")[0]

        # Read matrix first
        adata = read_raw_10x(raw)
        total_reads_unfiltered = float(adata.X.sum())

        if record_pre_filter_counts:
            read_counts_unfiltered[sample] = total_reads_unfiltered

        # Now apply the knee+GMM filter
        if cfg.cellbender_dir is None:  # same condition as before
            qc_plot_dir = Path(cfg.output_dir) / "figures" / "QC_plots"
            plot_path = qc_plot_dir / f"{sample}_barcode_knee"
            adata = filter_raw_barcodes(
                adata,
                plot=cfg.make_figures,
                plot_path=plot_path,
            )

        # Final filtered reads
        total_reads_filtered = float(adata.X.sum())
        read_counts_filtered[sample] = total_reads_filtered

        out[sample] = adata
        LOGGER.info(
            "Loaded raw %s: %d cells, %d genes, %.2e → %.2e reads "
            "(before → after filtering)",
            sample, adata.n_obs, adata.n_vars,
            total_reads_unfiltered, total_reads_filtered
        )

    return out, read_counts_filtered, read_counts_unfiltered


def load_filtered_data(cfg: LoadAndQCConfig) -> tuple[Dict[str, ad.AnnData], Dict[str, float]]:
    filtered_dirs = find_raw_dirs(cfg.filtered_sample_dir, cfg.filtered_pattern)
    out: Dict[str, ad.AnnData] = {}
    read_counts: Dict[str, float] = {}

    for fd in filtered_dirs:
        sample = fd.name.split(".filtered_feature_bc_matrix")[0]
        adata = read_raw_10x(fd)
        out[sample] = adata
        total_reads = float(adata.X.sum())
        read_counts[sample] = total_reads
        LOGGER.info("Loaded filtered %s: %d cells, %d genes, %.2e total reads",
                    sample, adata.n_obs, adata.n_vars, total_reads)
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


def merge_samples(sample_map: Dict[str, ad.AnnData], batch_key: str) -> ad.AnnData:
    adatas = []

    for sample, ad in sample_map.items():
        ad = ad.copy()

        import scipy.sparse as sp
        if not sp.issparse(ad.X):
            ad.X = sp.csr_matrix(ad.X)
        else:
            ad.X = ad.X.tocsr()

        ad.layers["counts_raw"] = ad.X.copy()

        ad.obs[batch_key] = sample
        ad.obs_names = [f"{sample}_{bc}" for bc in ad.obs_names]

        adatas.append(ad)

    if not adatas:
        raise RuntimeError("No samples loaded.")

    adata_all = sc.concat(adatas, axis=0, join="outer", merge="first")
    adata_all.obs_names_make_unique()
    return adata_all


def save_adata(adata: ad.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(out_path), compression="gzip")
    LOGGER.info("Wrote %s", out_path)