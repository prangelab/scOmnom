from __future__ import annotations
import glob
import logging
from typing import Dict, List, Optional
import anndata as ad
import scanpy as sc
from .config import LoadAndQCConfig
from pathlib import Path
import shutil

LOGGER = logging.getLogger(__name__)

# Official CellTypist model registry
CELLTYPIST_REGISTRY_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"

# Local cache directory for downloaded models
CELLTYPIST_CACHE = Path.home() / ".cache" / "scomnom" / "celltypist_models"
CELLTYPIST_CACHE.mkdir(parents=True, exist_ok=True)


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


def filter_raw_barcodes(
    adata: ad.AnnData,
    plot: bool = False,
    plot_path: Optional[Path] = None,
) -> ad.AnnData:
    """
    Approximate Cell Ranger 'cell calling' on a raw_feature_bc_matrix.
    Uses a hybrid of Gaussian mixture modeling and knee detection to
    determine a permissive UMI cutoff separating background from cells.
    Keeps all barcodes above the selected cutoff in total UMI counts.

    If plot=True and plot_path is provided, the figure is passed through
    save_multi() and ends up in <figdir>/<fmt>/subdirs consistently.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    from sklearn.mixture import GaussianMixture
    from .plot_utils import save_multi

    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    sorted_idx = np.argsort(total_counts)[::-1]
    sorted_counts = total_counts[sorted_idx]
    ranks = np.arange(1, len(sorted_counts) + 1)

    # --- 1. Fit GMM (log space) ---
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

    # --- 2. Intersection of Gaussians ---
    a = 1/(2*sd_bg**2) - 1/(2*sd_cell**2)
    b = mu_cell/(sd_cell**2) - mu_bg/(sd_bg**2)
    c = (mu_bg**2)/(2*sd_bg**2) - (mu_cell**2)/(2*sd_cell**2) - np.log((sd_cell*w_bg)/(sd_bg*w_cell))
    disc = b*b - 4*a*c

    if disc < 0:
        log_thresh = mu_cell
    else:
        log_thresh = (-b + np.sqrt(disc)) / (2*a)

    umi_thresh = 10 ** log_thresh

    # --- 3. Knee detection ---
    kl = KneeLocator(ranks, sorted_counts, curve="convex", direction="decreasing")
    knee_rank = kl.elbow or len(sorted_counts)
    knee_value = sorted_counts[knee_rank - 1]

    # --- 4. Choose cutoff (geometric mean of knee + GMM thresholds) ---
    cutoff_value = np.sqrt(umi_thresh * knee_value)

    keep_mask = total_counts >= cutoff_value
    adata_filtered = adata[keep_mask].copy()

    # --- 5. Plot using save_multi ---
    if plot and plot_path is not None:
        stem = plot_path.stem
        figdir = plot_path.parent

        plt.figure(figsize=(5, 4))
        plt.plot(ranks, sorted_counts, lw=1, label="All barcodes")
        plt.axhline(cutoff_value, color="red", linestyle="--", label=f"Cutoff = {cutoff_value:.0f}")
        plt.axvline(knee_rank, color="orange", linestyle=":", label=f"Knee rank = {knee_rank}")
        plt.xlabel("Barcode rank")
        plt.ylabel("Total UMI counts")
        plt.title("Barcode rank vs total UMIs")
        plt.legend()
        plt.tight_layout()

        save_multi(stem, figdir)

    LOGGER.info(
        "Cell-calling (Cell Ranger-like): retained %d / %d barcodes (%.1f%%)",
        adata_filtered.n_obs, adata.n_obs,
        100 * adata_filtered.n_obs / len(total_counts)
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
    plot_dir: Optional[Path] = None,
) -> tuple[
    Dict[str, ad.AnnData],
    Dict[str, float],
    Optional[Dict[str, float]],
]:
    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)

    raw_map: Dict[str, ad.AnnData] = {}
    read_counts_filtered: Dict[str, float] = {}
    read_counts_unfiltered: Optional[Dict[str, float]] = {} if record_pre_filter_counts else None

    for raw in raw_dirs:
        sample = raw.name.split(".raw_feature_bc_matrix")[0]
        LOGGER.info(f"Loading RAW sample: {sample}")

        # 1) Read unfiltered matrix
        adata = read_raw_10x(raw)
        total_reads_unfiltered = float(adata.X.sum())
        if record_pre_filter_counts:
            read_counts_unfiltered[sample] = total_reads_unfiltered

        # 2) Determine plot_path
        plot_dir = Path(cfg.output_dir) / cfg.figdir_name
        if getattr(cfg, "make_figures", False) and plot_dir is not None:
            # always include subdir under the format namespace
            plot_path = plot_dir / "cell_qc" / f"{sample}_barcode_knee"
        elif getattr(cfg, "make_figures", False):
            # Fallback for load-and-qc:
            fallback_dir = (
                Path(cfg.output_dir) / cfg.figdir_name / "QC_plots"
            )
            plot_path = fallback_dir / f"{sample}_barcode_knee"
        else:
            plot_path = None

        # 3) Apply knee+GMM
        adata_f = filter_raw_barcodes(
            adata,
            plot=cfg.make_figures,
            plot_path=plot_path,
        )

        # 4) Store counts
        read_counts_filtered[sample] = float(adata_f.X.sum())
        raw_map[sample] = adata_f

        LOGGER.info(
            "Raw sample %s: %.3e → %.3e reads (before → after filtering); %d cells retained",
            sample,
            total_reads_unfiltered,
            read_counts_filtered[sample],
            adata_f.n_obs,
        )

    return raw_map, read_counts_filtered, read_counts_unfiltered



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


# =====================================================================
# CellTypist model handling
# =====================================================================
import hashlib
import requests


def _celltypist_cache_dir() -> Path:
    """Return the cache directory for CellTypist models."""
    d = Path.home() / ".cache" / "scomnom" / "celltypist_models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_available_celltypist_models() -> List[Dict[str, str]]:
    """
    Retrieve CellTypist model list (v1.7.x compatible).
    Returns list of dicts: {"name": model_name, "description": None}
    """
    try:
        import celltypist
        import celltypist.models as m
        import pandas as pd

        # This works in CellTypist 1.7.x
        models_info = m.models_description()

        # Convert output to DataFrame (this is what worked earlier)
        df = pd.DataFrame(models_info).T  # Rows: ["model", "description"]

        if "model" not in df.index:
            LOGGER.error("Could not parse CellTypist model list (no 'model' row).")
            return []

        model_names = df.loc["model"].dropna().tolist()

        return [{"name": str(name), "description": None} for name in model_names]

    except Exception as e:
        LOGGER.warning(f"Failed to retrieve CellTypist model list: {e}")
        return []


def _download_celltypist_model(url: str, out_path: Path, timeout: int = 60) -> None:
    """
    Download a CellTypist model from the given URL to out_path.
    """
    LOGGER.info("Downloading CellTypist model from %s", url)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download CellTypist model from {url}: {e}")


def get_celltypist_model(model_name_or_path: str) -> Path:
    """
    Resolve a CellTypist model path for CellTypist v1.7.1.

    Cases:
      1) Local file path -> return
      2) Ensure model exists in registry
      3) If cached -> return cached
      4) If not cached -> bulk-download all CellTypist models
                         then copy the requested one into scomnom’s cache
    """
    from pathlib import Path
    import celltypist.models as ct_models
    import shutil

    # -------------------------------
    # 1) User-specified local file
    # -------------------------------
    local_path = Path(model_name_or_path)
    if local_path.exists():
        LOGGER.info("Using local CellTypist model: %s", local_path)
        return local_path.resolve()

    # -------------------------------
    # 2) Query registry
    # -------------------------------
    models = get_available_celltypist_models()  # [{"name","description","cached"},...]
    if not models:
        raise RuntimeError(
            f"Model '{model_name_or_path}' is not a local file and the "
            f"remote model list cannot be retrieved (offline?)."
        )

    available = [m["name"] for m in models]
    if model_name_or_path not in available:
        raise RuntimeError(
            f"Model '{model_name_or_path}' not found in CellTypist registry.\n"
            f"Available models: {', '.join(available)}"
        )

    # -------------------------------
    # Determine cache directory
    # -------------------------------
    try:
        cache_dir = Path(ct_models.MODELS_DIR)
    except Exception:
        cache_dir = Path.home() / ".cache" / "scomnom" / "celltypist_models"

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = cache_dir / model_name_or_path

    # -------------------------------
    # 3) Cached copy available?
    # -------------------------------
    if cached_model.exists():
        LOGGER.info("Using cached CellTypist model: %s", cached_model)
        return cached_model.resolve()

    # -------------------------------
    # 4) Not cached → bulk-download all models
    # -------------------------------
    from .io_utils import download_all_celltypist_models

    LOGGER.info(
        "Model '%s' not cached. Downloading all CellTypist models "
        "because CellTypist v1.x does not support single-model download.",
        model_name_or_path,
    )

    download_all_celltypist_models()

    # Source location from CellTypist's internal cache
    src = Path.home() / ".celltypist" / "data" / "models" / model_name_or_path

    if not src.exists():
        raise RuntimeError(
            f"Bulk download completed but required model not found:\n{src}\n"
            "This indicates CellTypist failed to download this specific file."
        )

    shutil.copy2(src, cached_model)
    LOGGER.info("Copied model to scomnom cache: %s", cached_model)

    return cached_model.resolve()


def download_all_celltypist_models() -> None:
    """
    Download ALL official CellTypist models into the scomnom cache.

    Notes:
    - CellTypist v1.x does NOT support downloading a single model.
    - `download_models()` always downloads the full model registry.
    - This function calls CellTypist's downloader, then copies all
      downloaded models into scomnom's cache directory.
    """
    import celltypist
    import celltypist.models as ct_models

    LOGGER.info("Using CellTypist version %s", celltypist.__version__)
    LOGGER.info("Invoking CellTypist bulk model downloader")

    # 1) Let CellTypist download its *entire* registry
    try:
        ct_models.download_models()   # v1.x always downloads all 59 models
    except Exception as e:
        raise RuntimeError(f"CellTypist failed while downloading models: {e}")

    # 2) Identify CellTypist’s own storage directory
    ct_home = Path.home() / ".celltypist" / "data" / "models"
    if not ct_home.exists():
        raise RuntimeError(
            f"CellTypist reports models downloaded, but directory does not exist: {ct_home}"
        )

    # 3) Ensure scomnom cache exists
    cache_dir = _celltypist_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 4) Copy *.pkl models into scomnom cache
    copied = 0
    skipped = 0
    for pkl in ct_home.glob("*.pkl"):
        dest = cache_dir / pkl.name
        if dest.exists():
            LOGGER.info("Cached model already exists, skipping: %s", dest)
            skipped += 1
            continue

        LOGGER.info("Copying model to scomnom cache: %s", pkl.name)
        shutil.copy2(pkl, dest)
        copied += 1

    LOGGER.info(
        "Model sync complete: %d copied, %d skipped. Cache at: %s",
        copied,
        skipped,
        cache_dir,
    )


# =====================================================================
# Export cluster annotations (unchanged)
# =====================================================================
def export_cluster_annotations(adata: ad.AnnData, columns: List[str], out_path: Path) -> None:
    df = adata.obs[columns].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    LOGGER.info("Exported cluster annotations → %s", out_path)

