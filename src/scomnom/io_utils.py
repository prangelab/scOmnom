from __future__ import annotations
import glob
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anndata as ad
import scanpy as sc
from .config import LoadAndQCConfig
from pathlib import Path
import shutil
import os
import re
import json
import urllib.request
import urllib.error
import zarr
import numpy as np
import pandas as pd
import scipy.sparse as sp


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


def load_raw_data(
    cfg: LoadAndQCConfig,
    record_pre_filter_counts: bool = False,
    plot_dir: Optional[Path] = None,
):
    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)

    raw_map = {}
    read_counts_filtered = {}
    read_counts_unfiltered = {} if record_pre_filter_counts else None

    n_workers = min(8, cfg.n_jobs) if cfg.n_jobs else 8
    LOGGER.info(f"Parallel RAW 10X loading with {n_workers} I/O threads")

    def _load_one_raw(raw_path: Path):
        sample = raw_path.name.split(".raw_feature_bc_matrix")[0]
        adata = read_raw_10x(raw_path)
        cnt_raw = float(adata.X.sum())

        # Determine plot_path using your existing routing logic
        if getattr(cfg, "make_figures", False):
            base = Path(cfg.output_dir) / cfg.figdir_name / "cell_qc"
            plot_path = base / f"{sample}_barcode_knee"
        else:
            plot_path = None

        # Knee+GMM filtering
        adata_f = filter_raw_barcodes(
            adata,
            plot=cfg.make_figures,
            plot_path=plot_path,
        )
        cnt_filt = float(adata_f.X.sum())
        return sample, adata_f, cnt_raw, cnt_filt

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one_raw, raw): raw for raw in raw_dirs}

        for fut in as_completed(futures):
            sample, adata_f, cnt_raw, cnt_filt = fut.result()
            raw_map[sample] = adata_f
            read_counts_filtered[sample] = cnt_filt
            if record_pre_filter_counts:
                read_counts_unfiltered[sample] = cnt_raw

            LOGGER.info(
                f"[I/O] RAW sample {sample}: {cnt_raw:.2e} → {cnt_filt:.2e} UMIs; "
                f"{adata_f.n_obs} cells retained"
            )

    return raw_map, read_counts_filtered, read_counts_unfiltered


def load_filtered_data(cfg: LoadAndQCConfig):
    filtered_dirs = find_raw_dirs(cfg.filtered_sample_dir, cfg.filtered_pattern)

    out = {}
    read_counts = {}

    n_workers = min(8, cfg.n_jobs) if cfg.n_jobs else 8
    LOGGER.info(f"Parallel filtered 10X loading with {n_workers} I/O threads")

    def _load_one(fd: Path):
        sample = fd.name.split(".filtered_feature_bc_matrix")[0]
        adata = read_raw_10x(fd)
        total_reads = float(adata.X.sum())
        return sample, adata, total_reads

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, fd): fd for fd in filtered_dirs}

        for fut in as_completed(futures):
            sample, adata, total_reads = fut.result()
            out[sample] = adata
            read_counts[sample] = total_reads

            LOGGER.info(
                f"[I/O] Loaded filtered {sample}: {adata.n_obs} cells, "
                f"{adata.n_vars} genes, {total_reads:.2e} UMIs"
            )

    return out, read_counts


def load_cellbender_data(cfg: LoadAndQCConfig):
    """
    Stable CellBender loader using Scanpy’s read_10x_h5.

    Why this works:
      • Scanpy automatically fixes malformed CSR matrices.
      • We reorder rows using the provided cell barcode CSV.
      • We avoid all low-level sparse reconstruction (which caused earlier failures).
    """

    import concurrent.futures
    import pandas as pd
    import scanpy as sc

    cb_dirs = sorted(
        [p for p in cfg.cellbender_dir.glob(cfg.cellbender_pattern) if p.is_dir()]
    )

    LOGGER.info(f"Found {len(cb_dirs)} CellBender output dirs")

    out = {}
    read_counts = {}
    failed = {}

    def _load_one(cb_path: Path):
        sample = cb_path.name.replace(".cellbender_filtered.output", "")
        try:
            # --- Locate H5 ---
            h5_files = list(cb_path.glob("*_out_filtered.h5"))
            if not h5_files:
                return ("fail", sample, "No *_out_filtered.h5 found")
            h5_path = h5_files[0]

            LOGGER.info(f"[{sample}] Loading filtered CellBender H5: {h5_path}")

            # --- Locate barcodes CSV ---
            bc_files = list(cb_path.glob("*_out_cell_barcodes.csv"))
            if not bc_files:
                return ("fail", sample, "No *_out_cell_barcodes.csv found")

            barcodes = pd.read_csv(bc_files[0], header=None)[0].astype(str).tolist()

            # --- Load via Scanpy (robust) ---
            adata = sc.read_10x_h5(str(h5_path))
            adata.var_names_make_unique()

            # --- Select rows according to barcode CSV order ---
            try:
                adata = adata[barcodes, :].copy()
            except Exception:
                # Fallback: intersect but preserve CSV ordering
                LOGGER.warning(f"[{sample}] Barcode mismatch; falling back to intersection.")
                obs_index = set(adata.obs_names)
                keep = [bc for bc in barcodes if bc in obs_index]
                if not keep:
                    return ("fail", sample, "No barcodes overlap between H5 and CSV")
                adata = adata[keep, :].copy()

            # --- Add metadata ---
            adata.obs["barcode"] = adata.obs_names
            adata.obs["sample_id"] = sample

            # --- Compute UMI counts ---
            reads = int(adata.X.sum())

            return ("ok", sample, adata, reads)

        except Exception as e:
            return ("fail", sample, str(e))

    n_workers = min(cfg.n_jobs or 8, 8)
    LOGGER.info(f"Parallel CellBender loading with {n_workers} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, p): p for p in cb_dirs}

        for fut in concurrent.futures.as_completed(futures):
            status, sample, *rest = fut.result()

            if status == "ok":
                adata, reads = rest
                out[sample] = adata
                read_counts[sample] = reads
            else:
                errmsg = rest[0]
                LOGGER.error(f"[FAIL] {sample}: {errmsg}")
                failed[sample] = errmsg

    if failed:
        raise RuntimeError(
            f"CellBender loading failed for {len(failed)} samples: "
            f"{', '.join(failed.keys())}"
        )

    return out, read_counts


def save_adata(adata: ad.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(out_path), compression="gzip")
    LOGGER.info("Wrote %s", out_path)


# =====================================================================
# ZARR-BACKED STREAMING MERGER
# =====================================================================

class ZarrBackedMerger:
    """
    Memory-safe streaming merger for large AnnData objects.

    Design:
      • Computes union gene space
      • Two-pass algorithm:
            1) Detect nnz_total and row layout
            2) Stream padded CSR blocks into a Zarr CSR layout
      • Produces a disk-backed AnnData that never materializes the full matrix.

    Result:
      merged.zarr/  (ZarrGroup)
         X/data
         X/indices
         X/indptr
         obs/...
         var/...
    """

    def __init__(
        self,
        sample_map: Dict[str, ad.AnnData],
        batch_key: str,
        out_store: Path,
    ):
        self.sample_map = sample_map
        self.batch_key = batch_key
        self.out_store = Path(out_store)
        self.out_store.mkdir(parents=True, exist_ok=True)

        if not self.sample_map:
            raise RuntimeError("ZarrBackedMerger: sample_map is empty.")

        # Compute union gene set
        self.union_genes = self._compute_union_genes()
        LOGGER.info(f"[ZarrMerge] Union genes: {len(self.union_genes)}")

        # Compute total rows
        self.N_total = sum(a.n_obs for a in self.sample_map.values())

    # --------------------------------------------------------
    # Union gene helper
    # --------------------------------------------------------
    def _compute_union_genes(self) -> List[str]:
        genes = set()
        for a in self.sample_map.values():
            genes.update(a.var_names)
        return sorted(genes)

    # --------------------------------------------------------
    # Padding helper (in-memory CSR transform)
    # --------------------------------------------------------
    def _pad_to_union(self, X: sp.spmatrix, var_names: pd.Index) -> sp.csr_matrix:
        """
        Pads CSR matrix X (shape n_obs × old_genes) to the union gene space.
        """
        X = X.tocsr()
        old_genes = var_names.values
        old_to_new = np.searchsorted(self.union_genes, old_genes)
        X_padded = sp.csr_matrix(
            (X.data, old_to_new[X.indices], X.indptr),
            shape=(X.shape[0], len(self.union_genes)),
        )
        return X_padded

    # --------------------------------------------------------
    # First pass: determine nnz_total to allocate Zarr arrays
    # --------------------------------------------------------
    def _compute_nnz(self) -> int:
        nnz_total = 0
        LOGGER.info("[ZarrMerge] First pass: measuring nnz for allocation")

        for sample, adata in self.sample_map.items():
            Xp = self._pad_to_union(adata.X, adata.var_names)
            nnz_total += Xp.nnz
            LOGGER.info(f"[ZarrMerge] Sample {sample}: padded nnz={Xp.nnz}")

        LOGGER.info(f"[ZarrMerge] Total nnz: {nnz_total}")
        return nnz_total

    # --------------------------------------------------------
    # Write obs + var tables
    # --------------------------------------------------------
    def _write_obs_var(self, root: zarr.Group):
        # Concatenate obs in correct row order
        obs_list = []
        for sample, adata in self.sample_map.items():
            df = adata.obs.copy()
            df[self.batch_key] = sample
            obs_list.append(df)
        obs = pd.concat(obs_list, axis=0)
        obs.to_zarr(root.create_group("obs"))
        LOGGER.info("[ZarrMerge] Wrote obs table")

        # Write var table
        var = pd.DataFrame(index=self.union_genes)
        var.to_zarr(root.create_group("var"))
        LOGGER.info("[ZarrMerge] Wrote var table")

    # --------------------------------------------------------
    # Main merge
    # --------------------------------------------------------
    def merge(self) -> ad.AnnData:
        # --------------------------
        # First pass
        # --------------------------
        nnz_total = self._compute_nnz()

        # --------------------------
        # Create Zarr store
        # --------------------------
        store = zarr.DirectoryStore(str(self.out_store))
        root = zarr.group(store=store, overwrite=True)

        X_data = root.create_dataset(
            "X/data",
            shape=(nnz_total,),
            dtype="float32",
            chunks=(10_000_000,),
        )
        X_indices = root.create_dataset(
            "X/indices",
            shape=(nnz_total,),
            dtype="int32",
            chunks=(10_000_000,),
        )
        X_indptr = root.create_dataset(
            "X/indptr",
            shape=(self.N_total + 1,),
            dtype="int64",
            chunks=(100_000,),
        )

        LOGGER.info("[ZarrMerge] Allocated Zarr CSR arrays")

        # --------------------------
        # Second pass: write CSR
        # --------------------------
        data_cursor = 0
        row_cursor = 0

        for sample, adata in self.sample_map.items():
            LOGGER.info(f"[ZarrMerge] Writing sample {sample}")
            Xp = self._pad_to_union(adata.X, adata.var_names).tocsr()
            rows = Xp.shape[0]
            L = Xp.nnz

            # Write data + indices
            X_data[data_cursor:data_cursor + L] = Xp.data
            X_indices[data_cursor:data_cursor + L] = Xp.indices

            # Write indptr block (shifted)
            # NOTE: Xp.indptr is length rows+1; we fill rows entries here.
            X_indptr[row_cursor:row_cursor + rows] = Xp.indptr[:-1] + data_cursor

            data_cursor += L
            row_cursor += rows

            del Xp

        # Final indptr entry
        X_indptr[self.N_total] = nnz_total

        LOGGER.info("[ZarrMerge] Finished writing CSR arrays")

        # --------------------------
        # Write obs + var tables
        # --------------------------
        self._write_obs_var(root)

        # --------------------------
        # Build AnnData reference (backed=read)
        # --------------------------
        LOGGER.info(f"[ZarrMerge] Building backed AnnData → {self.out_store}")
        adata = ad.AnnData(
            X=root["X"],
            obs=pd.read_zarr(root["obs"]),
            var=pd.read_zarr(root["var"]),
            filename=str(self.out_store),
            backed="r",
        )

        return adata


# Public merge_samples() function to handle the merge
def merge_samples(
    sample_map: Dict[str, ad.AnnData],
    batch_key: str,
    out_path: Path,
) -> ad.AnnData:
    """
    Merge samples using the ZarrBackedMerger.

    Parameters
    ----------
    sample_map : Dict[str, AnnData]
        Per-sample filtered AnnData objects.
    batch_key : str
        Column added to obs to indicate sample identity.
    out_path : Path
        Location of the final merged Zarr store. Will be overwritten.

    Returns
    -------
    AnnData (backed='r')
        Zarr-backed merged dataset living directly at out_path.
    """
    if not sample_map:
        raise RuntimeError("merge_samples: sample_map is empty.")

    out_path = Path(out_path)
    LOGGER.info("[merge_samples] Zarr merge -> %s", out_path)

    merger = ZarrBackedMerger(
        sample_map=sample_map,
        batch_key=batch_key,
        out_store=out_path,
    )
    return merger.merge()



# =====================================================================
# ZARR I/O UTILITIES
# =====================================================================

def save_zarr(adata: ad.AnnData, out_path: Path) -> None:
    """
    Save a (possibly backed) AnnData object to a Zarr store.

    Parameters
    ----------
    adata : AnnData
        Can be backed='r' (pointing to a zarr store) or in-memory.
    out_path : Path
        Directory to write the Zarr store to.

    Behavior
    --------
    • If adata is backed='r' and already points to `out_path`, nothing happens.
    • If adata is backed and filename != out_path, this copies the Zarr store.
    • If adata is in-memory, this writes a new Zarr store.
    """

    out_path = Path(out_path)
    if adata.isbacked and adata.filename == str(out_path):
        LOGGER.info(f"Zarr store already at {out_path}; nothing to do.")
        return

    LOGGER.info(f"Saving AnnData Zarr store → {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    adata.write_zarr(str(out_path), chunks=None)
    LOGGER.info(f"Saved Zarr store: {out_path}")


def load_zarr(path: Path) -> ad.AnnData:
    """
    Load a Zarr-backed AnnData object.

    Parameters
    ----------
    path : Path
        Directory of a Zarr store previously written.

    Returns
    -------
    AnnData (backed='r')
    """
    path = Path(path)
    LOGGER.info(f"Loading Zarr store → {path}")
    return ad.read_zarr(str(path), backed="r")


def convert_zarr_to_h5ad(zarr_path: Path, out_h5ad: Path) -> None:
    """
    Convert a Zarr-backed dataset to a full H5AD file.

    CAUTION
    -------
    This will load the ENTIRE expression matrix into RAM.
    For very large datasets this may cause OOM.

    Only call this if the user explicitly requests H5AD output.
    """

    LOGGER.warning(
        "Converting Zarr → H5AD forces full matrix load and may cause OOM "
        f"(input={zarr_path}). Proceeding…"
    )

    adata = ad.read_zarr(str(zarr_path))   # loads into memory
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Writing full H5AD → {out_h5ad}")
    adata.write(str(out_h5ad), compression="gzip")
    LOGGER.info(f"Wrote {out_h5ad}")


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


# =====================================================================
# ssGSEA IO helpers
# =====================================================================
MSIGDB_BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
MSIGDB_INDEX_FILENAME = "msigdb_index.json"


def _get_msigdb_cache_dir() -> Path:
    """
    Return the local cache directory for MSigDB gene sets.

    Uses:
      - env var SCOMNOM_MSIGDB_DIR if set
      - otherwise ~/.cache/scomnom/msigdb
    """
    override = os.environ.get("SCOMNOM_MSIGDB_DIR", None)
    if override is not None:
        cache_dir = Path(override).expanduser()
    else:
        cache_dir = Path.home() / ".cache" / "scomnom" / "msigdb"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _http_get(url: str) -> bytes:
    """
    Minimal HTTP GET helper with basic error handling.
    """
    LOGGER.debug("HTTP GET: %s", url)
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch URL {url!r}: {e}") from e


def _discover_latest_msigdb_release(species_code: str = "Hs") -> str:
    """
    Scrape the MSigDB release directory and return the latest <version>.<species> string,
    e.g. '2023.1.Hs'.

    If discovery fails, falls back to a hard-coded default.
    """
    url = MSIGDB_BASE_URL + "/"
    try:
        html = _http_get(url).decode("utf-8", errors="ignore")
        # Look for directory names like 2023.1.Hs/
        pattern = rf"(\d{{4}}\.\d+\.{re.escape(species_code)})/"
        candidates = set(re.findall(pattern, html))
        if not candidates:
            raise RuntimeError("No MSigDB release directories found in index HTML.")

        def _version_key(v: str) -> Tuple[int, int]:
            # v looks like "2023.1.Hs"
            parts = v.split(".")
            year = int(parts[0])
            sub = int(parts[1])
            return year, sub

        latest = sorted(candidates, key=_version_key)[-1]
        LOGGER.info("Detected latest MSigDB release: %s", latest)
        return latest
    except Exception as e:
        fallback = f"2023.1.{species_code}"
        LOGGER.warning(
            "Could not auto-discover latest MSigDB release (%s). "
            "Falling back to %s.",
            e,
            fallback,
        )
        return fallback


def _download_msigdb_release(release: str) -> Dict[str, str]:
    """
    Download ALL *.symbols.gmt files for the given MSigDB release into the cache,
    and build a keyword → filepath index.

    Returns
    -------
    index : Dict[str, str]
        Mapping from keyword (e.g. 'HALLMARK', 'REACTOME', 'C2_CP_REACTOME') to
        local .gmt file path (as str).
    """
    cache_dir = _get_msigdb_cache_dir()
    release_dir = cache_dir / release
    release_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / MSIGDB_INDEX_FILENAME

    LOGGER.info("Downloading MSigDB release %s into %s", release, release_dir)

    # Fetch directory listing for this release
    base_url = f"{MSIGDB_BASE_URL}/{release}/"
    html = _http_get(base_url).decode("utf-8", errors="ignore")

    # Find all *.symbols.gmt files
    gmt_files = sorted(set(re.findall(r'href="([^"]+\.symbols\.gmt)"', html)))
    if not gmt_files:
        raise RuntimeError(f"No .symbols.gmt files found in MSigDB release {release}")

    keyword_to_path: Dict[str, str] = {}

    for fname in gmt_files:
        file_url = base_url + fname
        dest = release_dir / fname
        if not dest.exists():
            LOGGER.info("Downloading MSigDB file: %s", fname)
            data = _http_get(file_url)
            dest.write_bytes(data)
        else:
            LOGGER.debug("MSigDB file already cached: %s", dest)

        # Build keyword aliases from filename
        # Example filenames:
        #   h.all.v2023.1.Hs.symbols.gmt
        #   c2.cp.reactome.v2023.1.Hs.symbols.gmt
        #   c5.go.bp.v2023.1.Hs.symbols.gmt
        base = fname.split(".v")[0]  # "h.all" or "c2.cp.reactome"
        parts = base.split(".")

        # Canonical key: whole prefix in upper snake-case
        canonical_key = "_".join(parts).upper()  # e.g. "H_ALL" or "C2_CP_REACTOME"
        keyword_to_path[canonical_key] = str(dest)

        # Also add "tail" keyword for convenience where meaningful
        # - hallmark: h.all -> HALLMARK
        # - reactome: c2.cp.reactome -> REACTOME
        # - wikipathways: c2.cp.wikipathways -> WIKIPATHWAYS
        if len(parts) >= 2:
            tail = parts[-1].upper()
            # Avoid overwriting if already mapped
            if tail not in keyword_to_path:
                keyword_to_path[tail] = str(dest)

        # Special case: hallmark collection is usually treated as "HALLMARK"
        if base.startswith("h.all") and "HALLMARK" not in keyword_to_path:
            keyword_to_path["HALLMARK"] = str(dest)

    # Persist index
    payload = {
        "release": release,
        "files": keyword_to_path,
    }
    index_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info(
        "MSigDB index written to %s (keywords: %d)", index_path, len(keyword_to_path)
    )
    return keyword_to_path


def _load_msigdb_index() -> Tuple[str, Dict[str, str]]:
    """
    Load the MSigDB keyword → filepath index from cache.

    Returns
    -------
    (release, index) where index is a dict: keyword -> filepath

    If index is missing or invalid, triggers a fresh download of the latest
    MSigDB release and rebuilds the index.
    """
    cache_dir = _get_msigdb_cache_dir()
    index_path = cache_dir / MSIGDB_INDEX_FILENAME

    if index_path.exists():
        try:
            obj = json.loads(index_path.read_text())
            release = obj.get("release", None)
            files = obj.get("files", {})
            if isinstance(release, str) and isinstance(files, dict) and files:
                LOGGER.info(
                    "Loaded MSigDB index from %s (release %s, keywords=%d)",
                    index_path,
                    release,
                    len(files),
                )
                return release, files
        except Exception as e:
            LOGGER.warning("Failed to parse MSigDB index %s: %s", index_path, e)

    # If we get here, we need to discover & download a release
    release = _discover_latest_msigdb_release(species_code="Hs")
    files = _download_msigdb_release(release)
    return release, files


def list_available_msigdb_keywords() -> List[str]:
    """
    Return all known MSigDB keyword aliases (e.g. 'HALLMARK', 'REACTOME', 'C2_CP_REACTOME').

    This is mainly useful for CLI autocompletion / help.
    """
    _, index = _load_msigdb_index()
    return sorted(index.keys())


def resolve_msigdb_gene_sets(
    user_spec: List[str] | None,
) -> Tuple[List[str], List[str]]:
    """
    Resolve user-provided ssGSEA gene set specifiers into local .gmt file paths.

    Parameters
    ----------
    user_spec : list[str] or None
        - None or [] -> default ['HALLMARK', 'REACTOME'] (MSigDB collections)
        - Each item can be:
            * an MSigDB keyword (e.g. 'HALLMARK', 'REACTOME', 'C2_CP_REACTOME')
            * a local/remote path to a .gmt file (endswith '.gmt')

    Returns
    -------
    (gmt_files, used_keywords)
        gmt_files     : list of file paths to feed into ssGSEA
        used_keywords : list of resolved keywords (for logging / metadata)

    Raises
    ------
    ValueError if no usable gene sets could be resolved.
    """
    if not user_spec:
        LOGGER.info(
            "ssGSEA: no gene sets provided; defaulting to MSigDB HALLMARK + REACTOME."
        )
        spec = ["HALLMARK", "REACTOME"]
    else:
        spec = [str(x).strip() for x in user_spec if str(x).strip()]

    if not spec:
        raise ValueError("Empty ssGSEA gene-set specification.")

    release, index = _load_msigdb_index()
    LOGGER.info("Resolving ssGSEA gene sets for MSigDB release %s", release)

    gmt_files: List[str] = []
    used_keywords: List[str] = []
    unresolved: List[str] = []

    for item in spec:
        # Custom GMT file: user path
        if item.lower().endswith(".gmt"):
            path = Path(item)
            if not path.is_file():
                LOGGER.warning(
                    "Custom GMT file '%s' does not exist; skipping.", path
                )
                unresolved.append(item)
                continue
            gmt_files.append(str(path))
            used_keywords.append(str(path))
            continue

        # MSigDB keyword
        key = item.upper()
        if key not in index:
            LOGGER.warning(
                "ssGSEA: unknown MSigDB keyword '%s'. Known examples: %s",
                key,
                ", ".join(sorted(index.keys())[:10]),
            )
            unresolved.append(item)
            continue

        gmt_files.append(index[key])
        used_keywords.append(key)

    if not gmt_files:
        raise ValueError(
            f"No resolvable MSigDB gene sets from spec: {spec}. "
            f"Unresolved: {unresolved}"
        )

    LOGGER.info(
        "Resolved ssGSEA gene sets: %s",
        ", ".join(f"{k} -> {p}" for k, p in zip(used_keywords, gmt_files)),
    )

    return gmt_files, used_keywords
