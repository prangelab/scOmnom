from __future__ import annotations
import glob
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
import re
import json
import urllib.request
import urllib.error


LOGGER = logging.getLogger(__name__)

# Official CellTypist model registry
CELLTYPIST_REGISTRY_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"

# Local cache directory for downloaded models
CELLTYPIST_CACHE = Path.home() / ".cache" / "scomnom" / "celltypist_models"
CELLTYPIST_CACHE.mkdir(parents=True, exist_ok=True)


def _downgrade_nullable_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]

        # 1. Handle Categoricals
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = s.cat.categories
            # Check if categories are the 'new' string type
            if hasattr(cats, "dtype") and str(cats.dtype).startswith(("string", "arrow")):
                # Convert categories to plain objects, preserving None/NaN properly
                new_cats = cats.astype(object).fillna(np.nan)
                out[col] = s.cat.rename_categories(new_cats)
            continue

        # 2. Handle standard nullable string columns
        if str(s.dtype).startswith(("string", "arrow")):
            # to_numpy(dtype=object) is the safest 'downgrade' path
            # as it handles the pd.NA -> np.nan conversion correctly
            out[col] = s.to_numpy(dtype=object)

    return out


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

    # --- 4. Choose cutoff (geometric mean of knee + GMM thresholds) + 20% to tighten it a little ---
    cutoff_value = 1.2 * np.sqrt(umi_thresh * knee_value)

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

    if "gene_ids" not in adata.var.columns:
        if "gene_ids" in adata.var_names:
            adata.var["gene_ids"] = adata.var_names.astype(str)
        elif "gene_ids" in adata.var.columns:
            adata.var["gene_ids"] = adata.var["gene_ids"].astype(str)
        else:
            raise RuntimeError("Cannot determine gene_ids from 10x input")

    adata.var_names_make_unique()
    return adata


def write_filtered_samples_parallel(
    sample_map: Dict[str, ad.AnnData],
    out_dir: Path,
    n_jobs: int | None = None,
    suffix: str = ".filtered.h5ad",
) -> Dict[str, Path]:
    """
    Write per-sample filtered AnnData objects to disk in parallel.

    Parameters
    ----------
    sample_map : Dict[str, AnnData]
        Mapping sample_id -> filtered AnnData (already QC'ed / filtered).
    out_dir : Path
        Directory where per-sample .h5ad files will be stored.
    n_jobs : int or None
        Max number of parallel writer threads. If None, defaults to 8.
    suffix : str
        Filename suffix to append to each sample id.

    Returns
    -------
    Dict[str, Path]
        Mapping sample_id -> path to written .h5ad file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_map:
        raise RuntimeError("write_filtered_samples_parallel: sample_map is empty.")

    max_workers = min(8, n_jobs or 8)
    LOGGER.info(
        "Writing %d filtered samples to disk (%d writer threads) → %s",
        len(sample_map),
        max_workers,
        out_dir,
    )

    def _write_one(sample: str, adata: ad.AnnData) -> tuple[str, Path]:
        # Use a simple, deterministic filename: <sample><suffix>
        out_path = out_dir / f"{sample}{suffix}"
        LOGGER.info("[Write] %s → %s", sample, out_path)
        adata.write_h5ad(str(out_path), compression="gzip")
        return sample, out_path

    sample_paths: Dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_write_one, sample, adata): sample
            for sample, adata in sample_map.items()
        }

        for fut in as_completed(futures):
            sample, out_path = fut.result()
            sample_paths[sample] = out_path

    LOGGER.info(
        "Finished writing %d filtered samples to %s", len(sample_paths), out_dir
    )
    return sample_paths


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


def load_raw_with_cellbender_barcodes(
    cfg: LoadAndQCConfig,
    plot_dir: Path | None = None,
) -> tuple[Dict[str, ad.AnnData], Dict[str, float]]:
    """
    Load RAW 10x matrices and immediately restrict to CellBender-called barcodes.
    Counts remain raw integers.

    Returns
    -------
    sample_map : Dict[sample_id, AnnData]
    read_counts : Dict[sample_id, total_UMIs_after_barcode_filter]
    """
    import pandas as pd
    import scanpy as sc
    from concurrent.futures import ThreadPoolExecutor, as_completed

    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)
    cb_dirs = find_cellbender_dirs(cfg.cellbender_dir, cfg.cellbender_pattern)

    cb_map = {
        p.name.replace(".cellbender_filtered.output", ""): p
        for p in cb_dirs
    }

    missing = set(d.name.split(".raw_feature_bc_matrix")[0] for d in raw_dirs) - cb_map.keys()
    if missing:
        raise RuntimeError(
            "Missing CellBender outputs for samples:\n"
            + "\n".join(sorted(missing))
        )

    out: Dict[str, ad.AnnData] = {}
    read_counts: Dict[str, float] = {}

    def _load_one(raw_path: Path):
        sample = raw_path.name.split(".raw_feature_bc_matrix")[0]
        cb_path = cb_map[sample]

        # ---- load raw counts ----
        adata = read_raw_10x(raw_path)   # integer counts
        adata.obs_names = adata.obs_names.astype(str)

        # ---- read CellBender barcodes ----
        bc_file = cb_path / f"{sample}{cfg.cellbender_barcode_suffix}"
        if not bc_file.exists():
            raise FileNotFoundError(
                f"Expected CellBender barcode file not found:\n"
                f"  {bc_file}\n"
                f"Check --cellbender-barcode-suffix."
            )

        barcodes = pd.read_csv(bc_file, header=None)[0].astype(str).tolist()
        LOGGER.info(
            "[%s] Using CellBender barcode file: %s",
            sample,
            bc_file.name,
        )

        # ---- restrict immediately ----
        keep = adata.obs_names.isin(barcodes)
        adata = adata[keep].copy()

        adata.obs["barcode"] = adata.obs_names
        adata.obs["sample_id"] = sample

        reads = float(adata.X.sum())
        return sample, adata, reads

    n_workers = min(cfg.n_jobs or 8, 8)
    LOGGER.info("Loading raw + CellBender barcodes with %d threads", n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_load_one, p) for p in raw_dirs]
        for fut in as_completed(futures):
            sample, adata, reads = fut.result()
            out[sample] = adata
            read_counts[sample] = reads
            LOGGER.info(
                "[CB+RAW] %s: %d cells, %.2e UMIs",
                sample, adata.n_obs, reads
            )

    return out, read_counts


def load_cellbender_filtered_data(
    cfg: LoadAndQCConfig,
) -> tuple[Dict[str, ad.AnnData], Dict[str, float]]:
    """
    Load CellBender-filtered expression matrices as the PRIMARY data space.

    Returns
    -------
    sample_map : Dict[sample_id, AnnData]
        AnnData objects with X = CellBender-denoised counts
    read_counts : Dict[sample_id, float]
        Total UMIs per sample (post-CellBender)
    """
    import scanpy as sc
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cb_dirs = find_cellbender_dirs(cfg.cellbender_dir, cfg.cellbender_pattern)
    if not cb_dirs:
        raise RuntimeError("No CellBender outputs found")

    sample_map: Dict[str, ad.AnnData] = {}
    read_counts: Dict[str, float] = {}

    def _load_one(cb_path: Path):
        sample = cb_path.name.replace(".cellbender_filtered.output", "")
        h5 = cb_path / f"{sample}{cfg.cellbender_h5_suffix}"
        if not h5.exists():
            raise FileNotFoundError(f"Missing CellBender H5 file: {h5}")

        adata = sc.read_10x_h5(h5)
        adata.var_names_make_unique()
        adata.obs_names = adata.obs_names.astype(str)

        adata.obs["barcode"] = adata.obs_names
        adata.obs["sample_id"] = sample

        total_reads = float(adata.X.sum())
        return sample, adata, total_reads

    n_workers = min(cfg.n_jobs or 8, 8)
    LOGGER.info("Loading CellBender filtered matrices with %d threads", n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_load_one, p) for p in cb_dirs]
        for fut in as_completed(futures):
            sample, adata, reads = fut.result()
            sample_map[sample] = adata
            read_counts[sample] = reads
            LOGGER.info(
                "[CB] %s: %d cells × %d genes, %.2e UMIs",
                sample, adata.n_obs, adata.n_vars, reads
            )

    return sample_map, read_counts


from concurrent.futures import ThreadPoolExecutor, as_completed

def attach_raw_counts_postfilter(
    cfg: LoadAndQCConfig,
    adata: ad.AnnData,
) -> ad.AnnData:

    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key is None:
        raise RuntimeError("attach_raw_counts_postfilter requires batch_key")

    required = {"barcode", batch_key}
    if not required.issubset(adata.obs.columns):
        raise RuntimeError(f"adata.obs must contain {required}")

    if not adata.var_names.is_unique:
        raise RuntimeError("adata.var_names must be unique")

    # --------------------------------------------------
    # Build global cell key
    # --------------------------------------------------
    cell_key = (
        adata.obs[batch_key].astype(str)
        + "::"
        + adata.obs["barcode"].astype(str)
    )
    cell_key_set = set(cell_key)

    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)
    if not raw_dirs:
        raise RuntimeError("No raw data directories found")

    sample_to_path = {
        p.name.split(".raw_feature_bc_matrix")[0]: p
        for p in raw_dirs
    }

    # --------------------------------------------------
    # Worker: load + subset ONE sample
    # --------------------------------------------------
    def _load_one_raw(sample: str, raw_path: Path):
        LOGGER.info("Loading raw counts for sample %s", sample)

        raw = read_raw_10x(raw_path)
        raw.var_names_make_unique()
        raw.obs_names = raw.obs_names.astype(str)

        raw.obs["_raw_key"] = sample + "::" + raw.obs_names
        raw = raw[raw.obs["_raw_key"].isin(cell_key_set)].copy()

        if raw.n_obs == 0:
            raise RuntimeError(
                f"No overlapping cells between raw and filtered data for {sample}"
            )

        return raw

    # --------------------------------------------------
    # Parallel load (THREADS, not processes)
    # --------------------------------------------------
    raw_layers = []
    max_workers = min(8, cfg.n_jobs or 8)

    LOGGER.info(
        "Loading raw counts for %d samples using %d threads",
        len(sample_to_path),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_one_raw, sample, path): sample
            for sample, path in sample_to_path.items()
        }

        for fut in as_completed(futures):
            raw = fut.result()
            raw_layers.append(raw)

    if not raw_layers:
        raise RuntimeError("No raw data could be aligned")

    # --------------------------------------------------
    # Concatenate + reorder (unchanged from your version)
    # --------------------------------------------------
    raw_all = ad.concat(raw_layers, axis=0, join="outer", fill_value=0)

    raw_all.obs_names = raw_all.obs["_raw_key"].astype(str)
    raw_all.obs_names_make_unique()

    missing = cell_key[~cell_key.isin(raw_all.obs_names)]
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} cells missing in raw counts "
            f"(example: {missing.iloc[0]})"
        )

    raw_all = raw_all[cell_key.values, :]
    raw_all = raw_all[:, adata.var_names]

    adata.layers["counts_raw"] = raw_all.X.copy()

    LOGGER.info(
        "Attached raw counts layer: %s",
        adata.layers["counts_raw"].shape,
    )

    return adata


def _prepare_sample_for_merge(
    sample_name: str,
    adata: ad.AnnData,
    union_genes: list[str],
    out_dir: Path,
    batch_key: str,
    input_layer_name: str,
) -> Path:
    import numpy as np
    import scipy.sparse as sp
    import pandas as pd
    import anndata as ad
    pd.set_option('future.no_silent_downcasting', True)

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    if adata.var_names.is_unique is False:
        raise RuntimeError(f"[{sample_name}] adata.var_names are not unique")

    if adata.obs_names.is_unique is False:
        raise RuntimeError(f"[{sample_name}] adata.obs_names are not unique")

    # ------------------------------------------------------------------
    # Pad X to union gene space
    # ------------------------------------------------------------------
    X = adata.X.tocsr()

    old_to_new = np.searchsorted(union_genes, adata.var_names)

    X_padded = sp.csr_matrix(
        (X.data, old_to_new[X.indices], X.indptr),
        shape=(adata.n_obs, len(union_genes)),
    )

    var = pd.DataFrame(index=union_genes)

    for col in adata.var.columns:
        s0 = adata.var[col]

        if pd.api.types.is_bool_dtype(s0):
            # Fill new genes with False for boolean masks
            s = s0.reindex(var.index, fill_value=False).astype(bool)

        elif pd.api.types.is_numeric_dtype(s0):
            # Fill missing numeric values with 0
            s = s0.reindex(var.index, fill_value=0)
            # Ensure we didn't end up with an Object array
            if s.dtype == object:
                s = s.infer_objects(copy=False)

        elif isinstance(s0.dtype, pd.CategoricalDtype):
            # Categoricals are special; missing values remain as 'NaN' categories
            # We reindex normally then ensure categorical type is preserved
            s = s0.reindex(var.index).astype("category")
            try:
                s = s.cat.set_categories(s0.cat.categories)
            except Exception:
                pass

        elif pd.api.types.is_string_dtype(s0):
            # For strings, Zarr/H5AD prefer empty strings over Float-NaNs
            # We convert to string first to handle nullable types safely
            s = s0.reindex(var.index).astype(str).replace("nan", "")

        else:
            # Fallback for any other complex types
            s = s0.reindex(var.index).fillna("")

        var[col] = s.values

    # Sanity: gene_ids must survive if they existed
    if "gene_ids" in adata.var.columns and "gene_ids" not in var.columns:
        raise RuntimeError(f"[{sample_name}] gene_ids lost during merge padding")

    # ------------------------------------------------------------------
    # Build padded AnnData
    # ------------------------------------------------------------------
    layers = {input_layer_name: X_padded.copy()}

    a = ad.AnnData(
        X=X_padded,
        obs=adata.obs.copy(),
        var=var,
        layers=layers,
    )

    a.obs[batch_key] = sample_name
    a.obs_names = [f"{sample_name}_{x}" for x in adata.obs_names]

    a.var = _downgrade_nullable_strings(a.var)
    a.obs = _downgrade_nullable_strings(a.obs)

    # ------------------------------------------------------------------
    # Write padded Zarr
    # ------------------------------------------------------------------
    out_path = out_dir / f"{sample_name}.padded.zarr"
    LOGGER.info("[Pad] %s → %s", sample_name, out_path)
    a.write_zarr(str(out_path), chunks=None)

    return out_path


def _compute_union_genes(sample_map: Dict[str, ad.AnnData]) -> List[str]:
    genes = set()
    for a in sample_map.values():
        genes.update(a.var_names)
    return sorted(genes)


def _merge_filtered_zarr_simple(padded_dirs: List[Path]) -> ad.AnnData:
    """
    Sequentially merge padded .zarr AnnData stores in-memory using ad.concat.
    """
    merged: Optional[ad.AnnData] = None

    for i, p in enumerate(padded_dirs, 1):
        LOGGER.info(f"[Merge {i:02d}/{len(padded_dirs)}] Loading {p.name}")
        a = ad.read_zarr(str(p))  # load into memory (NOT backed)

        if merged is None:
            merged = a
            continue

        LOGGER.info(f"[Merge {i:02d}/{len(padded_dirs)}] Concatenating in-memory...")
        merged = ad.concat(
            [merged, a],
            axis=0,
            join="outer",
            merge="first",
        )

        # Free memory aggressively
        del a

    if merged is None:
        raise RuntimeError("_merge_filtered_zarr_simple: no inputs?")

    return merged


def merge_samples(
    sample_map: Dict[str, ad.AnnData],
    batch_key: str,
    input_layer_name: str,
) -> ad.AnnData:
    """
    Memory-safe union-gene merge using disk-backed padded Zarr intermediates.

    Behavior:
      1. Compute union gene list across samples.
      2. Pad each sample to the union gene space → <sample>.padded.zarr.
      3. Sequential in-memory merge using ad.concat (reliable + fast).
      5. Temporary padded files are cleaned up automatically.

    Returns
    -------
    AnnData
        Fully in-memory merged AnnData. (backed=None)
    """

    import concurrent.futures

    if not sample_map:
        raise RuntimeError("merge_samples: sample_map is empty.")

    tmp_dir = Path.cwd() / "tmp_merge"
    padded_dir = tmp_dir / "padded"
    padded_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting merge of %d samples. tmp_merge=%s",
        len(sample_map), tmp_dir
    )

    # --------------------------------------------------------
    # 1. Compute union genes
    # --------------------------------------------------------
    union_genes = _compute_union_genes(sample_map)
    LOGGER.info("Union gene set contains %d genes", len(union_genes))

    # --------------------------------------------------------
    # 2. Prepare padded Zarrs in parallel
    # --------------------------------------------------------
    n_workers = min( 8, len(sample_map))
    LOGGER.info("Writing padded Zarrs with %d workers", n_workers)

    padded_files: List[Path] = []

    def _worker(sample: str, adata: ad.AnnData) -> Path:
        return _prepare_sample_for_merge(
            sample_name=sample,
            adata=adata,
            union_genes=union_genes,
            out_dir=padded_dir,
            batch_key=batch_key,
            input_layer_name=input_layer_name,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, s, a): s for s, a in sample_map.items()}
        count = 0
        for fut in concurrent.futures.as_completed(futures):
            sample = futures[fut]
            count += 1
            path = fut.result()
            padded_files.append(path)
            LOGGER.info("[Pad %02d/%02d] %s → %s", count, len(sample_map), sample, path.name)

    if not padded_files:
        raise RuntimeError("merge_samples: no padded Zarr stores were created.")

    padded_files = sorted(padded_files)

    # --------------------------------------------------------
    # 3. Merge all padded samples into memory
    # --------------------------------------------------------
    LOGGER.info(
        "Merging %d padded Zarrs via ad.concat (fully in-memory)...",
        len(padded_files)
    )
    merged = _merge_filtered_zarr_simple(padded_files)

    LOGGER.info(
        "Merged dataset: %d cells × %d genes",
        merged.n_obs, merged.n_vars
    )

    # --------------------------------------------------------
    # 4. Cleanup temporary padded files
    # --------------------------------------------------------
    try:
        LOGGER.info("Cleaning tmp_merge directory: %s", tmp_dir)
        shutil.rmtree(tmp_dir)
    except Exception as e:
        LOGGER.warning("Failed to clean temporary dir %s: %s", tmp_dir, e)

    # --------------------------------------------------------
    # 5. Return in-memory AnnData (caller will write final Zarr)
    # --------------------------------------------------------
    LOGGER.info(
        "merge_samples() complete. Returning in-memory AnnData. "
        "Caller is responsible for writing the final Zarr."
    )
    return merged


# ============================================================
# Dataset IO Helpers
# ============================================================
def save_dataset(adata: ad.AnnData, out_path: Path, fmt: str = "zarr") -> None:
    """
    Save an AnnData object either as .zarr or .h5ad.

    Automatically makes adata.uns (and nested structures) safe for H5AD/Zarr by:
      - converting any nested pandas.DataFrame into a tagged, json-compatible dict
      - converting numpy arrays to lists where needed (
      - ensuring keys are strings

    This is done on a shallow copy of adata so the in-memory object remains unchanged.
    Large numerical arrays should never be stored in .uns. Arrays found here are assumed to be small metadata.

    Additionally:
      - Emits WARNINGS (only) if adata.uns or individual entries are large.
    """
    import copy
    import sys
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------
    # Size estimation (warn-only)
    # ------------------------------------------------------------
    def _estimate_size_bytes(obj) -> int:
        try:
            if isinstance(obj, np.ndarray):
                return int(obj.nbytes)

            if isinstance(obj, pd.DataFrame):
                return int(obj.memory_usage(deep=True).sum())

            if isinstance(obj, dict):
                return sum(_estimate_size_bytes(v) for v in obj.values())

            if isinstance(obj, (list, tuple)):
                return sum(_estimate_size_bytes(v) for v in obj)

            return int(sys.getsizeof(obj))
        except Exception:
            return 0

    # ------------------------------------------------------------
    # Sanitization helpers
    # ------------------------------------------------------------
    def _df_to_tagged_payload(df: pd.DataFrame) -> dict:
        payload = df.to_dict(orient="split")  # keys: index, columns, data
        return {
            "__type__": "pandas.DataFrame",
            "orient": "split",
            "payload": payload,
        }

    def _ndarray_to_payload(a: np.ndarray) -> dict:
        return {
            "__type__": "numpy.ndarray",
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "data": a.tolist(),
        }

    def _sanitize(obj):
        """
        Recursively sanitize nested structures to be storage-friendly.
        """
        if isinstance(obj, pd.DataFrame):
            return _df_to_tagged_payload(obj)

        if isinstance(obj, pd.Series):
            return {
                "__type__": "pandas.Series",
                "name": None if obj.name is None else str(obj.name),
                "index": [str(x) for x in obj.index.astype(str).tolist()],
                "data": obj.tolist(),
            }

        if isinstance(obj, np.ndarray):
            return _ndarray_to_payload(obj)

        if isinstance(obj, (np.generic,)):
            try:
                return obj.item()
            except Exception:
                return str(obj)

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[str(k)] = _sanitize(v)
            return out

        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]

        if isinstance(obj, pd.Index):
            return obj.astype(str).tolist()

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        return str(obj)

    # ------------------------------------------------------------
    # Prepare output path
    # ------------------------------------------------------------
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Size warnings (before sanitization, warn-only)
    # ------------------------------------------------------------
    try:
        uns = adata.uns if hasattr(adata, "uns") else {}
        total_bytes = _estimate_size_bytes(uns)
        total_mb = total_bytes / (1024 ** 2)

        WARN_TOTAL_MB = 200.0
        WARN_SINGLE_MB = 50.0

        if total_mb >= WARN_TOTAL_MB:
            LOGGER.warning(
                "adata.uns is large (~%.1f MB). This is allowed, but save/load may be slow.",
                total_mb,
            )

        if isinstance(uns, dict):
            for k, v in uns.items():
                sz = _estimate_size_bytes(v) / (1024 ** 2)
                if sz >= WARN_SINGLE_MB:
                    LOGGER.warning(
                        "Large object in adata.uns['%s'] (~%.1f MB).",
                        k,
                        sz,
                    )
    except Exception as e:
        LOGGER.warning(
            "Could not estimate adata.uns size for warning purposes (continuing). (%s)",
            e,
        )

    # ------------------------------------------------------------
    # Work on a copy so we don't mutate the user's adata in memory
    # ------------------------------------------------------------
    adata_to_write = adata.copy()
    try:
        adata_to_write.uns = _sanitize(copy.deepcopy(adata_to_write.uns))
    except Exception as e:
        LOGGER.warning(
            "Failed to fully sanitize adata.uns; attempting best-effort write anyway. (%s)",
            e,
        )

    adata_to_write.var = _downgrade_nullable_strings(adata_to_write.var)
    adata_to_write.obs = _downgrade_nullable_strings(adata_to_write.obs)

    # ------------------------------------------------------------
    # Write dataset
    # ------------------------------------------------------------
    if fmt == "zarr":
        LOGGER.info(f"Saving dataset as Zarr → {out_path}")
        adata_to_write.write_zarr(str(out_path), chunks=None)

    elif fmt == "h5ad":
        LOGGER.info(f"Saving dataset as H5AD → {out_path}")
        adata_to_write.write_h5ad(str(out_path), compression="gzip")

    else:
        raise ValueError(f"Unknown dataset format '{fmt}'. Expected 'zarr' or 'h5ad'.")

    LOGGER.info(f"Finished writing dataset → {out_path}")


def load_dataset(path: Path) -> ad.AnnData:
    """
    Load a dataset from Zarr or H5AD into memory.

    Automatically rehydrates any tagged objects produced by save_dataset():
      - pandas.DataFrame
      - pandas.Series
      - numpy.ndarray
    """
    import numpy as np
    import pandas as pd

    def _rehydrate(obj):
        if isinstance(obj, dict) and "__type__" in obj:
            t = obj.get("__type__", None)

            if t == "pandas.DataFrame":
                orient = obj.get("orient", "split")
                payload = obj.get("payload", None)
                if orient == "split" and isinstance(payload, dict):
                    try:
                        df = pd.DataFrame(
                            data=payload.get("data", []),
                            index=payload.get("index", []),
                            columns=payload.get("columns", []),
                        )
                        df.index = df.index.astype(str)
                        df.columns = df.columns.astype(str)
                        return df
                    except Exception:
                        return obj  # fallback: leave tagged payload

            if t == "pandas.Series":
                try:
                    idx = obj.get("index", [])
                    data = obj.get("data", [])
                    name = obj.get("name", None)
                    return pd.Series(data=data, index=pd.Index(idx, dtype=str), name=name)
                except Exception:
                    return obj

            if t == "numpy.ndarray":
                try:
                    data = obj.get("data", [])
                    dtype = obj.get("dtype", None)
                    a = np.array(data, dtype=dtype)
                    # shape is best-effort; only reshape if consistent
                    shp = obj.get("shape", None)
                    if shp is not None:
                        try:
                            a = a.reshape(tuple(shp))
                        except Exception:
                            pass
                    return a
                except Exception:
                    return obj

        # normal recursion
        if isinstance(obj, dict):
            return {str(k): _rehydrate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_rehydrate(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_rehydrate(v) for v in obj)

        return obj

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".h5ad":
        LOGGER.info(f"Loading H5AD dataset → {path}")
        adata = ad.read_h5ad(str(path))
    elif path.suffix == ".zarr" or path.is_dir():
        LOGGER.info(f"Loading Zarr dataset → {path}")
        adata = ad.read_zarr(str(path))  # fully in-memory
    else:
        raise ValueError(f"Cannot load dataset from: {path}. Expected .zarr directory or .h5ad file.")

    # Rehydrate tagged structures in uns
    try:
        adata.uns = _rehydrate(dict(adata.uns))
    except Exception as e:
        LOGGER.warning("Failed to rehydrate tagged objects in adata.uns. (%s)", e)

    return adata


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
# Export cluster annotations
# =====================================================================
def export_cluster_annotations(adata: ad.AnnData, columns: List[str], out_path: Path) -> None:
    df = adata.obs[columns].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    LOGGER.info("Exported cluster annotations → %s", out_path)


# =====================================================================
# MSigDB (Decoupler) IO helpers
# =====================================================================

MSIGDB_BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
MSIGDB_INDEX_FILENAME = "msigdb_index.json"

def gmt_to_decoupler_net(gmt_path: str | Path) -> "pd.DataFrame":
    """
    Convert a GMT file into a decoupler net DataFrame.

    Uses decoupler.pp.read_gmt which returns a network DataFrame.
    Ensures columns: source, target, weight.
    """
    import pandas as pd
    import decoupler as dc

    net = dc.pp.read_gmt(str(gmt_path))
    if net is None or len(net) == 0:
        return pd.DataFrame(columns=["source", "target", "weight"])

    # decoupler returns at least source/target
    if "weight" not in net.columns:
        net = net.copy()
        net["weight"] = 1.0

    # standardize col order
    cols = [c for c in ["source", "target", "weight"] if c in net.columns]
    return net[cols].drop_duplicates().reset_index(drop=True)



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
    user_spec: "list[str] | str | None",
) -> tuple[list[Path], list[str], str | None]:
    """
    Resolve a user-provided MSigDB selector into concrete GMT file paths.

    Accepts:
      - None -> defaults to ["HALLMARK", "REACTOME"]
      - "HALLMARK,REACTOME" (comma-separated string)
      - ["HALLMARK", "REACTOME"]
      - explicit GMT paths mixed in

    Returns:
      (gmt_files, used_keywords, msigdb_release)

    msigdb_release is:
      - a string like "2025.1.Hs" if MSigDB keywords were used
      - None if ONLY explicit .gmt paths were provided
    """
    if user_spec is None:
        spec_items: list[str] = ["HALLMARK", "REACTOME"]
        LOGGER.info("MSigDB: no gene sets provided; defaulting to HALLMARK + REACTOME.")
    elif isinstance(user_spec, str):
        spec_items = [x.strip() for x in user_spec.split(",") if x.strip()]
    else:
        spec_items = [str(x).strip() for x in user_spec if str(x).strip()]

    if not spec_items:
        raise ValueError("Empty MSigDB gene-set specification.")

    release, index = _load_msigdb_index()
    LOGGER.info("Resolving MSigDB gene sets for release %s", release)

    gmt_files: list[Path] = []
    used_keywords: list[str] = []
    unresolved: list[str] = []
    used_any_msigdb_keyword = False

    for item in spec_items:
        # Custom GMT file path
        if item.lower().endswith(".gmt"):
            p = Path(item).expanduser().resolve()
            if not p.is_file():
                LOGGER.warning("Custom GMT file does not exist: %s (skipping)", p)
                unresolved.append(item)
                continue
            gmt_files.append(p)
            used_keywords.append(str(p))
            continue

        # MSigDB keyword
        key = item.upper()
        if key not in index:
            LOGGER.warning(
                "MSigDB: unknown keyword '%s'. Known examples: %s",
                key,
                ", ".join(sorted(index.keys())[:10]),
            )
            unresolved.append(item)
            continue

        used_any_msigdb_keyword = True
        gmt_files.append(Path(index[key]))
        used_keywords.append(key)

    # Deduplicate resolved GMTs
    uniq: dict[str, Path] = {}
    for p in gmt_files:
        uniq[str(p)] = p
    gmt_files_out = list(uniq.values())

    if not gmt_files_out:
        raise ValueError(
            f"No resolvable MSigDB gene sets from spec: {spec_items}. "
            f"Unresolved: {unresolved}"
        )

    LOGGER.info(
        "Resolved MSigDB gene sets: %s",
        ", ".join(used_keywords),
    )

    msigdb_release = release if used_any_msigdb_keyword else None
    return gmt_files_out, used_keywords, msigdb_release

