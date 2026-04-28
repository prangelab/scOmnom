from __future__ import annotations
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
import re
import json
import tarfile
import tempfile
import subprocess
import urllib.request
import urllib.error
import hashlib
from . import gene_utils


LOGGER = logging.getLogger(__name__)

_SIDECAR_REF_TYPE = "scomnom.ref.v1"
_SIDECAR_GROUP_ROOT = "__scomnom_payloads__"
_SIDECAR_GROUP_VERSION = "v1"

# Official CellTypist model registry
CELLTYPIST_REGISTRY_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"

# Local cache directory for downloaded models
CELLTYPIST_CACHE = Path.home() / ".cache" / "scomnom" / "celltypist_models"
CELLTYPIST_CACHE.mkdir(parents=True, exist_ok=True)

GENE_ANNOTATION_CACHE = Path.home() / ".cache" / "scomnom" / "gene_annotations"
GENE_ANNOTATION_CACHE.mkdir(parents=True, exist_ok=True)

_GENE_TYPE_COLS = (
    "gene_type",
    "gene_biotype",
    "biotype",
    "gene_type_ensembl",
)
_GENE_TYPE_MAP_CACHE: dict[tuple[str, str], dict[str, str]] = {}
_GENE_TYPE_SOURCE_CACHE: dict[tuple[str, str], str] = {}
_STANDARD_CHROMS = {*(str(i) for i in range(1, 23)), "X", "Y", "MT"}


def _add_gene_type_column(
    adata: ad.AnnData,
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    gene_type_col: str = "gene_type",
) -> pd.DataFrame:
    gene_map, source_label, chrom_map, chrom_source, gene_id_map = get_gene_type_map(
        adata,
        species="hsapiens",
        allow_fallback=True,
        force_download=False,
    )
    out = gene_utils.apply_gene_type_map(
        df,
        gene_map,
        gene_col=gene_col,
        gene_type_col=gene_type_col,
        gene_chrom_col="gene_chrom",
        chrom_map=chrom_map,
        gene_id_col="gene_id",
        gene_id_map=gene_id_map,
        add_source_cols=False,
        inplace=False,
    )
    for col in ("gene_type_source", "gene_chrom_source"):
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def _drop_redundant_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", False):
        return df
    drop_cols = [
        c
        for c in (
            "cluster",
            "group",
            "groupby",
            "contrast_key",
            "A",
            "B",
            "n_cells_A",
            "n_cells_B",
            "n_cells_A_used",
            "n_cells_B_used",
            "downsampled",
            "pct_nz_group",
            "pct_nz_reference",
            "key_added",
        )
        if c in df.columns
    ]
    if drop_cols:
        return df.drop(columns=drop_cols)
    return df


def _fallback_gene_type_map(genes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for g in genes:
        name = str(g)
        upper = name.upper()
        if upper.startswith("MT-"):
            out[name] = "mt_inferred"
        elif upper.startswith("LINC"):
            out[name] = "linc_inferred"
        elif upper.startswith("LOC"):
            out[name] = "loc_inferred"
        elif upper.startswith("MIR"):
            out[name] = "mir_inferred"
        elif upper.startswith("RNP"):
            out[name] = "rnp_inferred"
        elif upper.startswith("RNU"):
            out[name] = "rnu_inferred"
        elif upper.startswith(("SNORD", "SNORA", "SCARNA")):
            out[name] = "snorna_inferred"
        else:
            out[name] = "protein_coding"
    return out


def fetch_ensembl_gene_annotations(
    *,
    species: str = "hsapiens",
    force: bool = False,
) -> pd.DataFrame:
    species = str(species).strip().lower()
    dataset_name = f"{species}_gene_ensembl"
    cache_path = GENE_ANNOTATION_CACHE / f"{dataset_name}__gene_annotations.tsv.gz"

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, sep="\t")

    try:
        from pybiomart import Dataset
    except Exception as e:
        raise RuntimeError(f"pybiomart is required for Ensembl gene annotations: {e}") from e

    ds = Dataset(name=dataset_name, host="http://www.ensembl.org")
    df = ds.query(attributes=["external_gene_name", "gene_biotype", "chromosome_name", "ensembl_gene_id"])
    if df is None or df.empty:
        raise RuntimeError(f"BioMart query returned no results for dataset={dataset_name!r}")

    col_lc = {str(c).strip().lower(): c for c in df.columns}
    df = df.rename(
        columns={
            col_lc.get("external_gene_name", "external_gene_name"): "gene",
            col_lc.get("gene name", "gene name"): "gene",
            col_lc.get("gene_biotype", "gene_biotype"): "gene_type",
            col_lc.get("gene type", "gene type"): "gene_type",
            col_lc.get("chromosome_name", "chromosome_name"): "gene_chrom",
            col_lc.get("chromosome/scaffold name", "chromosome/scaffold name"): "gene_chrom",
            col_lc.get("ensembl_gene_id", "ensembl_gene_id"): "gene_id",
            col_lc.get("gene stable id", "gene stable id"): "gene_id",
        }
    )
    if "gene" not in df.columns or "gene_type" not in df.columns or "gene_chrom" not in df.columns:
        raise RuntimeError(
            f"BioMart columns not found after rename. Available columns: {list(df.columns)}"
        )
    if "gene_id" not in df.columns:
        raise RuntimeError(
            f"BioMart gene_id column not found after rename. Available columns: {list(df.columns)}"
        )
    df["gene"] = df["gene"].astype(str)
    df["gene_type"] = df["gene_type"].astype(str)
    df["gene_chrom"] = df["gene_chrom"].astype(str)
    df["gene_id"] = df["gene_id"].astype(str)

    chrom = df["gene_chrom"].astype(str).str.strip()
    chrom = chrom.str.replace(r"^chr", "", case=False, regex=True)
    is_standard = chrom.isin(_STANDARD_CHROMS)
    chrom = chrom.where(is_standard, "")
    df["gene_chrom"] = chrom
    df["__chrom_standard"] = is_standard

    df = df[df["gene"].astype(bool)]
    df = df.sort_values(["gene", "__chrom_standard"], ascending=[True, False])
    df = df.drop_duplicates(subset=["gene"], keep="first").drop(columns=["__chrom_standard"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, sep="\t", index=False, compression="gzip")
    return df


def get_gene_type_map(
    adata: ad.AnnData,
    *,
    species: str = "hsapiens",
    allow_fallback: bool = True,
    force_download: bool = False,
) -> tuple[dict[str, str], str, dict[str, str], str, dict[str, str]]:
    if adata is None:
        return {}, "unknown", {}, "unknown", {}

    raw_series = None
    for col in _GENE_TYPE_COLS:
        if col in adata.var.columns:
            raw_series = adata.var[col]
            break

    if raw_series is not None:
        gmap = raw_series.astype(str).to_dict()
        chrom_map = {}
        gene_id_map = {}
        if "chromosome" in adata.var.columns:
            chrom_map = adata.var["chromosome"].astype(str).to_dict()
        elif "chrom" in adata.var.columns:
            chrom_map = adata.var["chrom"].astype(str).to_dict()
        if "gene_id" in adata.var.columns:
            gene_id_map = adata.var["gene_id"].astype(str).to_dict()
        elif "ensembl_gene_id" in adata.var.columns:
            gene_id_map = adata.var["ensembl_gene_id"].astype(str).to_dict()
        return gmap, f"adata:{str(raw_series.name)}", chrom_map, "adata", gene_id_map

    cache_key = ("biomart", str(species))
    if cache_key in _GENE_TYPE_MAP_CACHE and not force_download:
        chrom_key = ("biomart_chrom", str(species))
        id_key = ("biomart_gene_id", str(species))
        chrom_map = _GENE_TYPE_MAP_CACHE.get(chrom_key, {})
        gene_id_map = _GENE_TYPE_MAP_CACHE.get(id_key, {})
        return (
            _GENE_TYPE_MAP_CACHE[cache_key],
            _GENE_TYPE_SOURCE_CACHE.get(cache_key, "biomart"),
            chrom_map,
            "biomart",
            gene_id_map,
        )

    try:
        df = fetch_ensembl_gene_annotations(species=species, force=force_download)
        gmap = dict(zip(df["gene"].astype(str), df["gene_type"].astype(str)))
        chrom_map = dict(zip(df["gene"].astype(str), df["gene_chrom"].astype(str)))
        gene_id_map = dict(zip(df["gene"].astype(str), df["gene_id"].astype(str)))
        _GENE_TYPE_MAP_CACHE[cache_key] = gmap
        _GENE_TYPE_SOURCE_CACHE[cache_key] = "biomart"
        _GENE_TYPE_MAP_CACHE[("biomart_chrom", str(species))] = chrom_map
        _GENE_TYPE_MAP_CACHE[("biomart_gene_id", str(species))] = gene_id_map
        return gmap, "biomart", chrom_map, "biomart", gene_id_map
    except Exception as e:
        if not allow_fallback:
            raise
        LOGGER.warning(
            "Gene annotation lookup failed for species=%r (%s). Falling back to simple pattern rules.",
            str(species),
            str(e),
        )

    cache_key = ("fallback", str(species))
    if cache_key in _GENE_TYPE_MAP_CACHE:
        chrom_key = ("fallback_chrom", str(species))
        chrom_map = _GENE_TYPE_MAP_CACHE.get(chrom_key, {})
        id_key = ("fallback_gene_id", str(species))
        gene_id_map = _GENE_TYPE_MAP_CACHE.get(id_key, {})
        return (
            _GENE_TYPE_MAP_CACHE[cache_key],
            _GENE_TYPE_SOURCE_CACHE.get(cache_key, "fallback"),
            chrom_map,
            "fallback",
            gene_id_map,
        )

    genes = list(adata.var_names.astype(str))
    gmap = _fallback_gene_type_map(genes)
    _GENE_TYPE_MAP_CACHE[cache_key] = gmap
    _GENE_TYPE_SOURCE_CACHE[cache_key] = "fallback"
    chrom_map = {}
    gene_id_map = {}
    _GENE_TYPE_MAP_CACHE[("fallback_chrom", str(species))] = chrom_map
    _GENE_TYPE_MAP_CACHE[("fallback_gene_id", str(species))] = gene_id_map
    return gmap, "fallback", chrom_map, "fallback", gene_id_map


def download_gene_models(*, species: str = "hsapiens") -> None:
    fetch_ensembl_gene_annotations(species=species, force=True)


def sanitize_identifier(
    x: Any,
    *,
    max_len: int = 180,
    allow_spaces: bool = True,
) -> str:
    """
    Shared sanitizer for:
      - filenames (cross-platform safe; no ':', '/', '\\')
      - Zarr group keys (no '/' or '\\')
      - general identifiers used in output artifacts

    Behavior:
      - replaces '/', '\\', ':' with '_'
      - normalizes whitespace
      - keeps only [A-Za-z0-9 ._-] (or [A-Za-z0-9._-] if allow_spaces=False)
      - collapses repeated underscores
      - strips leading/trailing punctuation
      - truncates to max_len

    Returns a non-empty string.
    """
    s = str(x)

    # hard disallow
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")

    # whitespace normalization
    s = re.sub(r"\s+", " ", s).strip()

    # conservative character set
    if allow_spaces:
        s = re.sub(r"[^A-Za-z0-9 ._-]+", "_", s)
    else:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    s = re.sub(r"_+", "_", s).strip("._-")

    if not s:
        s = "x"

    if len(s) > max_len:
        s = s[:max_len].rstrip("._-") or "x"

    return s


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
    from . import plot_utils

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

        plot_utils.record_plot_artifact(stem=stem, figdir=figdir)

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

    raw_glob = str(cfg.raw_sample_dir / cfg.raw_pattern)
    LOGGER.info(
        "attach_raw_counts_postfilter: cwd=%s raw_sample_dir=%r raw_pattern=%r raw_glob=%s",
        Path.cwd(),
        cfg.raw_sample_dir,
        cfg.raw_pattern,
        raw_glob,
    )
    raw_dirs = find_raw_dirs(cfg.raw_sample_dir, cfg.raw_pattern)
    LOGGER.info(
        "attach_raw_counts_postfilter: found %d raw dirs",
        len(raw_dirs),
    )
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
_ZARR_ARCHIVE_EXT = ".zarr.tar.zst"


def _is_zarr_archive_path(path: Path) -> bool:
    return str(path).endswith(_ZARR_ARCHIVE_EXT)


def _default_zarr_archive_path(path: Path) -> Path:
    p = Path(path)
    if _is_zarr_archive_path(p):
        return p
    if p.suffix == ".zarr" or p.name.endswith(".zarr"):
        return p.with_name(f"{p.name}.tar.zst")
    return p.with_name(f"{p.name}{_ZARR_ARCHIVE_EXT}")


def _archive_to_zarr_dir_name(archive_path: Path) -> str:
    return archive_path.name[: -len(".tar.zst")]


def _validate_tar_members_safe(member_names: List[str]) -> None:
    for name in member_names:
        if not name:
            continue
        if name.startswith("/"):
            raise RuntimeError(f"Refusing to extract archive with absolute member path: {name}")
        pure = Path(name)
        if ".." in pure.parts:
            raise RuntimeError(f"Refusing to extract archive with parent traversal member path: {name}")


def _use_system_tar_zstd() -> bool:
    return shutil.which("tar") is not None and shutil.which("zstd") is not None


def _zstd_threads_flag() -> str:
    return os.getenv("SCOMNOM_ZSTD_THREADS", "0")


def _zstd_level() -> str:
    return os.getenv("SCOMNOM_ZSTD_LEVEL", "19")


def _tar_create_zst(src_root: Path, item_name: str, out_archive: Path) -> None:
    if _use_system_tar_zstd():
        tar_cmd = ["tar", "-cf", "-", "-C", str(src_root), item_name]
        zstd_cmd = ["zstd", f"-{_zstd_level()}", f"-T{_zstd_threads_flag()}", "-o", str(out_archive)]
        with subprocess.Popen(tar_cmd, stdout=subprocess.PIPE) as tar_proc:
            assert tar_proc.stdout is not None
            zstd_res = subprocess.run(zstd_cmd, stdin=tar_proc.stdout, check=False, capture_output=True, text=True)
            tar_proc.stdout.close()
            tar_ret = tar_proc.wait()
        if tar_ret != 0:
            raise subprocess.CalledProcessError(tar_ret, tar_cmd)
        if zstd_res.returncode != 0:
            raise subprocess.CalledProcessError(
                zstd_res.returncode, zstd_cmd, output=zstd_res.stdout, stderr=zstd_res.stderr
            )
        return

    try:
        import zstandard as zstd
    except Exception as e:
        raise RuntimeError("zstandard is required for writing .zarr.tar.zst archives.") from e

    with out_archive.open("wb") as fh:
        cctx = zstd.ZstdCompressor(level=int(_zstd_level()), threads=max(1, (os.cpu_count() or 1)))
        with cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tf:
                tf.add(src_root / item_name, arcname=item_name)


def _tar_list_zst(archive_path: Path) -> List[str]:
    if _use_system_tar_zstd():
        zstd_cmd = ["zstd", "-d", "-c", str(archive_path)]
        tar_cmd = ["tar", "-tf", "-"]
        with subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as zstd_proc:
            assert zstd_proc.stdout is not None
            out = subprocess.run(tar_cmd, stdin=zstd_proc.stdout, check=False, capture_output=True, text=True)
            zstd_proc.stdout.close()
            zstd_stderr = zstd_proc.stderr.read().decode("utf-8", errors="replace") if zstd_proc.stderr else ""
            zstd_ret = zstd_proc.wait()
        if zstd_ret != 0:
            raise subprocess.CalledProcessError(zstd_ret, zstd_cmd, stderr=zstd_stderr)
        if out.returncode != 0:
            raise subprocess.CalledProcessError(out.returncode, tar_cmd, output=out.stdout, stderr=out.stderr)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    try:
        import zstandard as zstd
    except Exception as e:
        raise RuntimeError("zstandard is required for reading .zarr.tar.zst archives.") from e

    names: List[str] = []
    with archive_path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    names.append(member.name)
    return names


def _tar_extract_zst(archive_path: Path, out_dir: Path) -> None:
    if _use_system_tar_zstd():
        zstd_cmd = ["zstd", "-d", "-c", str(archive_path)]
        tar_cmd = ["tar", "-xf", "-", "-C", str(out_dir)]
        with subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as zstd_proc:
            assert zstd_proc.stdout is not None
            tar_res = subprocess.run(tar_cmd, stdin=zstd_proc.stdout, check=False, capture_output=True, text=True)
            zstd_proc.stdout.close()
            zstd_stderr = zstd_proc.stderr.read().decode("utf-8", errors="replace") if zstd_proc.stderr else ""
            zstd_ret = zstd_proc.wait()
        if zstd_ret != 0:
            raise subprocess.CalledProcessError(zstd_ret, zstd_cmd, stderr=zstd_stderr)
        if tar_res.returncode != 0:
            raise subprocess.CalledProcessError(
                tar_res.returncode, tar_cmd, output=tar_res.stdout, stderr=tar_res.stderr
            )
        return

    try:
        import zstandard as zstd
    except Exception as e:
        raise RuntimeError("zstandard is required for reading .zarr.tar.zst archives.") from e

    with archive_path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    _validate_tar_members_safe([member.name])
                    tf.extract(member, path=out_dir)


def save_dataset(adata: ad.AnnData, out_path: Path, fmt: str = "zarr", archive: bool = True) -> None:
    """
    Save an AnnData object either as .zarr/.zarr.tar.zst or .h5ad.

    Makes adata.uns safe for H5AD/Zarr by:
      - tagging pandas.DataFrame / Series
      - converting numpy arrays + scalars
      - sanitizing ALL dict keys using sanitize_identifier()
      - stabilizing key collisions with a hash suffix
    """
    import sys
    import numpy as np
    import pandas as pd
    import zarr
    import resource

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

    def _log_object_dtype_columns(frame: pd.DataFrame, *, label: str) -> None:
        try:
            if frame is None or frame.empty:
                return
            obj_cols = [str(c) for c in frame.columns if str(frame[c].dtype) == "object"]
            if obj_cols:
                LOGGER.error("save_dataset: detected object dtype columns in %s: %s", label, obj_cols[:20])
        except Exception:
            pass

    def _read_int_file(path: Path) -> int | None:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _format_gib(value_bytes: int | None) -> str:
        if value_bytes is None or value_bytes < 0:
            return "NA"
        return f"{(float(value_bytes) / (1024.0 ** 3)):.2f}"

    def _log_mem_checkpoint(stage: str) -> None:
        rss_gib = "NA"
        try:
            rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_bytes = rss_kib * 1024
            rss_gib = _format_gib(rss_bytes)
        except Exception:
            pass

        cgroup_current = _read_int_file(Path("/sys/fs/cgroup/memory.current"))
        cgroup_max = _read_int_file(Path("/sys/fs/cgroup/memory.max"))
        if cgroup_current is None:
            cgroup_current = _read_int_file(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        if cgroup_max is None:
            cgroup_max = _read_int_file(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        if cgroup_max is not None and cgroup_max >= (2**60):
            cgroup_max = None

        LOGGER.info(
            "save_dataset[%s]: rss_max=%s GiB cgroup_current=%s GiB cgroup_limit=%s GiB",
            str(stage),
            str(rss_gib),
            _format_gib(cgroup_current),
            _format_gib(cgroup_max),
        )

    def _coerce_object_columns_inplace_for_zarr(
        df: pd.DataFrame, *, label: str
    ) -> list[tuple[str, pd.Series]]:
        changed: list[tuple[str, pd.Series]] = []
        if df is None or df.empty:
            return changed
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        for col in obj_cols:
            original = df[col]
            s = original
            vals = s.dropna()
            if (not vals.empty) and (not vals.map(lambda x: isinstance(x, str)).all()):
                LOGGER.warning(
                    "save_dataset: coercing mixed object column in %s['%s'] to string categorical for zarr compatibility",
                    label,
                    col,
                )
                coerced = s.map(lambda v: "" if pd.isna(v) else str(v))
            else:
                coerced = s.astype("string", copy=False).fillna("")
            changed.append((str(col), original))
            df[col] = pd.Categorical(coerced.astype(str))
        return changed

    # ------------------------------------------------------------
    # Key sanitization + collision handling
    # ------------------------------------------------------------
    _keymap: dict[str, str] = {}
    _seen_keys: set[str] = set()
    _seen_sidecar_ids: set[str] = set()
    _sidecar_payloads: list[dict[str, object]] = []

    sidecar_enabled = fmt == "zarr"

    def _dedupe_key(safe: str, original: str) -> str:
        if safe not in _seen_keys:
            _seen_keys.add(safe)
            _keymap[safe] = original
            return safe

        if _keymap.get(safe) == original:
            return safe

        h = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        alt = f"{safe}__{h}"

        if alt in _seen_keys:
            h2 = hashlib.sha1((original + "|2").encode("utf-8")).hexdigest()[:8]
            alt = f"{safe}__{h2}"

        _seen_keys.add(alt)
        _keymap[alt] = original
        return alt

    def _sidecar_payload_id(path_tokens: tuple[str, ...]) -> str:
        base = "__".join(
            sanitize_identifier(str(tok), max_len=48, allow_spaces=False)
            for tok in path_tokens
            if str(tok).strip()
        )
        if not base:
            base = "payload"
        base = base[:180]
        raw = "/".join(path_tokens)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        pid = f"{base}__{digest}"
        if pid not in _seen_sidecar_ids:
            _seen_sidecar_ids.add(pid)
            return pid
        i = 2
        while f"{pid}__{i}" in _seen_sidecar_ids:
            i += 1
        deduped = f"{pid}__{i}"
        _seen_sidecar_ids.add(deduped)
        return deduped

    def _register_sidecar_payload(kind: str, obj, path_tokens: tuple[str, ...]) -> dict[str, object]:
        pid = _sidecar_payload_id(path_tokens)
        _sidecar_payloads.append(
            {
                "id": pid,
                "kind": str(kind),
                "obj": obj,
                "path_tokens": path_tokens,
            }
        )
        meta: dict[str, object] = {}
        if isinstance(obj, pd.DataFrame):
            meta["shape"] = [int(obj.shape[0]), int(obj.shape[1])]
            try:
                meta["nbytes_est"] = int(obj.memory_usage(deep=True).sum())
            except Exception:
                pass
        elif isinstance(obj, pd.Series):
            meta["shape"] = [int(obj.shape[0])]
            meta["dtype"] = str(obj.dtype)
            try:
                meta["nbytes_est"] = int(obj.memory_usage(deep=True))
            except Exception:
                pass
        elif isinstance(obj, np.ndarray):
            meta["shape"] = [int(x) for x in obj.shape]
            meta["dtype"] = str(obj.dtype)
            meta["nbytes_est"] = int(obj.nbytes)
        return {
            "__type__": _SIDECAR_REF_TYPE,
            "kind": str(kind),
            "path": f"{_SIDECAR_GROUP_ROOT}/{_SIDECAR_GROUP_VERSION}/{pid}",
            "meta": meta,
        }

    def _write_sidecar_payloads(zarr_dir: Path) -> None:
        if not _sidecar_payloads:
            return
        root = zarr.open_group(str(zarr_dir), mode="a", zarr_format=2)
        side_root = root.require_group(_SIDECAR_GROUP_ROOT).require_group(_SIDECAR_GROUP_VERSION)
        manifest_entries: list[dict[str, object]] = []
        
        def _to_unicode_array(values) -> np.ndarray:
            vals = [str(v) for v in values]
            max_len = max((len(v) for v in vals), default=1)
            return np.asarray(vals, dtype=f"<U{max(1, max_len)}")

        def _to_unicode_matrix(values_2d) -> np.ndarray:
            rows = [[str(v) for v in row] for row in values_2d]
            max_len = max((len(v) for row in rows for v in row), default=1)
            return np.asarray(rows, dtype=f"<U{max(1, max_len)}")

        def _to_unicode_ndarray(values_nd) -> np.ndarray:
            arr = np.asarray(values_nd, dtype=object)
            flat = [str(v) for v in arr.reshape(-1).tolist()]
            max_len = max((len(v) for v in flat), default=1)
            return np.asarray(flat, dtype=f"<U{max(1, max_len)}").reshape(arr.shape)

        def _dtype_has_object(dt: np.dtype) -> bool:
            if dt == object:
                return True
            if dt.fields:
                for _, field_info in dt.fields.items():
                    field_dt = np.dtype(field_info[0])
                    if _dtype_has_object(field_dt):
                        return True
            return False

        def _create_array_safe(grp, name: str, data) -> None:
            arr = np.asarray(data)
            if _dtype_has_object(arr.dtype):
                arr = _to_unicode_ndarray(arr)
            grp.create_array(name, data=arr, overwrite=True)

        for payload in _sidecar_payloads:
            pid = str(payload["id"])
            kind = str(payload["kind"])
            obj = payload["obj"]
            LOGGER.info("save_dataset: writing sidecar payload id=%s kind=%s", pid, kind)

            if pid in side_root:
                del side_root[pid]
            grp = side_root.create_group(pid)
            grp.attrs["kind"] = kind

            try:
                if kind == "dataframe":
                    df = obj
                    data = np.asarray(df.to_numpy(copy=False))
                    data_cast = "native"
                    if data.dtype == object:
                        data = _to_unicode_matrix(df.astype(str).to_numpy(copy=False))
                        data_cast = "str"
                    _create_array_safe(grp, "data", data)
                    idx = _to_unicode_array(df.index.astype(str).tolist())
                    cols = _to_unicode_array(df.columns.astype(str).tolist())
                    _create_array_safe(grp, "index", idx)
                    _create_array_safe(grp, "columns", cols)
                    grp.attrs["dtypes"] = json.dumps([str(x) for x in df.dtypes], ensure_ascii=True)
                    grp.attrs["data_cast"] = data_cast
                    manifest_entries.append({"id": pid, "kind": kind, "shape": [int(df.shape[0]), int(df.shape[1])]})
                elif kind == "series":
                    series = obj
                    data = np.asarray(series.to_numpy(copy=False))
                    data_cast = "native"
                    if data.dtype == object:
                        data = _to_unicode_array(series.astype(str).tolist())
                        data_cast = "str"
                    _create_array_safe(grp, "data", data)
                    idx = _to_unicode_array(series.index.astype(str).tolist())
                    _create_array_safe(grp, "index", idx)
                    grp.attrs["name"] = "" if series.name is None else str(series.name)
                    grp.attrs["dtype"] = str(series.dtype)
                    grp.attrs["data_cast"] = data_cast
                    manifest_entries.append({"id": pid, "kind": kind, "shape": [int(series.shape[0])]})
                elif kind == "ndarray":
                    arr = np.asarray(obj)
                    if arr.dtype == object:
                        arr = _to_unicode_ndarray(arr)
                    _create_array_safe(grp, "data", arr)
                    grp.attrs["dtype"] = str(arr.dtype)
                    manifest_entries.append({"id": pid, "kind": kind, "shape": [int(x) for x in arr.shape]})
                else:
                    raise RuntimeError(f"Unknown sidecar payload kind: {kind!r}")
            except Exception as e:
                LOGGER.warning(
                    "save_dataset: sidecar payload failed (id=%s kind=%s); skipping payload. (%s)",
                    pid,
                    kind,
                    e,
                )
                if pid in side_root:
                    del side_root[pid]
                continue

        if "__manifest__" in side_root:
            del side_root["__manifest__"]
        manifest_grp = side_root.create_group("__manifest__")
        manifest_grp.attrs["schema_version"] = "1"
        manifest_grp.attrs["created_by"] = "save_dataset"
        manifest_grp.attrs["payload_count"] = int(len(manifest_entries))
        manifest_grp.attrs["entries_json"] = json.dumps(manifest_entries, ensure_ascii=True)

    # ------------------------------------------------------------
    # Payload tagging
    # ------------------------------------------------------------
    def _df_to_tagged_payload(df: pd.DataFrame) -> dict:
        payload = df.to_dict(orient="split")
        return {"__type__": "pandas.DataFrame", "orient": "split", "payload": payload}

    def _ndarray_to_payload(a: np.ndarray) -> dict:
        return {
            "__type__": "numpy.ndarray",
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "data": a.tolist(),
        }

    def _sanitize(obj, path_tokens: tuple[str, ...] = ()):
        def _has_legacy_type_marker(d: dict) -> bool:
            for k, v in d.items():
                ks = str(k)
                if ks == "type" or ks.startswith("type__") or ks.startswith("type_"):
                    if isinstance(v, str) and "." in v:
                        return True
            return False

        if sidecar_enabled and isinstance(obj, pd.DataFrame):
            return _register_sidecar_payload("dataframe", obj, path_tokens)
        if isinstance(obj, pd.DataFrame):
            return _df_to_tagged_payload(obj)

        if sidecar_enabled and isinstance(obj, pd.Series):
            return _register_sidecar_payload("series", obj, path_tokens)
        if isinstance(obj, pd.Series):
            return {
                "__type__": "pandas.Series",
                "name": None if obj.name is None else str(obj.name),
                "index": [str(x) for x in obj.index.astype(str).tolist()],
                "data": obj.tolist(),
            }

        if sidecar_enabled and isinstance(obj, np.ndarray):
            return _register_sidecar_payload("ndarray", obj, path_tokens)
        if isinstance(obj, np.ndarray):
            return _ndarray_to_payload(obj)

        if isinstance(obj, (np.generic,)):
            try:
                return obj.item()
            except Exception:
                return str(obj)

        if isinstance(obj, dict):
            if "__type__" in obj or _has_legacy_type_marker(obj):
                # Preserve internal tag keys so load_dataset() can rehydrate payloads.
                return {str(k): _sanitize(v, path_tokens + (str(k),)) for k, v in obj.items()}
            out: dict[str, object] = {}
            for k, v in obj.items():
                orig = str(k)
                safe = sanitize_identifier(orig, max_len=180, allow_spaces=True)
                safe = _dedupe_key(safe, orig)
                out[safe] = _sanitize(v, path_tokens + (orig,))
            return out

        if isinstance(obj, (list, tuple)):
            return [_sanitize(v, path_tokens + (str(i),)) for i, v in enumerate(obj)]

        if isinstance(obj, pd.Index):
            return obj.astype(str).tolist()

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        return str(obj)

    # ------------------------------------------------------------
    # Prepare output path
    # ------------------------------------------------------------
    out_path = Path(out_path)
    if fmt == "zarr":
        zarr_archive_infix = f"{_ZARR_ARCHIVE_EXT}."
        out_name = out_path.name
        if zarr_archive_infix in out_name:
            while zarr_archive_infix in out_name:
                out_name = out_name.replace(zarr_archive_infix, ".")
            out_path = out_path.with_name(out_name)
            LOGGER.warning("save_dataset: normalized embedded archive suffix in output name → %s", out_path)
    if fmt == "zarr" and not archive and _is_zarr_archive_path(out_path):
        out_path = Path(str(out_path)[: -len(".tar.zst")])
    if fmt in ("zarr", "h5ad"):
        desired = f".{fmt}"
        dup = desired + desired
        name = out_path.name
        if name.endswith(dup):
            while name.endswith(dup):
                name = name[: -len(desired)]
            out_path = out_path.with_name(name)
            LOGGER.warning("save_dataset: normalized duplicate extension for %s", out_path)
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

        _ = WARN_SINGLE_MB
    except Exception as e:
        LOGGER.warning(
            "Could not estimate adata.uns size for warning purposes (continuing). (%s)",
            e,
        )

    # ------------------------------------------------------------
    # Prepare lightweight metadata replacements for write-out only.
    # Avoid adata.copy() here: duplicating X/layers/obsm can OOM on large runs.
    # ------------------------------------------------------------
    orig_uns = adata.uns
    orig_obs = adata.obs
    orig_var = adata.var
    restored_obs_cols: list[tuple[str, pd.Series]] = []
    restored_var_cols: list[tuple[str, pd.Series]] = []
    sanitized_uns = orig_uns
    downgraded_obs = orig_obs
    downgraded_var = orig_var
    try:
        _log_mem_checkpoint("pre_sanitize")
        sanitized_uns = _sanitize(orig_uns)
        _log_mem_checkpoint("post_sanitize")
        if _keymap:
            sanitized_uns["__scomnom_keymap__"] = {
                "__type__": "scomnom.keymap.v1",
                "safe_to_original": _keymap,
            }
    except Exception as e:
        LOGGER.warning(
            "Failed to fully sanitize adata.uns; attempting best-effort write anyway. (%s)",
            e,
        )
        sanitized_uns = orig_uns

    if fmt == "h5ad":
        try:
            downgraded_var = _downgrade_nullable_strings(orig_var)
        except Exception as e:
            LOGGER.warning("Failed to downgrade nullable string columns in adata.var; writing original var. (%s)", e)
            downgraded_var = orig_var

        try:
            downgraded_obs = _downgrade_nullable_strings(orig_obs)
        except Exception as e:
            LOGGER.warning("Failed to downgrade nullable string columns in adata.obs; writing original obs. (%s)", e)
            downgraded_obs = orig_obs

    if fmt == "zarr":
        try:
            restored_obs_cols = _coerce_object_columns_inplace_for_zarr(orig_obs, label="adata.obs")
            downgraded_obs = orig_obs
        except Exception as e:
            LOGGER.warning("Failed to coerce object columns in adata.obs for zarr; writing original obs. (%s)", e)
            downgraded_obs = orig_obs
        try:
            restored_var_cols = _coerce_object_columns_inplace_for_zarr(orig_var, label="adata.var")
            downgraded_var = orig_var
        except Exception as e:
            LOGGER.warning("Failed to coerce object columns in adata.var for zarr; writing original var. (%s)", e)
            downgraded_var = orig_var

    adata.uns = sanitized_uns
    adata.var = downgraded_var
    adata.obs = downgraded_obs

    # ------------------------------------------------------------
    # Write dataset
    # ------------------------------------------------------------
    try:
        _log_mem_checkpoint("pre_write_dispatch")
        if fmt == "zarr":
            if archive:
                archive_path = _default_zarr_archive_path(out_path)
                zarr_dir_name = _archive_to_zarr_dir_name(archive_path)
                tmp_root = Path(tempfile.mkdtemp(prefix="scomnom_save_", dir=str(archive_path.parent)))
                tmp_archive = archive_path.parent / f".{archive_path.name}.{os.getpid()}.tmp"
                try:
                    tmp_zarr_dir = tmp_root / zarr_dir_name
                    LOGGER.debug("save_dataset: staging zarr write at %s", tmp_zarr_dir)
                    _log_mem_checkpoint("pre_write_zarr_stage")
                    adata.write_zarr(str(tmp_zarr_dir), chunks=None)
                    _log_mem_checkpoint("post_write_zarr_stage")
                    if sidecar_enabled and _sidecar_payloads:
                        _log_mem_checkpoint("pre_write_sidecar_stage")
                        try:
                            _write_sidecar_payloads(tmp_zarr_dir)
                        except Exception as e:
                            LOGGER.warning("save_dataset: sidecar stage failed; continuing without sidecar payloads. (%s)", e)
                        _log_mem_checkpoint("post_write_sidecar_stage")

                    LOGGER.debug("save_dataset: archiving staged zarr to %s", archive_path)
                    _log_mem_checkpoint("pre_archive")
                    _tar_create_zst(tmp_root, zarr_dir_name, tmp_archive)
                    _log_mem_checkpoint("post_archive")
                    tmp_archive.replace(archive_path)
                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)
                    if tmp_archive.exists():
                        tmp_archive.unlink(missing_ok=True)
            else:
                LOGGER.debug("save_dataset: writing zarr directory %s", out_path)
                _log_mem_checkpoint("pre_write_zarr")
                adata.write_zarr(str(out_path), chunks=None)
                _log_mem_checkpoint("post_write_zarr")
                if sidecar_enabled and _sidecar_payloads:
                    _log_mem_checkpoint("pre_write_sidecar")
                    try:
                        _write_sidecar_payloads(out_path)
                    except Exception as e:
                        LOGGER.warning("save_dataset: sidecar stage failed; continuing without sidecar payloads. (%s)", e)
                    _log_mem_checkpoint("post_write_sidecar")
        elif fmt == "h5ad":
            LOGGER.debug("save_dataset: writing h5ad %s", out_path)
            _log_mem_checkpoint("pre_write_h5ad")
            adata.write_h5ad(str(out_path), compression="gzip")
            _log_mem_checkpoint("post_write_h5ad")
        else:
            raise ValueError(f"Unknown dataset format '{fmt}'. Expected 'zarr' or 'h5ad'.")
    except Exception:
        _log_object_dtype_columns(adata.obs, label="adata.obs")
        _log_object_dtype_columns(adata.var, label="adata.var")
        raise
    finally:
        adata.uns = orig_uns
        adata.var = orig_var
        adata.obs = orig_obs
        for col, series in restored_obs_cols:
            try:
                adata.obs[col] = series
            except Exception:
                pass
        for col, series in restored_var_cols:
            try:
                adata.var[col] = series
            except Exception:
                pass

    # Intentionally silent at INFO level to avoid duplicate save-path chatter.



def load_dataset(path: Path) -> ad.AnnData:
    """
    Load a dataset from Zarr or H5AD into memory.

    Automatically rehydrates any tagged objects produced by save_dataset():
      - pandas.DataFrame
      - pandas.Series
      - numpy.ndarray
    """
    import re
    import numpy as np
    import pandas as pd
    import zarr

    zarr_source_dir: Path | None = None
    zarr_root_cache = None

    def _load_sidecar_ref(ref_obj: dict):
        nonlocal zarr_root_cache
        if zarr_source_dir is None:
            raise RuntimeError(
                "Encountered sidecar reference in adata.uns, but dataset was not loaded from a Zarr store."
            )
        if zarr_root_cache is None:
            zarr_root_cache = zarr.open_group(
                str(zarr_source_dir),
                mode="r",
                zarr_format=2,
                use_consolidated=False,
            )
        rel_path = str(ref_obj.get("path", "")).strip()
        if not rel_path:
            raise RuntimeError("Sidecar reference is missing required 'path'.")
        node = zarr_root_cache
        for token in [x for x in rel_path.split("/") if x]:
            try:
                node = node[token]
            except Exception:
                raise RuntimeError(f"Sidecar payload not found at path={rel_path!r}")

        kind = str(ref_obj.get("kind", "")).strip() or str(node.attrs.get("kind", "")).strip()
        if kind == "dataframe":
            data = np.asarray(node["data"])
            index = np.asarray(node["index"]).astype(str)
            columns = np.asarray(node["columns"]).astype(str)
            df = pd.DataFrame(data=data, index=pd.Index(index, dtype=str), columns=pd.Index(columns, dtype=str))
            dtypes_raw = node.attrs.get("dtypes", None)
            if dtypes_raw:
                try:
                    dtypes_list = json.loads(dtypes_raw)
                    if isinstance(dtypes_list, list) and len(dtypes_list) == len(df.columns):
                        for col, dt in zip(df.columns, dtypes_list):
                            try:
                                df[col] = df[col].astype(str(dt))
                            except Exception:
                                continue
                except Exception:
                    pass
            return df
        if kind == "series":
            data = np.asarray(node["data"])
            index = np.asarray(node["index"]).astype(str)
            name = node.attrs.get("name", "")
            name = None if str(name) == "" else str(name)
            s = pd.Series(data=data, index=pd.Index(index, dtype=str), name=name)
            dtype_raw = node.attrs.get("dtype", None)
            if dtype_raw:
                try:
                    s = s.astype(str(dtype_raw))
                except Exception:
                    pass
            return s
        if kind == "ndarray":
            return np.asarray(node["data"])
        raise RuntimeError(f"Unsupported sidecar payload kind={kind!r} at path={rel_path!r}")

    def _rehydrate(obj):
        def _recover_stringified_legacy_array(s: str):
            if "numpy.ndarray" not in s:
                return None
            # Legacy payloads can be stringified; recover color palettes encoded as hex values.
            hex_colors = re.findall(r"#[0-9A-Fa-f]{6}", s)
            if hex_colors:
                uniq = list(dict.fromkeys(hex_colors))
                return np.array(uniq, dtype=str)
            return None

        def _tag_key_and_type(d: dict) -> tuple[str | None, str | None]:
            if "__type__" in d:
                tval = d.get("__type__", None)
                return "__type__", (str(tval) if tval is not None else None)
            # Backward-compatible recovery for legacy keys that were sanitized
            # from "__type__" to "type" (+ optional dedupe suffix).
            for k, v in d.items():
                ks = str(k)
                if ks == "type" or ks.startswith("type__") or ks.startswith("type_"):
                    if isinstance(v, str) and "." in v:
                        return ks, v
            return None, None

        if isinstance(obj, dict):
            _, t = _tag_key_and_type(obj)
            if t is None:
                # normal recursion
                return {str(k): _rehydrate(v) for k, v in obj.items()}

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
                    data = _rehydrate(obj.get("data", []))
                    dtype = obj.get("dtype", None)
                    try:
                        if isinstance(dtype, str) and "StringDType" in dtype:
                            a = np.array(data, dtype=str)
                        elif dtype is None:
                            a = np.array(data)
                        else:
                            a = np.array(data, dtype=dtype)
                    except Exception:
                        a = np.array(data)
                    # shape is best-effort; only reshape if consistent
                    shp = _rehydrate(obj.get("shape", None))
                    if shp is not None:
                        try:
                            if isinstance(shp, np.ndarray):
                                shp = shp.tolist()
                            if isinstance(shp, (int, np.integer)):
                                shp = [int(shp)]
                            a = a.reshape(tuple(int(x) for x in shp))
                        except Exception:
                            pass
                    return a
                except Exception:
                    return obj

            if t == _SIDECAR_REF_TYPE:
                return _load_sidecar_ref(obj)

        if isinstance(obj, list):
            return [_rehydrate(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_rehydrate(v) for v in obj)
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                try:
                    candidate = str(obj.reshape(-1)[0])
                except Exception:
                    candidate = None
                if candidate:
                    recovered = _recover_stringified_legacy_array(candidate)
                    if recovered is not None:
                        return recovered
            return obj
        if isinstance(obj, str):
            recovered = _recover_stringified_legacy_array(obj)
            if recovered is not None:
                return recovered

        return obj

    path = Path(path)

    if not path.exists() and str(path).endswith(".zarr"):
        candidate = Path(f"{path}.tar.zst")
        if candidate.exists():
            LOGGER.info("Resolved missing Zarr path %s to archived dataset %s", path, candidate)
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    tmp_dir: Path | None = None
    try:
        if path.suffix == ".h5ad":
            LOGGER.info(f"Loading H5AD dataset → {path}")
            adata = ad.read_h5ad(str(path))
        elif _is_zarr_archive_path(path):
            tmp_dir = Path(tempfile.mkdtemp(prefix="scomnom_load_"))
            member_names = _tar_list_zst(path)
            _validate_tar_members_safe(member_names)
            _tar_extract_zst(path, tmp_dir)

            expected = tmp_dir / _archive_to_zarr_dir_name(path)
            if expected.exists() and expected.is_dir():
                extracted = expected
            else:
                candidates = sorted(p for p in tmp_dir.iterdir() if p.is_dir() and p.name.endswith(".zarr"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Archive {path} did not extract to a single .zarr directory. Found: {[p.name for p in candidates]}"
                    )
                extracted = candidates[0]

            LOGGER.info("Loading archived Zarr dataset %s via extracted directory %s", path, extracted)
            zarr_source_dir = extracted
            adata = ad.read_zarr(str(extracted))
        elif path.suffix == ".zarr" or path.is_dir():
            LOGGER.info(f"Loading Zarr dataset → {path}")
            zarr_source_dir = path
            adata = ad.read_zarr(str(path))  # fully in-memory
        else:
            raise ValueError(
                f"Cannot load dataset from: {path}. Expected .zarr directory, .zarr.tar.zst archive, or .h5ad file."
            )

        # Rehydrate tagged structures in uns
        adata.uns = _rehydrate(dict(adata.uns))

        return adata
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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


def _safe_filename(s: str, max_len: int = 180) -> str:
    return sanitize_identifier(s, max_len=max_len, allow_spaces=False)


def _condition_key_label(condition_key: str) -> str:
    raw = str(condition_key).strip()
    if "^" in raw:
        parts = [p.strip() for p in raw.split("^") if p.strip()]
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1]}__interaction"
        return f"{raw}__interaction"
    return raw


def _short_cluster_sheet_name(group_value: str) -> str:
    s = str(group_value)
    m = re.match(r"^(C\d+)\b", s)
    if m:
        return m.group(1)
    return sanitize_identifier(s, max_len=28, allow_spaces=False)


def export_pseudobulk_de_tables(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    store_key: str = "scomnom_de",
    display_map: Optional[dict[str, str]] = None,
    groupby: Optional[str] = None,
    condition_key: Optional[str] = None,
    tables_root: Optional[Path] = None,
) -> None:
    """
    Write pseudobulk DE results stored in adata.uns[store_key] to CSV files.

    Outputs (relative to output_dir):
      - DE_tables/cluster_vs_rest/cluster_vs_rest__<cluster>.csv
      - DE_tables/cluster_vs_rest/__summary.csv (if present)
      - DE_tables/condition_within_cluster__<condition_key>/condition_within_cluster__<group>__<test>_vs_<ref>.csv
        (interaction: condition_within_cluster__<group>__interaction.csv)

    Assumes storage schema from de_utils:
      adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["results"] : dict[str, DataFrame]
      adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["summary"]  : DataFrame (optional)
      adata.uns[store_key]["pseudobulk_condition_within_group"]      : dict[key -> payload]
    """
    output_dir = Path(output_dir)
    base = Path(tables_root) if tables_root is not None else (output_dir / "DE_tables")
    base.mkdir(parents=True, exist_ok=True)

    block = adata.uns.get(store_key, {})
    if not isinstance(block, dict):
        return

    # -------------------------
    # Cluster vs rest
    # -------------------------
    if condition_key is None:
        pb = block.get("pseudobulk_cluster_vs_rest", {})
        if isinstance(pb, dict):
            out_cluster = base / "cluster_vs_rest"
            out_cluster.mkdir(parents=True, exist_ok=True)

            results_by_cluster = pb.get("results", {})
            if isinstance(results_by_cluster, dict):
                for cl, df in results_by_cluster.items():
                    cl_safe = _safe_filename(_cnn_label_for_group(str(cl), display_map))
                    out_csv = out_cluster / f"cluster_vs_rest__{cl_safe}.csv"

                    if df is None or getattr(df, "empty", True):
                        pd.DataFrame(
                            columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
                        ).to_csv(out_csv, index=False)
                    else:
                        df2 = df.copy()
                        if "gene" not in df2.columns:
                            df2["gene"] = df2.index.astype(str)
                        df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                        df2 = _drop_redundant_group_cols(df2)
                        df2.to_csv(out_csv, index=False)

            summ = pb.get("summary", None)
            if isinstance(summ, pd.DataFrame) and not summ.empty:
                summ.to_csv(out_cluster / "__summary.csv", index=False)

    # -------------------------
    # Condition within group
    # -------------------------
    if condition_key:
        cond = block.get("pseudobulk_condition_within_group_multi", {})
        if not cond:
            cond = block.get("pseudobulk_condition_within_group", {})
        if isinstance(cond, dict) and cond:
            cond_label = _condition_key_label(str(condition_key))
            out_cond = base / f"condition_within_cluster__{_safe_filename(cond_label)}"
            out_cond.mkdir(parents=True, exist_ok=True)

            # keys look like: "{group_key}={group_value}::{condition_key}::{test}_vs_{reference}"
            for k, payload in cond.items():
                if not isinstance(payload, dict):
                    continue
                payload_ck = payload.get("condition_key", None)
                if payload_ck is not None and str(payload_ck) != str(condition_key):
                    continue

                df = payload.get("results", None)
                if df is None:
                    continue

                group_value = payload.get("group_value", None)
                test = payload.get("test", None)
                ref = payload.get("reference", None)
                is_interaction = bool(payload.get("interaction", False))

                # fallback parse from key if payload missing
                if group_value is None and groupby and isinstance(k, str) and k.startswith(f"{groupby}="):
                    # "groupby=VALUE::..."
                    try:
                        group_value = k.split("::", 1)[0].split("=", 1)[1]
                    except Exception:
                        group_value = str(k)

                group_label = _cnn_label_for_group(str(group_value) if group_value is not None else str(k), display_map)
                stem = f"condition_within_cluster__{_safe_filename(str(group_label))}"
                if is_interaction:
                    stem += "__interaction"
                elif test is not None and ref is not None:
                    stem += f"__{_safe_filename(str(test))}_vs_{_safe_filename(str(ref))}"

                out_csv = out_cond / f"{stem}.csv"

                if getattr(df, "empty", True):
                    pd.DataFrame(
                        columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
                    ).to_csv(out_csv, index=False)
                else:
                    df2 = df.copy()
                    if "gene" not in df2.columns:
                        df2["gene"] = df2.index.astype(str)
                    df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                    df2 = _drop_redundant_group_cols(df2)
                    df2.to_csv(out_csv, index=False)


def export_rank_genes_groups_tables(
    adata: ad.AnnData,
    *,
    key_added: str,
    output_dir: Path,
    groupby: Optional[str] = None,
    display_map: Optional[dict[str, str]] = None,
    prefix: str = "celllevel_markers",
    tables_root: Optional[Path] = None,
) -> None:
    """
    Write scanpy rank_genes_groups results (adata.uns[key_added]) to CSV.

    Outputs (relative to output_dir):
      - marker_tables/<prefix>__all_groups.csv (long/tidy)

    This is notebook-friendly data in a stable tabular format.
    """
    output_dir = Path(output_dir)
    out_dir = Path(tables_root) if tables_root is not None else (output_dir / "marker_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    if key_added not in adata.uns:
        raise KeyError(f"export_rank_genes_groups_tables: key_added={key_added!r} not found in adata.uns")

    # Prefer scanpy helper if available
    try:
        import scanpy as sc  # noqa: F401
        from scanpy.get import rank_genes_groups_df

        # rank_genes_groups_df returns df for one group; iterate and concatenate
        # groups are stored in adata.uns[key_added]['names'].dtype.names
        rg = adata.uns[key_added]
        names = rg.get("names", None)
        if names is None or not hasattr(names, "dtype") or not getattr(names.dtype, "names", None):
            raise RuntimeError("rank_genes_groups results missing expected 'names' structured array")

        groups = list(names.dtype.names)
        dfs = []
        for g in groups:
            dfg = rank_genes_groups_df(adata, group=g, key=key_added)
            dfg.insert(0, "group", str(g))
            dfs.append(dfg)

        df_all = pd.concat(dfs, axis=0, ignore_index=True)

    except Exception:
        # Fallback: manual extraction from structured arrays
        rg = adata.uns[key_added]
        names = rg.get("names", None)
        if names is None or not hasattr(names, "dtype") or not getattr(names.dtype, "names", None):
            raise RuntimeError("rank_genes_groups results missing expected 'names' structured array")

        groups = list(names.dtype.names)
        cols = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
        # Some methods may not include all columns; guard each
        available = {c: rg.get(c, None) for c in cols}

        rows = []
        for g in groups:
            n = len(names[g])
            for i in range(n):
                row = {"group": str(g), "gene": str(names[g][i])}
                if available["scores"] is not None:
                    row["score"] = float(available["scores"][g][i])
                if available["logfoldchanges"] is not None:
                    row["logfoldchange"] = float(available["logfoldchanges"][g][i])
                if available["pvals"] is not None:
                    row["pval"] = float(available["pvals"][g][i])
                if available["pvals_adj"] is not None:
                    row["pvals_adj"] = float(available["pvals_adj"][g][i])
                rows.append(row)

        df_all = pd.DataFrame(rows)

    # Normalize columns a bit
    if "names" in df_all.columns and "gene" not in df_all.columns:
        df_all = df_all.rename(columns={"names": "gene"})

    # helpful metadata columns (best-effort)
    if groupby is not None:
        df_all.insert(1, "groupby", str(groupby))
    df_all.insert(2, "key_added", str(key_added))

    if "group" in df_all.columns:
        df_all["group"] = df_all["group"].astype(str).map(lambda x: _cnn_label_for_group(x, display_map))

    if "gene" in df_all.columns:
        df_all = _add_gene_type_column(adata, df_all, gene_col="gene")

    out_csv = out_dir / f"{_safe_filename(prefix)}__all_groups.csv"
    df_all.to_csv(out_csv, index=False)


def _safe_excel_sheet_name(name: str) -> str:
    """
    Excel constraints:
      - max 31 chars
      - cannot contain: : \\ / ? * [ ]
    """
    s = str(name)
    s = re.sub(r"[:\\/?*\[\]]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"_\s+|\s+_", "_", s)
    if not s:
        s = "Sheet"
    return s[:31]


def _display_label_for_group(group: str, display_map: Optional[dict[str, str]]) -> str:
    if isinstance(display_map, dict):
        return str(display_map.get(str(group), str(group)))
    return str(group)


def _cnn_label_for_group(group: str, display_map: Optional[dict[str, str]]) -> str:
    label = _display_label_for_group(group, display_map)
    m = re.search(r"\b(C\d+)\b", str(label))
    if m:
        return m.group(1)
    m2 = re.search(r"(C\d+)", str(label))
    if m2:
        return m2.group(1)
    return str(group)


def _sheet_name_for_group(
    group: str,
    *,
    display_map: Optional[dict[str, str]] = None,
    used: Optional[set[str]] = None,
) -> str:
    used = used if used is not None else set()
    label = _cnn_label_for_group(group, display_map)
    sheet = _safe_excel_sheet_name(label)
    if sheet in used:
        sheet = _safe_excel_sheet_name(f"{label}_{group}")
    if sheet in used:
        sheet = _safe_excel_sheet_name(str(group))
    used.add(sheet)
    return sheet


def export_pseudobulk_cluster_vs_rest_excel(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    store_key: str = "scomnom_de",
    display_map: Optional[dict[str, str]] = None,
    filename: str = "cluster_vs_rest.xlsx",
    tables_root: Optional[Path] = None,
) -> None:
    """
    Write ALL cluster-vs-rest DE tables into a single Excel workbook.
    One cluster per sheet.
    """
    output_dir = Path(output_dir)
    base = Path(tables_root) if tables_root is not None else (output_dir / "DE_tables")
    out_xlsx = base / filename
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    block = adata.uns.get(store_key, {})
    pb = block.get("pseudobulk_cluster_vs_rest", {})
    results = pb.get("results", {})

    if not isinstance(results, dict) or not results:
        return

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        used: set[str] = set()
        for cluster, df in results.items():
            sheet = _sheet_name_for_group(str(cluster), display_map=display_map, used=used)

            if df is None or getattr(df, "empty", True):
                pd.DataFrame(
                    columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
                ).to_excel(writer, sheet_name=sheet, index=False)
            else:
                df2 = df.copy()
                if "gene" not in df2.columns:
                    df2["gene"] = df2.index.astype(str)
                df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                df2 = _drop_redundant_group_cols(df2)
                df2.to_excel(writer, sheet_name=sheet, index=False)


def export_pseudobulk_condition_within_cluster_excel(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    store_key: str = "scomnom_de",
    condition_key: str,
    display_map: Optional[dict[str, str]] = None,
    filename: Optional[str] = None,
    tables_root: Optional[Path] = None,
) -> None:
    """
    One Excel file per contrast within condition_key.
    One sheet per cluster (Cnn).
    """
    if not condition_key:
        return

    output_dir = Path(output_dir)
    base = Path(tables_root) if tables_root is not None else (output_dir / "DE_tables")

    block = adata.uns.get(store_key, {})
    cond = block.get("pseudobulk_condition_within_group_multi", {})
    if not cond:
        cond = block.get("pseudobulk_condition_within_group", {})

    if not isinstance(cond, dict) or not cond:
        return

    cond_label = _condition_key_label(str(condition_key))

    by_contrast: dict[str, list[dict]] = {}
    for key, payload in cond.items():
        if not isinstance(payload, dict):
            continue
        payload_ck = payload.get("condition_key", None)
        if payload_ck is not None and str(payload_ck) != str(condition_key):
            continue
        df = payload.get("results", None)
        if df is None:
            continue
        is_interaction = bool(payload.get("interaction", False))
        if is_interaction:
            contrast_key = "interaction"
        else:
            test = payload.get("test", None)
            ref = payload.get("reference", None)
            if test and ref:
                contrast_key = f"{test}_vs_{ref}"
            else:
                contrast_key = "contrast"
        by_contrast.setdefault(str(contrast_key), []).append(payload)

    for contrast_key, payloads in by_contrast.items():
        if filename is None:
            if contrast_key == "interaction":
                fname = f"condition_within_cluster__{_safe_filename(cond_label)}.xlsx"
            else:
                fname = f"condition_within_cluster__{_safe_filename(cond_label)}__{_safe_filename(contrast_key)}.xlsx"
        else:
            fname = filename

        out_xlsx = base / fname
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            used: set[str] = set()
            for payload in payloads:
                df = payload.get("results", None)
                if df is None:
                    continue
                group_value = payload.get("group_value", "")
                sheet = _sheet_name_for_group(
                    str(group_value),
                    display_map=display_map,
                    used=used,
                )
                if getattr(df, "empty", True):
                    pd.DataFrame(
                        columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
                    ).to_excel(writer, sheet_name=sheet, index=False)
                else:
                    df2 = df.copy()
                    if "gene" not in df2.columns:
                        df2["gene"] = df2.index.astype(str)
                    df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                    df2 = _drop_redundant_group_cols(df2)
                    df2.to_excel(writer, sheet_name=sheet, index=False)


def export_rank_genes_groups_excel(
    adata: ad.AnnData,
    *,
    key_added: str,
    output_dir: Path,
    groupby: Optional[str] = None,
    display_map: Optional[dict[str, str]] = None,
    filename: Optional[str] = None,
    prefix: str = "celllevel_markers",
    max_genes: Optional[int] = None,
    tables_root: Optional[Path] = None,
) -> None:
    """
    Write scanpy rank_genes_groups results (adata.uns[key_added]) to a single Excel workbook.
    One group per sheet.

    Outputs (relative to output_dir):
      - marker_tables/<filename>.xlsx

    Notes:
      - Uses xlsxwriter.
      - Sheet names are sanitized + truncated to Excel limits.
      - If max_genes is set, truncates each sheet to that many rows.
    """
    output_dir = Path(output_dir)
    out_dir = Path(tables_root) if tables_root is not None else (output_dir / "marker_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    if key_added not in adata.uns:
        raise KeyError(f"export_rank_genes_groups_excel: key_added={key_added!r} not found in adata.uns")

    if filename is None:
        filename = f"{_safe_filename(prefix)}.xlsx"

    out_xlsx = out_dir / filename
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # Prefer scanpy helper for consistent columns
    df_all: pd.DataFrame
    try:
        from scanpy.get import rank_genes_groups_df

        rg = adata.uns[key_added]
        names = rg.get("names", None)
        if names is None or not hasattr(names, "dtype") or not getattr(names.dtype, "names", None):
            raise RuntimeError("rank_genes_groups results missing expected 'names' structured array")

        groups = list(names.dtype.names)
        dfs = []
        for g in groups:
            dfg = rank_genes_groups_df(adata, group=g, key=key_added)
            dfg.insert(0, "group", str(g))
            dfs.append(dfg)
        df_all = pd.concat(dfs, axis=0, ignore_index=True)

    except Exception:
        # Fallback: reconstruct a tidy table
        rg = adata.uns[key_added]
        names = rg.get("names", None)
        if names is None or not hasattr(names, "dtype") or not getattr(names.dtype, "names", None):
            raise RuntimeError("rank_genes_groups results missing expected 'names' structured array")

        groups = list(names.dtype.names)
        cols = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
        available = {c: rg.get(c, None) for c in cols}

        rows = []
        for g in groups:
            n = len(names[g])
            for i in range(n):
                row = {"group": str(g), "gene": str(names[g][i])}
                if available["scores"] is not None:
                    row["score"] = float(available["scores"][g][i])
                if available["logfoldchanges"] is not None:
                    row["logfoldchange"] = float(available["logfoldchanges"][g][i])
                if available["pvals"] is not None:
                    row["pval"] = float(available["pvals"][g][i])
                if available["pvals_adj"] is not None:
                    row["pvals_adj"] = float(available["pvals_adj"][g][i])
                rows.append(row)

        df_all = pd.DataFrame(rows)

    # Normalize columns
    if "names" in df_all.columns and "gene" not in df_all.columns:
        df_all = df_all.rename(columns={"names": "gene"})

    # Add metadata cols (best-effort)
    if groupby is not None and "groupby" not in df_all.columns:
        df_all.insert(1, "groupby", str(groupby))
    if "key_added" not in df_all.columns:
        df_all.insert(2, "key_added", str(key_added))
    if "group" in df_all.columns:
        df_all["group"] = df_all["group"].astype(str).map(lambda x: _cnn_label_for_group(x, display_map))
    if "gene" in df_all.columns:
        df_all = _add_gene_type_column(adata, df_all, gene_col="gene")

    # Write workbook: 1 sheet per group
    if "group" not in df_all.columns:
        # Degenerate case: just dump everything into one sheet
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            df_write = df_all.copy()
            if max_genes is not None and max_genes > 0:
                if "gene_type" in df_write.columns:
                    df_write = df_write[df_write["gene_type"] == "protein_coding"]
                df_write = df_write.head(int(max_genes))
            df_write = _drop_redundant_group_cols(df_write)
            df_write.to_excel(writer, sheet_name=_safe_excel_sheet_name("markers"), index=False)
        return

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        groups = list(pd.unique(df_all["group"].astype(str)))
        used: set[str] = set()
        for g in groups:
            dfg = df_all[df_all["group"].astype(str) == str(g)].copy()
            if max_genes is not None and max_genes > 0:
                if "gene_type" in dfg.columns:
                    dfg = dfg[dfg["gene_type"] == "protein_coding"]
                dfg = dfg.head(int(max_genes))
            dfg = _drop_redundant_group_cols(dfg)
            sheet = _sheet_name_for_group(str(g), display_map=display_map, used=used)
            dfg.to_excel(writer, sheet_name=sheet, index=False)


# --- add to io_utils.py ---

def export_contrast_conditional_markers_tables(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    store_key: str = "scomnom_de",
    display_map: Optional[dict[str, str]] = None,
    filename: Optional[str] = None,
    tables_root: Optional[Path] = None,
    contrast_key: Optional[str] = None,
) -> None:
    """
    Writes contrast-conditional (pairwise) marker results:
      - CSV folder per (cluster, A_vs_B)
      - One XLSX workbook with one sheet per (cluster, A_vs_B)
      - Summary CSV
    """
    output_dir = Path(output_dir)
    base = Path(tables_root) if tables_root is not None else (output_dir / "marker_tables")

    block = adata.uns.get(store_key, {})
    cc = block.get("contrast_conditional", {})
    if not isinstance(cc, dict):
        return

    results = cc.get("results", {})
    summary = cc.get("summary", None)

    if not isinstance(results, dict) or not results:
        return

    condition = str(contrast_key) if contrast_key else "contrast"
    pair_keys: set[str] = set()
    for pairs in results.values():
        if isinstance(pairs, dict):
            pair_keys.update(str(k) for k in pairs.keys())

    for pair_key in sorted(pair_keys):
        out_dir = base / f"{_safe_filename(condition)}_{_safe_filename(pair_key)}_DE"
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(summary, pd.DataFrame) and not summary.empty:
            summary.to_csv(out_dir / "__summary.csv", index=False)

        for cluster, pairs in results.items():
            if not isinstance(pairs, dict):
                continue
            payload = pairs.get(pair_key, None)
            if not isinstance(payload, dict):
                continue

            cluster_label = _cnn_label_for_group(str(cluster), display_map)
            stem = f"cluster__{_safe_filename(cluster_label)}__{_safe_filename(pair_key)}"
            subdir = out_dir / stem
            subdir.mkdir(parents=True, exist_ok=True)

            for kind in ["combined", "wilcoxon", "logreg", "pseudobulk_effect"]:
                df = payload.get(kind, None)
                if df is None or getattr(df, "empty", True):
                    pd.DataFrame().to_csv(subdir / f"{stem}__{kind}.csv", index=False)
                else:
                    df2 = df.copy()
                    if "gene" in df2.columns:
                        df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                    df2 = _drop_redundant_group_cols(df2)
                    df2.to_csv(subdir / f"{stem}__{kind}.csv", index=False)

        out_xlsx = out_dir / (filename or f"{_safe_filename(condition)}_{_safe_filename(pair_key)}_DE.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            used: set[str] = set()
            for cl2, pairs2 in results.items():
                if not isinstance(pairs2, dict):
                    continue
                payload2 = pairs2.get(pair_key, None)
                if not isinstance(payload2, dict):
                    continue
                df = payload2.get("combined", None)
                if df is None:
                    continue

                sheet = _sheet_name_for_group(str(cl2), display_map=display_map, used=used)
                if getattr(df, "empty", True):
                    pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
                else:
                    df2 = df.copy()
                    if "gene" in df2.columns:
                        df2 = _add_gene_type_column(adata, df2, gene_col="gene")
                    df2 = _drop_redundant_group_cols(df2)
                    df2.to_excel(writer, sheet_name=sheet, index=False)


def export_contrast_conditional_markers_tables_multi(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    store_key: str = "scomnom_de",
    display_map: Optional[dict[str, str]] = None,
    filename: Optional[str] = None,
    tables_root: Optional[Path] = None,
    contrast_key: Optional[str] = None,
) -> None:
    """
    Export contrast-conditional markers for a specific contrast_key from
    adata.uns[store_key]["contrast_conditional_multi"].
    """
    block = adata.uns.get(store_key, {})
    multi = block.get("contrast_conditional_multi", None)
    if not isinstance(multi, dict):
        return

    key = str(contrast_key) if contrast_key else None
    if key is None or key not in multi:
        return

    orig = block.get("contrast_conditional", None)
    block["contrast_conditional"] = multi.get(key)
    try:
        export_contrast_conditional_markers_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            display_map=display_map,
            filename=filename,
            tables_root=tables_root,
            contrast_key=key,
        )
    finally:
        if orig is None:
            block.pop("contrast_conditional", None)
        else:
            block["contrast_conditional"] = orig
