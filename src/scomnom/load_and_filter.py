# src/scomnom/load_and_filter.py

from __future__ import annotations

import logging
import warnings

from pathlib import Path
from typing import Dict, Optional, Literal
import numpy as np
import torch

import anndata as ad
import pandas as pd
import scanpy as sc

from . import io_utils
from . import plot_utils

LOGGER = logging.getLogger(__name__)


def _validate_metadata_samples(
    metadata_tsv: Path,
    batch_key: str,
    loaded_samples: Dict[str, ad.AnnData],
) -> None:
    """
    Ensure metadata_tsv contains exactly one row per sample and matches loaded sample IDs.
    """
    df_meta = pd.read_csv(metadata_tsv, sep="\t")

    if batch_key not in df_meta.columns:
        raise KeyError(
            f"Metadata TSV must contain the batch key column '{batch_key}'. "
            f"Found columns: {list(df_meta.columns)}"
        )

    meta_samples = pd.Index(df_meta[batch_key].astype(str))
    loaded = pd.Index(list(loaded_samples.keys())).astype(str)

    missing_rows = loaded.difference(meta_samples)
    extra_rows = meta_samples.difference(loaded)

    if len(missing_rows) > 0:
        raise ValueError(
            "The following samples were found in the input data but "
            "are missing from metadata_tsv:\n"
            f"  {list(missing_rows)}"
        )

    if len(extra_rows) > 0:
        raise ValueError(
            "The following samples exist in metadata_tsv but were not found "
            "in the input data folders:\n"
            f"  {list(extra_rows)}"
        )

    if len(meta_samples) != len(loaded):
        raise ValueError(
            f"metadata_tsv has {len(meta_samples)} rows for batch_key='{batch_key}', "
            f"but {len(loaded)} samples were loaded."
        )


def _add_metadata(adata: ad.AnnData, metadata_tsv: Path, sample_id_col: str) -> ad.AnnData:
    """
    Attach per-sample metadata from TSV to adata.obs, mirroring load_data.add_metadata.
    """
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
            continue
        adata.obs[col] = merged[col].values
        if (
            adata.obs[col].dtype == object
            and adata.obs[col].nunique() < 0.1 * len(adata.obs)
        ):
            adata.obs[col] = adata.obs[col].astype("category")

    return adata


def _per_sample_qc_and_filter(
    sample_map: Dict[str, ad.AnnData],
    cfg: LoadAndQCConfig,
) -> tuple[Dict[str, ad.AnnData], pd.DataFrame]:
    """
    Run QC + min_genes/min_cells filtering per sample (OOM-safe) and
    collect a lightweight QC dataframe for pre-filter plots.
    """
    import pandas as pd

    filtered_samples: Dict[str, ad.AnnData] = {}
    qc_rows = []

    for sample, a in sample_map.items():
        LOGGER.info(
            "[Per-sample QC] %s: %d cells × %d genes",
            sample,
            a.n_obs,
            a.n_vars,
        )

        # QC metrics (sparse-safe)
        a = compute_qc_metrics(a, cfg)  # uses mt_prefix/ribo_prefixes/hb_regex from cfg

        # small QC df for plotting
        qc_rows.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "total_counts": a.obs["total_counts"].to_numpy(),
                    "n_genes_by_counts": a.obs["n_genes_by_counts"].to_numpy(),
                    "pct_counts_mt": a.obs["pct_counts_mt"].to_numpy(),
                    "pct_counts_ribo": a.obs["pct_counts_ribo"].to_numpy(),
                    "pct_counts_hb": a.obs["pct_counts_hb"].to_numpy(),
                }
            )
        )

        # filtering
        a = sparse_filter_cells_and_genes(
            a,
            min_genes=cfg.min_genes,
            min_cells=cfg.min_cells,
            max_pct_mt=cfg.max_pct_mt,
        )

        if a.n_obs < cfg.min_cells_per_sample:
            LOGGER.warning(
                "[Per-sample QC] Dropping sample %s: %d cells < min_cells_per_sample=%d",
                sample,
                a.n_obs,
                cfg.min_cells_per_sample,
            )
            continue
        LOGGER.info(
            "[Per-sample QC] %s: %d cells × %d genes after filtering",
            sample,
            a.n_obs,
            a.n_vars,
        )

        filtered_samples[sample] = a

    qc_df = (
        pd.concat(qc_rows, axis=0, ignore_index=True)
        if qc_rows
        else pd.DataFrame(
            columns=[
                "sample",
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_mt",
                "pct_counts_ribo",
                "pct_counts_hb",
            ]
        )
    )

    return filtered_samples, qc_df


def _select_device():
    if torch.cuda.is_available():
        return "gpu", "auto"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def _is_oom_error(e: Exception) -> bool:
    txt = str(e).lower()
    return (
        "out of memory" in txt
        or "cuda error" in txt
        or "cublas_status_alloc_failed" in txt
        or ("mps" in txt and "oom" in txt)
    )


def _auto_scvi_epochs(n_cells: int) -> int:
    if n_cells < 50_000:
        return 80
    if n_cells < 200_000:
        return 60
    return 40


# ---------------------------------------------------------------------
# Generic SCVI trainer (used everywhere)
# ---------------------------------------------------------------------
def _train_scvi(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    layer: Optional[str],
    purpose: str,
):
    """
    Train an SCVI model with auto batch-size + auto epochs.

    purpose: "solo" | "integration" (logging only)
    """
    from scvi.model import SCVI

    accelerator, devices = _select_device()
    epochs = _auto_scvi_epochs(adata.n_obs)

    batch_ladder = [1024, 512, 256, 128, 64, 32]
    last_err = None

    LOGGER.info(
        "Training SCVI for %s (n_cells=%d, epochs=%d)",
        purpose,
        adata.n_obs,
        epochs,
    )

    for bsz in batch_ladder:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*setup_anndata is overwriting.*")
                SCVI.setup_anndata(
                    adata,
                    layer=layer,
                    batch_key=batch_key,
                )

            model = SCVI(adata)
            model.train(
                max_epochs=epochs,
                accelerator=accelerator,
                devices=devices,
                batch_size=bsz,
                enable_progress_bar=True,
            )

            LOGGER.info("SCVI trained successfully (batch_size=%d)", bsz)
            return model

        except RuntimeError as e:
            if _is_oom_error(e):
                LOGGER.warning("OOM at batch_size=%d, retrying smaller...", bsz)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_err = e
                continue
            raise

    raise RuntimeError("SCVI training failed") from last_err


# ---------------------------------------------------------------------
# QC metric computation (sparse-safe)
# ---------------------------------------------------------------------
def compute_qc_metrics(adata: ad.AnnData, cfg: QCFilterConfig) -> ad.AnnData:
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        LOGGER.warning("X is dense — QC may be slow and memory intensive.")

    n_cells, n_genes = X.shape

    # Gene categories
    mt_prefix = getattr(cfg, "mt_prefix", "MT-")
    ribo_prefixes = getattr(cfg, "ribo_prefixes", ["RPL", "RPS"])
    hb_regex = r"^(?:HB[AB]|HBA|HBB)"

    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(ribo_prefixes))
    adata.var["hb"] = adata.var_names.str.contains(hb_regex, regex=True)

    mt_idx = np.where(adata.var["mt"].values)[0]
    ribo_idx = np.where(adata.var["ribo"].values)[0]
    hb_idx = np.where(adata.var["hb"].values)[0]

    LOGGER.info("Computing sparse per-cell QC metrics...")

    total_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes_by_counts = np.diff(X.indptr)

    def pct_from_idx(idx):
        if len(idx) == 0:
            return np.zeros(n_cells)
        vals = np.asarray(X[:, idx].sum(axis=1)).ravel()
        return vals / np.maximum(total_counts, 1)

    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    adata.obs["pct_counts_mt"] = pct_from_idx(mt_idx) * 100
    adata.obs["pct_counts_ribo"] = pct_from_idx(ribo_idx) * 100
    adata.obs["pct_counts_hb"] = pct_from_idx(hb_idx) * 100

    LOGGER.info("Computing sparse per-gene QC metrics...")

    n_cells_by_counts = np.diff(X.tocsc().indptr)
    total_counts_gene = np.asarray(X.sum(axis=0)).ravel()
    mean_counts = total_counts_gene / max(n_cells, 1)
    pct_dropout = 100 * (1 - (n_cells_by_counts / max(n_cells, 1)))

    adata.var["n_cells_by_counts"] = n_cells_by_counts
    adata.var["mean_counts"] = mean_counts
    adata.var["total_counts"] = total_counts_gene
    adata.var["pct_dropout_by_counts"] = pct_dropout

    adata.uns["qc_metrics"] = {
        "qc_vars": ["mt", "ribo", "hb"],
        "percent_top": {},
        "log1p": False,
        "raw_qc_metrics": {},
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    }

    return adata


# ---------------------------------------------------------------------
# Sparse filtering
# ---------------------------------------------------------------------
def sparse_filter_cells_and_genes(
    adata: ad.AnnData,
    *,
    min_genes: int,
    min_cells: int,
    max_pct_mt: float | None = None,
) -> ad.AnnData:
    import numpy as np

    X = adata.X

    # --------------------------------------------------
    # Cell filtering: min_genes
    # --------------------------------------------------
    gene_counts = np.diff(X.indptr)
    cell_mask = gene_counts >= min_genes
    if cell_mask.sum() == 0:
        raise ValueError(f"All cells removed by min_genes={min_genes}.")
    adata = adata[cell_mask].copy()
    X = adata.X

    # --------------------------------------------------
    # Cell filtering: max_pct_mt
    # --------------------------------------------------
    if max_pct_mt is not None:
        if "pct_counts_mt" not in adata.obs:
            raise KeyError(
                "pct_counts_mt not found in adata.obs. "
                "Run compute_qc_metrics() before sparse filtering."
            )

        mt_mask = adata.obs["pct_counts_mt"].to_numpy() <= max_pct_mt
        if mt_mask.sum() == 0:
            raise ValueError(
                f"All cells removed by max_pct_mt={max_pct_mt}."
            )

        adata = adata[mt_mask].copy()
        X = adata.X

    # --------------------------------------------------
    # Gene filtering: min_cells
    # --------------------------------------------------
    gene_nnz = np.bincount(X.indices, minlength=adata.n_vars)
    gene_mask = gene_nnz >= min_cells
    if gene_mask.sum() == 0:
        raise ValueError(f"All genes removed by min_cells={min_cells}.")
    adata = adata[:, gene_mask].copy()

    return adata


# ---------------------------------------------------------------------
# SOLO (always global)
# ---------------------------------------------------------------------
def run_solo_with_scvi(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    doublet_mode: Literal["fixed", "rate", "gmm"],
    doublet_score_threshold: float,
    expected_doublet_rate: float,
) -> ad.AnnData:
    from scvi.external import SOLO

    LOGGER.info("Running SOLO doublet detection (global)")

    layer = "counts_raw" if "counts_raw" in adata.layers else None
    scvi_model = _train_scvi(
        adata,
        batch_key=batch_key,
        layer=layer,
        purpose="solo",
    )

    accelerator, devices = _select_device()

    solo = SOLO.from_scvi_model(scvi_model)
    solo.train(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
    )

    probs = solo.predict(soft=True)
    scores = probs["doublet"].to_numpy()

    adata.obs["doublet_score"] = scores

    mask = _call_doublets(
        scores,
        mode=doublet_mode,
        fixed_threshold=doublet_score_threshold,
        expected_rate=expected_doublet_rate,
    )

    adata.obs["predicted_doublet"] = pd.Categorical(
        mask,
        categories=[False, True],
        ordered=False,
    )

    LOGGER.info(
        "Doublet calling: mode=%s, detected=%d / %d (%.2f%%)",
        doublet_mode,
        mask.sum(),
        adata.n_obs,
        100 * mask.mean(),
    )

    del scvi_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adata


# ---------------------------------------------------------------------
# Cleanup after SOLO
# ---------------------------------------------------------------------
def cleanup_after_solo(
    adata: ad.AnnData,
    batch_key: str,
    min_cells_per_sample: int,
    doublet_mode: Literal["fixed", "rate", "gmm"],
    *,
    expected_doublet_rate: float | None = None,
) -> ad.AnnData:
    # Resolve batch_key: cfg > adata.uns
    if batch_key is None:
        batch_key = adata.uns.get("batch_key", None)

    if batch_key is None:
        raise RuntimeError(
            "Cannot apply min_cells_per_sample: batch_key is None and not "
            "found in adata.uns['batch_key']"
        )

    if batch_key not in adata.obs:
        raise RuntimeError(
            f"Resolved batch_key '{batch_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # -------------------------------------------------
    # Doublet removal + logging
    # -------------------------------------------------
    n_before = adata.n_obs

    if "predicted_doublet" in adata.obs:
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()

    n_after = adata.n_obs

    if n_after != n_before:
        _log_doublet_cleanup(
            n_before=n_before,
            n_after=n_after,
            doublet_mode=doublet_mode,
            inferred_threshold=adata.uns.get("doublet_threshold"),
            expected_rate=expected_doublet_rate if doublet_mode == "rate" else None,
        )

    # -------------------------------------------------
    # Min-cells-per-sample filtering
    # -------------------------------------------------
    if min_cells_per_sample > 0:
        vc = adata.obs[batch_key].value_counts()
        small = vc[vc < min_cells_per_sample].index
        if len(small):
            n0 = adata.n_obs
            adata = adata[~adata.obs[batch_key].isin(small)].copy()
            LOGGER.info(
                "Dropped %d small samples (<%d cells); kept %d / %d",
                len(small),
                min_cells_per_sample,
                adata.n_obs,
                n0,
            )

    # Persist for downstream plotting / resume
    adata.uns["doublet_mode"] = doublet_mode

    return adata


def infer_doublet_threshold(
    adata: ad.AnnData,
    *,
    mode: Literal["fixed", "rate", "gmm"],
    expected_doublet_rate: float | None = None,
    doublet_score_threshold: float | None = None,
) -> float | None:
    if mode == "fixed":
        return float(doublet_score_threshold) if doublet_score_threshold is not None else None

    if mode == "rate":
        if expected_doublet_rate is None:
            return None
        if "doublet_score" not in adata.obs:
            return None
        scores = adata.obs["doublet_score"].to_numpy()
        # threshold so that approx expected_doublet_rate are called doublets (upper tail)
        q = 1.0 - float(expected_doublet_rate)
        return float(np.quantile(scores, q))

    return adata.uns.get("doublet_threshold")



def _call_doublets(
    scores: np.ndarray,
    mode: str,
    *,
    fixed_threshold: float,
    expected_rate: float,
) -> np.ndarray:
    n = scores.size

    if mode == "fixed":
        return scores > fixed_threshold

    if mode == "rate":
        k = int(np.ceil(expected_rate * n))
        if k <= 0:
            return np.zeros(n, dtype=bool)
        idx = np.argsort(scores)[::-1][:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask

    raise ValueError(f"Unknown doublet threshold mode: {mode}")


# ---------------------------------------------------------------------
# Normalization + PCA
# ---------------------------------------------------------------------
def normalize_and_hvg(adata: ad.AnnData, n_top_genes: int, batch_key: str) -> ad.AnnData:
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
    )
    return adata


def pca_neighbors_umap(
    adata: ad.AnnData,
    *,
    var_explained: float = 0.85,
    min_pcs: int = 20,
    max_pcs: int = 50,
) -> ad.AnnData:

    if not (0 < var_explained <= 1):
        raise ValueError("var_explained must be in (0, 1]")

    sc.tl.pca(
        adata,
        n_comps=max_pcs,
        mask_var="highly_variable",
        svd_solver="arpack",
    )

    if "X_pca" not in adata.obsm:
        raise RuntimeError("PCA failed: adata.obsm['X_pca'] missing")

    vr = adata.uns["pca"]["variance_ratio"]
    cum = np.cumsum(vr)

    n_pcs = int(np.searchsorted(cum, var_explained) + 1)
    n_pcs = max(min_pcs, min(n_pcs, max_pcs))

    LOGGER.info(
        "Using n_pcs=%d (%.1f%% variance explained)",
        n_pcs,
        100 * cum[n_pcs - 1],
    )

    sc.pp.neighbors(
        adata,
        n_pcs=n_pcs,
        use_rep="X_pca",
    )

    sc.tl.umap(adata)

    adata.uns["n_pcs"] = n_pcs
    adata.uns["variance_explained"] = float(cum[n_pcs - 1])

    return adata




def cluster_unintegrated(adata: ad.AnnData) -> ad.AnnData:
    sc.tl.leiden(
        adata,
        resolution=1.0,
        key_added="leiden",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )

    return adata


def _log_doublet_cleanup(
    *,
    n_before: int,
    n_after: int,
    doublet_mode: str,
    inferred_threshold: float | None,
    expected_rate: float | None = None,
):
    removed = n_before - n_after
    frac_removed = removed / max(n_before, 1)

    lines = [
        "SOLO doublet filtering summary:",
        f"  cells before = {n_before}",
        f"  cells after  = {n_after}",
        f"  removed      = {removed} ({frac_removed:.2%})",
        f"  mode         = {doublet_mode}",
    ]

    if doublet_mode == "rate":
        # runtime intent
        if expected_rate is not None:
            lines.append(f"  expected rate (config) = {expected_rate:.2%}")

        # outcome always known
        lines.append(f"  observed rate (data)   = {frac_removed:.2%}")

        # threshold is secondary / diagnostic
        if inferred_threshold is not None:
            lines.append(f"  inferred threshold     = {inferred_threshold:.4f}")
        else:
            lines.append("  inferred threshold     = <not computed>")

    elif doublet_mode == "fixed":
        if inferred_threshold is not None:
            lines.append(f"  fixed threshold        = {inferred_threshold:.4f}")
        else:
            lines.append("  fixed threshold        = <none>")

        lines.append(f"  observed rate (data)   = {frac_removed:.2%}")

    else:
        # future-proofing (e.g. gmm)
        lines.append(f"  observed rate (data)   = {frac_removed:.2%}")

    LOGGER.info("\n".join(lines))



def run_load_and_filter(
    cfg: LoadAndFilterConfig) -> ad.AnnData:

    LOGGER.info("Starting load-and-filter")
    # Configure Scanpy/Matplotlib figure behavior + formats
    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    # If we are only applying a different doublet filter:
    if cfg.apply_doublet_score is not None:

        LOGGER.info(
            "Resuming from pre-doublet AnnData: %s",
            cfg.apply_doublet_score,
        )

        adata = io_utils.load_dataset(cfg.apply_doublet_score_path)

        # sanity check: SOLO must already have been run
        required = {"doublet_score", "predicted_doublet"}
        missing = required.difference(adata.obs.columns)
        if missing:
            raise RuntimeError(
                f"apply-doublet-score input is missing required SOLO fields: {missing}"
            )

        if cfg.apply_doublet_score:
            if cfg.raw_sample_dir is None:
                LOGGER.info(
                    "Resume mode without raw_sample_dir: "
                    "raw-based QC and CellBender comparison plots will be skipped."
                )


    # If we are running normally:
    else:
        # Infer batch key from metadata if needed
        batch_key = io_utils.infer_batch_key_from_metadata_tsv(
            cfg.metadata_tsv, cfg.batch_key
        )
        LOGGER.info("Using batch_key='%s'", batch_key)
        cfg.batch_key = batch_key

        # ---------------------------------------------------------
        # Select input source and load samples
        # ---------------------------------------------------------
        # raw only
        # filtered only
        # cellbender
        # raw + cellbender

        if cfg.filtered_sample_dir is not None:
            if cfg.raw_sample_dir or cfg.cellbender_dir:
                raise RuntimeError("--filtered-sample-dir cannot be combined with other inputs")

        elif cfg.cellbender_dir is not None:
            if cfg.raw_sample_dir is None:
                raise RuntimeError("--cellbender-dir requires --raw-sample-dir")

        elif cfg.raw_sample_dir is None:
            raise RuntimeError(
                "You must provide one of:\n"
                "  --raw-sample-dir\n"
                "  --filtered-sample-dir\n"
                "  --raw-sample-dir + --cellbender-dir"
            )

        if cfg.filtered_sample_dir is not None:
            LOGGER.info("Loading CellRanger filtered matrices...")
            sample_map, read_counts = io_utils.load_filtered_data(cfg)

        elif cfg.cellbender_dir is not None:
            LOGGER.info("Loading Cellbender filtered matrices...")
            sample_map, read_counts = io_utils.load_cellbender_filtered_data(cfg)

        else:
            LOGGER.info("Loading RAW matrices...")
            sample_map, read_counts, _ = io_utils.load_raw_data(
                cfg,
                plot_dir=cfg.figdir / "cell_qc",
            )

        # ---------------------------------------------------------
        # Validate metadata vs loaded samples
        # ---------------------------------------------------------
        LOGGER.info("Validating samples and metadata...")

        if cfg.metadata_tsv is None:
            raise RuntimeError("metadata_tsv is required but was None")

        _validate_metadata_samples(
            metadata_tsv=cfg.metadata_tsv,
            batch_key=cfg.batch_key,
            loaded_samples=sample_map,
        )

        LOGGER.info("Loaded %d samples.", len(sample_map))

        # ---------------------------------------------------------
        # Per-sample QC + sparse filtering (OOM-safe)
        # ---------------------------------------------------------
        LOGGER.info("Running per-sample QC and filtering...")
        filtered_sample_map, qc_df = _per_sample_qc_and_filter(sample_map, cfg)

        # ---------------------------------------------------------
        # Pre-filter QC plots (lightweight, from qc_df only)
        # ---------------------------------------------------------
        if cfg.make_figures:
            LOGGER.info("Plotting pre-filter QC...")
            plot_utils.run_qc_plots_pre_filter_df(qc_df, cfg)

        # ---------------------------------------------------------
        # Merge filtered samples into a single AnnData
        # ---------------------------------------------------------
        if cfg.cellbender_dir is not None:
            input_layer_name = "counts_cb"
        else:
            input_layer_name = "counts_raw"

        LOGGER.info("Merging filtered samples into a single AnnData...")
        adata = io_utils.merge_samples(
            filtered_sample_map,
            batch_key=cfg.batch_key,
            input_layer_name = input_layer_name,
        )

        LOGGER.info(
            "Merged filtered dataset: %d cells × %d genes",
            adata.n_obs,
            adata.n_vars,
        )

        # ---------------------------------------------------------
        # Attach per-sample metadata
        # ---------------------------------------------------------
        LOGGER.info("Adding metadata...")
        adata = _add_metadata(adata, cfg.metadata_tsv, sample_id_col=cfg.batch_key)
        adata.uns["batch_key"] = cfg.batch_key

        # canonical identifiers for downstream matching
        if "sample_id" not in adata.obs:
            adata.obs["sample_id"] = adata.obs[cfg.batch_key].astype(str)
        if "barcode" not in adata.obs:
            adata.obs["barcode"] = adata.obs_names.astype(str)

        # ---------------------------------------------------------
        # SOLO doublet detection (GLOBAL, RAW COUNTS)
        # ---------------------------------------------------------
        LOGGER.info("Running SOLO doublet detection")
        adata = run_solo_with_scvi(
            adata,
            batch_key=cfg.batch_key,
            doublet_mode=cfg.doublet_mode,
            doublet_score_threshold=cfg.doublet_score_threshold,
            expected_doublet_rate=cfg.expected_doublet_rate,
        )

        LOGGER.info("Saving Anndata with doublet scores...")
        pre_path = cfg.output_dir / "adata.merged.zarr"
        adata.write_zarr(pre_path, chunks=None)

        if cfg.save_h5ad:
            adata.write_h5ad(
                cfg.output_dir / "adata.merged.h5ad",
                compression="gzip",
            )

        LOGGER.info("Saved pre-doublet filter AnnData → %s", pre_path)

    # Here 'normal mode' and 'only apply doublet filter' merge again
    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if cfg.make_figures:
        plot_utils.doublet_plots(
            adata,
            batch_key=batch_key,
            figdir=cfg.figdir / "QC_plots" / "doublets",
        )

    adata = cleanup_after_solo(
        adata,
        batch_key=batch_key,
        min_cells_per_sample=cfg.min_cells_per_sample,
        doublet_mode=cfg.doublet_mode,
        expected_doublet_rate=cfg.expected_doublet_rate
        if cfg.doublet_mode == "rate"
        else None,
    )

    # ---------------------------------------------------------
    # Attach Raw counts if available
    # ---------------------------------------------------------
    if cfg.raw_sample_dir is not None and "counts_raw" not in adata.layers:
        adata = io_utils.attach_raw_counts_postfilter(cfg, adata)

    # ---------------------------------------------------------
    # Global QC on merged filtered data (for post-filter plots ONLY)
    # ---------------------------------------------------------
    LOGGER.info("Computing QC metrics on merged filtered data...")
    adata = compute_qc_metrics(adata, cfg)

    # ---------------------------------------------------------
    # Post-filter QC plots (NO additional filtering here)
    # ---------------------------------------------------------
    if cfg.make_figures:
        LOGGER.info("Plotting post-filter QC...")
        plot_utils.run_qc_plots_postfilter(adata, cfg)
        plot_utils.plot_cellbender_effects(
            adata,
            batch_key=batch_key,
            figdir=cfg.figdir / "QC_plots",
        )

        plot_utils.plot_final_cell_counts(adata, cfg)

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------
    adata = normalize_and_hvg(adata, cfg.n_top_genes, batch_key)
    adata = pca_neighbors_umap(adata, var_explained=0.85, min_pcs=20, max_pcs=50)
    adata = cluster_unintegrated(adata)

    if cfg.make_figures:
        plot_utils.umap_plots(
            adata,
            batch_key=batch_key,
            figdir=cfg.figdir / "QC_plots" / "overview",
        )

    # ---------------------------------------------------------
    # Save final filtered dataset
    # ---------------------------------------------------------
    out_zarr = cfg.output_dir / "adata.filtered.zarr"
    LOGGER.info("Saving filtered dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        out_h5ad = cfg.output_dir / "adata.filtered.h5ad"
        LOGGER.warning("Writing H5AD copy (loads data into RAM).")
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    plot_utils.cleanup_scanpy_tmp()

    LOGGER.info("Finished load-and-filter")
    return adata
