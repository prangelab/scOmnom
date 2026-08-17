# src/scomnom/load_and_filter.py

from __future__ import annotations

import logging
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import torch

import anndata as ad
import pandas as pd
import scanpy as sc

from scomnom import __version__
from . import io_utils
from . import plot_utils
from . import reporting
from .adata_ops import add_obs_metadata

LOGGER = logging.getLogger(__name__)


def _dataset_output_stem(output_name: str) -> str:
    stem = str(output_name).strip()
    for suffix in (".zarr.tar.zst", ".zarr", ".h5ad"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError("output_name must not be empty.")
    return stem


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
    _ = add_obs_metadata(
        adata,
        metadata_tsv,
        metadata_key=sample_id_col,
        obs_key=sample_id_col,
        columns=(),
        overwrite=True,
        require_exact_match=True,
        require_non_missing_values=False,
    )
    return adata


add_metadata = _add_metadata


def _per_sample_qc_and_filter(
    sample_map: Dict[str, ad.AnnData],
    cfg: LoadAndQCConfig,
    qc_filter_rows: list[Dict],
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
            min_counts=cfg.min_counts,
            min_counts_mad=cfg.min_counts_mad,
            min_counts_quantile=cfg.min_counts_quantile,
            min_counts_auto_activate_quantile=cfg.min_counts_auto_activate_quantile,
            min_counts_auto_activate_below=cfg.min_counts_auto_activate_below,
            max_pct_mt=cfg.max_pct_mt,
            max_genes_mad=cfg.max_genes_mad,
            max_genes_quantile=cfg.max_genes_quantile,
            max_counts_mad=cfg.max_counts_mad,
            max_counts_quantile=cfg.max_counts_quantile,
            qc_rows=qc_filter_rows,
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
    import numpy as np

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
def derive_lower_count_cutoff(
    total_counts,
    *,
    min_counts: int | None = None,
    min_counts_mad: float | None = 5.0,
    min_counts_quantile: float | None = None,
    min_counts_auto_activate_quantile: float | None = 0.01,
    min_counts_auto_activate_below: int | None = 1000,
) -> int | None:
    import numpy as np

    values = np.asarray(total_counts)
    cuts: list[float] = []

    if min_counts_mad is not None:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        cuts.append(med - min_counts_mad * mad)

    if min_counts_quantile is not None:
        cuts.append(np.quantile(values, min_counts_quantile))

    auto_cutoff = None
    if cuts:
        auto_cutoff = int(np.ceil(max(max(cuts), 0.0)))

    auto_active = True
    if (
        auto_cutoff is not None
        and min_counts_auto_activate_quantile is not None
        and min_counts_auto_activate_below is not None
    ):
        activation_value = float(
            np.quantile(values, min_counts_auto_activate_quantile)
        )
        auto_active = activation_value < float(min_counts_auto_activate_below)

    if not auto_active:
        auto_cutoff = None

    active_cuts = [cut for cut in (min_counts, auto_cutoff) if cut is not None]
    if not active_cuts:
        return None

    return int(max(active_cuts))


def sparse_filter_cells_and_genes(
    adata: ad.AnnData,
    *,
    min_genes: int,
    min_cells: int,
    min_counts: int | None = None,
    min_counts_mad: float | None = 5.0,
    min_counts_quantile: float | None = None,
    min_counts_auto_activate_quantile: float | None = 0.01,
    min_counts_auto_activate_below: int | None = 1000,
    max_pct_mt: float | None = None,
    # --- new: upper-cut filtering ---
    max_genes_mad: float | None = None,
    max_genes_quantile: float | None = None,
    max_counts_mad: float | None = None,
    max_counts_quantile: float | None = None,
    batch_key: str | None = "sample_id",
    qc_rows: list[Dict] | None = None,
) -> ad.AnnData:
    import numpy as np
    import pandas as pd
    from scipy import sparse

    def _ensure_csr(in_adata: ad.AnnData):
        X_local = in_adata.X
        if sparse.issparse(X_local):
            X_local = X_local.tocsr()
        else:
            X_local = sparse.csr_matrix(X_local)
        in_adata.X = X_local
        return X_local

    X = _ensure_csr(adata)

    # --------------------------------------------------
    # QC logging helpers
    # --------------------------------------------------
    if qc_rows is None:
        qc_rows = []

    def _log_cell_filter(
        *,
        filter_name: str,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        adata_obs,
    ):
        # global
        qc_rows.append(
            {
                "filter": filter_name,
                "scope": "cell",
                "batch": "ALL",
                "n_before": int(before_mask.sum()),
                "n_after": int(after_mask.sum()),
                "n_removed": int(before_mask.sum() - after_mask.sum()),
                "frac_removed": float(
                    (before_mask.sum() - after_mask.sum()) / before_mask.sum()
                ),
            }
        )

        # per batch
        if batch_key is not None and batch_key in adata_obs:
            for batch in adata_obs[batch_key].unique():
                bmask = adata_obs[batch_key].to_numpy() == batch
                nb = (before_mask & bmask).sum()
                na = (after_mask & bmask).sum()
                if nb == 0:
                    continue

                qc_rows.append(
                    {
                        "filter": filter_name,
                        "scope": "cell",
                        "batch": batch,
                        "n_before": int(nb),
                        "n_after": int(na),
                        "n_removed": int(nb - na),
                        "frac_removed": float((nb - na) / nb),
                    }
                )

    # --------------------------------------------------
    # Cell filtering: min_genes
    # --------------------------------------------------
    gene_counts = np.diff(X.indptr)
    before = np.ones(adata.n_obs, dtype=bool)
    cell_mask = gene_counts >= min_genes

    if cell_mask.sum() == 0:
        raise ValueError(f"All cells removed by min_genes={min_genes}.")

    _log_cell_filter(
        filter_name="min_genes",
        before_mask=before,
        after_mask=cell_mask,
        adata_obs=adata.obs,
    )

    adata = adata[cell_mask].copy()
    X = _ensure_csr(adata)

    # --------------------------------------------------
    # Cell filtering: min_counts
    # --------------------------------------------------
    total_counts = np.add.reduceat(X.data, X.indptr[:-1])
    effective_min_counts = derive_lower_count_cutoff(
        total_counts,
        min_counts=min_counts,
        min_counts_mad=min_counts_mad,
        min_counts_quantile=min_counts_quantile,
        min_counts_auto_activate_quantile=min_counts_auto_activate_quantile,
        min_counts_auto_activate_below=min_counts_auto_activate_below,
    )

    if effective_min_counts is not None:
        before = np.ones(adata.n_obs, dtype=bool)
        count_mask = total_counts >= effective_min_counts

        if count_mask.sum() == 0:
            raise ValueError(
                f"All cells removed by min_counts={effective_min_counts}."
            )

        _log_cell_filter(
            filter_name="min_counts",
            before_mask=before,
            after_mask=count_mask,
            adata_obs=adata.obs,
        )

        adata = adata[count_mask].copy()
        X = _ensure_csr(adata)

    # --------------------------------------------------
    # Cell filtering: max_pct_mt
    # --------------------------------------------------
    if max_pct_mt is not None:
        if "pct_counts_mt" not in adata.obs:
            raise KeyError(
                "pct_counts_mt not found in adata.obs. "
                "Run compute_qc_metrics() before sparse filtering."
            )

        before = np.ones(adata.n_obs, dtype=bool)
        mt_mask = adata.obs["pct_counts_mt"].to_numpy() <= max_pct_mt

        if mt_mask.sum() == 0:
            raise ValueError(
                f"All cells removed by max_pct_mt={max_pct_mt}."
            )

        _log_cell_filter(
            filter_name="max_pct_mt",
            before_mask=before,
            after_mask=mt_mask,
            adata_obs=adata.obs,
        )

        adata = adata[mt_mask].copy()
        X = _ensure_csr(adata)

    # --------------------------------------------------
    # Cell filtering: UPPER CUTS (MAD + quantile)
    # --------------------------------------------------
    def _upper_cut(values: np.ndarray, *, k_mad: float | None, q: float | None):
        if k_mad is None and q is None:
            return None

        med = np.median(values)
        cut_mad = None
        if k_mad is not None:
            mad = np.median(np.abs(values - med))
            if mad > 0:
                cut_mad = med + k_mad * mad

        cut_q = np.quantile(values, q) if q is not None else None
        cuts = [c for c in (cut_mad, cut_q) if c is not None]
        return min(cuts) if cuts else None

    # --- n_genes_by_counts upper cut ---
    if max_genes_mad is not None or max_genes_quantile is not None:
        n_genes = np.diff(X.indptr)
        cut = _upper_cut(
            n_genes,
            k_mad=max_genes_mad,
            q=max_genes_quantile,
        )

        if cut is not None:
            before = np.ones(adata.n_obs, dtype=bool)
            keep = n_genes <= cut

            if keep.sum() == 0:
                raise ValueError("All cells removed by max_genes upper-cut.")

            _log_cell_filter(
                filter_name="upper_cut_n_genes",
                before_mask=before,
                after_mask=keep,
                adata_obs=adata.obs,
            )

            adata = adata[keep].copy()
            X = _ensure_csr(adata)

    # --- total_counts upper cut ---
    if max_counts_mad is not None or max_counts_quantile is not None:
        total_counts = np.add.reduceat(X.data, X.indptr[:-1])
        cut = _upper_cut(
            total_counts,
            k_mad=max_counts_mad,
            q=max_counts_quantile,
        )

        if cut is not None:
            before = np.ones(adata.n_obs, dtype=bool)
            keep = total_counts <= cut

            if keep.sum() == 0:
                raise ValueError("All cells removed by max_counts upper-cut.")

            _log_cell_filter(
                filter_name="upper_cut_total_counts",
                before_mask=before,
                after_mask=keep,
                adata_obs=adata.obs,
            )

            adata = adata[keep].copy()
            X = _ensure_csr(adata)

    # --------------------------------------------------
    # Gene filtering: min_cells (not logged here)
    # --------------------------------------------------
    gene_nnz = np.bincount(X.indices, minlength=adata.n_vars)
    gene_mask = gene_nnz >= min_cells
    if gene_mask.sum() == 0:
        raise ValueError(f"All genes removed by min_cells={min_cells}.")
    adata = adata[:, gene_mask].copy()

    return adata


# ---------------------------------------------------------------------
# SOLO score generation
# ---------------------------------------------------------------------
_SOLO_DOUBLET_RATIO = 2


@dataclass(frozen=True)
class SoloScoreBlock:
    name: str
    indices: object
    n_cells: int
    nnz: int
    estimate: int
    batches: tuple[str, ...]
    split: bool = False


def _get_solo_layer(adata: ad.AnnData) -> str | None:
    if "counts_cb" in adata.layers:
        return "counts_cb"
    if "counts_raw" in adata.layers:
        return "counts_raw"
    return None


def _get_solo_matrix(adata: ad.AnnData):
    layer = _get_solo_layer(adata)
    if layer is not None:
        return adata.layers[layer]
    return adata.X


def _solo_row_nnz(adata: ad.AnnData):
    import numpy as np
    from scipy import sparse

    X = _get_solo_matrix(adata)
    if sparse.isspmatrix_csr(X):
        return np.asarray(np.diff(X.indptr), dtype=np.int64)
    if sparse.issparse(X):
        return np.asarray(X.getnnz(axis=1), dtype=np.int64).ravel()
    return np.full(adata.n_obs, adata.n_vars, dtype=np.int64)


def _estimate_solo_sparse_operation_nnz_from_nnz(
    nnz: int,
    *,
    doublet_ratio: int = _SOLO_DOUBLET_RATIO,
) -> int:
    return int(2 * int(doublet_ratio) * int(nnz))


def estimate_solo_sparse_operation_nnz(
    adata: ad.AnnData,
    *,
    doublet_ratio: int = _SOLO_DOUBLET_RATIO,
) -> int:
    matrix = _get_solo_matrix(adata)
    nnz = int(getattr(matrix, "nnz", adata.n_obs * adata.n_vars))
    return _estimate_solo_sparse_operation_nnz_from_nnz(
        nnz,
        doublet_ratio=doublet_ratio,
    )


def _split_indices_for_solo_block(
    indices,
    row_nnz,
    *,
    sparse_nnz_limit: int,
    max_cells_per_block: int | None,
    doublet_ratio: int,
) -> list:
    import numpy as np

    out = []
    current = []
    current_nnz = 0
    max_cells = max_cells_per_block or len(indices)

    for raw_idx in indices:
        idx = int(raw_idx)
        row_count = int(row_nnz[idx])
        row_estimate = _estimate_solo_sparse_operation_nnz_from_nnz(
            row_count,
            doublet_ratio=doublet_ratio,
        )
        if row_estimate > sparse_nnz_limit:
            raise RuntimeError(
                "A single cell is too large for SOLO blocked scoring under "
                f"solo_sparse_nnz_limit={sparse_nnz_limit}."
            )

        next_nnz = current_nnz + row_count
        next_estimate = _estimate_solo_sparse_operation_nnz_from_nnz(
            next_nnz,
            doublet_ratio=doublet_ratio,
        )
        next_too_large = next_estimate > sparse_nnz_limit
        next_too_many_cells = len(current) >= max_cells
        if current and (next_too_large or next_too_many_cells):
            out.append(np.asarray(current, dtype=np.int64))
            current = []
            current_nnz = 0

        current.append(idx)
        current_nnz += row_count

    if current:
        out.append(np.asarray(current, dtype=np.int64))

    return out


def plan_solo_score_blocks(
    adata: ad.AnnData,
    *,
    batch_key: str | None,
    sparse_nnz_limit: int,
    max_cells_per_block: int | None = None,
    doublet_ratio: int = _SOLO_DOUBLET_RATIO,
) -> list[SoloScoreBlock]:
    import numpy as np

    row_nnz = _solo_row_nnz(adata)

    if batch_key is not None and batch_key in adata.obs:
        grouped = adata.obs.groupby(batch_key, observed=True, sort=False).groups
        sample_chunks = []
        for batch, labels in grouped.items():
            positions = adata.obs_names.get_indexer(labels)
            parts = _split_indices_for_solo_block(
                positions,
                row_nnz,
                sparse_nnz_limit=sparse_nnz_limit,
                max_cells_per_block=max_cells_per_block,
                doublet_ratio=doublet_ratio,
            )
            was_split = len(parts) > 1
            for i, part in enumerate(parts, 1):
                sample_chunks.append((str(batch), i, was_split, part))
    else:
        parts = _split_indices_for_solo_block(
            np.arange(adata.n_obs, dtype=np.int64),
            row_nnz,
            sparse_nnz_limit=sparse_nnz_limit,
            max_cells_per_block=max_cells_per_block,
            doublet_ratio=doublet_ratio,
        )
        sample_chunks = [("ALL", i, len(parts) > 1, part) for i, part in enumerate(parts, 1)]

    blocks: list[SoloScoreBlock] = []
    current_parts = []
    current_nnz = 0
    current_batches: list[str] = []
    current_split = False
    max_cells = max_cells_per_block

    def _flush() -> None:
        nonlocal current_parts, current_nnz, current_batches, current_split
        if not current_parts:
            return
        indices = np.concatenate(current_parts)
        name = f"block_{len(blocks) + 1:04d}"
        blocks.append(
            SoloScoreBlock(
                name=name,
                indices=indices,
                n_cells=int(indices.size),
                nnz=int(current_nnz),
                estimate=_estimate_solo_sparse_operation_nnz_from_nnz(
                    current_nnz,
                    doublet_ratio=doublet_ratio,
                ),
                batches=tuple(dict.fromkeys(current_batches)),
                split=bool(current_split),
            )
        )
        current_parts = []
        current_nnz = 0
        current_batches = []
        current_split = False

    for batch, _part_idx, was_split, indices in sample_chunks:
        chunk_nnz = int(row_nnz[indices].sum())
        chunk_estimate = _estimate_solo_sparse_operation_nnz_from_nnz(
            chunk_nnz,
            doublet_ratio=doublet_ratio,
        )
        if chunk_estimate > sparse_nnz_limit:
            raise RuntimeError(
                "SOLO block planner produced an unsafe block estimate "
                f"({chunk_estimate} > {sparse_nnz_limit})."
            )

        next_nnz = current_nnz + chunk_nnz
        next_estimate = _estimate_solo_sparse_operation_nnz_from_nnz(
            next_nnz,
            doublet_ratio=doublet_ratio,
        )
        next_cells = sum(int(x.size) for x in current_parts) + int(indices.size)
        if current_parts and (
            next_estimate > sparse_nnz_limit
            or (max_cells is not None and next_cells > max_cells)
        ):
            _flush()

        current_parts.append(indices)
        current_nnz += chunk_nnz
        current_batches.append(batch)
        current_split = current_split or was_split

    _flush()

    if not blocks:
        raise RuntimeError("SOLO block planner produced no blocks.")

    return blocks


def _solo_block_summary(blocks: list[SoloScoreBlock]) -> list[dict]:
    return [
        {
            "name": block.name,
            "n_cells": block.n_cells,
            "nnz": block.nnz,
            "estimate": block.estimate,
            "batches": list(block.batches),
            "split": block.split,
        }
        for block in blocks
    ]


def _score_solo_with_scvi_model(
    scvi_model,
    adata: ad.AnnData,
    *,
    restrict_to_batch: str | None = None,
    doublet_ratio: int = _SOLO_DOUBLET_RATIO,
):
    from scvi.external import SOLO

    accelerator, devices = _select_device()

    solo = SOLO.from_scvi_model(
        scvi_model,
        adata=adata,
        restrict_to_batch=restrict_to_batch,
        doublet_ratio=doublet_ratio,
    )
    solo.train(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
    )

    probs = solo.predict(soft=True)
    return probs["doublet"].to_numpy()


def run_solo_with_scvi(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    doublet_score_mode: str = "auto",
    solo_sparse_nnz_limit: int = 1_500_000_000,
    solo_max_cells_per_block: int | None = None,
) -> ad.AnnData:
    import numpy as np

    if doublet_score_mode not in {"auto", "global", "blocked"}:
        raise ValueError("doublet_score_mode must be one of: auto, global, blocked")

    layer = _get_solo_layer(adata)
    global_estimate = estimate_solo_sparse_operation_nnz(adata)
    global_is_safe = global_estimate <= int(solo_sparse_nnz_limit)

    if doublet_score_mode == "global" and not global_is_safe:
        raise RuntimeError(
            "Global SOLO scoring is estimated to exceed the configured sparse "
            f"operation limit ({global_estimate} > {solo_sparse_nnz_limit}). "
            "Use --doublet-score-mode auto or --doublet-score-mode blocked."
        )

    effective_mode = "global"
    fallback_reason = None
    if doublet_score_mode == "blocked":
        effective_mode = "blocked"
        fallback_reason = "forced"
    elif doublet_score_mode == "auto" and not global_is_safe:
        effective_mode = "blocked"
        fallback_reason = "estimated_sparse_operation_exceeds_limit"

    LOGGER.info(
        "Running SOLO doublet detection (mode=%s, layer=%s, global_estimate=%d, limit=%d)",
        effective_mode,
        layer or "X",
        global_estimate,
        solo_sparse_nnz_limit,
    )

    scvi_model = _train_scvi(
        adata,
        batch_key=batch_key,
        layer=layer,
        purpose="solo",
    )

    try:
        if effective_mode == "global":
            scores = _score_solo_with_scvi_model(
                scvi_model,
                adata,
                doublet_ratio=_SOLO_DOUBLET_RATIO,
            )
            adata.obs["doublet_score"] = scores
            blocks: list[SoloScoreBlock] = [
                SoloScoreBlock(
                    name="global",
                    indices=np.arange(adata.n_obs, dtype=np.int64),
                    n_cells=int(adata.n_obs),
                    nnz=int(getattr(_get_solo_matrix(adata), "nnz", adata.n_obs * adata.n_vars)),
                    estimate=int(global_estimate),
                    batches=tuple(),
                    split=False,
                )
            ]
        else:
            blocks = plan_solo_score_blocks(
                adata,
                batch_key=batch_key,
                sparse_nnz_limit=int(solo_sparse_nnz_limit),
                max_cells_per_block=solo_max_cells_per_block,
                doublet_ratio=_SOLO_DOUBLET_RATIO,
            )
            LOGGER.info("SOLO blocked scoring will use %d blocks", len(blocks))
            adata.obs["doublet_score"] = np.nan
            for i, block in enumerate(blocks, 1):
                block_adata = adata[block.indices].copy()
                restrict_to_batch = block.batches[0] if len(block.batches) == 1 else None
                LOGGER.info(
                    "SOLO block %d/%d: %s (%d cells, estimate=%d, batches=%s)",
                    i,
                    len(blocks),
                    block.name,
                    block.n_cells,
                    block.estimate,
                    ",".join(block.batches),
                )
                scores = _score_solo_with_scvi_model(
                    scvi_model,
                    block_adata,
                    restrict_to_batch=restrict_to_batch,
                    doublet_ratio=_SOLO_DOUBLET_RATIO,
                )
                adata.obs.loc[block_adata.obs_names, "doublet_score"] = scores

            if adata.obs["doublet_score"].isna().any():
                n_missing = int(adata.obs["doublet_score"].isna().sum())
                raise RuntimeError(f"SOLO blocked scoring left {n_missing} cells without scores.")
    finally:
        del scvi_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adata.uns["solo_scoring"] = {
        "requested_mode": str(doublet_score_mode),
        "mode": effective_mode,
        "layer": layer or "X",
        "doublet_ratio": int(_SOLO_DOUBLET_RATIO),
        "sparse_nnz_limit": int(solo_sparse_nnz_limit),
        "global_nnz_estimate": int(global_estimate),
        "fallback_reason": fallback_reason,
        "blocks": _solo_block_summary(blocks),
    }


    return adata


# ---------------------------------------------------------------------
# Cleanup after SOLO
# ---------------------------------------------------------------------
def cleanup_after_solo(
    adata: ad.AnnData,
    batch_key: str,
    min_cells_per_sample: int,
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

    if "predicted_doublet" in adata.obs:
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()


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

    return adata


from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import logging

LOGGER = logging.getLogger(__name__)


def call_doublets(
    adata: ad.AnnData,
    *,
    batch_key: str = "sample_id",
    expected_doublet_rate: float,
    score_key: str = "doublet_score",
    out_stats: Path | None = None,
) -> None:
    """
    Call doublets PER SAMPLE using a fixed expected rate.

    Threshold per sample is inferred post-hoc as:
      min(doublet_score | predicted_doublet == True)

    This value is the canonical decision boundary and is:
      - written to TSV
      - stored in adata.uns
      - used by QC plots / report
    """

    import numpy as np
    import pandas as pd

    if score_key not in adata.obs:
        raise KeyError(f"Missing {score_key!r} in adata.obs")

    if batch_key not in adata.obs:
        raise KeyError(f"Missing {batch_key!r} in adata.obs")

    if not (0 < expected_doublet_rate < 1):
        raise ValueError("expected_doublet_rate must be in (0, 1)")

    # --------------------------------------------------
    # Initialize output
    # --------------------------------------------------
    adata.obs["predicted_doublet"] = False

    stats_rows = []
    thresholds_per_sample: dict[str, float] = {}

    total_cells = 0
    total_doublets = 0

    # --------------------------------------------------
    # Per-sample calling
    # --------------------------------------------------
    for sample, idx in adata.obs.groupby(batch_key, observed=True).groups.items():
        scores = adata.obs.loc[idx, score_key].to_numpy()
        n = scores.size

        if n == 0:
            continue

        k = int(np.ceil(expected_doublet_rate * n))
        k = max(k, 1)

        order = np.argsort(scores)[::-1]
        selected = order[:k]

        mask = np.zeros(n, dtype=bool)
        mask[selected] = True

        adata.obs.loc[idx, "predicted_doublet"] = mask

        # --- inferred threshold (decision boundary) ---
        thr = float(scores[selected].min())
        thresholds_per_sample[sample] = thr

        n_doublets = int(mask.sum())

        stats_rows.append(
            {
                batch_key: sample,
                "n_cells": n,
                "n_doublets": n_doublets,
                "doublet_rate_observed": float(n_doublets / n),
                "score_threshold_inferred": thr,
            }
        )

        total_cells += n
        total_doublets += n_doublets

    # --------------------------------------------------
    # Attach metadata to AnnData
    # --------------------------------------------------
    adata.uns["doublet_calling"] = {
        "method": "rate_per_batch",
        "batch_key": batch_key,
        "expected_doublet_rate": float(expected_doublet_rate),
        "score_key": score_key,
        "observed_global_rate": (
            float(total_doublets / total_cells) if total_cells > 0 else 0.0
        ),
        "thresholds_per_sample": thresholds_per_sample,
    }

    # --------------------------------------------------
    # Write per-sample stats table
    # --------------------------------------------------
    if out_stats is not None:
        out_stats = Path(out_stats)
        out_stats.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(stats_rows).sort_values(batch_key)
        df.to_csv(out_stats, sep="\t", index=False)

        LOGGER.info(
            "Wrote per-sample doublet statistics to %s (%d samples)",
            out_stats,
            len(df),
        )


def write_qc_filter_stats(
    adata,
    *,
    out_path: Path,
) -> None:
    if "qc_filter_stats" not in adata.uns:
        LOGGER.warning("No qc_filter_stats found in adata.uns; skipping QC table write.")
        return

    df = adata.uns["qc_filter_stats"]

    if not isinstance(df, pd.DataFrame) or df.empty:
        LOGGER.warning("qc_filter_stats is empty; skipping QC table write.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, sep="\t", index=False)

    LOGGER.info(
        "Wrote QC filter statistics to %s (%d rows)",
        out_path,
        len(df),
    )


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


def run_load_and_filter(
    cfg: LoadAndFilterConfig) -> ad.AnnData:

    LOGGER.info("Starting load-and-filter")
    # Configure Scanpy/Matplotlib figure behavior + formats
    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    qc_filter_rows: list[dict] = []

    # If we are only applying a different doublet filter:
    if cfg.apply_doublet_score is True:

        LOGGER.info(
            "Resuming from pre-doublet AnnData: %s",
            cfg.apply_doublet_score,
        )

        adata = io_utils.load_dataset(cfg.apply_doublet_score_path)

        # sanity check: SOLO must already have been run
        required = {"doublet_score"}
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
            with plot_utils.capture_plot_artifacts() as artifacts:
                sample_map, read_counts, _ = io_utils.load_raw_data(
                    cfg,
                    plot_dir=Path("QC_plots") / "cell_qc",
                )
            plot_utils.persist_plot_artifacts(artifacts)

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
        filtered_sample_map, qc_df = _per_sample_qc_and_filter(sample_map, cfg, qc_filter_rows)

        # ---------------------------------------------------------
        # Pre-filter QC plots (lightweight, from qc_df only)
        # ---------------------------------------------------------
        if cfg.make_figures:
            LOGGER.info("Plotting pre-filter QC...")
            artifacts = plot_utils.run_qc_plots_pre_filter_df(qc_df, cfg)
            plot_utils.persist_plot_artifacts(artifacts)

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

        # Write QC fitlers
        if qc_filter_rows:
            adata.uns["qc_filter_stats"] = pd.DataFrame(qc_filter_rows)

        # ---------------------------------------------------------
        # SOLO doublet score generation
        # ---------------------------------------------------------
        adata = run_solo_with_scvi(
            adata,
            batch_key=cfg.batch_key,
            doublet_score_mode=cfg.doublet_score_mode,
            solo_sparse_nnz_limit=cfg.solo_sparse_nnz_limit,
            solo_max_cells_per_block=cfg.solo_max_cells_per_block,
        )

        LOGGER.info("Saving Anndata with doublet scores...")
        pre_path = cfg.output_dir / "adata.merged.zarr"
        io_utils.save_dataset(adata, pre_path, fmt="zarr")

        if cfg.save_h5ad:
            io_utils.save_dataset(
                adata,
                cfg.output_dir / "adata.merged.h5ad",
                fmt="h5ad",
            )

        LOGGER.info("Saved pre-doublet filter AnnData → %s", pre_path)

    # Here 'normal mode' and 'only apply doublet filter' merge again
    batch_key = cfg.batch_key or adata.uns.get("batch_key")

    call_doublets(
        adata,
        batch_key=cfg.batch_key or "sample_id",
        expected_doublet_rate=cfg.expected_doublet_rate,
        out_stats=cfg.output_dir / "doublets_per_sample.tsv",
    )

    if cfg.make_figures:
        artifacts = plot_utils.doublet_plots(
            adata,
            batch_key=batch_key,
            figdir=Path("QC_plots") / "doublets",
        )
        plot_utils.persist_plot_artifacts(artifacts)

    adata = cleanup_after_solo(
        adata,
        batch_key=batch_key,
        min_cells_per_sample=cfg.min_cells_per_sample,
        expected_doublet_rate=cfg.expected_doublet_rate,
    )

    qc_stats_path = cfg.output_dir / "qc_filter_stats.tsv"
    write_qc_filter_stats(adata, out_path=qc_stats_path)

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
        artifacts = []
        artifacts.extend(plot_utils.run_qc_plots_postfilter(adata, cfg))
        artifacts.extend(
            plot_utils.plot_cellbender_effects(
                adata,
                batch_key=batch_key,
                figdir=Path("QC_plots") / "cellbender",
            )
        )
        artifacts.extend(plot_utils.plot_final_cell_counts(adata, cfg))
        artifacts.extend(
            plot_utils.plot_qc_filter_stack(
                adata,
                figdir=Path("QC_plots") / "overview",
            )
        )
        plot_utils.persist_plot_artifacts(artifacts)

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------
    adata = normalize_and_hvg(adata, cfg.n_top_genes, batch_key)
    adata = pca_neighbors_umap(adata, var_explained=0.85, min_pcs=20, max_pcs=50)
    adata = cluster_unintegrated(adata)

    if cfg.make_figures:
        artifacts = plot_utils.umap_plots(
            adata,
            batch_key=batch_key,
            figdir=Path("QC_plots") / "overview",
        )
        plot_utils.persist_plot_artifacts(artifacts)
        for fmt in cfg.figure_formats:
            reporting.generate_qc_report(
                fig_root=Path(cfg.figdir),
                fmt=fmt,
                cfg=cfg,
                version=__version__,
                adata=adata,
            )

    # ---------------------------------------------------------
    # Save final filtered dataset
    # ---------------------------------------------------------
    output_stem = _dataset_output_stem(cfg.output_name)
    out_zarr = cfg.output_dir / f"{output_stem}.zarr"
    LOGGER.info("Saving filtered dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if cfg.save_h5ad:
        out_h5ad = cfg.output_dir / f"{output_stem}.h5ad"
        LOGGER.warning("Writing H5AD copy (loads data into RAM).")
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished load-and-filter")
    return adata
