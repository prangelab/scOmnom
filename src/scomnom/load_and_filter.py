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

from scomnom import __version__
from . import io_utils
from . import plot_utils
from . import reporting

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

    X = adata.X  # assumed CSR

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
        X = adata.X

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
            X = adata.X

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
            X = adata.X

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
# SOLO (always global)
# ---------------------------------------------------------------------
def run_solo_with_scvi(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
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
        filtered_sample_map, qc_df = _per_sample_qc_and_filter(sample_map, cfg, qc_filter_rows)

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

        # Write QC fitlers
        if qc_filter_rows:
            adata.uns["qc_filter_stats"] = pd.DataFrame(qc_filter_rows)

        # ---------------------------------------------------------
        # SOLO doublet detection (GLOBAL, RAW COUNTS)
        # ---------------------------------------------------------
        adata = run_solo_with_scvi(
            adata,
            batch_key=cfg.batch_key,
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

    call_doublets(
        adata,
        batch_key=cfg.batch_key or "sample_id",
        expected_doublet_rate=cfg.expected_doublet_rate,
        out_stats=cfg.output_dir / "doublets_per_sample.tsv",
    )

    if cfg.make_figures:
        plot_utils.doublet_plots(
            adata,
            batch_key=batch_key,
            figdir=Path("QC_plots") / "doublets",
        )

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
        plot_utils.run_qc_plots_postfilter(adata, cfg)
        plot_utils.plot_cellbender_effects(
            adata,
            batch_key=batch_key,
            figdir=Path("QC_plots") / "cellbender",
        )

        plot_utils.plot_final_cell_counts(adata, cfg)
        plot_utils.plot_qc_filter_stack(
            adata,
            figdir=Path("QC_plots") / "overview",
        )

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
            figdir=Path("QC_plots") / "overview",
        )
        reporting.generate_qc_report(
            fig_root=cfg.output_dir / cfg.figdir_name,
            cfg=cfg,
            version=__version__,
            adata=adata,
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

    LOGGER.info("Finished load-and-filter")
    return adata
