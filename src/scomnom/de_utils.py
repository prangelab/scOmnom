# src/scomnom/de_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set
    pass


# -----------------------------------------------------------------------------
# Public API (notebook-first)
# -----------------------------------------------------------------------------
# Design goals:
# - Works on an already-loaded AnnData (no disk IO, no zarr discovery).
# - OOM-safe for 1M+ cells by:
#   * never looping with adata[mask].copy() over many clusters
#   * aggregating via sparse indicator matrices (G.T @ X)
#   * downsampling only for cell-level marker calling
# - Pseudobulk DE uses PyDESeq2 only (no rpy2 path).
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudobulkSpec:
    """Minimal schema describing where to find counts & keys in AnnData."""
    sample_key: str = "sample_id"
    counts_layer: Optional[str] = "counts_raw"  # or "counts_cb" / None (falls back to .X)
    # For DE you usually want raw counts. Layer is preferred over X.
    # If counts_layer is None, .X is used (must be raw counts).


@dataclass(frozen=True)
class CellLevelMarkerSpec:
    """Options for cell-level marker calling (discovery only)."""
    method: Literal["wilcoxon", "t-test", "logreg"] = "wilcoxon"
    n_genes: int = 100
    use_raw: bool = False
    layer: Optional[str] = None
    rankby_abs: bool = True
    max_cells_per_group: int = 2000  # stratified downsample per group if dataset is huge
    random_state: int = 0
    min_pct: float = 0.0
    min_diff_pct: float = 0.0
    positive_only: bool = True

@dataclass(frozen=True)
class PseudobulkDEOptions:
    min_cells_per_sample_group: int = 20
    min_samples_per_level: int = 2
    alpha: float = 0.05
    lfc_threshold: float = 0.0
    size_factors: str = "poscounts"
    shrink_lfc: bool = True
    min_total_counts: int = 10
    min_pct: float = 0.0
    min_diff_pct: float = 0.0
    positive_only: bool = True


def _pydeseq2_cluster_worker(payload: dict) -> tuple[str, pd.DataFrame, dict]:
    """
    Worker: run PyDESeq2 for a single cluster.
    Returns (cluster_id, result_df, summary_meta_updates)
    """
    cl = payload["cluster"]
    counts_np = payload["counts_np"]          # 2*n_samples x n_genes
    pb_index = payload["pb_index"]            # list[str]
    genes = payload["genes"]                  # list[str]
    metadata_rows = payload["metadata_rows"]  # list[dict]
    alpha = float(payload["alpha"])
    shrink_lfc = bool(payload["shrink_lfc"])
    positive_only = bool(payload["positive_only"])
    n_cpus = int(payload["n_cpus"])

    # NEW: size factor policy (default poscounts)
    size_factors = str(payload.get("size_factors", "poscounts"))

    metadata = pd.DataFrame(metadata_rows, index=pd.Index(pb_index, name="pb_id"))
    metadata["sample"] = metadata["sample"].astype("category")
    metadata["binary_cluster"] = pd.Categorical(metadata["binary_cluster"], categories=["rest", "target"])

    counts = pd.DataFrame(counts_np, index=metadata.index, columns=pd.Index(genes, name="gene"))

    try:
        res, meta = _run_pydeseq2(
            counts,
            metadata,
            design_factors=["sample", "binary_cluster"],
            contrast=("binary_cluster", "target", "rest"),
            alpha=alpha,
            shrink_lfc=shrink_lfc,
            n_cpus=n_cpus,
            size_factors=size_factors,
        )

        if positive_only and not res.empty and "log2FoldChange" in res.columns:
            res = res.loc[pd.to_numeric(res["log2FoldChange"], errors="coerce") > 0].copy()

        n_sig = int((pd.to_numeric(res["padj"], errors="coerce") < alpha).sum()) if not res.empty else 0

        # return enriched meta for summary table
        return cl, res, {
            "status": "ok",
            "n_sig": n_sig,
            "sf_policy": meta.get("sf_policy"),
            "sf_used": meta.get("sf_used"),
            "sf_forced": meta.get("sf_forced"),
            "warn_iterative_size_factors": meta.get("warn_iterative_size_factors"),
            "warn_low_df_dispersion": meta.get("warn_low_df_dispersion"),
        }

    except Exception as e:
        empty = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
        return cl, empty, {"status": "failed", "reason": str(e), "sf_policy": size_factors}


# -----------------------------------------------------------------------------
# Round-aware label resolution
# -----------------------------------------------------------------------------
def resolve_group_key(
    adata: ad.AnnData,
    *,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    prefer_pretty: bool = True,
) -> str:
    """
    Resolve which obs column to use as the grouping labels.

    Rules:
      1) If groupby is provided and exists -> use it.
      2) Else, if prefer_pretty and round_id can be resolved, use the round's pretty_cluster_key.
      3) Else, fall back to adata.uns['cluster_and_annotate'] pointers if present.
      4) Else, fall back to 'leiden' if present.
    """
    if groupby:
        if groupby not in adata.obs:
            raise KeyError(f"groupby={groupby!r} not in adata.obs")
        return groupby

    # Resolve round_id (explicit -> active -> None)
    rid = round_id
    if rid is None:
        rid0 = adata.uns.get("active_cluster_round", None)
        rid = str(rid0) if rid0 else None

    if prefer_pretty and rid is not None:
        rounds = adata.uns.get("cluster_rounds", {})
        if isinstance(rounds, dict) and rid in rounds:
            ann = rounds[rid].get("annotation", {}) if isinstance(rounds[rid], dict) else {}
            if isinstance(ann, dict):
                pretty = ann.get("pretty_cluster_key", None)
                if pretty and str(pretty) in adata.obs:
                    return str(pretty)

    ca = adata.uns.get("cluster_and_annotate", {})
    if isinstance(ca, dict):
        k = ca.get("cluster_label_key", None)
        if k and str(k) in adata.obs:
            return str(k)

    if "leiden" in adata.obs:
        return "leiden"

    raise RuntimeError(
        "Could not resolve group labels. Provide groupby=... or ensure clustering/annotation keys exist."
    )


# -----------------------------------------------------------------------------
# Counts access helpers
# -----------------------------------------------------------------------------

def _get_counts_matrix(
    adata: ad.AnnData,
    *,
    counts_layer: Optional[str],
) -> sp.csr_matrix:
    """
    Return counts matrix as CSR (cells x genes).
    Never densifies.
    """
    X = None
    if counts_layer:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"counts_layer={counts_layer!r} not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        X = adata.layers[counts_layer]
    else:
        X = adata.X

    if X is None:
        raise RuntimeError("Counts matrix is None (no .X and no counts layer).")

    if sp.issparse(X):
        return X.tocsr()
    # If user passed dense counts, convert (may be memory heavy; warn loudly)
    LOGGER.warning("Counts matrix is dense; converting to CSR (may use a lot of RAM).")
    return sp.csr_matrix(np.asarray(X))

def _pick_markers_layer(adata: ad.AnnData, requested_layer: Optional[str], *, use_raw: bool) -> Optional[str]:
    """
    Policy:
      - If use_raw=True: layer is irrelevant (Scanpy will use adata.raw); return requested_layer unchanged.
      - Else if requested_layer is explicitly provided: use it (must exist), do not override.
      - Else: prefer counts_cb, then counts_raw, else None -> falls back to adata.X
    """
    if use_raw:
        return requested_layer

    if requested_layer is not None:
        if requested_layer not in adata.layers:
            raise KeyError(
                f"markers layer={requested_layer!r} not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        return requested_layer

    if "counts_cb" in adata.layers:
        return "counts_cb"
    if "counts_raw" in adata.layers:
        return "counts_raw"
    return None


def _rebuild_rank_genes_groups_from_filtered(
    adata_run: ad.AnnData,
    *,
    key: str,
    filtered_by_group: dict[str, pd.DataFrame],
    n_genes: int,
) -> None:
    """
    Overwrite adata_run.uns[key] canonical rank_genes_groups arrays
    using already-filtered per-group DataFrames.

    This is the key to making plots/exports that rely on Scanpy's
    uns['rank_genes_groups'] see ONLY the filtered (and positive-only) hits.
    """
    if key not in adata_run.uns:
        return

    groups = list(adata_run.uns[key]["names"].dtype.names)  # scanpy's group names

    # Helper to allocate a structured 1D array of length n_genes
    def _alloc_struct(dtype_kind: str):
        # dtype_kind: "str" or "float"
        if dtype_kind == "str":
            dt = [(g, object) for g in groups]
            arr = np.empty((n_genes,), dtype=dt)
            for g in groups:
                arr[g] = ""
            return arr
        else:
            dt = [(g, np.float64) for g in groups]
            arr = np.empty((n_genes,), dtype=dt)
            for g in groups:
                arr[g] = np.nan
            return arr

    names = _alloc_struct("str")
    scores = _alloc_struct("float")
    logfcs = _alloc_struct("float")
    pvals = _alloc_struct("float")
    pvals_adj = _alloc_struct("float")
    pts = _alloc_struct("float")
    pts_rest = _alloc_struct("float")

    # Fill from filtered dfs
    for g in groups:
        df = filtered_by_group.get(str(g), None)
        if df is None or df.empty:
            continue

        # Take top n_genes after filtering
        d = df.copy()

        # Standardize column presence (rank_genes_groups_df varies)
        if "gene" not in d.columns and "names" in d.columns:
            d = d.rename(columns={"names": "gene"})

        # Some columns may be absent depending on method
        col_gene = "gene"
        col_score = "scores" if "scores" in d.columns else ("score" if "score" in d.columns else None)
        col_logfc = "logfoldchanges" if "logfoldchanges" in d.columns else None
        col_pval = "pvals" if "pvals" in d.columns else None
        col_padj = "pvals_adj" if "pvals_adj" in d.columns else None

        d = _coerce_pts_columns(d)
        d = d.head(int(n_genes))

        k = min(int(n_genes), int(d.shape[0]))
        if k <= 0:
            continue

        # Write values into the structured arrays
        names[g][:k] = d[col_gene].astype(str).to_numpy()[:k]

        if col_score is not None:
            scores[g][:k] = pd.to_numeric(d[col_score], errors="coerce").to_numpy()[:k]

        if col_logfc is not None:
            logfcs[g][:k] = pd.to_numeric(d[col_logfc], errors="coerce").to_numpy()[:k]

        if col_pval is not None:
            pvals[g][:k] = pd.to_numeric(d[col_pval], errors="coerce").to_numpy()[:k]

        if col_padj is not None:
            pvals_adj[g][:k] = pd.to_numeric(d[col_padj], errors="coerce").to_numpy()[:k]

        pts[g][:k] = pd.to_numeric(d["pts"], errors="coerce").to_numpy()[:k]
        pts_rest[g][:k] = pd.to_numeric(d["pts_rest"], errors="coerce").to_numpy()[:k]

    # Overwrite canonical fields Scanpy consumers use
    adata_run.uns[key]["names"] = names
    adata_run.uns[key]["scores"] = scores
    adata_run.uns[key]["logfoldchanges"] = logfcs
    adata_run.uns[key]["pvals"] = pvals
    adata_run.uns[key]["pvals_adj"] = pvals_adj
    adata_run.uns[key]["pts"] = pts
    adata_run.uns[key]["pts_rest"] = pts_rest


def _compute_cluster_parallelism(
    *,
    n_clusters: int,
    total_cpus: int,
) -> tuple[int, int]:
    """
    Decide (n_jobs, n_cpus_per_job) for cluster-wise pseudobulk DE.

    Rules:
      - If total_cpus <= n_clusters:
          n_jobs = total_cpus
          n_cpus = 1
      - Else:
          n_jobs = n_clusters
          n_cpus = 1 + floor((total_cpus - n_clusters) / n_clusters)
    """
    total_cpus = int(max(1, total_cpus))
    n_clusters = int(max(1, n_clusters))

    if total_cpus <= n_clusters:
        return total_cpus, 1

    extra = total_cpus - n_clusters
    n_cpus = 1 + (extra // n_clusters)
    return n_clusters, n_cpus


# -----------------------------------------------------------------------------
# OOM-safe pseudobulk aggregation (core engine)
# -----------------------------------------------------------------------------
def pseudobulk_aggregate(
    adata: ad.AnnData,
    *,
    sample_key: str,
    group_key: Optional[str] = None,
    counts_layer: Optional[str] = "counts_raw",
    min_cells_per_sample_group: int = 1,
    restrict_cells_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate raw counts into pseudobulk libraries.

    Returns:
      counts_df: DataFrame (n_pseudobulk_libs x n_genes) integer counts
      meta_df:   DataFrame (n_pseudobulk_libs x columns) with:
                sample_key, group_key (if provided), n_cells, pb_id

    This is OOM-safe for huge n_cells because it uses sparse matrix ops:
      PB = G.T @ X

    Notes:
      - If group_key is None, aggregation is by sample only.
      - min_cells_per_sample_group filters out tiny libraries *before* returning.
      - restrict_cells_mask optionally aggregates only a subset of cells (e.g. within a cluster).
    """
    if sample_key not in adata.obs:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")

    X = _get_counts_matrix(adata, counts_layer=counts_layer)
    n_cells, n_genes = X.shape

    if restrict_cells_mask is not None:
        restrict_cells_mask = np.asarray(restrict_cells_mask, dtype=bool)
        if restrict_cells_mask.shape != (n_cells,):
            raise ValueError("restrict_cells_mask has wrong shape")
        cell_idx = np.where(restrict_cells_mask)[0]
        if cell_idx.size == 0:
            # empty aggregation
            counts_df = pd.DataFrame(index=pd.Index([], name="pb_id"), columns=adata.var_names)
            meta_df = pd.DataFrame(index=pd.Index([], name="pb_id"))
            return counts_df, meta_df
        X = X[cell_idx, :]
        obs = adata.obs.iloc[cell_idx].copy()
    else:
        obs = adata.obs

    s = obs[sample_key].astype(str).to_numpy()

    if group_key is None:
        g = np.array(["ALL"] * obs.shape[0], dtype=object)
        group_key_out = None
    else:
        if group_key not in obs:
            raise KeyError(f"group_key={group_key!r} not in adata.obs")
        g = obs[group_key].astype(str).to_numpy()
        group_key_out = group_key

    # Build unique (sample, group) library ids
    # Use categorical codes to avoid slow python tuples on 1M cells.
    df_keys = pd.DataFrame({"sample": s, "group": g})
    # factorize the combined key
    combo = pd.Index(df_keys["sample"] + "||" + df_keys["group"])
    lib_codes, lib_uniques = pd.factorize(combo, sort=True)
    n_libs = int(lib_uniques.size)

    # Indicator matrix G: (cells x libs)
    rows = np.arange(obs.shape[0], dtype=np.int64)
    cols = lib_codes.astype(np.int64, copy=False)
    data = np.ones(rows.shape[0], dtype=np.int8)
    G = sp.csr_matrix((data, (rows, cols)), shape=(obs.shape[0], n_libs))

    # PB counts: (libs x genes)
    PB = (G.T @ X).tocsr()

    # n_cells per library
    n_cells_lib = np.asarray(G.sum(axis=0)).ravel().astype(int)

    # Library metadata
    lib_sample = np.array([u.split("||", 1)[0] for u in lib_uniques], dtype=object)
    lib_group = np.array([u.split("||", 1)[1] for u in lib_uniques], dtype=object)
    pb_id = pd.Index([f"{a}|{b}" for a, b in zip(lib_sample, lib_group)], name="pb_id")

    meta_df = pd.DataFrame(
        {
            sample_key: lib_sample,
            **({group_key_out: lib_group} if group_key_out else {}),
            "n_cells": n_cells_lib,
        },
        index=pb_id,
    )

    # Filter tiny libraries
    keep = meta_df["n_cells"].to_numpy() >= int(min_cells_per_sample_group)
    meta_df = meta_df.loc[keep].copy()
    PB = PB[keep, :]

    # Convert to pandas DataFrame (counts must be integer-ish)
    # PB is libs x genes; keep as int64 where possible.
    counts_df = pd.DataFrame.sparse.from_spmatrix(PB, index=meta_df.index, columns=adata.var_names)
    # Many downstream tools want dense ints; for DE we’ll densify per contrast (small #libs),
    # not here globally.
    return counts_df, meta_df


# -----------------------------------------------------------------------------
# Pseudobulk DE (PyDESeq2 only)
# -----------------------------------------------------------------------------
def _require_pydeseq2():
    try:
        import pydeseq2  # noqa: F401
    except Exception as e:
        raise ImportError(
            "PyDESeq2 is required for pseudobulk DE in scomnom. "
            "Install it (and its deps) in your environment."
        ) from e


def _run_pydeseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    design_factors: Sequence[str],
    contrast: Tuple[str, str, str],
    alpha: float = 0.05,
    shrink_lfc: bool = True,
    n_cpus: int = 1,
    size_factors: str = "poscounts",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run PyDESeq2 for one contrast.

    Returns: (results_df, meta)
      meta includes size factor mode actually used and captured warnings.
    """
    _require_pydeseq2()
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    # Defensive copies, ensure alignment
    counts = counts.loc[metadata.index]
    counts_i = counts.round().astype(np.int64)

    dds = DeseqDataSet(
        counts=counts_i,
        metadata=metadata.copy(),
        design_factors=list(design_factors),
        ref_level={contrast[0]: contrast[2]},
        n_cpus=int(n_cpus),
    )

    meta: Dict[str, Any] = {
        "sf_policy": str(size_factors),
        "sf_used": None,
        "sf_forced": False,
        "warn_iterative_size_factors": False,
        "warn_low_df_dispersion": False,
        "warnings": [],
    }

    # Capture warnings for provenance/debug
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        # ---- Force size factor estimation if supported
        # Prefer poscounts for sparse pseudobulk.
        def _try_fit_size_factors(mode: str) -> bool:
            try:
                fit_fn = getattr(dds, "fit_size_factors", None)
                if callable(fit_fn):
                    fit_fn(fit_type=str(mode))
                    meta["sf_used"] = str(mode)
                    meta["sf_forced"] = True
                    return True
                return False
            except Exception:
                return False

        sf = str(size_factors).lower().strip()

        if sf == "ratio_then_poscounts":
            ok = _try_fit_size_factors("ratio")
            if not ok:
                ok2 = _try_fit_size_factors("poscounts")
                if not ok2:
                    meta["sf_used"] = None
            else:
                meta["sf_used"] = "ratio"
        elif sf in ("poscounts", "ratio"):
            ok = _try_fit_size_factors(sf)
            if not ok:
                meta["sf_used"] = None
        else:
            # unknown policy -> do nothing, let pydeseq2 decide
            meta["sf_used"] = None

        # ---- Run DESeq2
        dds.deseq2()

        stat = DeseqStats(
            dds,
            contrast=list(contrast),
            alpha=float(alpha),
            n_cpus=int(n_cpus),
        )
        stat.summary(print_result=False)

        # Best-effort LFC shrinkage
        if shrink_lfc:
            try:
                stat.lfc_shrink()
            except Exception:
                pass

    # Process warnings after run
    for ww in wrec:
        msg = str(getattr(ww, "message", ww))
        meta["warnings"].append(msg)

        # Your two notable classes of warnings:
        if "Iterative size factor fitting did not converge" in msg:
            meta["warn_iterative_size_factors"] = True
        if "residual degrees of freedom is less than 3" in msg:
            meta["warn_low_df_dispersion"] = True

    # Extract results
    res = stat.results_df.copy()
    if "gene" not in res.columns:
        res["gene"] = res.index.astype(str)

    cols = [c for c in ["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"] if c in res.columns]
    res = res[cols].sort_values("padj", na_position="last")

    return res, meta



def de_cluster_vs_rest_pseudobulk(
    adata: ad.AnnData,
    *,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    spec: PseudobulkSpec = PseudobulkSpec(),
    opts: PseudobulkDEOptions = PseudobulkDEOptions(),
    store_key: Optional[str] = "scomnom_de",
    store: bool = True,
    n_jobs: int = 1,
) -> Dict[str, pd.DataFrame]:
    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    sample_key = spec.sample_key
    counts_layer = spec.counts_layer

    if sample_key not in adata.obs:
        raise KeyError(f"sample_key={sample_key!r} not in adata.obs")

    # 1) Aggregate per (sample, cluster)
    counts_sc, meta_sc = pseudobulk_aggregate(
        adata,
        sample_key=sample_key,
        group_key=group_key,
        counts_layer=counts_layer,
        min_cells_per_sample_group=1,
    )
    # 2) Aggregate per sample (totals)
    counts_s, meta_s = pseudobulk_aggregate(
        adata,
        sample_key=sample_key,
        group_key=None,
        counts_layer=counts_layer,
        min_cells_per_sample_group=1,
    )

    meta_sc2 = meta_sc.copy()
    meta_sc2["__sample"] = meta_sc2[sample_key].astype(str)
    meta_sc2["__group"] = meta_sc2[group_key].astype(str)
    meta_sc2["__key"] = meta_sc2["__sample"] + "||" + meta_sc2["__group"]
    sc_row_by_key = pd.Series(meta_sc2.index.to_numpy(), index=meta_sc2["__key"].to_numpy())

    meta_s2 = meta_s.copy()
    meta_s2["__sample"] = meta_s2[sample_key].astype(str)
    s_row_by_sample = pd.Series(meta_s2.index.to_numpy(), index=meta_s2["__sample"].to_numpy())

    clusters = pd.Index(pd.unique(meta_sc[group_key].astype(str))).sort_values()
    n_clusters = int(len(clusters))

    # --- smart core distribution ---
    n_jobs_eff, n_cpus_eff = _compute_cluster_parallelism(
        n_clusters=n_clusters,
        total_cpus=int(n_jobs),  # cfg.n_jobs passed through
    )

    LOGGER.info(
        "Pseudobulk DE parallelism: n_clusters=%d, total_cpus=%d → n_jobs=%d, n_cpus_per_job=%d",
        n_clusters,
        int(n_jobs),
        int(n_jobs_eff),
        int(n_cpus_eff),
    )

    results: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    min_pct = float(getattr(opts, "min_pct", 0.0))
    min_diff_pct = float(getattr(opts, "min_diff_pct", 0.0))
    min_total = int(getattr(opts, "min_total_counts", 10))

    # We build per-cluster payloads in the parent process (small) and then run PyDESeq2 in workers.
    payloads: list[dict] = []

    for cl in clusters:
        m_cl = meta_sc2["__group"].to_numpy() == str(cl)
        meta_cl = meta_sc2.loc[m_cl, [sample_key, group_key, "n_cells"]].copy()
        meta_cl = meta_cl.loc[meta_cl["n_cells"] >= int(opts.min_cells_per_sample_group)]
        n_target_samples = int(meta_cl.shape[0])

        if n_target_samples < int(opts.min_samples_per_level):
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "skipped",
                    "reason": f"only {n_target_samples} sample(s) with >= {opts.min_cells_per_sample_group} cells in target",
                    "n_target_samples": n_target_samples,
                    "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
                }
            )
            continue

        samples = np.unique(meta_cl[sample_key].astype(str).to_numpy())

        pb_index: list[str] = []
        metadata_rows: list[dict] = []
        target_rows: list[str] = []
        total_rows: list[str] = []

        for s in samples:
            key = f"{s}||{cl}"
            if key not in sc_row_by_key.index:
                continue
            rid_target = sc_row_by_key.loc[key]
            rid_total = s_row_by_sample.loc[s] if s in s_row_by_sample.index else None
            if rid_total is None:
                continue

            pb_index.append(f"{s}|{cl}|target")
            metadata_rows.append({"sample": str(s), "binary_cluster": "target"})
            target_rows.append(rid_target)
            total_rows.append(rid_total)

            pb_index.append(f"{s}|{cl}|rest")
            metadata_rows.append({"sample": str(s), "binary_cluster": "rest"})
            target_rows.append(rid_target)
            total_rows.append(rid_total)

        if len(pb_index) < 2 * int(opts.min_samples_per_level):
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "skipped",
                    "reason": "paired library construction failed",
                    "n_target_samples": n_target_samples,
                }
            )
            continue

        # densify only the needed rows (still small)
        target_mat = counts_sc.loc[pd.Index(target_rows)].sparse.to_coo().toarray()
        total_mat = counts_s.loc[pd.Index(total_rows)].sparse.to_coo().toarray()

        out = np.zeros_like(total_mat, dtype=np.int64)
        for i in range(out.shape[0]):
            if i % 2 == 0:
                out[i, :] = target_mat[i, :]
            else:
                v = total_mat[i, :] - target_mat[i, :]
                v[v < 0] = 0
                out[i, :] = v.astype(np.int64, copy=False)

        # gene prefilter: total counts
        keep_total = (out.sum(axis=0) >= min_total)
        if not np.any(keep_total):
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append({"cluster": str(cl), "status": "skipped", "reason": "no genes left after filtering"})
            continue

        out2 = out[:, keep_total]
        genes2 = adata.var_names.to_numpy()[keep_total].astype(str).tolist()

        # min_pct / min_diff_pct prevalence filter (library prevalence)
        metadata_tmp = pd.DataFrame(metadata_rows, index=pd.Index(pb_index, name="pb_id"))
        counts_tmp = pd.DataFrame(out2, index=metadata_tmp.index, columns=genes2)

        counts_tmp, prev_meta = _apply_min_pct_filters_pseudobulk(
            counts_tmp,
            labels=metadata_tmp["binary_cluster"],
            level_A="target",
            level_B="rest",
            min_pct=min_pct,
            min_diff_pct=min_diff_pct,
        )
        if counts_tmp.shape[1] == 0:
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "skipped",
                    "reason": "no genes left after min_pct/min_diff_pct filtering",
                    **prev_meta,
                }
            )
            continue

        vc = metadata_tmp["binary_cluster"].value_counts()
        if int(vc.get("target", 0)) < int(opts.min_samples_per_level) or int(vc.get("rest", 0)) < int(opts.min_samples_per_level):
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "skipped",
                    "reason": "insufficient libraries per level after filtering",
                    "n_target_libs": int(vc.get("target", 0)),
                    "n_rest_libs": int(vc.get("rest", 0)),
                }
            )
            continue

        # payload for worker (numpy + lists)
        payloads.append(
            {
                "cluster": str(cl),
                "counts_np": counts_tmp.to_numpy(dtype=np.int64, copy=False),
                "pb_index": counts_tmp.index.astype(str).tolist(),
                "size_factors": "poscounts",
                "genes": counts_tmp.columns.astype(str).tolist(),
                "metadata_rows": metadata_rows,
                "alpha": float(opts.alpha),
                "shrink_lfc": bool(opts.shrink_lfc),
                "positive_only": bool(opts.positive_only),
                "n_cpus": int(n_cpus_eff),
            }
        )

        # record partial summary now; finalize after worker returns
        summary_rows.append(
            {
                "cluster": str(cl),
                "status": "queued" if int(n_jobs) > 1 else "running",
                "n_target_samples": int(len(samples)),
                "n_libraries": int(counts_tmp.shape[0]),
                "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
                **prev_meta,
            }
        )

    # --- execute: serial or parallel ---
    import time
    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()
    total = int(len(payloads))
    if total == 0:
        LOGGER.info("Pseudobulk DE: no payloads queued (all clusters skipped earlier).")
    elif int(n_jobs) <= 1 or total <= 1:
        LOGGER.info("Pseudobulk DE: running serially (payloads=%d).", total)

        for i, p in enumerate(payloads, start=1):
            cl = str(p.get("cluster", ""))
            t_cl0 = time.perf_counter()

            # a little context up front
            n_libs = int(len(p.get("pb_index", [])))
            n_genes = int(len(p.get("genes", [])))
            LOGGER.info(
                "PB DE [%d/%d] start cluster=%s (libs=%d, genes=%d, n_cpus=%s)",
                i, total, cl, n_libs, n_genes, p.get("n_cpus", "?"),
            )

            cl2, res, meta_upd = _pydeseq2_cluster_worker(p)
            dt = time.perf_counter() - t_cl0

            results[cl2] = res

            # update summary entry
            for row in summary_rows:
                if row.get("cluster") == cl2 and row.get("status") in ("queued", "running"):
                    row.update(meta_upd)
                    row["runtime_s"] = float(dt)
                    break

            done = i
            elapsed = time.perf_counter() - t0
            # simple running average ETA
            eta_s = (elapsed / max(1, done)) * (total - done)

            LOGGER.info(
                "PB DE [%d/%d] done  cluster=%s status=%s n_sig=%s time=%.1fs elapsed=%.1fs eta=%.1fs",
                done, total, cl2,
                meta_upd.get("status", "unknown"),
                meta_upd.get("n_sig", "NA"),
                dt, elapsed, eta_s,
            )
    else:
        import time
        from concurrent.futures import TimeoutError

        max_workers = int(n_jobs_eff)  # effective workers decided earlier
        heartbeat_s = float(getattr(opts, "heartbeat_s", 60.0)) if "opts" in locals() else 60.0  # or just 60.0
        LOGGER.info(
            "Pseudobulk DE: running in parallel (payloads=%d, max_workers=%d, n_cpus_per_fit=%d, heartbeat=%.0fs).",
            total, max_workers, int(payloads[0].get("n_cpus", 1)) if payloads else 1, heartbeat_s,
        )

        submit_ts: dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = {}
            for p in payloads:
                cl = str(p.get("cluster", ""))
                submit_ts[cl] = time.perf_counter()
                futs[ex.submit(_pydeseq2_cluster_worker, p)] = cl
            done = 0

            # track pending futures so we can log “oldest running”
            pending = set(futs.keys())
            while pending:
                try:
                    # Wake up at least every heartbeat_s even if nothing finishes
                    for fut in as_completed(pending, timeout=heartbeat_s):
                        pending.remove(fut)
                        cl = futs[fut]
                        t_done = time.perf_counter()
                        dt = t_done - float(submit_ts.get(cl, t_done))
                        try:
                            cl2, res, meta_upd = fut.result()
                        except Exception as e:
                            cl2 = cl
                            res = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
                            meta_upd = {"status": "failed", "reason": str(e)}
                        results[cl2] = res
                        for row in summary_rows:
                            if row.get("cluster") == cl2 and row.get("status") in ("queued", "running"):
                                row.update(meta_upd)
                                row["runtime_s"] = float(dt)
                                break
                        done += 1
                        elapsed = time.perf_counter() - t0
                        eta_s = (elapsed / max(1, done)) * (total - done)
                        LOGGER.info(
                            "PB DE [%d/%d] done  cluster=%s status=%s n_sig=%s time=%.1fs elapsed=%.1fs eta=%.1fs",
                            done, total, cl2,
                            meta_upd.get("status", "unknown"),
                            meta_upd.get("n_sig", "NA"),
                            dt, elapsed, eta_s,
                        )
                except TimeoutError:
                    # Heartbeat: nothing finished recently
                    now = time.perf_counter()
                    elapsed = now - t0
                    # find oldest still-pending cluster
                    oldest_cl = None
                    oldest_dt = None
                    if pending:
                        # map pending futures -> cluster names
                        pending_cls = [futs[f] for f in pending]

                        # pick the one with smallest submit_ts
                        oldest_cl = min(pending_cls, key=lambda c: submit_ts.get(c, now))
                        oldest_dt = now - float(submit_ts.get(oldest_cl, now))
                    # (optional) show up to 3 longest-running pending clusters
                    longest = []
                    if pending:
                        pending_cls = [futs[f] for f in pending]
                        pending_cls.sort(key=lambda c: submit_ts.get(c, now))
                        for c in pending_cls[:3]:
                            longest.append(f"{c}:{now - float(submit_ts.get(c, now)):.0f}s")

                    LOGGER.info(
                        "PB DE heartbeat: done=%d/%d pending=%d elapsed=%.1fs oldest=%s (%.1fs) longest=%s",
                        done, total, int(len(pending)), elapsed,
                        str(oldest_cl) if oldest_cl is not None else "NA",
                        float(oldest_dt) if oldest_dt is not None else 0.0,
                        ", ".join(longest) if longest else "NA",
                    )

    # Ensure all clusters present in results dict (even skipped)
    for cl in clusters.astype(str).tolist():
        results.setdefault(cl, pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]))

    if store and store_key:
        adata.uns.setdefault(store_key, {})
        adata.uns[store_key].setdefault("pseudobulk_cluster_vs_rest", {})
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["group_key"] = str(group_key)
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["sample_key"] = str(sample_key)
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["counts_layer"] = str(counts_layer) if counts_layer else None
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["options"] = {
            "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
            "min_samples_per_level": int(opts.min_samples_per_level),
            "alpha": float(opts.alpha),
            "shrink_lfc": bool(opts.shrink_lfc),
            "min_total_counts": int(getattr(opts, "min_total_counts", 10)),
            "min_pct": float(getattr(opts, "min_pct", 0.0)),
            "min_diff_pct": float(getattr(opts, "min_diff_pct", 0.0)),
            "n_jobs": int(n_jobs_eff),
            "n_cpus_per_fit": 1 if int(n_jobs_eff) > 1 else int(n_cpus_eff),
        }
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["summary"] = pd.DataFrame(summary_rows)
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["results"] = results

    return results


def de_condition_within_group_pseudobulk(
    adata: ad.AnnData,
    *,
    group_value: str,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    condition_key: str,
    reference: str,
    spec: PseudobulkSpec = PseudobulkSpec(),
    opts: PseudobulkDEOptions = PseudobulkDEOptions(),
    store_key: Optional[str] = "scomnom_de",
    store: bool = True,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """
    Condition DE within a given group (e.g., cell type / cluster).
    """
    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    sample_key = spec.sample_key
    counts_layer = spec.counts_layer

    if condition_key not in adata.obs:
        raise KeyError(f"condition_key={condition_key!r} not in adata.obs")

    mask = adata.obs[group_key].astype(str).to_numpy() == str(group_value)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    counts_df, meta_df = pseudobulk_aggregate(
        adata,
        sample_key=sample_key,
        group_key=condition_key,
        counts_layer=counts_layer,
        min_cells_per_sample_group=int(opts.min_cells_per_sample_group),
        restrict_cells_mask=mask,
    )

    if meta_df.empty:
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    cond = meta_df[condition_key].astype(str)
    vc = cond.value_counts()
    if str(reference) not in vc.index:
        raise ValueError(f"reference={reference!r} not present in {condition_key!r} within group {group_value!r}")

    levels = list(vc.index.astype(str))
    other_levels = [x for x in levels if x != str(reference)]
    if len(other_levels) != 1:
        raise ValueError(
            f"{condition_key!r} within group {group_value!r} has levels={levels}; "
            "this helper currently supports exactly 2 levels (reference vs one other)."
        )
    test = other_levels[0]

    if int(vc.get(str(reference), 0)) < int(opts.min_samples_per_level) or int(vc.get(str(test), 0)) < int(opts.min_samples_per_level):
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    metadata = meta_df[[sample_key, condition_key]].copy()
    metadata[sample_key] = metadata[sample_key].astype("category")
    metadata[condition_key] = pd.Categorical(metadata[condition_key].astype(str), categories=[str(reference), str(test)])

    dense = counts_df.sparse.to_coo().toarray().astype(np.int64, copy=False)
    counts = pd.DataFrame(dense, index=counts_df.index, columns=counts_df.columns)

    min_total = int(getattr(opts, "min_total_counts", 10))
    keep = (counts.sum(axis=0) >= min_total)
    counts = counts.loc[:, keep]
    if counts.shape[1] == 0:
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    # Seurat-like prevalence gates (library prevalence by condition)
    min_pct = float(getattr(opts, "min_pct", 0.0))
    min_diff_pct = float(getattr(opts, "min_diff_pct", 0.0))
    counts, _prev_meta = _apply_min_pct_filters_pseudobulk(
        counts,
        labels=metadata[condition_key],
        level_A=str(test),
        level_B=str(reference),
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
    )
    if counts.shape[1] == 0:
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    try:
        res, meta = _run_pydeseq2(
            counts,
            metadata.rename(columns={sample_key: "sample"}),
            design_factors=["sample", condition_key],
            contrast=(condition_key, str(test), str(reference)),
            alpha=opts.alpha,
            shrink_lfc=opts.shrink_lfc,
            n_cpus=n_cpus,
            size_factors=str(getattr(opts, "size_factors", "poscounts")),
        )
    except Exception as e:
        LOGGER.warning("PyDESeq2 failed for condition DE within %s=%s: %s", group_key, group_value, e)
        res = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
        meta_df = {
            "sf_policy": str(getattr(opts, "size_factors", "poscounts")),
            "sf_used": None,
            "sf_forced": False,
            "warn_iterative_size_factors": False,
            "warn_low_df_dispersion": False,
            "warnings": [str(e)],
        }

    if store and store_key:
            adata.uns.setdefault(store_key, {})
            adata.uns[store_key].setdefault("pseudobulk_condition_within_group", {})
            key = f"{group_key}={group_value}::{condition_key}::{test}_vs_{reference}"
            adata.uns[store_key]["pseudobulk_condition_within_group"][key] = {
                "group_key": str(group_key),
                "group_value": str(group_value),
                "sample_key": str(sample_key),
                "condition_key": str(condition_key),
                "reference": str(reference),
                "test": str(test),
                "counts_layer": str(counts_layer) if counts_layer else None,
                "options": {
                    "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
                    "min_samples_per_level": int(opts.min_samples_per_level),
                    "alpha": float(opts.alpha),
                    "shrink_lfc": bool(opts.shrink_lfc),
                    "min_total_counts": int(getattr(opts, "min_total_counts", 10)),
                    "min_pct": float(getattr(opts, "min_pct", 0.0)),
                    "min_diff_pct": float(getattr(opts, "min_diff_pct", 0.0)),
                },
                "meta": meta_df,
                "results": res,
            }

    return res


# -----------------------------------------------------------------------------
# Cell-level markers (discovery; downsampled to be OOM-safe)
# -----------------------------------------------------------------------------

def _stratified_downsample_indices(
    labels: np.ndarray,
    *,
    max_per_label: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    Return indices of a stratified downsample of labels.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(random_state))
    idxs = []
    for lab in pd.unique(labels):
        m = np.where(labels == lab)[0]
        if m.size <= max_per_label:
            idxs.append(m)
        else:
            idxs.append(rng.choice(m, size=int(max_per_label), replace=False))
    out = np.concatenate(idxs) if idxs else np.array([], dtype=int)
    rng.shuffle(out)
    return out


def compute_markers_celllevel(
    adata: ad.AnnData,
    *,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    spec: CellLevelMarkerSpec = CellLevelMarkerSpec(),
    key_added: str = "scomnom_markers_celllevel",
    store: bool = True,
) -> str:
    """
    Cell-level discovery markers using scanpy.tl.rank_genes_groups.

    Adds Seurat-like prevalence filtering using scanpy pts/pts_rest:
      - min_pct
      - min_diff_pct

    Additionally:
      - if spec.positive_only=True (default), keeps only logFC > 0

    IMPORTANT:
      - After filtering, this function OVERWRITES the rank_genes_groups payload in adata.uns[key_added]
        so downstream plots/exports that read rank_genes_groups cannot "see" negative-FC genes.
    """
    import scanpy as sc
    from scanpy.get import rank_genes_groups_df

    def _pick_lfc_col(df: pd.DataFrame) -> Optional[str]:
        for c in ("logfoldchanges", "log2FoldChange", "logfoldchange"):
            if c in df.columns:
                return c
        return None

    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    labels = adata.obs[group_key].astype(str).to_numpy()

    # ---- downsample (OOM-safe)
    n = int(adata.n_obs)
    max_per = int(spec.max_cells_per_group)
    n_groups = int(pd.unique(labels).size)
    do_down = max_per > 0 and (n > max_per * max(2, n_groups))
    if do_down:
        idx = _stratified_downsample_indices(labels, max_per_label=max_per, random_state=spec.random_state)
        adata_run = adata[idx].copy()
        LOGGER.info(
            "Cell-level markers: downsampled %d → %d cells (max_per_group=%d, n_groups=%d).",
            n, int(adata_run.n_obs), max_per, n_groups,
        )
    else:
        adata_run = adata

    # If user wants positive-only, ranking by abs just promotes big negative genes into top-N
    rankby_abs = bool(spec.rankby_abs)
    if bool(spec.positive_only) and rankby_abs:
        LOGGER.info("Cell-level markers: positive_only=True → forcing rankby_abs=False.")
        rankby_abs = False

    # ---- run scanpy
    sc.tl.rank_genes_groups(
        adata_run,
        groupby=group_key,
        method=str(spec.method),
        use_raw=bool(spec.use_raw),
        layer=spec.layer,
        key_added=key_added,
        n_genes=int(spec.n_genes),
        rankby_abs=bool(rankby_abs),
        pts=True,
    )

    if not store:
        return key_added

    # ---- store raw output payload first
    adata.uns[key_added] = dict(adata_run.uns[key_added])

    adata.uns[key_added]["scomnom_meta"] = {
        "group_key": str(group_key),
        "method": str(spec.method),
        "n_genes": int(spec.n_genes),
        "use_raw": bool(spec.use_raw),
        "layer": spec.layer,
        "rankby_abs": bool(rankby_abs),
        "max_cells_per_group": int(spec.max_cells_per_group),
        "random_state": int(spec.random_state),
        "downsampled": bool(do_down),
        "n_cells_used": int(adata_run.n_obs),
        "n_cells_total": int(adata.n_obs),
        "min_pct": float(spec.min_pct),
        "min_diff_pct": float(spec.min_diff_pct),
        "positive_only": bool(spec.positive_only),
    }

    # ---- filtering + payload rewrite
    min_pct = float(spec.min_pct)
    min_diff_pct = float(spec.min_diff_pct)
    do_filter = (min_pct > 0.0) or (min_diff_pct > 0.0) or bool(spec.positive_only)

    if not do_filter:
        return key_added

    # Scanpy stores groups as field names in a structured array
    groups = list(adata_run.uns[key_added]["names"].dtype.names)

    filtered_by_group: dict[str, pd.DataFrame] = {}
    all_rows = []

    # We will rebuild these arrays to EXACTLY spec.n_genes per group (padding with NaN/None)
    n_keep = int(spec.n_genes)
    dtype_obj = [(g, object) for g in groups]
    dtype_flt = [(g, np.float64) for g in groups]

    names_arr = np.empty(n_keep, dtype=dtype_obj)
    scores_arr = np.full(n_keep, np.nan, dtype=dtype_flt)
    lfc_arr = np.full(n_keep, np.nan, dtype=dtype_flt)
    pvals_arr = np.full(n_keep, np.nan, dtype=dtype_flt)
    padj_arr = np.full(n_keep, np.nan, dtype=dtype_flt)
    pts_arr = np.full(n_keep, np.nan, dtype=dtype_flt)
    pts_rest_arr = np.full(n_keep, np.nan, dtype=dtype_flt)

    for g in groups:
        names_arr[g] = np.array([None] * n_keep, dtype=object)

    for g in groups:
        df = rank_genes_groups_df(adata_run, group=g, key=key_added)
        df = _coerce_pts_columns(df)

        # normalize gene column name
        if "names" in df.columns and "gene" not in df.columns:
            df = df.rename(columns={"names": "gene"})
        df["gene"] = df["gene"].astype(str)

        # apply Seurat-like prevalence gates
        df_f = _apply_min_pct_filters_celllevel(df, min_pct=min_pct, min_diff_pct=min_diff_pct)

        # positive-only post hoc
        if bool(spec.positive_only):
            lfc_col = _pick_lfc_col(df_f)
            if lfc_col is not None:
                df_f[lfc_col] = pd.to_numeric(df_f[lfc_col], errors="coerce")
                df_f = df_f.loc[df_f[lfc_col] > 0].copy()

        df_f.insert(0, "group", str(g))
        filtered_by_group[str(g)] = df_f
        all_rows.append(df_f)

        # ---- write into rebuilt arrays (top n_keep)
        k = min(n_keep, int(df_f.shape[0]))
        if k <= 0:
            continue

        genes = df_f["gene"].to_numpy()[:k]

        # columns: scanpy usually uses logfoldchanges/scores/pvals/pvals_adj
        lfc_col = _pick_lfc_col(df_f)
        lfc = pd.to_numeric(df_f[lfc_col], errors="coerce").to_numpy()[:k] if lfc_col else np.full(k, np.nan)

        scores = pd.to_numeric(df_f.get("scores", np.nan), errors="coerce").to_numpy()[:k] if "scores" in df_f.columns else np.full(k, np.nan)
        pvals = pd.to_numeric(df_f.get("pvals", np.nan), errors="coerce").to_numpy()[:k] if "pvals" in df_f.columns else np.full(k, np.nan)
        padj = pd.to_numeric(df_f.get("pvals_adj", np.nan), errors="coerce").to_numpy()[:k] if "pvals_adj" in df_f.columns else np.full(k, np.nan)

        pts = pd.to_numeric(df_f.get("pts", np.nan), errors="coerce").to_numpy()[:k] if "pts" in df_f.columns else np.full(k, np.nan)
        pts_rest = pd.to_numeric(df_f.get("pts_rest", np.nan), errors="coerce").to_numpy()[:k] if "pts_rest" in df_f.columns else np.full(k, np.nan)

        names_arr[g][:k] = genes
        lfc_arr[g][:k] = lfc
        scores_arr[g][:k] = scores
        pvals_arr[g][:k] = pvals
        padj_arr[g][:k] = padj
        pts_arr[g][:k] = pts
        pts_rest_arr[g][:k] = pts_rest

    adata.uns[key_added]["filtered_by_group"] = filtered_by_group
    adata.uns[key_added]["filtered"] = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()

    # ---- CRITICAL: overwrite the rank_genes_groups payload so exports/plots can't see negatives
    adata.uns[key_added]["names"] = names_arr
    adata.uns[key_added]["logfoldchanges"] = lfc_arr
    adata.uns[key_added]["scores"] = scores_arr
    adata.uns[key_added]["pvals"] = pvals_arr
    adata.uns[key_added]["pvals_adj"] = padj_arr
    adata.uns[key_added]["pts"] = pts_arr
    adata.uns[key_added]["pts_rest"] = pts_rest_arr

    return key_added


# -----------------------------------------------------------------------------
# Optional: combined view helper (cell-level + pseudobulk)
# -----------------------------------------------------------------------------

def combine_celllevel_and_pseudobulk_hits(
    celllevel_df: pd.DataFrame,
    pseudobulk_df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    pb_prefix: str = "pb_",
    cl_prefix: str = "cl_",
) -> pd.DataFrame:
    """
    Convenience: merge two result tables on gene, prefixing columns.

    Intended for a “confidence ladder” table where users see agreement.
    """
    a = celllevel_df.copy()
    b = pseudobulk_df.copy()

    if gene_col not in a.columns:
        raise KeyError(f"{gene_col!r} not in celllevel_df")
    if gene_col not in b.columns:
        raise KeyError(f"{gene_col!r} not in pseudobulk_df")

    a = a.rename(columns={c: f"{cl_prefix}{c}" for c in a.columns if c != gene_col})
    b = b.rename(columns={c: f"{pb_prefix}{c}" for c in b.columns if c != gene_col})

    out = a.merge(b, on=gene_col, how="outer")
    return out


def _downsample_two_level_subset(
    adata: ad.AnnData,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    *,
    max_per_level: int,
    random_state: int = 0,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """
    Returns (adata_subset, meta) where meta has counts and downsampling info.
    Downsamples independently within A and B to max_per_level.
    """
    rng = np.random.default_rng(int(random_state))
    idx_A = np.where(mask_A)[0]
    idx_B = np.where(mask_B)[0]

    nA = int(idx_A.size)
    nB = int(idx_B.size)

    used_A = idx_A
    used_B = idx_B

    downsampled = False
    if max_per_level > 0 and nA > max_per_level:
        used_A = rng.choice(idx_A, size=int(max_per_level), replace=False)
        downsampled = True
    if max_per_level > 0 and nB > max_per_level:
        used_B = rng.choice(idx_B, size=int(max_per_level), replace=False)
        downsampled = True

    idx = np.concatenate([used_A, used_B])
    rng.shuffle(idx)

    meta = dict(
        n_cells_A=nA,
        n_cells_B=nB,
        n_cells_A_used=int(used_A.size),
        n_cells_B_used=int(used_B.size),
        downsampled=bool(downsampled),
    )

    return adata[idx].copy(), meta


def _pb_effect_log2fc(
    adata: ad.AnnData,
    *,
    cluster_mask: np.ndarray,
    contrast_key: str,
    level_A: str,
    level_B: str,
    counts_layer: Optional[str],
    min_total_counts: int = 10,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Pseudobulk effect size only (no p-values):
      log2((sum_A + pc)/(sum_B + pc)) within the cluster.
    """
    # aggregate within cluster by contrast_key by treating contrast_key as sample_key
    counts_df, meta_df = pseudobulk_aggregate(
        adata,
        sample_key=str(contrast_key),
        group_key=None,
        counts_layer=counts_layer,
        min_cells_per_sample_group=1,
        restrict_cells_mask=cluster_mask,
    )
    if meta_df.empty or counts_df.shape[0] == 0:
        return pd.DataFrame(columns=["gene", "pb_log2fc", "pb_sum_A", "pb_sum_B", "pb_cpm_A", "pb_cpm_B"])

    # meta_df index pb_id is "<level>|ALL" because group_key=None
    # meta_df[contrast_key] contains the level
    meta_df2 = meta_df.copy()
    meta_df2["__level"] = meta_df2[str(contrast_key)].astype(str)

    def _row_for(level: str) -> Optional[str]:
        hits = meta_df2.index[meta_df2["__level"].to_numpy() == str(level)]
        return str(hits[0]) if len(hits) > 0 else None

    rid_A = _row_for(level_A)
    rid_B = _row_for(level_B)
    if rid_A is None or rid_B is None:
        return pd.DataFrame(columns=["gene", "pb_log2fc", "pb_sum_A", "pb_sum_B", "pb_cpm_A", "pb_cpm_B"])

    # densify 2 rows only
    sub = counts_df.loc[[rid_A, rid_B]]
    dense = sub.sparse.to_coo().toarray().astype(np.int64, copy=False)
    sumA = dense[0, :]
    sumB = dense[1, :]

    # filter genes by total
    tot = sumA + sumB
    keep = tot >= int(min_total_counts)
    if keep.sum() == 0:
        return pd.DataFrame(columns=["gene", "pb_log2fc", "pb_sum_A", "pb_sum_B", "pb_cpm_A", "pb_cpm_B"])

    sumA = sumA[keep]
    sumB = sumB[keep]
    genes = adata.var_names.to_numpy()[keep]

    pc = float(pseudocount)
    pb_log2fc = np.log2((sumA + pc) / (sumB + pc))

    totA = float(sumA.sum()) if float(sumA.sum()) > 0 else 1.0
    totB = float(sumB.sum()) if float(sumB.sum()) > 0 else 1.0
    cpmA = (sumA / totA) * 1e6
    cpmB = (sumB / totB) * 1e6

    return pd.DataFrame(
        {
            "gene": genes.astype(str),
            "pb_log2fc": pb_log2fc.astype(float),
            "pb_sum_A": sumA.astype(np.int64),
            "pb_sum_B": sumB.astype(np.int64),
            "pb_cpm_A": cpmA.astype(float),
            "pb_cpm_B": cpmB.astype(float),
        }
    )

def _normalize_pair(s: str) -> tuple[str, str]:
    s = str(s).strip()
    if "_vs_" in s:
        a, b = s.split("_vs_", 1)
    elif " vs " in s:
        a, b = s.split(" vs ", 1)
    else:
        raise ValueError(f"Invalid contrast spec {s!r}. Use 'A_vs_B'.")
    a = a.strip()
    b = b.strip()
    if not a or not b:
        raise ValueError(f"Invalid contrast spec {s!r}. Use 'A_vs_B'.")
    if a == b:
        raise ValueError(f"Invalid contrast spec {s!r}: A and B must differ.")
    return a, b


def _select_pairs(levels: Sequence[str], requested: Sequence[str] | None) -> list[tuple[str, str]]:
    lv = [str(x) for x in levels]
    all_pairs = [(lv[i], lv[j]) for i in range(len(lv)) for j in range(i + 1, len(lv))]

    if not requested:
        return all_pairs

    req_pairs = [_normalize_pair(x) for x in requested]
    # Validate membership
    lv_set = set(lv)
    out = []
    for a, b in req_pairs:
        if a not in lv_set or b not in lv_set:
            raise ValueError(f"Requested contrast {a}_vs_{b} not in levels={sorted(lv_set)}")
        # keep direction as requested (important for sign)
        out.append((a, b))
    return out


@dataclass(frozen=True)
class ContrastConditionalSpec:
    """
    Pairwise (A vs B) markers within each cluster, when sample-level replicates don't exist.
    """
    contrast_key: str  # e.g. batch_key / sample_id / group
    contrasts: Tuple[str, ...] = ()
    methods: Tuple[Literal["wilcoxon", "logreg"], ...] = ("wilcoxon", "logreg")

    # guards
    min_cells_per_level_in_cluster: int = 50
    max_cells_per_level_in_cluster: int = 2000  # downsample per (cluster, level) for cell-level tests

    # gene filter for pb effect (and optionally for cell-level tests)
    min_total_counts: int = 10
    pseudocount: float = 1.0

    # Seurat-like prevalence gates for *cell-level* tests within cluster (uses scanpy pts/pts_rest)
    min_pct: float = 0.0
    min_diff_pct: float = 0.0

    # thresholds for combined tiering
    cl_alpha: float = 0.05
    cl_min_abs_logfc: float = 0.25
    lr_min_abs_coef: float = 0.25
    pb_min_abs_log2fc: float = 0.5

    random_state: int = 0



def contrast_conditional_markers(
    adata: ad.AnnData,
    *,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    spec: ContrastConditionalSpec,
    pb_spec: Optional[PseudobulkSpec] = None,
    store_key: str = "scomnom_de",
    store: bool = True,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Pairwise (A vs B) markers within each cluster.

    Returns nested dict:
      out[cluster][f"{A}_vs_{B}"]["combined" | "wilcoxon" | "logreg" | "pseudobulk_effect"] = DataFrame

    Notes:
      - If pb_spec is None, pseudobulk_effect is empty and tiering ignores pseudobulk.
      - Cell-level prevalence gates (min_pct/min_diff_pct) are applied when prevalence columns exist.
    """
    import scanpy as sc
    from scanpy.get import rank_genes_groups_df

    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    contrast_key = str(spec.contrast_key)

    if contrast_key not in adata.obs:
        raise KeyError(f"contrast_key={contrast_key!r} not in adata.obs")

    levels = pd.Index(pd.unique(adata.obs[contrast_key].astype(str))).sort_values()
    if len(levels) < 2:
        raise ValueError(f"contrast_key={contrast_key!r} needs >=2 levels, got {list(levels)}")

    clusters = pd.Index(pd.unique(adata.obs[group_key].astype(str))).sort_values()

    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    summary_rows: list[dict[str, Any]] = []

    counts_layer = pb_spec.counts_layer if pb_spec is not None else None

    # cell-level prevalence gates (Seurat-like)
    cl_min_pct = float(getattr(spec, "min_pct", 0.0))
    cl_min_diff_pct = float(getattr(spec, "min_diff_pct", 0.0))

    # choose which pairs to run
    pairs = _select_pairs(levels.tolist(), list(spec.contrasts) if spec.contrasts else None)

    def _sgn(x: Any) -> int:
        try:
            if x is None or not np.isfinite(x):
                return 0
        except Exception:
            return 0
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for cl in clusters:
        cl_mask = (adata.obs[group_key].astype(str).to_numpy() == str(cl))
        if int(cl_mask.sum()) == 0:
            continue

        out[str(cl)] = {}

        cl_contrast = adata.obs.loc[cl_mask, contrast_key].astype(str)
        global_idx = np.where(cl_mask)[0]

        by_level_idx: dict[str, np.ndarray] = {}
        for lv in levels:
            m = (cl_contrast.to_numpy() == str(lv))
            by_level_idx[str(lv)] = global_idx[np.where(m)[0]]

        for (A, B) in pairs:
            pair_key = f"{A}_vs_{B}"

            idxA = by_level_idx.get(A, np.array([], dtype=int))
            idxB = by_level_idx.get(B, np.array([], dtype=int))

            nA = int(idxA.size)
            nB = int(idxB.size)

            if nA < int(spec.min_cells_per_level_in_cluster) or nB < int(spec.min_cells_per_level_in_cluster):
                out[str(cl)][pair_key] = {
                    "wilcoxon": pd.DataFrame(),
                    "logreg": pd.DataFrame(),
                    "pseudobulk_effect": pd.DataFrame(),
                    "combined": pd.DataFrame(),
                }
                summary_rows.append(
                    dict(
                        cluster=str(cl),
                        contrast_key=contrast_key,
                        A=A,
                        B=B,
                        status="skipped",
                        reason="min_cells_per_level_in_cluster",
                        n_cells_A=nA,
                        n_cells_B=nB,
                    )
                )
                continue

            maskA = np.zeros(adata.n_obs, dtype=bool)
            maskA[idxA] = True
            maskB = np.zeros(adata.n_obs, dtype=bool)
            maskB[idxB] = True

            adata_sub, ds_meta = _downsample_two_level_subset(
                adata,
                maskA,
                maskB,
                max_per_level=int(spec.max_cells_per_level_in_cluster),
                random_state=int(spec.random_state),
            )

            # keep only the two levels explicitly (defensive)
            adata_sub = adata_sub[adata_sub.obs[contrast_key].astype(str).isin([A, B])].copy()

            # ----------------------------
            # pseudobulk effect (optional)
            # ----------------------------
            if pb_spec is not None:
                pb_df = _pb_effect_log2fc(
                    adata,
                    cluster_mask=cl_mask,
                    contrast_key=contrast_key,
                    level_A=A,
                    level_B=B,
                    counts_layer=counts_layer,
                    min_total_counts=int(spec.min_total_counts),
                    pseudocount=float(spec.pseudocount),
                )
            else:
                pb_df = pd.DataFrame(
                    columns=["gene", "pb_log2fc", "pb_sum_A", "pb_sum_B", "pb_cpm_A", "pb_cpm_B"]
                )

            # ----------------------------
            # Wilcoxon (cell-level)
            # ----------------------------
            wilcoxon_df = pd.DataFrame()
            if "wilcoxon" in spec.methods:
                sc.tl.rank_genes_groups(
                    adata_sub,
                    groupby=contrast_key,
                    groups=[A],
                    reference=B,
                    method="wilcoxon",
                    use_raw=False,
                    key_added="__tmp_wilcoxon",
                    n_genes=adata_sub.n_vars,
                    rankby_abs=False,
                    pts=True,
                )
                d = rank_genes_groups_df(adata_sub, group=A, key="__tmp_wilcoxon")
                d = _coerce_pts_columns(d)

                wilcoxon_df = d.rename(
                    columns={
                        "names": "gene",
                        "logfoldchanges": "cl_logfc",
                        "scores": "cl_score",
                        "pvals": "cl_pval",
                        "pvals_adj": "cl_padj",
                        "pts": "cl_pts",
                        "pts_rest": "cl_pts_rest",
                    }
                )
                if "gene" in wilcoxon_df.columns:
                    wilcoxon_df["gene"] = wilcoxon_df["gene"].astype(str)

                # Apply Seurat-like gates using prevalence columns
                if ("cl_pts" in wilcoxon_df.columns) and ("cl_pts_rest" in wilcoxon_df.columns):
                    _tmp = wilcoxon_df.rename(columns={"cl_pts": "pts", "cl_pts_rest": "pts_rest"})
                    _tmp = _apply_min_pct_filters_celllevel(_tmp, min_pct=cl_min_pct, min_diff_pct=cl_min_diff_pct)
                    wilcoxon_df = _tmp.rename(columns={"pts": "cl_pts", "pts_rest": "cl_pts_rest"}).copy()

            # ----------------------------
            # Logreg (cell-level)
            # ----------------------------
            logreg_df = pd.DataFrame()
            if "logreg" in spec.methods:
                sc.tl.rank_genes_groups(
                    adata_sub,
                    groupby=contrast_key,
                    groups=[A],
                    reference=B,
                    method="logreg",
                    use_raw=False,
                    key_added="__tmp_logreg",
                    n_genes=adata_sub.n_vars,
                    rankby_abs=False,
                    pts=True,
                )
                d = rank_genes_groups_df(adata_sub, group=A, key="__tmp_logreg")
                d = _coerce_pts_columns(d)

                d = d.rename(columns={"names": "gene"})
                if "logfoldchanges" in d.columns:
                    d = d.rename(columns={"logfoldchanges": "lr_coef"})
                if "scores" in d.columns:
                    d = d.rename(columns={"scores": "lr_score"})

                # carry prevalence if present
                rename_prev = {}
                if "pts" in d.columns:
                    rename_prev["pts"] = "cl_pts"
                if "pts_rest" in d.columns:
                    rename_prev["pts_rest"] = "cl_pts_rest"
                if rename_prev:
                    d = d.rename(columns=rename_prev)

                cols = ["gene"] + [c for c in ["lr_coef", "lr_score", "cl_pts", "cl_pts_rest"] if c in d.columns]
                logreg_df = d[cols].copy()
                if "gene" in logreg_df.columns:
                    logreg_df["gene"] = logreg_df["gene"].astype(str)

                # Apply Seurat-like gates if prevalence columns exist
                if ("cl_pts" in logreg_df.columns) and ("cl_pts_rest" in logreg_df.columns):
                    _tmp = logreg_df.rename(columns={"cl_pts": "pts", "cl_pts_rest": "pts_rest"})
                    _tmp = _apply_min_pct_filters_celllevel(_tmp, min_pct=cl_min_pct, min_diff_pct=cl_min_diff_pct)
                    logreg_df = _tmp.rename(columns={"pts": "cl_pts", "pts_rest": "cl_pts_rest"}).copy()

            # ----------------------------
            # Combined merge
            # ----------------------------
            if pb_df is not None and not pb_df.empty:
                combined = pb_df.copy()
            else:
                combined = pd.DataFrame({"gene": []})

            if wilcoxon_df is not None and not wilcoxon_df.empty:
                combined = combined.merge(wilcoxon_df, on="gene", how="outer")

            if logreg_df is not None and not logreg_df.empty:
                combined = combined.merge(logreg_df, on="gene", how="outer", suffixes=("", "_lr"))

                # if logreg carried cl_pts/cl_pts_rest with suffixes, prefer wilcoxon’s if present
                if "cl_pts_lr" in combined.columns and "cl_pts" not in combined.columns:
                    combined = combined.rename(columns={"cl_pts_lr": "cl_pts"})
                if "cl_pts_rest_lr" in combined.columns and "cl_pts_rest" not in combined.columns:
                    combined = combined.rename(columns={"cl_pts_rest_lr": "cl_pts_rest"})
                for c in ["cl_pts_lr", "cl_pts_rest_lr"]:
                    if c in combined.columns:
                        combined = combined.drop(columns=[c])

            # Add provenance columns
            combined.insert(0, "cluster", str(cl))
            combined.insert(1, "contrast_key", contrast_key)
            combined.insert(2, "A", A)
            combined.insert(3, "B", B)
            combined.insert(4, "n_cells_A", int(ds_meta["n_cells_A"]))
            combined.insert(5, "n_cells_B", int(ds_meta["n_cells_B"]))
            combined.insert(6, "downsampled", bool(ds_meta["downsampled"]))
            combined.insert(7, "n_cells_A_used", int(ds_meta["n_cells_A_used"]))
            combined.insert(8, "n_cells_B_used", int(ds_meta["n_cells_B_used"]))

            # numeric coercions
            for col in ["cl_logfc", "cl_padj", "lr_coef", "pb_log2fc", "cl_pts", "cl_pts_rest"]:
                if col in combined.columns:
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")

            # Seurat-like gating at hit layer (so tiering respects it)
            combined["pass_minpct"] = True
            if (cl_min_pct > 0.0 or cl_min_diff_pct > 0.0) and ("cl_pts" in combined.columns) and ("cl_pts_rest" in combined.columns):
                pass_mask = pd.Series(True, index=combined.index)
                if cl_min_pct > 0.0:
                    pass_mask &= (combined["cl_pts"] >= cl_min_pct) | (combined["cl_pts_rest"] >= cl_min_pct)
                if cl_min_diff_pct > 0.0:
                    pass_mask &= (combined["cl_pts"] - combined["cl_pts_rest"]).abs() >= cl_min_diff_pct
                combined["pass_minpct"] = pass_mask.fillna(True)

            # hits
            combined["hit_wilcoxon"] = False
            if "cl_padj" in combined.columns and "cl_logfc" in combined.columns:
                combined["hit_wilcoxon"] = (
                    (combined["cl_padj"] < float(spec.cl_alpha))
                    & (combined["cl_logfc"].abs() >= float(spec.cl_min_abs_logfc))
                    & (combined["pass_minpct"])
                )

            combined["hit_logreg"] = False
            if "lr_coef" in combined.columns:
                combined["hit_logreg"] = (combined["lr_coef"].abs() >= float(spec.lr_min_abs_coef)) & (combined["pass_minpct"])

            combined["hit_pseudobulk"] = False
            if pb_spec is not None and "pb_log2fc" in combined.columns:
                combined["hit_pseudobulk"] = combined["pb_log2fc"].abs() >= float(spec.pb_min_abs_log2fc)

            combined["sign_agree_pb_wilcoxon"] = False
            if pb_spec is not None and "pb_log2fc" in combined.columns and "cl_logfc" in combined.columns:
                combined["sign_agree_pb_wilcoxon"] = (
                    combined["pb_log2fc"].apply(_sgn) == combined["cl_logfc"].apply(_sgn)
                )

            # tiering
            hits = combined[["hit_wilcoxon", "hit_logreg", "hit_pseudobulk"]].sum(axis=1)
            if pb_spec is None:
                # only two methods in play (wilcoxon/logreg)
                tier = np.where(hits >= 2, "Tier1", np.where(hits >= 1, "Tier2", "None"))
                combined["consensus_tier"] = tier
                score = hits.astype(int) + np.where(combined["consensus_tier"] == "Tier1", 1, 0)
            else:
                tier = np.where(
                    (hits == 3) & (combined.get("sign_agree_pb_wilcoxon", True)),
                    "Tier1",
                    np.where(hits >= 2, "Tier2", np.where(hits >= 1, "Tier3", "None")),
                )
                combined["consensus_tier"] = tier
                score = hits.astype(int) + np.where(combined["consensus_tier"] == "Tier1", 2, 0)

                # penalize strong sign disagreements (optional)
                if "pb_log2fc" in combined.columns and "cl_logfc" in combined.columns:
                    disagree = (
                        (combined["sign_agree_pb_wilcoxon"] == False)
                        & (combined["pb_log2fc"].abs() > 0.5)
                        & (combined["cl_logfc"].abs() > 0.25)
                    )
                    score = score - np.where(disagree, 1, 0)

            combined["consensus_score"] = score.astype(int)

            # sort
            if pb_spec is None:
                tier_cat = pd.Categorical(
                    combined["consensus_tier"],
                    categories=["Tier1", "Tier2", "None"],
                    ordered=True,
                )
            else:
                tier_cat = pd.Categorical(
                    combined["consensus_tier"],
                    categories=["Tier1", "Tier2", "Tier3", "None"],
                    ordered=True,
                )

            combined["__tier_order"] = tier_cat
            sort_cols = ["__tier_order"]
            asc = [True]
            if "cl_padj" in combined.columns:
                sort_cols.append("cl_padj"); asc.append(True)
            if "pb_log2fc" in combined.columns:
                sort_cols.append("pb_log2fc"); asc.append(False)
            if "lr_coef" in combined.columns:
                sort_cols.append("lr_coef"); asc.append(False)

            combined = combined.sort_values(sort_cols, ascending=asc).drop(columns=["__tier_order"])

            out[str(cl)][pair_key] = {
                "wilcoxon": wilcoxon_df,
                "logreg": logreg_df,
                "pseudobulk_effect": pb_df,
                "combined": combined,
            }

            # summary
            if "consensus_tier" in combined.columns:
                n_t1 = int((combined["consensus_tier"] == "Tier1").sum())
                n_t2 = int((combined["consensus_tier"] == ("Tier2" if pb_spec is not None else "Tier2")).sum())
                n_t3 = int((combined["consensus_tier"] == "Tier3").sum()) if pb_spec is not None else 0
                top10 = combined.loc[combined["consensus_tier"] == "Tier1", "gene"].head(10).tolist() if "gene" in combined.columns else []
            else:
                n_t1 = n_t2 = n_t3 = 0
                top10 = []

            summary_rows.append(
                dict(
                    cluster=str(cl),
                    contrast_key=contrast_key,
                    A=A,
                    B=B,
                    status="ok",
                    reason="",
                    n_cells_A=nA,
                    n_cells_B=nB,
                    downsampled=bool(ds_meta["downsampled"]),
                    n_genes_tested=int(combined.shape[0]),
                    n_tier1=int(n_t1),
                    n_tier2=int(n_t2),
                    n_tier3=int(n_t3),
                    min_pct=float(cl_min_pct),
                    min_diff_pct=float(cl_min_diff_pct),
                    pseudobulk_effect_included=bool(pb_spec is not None),
                    top10_tier1_genes=",".join(map(str, top10)),
                )
            )

    if store:
        adata.uns.setdefault(store_key, {})
        adata.uns[store_key].setdefault("contrast_conditional", {})
        adata.uns[store_key]["contrast_conditional"]["group_key"] = str(group_key)
        adata.uns[store_key]["contrast_conditional"]["contrast_key"] = str(contrast_key)
        adata.uns[store_key]["contrast_conditional"]["counts_layer"] = str(counts_layer) if (pb_spec is not None and counts_layer) else None
        adata.uns[store_key]["contrast_conditional"]["spec"] = {**spec.__dict__, "methods": list(spec.methods)}
        adata.uns[store_key]["contrast_conditional"]["pseudobulk_effect_included"] = bool(pb_spec is not None)
        adata.uns[store_key]["contrast_conditional"]["summary"] = pd.DataFrame(summary_rows)
        adata.uns[store_key]["contrast_conditional"]["results"] = out

    return out


def de_condition_within_group_pseudobulk_multi(
    adata: ad.AnnData,
    *,
    group_value: str,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    condition_key: str,
    spec: PseudobulkSpec = PseudobulkSpec(),
    opts: PseudobulkDEOptions = PseudobulkDEOptions(),
    contrasts: Sequence[str] | None = None,   # "A_vs_B"
    store_key: Optional[str] = "scomnom_de",
    store: bool = True,
    n_cpus: int = 1,
) -> Dict[str, pd.DataFrame]:
    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)

    mask = adata.obs[group_key].astype(str).to_numpy() == str(group_value)
    if mask.sum() == 0:
        return {}

    # get levels present *within this cluster*
    levels = pd.Index(pd.unique(adata.obs.loc[mask, condition_key].astype(str))).sort_values().tolist()
    if len(levels) < 2:
        return {}

    pairs = _select_pairs(levels, contrasts)

    out: Dict[str, pd.DataFrame] = {}
    for A, B in pairs:
        # Run A vs B by calling your existing function,
        # but it currently wants reference and assumes exactly 2 levels.
        # We can *temporarily* treat reference=B and restrict to only A/B cells:
        m2 = mask & adata.obs[condition_key].astype(str).isin([A, B]).to_numpy()
        if m2.sum() == 0:
            continue

        # Call a slightly refactored internal helper that accepts restrict mask,
        # OR easiest: duplicate the aggregation logic here (small).
        res = de_condition_within_group_pseudobulk(
            adata[m2].copy(),
            group_value=str(group_value),
            groupby=groupby,          # group_key resolution still OK, but now only one cluster present
            round_id=round_id,
            condition_key=condition_key,
            reference=str(B),
            spec=spec,
            opts=opts,
            store_key=None,           # store ourselves with a better key
            store=False,
            n_cpus=n_cpus,
        )

        out[f"{A}_vs_{B}"] = res

        if store and store_key:
            adata.uns.setdefault(store_key, {})
            adata.uns[store_key].setdefault("pseudobulk_condition_within_group_multi", {})
            key = f"{group_key}={group_value}::{condition_key}::{A}_vs_{B}"
            adata.uns[store_key]["pseudobulk_condition_within_group_multi"][key] = {
                "group_key": str(group_key),
                "group_value": str(group_value),
                "condition_key": str(condition_key),
                "test": str(A),
                "reference": str(B),
                "results": res,
                "options": {
                    "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
                    "min_samples_per_level": int(opts.min_samples_per_level),
                    "alpha": float(opts.alpha),
                    "shrink_lfc": bool(opts.shrink_lfc),
                },
            }

    return out


def _coerce_pts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Scanpy rank_genes_groups_df prevalence columns to:
      - pts (group)
      - pts_rest (reference/rest)

    Handles common Scanpy variants:
      - pts / pts_rest
      - pct_nz_group / pct_nz_reference
    """
    out = df.copy()

    # common in newer scanpy when pts=True
    if "pts" in out.columns and "pts_rest" in out.columns:
        return out

    # common alternative naming
    if "pct_nz_group" in out.columns and "pct_nz_reference" in out.columns:
        out = out.rename(columns={"pct_nz_group": "pts", "pct_nz_reference": "pts_rest"})
        return out

    # If not present, create NaNs so downstream gating becomes a no-op unless user insists.
    if "pts" not in out.columns:
        out["pts"] = np.nan
    if "pts_rest" not in out.columns:
        out["pts_rest"] = np.nan
    return out


def _apply_min_pct_filters_celllevel(
    df: pd.DataFrame,
    *,
    min_pct: float = 0.0,
    min_diff_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Seurat-like filtering for cell-level rank_genes_groups tables.

    Requires df to have:
      - gene (or names renamed)
      - pts, pts_rest (fraction of cells expressing)
    """
    if df is None or getattr(df, "empty", True):
        return df

    min_pct = float(min_pct)
    min_diff_pct = float(min_diff_pct)
    if min_pct <= 0.0 and min_diff_pct <= 0.0:
        return df

    d = _coerce_pts_columns(df)

    # robust numeric conversion
    d["pts"] = pd.to_numeric(d["pts"], errors="coerce")
    d["pts_rest"] = pd.to_numeric(d["pts_rest"], errors="coerce")

    # If pts are missing (all NaN), do not filter (avoid silently dropping everything)
    if d["pts"].notna().sum() == 0 and d["pts_rest"].notna().sum() == 0:
        return df

    keep = pd.Series(True, index=d.index)

    if min_pct > 0.0:
        keep &= (d["pts"] >= min_pct) | (d["pts_rest"] >= min_pct)

    if min_diff_pct > 0.0:
        keep &= (d["pts"] - d["pts_rest"]).abs() >= min_diff_pct

    return d.loc[keep].copy()


def _apply_min_pct_filters_pseudobulk(
    counts: pd.DataFrame,
    labels: pd.Series,
    *,
    level_A: str,
    level_B: str,
    min_pct: float = 0.0,
    min_diff_pct: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Seurat-like filtering for pseudobulk DE (library/sample prevalence).

    counts: (libs x genes), dense int DataFrame (small)
    labels: (libs,) categorical/str labels for the two levels (e.g. "target"/"rest")
    level_A, level_B: the two levels to compare
    """
    min_pct = float(min_pct)
    min_diff_pct = float(min_diff_pct)

    meta = {
        "min_pct": min_pct,
        "min_diff_pct": min_diff_pct,
        "n_genes_before": int(counts.shape[1]),
        "n_genes_after": int(counts.shape[1]),
    }

    if counts.shape[1] == 0:
        return counts, meta

    if (min_pct <= 0.0) and (min_diff_pct <= 0.0):
        return counts, meta

    lab = labels.astype(str)
    mA = (lab == str(level_A)).to_numpy()
    mB = (lab == str(level_B)).to_numpy()

    # if either group empty, leave unchanged (upstream should skip anyway)
    if mA.sum() == 0 or mB.sum() == 0:
        return counts, meta

    X = counts.to_numpy(copy=False)  # small dense
    det = (X > 0)

    prevA = det[mA, :].mean(axis=0)
    prevB = det[mB, :].mean(axis=0)

    keep = np.ones(counts.shape[1], dtype=bool)

    if min_pct > 0.0:
        keep &= (prevA >= min_pct) | (prevB >= min_pct)

    if min_diff_pct > 0.0:
        keep &= (np.abs(prevA - prevB) >= min_diff_pct)

    out = counts.loc[:, keep].copy()
    meta["n_genes_after"] = int(out.shape[1])
    return out, meta
