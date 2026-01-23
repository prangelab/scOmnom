# src/scomnom/de_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


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
    # OOM guard:
    max_cells_per_group: int = 2000  # stratified downsample per group if dataset is huge
    random_state: int = 0


@dataclass(frozen=True)
class PseudobulkDEOptions:
    min_cells_per_sample_group: int = 20
    min_samples_per_level: int = 2
    alpha: float = 0.05
    lfc_threshold: float = 0.0
    shrink_lfc: bool = True
    min_total_counts: int = 10


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

    # legacy-ish pointers
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
) -> pd.DataFrame:
    """
    Run PyDESeq2 for one contrast.

    counts: samples x genes (integers)
    metadata: samples x covariates (categorical recommended)
    contrast: (factor_name, test_level, ref_level)
    """
    _require_pydeseq2()
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    # Defensive copies, ensure alignment
    counts = counts.loc[metadata.index]
    # PyDESeq2 expects non-negative ints
    counts_i = counts.round().astype(np.int64)

    dds = DeseqDataSet(
        counts=counts_i,
        metadata=metadata.copy(),
        design_factors=list(design_factors),
        ref_level={contrast[0]: contrast[2]},
        n_cpus=int(n_cpus),    )
    dds.deseq2()

    stat = DeseqStats(
        dds,
        contrast=list(contrast),
        alpha=float(alpha),
        # shrinkage varies across versions; best-effort below
        n_cpus=int(n_cpus),
    )
    stat.summary()

    res = stat.results_df.copy()
    # Standardize column names to match your older plotting expectations
    # PyDESeq2 uses: log2FoldChange, lfcSE, stat, pvalue, padj
    if "gene" not in res.columns:
        res["gene"] = res.index.astype(str)

    # Best-effort LFC shrinkage if available
    if shrink_lfc:
        try:
            # Some versions support lfc_shrink(); others don't.
            # If supported, it updates results_df.
            stat.lfc_shrink()
            res2 = stat.results_df.copy()
            if "gene" not in res2.columns:
                res2["gene"] = res2.index.astype(str)
            res = res2
        except Exception:
            # keep unshrunk results
            pass

    # Sort + keep canonical columns
    cols = [c for c in ["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"] if c in res.columns]
    res = res[cols].sort_values("padj", na_position="last")
    return res


def de_cluster_vs_rest_pseudobulk(
    adata: ad.AnnData,
    *,
    groupby: Optional[str] = None,
    round_id: Optional[str] = None,
    spec: PseudobulkSpec = PseudobulkSpec(),
    opts: PseudobulkDEOptions = PseudobulkDEOptions(),
    store_key: Optional[str] = "scomnom_de",
    store: bool = True,
    n_cpus: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Rigorous “cluster markers” via paired pseudobulk DE:
      For each cluster C:
        within each sample S:
          library1 = cells in (S, C)
          library2 = cells in (S, not C)
        Fit: ~ sample + binary_cluster
        Contrast: C vs Rest (ref=Rest)

    OOM-safe approach:
      1) aggregate counts once per (sample, cluster) across ALL cells
      2) aggregate totals once per sample
      3) for each cluster, compute rest = total - target (no re-reading X)
    """
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

    # Convert sparse DF to something indexable; keep sparse until per-cluster subset
    # Build fast lookup: (sample, cluster) -> row id
    meta_sc2 = meta_sc.copy()
    meta_sc2["__sample"] = meta_sc2[sample_key].astype(str)
    meta_sc2["__group"] = meta_sc2[group_key].astype(str)
    meta_sc2["__key"] = meta_sc2["__sample"] + "||" + meta_sc2["__group"]
    sc_row_by_key = pd.Series(meta_sc2.index.to_numpy(), index=meta_sc2["__key"].to_numpy())

    # Totals by sample: sample -> row id
    meta_s2 = meta_s.copy()
    meta_s2["__sample"] = meta_s2[sample_key].astype(str)
    s_row_by_sample = pd.Series(meta_s2.index.to_numpy(), index=meta_s2["__sample"].to_numpy())

    clusters = pd.Index(pd.unique(meta_sc[group_key].astype(str))).sort_values()
    results: Dict[str, pd.DataFrame] = {}

    # For storage
    summary_rows = []

    for cl in clusters:
        # eligible samples where (sample, cl) has enough cells
        m_cl = meta_sc2["__group"].to_numpy() == str(cl)
        meta_cl = meta_sc2.loc[m_cl, [sample_key, group_key, "n_cells"]].copy()

        # Filter by min cells per (sample, cluster)
        meta_cl = meta_cl.loc[meta_cl["n_cells"] >= int(opts.min_cells_per_sample_group)]
        n_target_samples = int(meta_cl.shape[0])

        if n_target_samples < int(opts.min_samples_per_level):
            results[str(cl)] = pd.DataFrame(
                columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
            )
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

        # Determine which samples participate
        samples = meta_cl[sample_key].astype(str).to_numpy()
        samples = np.unique(samples)

        # Build the paired libraries (target + rest) per sample
        pb_index = []
        metadata_rows = []
        target_rows = []
        total_rows = []

        for s in samples:
            # target row id
            key = f"{s}||{cl}"
            if key not in sc_row_by_key.index:
                continue
            rid_target = sc_row_by_key.loc[key]
            rid_total = s_row_by_sample.loc[s] if s in s_row_by_sample.index else None
            if rid_total is None:
                continue

            pb_index.append(f"{s}|{cl}|target")
            metadata_rows.append({sample_key: str(s), "binary_cluster": "target"})
            target_rows.append(rid_target)
            total_rows.append(rid_total)

            pb_index.append(f"{s}|{cl}|rest")
            metadata_rows.append({sample_key: str(s), "binary_cluster": "rest"})
            target_rows.append(rid_target)  # reuse for shape; we compute rest later
            total_rows.append(rid_total)

        if len(pb_index) < 2 * int(opts.min_samples_per_level):
            results[str(cl)] = pd.DataFrame(
                columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
            )
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "skipped",
                    "reason": "paired library construction failed",
                    "n_target_samples": n_target_samples,
                }
            )
            continue

        metadata = pd.DataFrame(metadata_rows, index=pd.Index(pb_index, name="pb_id"))
        metadata[sample_key] = metadata[sample_key].astype("category")
        metadata["binary_cluster"] = pd.Categorical(metadata["binary_cluster"], categories=["rest", "target"])

        # Slice counts (small): use sparse frames and densify now
        # Get dense arrays for target and total (same order as metadata rows, but we want per-row)
        # Build counts as samples x genes DataFrame
        # target rows alternate with rest rows; compute row-wise accordingly.
        target_mat = counts_sc.loc[pd.Index(target_rows)].sparse.to_coo().toarray()
        total_mat = counts_s.loc[pd.Index(total_rows)].sparse.to_coo().toarray()

        # Compose final counts:
        # even idx = target, odd idx = rest
        out = np.zeros_like(total_mat, dtype=np.int64)
        for i in range(out.shape[0]):
            if i % 2 == 0:
                out[i, :] = target_mat[i, :]
            else:
                # rest = total - target
                v = total_mat[i, :] - target_mat[i, :]
                v[v < 0] = 0
                out[i, :] = v.astype(np.int64, copy=False)

        counts = pd.DataFrame(out, index=metadata.index, columns=adata.var_names)

        # Drop genes with extremely low total counts (helps speed/stability)
        min_total = int(getattr(opts, "min_total_counts", 10))  # optional new option
        keep = (counts.sum(axis=0) >= min_total)
        counts = counts.loc[:, keep]
        if counts.shape[1] == 0:
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append({"cluster": str(cl), "status": "skipped", "reason": "no genes left after filtering"})
            continue

        # Replicate check for both levels
        vc = metadata["binary_cluster"].value_counts()
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

        # Run PyDESeq2
        try:
            res = _run_pydeseq2(
                counts,
                metadata.rename(columns={sample_key: "sample"}),  # keep sample column stable
                design_factors=["sample", "binary_cluster"],
                contrast=("binary_cluster", "target", "rest"),
                alpha=opts.alpha,
                shrink_lfc=opts.shrink_lfc,
                n_cpus=n_cpus,
            )
            results[str(cl)] = res

            n_sig = int((pd.to_numeric(res["padj"], errors="coerce") < opts.alpha).sum()) if not res.empty else 0
            summary_rows.append(
                {
                    "cluster": str(cl),
                    "status": "ok",
                    "n_target_samples": int(len(samples)),
                    "n_libraries": int(counts.shape[0]),
                    "min_cells_per_sample_group": int(opts.min_cells_per_sample_group),
                    "n_sig": n_sig
                }
            )
        except Exception as e:
            LOGGER.warning("PyDESeq2 failed for cluster %s vs rest: %s", str(cl), e)
            results[str(cl)] = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
            summary_rows.append(
                {"cluster": str(cl), "status": "failed", "reason": str(e)}
            )

    # Optional store (kept structured)
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
        }
        adata.uns[store_key]["pseudobulk_cluster_vs_rest"]["summary"] = pd.DataFrame(summary_rows)
        # Store results per cluster as dict-of-DataFrames
        # (AnnData can store DataFrames in .uns; for zarr/h5 this may need serialization later)
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

    Pipeline:
      - restrict cells to group == group_value
      - aggregate counts per (sample, condition)
      - fit: ~ sample + condition
      - contrast: condition (test vs reference)

    Requirements:
      - at least opts.min_samples_per_level libraries for each condition level
      - each (sample, condition) library must have >= opts.min_cells_per_sample_group cells
    """
    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    sample_key = spec.sample_key
    counts_layer = spec.counts_layer

    if condition_key not in adata.obs:
        raise KeyError(f"condition_key={condition_key!r} not in adata.obs")

    mask = adata.obs[group_key].astype(str).to_numpy() == str(group_value)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    # Aggregate within-group by (sample, condition)
    # We reuse pseudobulk_aggregate by treating condition as "group_key" while restricting cells.
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

    # Replicate checks per condition level
    cond = meta_df[condition_key].astype(str)
    vc = cond.value_counts()
    if str(reference) not in vc.index:
        raise ValueError(f"reference={reference!r} not present in {condition_key!r} within group {group_value!r}")
    # pick test as the "other" level if exactly two; else caller should handle multi-level later
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

    # Densify counts (small)
    dense = counts_df.sparse.to_coo().toarray().astype(np.int64, copy=False)
    counts = pd.DataFrame(dense, index=counts_df.index, columns=counts_df.columns)

    # Drop genes with extremely low total counts (helps speed/stability)
    min_total = int(getattr(opts, "min_total_counts", 10))  # optional new option
    keep = (counts.sum(axis=0) >= min_total)
    counts = counts.loc[:, keep]

    if counts.shape[1] == 0:
        # No genes left after filtering; nothing to test
        return pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

    # Run PyDESeq2
    try:
        res = _run_pydeseq2(
            counts,
            metadata.rename(columns={sample_key: "sample"}),
            design_factors=["sample", condition_key],
            contrast=(condition_key, str(test), str(reference)),
            alpha=opts.alpha,
            shrink_lfc=opts.shrink_lfc,
            n_cpus=n_cpus,
        )
    except Exception as e:
        LOGGER.warning("PyDESeq2 failed for condition DE within %s=%s: %s", group_key, group_value, e)
        res = pd.DataFrame(columns=["gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])

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

    OOM-safety:
      - If dataset is huge, do stratified downsampling per group (max_cells_per_group)
      - Run rank_genes_groups on the downsampled AnnData copy only

    Returns:
      key_added used in adata.uns
    """
    import scanpy as sc

    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    labels = adata.obs[group_key].astype(str).to_numpy()

    # Downsample if needed
    n = int(adata.n_obs)
    max_per = int(spec.max_cells_per_group)
    # Always downsample if it would be too large; keep conservative defaults.
    do_down = max_per > 0 and (n > max_per * max(2, int(pd.unique(labels).size)))
    if do_down:
        idx = _stratified_downsample_indices(labels, max_per_label=max_per, random_state=spec.random_state)
        if idx.size == 0:
            raise RuntimeError("Downsampling produced 0 cells (unexpected).")
        adata_run = adata[idx].copy()
        LOGGER.info(
            "Cell-level markers: downsampled %d → %d cells (max_per_group=%d, n_groups=%d).",
            n, int(adata_run.n_obs), max_per, int(pd.unique(labels).size),
        )
    else:
        # Warning: this may be expensive on very large data; caller can raise max_per
        adata_run = adata

    sc.tl.rank_genes_groups(
        adata_run,
        groupby=group_key,
        method=str(spec.method),
        use_raw=bool(spec.use_raw),
        layer=spec.layer,
        key_added=key_added,
        n_genes=int(spec.n_genes),
        rankby_abs=bool(spec.rankby_abs),
    )

    if store:
        # Store the result back on the original adata, but keep provenance about downsampling
        adata.uns[key_added] = dict(adata_run.uns[key_added])
        adata.uns[key_added]["scomnom_meta"] = {
            "group_key": str(group_key),
            "method": str(spec.method),
            "n_genes": int(spec.n_genes),
            "use_raw": bool(spec.use_raw),
            "layer": spec.layer,
            "rankby_abs": bool(spec.rankby_abs),
            "max_cells_per_group": int(spec.max_cells_per_group),
            "random_state": int(spec.random_state),
            "downsampled": bool(do_down),
            "n_cells_used": int(adata_run.n_obs),
            "n_cells_total": int(adata.n_obs),
        }

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
    pb_spec: PseudobulkSpec,
    store_key: str = "scomnom_de",
    store: bool = True,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Returns nested dict:
      out[cluster][f"{A}_vs_{B}"]["combined" | "wilcoxon" | "logreg" | "pseudobulk_effect"] = DataFrame
    """
    import scanpy as sc
    from scanpy.get import rank_genes_groups_df

    group_key = resolve_group_key(adata, groupby=groupby, round_id=round_id, prefer_pretty=True)
    contrast_key = str(spec.contrast_key)

    if contrast_key not in adata.obs:
        raise KeyError(f"contrast_key={contrast_key!r} not in adata.obs")

    # levels
    levels = pd.Index(pd.unique(adata.obs[contrast_key].astype(str))).sort_values()
    if len(levels) < 2:
        raise ValueError(f"contrast_key={contrast_key!r} needs >=2 levels, got {list(levels)}")

    clusters = pd.Index(pd.unique(adata.obs[group_key].astype(str))).sort_values()

    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    counts_layer = pb_spec.counts_layer  # can be None

    for cl in clusters:
        cl_mask = (adata.obs[group_key].astype(str).to_numpy() == str(cl))
        if cl_mask.sum() == 0:
            continue

        out[str(cl)] = {}

        # precompute per-level masks within cluster
        cl_contrast = adata.obs.loc[cl_mask, contrast_key].astype(str)
        # map from global cell indices
        global_idx = np.where(cl_mask)[0]
        by_level_idx = {}
        for lv in levels:
            m = (cl_contrast.to_numpy() == str(lv))
            by_level_idx[str(lv)] = global_idx[np.where(m)[0]]

        # pairwise A vs B
        levels = pd.Index(pd.unique(adata.obs[contrast_key].astype(str))).sort_values()
        pairs = _select_pairs(levels.tolist(), list(spec.contrasts) if spec.contrasts else None)

        for (A, B) in pairs:
            pair_key = f"{A}_vs_{B}"

            idxA = by_level_idx.get(A, np.array([], dtype=int))
            idxB = by_level_idx.get(B, np.array([], dtype=int))

            nA = int(idxA.size)
            nB = int(idxB.size)

            if nA < spec.min_cells_per_level_in_cluster or nB < spec.min_cells_per_level_in_cluster:
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

            # Build subset AnnData for cell-level tests (downsampled)
            maskA = np.zeros(adata.n_obs, dtype=bool); maskA[idxA] = True
            maskB = np.zeros(adata.n_obs, dtype=bool); maskB[idxB] = True

            adata_sub, ds_meta = _downsample_two_level_subset(
                adata, maskA, maskB,
                max_per_level=int(spec.max_cells_per_level_in_cluster),
                random_state=int(spec.random_state),
            )

            # Ensure contrast_key exists and only A/B levels in subset
            # (it should, but be explicit)
            adata_sub = adata_sub[adata_sub.obs[contrast_key].astype(str).isin([A, B])].copy()

            # ---- pseudobulk effect (all cells)
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

            # ---- Wilcoxon
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
                )
                d = rank_genes_groups_df(adata_sub, group=A, key="__tmp_wilcoxon")
                wilcoxon_df = d.rename(
                    columns={
                        "names": "gene",
                        "logfoldchanges": "cl_logfc",
                        "scores": "cl_score",
                        "pvals": "cl_pval",
                        "pvals_adj": "cl_padj",
                    }
                )
                wilcoxon_df["gene"] = wilcoxon_df["gene"].astype(str)

            # ---- Logreg
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
                )
                d = rank_genes_groups_df(adata_sub, group=A, key="__tmp_logreg")
                # scanpy logreg exposes "scores" but coefficients may be stored in "logfoldchanges" depending on version.
                # We store both if present.
                d = d.rename(columns={"names": "gene"})
                if "logfoldchanges" in d.columns:
                    d = d.rename(columns={"logfoldchanges": "lr_coef"})
                if "scores" in d.columns:
                    d = d.rename(columns={"scores": "lr_score"})
                logreg_df = d[["gene"] + [c for c in ["lr_coef", "lr_score"] if c in d.columns]].copy()
                logreg_df["gene"] = logreg_df["gene"].astype(str)

                # ---- Combined merge
                combined = None
                # Start from pseudobulk (stable base)
                if pb_df is None or pb_df.empty:
                    combined = pd.DataFrame({"gene": []})
                else:
                    combined = pb_df.copy()

                if wilcoxon_df is not None and not wilcoxon_df.empty:
                    combined = combined.merge(wilcoxon_df, on="gene", how="outer")

                if logreg_df is not None and not logreg_df.empty:
                    combined = combined.merge(logreg_df, on="gene", how="outer")

                # attach meta columns
                combined.insert(0, "cluster", str(cl))
                combined.insert(1, "contrast_key", contrast_key)
                combined.insert(2, "A", A)
                combined.insert(3, "B", B)
                combined.insert(4, "n_cells_A", int(ds_meta["n_cells_A"]))
                combined.insert(5, "n_cells_B", int(ds_meta["n_cells_B"]))
                combined.insert(6, "downsampled", bool(ds_meta["downsampled"]))
                combined.insert(7, "n_cells_A_used", int(ds_meta["n_cells_A_used"]))
                combined.insert(8, "n_cells_B_used", int(ds_meta["n_cells_B_used"]))

                # consensus flags
                def _sgn(x):
                    if x is None or not np.isfinite(x):
                        return 0
                    if x > 0:
                        return 1
                    if x < 0:
                        return -1
                    return 0

                # safe numeric columns
                for col in ["cl_logfc", "cl_padj", "lr_coef", "pb_log2fc"]:
                    if col in combined.columns:
                        combined[col] = pd.to_numeric(combined[col], errors="coerce")

                combined["hit_wilcoxon"] = False
                if "cl_padj" in combined.columns and "cl_logfc" in combined.columns:
                    combined["hit_wilcoxon"] = (combined["cl_padj"] < float(spec.cl_alpha)) & (
                        combined["cl_logfc"].abs() >= float(spec.cl_min_abs_logfc)
                    )

                combined["hit_logreg"] = False
                if "lr_coef" in combined.columns:
                    combined["hit_logreg"] = combined["lr_coef"].abs() >= float(spec.lr_min_abs_coef)

                combined["hit_pseudobulk"] = False
                if "pb_log2fc" in combined.columns:
                    combined["hit_pseudobulk"] = combined["pb_log2fc"].abs() >= float(spec.pb_min_abs_log2fc)

                combined["sign_agree_pb_wilcoxon"] = False
                if "pb_log2fc" in combined.columns and "cl_logfc" in combined.columns:
                    combined["sign_agree_pb_wilcoxon"] = (
                        combined["pb_log2fc"].apply(_sgn) == combined["cl_logfc"].apply(_sgn)
                    )

                # tier
                hits = combined[["hit_wilcoxon", "hit_logreg", "hit_pseudobulk"]].sum(axis=1)
                tier = np.where(
                    (hits == 3) & (combined.get("sign_agree_pb_wilcoxon", True)),
                    "Tier1",
                    np.where(hits >= 2, "Tier2", np.where(hits >= 1, "Tier3", "None")),
                )
                combined["consensus_tier"] = tier

                # consensus score
                score = hits.astype(int)
                score = score + np.where(combined["consensus_tier"] == "Tier1", 2, 0)

                # penalty for sign disagreement when both effects meaningful
                if "pb_log2fc" in combined.columns and "cl_logfc" in combined.columns:
                    disagree = (combined["sign_agree_pb_wilcoxon"] == False) & (
                        combined["pb_log2fc"].abs() > 0.5
                    ) & (combined["cl_logfc"].abs() > 0.25)
                    score = score - np.where(disagree, 1, 0)

                combined["consensus_score"] = score.astype(int)

                # sort
                # tier order: Tier1, Tier2, Tier3, None
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

                # summary row
                n_t1 = int((combined["consensus_tier"] == "Tier1").sum()) if "consensus_tier" in combined.columns else 0
                n_t2 = int((combined["consensus_tier"] == "Tier2").sum()) if "consensus_tier" in combined.columns else 0
                n_t3 = int((combined["consensus_tier"] == "Tier3").sum()) if "consensus_tier" in combined.columns else 0
                top10 = combined.loc[combined["consensus_tier"] == "Tier1", "gene"].head(10).tolist() if "gene" in combined.columns else []
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
                        n_tier1=n_t1,
                        n_tier2=n_t2,
                        n_tier3=n_t3,
                        top10_tier1_genes=",".join(map(str, top10)),
                    )
                )

    if store:
        adata.uns.setdefault(store_key, {})
        adata.uns[store_key].setdefault("contrast_conditional", {})
        adata.uns[store_key]["contrast_conditional"]["group_key"] = str(group_key)
        adata.uns[store_key]["contrast_conditional"]["contrast_key"] = str(contrast_key)
        adata.uns[store_key]["contrast_conditional"]["counts_layer"] = str(counts_layer) if counts_layer else None
        adata.uns[store_key]["contrast_conditional"]["spec"] = {
            **spec.__dict__,
            "methods": list(spec.methods),
        }
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
