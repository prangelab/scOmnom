from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence
import json

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from .config import ClusterAnnotateConfig
from .logging_utils import init_logging
from . import io_utils, plot_utils
from .io_utils import get_celltypist_model, resolve_msigdb_gene_sets
from .plot_utils import _extract_series, _normalize_array
from datetime import datetime, timezone



LOGGER = logging.getLogger(__name__)

# Single pretty cluster label column
CLUSTER_LABEL_KEY = "cluster_label"


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_active_round(
    adata: ad.AnnData,
    round_id: str,
    *,
    publish_decoupler: bool = True,
) -> None:
    """
    Canonical linkage contract.
    - Sets active round
    - Mirrors round labels into canonical obs[cluster_key]
    - Mirrors round pretty labels into canonical obs["cluster_label"]
    - Best-effort color sync for both cluster_key and cluster_label
    - Publishes decoupler/pseudobulk into top-level uns for forward compatibility
    """
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"set_active_round: round_id {round_id!r} not found")

    r = rounds[round_id]
    adata.uns["active_cluster_round"] = round_id

    # -----------------------------
    # 1) Canonical clustering labels
    # -----------------------------
    cluster_key = r.get("cluster_key", None)
    labels_obs_key = r.get("labels_obs_key", None)

    if cluster_key and labels_obs_key and labels_obs_key in adata.obs:
        adata.obs[str(cluster_key)] = adata.obs[str(labels_obs_key)]
    elif cluster_key and cluster_key in adata.obs:
        # already present
        pass

    if cluster_key and cluster_key in adata.obs:
        if not pd.api.types.is_categorical_dtype(adata.obs[cluster_key]):
            adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

        # palette sync: if per-round colors exist, mirror to canonical cluster_key colors
        try:
            if labels_obs_key:
                src = f"{labels_obs_key}_colors"
                dst = f"{cluster_key}_colors"
                if src in adata.uns:
                    adata.uns[dst] = list(adata.uns[src])
        except Exception:
            pass

    # -----------------------------
    # 2) Canonical pretty labels (cluster_label)
    # -----------------------------
    # Prefer the round-linked annotation key, else fallback to the conventional name.
    ann = r.get("annotation", {}) if isinstance(r.get("annotation", {}), dict) else {}
    pretty_key = ann.get("pretty_cluster_key", f"{CLUSTER_LABEL_KEY}__{round_id}")

    if isinstance(pretty_key, str) and pretty_key in adata.obs:
        adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]
        if not pd.api.types.is_categorical_dtype(adata.obs[CLUSTER_LABEL_KEY]):
            adata.obs[CLUSTER_LABEL_KEY] = adata.obs[CLUSTER_LABEL_KEY].astype("category")

        # palette sync for cluster_label alias
        try:
            src = f"{pretty_key}_colors"
            dst = f"{CLUSTER_LABEL_KEY}_colors"
            if src in adata.uns:
                adata.uns[dst] = list(adata.uns[src])
        except Exception:
            pass

    # -----------------------------
    # 3) Publish decoupler active view to top-level uns
    # -----------------------------
    if publish_decoupler:
        _publish_decoupler_from_round_to_top_level(
            adata,
            round_id=round_id,
            resources=("msigdb", "progeny", "dorothea"),
            publish_pseudobulk=True,
            clear_missing=True,
        )



def _ensure_cluster_rounds(adata: ad.AnnData) -> None:
    """
    Ensure a minimal rounds scaffold exists in .uns.

    Layout:
      adata.uns["cluster_rounds"] : dict[round_id -> metadata]
      adata.uns["cluster_round_order"] : list[str]
      adata.uns["active_cluster_round"] : str | None
    """
    adata.uns.setdefault("cluster_rounds", {})
    adata.uns.setdefault("cluster_round_order", [])
    adata.uns.setdefault("active_cluster_round", None)


def _next_round_index(adata: ad.AnnData) -> int:
    _ensure_cluster_rounds(adata)
    existing = list(adata.uns["cluster_rounds"].keys())

    idxs: list[int] = []
    for rid in existing:
        if not isinstance(rid, str) or not rid.startswith("r"):
            continue
        i = 1
        while i < len(rid) and rid[i].isdigit():
            i += 1
        if i == 1:
            continue
        try:
            idxs.append(int(rid[1:i]))
        except Exception:
            pass

    return (max(idxs) + 1) if idxs else 0


def _make_round_id(idx: int, suffix: str) -> str:
    return f"r{idx}_{suffix}"


def _register_round(
    adata: ad.AnnData,
    *,
    round_id: str,
    cluster_key: str,
    labels_obs_key: str,
    kind: str,
    best_resolution: float | None,
    sweep: dict | None,
    cfg_snapshot: dict | None,
    parent_round_id: str | None = None,
    notes: str | None = None,
    # --- optional schema extras (safe defaults) ---
    cluster_id_map: dict[str, str] | None = None,
    cluster_renumbering: dict[str, str] | None = None,
    cache_labels: bool = False,
    compacting: dict | None = None,
) -> None:
    _ensure_cluster_rounds(adata)

    # -----------------------------
    # Cluster sizes (for audit + downstream sanity checks)
    # -----------------------------
    cluster_sizes: dict[str, int] = {}
    if isinstance(labels_obs_key, str) and labels_obs_key in adata.obs:
        try:
            vc = adata.obs[labels_obs_key].astype(str).value_counts()
            cluster_sizes = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            cluster_sizes = {}

    # -----------------------------
    # Identity maps if not provided
    # -----------------------------
    if cluster_id_map is None:
        # Default: identity mapping over observed cluster ids (if present), else empty
        if cluster_sizes:
            cluster_id_map = {cid: cid for cid in cluster_sizes.keys()}
        else:
            cluster_id_map = {}

    if cluster_renumbering is None:
        # Default: identity renumbering for the NEW ids (values of cluster_id_map)
        new_ids = sorted({str(v) for v in cluster_id_map.values()})
        cluster_renumbering = {nid: nid for nid in new_ids}

    # -----------------------------
    # Optional cached labels copy (convenience; keep off by default)
    # -----------------------------
    labels_cache: list[str] | None = None
    if cache_labels and isinstance(labels_obs_key, str) and labels_obs_key in adata.obs:
        try:
            labels_cache = adata.obs[labels_obs_key].astype(str).tolist()
        except Exception:
            labels_cache = None

    # -----------------------------
    # Round payload
    # -----------------------------
    payload: dict[str, object] = {
        "round_id": str(round_id),
        "parent_round_id": None if parent_round_id is None else str(parent_round_id),
        "created_utc": _utc_now_iso(),
        "notes": "" if notes is None else str(notes),

        "cluster_key": str(cluster_key),
        "labels_obs_key": str(labels_obs_key),

        "kind": str(kind),
        "best_resolution": None if best_resolution is None else float(best_resolution),
        "sweep": sweep,
        "cfg": cfg_snapshot,

        # --- spec-ish linkage & audit helpers ---
        "cluster_sizes": cluster_sizes,                 # {cluster_id: n_cells}
        "cluster_id_map": dict(cluster_id_map),         # {old_id: new_id} (identity for non-compacted rounds)
        "cluster_renumbering": dict(cluster_renumbering),  # {new_id: renumbered_id} (often identity)
        "labels": labels_cache,                         # optional convenience copy (can be large)

        # precreate slots for later (existing)
        "annotation": {},
        "decoupler": {},
        "qc": {},
        "stability": {},
        "diagnostics": {},

        "compacting": {} if compacting is None else dict(compacting),
    }

    adata.uns["cluster_rounds"][round_id] = payload

    if round_id not in adata.uns["cluster_round_order"]:
        adata.uns["cluster_round_order"].append(round_id)

    adata.uns["active_cluster_round"] = round_id



def _get_cluster_pseudobulk_df(
    adata: ad.AnnData,
    *,
    store_key: str = "pseudobulk",
) -> pd.DataFrame:
    """
    Reconstruct genes x clusters DataFrame from adata.uns[store_key].

    Raises if missing or malformed.
    """
    pb = adata.uns.get(store_key, None)
    if not isinstance(pb, dict):
        raise KeyError(f"Missing pseudobulk store at adata.uns[{store_key!r}]")

    required = {"genes", "clusters", "expr"}
    missing = required.difference(pb.keys())
    if missing:
        raise KeyError(f"Malformed pseudobulk store at adata.uns[{store_key!r}]; missing: {sorted(missing)}")

    genes = np.asarray(pb["genes"])
    clusters = np.asarray(pb["clusters"])
    expr = np.asarray(pb["expr"])

    if expr.ndim != 2 or expr.shape[0] != genes.size or expr.shape[1] != clusters.size:
        raise ValueError(
            f"Malformed pseudobulk matrix: expr{expr.shape}, genes={genes.size}, clusters={clusters.size}"
        )

    return pd.DataFrame(
        expr,
        index=pd.Index(genes, name="gene"),
        columns=pd.Index(clusters, name="cluster"),
    )


def _store_cluster_pseudobulk(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    agg: str = "mean",
    use_raw_like: bool = True,
    prefer_layers: tuple[str, ...] = ("counts_raw", "counts_cb"),
    store_key: str = "pseudobulk",
) -> None:
    """
    Compute and store cluster-level pseudobulk expression (genes x clusters) in adata.uns[store_key].

    Storage format is compact + reusable:
      adata.uns[store_key] = {
        "level": "cluster",
        "cluster_key": ...,
        "agg": ...,
        "use_raw_like": ...,
        "prefer_layers": [...],
        "genes": np.ndarray[str],
        "clusters": np.ndarray[str],
        "expr": np.ndarray[float32]  # shape (n_genes, n_clusters)
      }

    This does NOT return anything; downstream consumers reconstruct a DataFrame via
    `_get_cluster_pseudobulk_df()`.
    """
    from scipy import sparse

    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in adata.obs")

    agg = (agg or "mean").lower().strip()
    if agg not in {"mean", "median"}:
        agg = "mean"

    # -----------------------------
    # Choose expression source
    # -----------------------------
    X = adata.X
    genes = adata.var_names

    if use_raw_like:
        picked = None
        for layer in prefer_layers:
            if layer in adata.layers:
                picked = layer
                X = adata.layers[layer]
                genes = adata.var_names
                break

        if picked is not None:
            LOGGER.info("Pseudobulk using adata.layers[%r].", picked)
        elif adata.raw is not None:
            LOGGER.info("Pseudobulk using adata.raw.X.")
            X = adata.raw.X
            genes = adata.raw.var_names
        else:
            LOGGER.info("Pseudobulk requested raw-like but none found; using adata.X.")

    cl = adata.obs[cluster_key].astype(str).to_numpy()
    clusters = pd.Index(pd.unique(cl), dtype=str, name="cluster")

    if sparse.issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = np.asarray(X)

    LOGGER.info(
        "Computing cluster pseudobulk: cluster_key=%r, agg=%s over %d clusters.",
        cluster_key,
        agg,
        len(clusters),
    )

    expr_cols: dict[str, np.ndarray] = {}

    for c in clusters:
        idx = np.where(cl == c)[0]
        if idx.size == 0:
            continue

        if sparse.issparse(X_csr):
            sub = X_csr[idx, :]
            if agg == "mean":
                vec = np.asarray(sub.mean(axis=0)).ravel()
            else:
                vec = np.median(sub.toarray(), axis=0)
        else:
            sub = X_csr[idx, :]
            vec = sub.mean(axis=0) if agg == "mean" else np.median(sub, axis=0)

        expr_cols[str(c)] = vec.astype(np.float32, copy=False)

    if not expr_cols:
        raise RuntimeError("Pseudobulk: no clusters produced aggregated expression.")

    expr_mat = np.stack([expr_cols[c] for c in expr_cols.keys()], axis=1).astype(np.float32, copy=False)
    clusters_out = np.array(list(expr_cols.keys()), dtype=object)

    adata.uns[store_key] = {
        "level": "cluster",
        "cluster_key": cluster_key,
        "agg": agg,
        "use_raw_like": bool(use_raw_like),
        "prefer_layers": list(prefer_layers),
        "genes": np.asarray(genes, dtype=object),
        "clusters": clusters_out,
        "expr": expr_mat,  # (genes x clusters)
    }

    LOGGER.info(
        "Stored pseudobulk in adata.uns[%r]: %d genes × %d clusters.",
        store_key,
        expr_mat.shape[0],
        expr_mat.shape[1],
    )


def _celltypist_entropy_margin_mask(
    prob_matrix: pd.DataFrame,
    *,
    entropy_abs_limit: float,
    entropy_quantile: float,
    margin_min: float,
) -> tuple[np.ndarray, dict]:
    """
    Compute a boolean mask of "good" CellTypist predictions based on:
      good = (H <= max(H_abs, H_q)) AND (margin >= margin_min)

    where:
      H = -sum(p log p)
      margin = p1 - p2  (top1 minus top2 probabilities)

    Returns:
      mask (np.ndarray[bool]) aligned to prob_matrix rows
      stats (dict) with thresholds and summary
    """
    P = prob_matrix.to_numpy(dtype=np.float64, copy=False)  # (n_cells, n_classes)
    n = P.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool), {"n_cells": 0}

    # --- entropy ---
    eps = 1e-12
    P_clip = np.clip(P, eps, 1.0)
    entropy = -np.sum(P_clip * np.log(P_clip), axis=1)

    # --- margin via top-2 (O(n) per row using partition) ---
    # top2: two largest values, unsorted
    top2 = np.partition(P, kth=-2, axis=1)[:, -2:]  # (n, 2)
    p1 = np.max(top2, axis=1)
    p2 = np.min(top2, axis=1)
    margin = p1 - p2

    # hybrid entropy cutoff
    H_q = float(np.quantile(entropy, float(entropy_quantile)))
    H_abs = float(entropy_abs_limit)
    H_cut = max(H_abs, H_q)

    mask = (entropy <= H_cut) & (margin >= float(margin_min))

    stats = {
        "n_cells": int(n),
        "kept": int(mask.sum()),
        "kept_frac": float(mask.mean()),
        "entropy_abs_limit": H_abs,
        "entropy_quantile": float(entropy_quantile),
        "entropy_q_value": H_q,
        "entropy_cut_used": H_cut,
        "margin_min": float(margin_min),
        "entropy_summary": {
            "min": float(np.min(entropy)),
            "p10": float(np.percentile(entropy, 10)),
            "median": float(np.median(entropy)),
            "p90": float(np.percentile(entropy, 90)),
            "max": float(np.max(entropy)),
        },
        "margin_summary": {
            "min": float(np.min(margin)),
            "p10": float(np.percentile(margin, 10)),
            "median": float(np.median(margin)),
            "p90": float(np.percentile(margin, 90)),
            "max": float(np.max(margin)),
        },
    }
    return mask, stats


def _maybe_build_bio_mask(
    cfg: ClusterAnnotateConfig,
    celltypist_proba: Optional[pd.DataFrame],
    n_obs: int,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Build a bio mask once per run. If unavailable or unsafe, returns (None, stats).
    """
    stats: dict = {"mode": getattr(cfg, "bio_mask_mode", "entropy_margin")}

    if not getattr(cfg, "bio_guided_clustering", False):
        stats["disabled_reason"] = "bio_guided_clustering=False"
        return None, stats

    mode = getattr(cfg, "bio_mask_mode", "entropy_margin")
    if mode == "none":
        stats["disabled_reason"] = "bio_mask_mode=none"
        return None, stats

    if celltypist_proba is None or celltypist_proba.empty:
        stats["disabled_reason"] = "no_celltypist_probability_matrix"
        return None, stats

    if mode != "entropy_margin":
        stats["disabled_reason"] = f"unknown_mode={mode}"
        return None, stats

    mask, mstats = _celltypist_entropy_margin_mask(
        celltypist_proba,
        entropy_abs_limit=float(getattr(cfg, "bio_entropy_abs_limit", 0.5)),
        entropy_quantile=float(getattr(cfg, "bio_entropy_quantile", 0.7)),
        margin_min=float(getattr(cfg, "bio_margin_min", 0.10)),
    )
    stats.update(mstats)

    # safety gate: if too few pass, disable bio metrics
    min_cells = int(getattr(cfg, "bio_mask_min_cells", 500))
    min_frac = float(getattr(cfg, "bio_mask_min_frac", 0.05))
    kept = int(stats.get("kept", 0))
    kept_frac = float(stats.get("kept_frac", 0.0))

    if kept < min_cells or kept_frac < min_frac:
        stats["disabled_reason"] = (
            f"too_few_cells_passed (kept={kept}, kept_frac={kept_frac:.3f}, "
            f"min_cells={min_cells}, min_frac={min_frac})"
        )
        return None, stats

    stats["disabled_reason"] = None
    return mask, stats


def _ensure_embedding(adata: ad.AnnData, embedding_key: str) -> str:
    """
    Ensure the chosen embedding exists; if not, try to recover from integration metadata.
    Returns the actual embedding key to use.
    """
    if embedding_key in adata.obsm:
        return embedding_key

    # Try integration metadata if present
    if "integration" in adata.uns:
        best = adata.uns["integration"].get("best_embedding")
        if best and best in adata.obsm:
            LOGGER.warning(
                "Embedding key '%s' not found. Falling back to integration best_embedding='%s'.",
                embedding_key,
                best,
            )
            return best

    raise KeyError(
        f"Embedding key '{embedding_key}' not found in adata.obsm and no usable fallback found."
    )


def _compute_resolutions(cfg: ClusterAnnotateConfig) -> np.ndarray:
    return np.linspace(cfg.res_min, cfg.res_max, cfg.n_resolutions, endpoint=True)


def _res_key(r: float | str) -> str:
    """Canonical resolution key string (3 decimals, for external-facing keys)."""
    return f"{float(r):.3f}"


def _centroid_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute a centroid-based separation score in the given embedding X.

    Parameters
    ----------
    X : ndarray, shape (n_cells, n_dims)
    labels : ndarray, shape (n_cells,)

    Returns
    -------
    float
        Mean centroid-based separation across clusters. NaN if <2 clusters.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    centroids = []
    for cid in unique:
        mask = labels == cid
        if not np.any(mask):
            continue
        centroids.append(X[mask].mean(axis=0))
    centroids = np.vstack(centroids)
    k = centroids.shape[0]
    if k < 2:
        return float("nan")

    diff = centroids[:, None, :] - centroids[None, :, :]
    D = np.linalg.norm(diff, axis=2)  # (k, k)

    s_vals = []
    for i in range(k):
        d_i = D[i].copy()
        d_i[i] = np.inf
        a_i = float(np.min(d_i))  # nearest other centroid
        b_i = float(np.mean(d_i[np.isfinite(d_i)])) if np.isfinite(d_i).any() else 0.0
        denom = max(a_i, b_i)
        if denom <= 0.0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom
        s_vals.append(s_i)

    return float(np.mean(s_vals)) if s_vals else float("nan")


# -------------------------------------------------------------------------
# CellTypist: single precompute (per-cell labels + probabilities)
# -------------------------------------------------------------------------
def _precompute_celltypist(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Run CellTypist once to obtain per-cell predictions and probability matrix.

    Policy (per your requirement):
      - Only trust counts-like layers: "counts_raw" then "counts_cb".
      - NEVER assume adata.raw is counts-like.
      - If neither counts layer exists, fall back to adata.X *as-is* (no normalize/log),
        because X is integrated and may already be log-transformed.

    Returns
    -------
    labels : np.ndarray or None
        1-D array of per-cell predicted labels.
    prob_matrix : pd.DataFrame or None
        Probability matrix with shape (n_obs, n_classes), index aligned to adata.obs_names.
    """
    if cfg.celltypist_model is None:
        LOGGER.info("No CellTypist model provided; skipping CellTypist precompute.")
        return None, None

    try:
        LOGGER.info("Running CellTypist precompute (predictions + probabilities).")

        # -----------------------------
        # Select expression source
        # -----------------------------
        picked_layer: Optional[str] = None
        X_src = None

        for layer in ("counts_raw", "counts_cb"):
            if layer in adata.layers:
                picked_layer = layer
                X_src = adata.layers[layer]
                break

        if picked_layer is not None:
            LOGGER.info("CellTypist input: using counts-like layer adata.layers[%r].", picked_layer)
            # Make a lightweight AnnData with the same obs/var but counts-like X
            adata_ct = ad.AnnData(
                X=X_src,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata_ct.obs_names = adata.obs_names.copy()
            adata_ct.var_names = adata.var_names.copy()

            # Correct preprocessing for CellTypist (expects log1p normalized counts)
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)

        else:
            # Fall back to integrated assay (adata.X) WITHOUT reprocessing
            LOGGER.warning(
                "CellTypist input: no counts-like layers found ('counts_raw'/'counts_cb'). "
                "Falling back to adata.X as-is (no normalize_total/log1p), since X is typically integrated/log space."
            )
            # Copy just enough to be safe; keep X as-is
            adata_ct = adata.copy()

        # -----------------------------
        # Run CellTypist
        # -----------------------------
        model_path = get_celltypist_model(cfg.celltypist_model)
        from celltypist.models import Model
        import celltypist

        LOGGER.info("Loading CellTypist model from %s", model_path)
        model = Model.load(str(model_path))

        preds = celltypist.annotate(
            adata_ct,
            model=model,
            majority_voting=False,
        )

        # -----------------------------
        # Extract labels
        # -----------------------------
        raw = preds.predicted_labels
        if isinstance(raw, pd.DataFrame):
            labels = raw.squeeze(axis=1).to_numpy().ravel()
        elif isinstance(raw, pd.Series):
            labels = raw.to_numpy().ravel()
        else:
            labels = np.asarray(raw).ravel()

        if labels.size != adata.n_obs:
            LOGGER.warning(
                "CellTypist returned %d labels for %d cells; ignoring CellTypist outputs.",
                int(labels.size),
                int(adata.n_obs),
            )
            return None, None

        # -----------------------------
        # Probability matrix (align to original adata)
        # -----------------------------
        prob_matrix = preds.probability_matrix
        if not isinstance(prob_matrix, pd.DataFrame) or prob_matrix.empty:
            LOGGER.warning("CellTypist returned no/empty probability_matrix; returning labels only.")
            return labels, None

        # Ensure index alignment to original adata.obs_names
        # (celltypist should preserve obs_names, but be defensive)
        try:
            prob_matrix = prob_matrix.loc[adata.obs_names]
        except Exception:
            # last resort: reindex (may introduce NaNs if mismatch)
            prob_matrix = prob_matrix.reindex(adata.obs_names)

        LOGGER.info(
            "CellTypist precompute completed: %d labels, probability_matrix shape=%s (input=%s).",
            int(labels.size),
            tuple(prob_matrix.shape),
            f"layer:{picked_layer}" if picked_layer is not None else "adata.X(as-is)",
        )
        return labels, prob_matrix

    except Exception as e:
        LOGGER.warning(
            "CellTypist precompute failed: %s. Proceeding without biological metrics.",
            e,
        )
        return None, None


# -------------------------------------------------------------------------
# Biological metrics (per resolution)
# -------------------------------------------------------------------------
def _compute_bio_homogeneity(
    labels: np.ndarray,
    bio_labels: np.ndarray,
) -> float:
    """
    Cluster-level biological homogeneity metric.

    For each cluster:
    - find majority CellTypist label
    - take fraction of cells with that label
    Returns mean fraction across clusters in [0, 1].
    """
    df = pd.DataFrame({"cl": labels, "bio": bio_labels})
    groups = df.groupby("cl")
    homs: List[float] = []

    for _, g in groups:
        vc = g["bio"].value_counts()
        if vc.empty:
            continue
        maj = vc.iloc[0] / len(g)
        homs.append(float(maj))

    return float(np.mean(homs)) if homs else 0.0


def _compute_bio_fragmentation(
    labels: np.ndarray,
    bio_labels: np.ndarray,
    frac_thr: float = 0.15,
) -> float:
    """
    Biological fragmentation penalty.

    For each cluster C_j:
        - count how many CellTypist labels have fraction >= frac_thr
        - let k_j = this count - 1 (minimum 0)
    Return mean k_j across clusters (0 = perfectly homogeneous).
    """
    df = pd.DataFrame({"cl": labels, "bio": bio_labels})
    groups = df.groupby("cl")
    frags: List[float] = []

    for _, g in groups:
        vc = g["bio"].value_counts(normalize=True)
        if vc.empty:
            continue
        k = int((vc >= frac_thr).sum()) - 1
        frags.append(float(max(k, 0)))

    return float(np.mean(frags)) if frags else 0.0


# -------------------------------------------------------------------------
# Resolution-selection data structures and helpers
# -------------------------------------------------------------------------
@dataclass
class ResolutionMetrics:
    resolutions: List[float]
    silhouette: Dict[float, float]
    cluster_counts: Dict[float, int]
    cluster_sizes: Dict[float, np.ndarray]
    labels_per_resolution: Dict[float, np.ndarray]
    ari_adjacent: Optional[Dict[Tuple[float, float], float]] = None
    # Optional biological metrics (per resolution)
    penalized: Optional[Dict[float, float]] = None
    bio_homogeneity: Optional[Dict[float, float]] = None
    bio_fragmentation: Optional[Dict[float, float]] = None
    bio_ari: Optional[Dict[float, float]] = None
    n_bio_labels: Optional[int] = None


@dataclass
class ResolutionSelectionConfig:
    stability_threshold: float = 0.85
    min_plateau_len: int = 3
    max_cluster_jump_frac: float = 0.4
    min_cluster_size: int = 20
    tiny_cluster_size: int = 20
    w_stab: float = 0.50
    w_sil: float = 0.35
    w_tiny: float = 0.15
    # Biological weights + flag
    w_hom: float = 0.0
    w_frag: float = 0.0
    w_bioari: float = 0.0
    use_bio: bool = False


@dataclass
class Plateau:
    resolutions: List[float]
    mean_stability: float


@dataclass
class ResolutionSelectionResult:
    best_resolution: float
    scores: Dict[float, float]
    stability: Dict[float, float]
    tiny_cluster_penalty: Dict[float, float]
    plateaus: List[Plateau]
    bio_homogeneity: Optional[Dict[float, float]] = None
    bio_fragmentation: Optional[Dict[float, float]] = None
    bio_ari: Optional[Dict[float, float]] = None


def _compute_ari_adjacent(
    resolutions: Sequence[float],
    labels_per_resolution: Dict[float, np.ndarray],
) -> Dict[Tuple[float, float], float]:
    """Compute ARI between adjacent resolutions."""
    ari_adjacent: Dict[Tuple[float, float], float] = {}
    sorted_res = sorted(resolutions)
    for r1, r2 in zip(sorted_res[:-1], sorted_res[1:]):
        labels1 = labels_per_resolution[r1]
        labels2 = labels_per_resolution[r2]
        ari = adjusted_rand_score(labels1, labels2)
        ari_adjacent[(r1, r2)] = float(ari)
    return ari_adjacent


def _compute_smoothed_stability(
    resolutions: Sequence[float],
    ari_adjacent: Dict[Tuple[float, float], float],
) -> Dict[float, float]:
    """
    Smoothed ARI-based stability per resolution.
    For resolution r_i we average ARI(r_{i-1}, r_i) and ARI(r_i, r_{i+1}) where available.
    """
    sorted_res = sorted(resolutions)
    stab: Dict[float, float] = {}

    for i, r in enumerate(sorted_res):
        terms: List[float] = []
        if i > 0:
            r_prev = sorted_res[i - 1]
            key = (r_prev, r)
            if key in ari_adjacent:
                terms.append(ari_adjacent[key])
        if i < len(sorted_res) - 1:
            r_next = sorted_res[i + 1]
            key = (r, r_next)
            if key in ari_adjacent:
                terms.append(ari_adjacent[key])
        stab[r] = float(np.mean(terms)) if terms else 0.0
    return stab


def _detect_plateaus(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
    stability: Dict[float, float],
) -> List[Plateau]:
    """
    Detect plateau segments:
    - contiguous in sorted resolutions
    - stability >= threshold
    - cluster count does not jump more than max_cluster_jump_frac
    - median cluster size >= min_cluster_size
    - minimum cluster size >= 5

    NOTE: deliberately structural-only; no biological metrics used here.
    """
    sorted_res = sorted(metrics.resolutions)
    plateaus: List[Plateau] = []
    current_segment: List[float] = []

    for idx, r in enumerate(sorted_res):
        stab_ok = stability.get(r, 0.0) >= config.stability_threshold

        if idx > 0:
            r_prev = sorted_res[idx - 1]
            n_prev = metrics.cluster_counts[r_prev]
            n_curr = metrics.cluster_counts[r]
            jump = robust_cluster_jump(n_prev, n_curr, alpha=10)
            jump_ok = jump <= config.max_cluster_jump_frac
        else:
            jump_ok = True

        sizes = metrics.cluster_sizes[r]
        median_size = float(np.median(sizes)) if sizes.size > 0 else 0.0
        size_ok = median_size >= config.min_cluster_size
        min_ok = (sizes.size == 0) or (sizes.min() >= 5)

        if stab_ok and jump_ok and size_ok and min_ok:
            current_segment.append(r)
        else:
            if len(current_segment) >= config.min_plateau_len:
                mean_stab = float(np.mean([stability[x] for x in current_segment]))
                plateaus.append(
                    Plateau(
                        resolutions=current_segment.copy(),
                        mean_stability=mean_stab,
                    )
                )
            current_segment = []

    if len(current_segment) >= config.min_plateau_len:
        mean_stab = float(np.mean([stability[x] for x in current_segment]))
        plateaus.append(
            Plateau(
                resolutions=current_segment.copy(),
                mean_stability=mean_stab,
            )
        )

    return plateaus


def _normalize_scores(d: Dict[float, float]) -> Dict[float, float]:
    """Normalize dict values to [0, 1] to make them comparable."""
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax == vmin:
        return {k: 0.0 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}


def compute_tiny_cluster_penalty(cluster_sizes: np.ndarray, tiny_threshold: int) -> float:
    """
    Combined tiny-cluster penalty:
    1. penalty_cluster_fraction: fraction of clusters that are NOT tiny
    2. penalty_cell_fraction: fraction of cells NOT inside tiny clusters

    Both are in [0,1]. Final score is the mean of the two.
    """
    total_clusters = len(cluster_sizes)
    total_cells = np.sum(cluster_sizes)

    if total_clusters == 0 or total_cells == 0:
        return 1.0

    tiny_mask = cluster_sizes < tiny_threshold
    n_tiny = np.sum(tiny_mask)
    cells_in_tiny = np.sum(cluster_sizes[tiny_mask])

    frac_tiny_clusters = n_tiny / total_clusters
    penalty_cluster_fraction = 1.0 - frac_tiny_clusters

    frac_cells_in_tiny = cells_in_tiny / total_cells
    penalty_cell_fraction = 1.0 - frac_cells_in_tiny

    combined_penalty = 0.5 * (penalty_cluster_fraction + penalty_cell_fraction)
    return float(combined_penalty)


def robust_cluster_jump(k_prev, k_curr, alpha=10) -> float:
    """
    Robust jump metric:
    jump = |k_curr - k_prev| / max(k_prev, alpha)
    Prevents division by very small k.
    """
    denom = max(k_prev, alpha)
    return abs(k_curr - k_prev) / denom


def select_best_resolution(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
) -> ResolutionSelectionResult:
    """
    Resolution selection with strict structural primacy, plateau-like bio guidance,
    explicit SearchSet construction, and parsimony-aware selection.

    Key invariants:
      - Structural plateau is the first and strongest signal.
      - Bio plateau only exists if biological metrics are FLAT + CONTIGUOUS.
      - If bio plateau is absent, biology acts ONLY as a guardrail (never an attractor).
      - All snapping / vetoes are constrained to the SearchSet.
      - Parsimony: smallest resolution within ε of max score is preferred.
    """

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------
    ari_adjacent = metrics.ari_adjacent or _compute_ari_adjacent(
        resolutions=metrics.resolutions,
        labels_per_resolution=metrics.labels_per_resolution,
    )
    stability = _compute_smoothed_stability(metrics.resolutions, ari_adjacent)

    # ------------------------------------------------------------------
    # Structural metrics
    # ------------------------------------------------------------------
    sil_norm = _normalize_scores(metrics.silhouette)

    tiny_penalty = {
        float(r): compute_tiny_cluster_penalty(
            metrics.cluster_sizes[float(r)], config.tiny_cluster_size
        )
        for r in metrics.resolutions
    }
    tiny_norm = _normalize_scores(tiny_penalty)
    stab_norm = _normalize_scores(stability)

    # ------------------------------------------------------------------
    # Bio availability
    # ------------------------------------------------------------------
    use_bio = (
        config.use_bio
        and metrics.bio_homogeneity is not None
        and metrics.bio_fragmentation is not None
        and metrics.bio_ari is not None
    )

    hom_norm = _normalize_scores(metrics.bio_homogeneity or {})
    frag_norm = _normalize_scores(metrics.bio_fragmentation or {})
    frag_good = {r: 1.0 - frag_norm.get(r, 0.0) for r in frag_norm}
    bioari_norm = _normalize_scores(metrics.bio_ari or {})

    # ------------------------------------------------------------------
    # Composite score (ranking only, never global veto)
    # ------------------------------------------------------------------
    def composite(r: float) -> float:
        s = (
            config.w_stab * stab_norm.get(r, 0.0)
            + config.w_sil * sil_norm.get(r, 0.0)
            + config.w_tiny * tiny_norm.get(r, 0.0)
        )
        if use_bio:
            s += (
                config.w_hom * hom_norm.get(r, 0.0)
                + config.w_frag * frag_good.get(r, 0.0)
                + config.w_bioari * bioari_norm.get(r, 0.0)
            )
        return float(s)

    all_scores = {float(r): composite(float(r)) for r in metrics.resolutions}

    # ------------------------------------------------------------------
    # Resolution sets
    # ------------------------------------------------------------------
    sorted_res = sorted(float(r) for r in metrics.resolutions)
    interior = sorted_res[1:-1]

    min_stab_ok = 0.60
    feasible = [r for r in interior if stability.get(r, 0.0) >= min_stab_ok]

    # Complexity guardrail (always applied if bio available)
    if use_bio and metrics.n_bio_labels:
        max_clusters = 2.5 * metrics.n_bio_labels
        feasible = [
            r for r in feasible
            if metrics.cluster_counts.get(r, 0) <= max_clusters
        ]

    SearchSet = feasible if feasible else interior

    # ------------------------------------------------------------------
    # Helper: parsimony-aware argmax
    # ------------------------------------------------------------------
    def pick_parsimonious(cands, eps=0.03):
        if not cands:
            return None
        best = max(cands, key=lambda r: all_scores.get(r, -np.inf))
        best_val = all_scores[best]
        near = [r for r in cands if all_scores.get(r, -np.inf) >= (1 - eps) * best_val]
        return min(near) if near else best

    # ------------------------------------------------------------------
    # Structural plateau (gold standard)
    # ------------------------------------------------------------------
    plateaus = _detect_plateaus(metrics, config, stability)
    if plateaus:
        best_plateau = max(
            plateaus,
            key=lambda p: (p.mean_stability, len(p.resolutions)),
        )
        plateau_res = [float(r) for r in best_plateau.resolutions]
        plateau_res = [r for r in plateau_res if r in SearchSet]

        # ---- BIO PLATEAU (plateau-like, not permissive) ----
        bio_plateau = None
        if use_bio and plateau_res:
            bioari_vals = [bioari_norm.get(r, np.nan) for r in plateau_res]
            if np.isfinite(bioari_vals).all():
                spread = max(bioari_vals) - min(bioari_vals)
                if spread < 0.05:  # flatness criterion
                    bio_plateau = plateau_res

        cands = bio_plateau if bio_plateau else plateau_res
        best = pick_parsimonious(cands)
        return ResolutionSelectionResult(
            best_resolution=best,
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            bio_homogeneity=metrics.bio_homogeneity,
            bio_fragmentation=metrics.bio_fragmentation,
            bio_ari=metrics.bio_ari,
        )

    # ------------------------------------------------------------------
    # No structural plateau → knee / fallback (biology is guardrail only)
    # ------------------------------------------------------------------
    def stability_knee(cands):
        vals = [stability.get(r, 0.0) for r in cands]
        if not vals:
            return None
        m = max(vals)
        thr = 0.95 * m
        for r in sorted(cands):
            if stability.get(r, 0.0) >= thr:
                return r
        return None

    knee = stability_knee(SearchSet)
    if knee is not None:
        best = knee
    else:
        best = pick_parsimonious(SearchSet)

    return ResolutionSelectionResult(
        best_resolution=best,
        scores=all_scores,
        stability=stability,
        tiny_cluster_penalty=tiny_penalty,
        plateaus=plateaus,
        bio_homogeneity=metrics.bio_homogeneity,
        bio_fragmentation=metrics.bio_fragmentation,
        bio_ari=metrics.bio_ari,
    )


# -------------------------------------------------------------------------
# Resolution sweep and stability
# -------------------------------------------------------------------------
def _resolution_sweep(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    celltypist_labels: Optional[np.ndarray],
    bio_mask: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, object], Dict[str, np.ndarray]]:
    """
    Sweep over a range of Leiden resolutions and compute:
    - Centroid-based separation score
    - Number of clusters and cluster sizes
    - ARI matrix between all resolutions
    - Composite stability score with plateau-aware selection
    - OPTIONAL: CellTypist-guided biological metrics (homogeneity, fragmentation, ARI_bio)
    """
    resolutions = _compute_resolutions(cfg)
    res_list = [float(r) for r in resolutions]

    clusterings_float: Dict[float, np.ndarray] = {}
    silhouette_scores: List[float] = []
    n_clusters_list: List[int] = []
    penalized_scores: List[float] = []
    cluster_sizes: Dict[float, np.ndarray] = {}

    bio_hom: Dict[float, float] = {}
    bio_frag: Dict[float, float] = {}
    bio_ari: Dict[float, float] = {}

    X = adata.obsm[embedding_key]

    use_bio = (
            bool(getattr(cfg, "bio_guided_clustering", False))
            and (celltypist_labels is not None)
            and (bio_mask is not None)
    )

    if getattr(cfg, "bio_guided_clustering", False) and not use_bio:
        LOGGER.warning(
            "bio_guided_clustering=True, but biological metrics are unavailable "
            "(missing CellTypist labels and/or bio_mask). Using structural metrics only."
        )

    if getattr(cfg, "bio_guided_clustering", False) and celltypist_labels is None:
        LOGGER.warning(
            "bio_guided_clustering=True, but CellTypist labels are unavailable. "
            "Resolution sweep will use structural metrics only."
        )

    n_bio_labels_masked: Optional[int] = None
    if use_bio:
        # number of unique CellTypist labels among MASKED cells
        n_bio_labels_masked = int(
            pd.unique(celltypist_labels[bio_mask]).size
        )

    for res in resolutions:
        res_f = float(res)
        key = f"{cfg.label_key}_{res_f:.2f}"
        LOGGER.info("Running Leiden clustering at resolution %.2f -> key '%s'", res_f, key)
        sc.tl.leiden(
            adata,
            resolution=res_f,
            key_added=key,
            random_state=cfg.random_state,
            flavor="igraph",
        )
        labels = adata.obs[key].to_numpy()
        clusterings_float[res_f] = labels

        vc = pd.Series(labels).value_counts().sort_index()
        n_clusters = int(vc.size)
        n_clusters_list.append(n_clusters)
        sizes = vc.to_numpy(dtype=int)
        cluster_sizes[res_f] = sizes

        sil = _centroid_silhouette(X, labels)
        silhouette_scores.append(sil)
        penalized_scores.append(sil - cfg.penalty_alpha * n_clusters)

        LOGGER.info(
            "Resolution %.2f: %d clusters, centroid_score=%.3f, penalized=%.3f",
            res_f,
            n_clusters,
            sil,
            penalized_scores[-1],
        )

        if use_bio:
            m = bio_mask
            # defensive alignment check
            if m.shape[0] != labels.shape[0]:
                raise ValueError("bio_mask length does not match number of cells.")

            labels_m = labels[m]
            bio_m = celltypist_labels[m]

            # If mask makes this resolution unusable (should be rare), skip bio metrics
            if labels_m.size < 2 or np.unique(labels_m).size < 2:
                LOGGER.warning(
                    "Resolution %.2f: bio mask leaves too few usable cells; skipping bio metrics.",
                    res_f,
                )
            else:
                hom = _compute_bio_homogeneity(labels_m, bio_m)
                frag = _compute_bio_fragmentation(labels_m, bio_m)
                ari_bio = adjusted_rand_score(labels_m, bio_m)

                bio_hom[res_f] = float(hom)
                bio_frag[res_f] = float(frag)
                bio_ari[res_f] = float(ari_bio)

                LOGGER.info(
                    "Resolution %.2f (masked): bio_homogeneity=%.3f, bio_fragmentation=%.3f, bio_ARI=%.3f",
                    res_f,
                    hom,
                    frag,
                    ari_bio,
                )

    # ARI across all resolutions
    col_names = [f"{r:.2f}" for r in res_list]
    ari_matrix = pd.DataFrame(index=col_names, columns=col_names, dtype=float)
    for i, r1 in enumerate(res_list):
        for j, r2 in enumerate(res_list):
            ari = adjusted_rand_score(clusterings_float[r1], clusterings_float[r2])
            ari_matrix.iat[i, j] = float(ari)

    metrics = ResolutionMetrics(
        resolutions=res_list,
        silhouette={r: s for r, s in zip(res_list, silhouette_scores)},
        penalized={r: p for r, p in zip(res_list, penalized_scores)},
        cluster_counts={r: n for r, n in zip(res_list, n_clusters_list)},
        cluster_sizes=cluster_sizes,
        labels_per_resolution=clusterings_float,
        n_bio_labels=n_bio_labels_masked,
    )

    if bio_hom and bio_frag and bio_ari:
        metrics.bio_homogeneity = bio_hom
        metrics.bio_fragmentation = bio_frag
        metrics.bio_ari = bio_ari
    else:
        metrics.bio_homogeneity = None
        metrics.bio_fragmentation = None
        metrics.bio_ari = None

    sel_cfg = ResolutionSelectionConfig(
        stability_threshold=getattr(cfg, "stability_threshold", 0.85),
        min_plateau_len=getattr(cfg, "min_plateau_len", 3),
        max_cluster_jump_frac=getattr(cfg, "max_cluster_jump_frac", 0.4),
        min_cluster_size=getattr(cfg, "min_cluster_size", 20),
        tiny_cluster_size=getattr(cfg, "tiny_cluster_size", 20),
        w_stab=getattr(cfg, "w_stab", 0.50),
        w_sil=getattr(cfg, "w_sil", 0.35),
        w_tiny=getattr(cfg, "w_tiny", 0.15),
        w_hom=getattr(cfg, "w_hom", 0.0),
        w_frag=getattr(cfg, "w_frag", 0.0),
        w_bioari=getattr(cfg, "w_bioari", 0.0),
        use_bio=use_bio,
    )

    selection = select_best_resolution(metrics, sel_cfg)
    best_res = float(selection.best_resolution)

    LOGGER.info(
        "Selected optimal resolution %.3f (composite score=%.3f)",
        best_res,
        selection.scores[best_res],
    )

    sweep: Dict[str, object] = {
        "resolutions": np.array(res_list, dtype=float),
        "silhouette_scores": silhouette_scores,
        "n_clusters": n_clusters_list,
        "penalized_scores": penalized_scores,
        "ari_matrix": ari_matrix,
        "composite_scores": [selection.scores[r] for r in res_list],
        "stability_scores": [selection.stability[r] for r in res_list],
        "tiny_cluster_penalty": [selection.tiny_cluster_penalty[r] for r in res_list],
        "cluster_sizes": cluster_sizes,
        "plateaus": [
            {"resolutions": p.resolutions, "mean_stability": p.mean_stability}
            for p in selection.plateaus
        ],
        "selection_config": asdict(sel_cfg),
    }

    if selection.bio_homogeneity is not None:
        sweep["bio_homogeneity"] = [selection.bio_homogeneity.get(r, np.nan) for r in res_list]
        sweep["bio_fragmentation"] = [selection.bio_fragmentation.get(r, np.nan) for r in res_list]
        sweep["bio_ari"] = [selection.bio_ari.get(r, np.nan) for r in res_list]
    else:
        sweep["bio_homogeneity"] = None
        sweep["bio_fragmentation"] = None
        sweep["bio_ari"] = None

    if use_bio:
        sweep["bio_mask_stats"] = adata.uns.get("cluster_and_annotate", {}).get("bio_mask", None)
    else:
        sweep["bio_mask_stats"] = None

    clusterings_str: Dict[str, np.ndarray] = {
        _res_key(r): labs for r, labs in clusterings_float.items()
    }

    return best_res, sweep, clusterings_str


def _subsampling_stability(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    best_res: float,
) -> List[float]:
    """
    Subsampling stability analysis:
    - Cluster full data at best_res (reference)
    - Repeat subsampling of cells and recompute clustering
    - Compute ARI vs reference clustering on overlapping cells
    """
    ref_key = f"{cfg.label_key}_stab_ref"
    LOGGER.info("Computing reference clustering for stability at resolution %.3f", best_res)
    sc.tl.leiden(
        adata,
        resolution=float(best_res),
        key_added=ref_key,
        random_state=cfg.random_state,
        flavor="igraph",
    )
    ref_labels = adata.obs[ref_key].copy()

    rng = np.random.default_rng(cfg.random_state)
    stability_aris: List[float] = []

    for i in range(cfg.stability_repeats):
        rng_i = np.random.default_rng(cfg.random_state + i)
        n_sub = int(round(cfg.subsample_frac * adata.n_obs))
        cells = rng_i.choice(adata.obs_names.to_numpy(), size=n_sub, replace=False)
        sub = adata[cells].copy()

        LOGGER.info("Stability repeat %d/%d: %d cells", i + 1, cfg.stability_repeats, n_sub)

        sc.pp.neighbors(sub, use_rep=embedding_key)
        sc.tl.leiden(
            sub,
            resolution=float(best_res),
            key_added=f"{cfg.label_key}_sub",
            random_state=cfg.random_state + i,
            flavor="igraph",
        )

        overlap = adata.obs_names.intersection(sub.obs_names)
        ari = adjusted_rand_score(
            ref_labels.loc[overlap],
            sub.obs.loc[overlap, f"{cfg.label_key}_sub"],
        )
        stability_aris.append(float(ari))

    mean_ari = float(np.mean(stability_aris)) if stability_aris else float("nan")
    LOGGER.info(
        "Subsampling stability: mean ARI over %d repeats = %.3f",
        cfg.stability_repeats,
        mean_ari,
    )
    return stability_aris


def _apply_final_clustering(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    best_res: float,
) -> None:
    """
    Apply the chosen resolution as the final clustering into cfg.label_key
    and set a consistent color palette.
    """
    LOGGER.info(
        "Applying final Leiden clustering at resolution %.3f -> key '%s'",
        best_res,
        cfg.label_key,
    )
    sc.tl.leiden(
        adata,
        resolution=float(best_res),
        key_added=cfg.label_key,
        random_state=cfg.random_state,
        flavor="igraph",
    )

    try:
        from scanpy.plotting.palettes import default_102

        cats = adata.obs[cfg.label_key].cat.categories
        adata.uns[f"{cfg.label_key}_colors"] = default_102[: len(cats)]
    except Exception as e:
        LOGGER.warning("Could not set Leiden color palette: %s", e)


def _run_celltypist_annotation(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    cluster_key: str,
    round_id: str | None = None,
    precomputed_labels: Optional[np.ndarray] = None,
    precomputed_proba: Optional[pd.DataFrame] = None,
) -> dict[str, str] | None:
    """
    Attach CellTypist annotations to AnnData, *round-aware*.

    Writes:
      - cell-level labels: adata.obs[cfg.celltypist_label_key]
      - cluster-majority labels (round-scoped): adata.obs[f"{cfg.celltypist_cluster_label_key}__{round_id}"]
        (also updates adata.obs[cfg.celltypist_cluster_label_key] as alias to latest)
      - pretty cluster labels (round-scoped): adata.obs[f"{CLUSTER_LABEL_KEY}__{round_id}"]
        (also updates adata.obs[CLUSTER_LABEL_KEY] as alias to latest)

    Returns a dict of keys created (for plotting), or None if skipped.
    """
    if cfg.celltypist_model is None:
        LOGGER.info("No CellTypist model provided; skipping annotation.")
        return None

    if cluster_key not in adata.obs:
        raise KeyError(
            f"_run_celltypist_annotation: cluster_key '{cluster_key}' not found in adata.obs"
        )

    # Determine round_id (best effort)
    if round_id is None:
        rid = adata.uns.get("active_cluster_round", None)
        round_id = str(rid) if rid else "r0"

    # Round-scoped keys (avoid overwriting across rounds)
    cell_key = cfg.celltypist_label_key
    cluster_ct_base = cfg.celltypist_cluster_label_key
    cluster_ct_key = f"{cluster_ct_base}__{round_id}"
    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"

    # --------------------------------------------------------------
    # A) CellTypist predictions (cell-level + probabilities)
    # --------------------------------------------------------------
    if precomputed_labels is not None:
        if precomputed_labels.shape[0] != adata.n_obs:
            raise ValueError("precomputed_labels length does not match adata.n_obs.")
        LOGGER.info("Using precomputed CellTypist labels for final annotation.")
        adata.obs[cell_key] = precomputed_labels

        if precomputed_proba is not None:
            try:
                pm = precomputed_proba.loc[adata.obs_names]
            except Exception:
                pm = precomputed_proba.reindex(adata.obs_names)
            adata.obsm["celltypist_proba"] = pm.to_numpy()
            adata.uns["celltypist_proba_columns"] = list(pm.columns)

    else:
        # Fallback path (kept for safety; your pipeline usually uses precompute)
        LOGGER.info("Running CellTypist on main AnnData for final annotation (fallback).")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        model_path = get_celltypist_model(cfg.celltypist_model)
        from celltypist.models import Model
        import celltypist

        model = Model.load(str(model_path))

        predictions = celltypist.annotate(
            adata,
            model=model,
            majority_voting=cfg.celltypist_majority_voting,
        )

        raw = predictions.predicted_labels
        if isinstance(raw, dict) and "majority_voting" in raw:
            cell_level_labels = raw["majority_voting"]
        else:
            cell_level_labels = raw

        adata.obs[cell_key] = cell_level_labels

        if hasattr(predictions, "probability_matrix"):
            pm = predictions.probability_matrix
            try:
                pm = pm.loc[adata.obs_names]
            except Exception:
                pm = pm.reindex(adata.obs_names)
            adata.obsm["celltypist_proba"] = pm.to_numpy()
            adata.uns["celltypist_proba_columns"] = list(pm.columns)

    # --------------------------------------------------------------
    # B) Cluster-level majority CellTypist label (ROUND-SCOPED)
    #    Mask-aware: only use "good" cells (high-confidence CellTypist)
    #    to decide the cluster label; otherwise mark as Unknown.
    # --------------------------------------------------------------
    # Rebuild mask from stored proba if available; fall back to "all True".
    bio_mask = None
    bio_mask_stats = None
    try:
        # If probabilities were stored (either from precompute or fallback), reconstruct a DataFrame
        # so we can apply the exact same entropy+margin rule used during BISC.
        if "celltypist_proba" in adata.obsm and "celltypist_proba_columns" in adata.uns:
            pm = pd.DataFrame(
                adata.obsm["celltypist_proba"],
                index=adata.obs_names,
                columns=list(adata.uns["celltypist_proba_columns"]),
            )
            bio_mask, bio_mask_stats = _celltypist_entropy_margin_mask(
                pm,
                entropy_abs_limit=float(getattr(cfg, "bio_entropy_abs_limit", 0.5)),
                entropy_quantile=float(getattr(cfg, "bio_entropy_quantile", 0.7)),
                margin_min=float(getattr(cfg, "bio_margin_min", 0.10)),
            )
    except Exception as e:
        LOGGER.warning("CellTypist mask reconstruction failed; proceeding unmasked. (%s)", e)
        bio_mask = None
        bio_mask_stats = None

    if bio_mask is None or bio_mask.shape[0] != adata.n_obs:
        bio_mask = np.ones((adata.n_obs,), dtype=bool)

    # Cluster-wise label: majority over masked cells only.
    # If a cluster has too few masked cells, assign Unknown.
    min_masked_cells = int(getattr(cfg, "pretty_label_min_masked_cells", 25) or 25)
    min_masked_frac = float(getattr(cfg, "pretty_label_min_masked_frac", 0.10) or 0.10)

    clust_vals = adata.obs[cluster_key].astype(str)
    ct_vals = adata.obs[cell_key].astype(str)

    tmp = pd.DataFrame(
        {
            "cluster": clust_vals.to_numpy(),
            "ct": ct_vals.to_numpy(),
            "masked": bio_mask,
        },
        index=adata.obs_names,
    )

    # Precompute cluster sizes to enforce min fraction
    cluster_sizes = tmp.groupby("cluster").size().to_dict()

    majority_map: dict[str, str] = {}
    for c, g in tmp.groupby("cluster", sort=False):
        g_masked = g[g["masked"]]
        n_total = int(cluster_sizes.get(c, len(g)))
        n_masked = int(g_masked.shape[0])

        if n_masked < min_masked_cells or (n_total > 0 and (n_masked / n_total) < min_masked_frac):
            majority_map[str(c)] = "Unknown"
            continue

        vc = g_masked["ct"].value_counts()
        majority_map[str(c)] = str(vc.idxmax()) if not vc.empty else "Unknown"

    adata.obs[cluster_ct_key] = clust_vals.map(majority_map).astype("category")

    # Alias for "latest" behavior
    adata.obs[cluster_ct_base] = adata.obs[cluster_ct_key]

    # --------------------------------------------------------------
    # C) Pretty labels (ROUND-SCOPED), also keep CLUSTER_LABEL_KEY alias
    #    Cluster name becomes Unknown if its masked-majority is Unknown.
    # --------------------------------------------------------------
    cl_ids = clust_vals
    maj = adata.obs[cluster_ct_key].astype(str).fillna("Unknown")

    pretty_labels = "C" + cl_ids.str.zfill(2) + ": " + maj
    adata.obs[pretty_key] = pretty_labels.astype("category")
    adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]  # alias to latest round

    # Stable palette for round-scoped pretty labels + alias
    try:
        from scanpy.plotting.palettes import default_102
        cats = adata.obs[pretty_key].cat.categories
        adata.uns[f"{pretty_key}_colors"] = default_102[: len(cats)]
        adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = adata.uns[f"{pretty_key}_colors"]
    except Exception as e:
        LOGGER.warning("Could not set pretty-label palette: %s", e)

    # --------------------------------------------------------------
    # D) Store linkage + mask stats into the round dict (if present)
    # --------------------------------------------------------------
    try:
        rounds = adata.uns.get("cluster_rounds", {})
        if isinstance(rounds, dict) and round_id in rounds and isinstance(rounds[round_id], dict):
            rounds[round_id].setdefault("annotation", {})
            rounds[round_id]["annotation"].update(
                {
                    "celltypist_cell_key": cell_key,
                    "celltypist_cluster_key": cluster_ct_key,
                    "pretty_cluster_key": pretty_key,
                    "cluster_key_used": cluster_key,
                    "pretty_label_masked": True,
                    "pretty_label_min_masked_cells": int(min_masked_cells),
                    "pretty_label_min_masked_frac": float(min_masked_frac),
                }
            )
            if bio_mask_stats is not None:
                rounds[round_id].setdefault("bio_mask", {})
                rounds[round_id]["bio_mask"]["annotation_mask_stats"] = bio_mask_stats
            adata.uns["cluster_rounds"] = rounds
    except Exception as e:
        LOGGER.warning("Failed to store round annotation linkage/mask stats: %s", e)

    LOGGER.info(
        "CellTypist annotation done for round '%s' using cluster_key='%s'. "
        "Wrote: cell='%s', cluster='%s', pretty='%s' (+ aliases).",
        round_id,
        cluster_key,
        cell_key,
        cluster_ct_key,
        pretty_key,
    )

    return {
        "round_id": str(round_id),
        "cluster_key": str(cluster_key),
        "celltypist_cell_key": str(cell_key),
        "celltypist_cluster_key": str(cluster_ct_key),
        "pretty_cluster_key": str(pretty_key),
    }



def _final_real_silhouette_qc(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    figdir: Path,
    *,
    cluster_key: str,
    round_id: str | None = None,
) -> Optional[float]:
    """
    Compute true silhouette for the final clustering (QC only) and optionally plot histogram.

    ROUNDS-ONLY:
      Stores under adata.uns["cluster_rounds"][round_id]["qc"]["real_silhouette_*"].
    """
    if cluster_key not in adata.obs:
        LOGGER.warning(
            "final_real_silhouette_qc: cluster_key '%s' not in adata.obs; skipping.",
            cluster_key,
        )
        return None

    if round_id is None:
        rid = adata.uns.get("active_cluster_round", None)
        round_id = str(rid) if rid else None

    labels = adata.obs[cluster_key].to_numpy()
    unique = np.unique(labels)
    if unique.size < 2:
        LOGGER.warning(
            "final_real_silhouette_qc: <2 clusters (%d); skipping.", unique.size
        )
        return None

    X = adata.obsm[embedding_key]
    LOGGER.info("Computing true silhouette for final clustering (QC only)...")
    sil_values = silhouette_samples(X, labels, metric="euclidean")
    sil_mean = float(np.mean(sil_values))

    # ---- store into round ----
    if round_id:
        rounds = adata.uns.get("cluster_rounds", {})
        if isinstance(rounds, dict) and round_id in rounds and isinstance(rounds[round_id], dict):
            rounds[round_id].setdefault("qc", {})
            rounds[round_id]["qc"]["real_silhouette_final"] = sil_mean
            rounds[round_id]["qc"]["real_silhouette_summary"] = {
                "mean": sil_mean,
                "median": float(np.median(sil_values)),
                "p10": float(np.percentile(sil_values, 10)),
                "p90": float(np.percentile(sil_values, 90)),
            }
            adata.uns["cluster_rounds"] = rounds

    LOGGER.info("Final true silhouette (mean) = %.3f", sil_mean)

    if cfg.make_figures:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sil_values, bins=40, color="steelblue", alpha=0.85)
        ax.axvline(sil_mean, color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Silhouette value")
        ax.set_ylabel("Number of cells")
        title_key = f"{cluster_key}" + (f" [{round_id}]" if round_id else "")
        ax.set_title(f"Final clustering: true silhouette ({title_key}, mean={sil_mean:.3f})")
        fig.tight_layout()
        plot_utils.save_multi("final_real_silhouette", figdir, fig)

    return sil_mean


def _dc_run_method(
    *,
    method: str,
    mat: "pd.DataFrame",
    net: "pd.DataFrame",
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
    min_n: int = 5,
    consensus_methods: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> "pd.DataFrame":
    """
    Run a decoupler method on an expression matrix + network and return activities.

    Returns activities as (sources x samples).
    """
    import numpy as np
    import pandas as pd
    import decoupler as dc

    if mat is None or not isinstance(mat, pd.DataFrame) or mat.empty:
        raise ValueError("decoupler: 'mat' must be a non-empty pandas DataFrame.")
    if net is None or not isinstance(net, pd.DataFrame) or net.empty:
        raise ValueError("decoupler: 'net' must be a non-empty pandas DataFrame.")

    method = (method or "consensus").lower().strip()

    # -----------------------------
    # Normalize / validate network
    # -----------------------------
    for col in (source, target):
        if col not in net.columns:
            raise KeyError(f"decoupler: net missing required column '{col}'")

    net_use = net.copy()
    net_use[source] = net_use[source].astype(str)
    net_use[target] = net_use[target].astype(str)

    # decoupler 2.x expects standard column names: source/target/weight
    if source != "source":
        net_use = net_use.rename(columns={source: "source"})
    if target != "target":
        net_use = net_use.rename(columns={target: "target"})
    if weight is None or weight not in net_use.columns:
        net_use["weight"] = 1.0
    elif weight != "weight":
        net_use = net_use.rename(columns={weight: "weight"})

    # -----------------------------
    # Normalize expression to samples x genes
    # -----------------------------
    genes_in_net = set(net_use["target"].unique().tolist())
    idx_overlap = len(set(map(str, mat.index)).intersection(genes_in_net))
    col_overlap = len(set(map(str, mat.columns)).intersection(genes_in_net))

    mat_sxg = mat.T.copy() if idx_overlap >= col_overlap else mat.copy()
    mat_sxg = mat_sxg.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    common_genes = [g for g in mat_sxg.columns.astype(str) if g in genes_in_net]
    if not common_genes:
        raise ValueError("decoupler: no overlap between mat genes and net targets.")

    mat_sxg = mat_sxg.loc[:, common_genes]
    net_use = net_use[net_use["target"].isin(common_genes)].copy()

    # Filter sources by min_n targets
    tgt_counts = net_use.groupby("source")["target"].nunique()
    keep_sources = tgt_counts[tgt_counts >= int(min_n)].index.astype(str).tolist()
    net_use = net_use[net_use["source"].isin(keep_sources)].copy()
    if net_use.empty:
        raise ValueError(f"decoupler: after min_n={min_n} filtering, net is empty.")

    # -----------------------------
    # Available methods (decoupler 2.x)
    # -----------------------------
    try:
        available = {m.lower() for m in dc.mt.show_methods()}
    except Exception:
        # Very defensive fallback
        available = {m.lower() for m in dir(dc.mt) if not m.startswith("_")}

    def _run_one(m: str) -> pd.DataFrame:
        m = (m or "").lower().strip()
        if m not in available:
            raise ValueError(
                f"decoupler: method '{m}' not available. "
                f"Use decoupler.mt.show_methods() to inspect available methods."
            )

        func = getattr(dc.mt, m)

        # decoupler 2.x methods: func(data=..., net=...) -> (acts, pvals) or acts
        out = func(data=mat_sxg, net=net_use, verbose=verbose)

        acts = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(acts, pd.DataFrame) or acts.empty:
            raise RuntimeError(f"decoupler: method '{m}' returned empty/invalid activities.")

        # acts is samples x sources -> convert to sources x samples
        return acts.T

    if method == "consensus":
        methods = list(consensus_methods) if consensus_methods else ["ulm", "mlm", "wsum"]
        methods = [m.lower().strip() for m in methods if m and str(m).strip()]
        if not methods:
            raise ValueError("decoupler consensus: consensus_methods is empty.")

        ests: list[pd.DataFrame] = []
        errs: list[str] = []
        for m in methods:
            try:
                ests.append(_run_one(m))
            except Exception as e:
                errs.append(f"{m}: {e}")

        if not ests:
            raise RuntimeError("decoupler consensus: all constituent methods failed: " + " | ".join(errs))

        # Align + average
        common_sources = set(ests[0].index)
        common_samples = set(ests[0].columns)
        for e in ests[1:]:
            common_sources &= set(e.index)
            common_samples &= set(e.columns)

        if not common_sources or not common_samples:
            raise RuntimeError("decoupler consensus: no common sources/samples across methods.")

        common_sources = sorted(common_sources)
        common_samples = sorted(common_samples)

        stack = np.stack([e.loc[common_sources, common_samples].to_numpy() for e in ests], axis=0)
        mean_est = np.nanmean(stack, axis=0)

        return pd.DataFrame(mean_est, index=common_sources, columns=common_samples)

    return _run_one(method)


def _get_round_id_default(adata: ad.AnnData, round_id: str | None) -> str:
    if round_id is not None:
        return str(round_id)
    rid = adata.uns.get("active_cluster_round", None)
    if rid is None:
        raise KeyError("No round_id provided and adata.uns['active_cluster_round'] is None.")
    return str(rid)


def _round_put_decoupler(
    adata: ad.AnnData,
    *,
    round_id: str,
    resource: str,
    payload: dict,
) -> None:
    """
    Store decoupler outputs in the round dict:
      adata.uns["cluster_rounds"][round_id]["decoupler"][resource] = payload
    """
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"Round {round_id!r} not found in adata.uns['cluster_rounds'].")

    r = rounds[round_id]
    if not isinstance(r, dict):
        raise TypeError(f"Round entry for {round_id!r} is not a dict.")

    r.setdefault("decoupler", {})
    if not isinstance(r["decoupler"], dict):
        r["decoupler"] = {}

    r["decoupler"][str(resource)] = payload
    rounds[round_id] = r
    adata.uns["cluster_rounds"] = rounds


def _publish_decoupler_from_round_to_top_level(
    adata: ad.AnnData,
    *,
    round_id: str,
    resources: tuple[str, ...] = ("msigdb", "progeny", "dorothea"),
    publish_pseudobulk: bool = True,
    clear_missing: bool = True,
) -> None:
    """
    Publish decoupler outputs from a round to top-level adata.uns so legacy code can find them:
      adata.uns["msigdb"], adata.uns["progeny"], adata.uns["dorothea"], (optionally) adata.uns["pseudobulk"]

    If clear_missing=True, deletes top-level keys for resources that are absent in the round.
    """
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"Round {round_id!r} not found in adata.uns['cluster_rounds'].")

    rinfo = rounds[round_id]
    dec = rinfo.get("decoupler", {}) if isinstance(rinfo.get("decoupler", {}), dict) else {}

    # Pseudobulk (optional publish)
    if publish_pseudobulk:
        pb_key = dec.get("pseudobulk_store_key", None)
        if isinstance(pb_key, str) and pb_key in adata.uns:
            adata.uns["pseudobulk"] = adata.uns[pb_key]
        elif clear_missing and "pseudobulk" in adata.uns:
            # Only clear if the round does not provide a usable pseudobulk pointer
            del adata.uns["pseudobulk"]

    # Nets
    for res in resources:
        payload = dec.get(res, None)
        if isinstance(payload, dict) and "activity" in payload:
            # Shallow copy is enough; activity is a DF
            adata.uns[res] = payload
            # publish display hints for plotting
            try:
                adata.uns[res].setdefault("config", {})
                adata.uns[res]["config"]["cluster_display_map"] = dec.get("cluster_display_map", None)
                adata.uns[res]["config"]["cluster_display_labels"] = dec.get("cluster_display_labels", None)
            except Exception:
                pass
        else:
            if clear_missing and res in adata.uns:
                del adata.uns[res]



def _run_msigdb(
    adata: "ad.AnnData",
    cfg: "ClusterAnnotateConfig",
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "msigdb",
) -> None:
    """
    Round-native MSigDB decoupler run.

    Reads pseudobulk from:
      adata.uns[store_key]  (genes x clusters store dict)

    Stores into:
      adata.uns["cluster_rounds"][round_id]["decoupler"][out_resource] = {
          "activity": DataFrame (clusters x pathways),
          "config": dict,
      }

    NOTE: Publishing to top-level adata.uns["msigdb"] is handled by set_active_round().
    """
    import pandas as pd

    rid = _get_round_id_default(adata, round_id)

    # -----------------------------
    # Pull pseudobulk from .uns
    # -----------------------------
    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)  # genes x clusters
    except Exception as e:
        LOGGER.warning("MSigDB: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("MSigDB: pseudobulk expr is empty; skipping.")
        return

    # -----------------------------
    # Resolve gene sets (GMT files)
    # -----------------------------
    gene_sets = getattr(cfg, "msigdb_gene_sets", None) or ["HALLMARK", "REACTOME"]

    try:
        gmt_files, used_keywords, msigdb_release = resolve_msigdb_gene_sets(gene_sets)
    except Exception as e:
        LOGGER.warning("MSigDB: failed to resolve gene sets: %s", e)
        return
    if not gmt_files:
        LOGGER.warning("MSigDB: no gene set files resolved; skipping.")
        return

    # -----------------------------
    # Method config
    # -----------------------------
    method = (
        getattr(cfg, "msigdb_method", None)
        or getattr(cfg, "decoupler_method", None)
        or "consensus"
    )
    min_n = int(getattr(cfg, "msigdb_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    # -----------------------------
    # Build decoupler network from GMTs
    # -----------------------------
    nets = []
    for gmt in gmt_files:
        try:
            net_i = io_utils.gmt_to_decoupler_net(gmt)
            nets.append(net_i)
        except Exception as e:
            LOGGER.warning("MSigDB: failed to parse GMT '%s': %s", str(gmt), e)

    if not nets:
        LOGGER.warning("MSigDB: no GMTs could be parsed into a decoupler net; skipping.")
        return

    net = pd.concat(nets, axis=0, ignore_index=True)

    for col in ("source", "target"):
        if col not in net.columns:
            LOGGER.warning("MSigDB: net missing required column '%s'; skipping.", col)
            return
    if "weight" not in net.columns:
        net["weight"] = 1.0

    # -----------------------------
    # Run decoupler via shared helper
    # -----------------------------
    try:
        est = _dc_run_method(
            method=method,
            mat=expr,
            net=net,
            source="source",
            target="target",
            weight="weight",
            min_n=min_n,
            verbose=False,
            consensus_methods=getattr(cfg, "decoupler_consensus_methods", None),
        )
        activity = est.T  # clusters x pathways
        activity.index.name = "cluster"
        activity.columns.name = "pathway"

    except Exception as e:
        LOGGER.warning("MSigDB: decoupler run failed (method=%s): %s", str(method), e)
        return

    # Build activity_by_gmt
    try:
        feature_meta = pd.DataFrame(
            {
                "term": activity.columns.astype(str),
                "gmt": [_infer_msigdb_gmt(c) for c in activity.columns.astype(str)],
            }
        ).set_index("term", drop=False)

        activity_by_gmt = msigdb_activity_by_gmt_from_activity_and_meta(
            activity,
            feature_meta=feature_meta,
            gmt_col="gmt",
        )
    except Exception as e:
        LOGGER.warning("MSigDB: failed to build activity_by_gmt; leaving empty. (%s)", e)
        feature_meta = None
        activity_by_gmt = {}

    payload = {
        "activity": activity,
        "config": {
            "method": str(method),
            "min_n_targets": int(min_n),
            "msigdb_release": msigdb_release,
            "gene_sets": [str(g) for g in gmt_files],
            "used_keywords": used_keywords,
            "input": f"{store_key}:pseudobulk_expr(genes_x_clusters)",
            "resource": "msigdb_gmt",
            "round_id": rid,
        },
        "activity_by_gmt": activity_by_gmt,  # <-- populated
        "feature_meta": feature_meta,  # <-- optional but helps audits/debug
    }

    _round_put_decoupler(adata, round_id=rid, resource=out_resource, payload=payload)

    LOGGER.info(
        "MSigDB stored in round '%s' decoupler[%r]: activity shape=%s using method=%s (min_n=%d; GMTs=%d).",
        rid,
        out_resource,
        tuple(activity.shape),
        str(method),
        int(min_n),
        len(gmt_files),
    )


def _run_progeny(
    adata: "ad.AnnData",
    cfg: "ClusterAnnotateConfig",
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "progeny",
) -> None:
    """
    Round-native PROGENy decoupler run.

    Reads pseudobulk from adata.uns[store_key].
    Stores into adata.uns["cluster_rounds"][round_id]["decoupler"][out_resource].
    """
    import decoupler as dc

    rid = _get_round_id_default(adata, round_id)

    # -----------------------------
    # Pull pseudobulk from .uns
    # -----------------------------
    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)  # genes x clusters
    except Exception as e:
        LOGGER.warning("PROGENy: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("PROGENy: pseudobulk expr is empty; skipping.")
        return

    # -----------------------------
    # Method config
    # -----------------------------
    method = (
        getattr(cfg, "progeny_method", None)
        or getattr(cfg, "decoupler_method", None)
        or "consensus"
    )
    min_n = int(getattr(cfg, "progeny_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    top_n = int(getattr(cfg, "progeny_top_n", 100) or 100)
    organism = getattr(cfg, "progeny_organism", "human") or "human"

    # -----------------------------
    # Load resource
    # -----------------------------
    try:
        net = dc.op.progeny(organism=str(organism), top=top_n)
    except Exception as e:
        LOGGER.warning("PROGENy: failed to load resource via dc.op.progeny: %s", e)
        return

    if net is None or len(net) == 0:
        LOGGER.warning("PROGENy: resource empty; skipping.")
        return

    wcol = "weight" if "weight" in net.columns else ("mor" if "mor" in net.columns else None)
    if wcol is None:
        LOGGER.warning("PROGENy: no usable weight column found (expected 'weight' or 'mor'); skipping.")
        return

    # -----------------------------
    # Run decoupler
    # -----------------------------
    try:
        est = _dc_run_method(
            method=method,
            mat=expr,
            net=net,
            source="source",
            target="target",
            weight=wcol,
            min_n=min_n,
            verbose=False,
            consensus_methods=getattr(cfg, "decoupler_consensus_methods", None),
        )
        activity = est.T  # clusters x pathways
        activity.index.name = "cluster"
        activity.columns.name = "pathway"

    except Exception as e:
        LOGGER.warning("PROGENy: decoupler run failed (method=%s): %s", str(method), e)
        return

    payload = {
        "activity": activity,
        "config": {
            "method": str(method),
            "min_n_targets": int(min_n),
            "top_n": int(top_n),
            "input": f"{store_key}:pseudobulk_expr(genes_x_clusters)",
            "resource": "progeny",
            "organism": str(organism),
            "round_id": rid,
        },
        "feature_meta": None,
    }

    _round_put_decoupler(adata, round_id=rid, resource=out_resource, payload=payload)

    LOGGER.info(
        "PROGENy stored in round '%s' decoupler[%r]: activity shape=%s using method=%s (top_n=%d, min_n=%d).",
        rid,
        out_resource,
        tuple(activity.shape),
        str(method),
        int(top_n),
        int(min_n),
    )


def _run_dorothea(
    adata: "ad.AnnData",
    cfg: "ClusterAnnotateConfig",
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "dorothea",
) -> None:
    """
    Round-native DoRothEA decoupler run.

    Reads pseudobulk from adata.uns[store_key].
    Stores into adata.uns["cluster_rounds"][round_id]["decoupler"][out_resource].
    """
    import decoupler as dc

    rid = _get_round_id_default(adata, round_id)

    # -----------------------------
    # Pull pseudobulk from .uns
    # -----------------------------
    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)  # genes x clusters
    except Exception as e:
        LOGGER.warning("DoRothEA: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("DoRothEA: pseudobulk expr is empty; skipping.")
        return

    # -----------------------------
    # Method config
    # -----------------------------
    method = (
        getattr(cfg, "dorothea_method", None)
        or getattr(cfg, "decoupler_method", None)
        or "consensus"
    )
    min_n = int(getattr(cfg, "dorothea_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    conf = getattr(cfg, "dorothea_confidence", None)
    if conf is None:
        conf = ["A", "B", "C"]
    if isinstance(conf, str):
        conf = [x.strip().upper() for x in conf.split(",") if x.strip()]
    conf = [str(x).upper() for x in conf]

    organism = getattr(cfg, "dorothea_organism", "human") or "human"

    # -----------------------------
    # Load resource + filter
    # -----------------------------
    try:
        net = dc.op.dorothea(organism=str(organism), levels=conf)
    except Exception as e:
        LOGGER.warning("DoRothEA: failed to load resource via dc.op.dorothea: %s", e)
        return

    if net is None or net.empty:
        LOGGER.warning("DoRothEA: resource empty after confidence filtering (%s); skipping.", conf)
        return

    wcol = "weight" if "weight" in net.columns else ("mor" if "mor" in net.columns else None)
    if wcol is None:
        LOGGER.warning("DoRothEA: no usable weight column found (expected 'weight' or 'mor'); skipping.")
        return

    # -----------------------------
    # Run decoupler
    # -----------------------------
    try:
        est = _dc_run_method(
            method=method,
            mat=expr,
            net=net,
            source="source",
            target="target",
            weight=wcol,
            min_n=min_n,
            verbose=False,
            consensus_methods=getattr(cfg, "decoupler_consensus_methods", None),
        )
        activity = est.T  # clusters x TFs
        activity.index.name = "cluster"
        activity.columns.name = "TF"

    except Exception as e:
        LOGGER.warning("DoRothEA: decoupler run failed (method=%s): %s", str(method), e)
        return

    payload = {
        "activity": activity,
        "config": {
            "method": str(method),
            "organism": str(organism),
            "min_n_targets": int(min_n),
            "confidence": list(conf),
            "input": f"{store_key}:pseudobulk_expr(genes_x_clusters)",
            "resource": "dorothea",
            "round_id": rid,
        },
        "feature_meta": None,
    }

    _round_put_decoupler(adata, round_id=rid, resource=out_resource, payload=payload)

    LOGGER.info(
        "DoRothEA stored in round '%s' decoupler[%r]: activity shape=%s using method=%s (conf=%s, min_n=%d).",
        rid,
        out_resource,
        tuple(activity.shape),
        str(method),
        ",".join(conf),
        int(min_n),
    )


def _round_cluster_display_map(
    adata: ad.AnnData,
    *,
    round_id: str,
    labels_obs_key: str,
) -> dict[str, str]:
    """
    Build mapping {cluster_id -> display_label} for a round.

    Prefers the round-scoped pretty labels column (CLUSTER_LABEL_KEY__{round_id}).
    Falls back to {cluster_id -> cluster_id} if pretty labels are missing.

    Assumes:
      - labels_obs_key is per-cell cluster ids (strings / categoricals)
      - pretty labels, if present, are constant within cluster ids
    """
    if labels_obs_key not in adata.obs:
        raise KeyError(f"labels_obs_key '{labels_obs_key}' not found in adata.obs")

    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"
    if pretty_key not in adata.obs:
        # no pretty labels yet
        cl = adata.obs[labels_obs_key].astype(str)
        cats = pd.unique(cl)
        return {str(c): str(c) for c in cats}

    tmp = pd.DataFrame(
        {
            "cluster_id": adata.obs[labels_obs_key].astype(str).to_numpy(),
            "pretty": adata.obs[pretty_key].astype(str).to_numpy(),
        },
        index=adata.obs_names,
    )

    # take first pretty label per cluster defensively
    m = (
        tmp.groupby("cluster_id", observed=True)["pretty"]
        .agg(lambda x: str(x.iloc[0]) if len(x) else "Unknown")
        .astype(str)
        .to_dict()
    )
    return {str(k): str(v) for k, v in m.items()}


def run_decoupler_for_round(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    round_id: str | None = None,
    publish_to_top_level_if_active: bool = True,
) -> None:
    """
    Round-native decoupler orchestrator:

    1) Computes round-scoped pseudobulk store: adata.uns[f"pseudobulk__{round_id}"]
       IMPORTANT: pseudobulk is ALWAYS computed on the round-stable cluster ids
       stored in round["labels_obs_key"] (NOT pretty labels), so downstream
       compaction/stability has stable identifiers.

    2) Runs enabled decoupler resources and stores them in:
         adata.uns["cluster_rounds"][round_id]["decoupler"][resource]

    3) Records pointers + plotting display metadata in the round:
         round["decoupler"]["pseudobulk_store_key"] = ...
         round["decoupler"]["cluster_display_map"] = {cluster_id -> pretty_label}
         round["decoupler"]["cluster_display_labels"] = [...] aligned to pseudobulk cluster order

    4) If round_id is active and publish_to_top_level_if_active=True, publishes
       the round's decoupler payloads to top-level adata.uns["msigdb"/"progeny"/"dorothea"/"pseudobulk"].

    Contract:
      - Round dict remains source of truth.
      - Plotting should use display_map/display_labels, but activities remain indexed by cluster_id.
    """
    if not getattr(cfg, "run_decoupler", False):
        LOGGER.info("Decoupler: disabled (run_decoupler=False).")
        return

    _ensure_cluster_rounds(adata)
    rid = _get_round_id_default(adata, round_id)

    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or rid not in rounds:
        raise KeyError(f"Decoupler: round {rid!r} not found in adata.uns['cluster_rounds'].")

    rinfo = rounds[rid]
    if not isinstance(rinfo, dict):
        raise TypeError(f"Decoupler: round entry for {rid!r} is not a dict.")

    labels_obs_key = rinfo.get("labels_obs_key", None)
    if not labels_obs_key or str(labels_obs_key) not in adata.obs:
        raise KeyError(
            f"Decoupler: round {rid!r} missing labels_obs_key (or not present in adata.obs). "
            f"labels_obs_key={labels_obs_key!r}"
        )
    labels_obs_key = str(labels_obs_key)

    # ------------------------------------------------------------------
    # 1) Pseudobulk on STABLE cluster ids (round labels_obs_key)
    # ------------------------------------------------------------------
    pb_key = f"pseudobulk__{rid}"

    LOGGER.info(
        "Decoupler[%s]: computing pseudobulk using labels_obs_key=%r -> store_key=%r",
        rid,
        labels_obs_key,
        pb_key,
    )

    _store_cluster_pseudobulk(
        adata,
        cluster_key=labels_obs_key,
        agg=getattr(cfg, "decoupler_pseudobulk_agg", "mean"),
        use_raw_like=bool(getattr(cfg, "decoupler_use_raw", True)),
        prefer_layers=("counts_raw", "counts_cb"),
        store_key=pb_key,
    )

    # Ensure round decoupler dict
    rinfo.setdefault("decoupler", {})
    if not isinstance(rinfo["decoupler"], dict):
        rinfo["decoupler"] = {}

    # Record pseudobulk pointer in the round (so publishing can find it)
    rinfo["decoupler"]["pseudobulk_store_key"] = pb_key

    # ------------------------------------------------------------------
    # 2) Always store DISPLAY metadata (pretty labels) for plotting
    # ------------------------------------------------------------------
    # {cluster_id -> display_label}; falls back to identity if pretty labels missing
    display_map = _round_cluster_display_map(
        adata,
        round_id=rid,
        labels_obs_key=labels_obs_key,
    )
    rinfo["decoupler"]["cluster_display_map"] = dict(display_map)

    # Also store a display labels vector aligned to the pseudobulk cluster order
    # (nice for quick plotting without re-deriving order)
    try:
        pb_store = adata.uns.get(pb_key, None)
        clusters = []
        if isinstance(pb_store, dict) and "clusters" in pb_store:
            clusters = list(np.asarray(pb_store["clusters"], dtype=object))
        rinfo["decoupler"]["cluster_display_labels"] = [
            display_map.get(str(c), str(c)) for c in clusters
        ] if clusters else None
    except Exception:
        rinfo["decoupler"]["cluster_display_labels"] = None

    rounds[rid] = rinfo
    adata.uns["cluster_rounds"] = rounds

    # ------------------------------------------------------------------
    # 3) Run resources (round-native storage)
    # ------------------------------------------------------------------
    # MSigDB
    _run_msigdb(adata, cfg, store_key=pb_key, round_id=rid, out_resource="msigdb")

    # DoRothEA
    if getattr(cfg, "run_dorothea", True):
        _run_dorothea(adata, cfg, store_key=pb_key, round_id=rid, out_resource="dorothea")

    # PROGENy
    if getattr(cfg, "run_progeny", True):
        _run_progeny(adata, cfg, store_key=pb_key, round_id=rid, out_resource="progeny")

    # ------------------------------------------------------------------
    # 4) Publish to top-level if this is the active round
    # ------------------------------------------------------------------
    active = adata.uns.get("active_cluster_round", None)
    if publish_to_top_level_if_active and (active is not None) and (str(active) == rid):
        _publish_decoupler_from_round_to_top_level(
            adata,
            round_id=rid,
            resources=("msigdb", "progeny", "dorothea"),
            publish_pseudobulk=True,
            clear_missing=True,
        )

        # Also publish display hints into each top-level resource payload (best-effort)
        try:
            dec = adata.uns["cluster_rounds"][rid].get("decoupler", {})
            dm = dec.get("cluster_display_map", None)
            dl = dec.get("cluster_display_labels", None)
            for res in ("msigdb", "progeny", "dorothea"):
                if res in adata.uns and isinstance(adata.uns[res], dict):
                    adata.uns[res].setdefault("config", {})
                    if isinstance(adata.uns[res]["config"], dict):
                        adata.uns[res]["config"]["cluster_display_map"] = dm
                        adata.uns[res]["config"]["cluster_display_labels"] = dl
        except Exception:
            pass

        LOGGER.info("Decoupler[%s]: published to top-level adata.uns for forward compatibility.", rid)


# -------------------------------------------------------------------------
# Public: Biology Informed Structural Clustering (BISC)
# -------------------------------------------------------------------------
def run_BISC(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    embedding_key: str,
    celltypist_labels: Optional[np.ndarray],
    celltypist_proba: Optional[pd.DataFrame],
    round_suffix: str = "initial_clustering",
) -> ad.AnnData:
    """
    Biology Informed Structural Clustering (BISC) — ROUNDS-ONLY

    Stores EVERYTHING needed for downstream plots/exports in the created round:
      adata.uns["cluster_rounds"][round_id] = {
        cluster_key, kind, best_resolution, sweep, cfg,
        inputs, bio_mask, stability, diagnostics, qc (added later)
      }
    """
    _ensure_cluster_rounds(adata)

    # --------------------------------------------------------------
    # Build neighbors/UMAP if missing (manual convenience)
    # --------------------------------------------------------------
    if "neighbors" not in adata.uns:
        LOGGER.info("BISC: neighbors not found; computing neighbors using embedding_key=%r", embedding_key)
        sc.pp.neighbors(adata, use_rep=embedding_key)

    if "X_umap" not in adata.obsm:
        LOGGER.info("BISC: UMAP not found; computing UMAP.")
        sc.tl.umap(adata)

    # --------------------------------------------------------------
    # Bio mask (one-time) for bio metrics during sweep
    # --------------------------------------------------------------
    bio_mask, bio_mask_stats = _maybe_build_bio_mask(cfg, celltypist_proba, adata.n_obs)

    if getattr(cfg, "bio_guided_clustering", False):
        mode = bio_mask_stats.get("mode", "unknown")
        disabled = bio_mask_stats.get("disabled_reason", None)

        if bio_mask is None:
            LOGGER.info(
                "CellTypist bio mask: DISABLED (mode=%s) — %s",
                mode,
                disabled or "unknown reason",
            )
        else:
            kept = int(bio_mask_stats.get("kept", int(np.sum(bio_mask))))
            n_cells = int(bio_mask_stats.get("n_cells", bio_mask.size))
            frac = float(bio_mask_stats.get("kept_frac", kept / max(1, n_cells)))
            LOGGER.info(
                "CellTypist bio mask: kept %d/%d cells (%.1f%%) [mode=%s]",
                kept,
                n_cells,
                100.0 * frac,
                mode,
            )

    # --------------------------------------------------------------
    # Resolution sweep + best res selection
    # --------------------------------------------------------------
    best_res, sweep, _clusterings = _resolution_sweep(
        adata,
        cfg,
        embedding_key,
        celltypist_labels=celltypist_labels,
        bio_mask=bio_mask,
    )

    # Make sweep fully round-self-contained (avoid reading any global state)
    sweep = dict(sweep) if isinstance(sweep, dict) else {}
    sweep["bio_mask_stats"] = bio_mask_stats

    # --------------------------------------------------------------
    # Subsampling stability (uses best_res)
    # --------------------------------------------------------------
    stability_aris = _subsampling_stability(adata, cfg, embedding_key, best_res)

    # --------------------------------------------------------------
    # Apply final clustering at best_res into cfg.label_key
    # --------------------------------------------------------------
    _apply_final_clustering(adata, cfg, best_res)

    # --------------------------------------------------------------
    # Create round_id + round-scoped labels key
    # --------------------------------------------------------------
    idx = _next_round_index(adata)
    round_id = _make_round_id(idx, round_suffix)

    labels_obs_key = f"{cfg.label_key}__{round_id}"
    adata.obs[labels_obs_key] = adata.obs[cfg.label_key].astype(str).astype("category")

    # Mirror colors to per-round obs key if present
    try:
        if f"{cfg.label_key}_colors" in adata.uns:
            adata.uns[f"{labels_obs_key}_colors"] = list(adata.uns[f"{cfg.label_key}_colors"])
    except Exception:
        pass

    # --------------------------------------------------------------
    # Register round (now includes identity mapping fields)
    # --------------------------------------------------------------
    # Identity map for the initial round: every cluster maps to itself
    try:
        cats = adata.obs[labels_obs_key].astype(str).astype("category").cat.categories.astype(str).tolist()
    except Exception:
        cats = sorted(pd.unique(adata.obs[labels_obs_key].astype(str)).tolist())

    identity_map = {c: c for c in cats}
    identity_renumbering = {c: c for c in sorted(set(identity_map.values()))}

    _register_round(
        adata,
        round_id=round_id,
        cluster_key=cfg.label_key,
        labels_obs_key=labels_obs_key,
        kind="BISC",
        best_resolution=float(best_res),
        sweep={
            # compact sweep snapshot (HDF5-safe later)
            "resolutions": [float(r) for r in sweep.get("resolutions", [])],
            "silhouette_scores": [float(x) for x in sweep.get("silhouette_scores", [])],
            "n_clusters": [int(x) for x in sweep.get("n_clusters", [])],
            "penalized_scores": [float(x) for x in sweep.get("penalized_scores", [])],
            "plateaus": sweep.get("plateaus", None),
            "selection_config": sweep.get("selection_config", {}),
            "bio_mask_stats": sweep.get("bio_mask_stats", None),
            "bio_homogeneity": sweep.get("bio_homogeneity", None),
            "bio_fragmentation": sweep.get("bio_fragmentation", None),
            "bio_ari": sweep.get("bio_ari", None),
        },
        cfg_snapshot=asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,
        parent_round_id=None,

        # NEW: schema-complete fields (explicit identity mapping)
        cluster_id_map=identity_map,
        cluster_renumbering=identity_renumbering,
        compacting=None,  # no compaction in initial round
        cache_labels=False,  # keep uns small
    )

    # Ensure canonical view points at this round
    set_active_round(adata, round_id, publish_decoupler=False)


    # --------------------------------------------------------------
    # Enrich round with plot-friendly diagnostics and stability/QC pointers
    # --------------------------------------------------------------
    rounds = adata.uns.get("cluster_rounds", {})
    rinfo = rounds.get(round_id, {}) if isinstance(rounds, dict) else {}

    rinfo.setdefault("inputs", {})
    rinfo["inputs"].update(
        {
            "embedding_key": str(embedding_key),
            "batch_key": getattr(cfg, "batch_key", None),
        }
    )

    rinfo.setdefault("bio_mask", {})
    rinfo["bio_mask"] = bio_mask_stats

    rinfo.setdefault("stability", {})
    rinfo["stability"]["subsampling_ari"] = [float(x) for x in stability_aris]

    # A single dict with everything plotting needs, already aligned + keyed
    res_list = [float(r) for r in rinfo.get("sweep", {}).get("resolutions", [])]
    sil_list = rinfo.get("sweep", {}).get("silhouette_scores", []) or []
    n_list = rinfo.get("sweep", {}).get("n_clusters", []) or []
    pen_list = rinfo.get("sweep", {}).get("penalized_scores", []) or []

    rinfo.setdefault("diagnostics", {})
    rinfo["diagnostics"]["tested_resolutions"] = res_list
    rinfo["diagnostics"]["silhouette_centroid"] = {
        _res_key(r): float(s) for r, s in zip(res_list, sil_list)
    }
    rinfo["diagnostics"]["cluster_counts"] = {
        _res_key(r): int(n) for r, n in zip(res_list, n_list)
    }
    rinfo["diagnostics"]["penalized_scores"] = {
        _res_key(r): float(p) for r, p in zip(res_list, pen_list)
    }

    # If your sweep stored these arrays, include them too
    # (Note: your current sweep stores tiny penalties + composite + stability arrays)
    comp_list = sweep.get("composite_scores", None)
    stab_list = sweep.get("stability_scores", None)
    tiny_list = sweep.get("tiny_cluster_penalty", None)
    if isinstance(comp_list, list) and len(comp_list) == len(res_list):
        rinfo["diagnostics"]["composite_scores"] = {
            _res_key(r): float(v) for r, v in zip(res_list, comp_list)
        }
    if isinstance(stab_list, list) and len(stab_list) == len(res_list):
        rinfo["diagnostics"]["resolution_stability"] = {
            _res_key(r): float(v) for r, v in zip(res_list, stab_list)
        }
    if isinstance(tiny_list, list) and len(tiny_list) == len(res_list):
        rinfo["diagnostics"]["tiny_cluster_penalty"] = {
            _res_key(r): float(v) for r, v in zip(res_list, tiny_list)
        }

    rounds[round_id] = rinfo
    adata.uns["cluster_rounds"] = rounds

    # --------------------------------------------------------------
    # QC silhouette -> store in round + optional plot
    # --------------------------------------------------------------
    _final_real_silhouette_qc(
        adata,
        cfg,
        embedding_key,
        Path("cluster_and_annotate") / round_id / "clustering",
        cluster_key=cfg.label_key,
        round_id=round_id,
    )

    LOGGER.info(
        "BISC complete: best_res=%.3f stored as round '%s' using cluster_key='%s'",
        float(best_res),
        round_id,
        cfg.label_key,
    )

    return adata


# =============================================================================
# Utilities: colors + timestamps
# =============================================================================
def _ensure_category_with_order(s: pd.Series, ordered_categories: Optional[List[str]] = None) -> pd.Series:
    s = s.astype(str)
    if ordered_categories is None:
        cats = sorted(pd.unique(s).tolist())
    else:
        cats = list(ordered_categories)
    return pd.Categorical(s, categories=cats, ordered=False)


def _get_palette(n: int) -> List[str]:
    """
    Deterministic palette (hex) for n categories.
    Uses matplotlib if available; otherwise falls back.
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(n)]
        # rgba -> hex
        def to_hex(rgba):
            r, g, b = rgba[:3]
            return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        return [to_hex(c) for c in colors]
    except Exception:
        # fallback
        base = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        return [base[i % len(base)] for i in range(n)]


def _sync_cluster_colors(adata, cluster_key: str) -> None:
    """
    Ensure adata.uns[f"{cluster_key}_colors"] aligns with the categorical ordering
    of adata.obs[cluster_key].
    """
    if cluster_key not in adata.obs:
        return
    s = adata.obs[cluster_key]
    if not pd.api.types.is_categorical_dtype(s):
        s = pd.Series(_ensure_category_with_order(pd.Series(s, index=adata.obs.index))).astype("category")
        adata.obs[cluster_key] = s

    cats = list(adata.obs[cluster_key].cat.categories.astype(str))
    pal_key = f"{cluster_key}_colors"
    existing = adata.uns.get(pal_key, None)

    if isinstance(existing, (list, tuple)) and len(existing) >= len(cats):
        adata.uns[pal_key] = list(existing)[: len(cats)]
        return

    adata.uns[pal_key] = _get_palette(len(cats))


def _infer_msigdb_gmt(term: str) -> str:
    """
    Best-effort GMT family inference from a decoupler 'source' / term string.
    Handles:
      - PREFIX::TERM  -> PREFIX
      - PREFIX_TERM   -> PREFIX
    """
    t = str(term)
    if "::" in t:
        p = t.split("::", 1)[0].strip()
        return p or "UNKNOWN"
    return (t.split("_", 1)[0].strip() or "UNKNOWN")


def msigdb_activity_by_gmt_from_activity_and_meta(
    activity: pd.DataFrame,
    *,
    feature_meta: Optional[pd.DataFrame] = None,
    gmt_col: str = "gmt",
) -> Dict[str, pd.DataFrame]:
    """
    Build {GMT -> activity_sub} from:
      - activity: clusters x terms
      - feature_meta: dataframe indexed by term OR has column 'term', with a gmt_col giving family.
    If feature_meta not provided, falls back to prefix inference:
      - "HALLMARK_*" -> HALLMARK
      - "REACTOME_*" -> REACTOME
      - "KEGG_*" -> KEGG
      - "WIKIPATHWAYS_*" -> WIKIPATHWAYS
      - "BIOCARTA_*" -> BIOCARTA
      - "GOBP_*"/"GOCC_*"/"GOMF_*" -> GOBP/GOCC/GOMF
      - "PREFIX::TERM" -> PREFIX
      - else first token before '_' -> token
    """
    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        return {}

    A = activity.copy()
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)

    if feature_meta is not None and isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        fm = feature_meta.copy()
        # normalize to index = term
        if fm.index.name is None or fm.index.astype(str).tolist() != fm.index.tolist():
            fm.index = fm.index.astype(str)
        if "term" in fm.columns and fm.index.name != "term":
            # if term column exists and index isn't term-like, set it
            try:
                fm = fm.set_index("term", drop=False)
            except Exception:
                pass
        fm.index = fm.index.astype(str)

        if gmt_col not in fm.columns:
            # fallback to inference
            prefixes = pd.Series(A.columns, index=A.columns).map(_infer_msigdb_gmt)
        else:
            prefixes = pd.Series(A.columns, index=A.columns).map(lambda c: fm.loc[c, gmt_col] if c in fm.index else np.nan)
            prefixes = prefixes.fillna(pd.Series(A.columns, index=A.columns).map(_infer_msigdb_gmt))
    else:
        prefixes = pd.Series(A.columns, index=A.columns).map(_infer_msigdb_gmt)

    out: Dict[str, pd.DataFrame] = {}
    for gmt, cols in prefixes.groupby(prefixes).groups.items():
        cols = list(cols)
        if not cols:
            continue
        out[str(gmt)] = A.loc[:, cols].copy()

    return out


def ensure_round_msigdb_activity_by_gmt(
    round_snapshot: Dict[str, Any],
    *,
    msigdb_key: str = "msigdb",
    activity_key: str = "activity",
    feature_meta_key: str = "feature_meta",
    gmt_col: str = "gmt",
) -> None:
    """
    Mutates round_snapshot in-place:
      round["decoupler"]["msigdb"]["activity_by_gmt"] = {...}
    """
    dec = round_snapshot.setdefault("decoupler", {})
    ms = dec.setdefault(msigdb_key, {})

    if "activity_by_gmt" in ms and isinstance(ms["activity_by_gmt"], dict) and ms["activity_by_gmt"]:
        return

    activity = ms.get(activity_key, None)
    feature_meta = ms.get(feature_meta_key, None)
    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        raise KeyError("Cannot build activity_by_gmt: round['decoupler']['msigdb']['activity'] missing/empty.")

    if feature_meta is not None and isinstance(feature_meta, dict):
        # allow dict feature_meta -> dataframe
        feature_meta = pd.DataFrame(feature_meta)

    by_gmt = msigdb_activity_by_gmt_from_activity_and_meta(activity, feature_meta=feature_meta, gmt_col=gmt_col)
    if not by_gmt:
        raise ValueError("Failed to build MSigDB activity_by_gmt (no GMT blocks produced).")

    ms["activity_by_gmt"] = by_gmt


# =============================================================================
# Compaction decision engine
# =============================================================================

@dataclass
class CompactionOutputs:
    components: List[List[str]]
    cluster_id_map: Dict[str, str]
    reverse_map: Dict[str, List[str]]
    cluster_renumbering: Dict[str, str]
    edges: pd.DataFrame
    adjacency: Dict[str, List[Tuple[str, str]]]
    decision_log: List[Dict[str, Any]]


def _as_float_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.index = x.index.astype(str)
    x.columns = x.columns.astype(str)
    return x.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _zscore_cols(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, np.nan)
    z = (df - mu) / (sd + eps)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _pair_iter(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((items[i], items[j]))
    return out


def _connected_components(nodes: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    if not nodes:
        return []
    nbr: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        nbr[u].append(v)
        nbr[v].append(u)

    seen = set()
    comps: List[List[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in nbr[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _clique_components(nodes: List[str], pass_edges_set: set[Tuple[str, str]]) -> List[List[str]]:
    if not nodes:
        return []

    adj: Dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in pass_edges_set:
        adj[u].add(v)
        adj[v].add(u)

    remaining = set(nodes)
    cliques: List[List[str]] = []

    def degree(n: str) -> int:
        return len(adj[n] & remaining)

    while remaining:
        seed = sorted(remaining, key=lambda x: (-degree(x), x))[0]
        clique = {seed}
        candidates = (adj[seed] & remaining) - clique
        while candidates:
            cand = sorted(candidates, key=lambda x: (-degree(x), x))[0]
            if all((cand in adj[m]) for m in clique):
                clique.add(cand)
            candidates = {c for c in candidates if all((c in adj[m]) for m in clique)}
            candidates.discard(cand)

        cliques.append(sorted(clique))
        remaining -= clique

    return cliques


def compact_clusters_by_multiview_agreement(
    *,
    adata,
    round_snapshot: Dict[str, Any],
    celltypist_obs_key: str,
    min_cells: int = 0,
    zscore_scope: str = "within_celltypist_label",
    similarity_metric: str = "cosine_zscore",
    grouping: str = "connected_components",
    thr_progeny: float = 0.98,
    thr_dorothea: float = 0.98,
    thr_msigdb_default: float = 0.98,
    thr_msigdb_by_gmt: Optional[Dict[str, float]] = None,
    msigdb_required: bool = True,
    # NEW: gated by cfg at callsite (pass bool from cfg)
    skip_unknown_celltypist_groups: bool = False,
) -> CompactionOutputs:
    """
    Compaction decision engine.

    Key change vs older versions:
      - Groups clusters by the ROUND-SCOPED cluster-level CellTypist label if available
        (round_snapshot["annotation"]["celltypist_cluster_key"] in adata.obs).
      - Falls back to per-cell majority (celltypist_obs_key) only if the round-scoped
        cluster-level key is missing.
      - Optionally skips Unknown/UNKNOWN groups (cfg-gated via skip_unknown_celltypist_groups).
    """
    thr_msigdb_by_gmt = dict(thr_msigdb_by_gmt or {})

    labels_obs_key = round_snapshot.get("labels_obs_key", None)
    if not labels_obs_key or labels_obs_key not in adata.obs:
        raise KeyError("round_snapshot['labels_obs_key'] missing or not in adata.obs")

    if celltypist_obs_key not in adata.obs:
        raise KeyError(f"celltypist_obs_key '{celltypist_obs_key}' not found in adata.obs")

    cluster_per_cell = adata.obs[labels_obs_key].astype(str)
    celltypist_per_cell = adata.obs[celltypist_obs_key].astype(str)

    cluster_sizes = cluster_per_cell.value_counts()
    all_clusters = sorted(cluster_sizes.index.astype(str).tolist())

    dec = round_snapshot.get("decoupler", {})
    prog = dec.get("progeny", {}).get("activity", None)
    doro = dec.get("dorothea", {}).get("activity", None)
    msig_by_gmt = dec.get("msigdb", {}).get("activity_by_gmt", None)

    if prog is None or doro is None:
        raise KeyError("Missing progeny/dorothea activity in round_snapshot['decoupler']")

    prog = _as_float_df(prog).reindex(index=all_clusters).fillna(0.0)
    doro = _as_float_df(doro).reindex(index=all_clusters).fillna(0.0)

    if msigdb_required:
        if not isinstance(msig_by_gmt, dict) or not msig_by_gmt:
            raise KeyError("Missing msigdb activity_by_gmt in round_snapshot['decoupler']['msigdb']")
        msig_by_gmt_clean: Dict[str, pd.DataFrame] = {}
        for gmt, df in msig_by_gmt.items():
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            msig_by_gmt_clean[str(gmt)] = _as_float_df(df).reindex(index=all_clusters).fillna(0.0)
        if not msig_by_gmt_clean:
            raise ValueError("msigdb_required=True but no non-empty GMT blocks found")
        msig_by_gmt = msig_by_gmt_clean
    else:
        msig_by_gmt = {}

    if zscore_scope == "global":
        prog_z_global = _zscore_cols(prog)
        doro_z_global = _zscore_cols(doro)
        msig_z_global = {g: _zscore_cols(df) for g, df in msig_by_gmt.items()}
    elif zscore_scope == "within_celltypist_label":
        prog_z_global = doro_z_global = None
        msig_z_global = None
    else:
        raise ValueError("zscore_scope must be 'within_celltypist_label' or 'global'")

    # ------------------------------------------------------------------
    # Grouping label per cluster (CellTypist)
    #
    # Prefer round-scoped cluster-level CellTypist key (mask-aware, can be Unknown)
    # from: round_snapshot["annotation"]["celltypist_cluster_key"].
    #
    # Fallback: derive grouping label per cluster from per-cell majority of celltypist_obs_key.
    # ------------------------------------------------------------------
    ann = round_snapshot.get("annotation", {})
    ann = ann if isinstance(ann, dict) else {}
    ct_cluster_key = ann.get("celltypist_cluster_key", None)

    majority_ct: Dict[str, str] = {}

    if isinstance(ct_cluster_key, str) and ct_cluster_key in adata.obs:
        # This column should be constant within each cluster; we take the first value defensively.
        tmp = pd.DataFrame(
            {
                "cluster": cluster_per_cell.values,
                "ct_cluster": adata.obs[ct_cluster_key].astype(str).values,
            }
        )
        majority_ct = (
            tmp.groupby("cluster", observed=True)["ct_cluster"]
            .agg(lambda x: str(x.iloc[0]) if len(x) else "UNKNOWN")
            .astype(str)
            .to_dict()
        )
    else:
        # Fallback: per-cell majority vote within each cluster
        tmp = pd.DataFrame({"cluster": cluster_per_cell.values, "ct": celltypist_per_cell.values})
        majority_ct = (
            tmp.groupby("cluster", observed=True)["ct"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "UNKNOWN")
            .astype(str)
            .to_dict()
        )

    # Map CellTypist-group label -> list of clusters in that group
    label_to_clusters: Dict[str, List[str]] = {}
    for cl in all_clusters:
        if min_cells and int(cluster_sizes.get(cl, 0)) < int(min_cells):
            continue
        lab = str(majority_ct.get(cl, "UNKNOWN"))
        label_to_clusters.setdefault(lab, []).append(cl)

    # Optional: do NOT compact Unknown-labeled clusters with each other
    if skip_unknown_celltypist_groups:
        label_to_clusters.pop("Unknown", None)
        label_to_clusters.pop("UNKNOWN", None)

    edge_rows: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[Tuple[str, str]]] = {}

    def _thr_for_gmt(gmt: str) -> float:
        return float(thr_msigdb_by_gmt.get(gmt, thr_msigdb_default))

    pass_edges_by_label: Dict[str, List[Tuple[str, str]]] = {}

    for ct_label, clusters in sorted(label_to_clusters.items(), key=lambda kv: kv[0]):
        clusters = sorted(clusters)
        if len(clusters) < 2:
            continue

        if zscore_scope == "within_celltypist_label":
            prog_z = _zscore_cols(prog.loc[clusters])
            doro_z = _zscore_cols(doro.loc[clusters])
            msig_z = {g: _zscore_cols(df.loc[clusters]) for g, df in msig_by_gmt.items()}
        else:
            prog_z = prog_z_global.loc[clusters]
            doro_z = doro_z_global.loc[clusters]
            msig_z = {g: msig_z_global[g].loc[clusters] for g in msig_by_gmt.keys()}

        passed_edges: List[Tuple[str, str]] = []

        for a, b in _pair_iter(clusters):
            sim_prog = _cosine(prog_z.loc[a].to_numpy(), prog_z.loc[b].to_numpy())
            sim_doro = _cosine(doro_z.loc[a].to_numpy(), doro_z.loc[b].to_numpy())

            msig_sims: Dict[str, float] = {}
            msig_pass_all = True
            msig_fail_gmts: List[str] = []
            for gmt, dfz in msig_z.items():
                s = _cosine(dfz.loc[a].to_numpy(), dfz.loc[b].to_numpy())
                msig_sims[gmt] = float(s)
                if s < _thr_for_gmt(gmt):
                    msig_pass_all = False
                    msig_fail_gmts.append(gmt)

            pass_prog = sim_prog >= float(thr_progeny)
            pass_doro = sim_doro >= float(thr_dorothea)
            pass_all = bool(pass_prog and pass_doro and msig_pass_all)

            if pass_all:
                passed_edges.append((a, b))

            row = {
                "celltypist_label": ct_label,
                "a": a,
                "b": b,
                "n_a": int(cluster_sizes.get(a, 0)),
                "n_b": int(cluster_sizes.get(b, 0)),
                "sim_progeny": float(sim_prog),
                "thr_progeny": float(thr_progeny),
                "pass_progeny": bool(pass_prog),
                "sim_dorothea": float(sim_doro),
                "thr_dorothea": float(thr_dorothea),
                "pass_dorothea": bool(pass_doro),
                "pass_msigdb_all_gmt": bool(msig_pass_all),
                "fail_msigdb_gmts": ",".join(msig_fail_gmts) if msig_fail_gmts else "",
                "pass_all": bool(pass_all),
                "similarity_metric": similarity_metric,
                "zscore_scope": zscore_scope,
                "grouping": grouping,
            }
            for gmt, s in msig_sims.items():
                row[f"sim_msigdb__{gmt}"] = float(s)
                row[f"thr_msigdb__{gmt}"] = float(_thr_for_gmt(gmt))
                row[f"pass_msigdb__{gmt}"] = bool(s >= _thr_for_gmt(gmt))

            edge_rows.append(row)

        pass_edges_by_label[ct_label] = passed_edges
        adjacency[ct_label] = passed_edges

    edges_df = pd.DataFrame(edge_rows)

    decision_log: List[Dict[str, Any]] = []
    all_components: List[List[str]] = []

    for ct_label, clusters in sorted(label_to_clusters.items(), key=lambda kv: kv[0]):
        clusters = sorted(clusters)
        if len(clusters) == 0:
            continue
        passed_edges = pass_edges_by_label.get(ct_label, [])

        if grouping == "connected_components":
            comps = _connected_components(clusters, passed_edges)
        elif grouping == "clique":
            s = set((min(u, v), max(u, v)) for (u, v) in passed_edges)
            comps = _clique_components(clusters, s)
        else:
            raise ValueError("grouping must be 'connected_components' or 'clique'")

        for comp in comps:
            if len(comp) <= 1:
                continue
            decision_log.append(
                {
                    "celltypist_label": ct_label,
                    "members": list(comp),
                    "n_members": int(len(comp)),
                    "reason": "all-pass multiview agreement within CellTypist label",
                    "grouping": grouping,
                    "thresholds": {
                        "thr_progeny": float(thr_progeny),
                        "thr_dorothea": float(thr_dorothea),
                        "thr_msigdb_default": float(thr_msigdb_default),
                        "thr_msigdb_by_gmt": dict(thr_msigdb_by_gmt),
                    },
                }
            )

        all_components.extend([list(c) for c in comps])

    covered = set(c for comp in all_components for c in comp)
    missing = [
        c
        for c in all_clusters
        if (min_cells == 0 or int(cluster_sizes.get(c, 0)) >= int(min_cells)) and c not in covered
    ]
    for c in missing:
        all_components.append([c])

    def comp_size(members: List[str]) -> int:
        return int(sum(int(cluster_sizes.get(m, 0)) for m in members))

    comps_sorted = sorted(all_components, key=lambda comp: (-comp_size(comp), [str(x) for x in comp]))

    cluster_id_map: Dict[str, str] = {}
    reverse_map: Dict[str, List[str]] = {}

    for i, members in enumerate(comps_sorted):
        new_id = f"C{i:02d}"
        reverse_map[new_id] = list(members)
        for old in members:
            cluster_id_map[str(old)] = new_id

    cluster_renumbering = {k: k for k in reverse_map.keys()}

    return CompactionOutputs(
        components=[list(m) for m in reverse_map.values()],
        cluster_id_map=cluster_id_map,
        reverse_map=reverse_map,
        cluster_renumbering=cluster_renumbering,
        edges=edges_df,
        adjacency=adjacency,
        decision_log=decision_log,
    )


def _apply_cluster_id_map_to_obs(
    adata,
    *,
    src_labels_obs_key: str,
    dst_labels_obs_key: str,
    cluster_id_map: Dict[str, str],
) -> None:
    src = adata.obs[src_labels_obs_key].astype(str)
    dst = src.map(lambda x: cluster_id_map.get(str(x), str(x))).astype(str)
    adata.obs[dst_labels_obs_key] = dst


def create_compacted_round_from_parent_round(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    parent_round_id: str,
    new_round_id: str,
    celltypist_obs_key: str,
    notes: str = "",
    labels_obs_key_new: str | None = None,
    # compaction params
    min_cells: int = 0,
    zscore_scope: str = "within_celltypist_label",
    grouping: str = "connected_components",
    skip_unknown_celltypist_groups: bool = False,
    thr_progeny: float = 0.98,
    thr_dorothea: float = 0.98,
    thr_msigdb_default: float = 0.98,
    thr_msigdb_by_gmt: dict[str, float] | None = None,
    msigdb_required: bool = True,
) -> None:
    _ensure_cluster_rounds(adata)

    rounds = adata.uns.get("cluster_rounds", {})
    if parent_round_id not in rounds:
        raise KeyError(f"Parent round '{parent_round_id}' not found in adata.uns['cluster_rounds'].")

    parent = rounds[parent_round_id]
    parent_labels_obs_key = parent.get("labels_obs_key")
    if not parent_labels_obs_key or parent_labels_obs_key not in adata.obs:
        raise KeyError("Parent round missing labels_obs_key or it is not present in adata.obs.")

    parent_cluster_key = parent.get("cluster_key")
    if not parent_cluster_key:
        raise KeyError("Parent round missing 'cluster_key'.")

    # Ensure MSigDB activity_by_gmt exists in parent snapshot if required
    if msigdb_required:
        ensure_round_msigdb_activity_by_gmt(parent)

    outputs = compact_clusters_by_multiview_agreement(
        adata=adata,
        round_snapshot=parent,
        celltypist_obs_key=celltypist_obs_key,
        min_cells=min_cells,
        zscore_scope=zscore_scope,
        grouping=grouping,
        thr_progeny=thr_progeny,
        thr_dorothea=thr_dorothea,
        thr_msigdb_default=thr_msigdb_default,
        thr_msigdb_by_gmt=thr_msigdb_by_gmt,
        msigdb_required=msigdb_required,
        skip_unknown_celltypist_groups=skip_unknown_celltypist_groups,
    )

    # New obs key storing compacted cluster ids for this round
    if labels_obs_key_new is None:
        labels_obs_key_new = f"{parent_cluster_key}__{new_round_id}"
    if labels_obs_key_new in adata.obs:
        raise ValueError(f"labels_obs_key_new '{labels_obs_key_new}' already exists in adata.obs.")

    _apply_cluster_id_map_to_obs(
        adata,
        src_labels_obs_key=parent_labels_obs_key,
        dst_labels_obs_key=labels_obs_key_new,
        cluster_id_map=outputs.cluster_id_map,
    )
    adata.obs[labels_obs_key_new] = adata.obs[labels_obs_key_new].astype(str).astype("category")

    # Compaction audit payload (stored in round["compacting"])
    compacting_payload = {
        "parent_round_id": str(parent_round_id),
        "within_celltypist_label_only": True,
        "similarity_metric": "cosine_zscore",
        "params": {
            "min_cells": int(min_cells),
            "zscore_scope": str(zscore_scope),
            "grouping": str(grouping),
            "skip_unknown_celltypist_groups": bool(skip_unknown_celltypist_groups),
            "msigdb_required": bool(msigdb_required),
        },
        "thresholds": {
            "thr_progeny": float(thr_progeny),
            "thr_dorothea": float(thr_dorothea),
            "thr_msigdb_default": float(thr_msigdb_default),
            "thr_msigdb_by_gmt": dict(thr_msigdb_by_gmt or {}),
        },
        # heavy objects are still useful; you already store DataFrames in uns elsewhere
        "pairwise": {
            "edges": outputs.edges,
            "adjacency": outputs.adjacency,
        },
        "components": outputs.components,
        "decision_log": outputs.decision_log,
        "reverse_map": outputs.reverse_map,
    }

    # Register new round with schema-complete fields
    _register_round(
        adata,
        round_id=new_round_id,
        parent_round_id=parent_round_id,
        cluster_key=str(parent_cluster_key),
        labels_obs_key=str(labels_obs_key_new),
        kind="COMPACTED",
        best_resolution=None,
        sweep=None,
        cfg_snapshot=asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,
        notes=notes,

        # NEW: mapping fields + compacting payload
        cluster_id_map=dict(outputs.cluster_id_map),
        cluster_renumbering=dict(outputs.cluster_renumbering),
        compacting=compacting_payload,
        cache_labels=False,
    )




# -------------------------------------------------------------------------
# Public orchestrator
# -------------------------------------------------------------------------
def run_clustering(cfg: ClusterAnnotateConfig) -> ad.AnnData:
    """
    Full clustering + annotation pipeline:

    - Load integrated AnnData
    - Infer batch key
    - Precompute CellTypist predictions (optional, once)
    - Build neighbors/UMAP
    - Resolution sweep (structural metrics + optional bio metrics)
    - Subsampling stability
    - Apply final clustering
    - Final silhouette QC
    - CellTypist annotation using precomputed labels
    - Plots + save outputs
    """
    init_logging(cfg.logfile)
    LOGGER.info("Starting cluster_and_annotate")

    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)
    adata = io_utils.load_dataset(cfg.input_path)

    batch_key = io_utils.infer_batch_key(adata, cfg.batch_key)
    cfg.batch_key = batch_key

    embedding_key = _ensure_embedding(adata, cfg.embedding_key)
    LOGGER.info("Using embedding_key='%s', batch_key='%s'", embedding_key, batch_key)

    # CellTypist precompute (must happen BEFORE BISC for bioARI)
    celltypist_labels, celltypist_proba = _precompute_celltypist(adata, cfg)

    # Build neighbors + UMAP (BISC will also defensively compute if missing,
    # but we keep it here for the pipeline’s expected flow)
    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata)

    # --- BISC: resolution sweep + select best res + apply final clustering + store round ---
    run_BISC(
        adata,
        cfg,
        embedding_key=embedding_key,
        celltypist_labels=celltypist_labels,
        celltypist_proba=celltypist_proba,
        round_suffix="initial_clustering",
    )

    # ------------------------------------------------------------------
    # Figures for BISC + clustering diagnostics
    # ------------------------------------------------------------------
    def _plot_round_clustering_diagnostics(
            adata: ad.AnnData,
            cfg: ClusterAnnotateConfig,
            *,
            embedding_key: str,
            batch_key: str | None,
    ) -> None:
        """
        ROUNDS-ONLY plotting for the active round.
        """
        if not cfg.make_figures:
            return

        active_round_id = adata.uns.get("active_cluster_round", None)
        active_round_id = str(active_round_id) if active_round_id else None
        rounds = adata.uns.get("cluster_rounds", {})
        if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
            LOGGER.warning("Plots: no active cluster round found; skipping clustering diagnostics.")
            return

        rinfo = rounds[active_round_id]
        sweep = rinfo.get("sweep", {}) if isinstance(rinfo.get("sweep", {}), dict) else {}
        diag = rinfo.get("diagnostics", {}) if isinstance(rinfo.get("diagnostics", {}), dict) else {}

        figdir_cluster = Path("cluster_and_annotate") / active_round_id / "clustering"

        tested_res = diag.get("tested_resolutions", sweep.get("resolutions", [])) or []
        res_sorted = sorted(float(r) for r in tested_res) if tested_res else []
        if not res_sorted:
            LOGGER.warning("Plots: no tested resolutions found in round '%s'.", active_round_id)
            return

        best_res = rinfo.get("best_resolution", None)

        # dicts keyed by _res_key (%.3f)
        sil_dict = diag.get("silhouette_centroid", {}) or {}
        n_dict = diag.get("cluster_counts", {}) or {}
        comp_dict = diag.get("composite_scores", {}) or {}
        stab_dict = diag.get("resolution_stability", {}) or {}
        tiny_dict = diag.get("tiny_cluster_penalty", {}) or {}

        # Arrays aligned to res_sorted
        sil_arr = _extract_series(res_sorted, sil_dict)
        n_arr = _extract_series(res_sorted, n_dict)

        # Penalized: either stored as dict, or reconstruct
        pen_dict = diag.get("penalized_scores", None)
        if isinstance(pen_dict, dict) and pen_dict:
            pen_arr = _extract_series(res_sorted, pen_dict)
        else:
            alpha = float(getattr(cfg, "penalty_alpha", 0.0))
            pen_arr = np.array([float(s) - alpha * float(n) for s, n in zip(sil_arr, n_arr)], dtype=float)

        plot_utils.plot_clustering_resolution_sweep(
            resolutions=np.array(res_sorted, dtype=float),
            silhouette_scores=[float(x) for x in sil_arr],
            n_clusters=[int(round(x)) if np.isfinite(x) else 0 for x in n_arr],
            penalized_scores=[float(x) for x in pen_arr],
            figdir=figdir_cluster,
        )

        # Cluster tree (rebuild from existing sweep-added obs keys)
        labels_per_resolution: dict[str, np.ndarray] = {}
        for r in res_sorted:
            obs_key = f"{cfg.label_key}_{float(r):.2f}"
            if obs_key in adata.obs:
                labels_per_resolution[_res_key(r)] = adata.obs[obs_key].to_numpy()

        if len(labels_per_resolution) >= 2:
            plot_utils.plot_cluster_tree(
                labels_per_resolution=labels_per_resolution,
                resolutions=res_sorted,
                figdir=figdir_cluster,
                best_resolution=float(best_res) if best_res is not None else None,
            )

        # UMAPs for final clustering + batch
        plot_utils.plot_cluster_umaps(
            adata=adata,
            label_key=cfg.label_key,
            batch_key=batch_key,
            figdir=figdir_cluster,
        )

        # Subsampling stability ARI curve (round-owned)
        stability = rinfo.get("stability", {}) if isinstance(rinfo.get("stability", {}), dict) else {}
        stability_aris = stability.get("subsampling_ari", []) or []
        plot_utils.plot_clustering_stability_ari(
            stability_aris=[float(x) for x in stability_aris],
            figdir=figdir_cluster,
        )

        # Stability curves if available
        if best_res is not None and res_sorted and stab_dict and comp_dict and tiny_dict:
            plateaus = sweep.get("plateaus", None)
            if isinstance(plateaus, str):
                # already JSON-encoded
                try:
                    plateaus = json.loads(plateaus)
                except Exception:
                    plateaus = None

            plot_utils.plot_stability_curves(
                resolutions=res_sorted,
                silhouette=sil_dict,
                stability=stab_dict,
                composite=comp_dict,
                tiny_cluster_penalty=tiny_dict,
                best_resolution=float(best_res),
                plateaus=plateaus if isinstance(plateaus, list) else None,
                figdir=figdir_cluster,
            )

        # Biological metrics plot (if present in sweep arrays)
        if (
                getattr(cfg, "bio_guided_clustering", False)
                and sweep.get("bio_homogeneity") is not None
                and sweep.get("bio_fragmentation") is not None
                and sweep.get("bio_ari") is not None
                and best_res is not None
                and res_sorted
        ):
            # Convert sweep arrays -> dict keyed by _res_key
            bh = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_homogeneity"])}
            bf = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_fragmentation"])}
            ba = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_ari"])}

            plateaus = sweep.get("plateaus", None)
            if isinstance(plateaus, str):
                try:
                    plateaus = json.loads(plateaus)
                except Exception:
                    plateaus = None

            plot_utils.plot_biological_metrics(
                resolutions=res_sorted,
                bio_homogeneity=bh,
                bio_fragmentation=bf,
                bio_ari=ba,
                selection_config=sweep.get("selection_config", {}),
                best_resolution=float(best_res),
                plateaus=plateaus if isinstance(plateaus, list) else None,
                figdir=figdir_cluster,
                figure_formats=cfg.figure_formats,
            )

    if cfg.make_figures:
        _plot_round_clustering_diagnostics(adata, cfg, embedding_key=embedding_key, batch_key=batch_key)

    # ------------------------------------------------------------------
    # CellTypist annotation + pretty cluster labels (ROUND-AWARE)
    # ------------------------------------------------------------------
    active_round_id = adata.uns.get("active_cluster_round", None)
    active_round_id = str(active_round_id) if active_round_id else None

    # Determine which clustering to annotate: use active round's cluster_key if available
    cluster_key_for_annotation = cfg.label_key
    try:
        rounds = adata.uns.get("cluster_rounds", {})
        if active_round_id and isinstance(rounds, dict) and active_round_id in rounds:
            rk = rounds[active_round_id].get("cluster_key", None)
            if rk and rk in adata.obs:
                cluster_key_for_annotation = str(rk)
    except Exception:
        pass

    ann_keys = _run_celltypist_annotation(
        adata,
        cfg,
        cluster_key=cluster_key_for_annotation,
        round_id=active_round_id,
        precomputed_labels=celltypist_labels,
        precomputed_proba=celltypist_proba,
    )

    if cfg.make_figures and ann_keys is not None:
        # Put plots under: cluster_and_annotate/<round_id>/clustering/
        figdir_cluster = Path("cluster_and_annotate")
        round_part = ann_keys.get("round_id", active_round_id) or "r0"
        figdir_ct = figdir_cluster / str(round_part) / "clustering"

        # UMAPs for CellTypist outputs
        plot_utils.umap_by(
            adata,
            keys=ann_keys["celltypist_cell_key"],
            figdir=figdir_ct,
            stem="umap_celltypist_celllevel",
        )
        plot_utils.umap_by(
            adata,
            keys=ann_keys["celltypist_cluster_key"],
            figdir=figdir_ct,
            stem="umap_celltypist_clusterlevel",
        )

        # UMAP using pretty cluster labels (round-scoped)
        plot_utils.umap_by(
            adata,
            keys=ann_keys["pretty_cluster_key"],
            figdir=figdir_ct,
            stem="umap_pretty_cluster_label",
        )

        # Cluster-level statistics using pretty cluster labels (round-scoped)
        id_key = ann_keys["pretty_cluster_key"]
        plot_utils.plot_cluster_sizes(adata, id_key, figdir_ct)
        plot_utils.plot_cluster_qc_summary(adata, id_key, figdir_ct)
        plot_utils.plot_cluster_silhouette_by_cluster(adata, id_key, embedding_key, figdir_ct)
        if batch_key is not None:
            plot_utils.plot_cluster_batch_composition(adata, id_key, batch_key, figdir_ct)

    # Store pointers in cluster_and_annotate uns (keep old keys as aliases)
    ca_uns = adata.uns.get("cluster_and_annotate", {})
    if ann_keys is not None:
        ca_uns["celltypist_label_key"] = ann_keys["celltypist_cell_key"]
        ca_uns["celltypist_cluster_label_key"] = ann_keys["celltypist_cluster_key"]
        ca_uns["cluster_label_key"] = ann_keys["pretty_cluster_key"]
        adata.uns["cluster_and_annotate"] = ca_uns

    # ------------------------------------------------------------------
    # Optional CSV export of cluster annotations (ROUND-AWARE)
    # ------------------------------------------------------------------
    def _export_round_annotations_csv(adata: ad.AnnData, cfg: ClusterAnnotateConfig) -> None:
        if cfg.annotation_csv is None:
            return

        active_round_id = adata.uns.get("active_cluster_round", None)
        active_round_id = str(active_round_id) if active_round_id else None
        rounds = adata.uns.get("cluster_rounds", {})

        if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
            LOGGER.warning("annotation_csv requested but no active cluster round found; skipping export.")
            return

        # Prefer round-linked keys if present
        rinfo = rounds[active_round_id]
        ann = rinfo.get("annotation", {}) if isinstance(rinfo.get("annotation", {}), dict) else {}
        cluster_key = rinfo.get("cluster_key", None)

        cols: list[str] = []
        if cluster_key and cluster_key in adata.obs:
            cols.append(str(cluster_key))

        ct_cluster_key = ann.get("celltypist_cluster_key", f"{cfg.celltypist_cluster_label_key}__{active_round_id}")
        pretty_key = ann.get("pretty_cluster_key", f"{CLUSTER_LABEL_KEY}__{active_round_id}")

        if ct_cluster_key in adata.obs:
            cols.append(ct_cluster_key)
        if pretty_key in adata.obs:
            cols.append(pretty_key)

        if not cols:
            LOGGER.warning("annotation_csv requested, but no annotation columns found for round '%s'.", active_round_id)
            return

        LOGGER.info("Exporting cluster annotations for round '%s' with columns: %s", active_round_id, cols)
        io_utils.export_cluster_annotations(adata, columns=cols, out_path=cfg.annotation_csv)

    _export_round_annotations_csv(adata, cfg)

    # ------------------------------------------------------------------
    # Decoupler nets (cluster-level) — ROUND-AWARE, gated by cfg.run_decoupler
    # ------------------------------------------------------------------
    if getattr(cfg, "run_decoupler", False):
        run_decoupler_for_round(adata, cfg, round_id=None)  # uses active round

    else:
        LOGGER.info("Decoupler: disabled (run_decoupler=False).")

    # ------------------------------------------------------------------
    # Decoupler net plots (heatmap + per-cluster bars + dotplot) — ROUND-AWARE
    # ------------------------------------------------------------------
    if cfg.make_figures and getattr(cfg, "run_decoupler", False):
        active_round_id = adata.uns.get("active_cluster_round", None)
        active_round_id = str(active_round_id) if active_round_id else None

        rounds = adata.uns.get("cluster_rounds", {})
        if not active_round_id or active_round_id not in rounds:
            LOGGER.warning("Decoupler plots: no active cluster round found; skipping.")
        else:
            figdir_round = Path("cluster_and_annotate") / active_round_id

            if "msigdb" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="msigdb",
                    net_name="MSigDB",
                    figdir=figdir_round,
                    heatmap_top_k=30,
                    bar_top_n=10,
                    dotplot_top_k=30,
                )

            if "progeny" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="progeny",
                    net_name="PROGENy",
                    figdir=figdir_round,
                    heatmap_top_k=14,
                    bar_top_n=8,
                    dotplot_top_k=14,
                )

            if "dorothea" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="dorothea",
                    net_name="DoRothEA",
                    figdir=figdir_round,
                    heatmap_top_k=40,
                    bar_top_n=10,
                    dotplot_top_k=35,
                )

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------
    if getattr(cfg, "enable_compacting", False):
        # Compaction requires decoupler outputs (progeny/dorothea and usually msigdb)
        if not getattr(cfg, "run_decoupler", False):
            LOGGER.warning(
                "Compaction requested (enable_compacting=True) but run_decoupler=False. "
                "Compaction requires decoupler activities; skipping."
            )
        else:
            parent_round_id = adata.uns.get("active_cluster_round", None)
            parent_round_id = str(parent_round_id) if parent_round_id else None
            if not parent_round_id:
                LOGGER.warning("Compaction requested but no active_cluster_round found; skipping.")
            else:
                rounds = adata.uns.get("cluster_rounds", {})
                if not isinstance(rounds, dict) or parent_round_id not in rounds:
                    LOGGER.warning(
                        "Compaction requested but parent round '%s' not found; skipping.",
                        parent_round_id,
                    )
                else:
                    # Ensure parent has decoupler results (if user disabled plots/earlier steps, be defensive)
                    try:
                        run_decoupler_for_round(adata, cfg, round_id=parent_round_id)
                    except Exception as e:
                        LOGGER.warning(
                            "Compaction: failed to ensure decoupler outputs for parent round '%s': %s",
                            parent_round_id,
                            e,
                        )

                    parent = rounds[parent_round_id]
                    cluster_key = str(parent.get("cluster_key", getattr(cfg, "label_key", "leiden")))

                    if cluster_key not in adata.obs:
                        LOGGER.warning(
                            "Compaction: parent cluster_key '%s' not found in adata.obs; skipping.",
                            cluster_key,
                        )
                    else:
                        celltypist_obs_key = str(getattr(cfg, "celltypist_label_key", "") or "")
                        if not celltypist_obs_key or celltypist_obs_key not in adata.obs:
                            LOGGER.warning(
                                "Compaction: celltypist_obs_key '%s' missing in adata.obs; skipping.",
                                celltypist_obs_key,
                            )
                        else:
                            new_round_id = _make_round_id(_next_round_index(adata), "compacted")

                            LOGGER.info(
                                "Compaction: creating new round '%s' from parent '%s' (cluster_key=%s, celltypist=%s).",
                                new_round_id,
                                parent_round_id,
                                cluster_key,
                                celltypist_obs_key,
                            )

                            try:
                                create_compacted_round_from_parent_round(
                                    adata,
                                    cfg,
                                    parent_round_id=parent_round_id,
                                    new_round_id=new_round_id,
                                    celltypist_obs_key=celltypist_obs_key,
                                    notes=f"Compacted from {parent_round_id}",
                                    min_cells=int(getattr(cfg, "compact_min_cells", 0) or 0),
                                    zscore_scope=str(
                                        getattr(cfg, "compact_zscore_scope", "within_celltypist_label")
                                        or "within_celltypist_label"
                                    ),
                                    grouping=str(
                                        getattr(cfg, "compact_grouping", "connected_components")
                                        or "connected_components"
                                    ),
                                    skip_unknown_celltypist_groups=bool(
                                        getattr(cfg, "compact_skip_unknown_celltypist_groups", False)
                                    ),
                                    thr_progeny=float(getattr(cfg, "thr_progeny", 0.98) or 0.98),
                                    thr_dorothea=float(getattr(cfg, "thr_dorothea", 0.98) or 0.98),
                                    thr_msigdb_default=float(getattr(cfg, "thr_msigdb_default", 0.98) or 0.98),
                                    thr_msigdb_by_gmt=getattr(cfg, "thr_msigdb_by_gmt", None),
                                    msigdb_required=bool(getattr(cfg, "msigdb_required", True)),
                                )
                            except Exception as e:
                                LOGGER.warning("Compaction failed while creating compacted round: %s", e)
                            else:
                                # Activate the compacted round so canonical keys point at it
                                set_active_round(adata, new_round_id, publish_decoupler=False)

                                # Re-run CellTypist cluster-level labels + pretty labels for the new round
                                try:
                                    rk = adata.uns["cluster_rounds"][new_round_id]["cluster_key"]
                                    _run_celltypist_annotation(
                                        adata,
                                        cfg,
                                        cluster_key=str(rk),
                                        round_id=new_round_id,
                                        precomputed_labels=celltypist_labels,
                                        precomputed_proba=celltypist_proba,
                                    )
                                except Exception as e:
                                    LOGGER.warning(
                                        "Compaction: failed to rebuild CellTypist cluster/pretty labels for '%s': %s",
                                        new_round_id,
                                        e,
                                    )

                                # Run decoupler for the compacted round (now active)
                                try:
                                    run_decoupler_for_round(adata, cfg, round_id=new_round_id)
                                except Exception as e:
                                    LOGGER.warning(
                                        "Compaction: decoupler failed for compacted round '%s': %s",
                                        new_round_id,
                                        e,
                                    )

                                # Plot decoupler under the compacted round folder
                                if cfg.make_figures and getattr(cfg, "run_decoupler", False):
                                    figdir_round = Path("cluster_and_annotate") / new_round_id

                                    if "msigdb" in adata.uns:
                                        plot_utils.plot_decoupler_all_styles(
                                            adata,
                                            net_key="msigdb",
                                            net_name="MSigDB",
                                            figdir=figdir_round,
                                            heatmap_top_k=30,
                                            bar_top_n=10,
                                            dotplot_top_k=30,
                                        )

                                    if "progeny" in adata.uns:
                                        plot_utils.plot_decoupler_all_styles(
                                            adata,
                                            net_key="progeny",
                                            net_name="PROGENy",
                                            figdir=figdir_round,
                                            heatmap_top_k=14,
                                            bar_top_n=8,
                                            dotplot_top_k=14,
                                        )

                                    if "dorothea" in adata.uns:
                                        plot_utils.plot_decoupler_all_styles(
                                            adata,
                                            net_key="dorothea",
                                            net_name="DoRothEA",
                                            figdir=figdir_round,
                                            heatmap_top_k=40,
                                            bar_top_n=10,
                                            dotplot_top_k=35,
                                        )
    else:
        LOGGER.info("Compaction: disabled (enable_compacting=False).")

    # ------------------------------------------------------------------
    # Make 'plateaus' HDF5/Zarr-safe — ROUND-AWARE (encode IN PLACE)
    # ------------------------------------------------------------------
    def _json_encode_round_plateaus_in_place(adata: ad.AnnData) -> None:
        """
        Make plateaus HDF5/Zarr-safe by JSON encoding:
          adata.uns["cluster_rounds"][round_id]["sweep"]["plateaus"]
        """
        active_round_id = adata.uns.get("active_cluster_round", None)
        active_round_id = str(active_round_id) if active_round_id else None
        rounds = adata.uns.get("cluster_rounds", {})

        if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
            return

        rinfo = rounds[active_round_id]
        sweep = rinfo.get("sweep", None)
        if not isinstance(sweep, dict):
            return

        plateaus = sweep.get("plateaus", None)
        if isinstance(plateaus, list):
            sweep["plateaus"] = json.dumps(plateaus)
            rinfo["sweep"] = sweep
            rounds[active_round_id] = rinfo
            adata.uns["cluster_rounds"] = rounds

    _json_encode_round_plateaus_in_place(adata)

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    out_zarr = cfg.resolved_output_dir / (cfg.output_name + ".zarr")
    LOGGER.info("Saving clustered/annotated dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if getattr(cfg, "save_h5ad", False):
        out_h5ad = cfg.resolved_output_dir / (cfg.output_name + ".h5ad")
        LOGGER.warning(
            "Writing additional H5AD output (loads full matrix into RAM): %s",
            out_h5ad,
        )
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
        LOGGER.info("Saved clustered/annotated H5AD → %s", out_h5ad)

    LOGGER.info("Finished cluster_and_annotate")

    return adata
