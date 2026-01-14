from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional
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


LOGGER = logging.getLogger(__name__)

# Single pretty cluster label column
CLUSTER_LABEL_KEY = "cluster_label"


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
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
    precomputed_labels: Optional[np.ndarray] = None,
    precomputed_proba: Optional[pd.DataFrame] = None,
) -> str | None:
    """
    Attach CellTypist annotations to the main AnnData object.

    If `precomputed_labels`/`precomputed_proba` are provided (from _precompute_celltypist),
    they are used directly; otherwise a fallback CellTypist run is performed on `adata`.

    Steps:
    - set per-cell labels in adata.obs[cfg.celltypist_label_key]
    - optionally store probabilities in adata.obsm["celltypist_proba"]
    - perform cluster-level majority voting to derive cluster labels
    - create pretty cluster labels in adata.obs[CLUSTER_LABEL_KEY]
    """
    if cfg.celltypist_model is None:
        LOGGER.info("No CellTypist model provided; skipping annotation.")
        return None

    # Path A: use precomputed predictions
    if precomputed_labels is not None:
        if precomputed_labels.shape[0] != adata.n_obs:
            raise ValueError(
                "precomputed_labels length does not match number of cells in adata."
            )
        LOGGER.info("Using precomputed CellTypist labels for final annotation.")
        adata.obs[cfg.celltypist_label_key] = precomputed_labels

        if precomputed_proba is not None:
            # Store probabilities as obsm + column names in uns for interpretability
            adata.obsm["celltypist_proba"] = precomputed_proba.loc[
                adata.obs_names
            ].to_numpy()
            adata.uns["celltypist_proba_columns"] = list(precomputed_proba.columns)

    # Path B: fallback -> run CellTypist on main adata
    else:
        LOGGER.info("Running CellTypist on main AnnData for final annotation.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        LOGGER.info("Resolving CellTypist model: %s", cfg.celltypist_model)
        model_path = get_celltypist_model(cfg.celltypist_model)

        from celltypist.models import Model
        import celltypist

        LOGGER.info("Loading CellTypist model from %s", model_path)
        model = Model.load(str(model_path))

        predictions = celltypist.annotate(
            adata,
            model=model,
            majority_voting=cfg.celltypist_majority_voting,
        )

        if (
            isinstance(predictions.predicted_labels, dict)
            and "majority_voting" in predictions.predicted_labels
        ):
            cell_level_labels = predictions.predicted_labels["majority_voting"]
        else:
            cell_level_labels = predictions.predicted_labels

        adata.obs[cfg.celltypist_label_key] = cell_level_labels

        if hasattr(predictions, "probability_matrix"):
            pm = predictions.probability_matrix
            adata.obsm["celltypist_proba"] = pm.loc[adata.obs_names].to_numpy()
            adata.uns["celltypist_proba_columns"] = list(pm.columns)

    if cfg.label_key not in adata.obs:
        raise KeyError(
            f"label_key '{cfg.label_key}' not found in adata.obs; "
            "cannot compute cluster-level CellTypist labels."
        )

    # Cluster-level majority CellTypist label
    cluster_majority = (
        adata.obs[[cfg.label_key, cfg.celltypist_label_key]]
        .groupby(cfg.label_key)[cfg.celltypist_label_key]
        .agg(lambda x: x.value_counts().idxmax())
    )

    adata.obs[cfg.celltypist_cluster_label_key] = (
        adata.obs[cfg.label_key].map(cluster_majority)
    )

    # ------------------------------------------------------------------
    # Pretty final cluster labels:
    #   "C{leiden_id:02d}: {majority_celltypist_label}"
    # This keeps the original Leiden cluster numbers as reference.
    # ------------------------------------------------------------------
    leiden_ids = adata.obs[cfg.label_key].astype(str)
    maj_labels = adata.obs[cfg.celltypist_cluster_label_key].astype(str).fillna("Unknown")

    pretty_labels = "C" + leiden_ids.str.zfill(2) + ": " + maj_labels
    adata.obs[CLUSTER_LABEL_KEY] = pretty_labels.astype("category")

    # Stable color palette for cluster_label
    try:
        from scanpy.plotting.palettes import default_102

        cats = adata.obs[CLUSTER_LABEL_KEY].cat.categories
        adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = default_102[: len(cats)]
    except Exception as e:
        LOGGER.warning("Could not set cluster_label color palette: %s", e)

    LOGGER.info(
        "Added CellTypist labels to adata.obs['%s'] (cell level) and "
        "cluster-level majority labels to adata.obs['%s'].",
        cfg.celltypist_label_key,
        cfg.celltypist_cluster_label_key,
    )
    LOGGER.info(
        "Added pretty cluster labels to adata.obs['%s'] using Leiden IDs + majority CellTypist label.",
        CLUSTER_LABEL_KEY,
    )

    return cfg.celltypist_label_key


def _final_real_silhouette_qc(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    figdir: Path,
) -> Optional[float]:
    """
    Compute true silhouette for the final clustering (QC only) and optionally plot histogram.
    """
    if cfg.label_key not in adata.obs:
        LOGGER.warning(
            "final_real_silhouette_qc: label_key '%s' not in adata.obs; skipping.",
            cfg.label_key,
        )
        return None

    labels = adata.obs[cfg.label_key].to_numpy()
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

    adata.uns.setdefault("cluster_and_annotate", {})
    ca_uns = adata.uns["cluster_and_annotate"]
    ca_uns["real_silhouette_final"] = sil_mean
    ca_uns["real_silhouette_summary"] = {
        "mean": sil_mean,
        "median": float(np.median(sil_values)),
        "p10": float(np.percentile(sil_values, 10)),
        "p90": float(np.percentile(sil_values, 90)),
    }

    LOGGER.info("Final true silhouette (mean) = %.3f", sil_mean)

    if cfg.make_figures:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sil_values, bins=40, color="steelblue", alpha=0.85)
        ax.axvline(sil_mean, color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Silhouette value")
        ax.set_ylabel("Number of cells")
        ax.set_title(f"Final clustering: true silhouette (mean = {sil_mean:.3f})")

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


def _run_msigdb(adata: "ad.AnnData", cfg: "ClusterAnnotateConfig") -> None:
    """
    Compute MSigDB pathway activity on CLUSTER-level pseudobulk expression using decoupler.

    Expects pseudobulk expression already stored in:
      adata.uns["pseudobulk"]["expr"]  -> DataFrame (genes x clusters)

    Resolves GMTs using your resolve_msigdb_gene_sets(...).

    Stores:
      adata.uns["msigdb"]["activity"] -> DataFrame (clusters x pathways)
      adata.uns["msigdb"]["config"]   -> dict snapshot
    """
    import pandas as pd

    # -----------------------------
    # Pull pseudobulk from .uns
    # -----------------------------
    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key="pseudobulk")  # genes x clusters DataFrame
    except Exception as e:
        LOGGER.warning("...: missing/invalid pseudobulk store; skipping. (%s)", e)
        return
    if expr.empty:
        LOGGER.warning("...: pseudobulk expr is empty; skipping.")
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
    method = getattr(cfg, "msigdb_method", None) or getattr(cfg, "decoupler_method", None) or "consensus"
    min_n = int(getattr(cfg, "msigdb_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    # -----------------------------
    # Build decoupler network from GMTs
    # -----------------------------
    # Network columns: source (pathway), target (gene), weight (1.0)
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

    # Defensive: ensure required cols
    for col in ("source", "target"):
        if col not in net.columns:
            raise KeyError(f"MSigDB net missing required column '{col}'")
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

        activity = est.T      # clusters x pathways
        activity.index.name = "cluster"
        activity.columns.name = "pathway"

    except Exception as e:
        LOGGER.warning("MSigDB: decoupler run failed (method=%s): %s", str(method), e)
        return

    # -----------------------------
    # Store
    # -----------------------------
    adata.uns.setdefault("msigdb", {})
    adata.uns["msigdb"]["activity"] = activity
    adata.uns["msigdb"]["config"] = {
        "method": str(method),
        "min_n_targets": int(min_n),
        "msigdb_release": msigdb_release,
        "gene_sets": [str(g) for g in gmt_files],
        "used_keywords": used_keywords,
        "input": "pseudobulk_expr(genes_x_clusters)",
        "resource": "msigdb_gmt",
    }

    LOGGER.info(
        "MSigDB stored: activity shape=%s using method=%s (min_n=%d; GMTs=%d).",
        tuple(activity.shape),
        str(method),
        int(min_n),
        len(gmt_files),
    )



def _run_progeny(adata: "ad.AnnData", cfg: "ClusterAnnotateConfig") -> None:
    """
    Compute PROGENy pathway activity on CLUSTER-level pseudobulk expression using decoupler.

    Expects pseudobulk expression already stored in:
      adata.uns["pseudobulk"]["expr"]  -> DataFrame (genes x clusters)

    Stores:
      adata.uns["progeny"]["activity"] -> DataFrame (clusters x pathways)
      adata.uns["progeny"]["config"]   -> dict snapshot
    """
    import pandas as pd
    import decoupler as dc

    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key="pseudobulk")  # genes x clusters DataFrame
    except Exception as e:
        LOGGER.warning("...: missing/invalid pseudobulk store; skipping. (%s)", e)
        return
    if expr.empty:
        LOGGER.warning("...: pseudobulk expr is empty; skipping.")
        return

    # -----------------------------
    # Method config (NEW)
    # -----------------------------
    method = getattr(cfg, "progeny_method", None) or getattr(cfg, "decoupler_method", None) or "consensus"
    min_n = int(getattr(cfg, "progeny_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    # PROGENy-specific knobs
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
            weight=wcol or "weight",
            min_n=min_n,
            verbose=False,
            consensus_methods=getattr(cfg, "decoupler_consensus_methods", None),
        )

        activity = est.T
        activity.index.name = "cluster"
        activity.columns.name = "pathway"

    except Exception as e:
        LOGGER.warning("PROGENy: decoupler run failed (method=%s): %s", method, e)
        return

    # -----------------------------
    # Store
    # -----------------------------
    adata.uns.setdefault("progeny", {})
    adata.uns["progeny"]["activity"] = activity
    adata.uns["progeny"]["config"] = {
        "method": str(method),
        "min_n_targets": int(min_n),
        "top_n": int(top_n),
        "input": "pseudobulk_expr(genes_x_clusters)",
        "resource": "progeny",
        "organism": str(organism),
    }

    LOGGER.info(
        "PROGENy stored: activity shape=%s using method=%s (top_n=%d, min_n=%d).",
        tuple(activity.shape),
        str(method),
        int(top_n),
        int(min_n),
    )



def _run_dorothea(adata: "ad.AnnData", cfg: "ClusterAnnotateConfig") -> None:
    """
    Compute DoRothEA TF activity on CLUSTER-level pseudobulk expression using decoupler.

    Expects pseudobulk expression already stored in:
      adata.uns["pseudobulk"]["expr"]  -> DataFrame (genes x clusters)

    Stores:
      adata.uns["dorothea"]["activity"] -> DataFrame (clusters x TFs)
      adata.uns["dorothea"]["config"]   -> dict snapshot
    """
    import decoupler as dc

    # -----------------------------
    # Pull pseudobulk from .uns
    # -----------------------------
    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key="pseudobulk")  # genes x clusters DataFrame
    except Exception as e:
        LOGGER.warning("...: missing/invalid pseudobulk store; skipping. (%s)", e)
        return
    if expr.empty:
        LOGGER.warning("...: pseudobulk expr is empty; skipping.")
        return

    # -----------------------------
    # Method config (NEW)
    # -----------------------------
    method = getattr(cfg, "dorothea_method", None) or getattr(cfg, "decoupler_method", None) or "consensus"
    min_n = int(getattr(cfg, "dorothea_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

    # Confidence filtering (common for DoRothEA)
    # DoRothEA confidence levels are typically {"A","B","C","D","E"}.
    # Default to A/B/C unless user overrides.
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
        # mat must be genes x samples
        est = _dc_run_method(
            method=method,
            mat=expr,
            net=net,
            source="source",
            target="target",
            weight="weight" if "weight" in net.columns else "mor",
            min_n=min_n,
            verbose=False,
            consensus_methods=getattr(cfg, "decoupler_consensus_methods", None),

        )

        # est is typically (sources x samples). We want samples x sources for plotting consistency.
        # So: clusters x TFs
        activity = est.T
        activity.index.name = "cluster"
        activity.columns.name = "TF"

    except Exception as e:
        LOGGER.warning("DoRothEA: decoupler run failed (method=%s): %s", method, e)
        return

    # -----------------------------
    # Store
    # -----------------------------
    adata.uns.setdefault("dorothea", {})
    adata.uns["dorothea"]["activity"] = activity
    adata.uns["dorothea"]["config"] = {
        "method": str(method),
        "organism": str(organism),
        "min_n_targets": int(min_n),
        "confidence": list(conf),
        "input": "pseudobulk_expr(genes_x_clusters)",
        "resource": "dorothea",
    }

    LOGGER.info(
        "DoRothEA stored: activity shape=%s using method=%s (conf=%s, min_n=%d).",
        tuple(activity.shape),
        str(method),
        ",".join(conf),
        int(min_n),
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

    # CellTypist precompute (may be None, None)
    celltypist_labels, celltypist_proba = _precompute_celltypist(adata, cfg)

    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata)

    if cfg.make_figures:
        plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)
    figdir_cluster = Path("cluster_and_annotate")

    # Build "good CellTypist" mask once (for bio metrics only)
    bio_mask, bio_mask_stats = _maybe_build_bio_mask(cfg, celltypist_proba, adata.n_obs)

    adata.uns.setdefault("cluster_and_annotate", {})
    adata.uns["cluster_and_annotate"].setdefault("bio_mask", {})
    adata.uns["cluster_and_annotate"]["bio_mask"] = bio_mask_stats

    # --- summary of mask before sweep ---
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
                "CellTypist bio mask: kept %d/%d cells (%.1f%%) "
                "[mode=%s, entropy_cut=%.3f (abs=%.3f, q=%.2f→%.3f), margin_min=%.3f]",
                kept,
                n_cells,
                100.0 * frac,
                mode,
                float(bio_mask_stats.get("entropy_cut_used", float("nan"))),
                float(bio_mask_stats.get("entropy_abs_limit", float("nan"))),
                float(bio_mask_stats.get("entropy_quantile", float("nan"))),
                float(bio_mask_stats.get("entropy_q_value", float("nan"))),
                float(bio_mask_stats.get("margin_min", float("nan"))),
            )

    # Perform resolution sweep
    best_res, sweep, clusterings = _resolution_sweep(
        adata,
        cfg,
        embedding_key,
        celltypist_labels=celltypist_labels,
        bio_mask=bio_mask,
    )

    res_list = [float(r) for r in sweep["resolutions"]]

    adata.uns.setdefault("cluster_and_annotate", {})
    ca_uns = adata.uns["cluster_and_annotate"]

    ca_uns.update(
        {
            "embedding_key": embedding_key,
            "batch_key": batch_key,
            "label_key": cfg.label_key,
            "cluster_label_key": CLUSTER_LABEL_KEY,
            "best_resolution": float(best_res),
            "resolutions": [float(r) for r in sweep["resolutions"]],
            "silhouette_scores": [float(x) for x in sweep["silhouette_scores"]],
            "n_clusters": [int(x) for x in sweep["n_clusters"]],
            "penalized_scores": [float(x) for x in sweep["penalized_scores"]],
            "stability_ari": [],
            "celltypist_model": cfg.celltypist_model,
            "celltypist_label_key": None,
            "celltypist_cluster_label_key": None,
        }
    )

    ca_uns["clustering"] = {
        "tested_resolutions": res_list,
        "best_resolution": float(best_res),
        "silhouette_centroid": {
            _res_key(r): float(s)
            for r, s in zip(res_list, sweep["silhouette_scores"])
        },
        "cluster_counts": {
            _res_key(r): int(n)
            for r, n in zip(res_list, sweep["n_clusters"])
        },
        "cluster_sizes": {
            _res_key(r): [int(x) for x in sweep["cluster_sizes"][float(r)]]
            for r in res_list
        },
        "composite_scores": {
            _res_key(r): float(s)
            for r, s in zip(res_list, sweep["composite_scores"])
        },
        "resolution_stability": {
            _res_key(r): float(s)
            for r, s in zip(res_list, sweep["stability_scores"])
        },
        "tiny_cluster_penalty": {
            _res_key(r): float(s)
            for r, s in zip(res_list, sweep["tiny_cluster_penalty"])
        },
        "plateaus": sweep["plateaus"],
        "selection_config": sweep["selection_config"],
        "bio_homogeneity": None,
        "bio_fragmentation": None,
        "bio_ari": None,
    }

    if sweep.get("bio_homogeneity") is not None:
        ca_uns["clustering"]["bio_homogeneity"] = {
            _res_key(r): float(v)
            for r, v in zip(res_list, sweep["bio_homogeneity"])
        }
        ca_uns["clustering"]["bio_fragmentation"] = {
            _res_key(r): float(v)
            for r, v in zip(res_list, sweep["bio_fragmentation"])
        }
        ca_uns["clustering"]["bio_ari"] = {
            _res_key(r): float(v)
            for r, v in zip(res_list, sweep["bio_ari"])
        }

    if cfg.make_figures:
        plot_utils.plot_clustering_resolution_sweep(
            resolutions=sweep["resolutions"],
            silhouette_scores=sweep["silhouette_scores"],
            n_clusters=sweep["n_clusters"],
            penalized_scores=sweep["penalized_scores"],
            figdir=figdir_cluster,
        )
        plot_utils.plot_cluster_tree(
            labels_per_resolution=clusterings,
            resolutions=sweep["resolutions"],
            figdir=figdir_cluster,
            best_resolution=best_res,
        )

    stability_aris = _subsampling_stability(adata, cfg, embedding_key, best_res)
    ca_uns["stability_ari"] = [float(x) for x in stability_aris]

    _apply_final_clustering(adata, cfg, best_res)
    _final_real_silhouette_qc(adata, cfg, embedding_key, figdir_cluster)

    if cfg.make_figures:
        # UMAP with raw Leiden clusters (reference)
        plot_utils.plot_cluster_umaps(
            adata=adata,
            label_key=cfg.label_key,
            batch_key=batch_key,
            figdir=figdir_cluster,
        )
        # Stability curves, composite metrics, etc.
        clust = ca_uns["clustering"]
        plot_utils.plot_clustering_stability_ari(
            stability_aris=stability_aris,
            figdir=figdir_cluster,
        )
        plot_utils.plot_stability_curves(
            resolutions=clust["tested_resolutions"],
            silhouette=clust["silhouette_centroid"],
            stability=clust["resolution_stability"],
            composite=clust["composite_scores"],
            tiny_cluster_penalty=clust["tiny_cluster_penalty"],
            best_resolution=clust["best_resolution"],
            plateaus=clust["plateaus"],
            figdir=figdir_cluster,
        )

        # Biological metrics plot (only when bio-guided clustering is enabled & metrics present)
        if (
            getattr(cfg, "bio_guided_clustering", False)
            and clust.get("bio_homogeneity") is not None
            and clust.get("bio_fragmentation") is not None
            and clust.get("bio_ari") is not None
        ):
            plot_utils.plot_biological_metrics(
                resolutions=clust["tested_resolutions"],
                bio_homogeneity=clust["bio_homogeneity"],
                bio_fragmentation=clust["bio_fragmentation"],
                bio_ari=clust["bio_ari"],
                selection_config=clust["selection_config"],
                best_resolution=clust["best_resolution"],
                plateaus=clust["plateaus"],
                figdir=figdir_cluster,
                figure_formats=cfg.figure_formats,
            )

        # === Build structural composite series ===
        res_list = clust["tested_resolutions"]

        sil_dict = clust["silhouette_centroid"]
        stab_dict = clust["resolution_stability"]
        tiny_dict = clust["tiny_cluster_penalty"]
        cfg_sel = clust["selection_config"]

        # Min–max normalize each metric using the existing helper
        sil_norm_array = _normalize_array(_extract_series(res_list, sil_dict))
        stab_norm_array = _normalize_array(_extract_series(res_list, stab_dict))
        tiny_norm_array = _normalize_array(_extract_series(res_list, tiny_dict))

        w_sil = float(cfg_sel.get("w_sil", 0.0))
        w_stab = float(cfg_sel.get("w_stab", 0.0))
        w_tiny = float(cfg_sel.get("w_tiny", 0.0))

        structural_comp = {
            _res_key(r): (
                w_sil * sil_norm_array[i]
                + w_stab * stab_norm_array[i]
                + w_tiny * tiny_norm_array[i]
            )
            for i, r in enumerate(res_list)
        }

        # === Build biological composite if available ===
        bio_comp = None
        if (
            clust.get("bio_homogeneity") is not None
            and clust.get("bio_fragmentation") is not None
            and clust.get("bio_ari") is not None
        ):
            hom = clust["bio_homogeneity"]
            frag = clust["bio_fragmentation"]
            bioari = clust["bio_ari"]

            hom_norm = _normalize_array(_extract_series(res_list, hom))
            frag_norm = _normalize_array(_extract_series(res_list, frag))
            ari_norm = _normalize_array(_extract_series(res_list, bioari))

            w_hom = float(cfg_sel.get("w_hom", 0.0))
            w_frag = float(cfg_sel.get("w_frag", 0.0))
            w_bioari = float(cfg_sel.get("w_bioari", 0.0))

            bio_comp = {
                _res_key(r): (
                    w_hom * hom_norm[i]
                    + w_frag * (1 - frag_norm[i])
                    + w_bioari * ari_norm[i]
                )
                for i, r in enumerate(res_list)
            }

        # Composite-only diagnostic plot
        plot_utils.plot_composite_only(
            resolutions=clust["tested_resolutions"],
            structural_comp=structural_comp,
            biological_comp=bio_comp,
            total_comp=clust["composite_scores"],
            best_resolution=clust["best_resolution"],
            plateaus=clust["plateaus"],
            figdir=figdir_cluster,
        )

        # Plateau diagnostic plot
        plot_utils.plot_plateau_highlights(
            resolutions=clust["tested_resolutions"],
            silhouette=clust["silhouette_centroid"],
            stability=clust["resolution_stability"],
            composite=clust["composite_scores"],
            best_resolution=clust["best_resolution"],
            plateaus=clust["plateaus"],
            figdir=figdir_cluster,
            figure_formats=cfg.figure_formats,
        )

    # ------------------------------------------------------------------
    # CellTypist annotation + pretty cluster_label
    # ------------------------------------------------------------------
    annotation_col = _run_celltypist_annotation(
        adata,
        cfg,
        precomputed_labels=celltypist_labels,
        precomputed_proba=celltypist_proba,
    )

    if cfg.make_figures and annotation_col is not None:
        # UMAPs for CellTypist outputs
        plot_utils.umap_by(
            adata,
            keys=cfg.celltypist_label_key,
            figdir=figdir_cluster,
            stem="umap_celltypist_celllevel",
        )
        plot_utils.umap_by(
            adata,
            keys=cfg.celltypist_cluster_label_key,
            figdir=figdir_cluster,
            stem="umap_celltypist_clusterlevel",
        )
        # UMAP using pretty cluster labels
        plot_utils.umap_by(
            adata,
            keys=CLUSTER_LABEL_KEY,
            figdir=figdir_cluster,
            stem="umap_cluster_label",
        )

        # Cluster-level statistics using pretty cluster labels
        id_key = CLUSTER_LABEL_KEY
        plot_utils.plot_cluster_sizes(adata, id_key, figdir_cluster)
        plot_utils.plot_cluster_qc_summary(adata, id_key, figdir_cluster)
        plot_utils.plot_cluster_silhouette_by_cluster(
            adata, id_key, embedding_key, figdir_cluster
        )
        if batch_key is not None:
            plot_utils.plot_cluster_batch_composition(
                adata, id_key, batch_key, figdir_cluster
            )

    if annotation_col is not None:
        ca_uns["celltypist_label_key"] = cfg.celltypist_label_key
        ca_uns["celltypist_cluster_label_key"] = cfg.celltypist_cluster_label_key
        ca_uns["cluster_label_key"] = CLUSTER_LABEL_KEY

    # Optional CSV with cluster annotations
    if cfg.annotation_csv is not None and annotation_col is not None:
        io_utils.export_cluster_annotations(
            adata,
            columns=[cfg.label_key, annotation_col],
            out_path=cfg.annotation_csv,
        )

    # ------------------------------------------------------------------
    # Decoupler nets (cluster-level) — gated by cfg.run_decoupler
    # ------------------------------------------------------------------
    if getattr(cfg, "run_decoupler", False):
        # Store cluster pseudobulk once for all downstream nets
        cluster_key = None
        if "cluster_label" in adata.obs:
            cluster_key = "cluster_label"
        elif getattr(cfg, "final_auto_idents_key", None) and cfg.final_auto_idents_key in adata.obs:
            cluster_key = cfg.final_auto_idents_key
        elif getattr(cfg, "label_key", None) and cfg.label_key in adata.obs:
            cluster_key = cfg.label_key

        if cluster_key is None:
            LOGGER.warning("Decoupler: no cluster key found; skipping pseudobulk + nets.")
        else:
            _store_cluster_pseudobulk(
                adata,
                cluster_key=cluster_key,
                agg=getattr(cfg, "decoupler_pseudobulk_agg", "mean"),
                use_raw_like=bool(getattr(cfg, "decoupler_use_raw", True)),
                prefer_layers=("counts_raw", "counts_cb"),
                store_key="pseudobulk",
            )

            # Run selected nets (each function also reads cfg for method overrides)
            _run_msigdb(adata, cfg)

            if getattr(cfg, "run_dorothea", True):
                _run_dorothea(adata, cfg)

            if getattr(cfg, "run_progeny", True):
                _run_progeny(adata, cfg)
    else:
        LOGGER.info("Decoupler: disabled (run_decoupler=False).")

    # ------------------------------------------------------------------
    # Decoupler net plots (heatmap + per-cluster bars + dotplot)
    # ------------------------------------------------------------------
    if cfg.make_figures and getattr(cfg, "run_decoupler", False):
        # Tune defaults per resource (PROGENy is small; DoRothEA can be large)
        plot_utils.plot_decoupler_all_styles(
            adata,
            net_key="msigdb",
            net_name="MSigDB",
            figdir=Path("cluster_and_annotate"),
            heatmap_top_k=30,
            bar_top_n=10,
            dotplot_top_k=30,
        )

        plot_utils.plot_decoupler_all_styles(
            adata,
            net_key="progeny",
            net_name="PROGENy",
            figdir=Path("cluster_and_annotate"),
            heatmap_top_k=14,   # PROGENy pathways are ~14 (human)
            bar_top_n=8,
            dotplot_top_k=14,
        )

        plot_utils.plot_decoupler_all_styles(
            adata,
            net_key="dorothea",
            net_name="DoRothEA",
            figdir=Path("cluster_and_annotate"),
            heatmap_top_k=40,
            bar_top_n=10,
            dotplot_top_k=35,
        )

    # Make 'plateaus' HDF5-safe: JSON-encode list of dicts if present
    ca_uns = adata.uns.get("cluster_and_annotate", {})
    clustering = ca_uns.get("clustering", {})
    plateaus = clustering.get("plateaus", None)
    if isinstance(plateaus, list):
        clustering["plateaus"] = json.dumps(plateaus)
        ca_uns["clustering"] = clustering
        adata.uns["cluster_and_annotate"] = ca_uns

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
