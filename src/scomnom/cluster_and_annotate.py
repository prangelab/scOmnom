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
        adata_ct = adata.copy()

        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)

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

        raw = preds.predicted_labels

        if isinstance(raw, pd.DataFrame):
            # usually a single column
            labels = raw.squeeze(axis=1).to_numpy().ravel()
        elif isinstance(raw, pd.Series):
            labels = raw.to_numpy().ravel()
        else:
            # ndarray-like; flatten
            labels = np.asarray(raw).ravel()

        if labels.size != adata.n_obs:
            LOGGER.warning(
                "CellTypist returned %d labels for %d cells; ignoring CellTypist outputs.",
                labels.size,
                adata.n_obs,
            )
            return None, None

        # probability_matrix: pd.DataFrame, index must match adata.obs_names
        prob_matrix = preds.probability_matrix
        prob_matrix = prob_matrix.loc[adata.obs_names]

        LOGGER.info(
            "CellTypist precompute completed: %d labels, probability_matrix shape=%s.",
            labels.size,
            prob_matrix.shape,
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
    bio_homogeneity: Optional[Dict[float, float]] = None
    bio_fragmentation: Optional[Dict[float, float]] = None
    bio_ari: Optional[Dict[float, float]] = None


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
                mean_stability=mean_stability,
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
    Main entry point for resolution selection.

    Uses:
    - smoothed ARI stability
    - centroid-based silhouette
    - tiny-cluster penalty
    - structural plateau detection
    - optional biological metrics:
        * homogeneity
        * fragmentation penalty
        * ARI vs CellTypist labels
    - local-maxima fallback
    - border guard to avoid selecting max resolution spuriously
    """
    if metrics.ari_adjacent is None:
        ari_adjacent = _compute_ari_adjacent(
            resolutions=metrics.resolutions,
            labels_per_resolution=metrics.labels_per_resolution,
        )
    else:
        ari_adjacent = metrics.ari_adjacent

    stability = _compute_smoothed_stability(
        resolutions=metrics.resolutions,
        ari_adjacent=ari_adjacent,
    )

    tiny_penalty: Dict[float, float] = {}
    for r in sorted(metrics.resolutions):
        tiny_penalty[r] = compute_tiny_cluster_penalty(
            metrics.cluster_sizes[r],
            config.tiny_cluster_size,
        )

    tiny_vals = np.array(list(tiny_penalty.values()), dtype=float)
    tiny_min, tiny_max = float(np.nanmin(tiny_vals)), float(np.nanmax(tiny_vals))
    if tiny_max - tiny_min > 1e-9:
        tiny_norm = {
            r: (tiny_penalty[r] - tiny_min) / (tiny_max - tiny_min)
            for r in tiny_penalty
        }
    else:
        tiny_norm = {r: 1.0 for r in tiny_penalty}

    sil_norm = _normalize_scores(metrics.silhouette)
    stab_norm = _normalize_scores(stability)

    use_bio_effective = (
        config.use_bio
        and metrics.bio_homogeneity is not None
        and metrics.bio_fragmentation is not None
        and metrics.bio_ari is not None
    )

    if config.use_bio and not use_bio_effective:
        LOGGER.warning(
            "bio_guided_clustering=True, but biological metrics are unavailable. "
            "Falling back to structural-only resolution selection."
        )

    hom_norm: Dict[float, float] = {}
    frag_good_norm: Dict[float, float] = {}
    bioari_norm: Dict[float, float] = {}

    if use_bio_effective:
        hom_norm = _normalize_scores(metrics.bio_homogeneity)
        frag_raw_norm = _normalize_scores(metrics.bio_fragmentation)
        frag_good_norm = {r: 1.0 - frag_raw_norm.get(r, 0.0) for r in frag_raw_norm}
        bioari_norm = _normalize_scores(metrics.bio_ari)

    def composite_score(r: float) -> float:
        s = (
            config.w_stab * stab_norm.get(r, 0.0)
            + config.w_sil * sil_norm.get(r, 0.0)
            + config.w_tiny * tiny_norm.get(r, 0.0)
        )
        if use_bio_effective:
            s += (
                config.w_hom * hom_norm.get(r, 0.0)
                + config.w_frag * frag_good_norm.get(r, 0.0)
                + config.w_bioari * bioari_norm.get(r, 0.0)
            )
        return float(s)

    all_scores = {r: composite_score(r) for r in metrics.resolutions}

    # Plateau-first strategy (structural-only)
    plateaus = _detect_plateaus(metrics, config, stability)

    if plateaus:
        plateaus_sorted = sorted(
            plateaus,
            key=lambda p: (p.mean_stability, len(p.resolutions)),
            reverse=True,
        )
        best_plateau = plateaus_sorted[0]
        plateau_res = sorted(best_plateau.resolutions)

        mid_idx = len(plateau_res) // 2
        best_resolution = float(plateau_res[mid_idx])

        LOGGER.info(
            "Selected resolution %.3f from plateau [%s] with mean stability=%.3f.",
            best_resolution,
            ", ".join(f"{r:.3f}" for r in plateau_res),
            best_plateau.mean_stability,
        )

        return ResolutionSelectionResult(
            best_resolution=best_resolution,
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            bio_homogeneity=metrics.bio_homogeneity,
            bio_fragmentation=metrics.bio_fragmentation,
            bio_ari=metrics.bio_ari,
        )

    # No plateau: fallback logic
    sorted_res = sorted(metrics.resolutions)
    max_r = max(sorted_res)
    min_r = min(sorted_res)

    best_r_by_score = max(all_scores, key=all_scores.get)

    if (
        float(best_r_by_score) == float(max_r)
        and metrics.silhouette.get(max_r, -1.0) < 0.15
    ):
        LOGGER.info(
            "Border guard: composite optimum at max resolution %.3f but silhouette=%.3f < 0.15. "
            "Falling back to lowest resolution.",
            max_r,
            metrics.silhouette.get(max_r, float("nan")),
        )
        best_resolution = float(min_r)
        return ResolutionSelectionResult(
            best_resolution=best_resolution,
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            bio_homogeneity=metrics.bio_homogeneity,
            bio_fragmentation=metrics.bio_fragmentation,
            bio_ari=metrics.bio_ari,
        )

    scores_arr = np.array([all_scores[r] for r in sorted_res])
    n = len(sorted_res)

    local_max_idx = []
    for i in range(1, n - 1):
        if scores_arr[i] > scores_arr[i - 1] and scores_arr[i] > scores_arr[i + 1]:
            local_max_idx.append(i)

    local_max_idx = [i for i in local_max_idx if i not in (0, n - 1)]

    if local_max_idx:
        best_idx = max(local_max_idx, key=lambda k: scores_arr[k])
        best_resolution = float(sorted_res[best_idx])

        LOGGER.info(
            "No plateau; selecting strongest local maximum at %.3f (score=%.3f).",
            best_resolution,
            scores_arr[best_idx],
        )

        return ResolutionSelectionResult(
            best_resolution=best_resolution,
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            bio_homogeneity=metrics.bio_homogeneity,
            bio_fragmentation=metrics.bio_fragmentation,
            bio_ari=metrics.bio_ari,
        )

    if float(best_r_by_score) in (float(min_r), float(max_r)):
        interior = sorted_res[1:-1]
        interior_best = max(interior, key=lambda r: all_scores[r])
        best_resolution = float(interior_best)

        LOGGER.info(
            "No plateau and no local maxima; global score peak was at border. "
            "Selecting best interior resolution %.3f (score=%.3f).",
            best_resolution,
            all_scores[best_resolution],
        )
    else:
        best_resolution = float(best_r_by_score)
        LOGGER.info(
            "No plateau and no local maxima; selected %.3f by global composite score.",
            best_resolution,
        )

    return ResolutionSelectionResult(
        best_resolution=best_resolution,
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

    use_bio = bool(getattr(cfg, "bio_guided_clustering", False)) and (
        celltypist_labels is not None
    )
    if getattr(cfg, "bio_guided_clustering", False) and celltypist_labels is None:
        LOGGER.warning(
            "bio_guided_clustering=True, but CellTypist labels are unavailable. "
            "Resolution sweep will use structural metrics only."
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
            hom = _compute_bio_homogeneity(labels, celltypist_labels)
            frag = _compute_bio_fragmentation(labels, celltypist_labels)
            ari_bio = adjusted_rand_score(labels, celltypist_labels)

            bio_hom[res_f] = hom
            bio_frag[res_f] = frag
            bio_ari[res_f] = float(ari_bio)

            LOGGER.info(
                "Resolution %.2f: bio_homogeneity=%.3f, bio_fragmentation=%.3f, bio_ARI=%.3f",
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
        cluster_counts={r: n for r, n in zip(res_list, n_clusters_list)},
        cluster_sizes=cluster_sizes,
        labels_per_resolution=clusterings_float,
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
        use_bio=getattr(cfg, "bio_guided_clustering", False),
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


def _ssgsea_single_gmt(
    expr_df: pd.DataFrame,
    gmt: Path | str,
    cfg: ClusterAnnotateConfig,
):
    """
    Run ssGSEA for a single GMT file on the provided expression DataFrame.

    Parameters
    ----------
    expr_df
        Expression matrix (genes x cells), columns are cell IDs.
    gmt
        Path or identifier of the GMT file.
    cfg
        ClusterAnnotateConfig with ssGSEA parameters.

    Returns
    -------
    pd.DataFrame
        ES scores with shape (n_cells x n_pathways), columns are pathway names.
    """
    import gseapy as gp

    gmt_str = str(gmt)
    LOGGER.info(
        "ssGSEA: running on gene_sets='%s' for %d cells and %d genes.",
        gmt_str,
        expr_df.shape[1],
        expr_df.shape[0],
    )

    ss = gp.ssgsea(
        data=expr_df,
        gene_sets=gmt_str,
        outdir=None,
        sample_norm_method=getattr(cfg, "ssgsea_sample_norm_method", "rank"),
        min_size=int(getattr(cfg, "ssgsea_min_size", 10)),
        max_size=int(getattr(cfg, "ssgsea_max_size", 500)),
        threads=1,  # we parallelize outside gseapy
        no_plot=True,
    )

    res2d = ss.res2d
    if not isinstance(res2d, pd.DataFrame):
        res2d = pd.DataFrame(res2d)

    # pathways x samples (ES only)
    es = res2d.pivot(index="Term", columns="Name", values="ES")

    # Ensure sample order matches expr_df columns (cells)
    es = es.reindex(columns=expr_df.columns)

    # Transpose to cells x pathways
    es = es.T

    # Prefix columns with library stem to avoid collisions
    lib_prefix = Path(gmt_str).stem
    es.columns = [f"{lib_prefix}::{term}" for term in es.columns]

    return es


def _ssgsea_on_cell_chunk(
    expr_df: pd.DataFrame,
    gmt_files: Sequence[Path | str],
    cfg: ClusterAnnotateConfig,
) -> pd.DataFrame:
    """
    Run ssGSEA on a subset of cells (columns) for all GMT files.

    Parameters
    ----------
    expr_df
        Expression matrix (genes x cells) for this chunk.
    gmt_files
        Sequence of GMT file paths.
    cfg
        ClusterAnnotateConfig.

    Returns
    -------
    pd.DataFrame
        ES scores with shape (cells_in_chunk x total_pathways).
    """
    all_scores: list[pd.DataFrame] = []

    for gmt in gmt_files:
        try:
            es = _ssgsea_single_gmt(expr_df, gmt, cfg)
            all_scores.append(es)
        except Exception as e:
            LOGGER.warning(
                "ssGSEA failed for gene_sets='%s' on chunk: %s",
                gmt,
                e,
            )

    if not all_scores:
        raise RuntimeError("No successful ssGSEA results for this chunk.")

    return pd.concat(all_scores, axis=1)

def _run_ssgsea(adata: ad.AnnData, cfg: ClusterAnnotateConfig) -> None:
    """
    Run ssGSEA on CLUSTER-AGGREGATED expression

    Stores:
      - adata.uns["ssgsea_cluster_means"] : DataFrame (clusters x pathways)
      - adata.uns["ssgsea"]["config"]     : config snapshot
    """
    if not getattr(cfg, "run_ssgsea", False):
        LOGGER.info("ssGSEA disabled (run_ssgsea=False); skipping.")
        return

    try:
        import gseapy as gp  # noqa: F401
    except ImportError:
        LOGGER.warning(
            "gseapy is not installed; cannot run ssGSEA. Install with `pip install gseapy`."
        )
        return

    # --------------------------------------------------
    # 1) Resolve gene set GMT files
    # --------------------------------------------------
    try:
        gmt_files, used_keywords = resolve_msigdb_gene_sets(cfg.ssgsea_gene_sets)
    except Exception as e:
        LOGGER.warning("ssGSEA: failed to resolve gene sets: %s", e)
        return
    if not gmt_files:
        LOGGER.warning("ssGSEA: no gene set files resolved; skipping.")
        return

    # --------------------------------------------------
    # 2) Choose clustering key (must exist)
    # --------------------------------------------------
    cluster_key = None
    if "cluster_label" in adata.obs:
        cluster_key = "cluster_label"
    elif getattr(cfg, "final_auto_idents_key", None) and cfg.final_auto_idents_key in adata.obs:
        cluster_key = cfg.final_auto_idents_key
    elif getattr(cfg, "label_key", None) and cfg.label_key in adata.obs:
        cluster_key = cfg.label_key

    if cluster_key is None:
        LOGGER.warning("ssGSEA: no cluster key found in adata.obs; skipping.")
        return

    # --------------------------------------------------
    # 3) Pick expression source (raw counts layer if possible)
    # --------------------------------------------------
    from scipy import sparse

    X = adata.X
    genes = adata.var_names

    if getattr(cfg, "ssgsea_use_raw", True):
        if "counts_raw" in adata.layers:
            LOGGER.info("ssGSEA using adata.layers['counts_raw'] (counts_raw layer).")
            X = adata.layers["counts_raw"]
            genes = adata.var_names
        elif "counts_cb" in adata.layers:
            LOGGER.info("ssGSEA using adata.layers['counts_cb'] (cellbender layer).")
            X = adata.layers["counts_cb"]
            genes = adata.var_names
        elif adata.raw is not None:
            LOGGER.info("ssGSEA using adata.raw.X.")
            X = adata.raw.X
            genes = adata.raw.var_names
        else:
            LOGGER.info(
                "ssGSEA: requested raw, but no counts_raw/counts_cb/adata.raw; using adata.X."
            )
            X = adata.X
            genes = adata.var_names

    # --------------------------------------------------
    # 4) Aggregate expression by cluster WITHOUT densifying full matrix
    #    Output: genes x clusters DataFrame
    # --------------------------------------------------
    agg = getattr(cfg, "ssgsea_aggregate", "mean").lower().strip()
    if agg not in {"mean", "median"}:
        agg = "mean"

    cl = adata.obs[cluster_key].astype(str).to_numpy()
    clusters = pd.Index(pd.unique(cl), dtype=str)

    X_csr = X.tocsr() if sparse.issparse(X) else np.asarray(X)

    expr_cols = {}
    LOGGER.info(
        "ssGSEA: aggregating expression by '%s' using %s over %d clusters.",
        cluster_key,
        agg,
        len(clusters),
    )

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

        expr_cols[c] = vec

    if not expr_cols:
        LOGGER.warning("ssGSEA: no clusters produced aggregated expression; skipping.")
        return

    expr_df = pd.DataFrame(expr_cols, index=pd.Index(genes, name="gene"))
    LOGGER.info(
        "ssGSEA: running on aggregated matrix: %d genes × %d clusters.",
        expr_df.shape[0],
        expr_df.shape[1],
    )

    # --------------------------------------------------
    # 5) Run ssGSEA over GMTs in parallel (always) and concatenate pathways
    # --------------------------------------------------
    def _ssgsea_one_gmt(expr_df_: pd.DataFrame, gmt: str, *, threads: int) -> pd.DataFrame:
        # Prevent oversubscription from BLAS/OpenMP inside each job
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        import gseapy as gp

        ss = gp.ssgsea(
            data=expr_df_,
            gene_sets=str(gmt),
            outdir=None,
            sample_norm_method=getattr(cfg, "ssgsea_sample_norm_method", "rank"),
            min_size=int(getattr(cfg, "ssgsea_min_size", 10)),
            max_size=int(getattr(cfg, "ssgsea_max_size", 500)),
            threads=int(max(1, threads)),
            no_plot=True,
        )

        res2d = ss.res2d
        if not isinstance(res2d, pd.DataFrame):
            res2d = pd.DataFrame(res2d)

        es = res2d.pivot(index="Term", columns="Name", values="ES")  # pathways x clusters
        es = es.reindex(columns=expr_df_.columns)  # ensure cluster order
        es = es.T  # clusters x pathways

        lib_prefix = Path(str(gmt)).stem
        es.columns = [f"{lib_prefix}::{term}" for term in es.columns]
        return es

    # Decide parallel shape
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None

    nproc = int(getattr(cfg, "ssgsea_nproc", 1) or 1)
    nproc_eff = max(1, nproc - 1)  # <-- your rule: avoid oversubscribing, leave 1 core
    n_jobs = min(len(gmt_files), nproc_eff)
    threads_per_job = max(1, nproc_eff // n_jobs)

    LOGGER.info(
        "ssGSEA parallel (over GMTs): G=%d, nproc=%d → nproc_eff=%d, n_jobs=%d, threads/job=%d",
        len(gmt_files),
        nproc,
        nproc_eff,
        n_jobs,
        threads_per_job,
    )

    all_es: list[pd.DataFrame] = []

    if Parallel is None or delayed is None or n_jobs <= 1:
        # Fallback: sequential GMTs, but still use threads_per_job to exploit cores safely
        for gmt in gmt_files:
            try:
                all_es.append(_ssgsea_one_gmt(expr_df, gmt, threads=threads_per_job))
            except Exception as e:
                LOGGER.warning("ssGSEA failed for gene_sets='%s': %s", gmt, e)
    else:
        # Always parallel over GMTs
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_ssgsea_one_gmt)(expr_df, gmt, threads=threads_per_job)
            for gmt in gmt_files
        )
        # Filter any Nones (shouldn't happen unless you change worker)
        all_es = [x for x in results if isinstance(x, pd.DataFrame) and not x.empty]

    if not all_es:
        LOGGER.warning("ssGSEA: no successful results; skipping.")
        return

    ssgsea_cluster = pd.concat(all_es, axis=1)

    # --------------------------------------------------
    # 6) Store only cluster-level results
    # --------------------------------------------------
    adata.uns["ssgsea_cluster_means"] = ssgsea_cluster

    adata.uns.setdefault("ssgsea", {})
    adata.uns["ssgsea"]["config"] = {
        "run_ssgsea": True,
        "mode": "cluster",
        "cluster_key": cluster_key,
        "aggregate": agg,
        "gene_sets": [str(g) for g in gmt_files],
        "used_keywords": used_keywords,
        "use_raw": bool(getattr(cfg, "ssgsea_use_raw", True)),
        "min_size": int(getattr(cfg, "ssgsea_min_size", 10)),
        "max_size": int(getattr(cfg, "ssgsea_max_size", 500)),
        "sample_norm_method": getattr(cfg, "ssgsea_sample_norm_method", "rank"),
        "backend": "gseapy_ssgsea_cluster_aggregate_parallel_over_gmts",
        "nproc": nproc,
        "nproc_eff": nproc_eff,
        "n_jobs": n_jobs,
        "threads_per_job": threads_per_job,
    }

    LOGGER.info(
        "Stored ssGSEA cluster means in adata.uns['ssgsea_cluster_means']: %d clusters × %d pathways.",
        ssgsea_cluster.shape[0],
        ssgsea_cluster.shape[1],
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

    best_res, sweep, clusterings = _resolution_sweep(
        adata,
        cfg,
        embedding_key,
        celltypist_labels=celltypist_labels,
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

    # Optional: per-cell ssGSEA enrichment (Hallmark/Reactome)
    try:
        _run_ssgsea(adata, cfg)
    except Exception as e:
        LOGGER.warning("ssGSEA step failed: %s", e)

    if cfg.make_figures and getattr(cfg, "run_ssgsea", False) and "ssgsea_cluster_means" in adata.uns:
        plot_utils.plot_ssgsea_cluster_topn_heatmap(
            adata,
            cluster_key = "cluster_label",
            figdir = figdir_cluster,
            n = 5,
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
