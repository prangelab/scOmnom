from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from .config import ClusterAnnotateConfig
from .io_utils import get_celltypist_model
from . import plot_utils

LOGGER = logging.getLogger(__name__)

# Canonical pretty cluster label column
CLUSTER_LABEL_KEY = "cluster_label"


# -------------------------------------------------------------------------
# Rounds plumbing (cluster rounds scaffold + registration + activation helpers)
# -------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
        if cluster_sizes:
            cluster_id_map = {cid: cid for cid in cluster_sizes.keys()}
        else:
            cluster_id_map = {}

    if cluster_renumbering is None:
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
        # schema-ish linkage & audit helpers
        "cluster_sizes": cluster_sizes,
        "cluster_id_map": dict(cluster_id_map),
        "cluster_renumbering": dict(cluster_renumbering),
        "labels": labels_cache,
        # precreate slots
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


def set_active_round(
    adata: ad.AnnData,
    round_id: str,
    *,
    publish_decoupler: bool = True,
) -> None:
    """
    Canonical linkage contract (CLUSTER_LABEL aliasing is owned by annotation module).
    This clustering_utils version only mirrors round labels into canonical cluster_key
    and syncs cluster_key colors (best-effort).

    If your main module previously also published decoupler to top-level here,
    keep that behavior in the orchestrator (or move publish helper into annotation_utils).
    """
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"set_active_round: round_id {round_id!r} not found")

    r = rounds[round_id]
    adata.uns["active_cluster_round"] = round_id

    cluster_key = r.get("cluster_key", None)
    labels_obs_key = r.get("labels_obs_key", None)

    if cluster_key and labels_obs_key and labels_obs_key in adata.obs:
        adata.obs[str(cluster_key)] = adata.obs[str(labels_obs_key)]
    elif cluster_key and cluster_key in adata.obs:
        pass

    if cluster_key and cluster_key in adata.obs:
        if not pd.api.types.is_categorical_dtype(adata.obs[cluster_key]):
            adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

        try:
            if labels_obs_key:
                src = f"{labels_obs_key}_colors"
                dst = f"{cluster_key}_colors"
                if src in adata.uns:
                    adata.uns[dst] = list(adata.uns[src])
        except Exception:
            pass

    # keep signature parity; publishing decoupler is done elsewhere
    _ = publish_decoupler


# -------------------------------------------------------------------------
# Embedding + clustering helpers
# -------------------------------------------------------------------------
def _ensure_embedding(adata: ad.AnnData, embedding_key: str) -> str:
    """
    Ensure the chosen embedding exists; if not, try to recover from integration metadata.
    Returns the actual embedding key to use.
    """
    if embedding_key in adata.obsm:
        return embedding_key

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


def robust_cluster_jump(k_prev, k_curr, alpha=10) -> float:
    """
    Robust jump metric:
    jump = |k_curr - k_prev| / max(k_prev, alpha)
    Prevents division by very small k.
    """
    denom = max(k_prev, alpha)
    return abs(k_curr - k_prev) / denom


def _centroid_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute a centroid-based separation score in the given embedding X.
    Mean "separation" based on nearest centroid vs mean centroid distance.
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
    D = np.linalg.norm(diff, axis=2)

    s_vals = []
    for i in range(k):
        d_i = D[i].copy()
        d_i[i] = np.inf
        a_i = float(np.min(d_i))
        b_i = float(np.mean(d_i[np.isfinite(d_i)])) if np.isfinite(d_i).any() else 0.0
        denom = max(a_i, b_i)
        s_i = 0.0 if denom <= 0.0 else (b_i - a_i) / denom
        s_vals.append(s_i)

    return float(np.mean(s_vals)) if s_vals else float("nan")


# -------------------------------------------------------------------------
# CellTypist precompute + bio mask (used by BISC sweep only; not decoupler)
# -------------------------------------------------------------------------
def _precompute_celltypist(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Run CellTypist once to obtain per-cell predictions and probability matrix.

    Policy:
      - Trust counts-like layers only: "counts_raw" then "counts_cb".
      - NEVER assume adata.raw is counts-like.
      - If neither exists, fall back to adata.X as-is (no normalize/log).
    """
    if cfg.celltypist_model is None:
        LOGGER.info("No CellTypist model provided; skipping CellTypist precompute.")
        return None, None

    try:
        LOGGER.info("Running CellTypist precompute (predictions + probabilities).")

        picked_layer: Optional[str] = None
        X_src = None

        for layer in ("counts_raw", "counts_cb"):
            if layer in adata.layers:
                picked_layer = layer
                X_src = adata.layers[layer]
                break

        if picked_layer is not None:
            LOGGER.info("CellTypist input: using counts-like layer adata.layers[%r].", picked_layer)
            adata_ct = ad.AnnData(
                X=X_src,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata_ct.obs_names = adata.obs_names.copy()
            adata_ct.var_names = adata.var_names.copy()
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)
        else:
            LOGGER.warning(
                "CellTypist input: no counts-like layers found ('counts_raw'/'counts_cb'). "
                "Falling back to adata.X as-is (no normalize_total/log1p)."
            )
            adata_ct = adata.copy()

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

        prob_matrix = preds.probability_matrix
        if not isinstance(prob_matrix, pd.DataFrame) or prob_matrix.empty:
            LOGGER.warning("CellTypist returned no/empty probability_matrix; returning labels only.")
            return labels, None

        try:
            prob_matrix = prob_matrix.loc[adata.obs_names]
        except Exception:
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

    HARD GUARANTEE (per your requirement):
      - Always creates round-scoped cluster-level labels and pretty labels,
        even if CellTypist is disabled or fails.

    Writes (always):
      - adata.obs[f"{cfg.celltypist_cluster_label_key}__{round_id}"]   (cluster-level CT label; often "Unknown")
      - adata.obs[f"{CLUSTER_LABEL_KEY}__{round_id}"]                  (pretty label; always string/categorical)
      - plus aliases:
          adata.obs[cfg.celltypist_cluster_label_key]  -> latest round
          adata.obs[CLUSTER_LABEL_KEY]                 -> latest round

    Writes (best-effort):
      - adata.obs[cfg.celltypist_label_key] (cell-level CT label; "Unknown" if unavailable)
      - adata.obsm["celltypist_proba"] + adata.uns["celltypist_proba_columns"] if available

    Returns plotting keys dict (never None unless cluster_key missing).
    """
    if cluster_key not in adata.obs:
        raise KeyError(
            f"_run_celltypist_annotation: cluster_key '{cluster_key}' not found in adata.obs"
        )

    # Determine round_id (best effort)
    if round_id is None:
        rid = adata.uns.get("active_cluster_round", None)
        round_id = str(rid) if rid else "r0"
    else:
        round_id = str(round_id)

    # Round-scoped keys (avoid overwriting across rounds)
    cell_key = str(cfg.celltypist_label_key)
    cluster_ct_base = str(cfg.celltypist_cluster_label_key)
    cluster_ct_key = f"{cluster_ct_base}__{round_id}"
    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"

    # --------------------------------------------------------------
    # Helper: stable "Leiden-style" cluster ordering by size
    # --------------------------------------------------------------
    def _cluster_order_by_size(labels: pd.Series) -> list[str]:
        """
        Return cluster ids ordered by:
          1) descending size
          2) stable tie-break by cluster id (string)
        """
        s = labels.astype(str)
        vc = s.value_counts(dropna=False)  # desc by default
        df = pd.DataFrame({"cluster": vc.index.astype(str), "n": vc.values.astype(int)})
        df["cluster_sort"] = df["cluster"].astype(str)
        df = df.sort_values(["n", "cluster_sort"], ascending=[False, True], kind="mergesort")
        return df["cluster"].astype(str).tolist()

    # --------------------------------------------------------------
    # A) CellTypist predictions (cell-level + probabilities) - BEST EFFORT
    #    If unavailable, we fill cell_key with "Unknown".
    # --------------------------------------------------------------
    celltypist_ok = False
    try:
        if cfg.celltypist_model is None:
            raise RuntimeError("CellTypist disabled (cfg.celltypist_model is None).")

        if precomputed_labels is not None:
            if precomputed_labels.shape[0] != adata.n_obs:
                raise ValueError("precomputed_labels length does not match adata.n_obs.")
            LOGGER.info("Using precomputed CellTypist labels for annotation.")
            adata.obs[cell_key] = pd.Series(precomputed_labels, index=adata.obs_names).astype(str).astype("category")

            if precomputed_proba is not None:
                try:
                    pm = precomputed_proba.loc[adata.obs_names]
                except Exception:
                    pm = precomputed_proba.reindex(adata.obs_names)
                if isinstance(pm, pd.DataFrame) and not pm.empty:
                    adata.obsm["celltypist_proba"] = pm.to_numpy()
                    adata.uns["celltypist_proba_columns"] = list(pm.columns.astype(str))
            celltypist_ok = True


        else:

            # Fallback path (kept for safety) — IMPORTANT:
            # Do NOT mutate `adata` (normalize_total/log1p are in-place).
            # Build a minimal scratch AnnData that copies ONLY the expression matrix.
            LOGGER.info("Running CellTypist on scratch AnnData (fallback; non-mutating).")
            picked_layer: Optional[str] = None
            X_src = None
            for layer in ("counts_raw", "counts_cb"):
                if layer in adata.layers:
                    picked_layer = layer
                    X_src = adata.layers[layer]
                    break

            if X_src is None:
                # Last resort: use adata.X, but must copy to avoid mutating original
                X_src = adata.X
                LOGGER.warning(
                    "CellTypist fallback input: no counts-like layers found ('counts_raw'/'counts_cb'). "
                    "Using adata.X, but copying matrix to avoid in-place mutation."
                )
            else:
                LOGGER.info("CellTypist fallback input: using counts-like layer adata.layers[%r].", picked_layer)
            # Copy ONLY the matrix (sparse-preserving). This is the minimal safe copy.
            try:
                X_ct = X_src.copy()
            except Exception:
                # extremely defensive fallback; should be rare
                import numpy as _np
                X_ct = _np.array(X_src, copy=True)
            # Build minimal AnnData: avoids copying obsm/uns/etc; just what CellTypist needs
            adata_ct = ad.AnnData(
                X=X_ct,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata_ct.obs_names = adata.obs_names.copy()
            adata_ct.var_names = adata.var_names.copy()

            # Apply standard preproc on scratch object only
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)
            model_path = get_celltypist_model(cfg.celltypist_model)
            from celltypist.models import Model
            import celltypist
            model = Model.load(str(model_path))
            predictions = celltypist.annotate(
                adata_ct,
                model=model,
                majority_voting=cfg.celltypist_majority_voting,
            )

            raw = predictions.predicted_labels
            if isinstance(raw, dict) and "majority_voting" in raw:
                cell_level_labels = raw["majority_voting"]

            else:
                cell_level_labels = raw

            # write cell-level labels (to real adata)
            if isinstance(cell_level_labels, (pd.Series, pd.DataFrame)):
                s = cell_level_labels.squeeze()
                adata.obs[cell_key] = s.astype(str).astype("category")

            else:
                adata.obs[cell_key] = pd.Series(
                    np.asarray(cell_level_labels).ravel(), index=adata.obs_names
                ).astype(str).astype("category")

            # probability matrix if available (to real adata)
            if hasattr(predictions, "probability_matrix"):
                pm = predictions.probability_matrix
                if isinstance(pm, pd.DataFrame) and not pm.empty:
                    try:
                        pm = pm.loc[adata.obs_names]
                    except Exception:
                        pm = pm.reindex(adata.obs_names)
                    adata.obsm["celltypist_proba"] = pm.to_numpy()
                    adata.uns["celltypist_proba_columns"] = list(pm.columns.astype(str))

            celltypist_ok = True

    except Exception as e:
        LOGGER.warning("CellTypist unavailable/failed; proceeding with Unknown labels. (%s)", e)
        # Ensure cell_key exists as "Unknown" for all cells (so downstream always has a string column)
        adata.obs[cell_key] = pd.Series(["Unknown"] * adata.n_obs, index=adata.obs_names).astype("category")
        # Do NOT delete any existing proba; but don't assume it's valid either.
        celltypist_ok = False

    # --------------------------------------------------------------
    # B) Cluster-level majority CellTypist label (ROUND-SCOPED)
    #    Mask-aware if probability matrix exists; otherwise unmasked.
    # --------------------------------------------------------------
    bio_mask = None
    bio_mask_stats = None
    try:
        if "celltypist_proba" in adata.obsm and "celltypist_proba_columns" in adata.uns:
            pm = pd.DataFrame(
                adata.obsm["celltypist_proba"],
                index=adata.obs_names,
                columns=list(map(str, adata.uns["celltypist_proba_columns"])),
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

    if bio_mask is None or getattr(bio_mask, "shape", (None,))[0] != adata.n_obs:
        bio_mask = np.ones((adata.n_obs,), dtype=bool)

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

    cluster_sizes = tmp.groupby("cluster").size().to_dict()

    majority_map: dict[str, str] = {}
    for c, g in tmp.groupby("cluster", sort=False):
        g_masked = g[g["masked"]]
        n_total = int(cluster_sizes.get(c, len(g)))
        n_masked = int(g_masked.shape[0])

        # If too few confident cells OR CellTypist not actually OK -> Unknown
        if (not celltypist_ok) or (n_masked < min_masked_cells) or (n_total > 0 and (n_masked / n_total) < min_masked_frac):
            majority_map[str(c)] = "Unknown"
            continue

        vc = g_masked["ct"].value_counts()
        majority_map[str(c)] = str(vc.idxmax()) if not vc.empty else "Unknown"

    adata.obs[cluster_ct_key] = clust_vals.map(majority_map).astype("category")
    adata.obs[cluster_ct_base] = adata.obs[cluster_ct_key]  # alias to latest round

    # --------------------------------------------------------------
    # C) Pretty labels (ROUND-SCOPED) — ALWAYS
    #     IMPORTANT: numbering follows Leiden practice: sort clusters by size (desc).
    # --------------------------------------------------------------
    # Stable cluster order (desc size, tie-break by original cluster id)
    try:
        cluster_order = _cluster_order_by_size(clust_vals)
    except Exception:
        # fallback: stable string sort
        cluster_order = sorted(pd.unique(clust_vals.astype(str)).astype(str).tolist())

    ord_map = {c: f"C{i:02d}" for i, c in enumerate(cluster_order)}

    # Build pretty label per cell: "C00: <majority_label>"
    cl_to_maj = {str(k): str(v) for k, v in majority_map.items()}
    pretty = clust_vals.map(lambda c: f"{ord_map.get(str(c), 'C??')}: {cl_to_maj.get(str(c), 'Unknown')}")

    # Make categorical with categories ordered by size
    pretty_categories = [f"{ord_map[c]}: {cl_to_maj.get(str(c), 'Unknown')}" for c in cluster_order]
    adata.obs[pretty_key] = pd.Categorical(pretty.astype(str), categories=pretty_categories, ordered=False)
    adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]  # alias to latest round

    # Palette for round-scoped pretty labels + alias
    try:
        from scanpy.plotting.palettes import default_102
        cats_pretty = list(adata.obs[pretty_key].cat.categories)
        adata.uns[f"{pretty_key}_colors"] = list(default_102[: len(cats_pretty)])
        adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = adata.uns[f"{pretty_key}_colors"]
    except Exception as e:
        LOGGER.warning("Could not set pretty-label palette: %s", e)

    # --------------------------------------------------------------
    # D) Store linkage + mask stats into the round dict (if present)
    #    (Also stores the size-sorted cluster_order + display map for downstream consumers.)
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
                    "cluster_key_used": str(cluster_key),
                    "pretty_label_masked": True,
                    "pretty_label_min_masked_cells": int(min_masked_cells),
                    "pretty_label_min_masked_frac": float(min_masked_frac),
                    "celltypist_ok": bool(celltypist_ok),
                }
            )

            # NEW: stable ordering + mapping (useful for decoupler/pseudobulk/plots)
            rounds[round_id]["cluster_order"] = list(map(str, cluster_order))
            rounds[round_id]["cluster_display_map"] = {
                str(cid): f"{ord_map.get(str(cid), 'C??')}: {cl_to_maj.get(str(cid), 'Unknown')}"
                for cid in cluster_order
            }

            if bio_mask_stats is not None:
                rounds[round_id].setdefault("bio_mask", {})
                rounds[round_id]["bio_mask"]["annotation_mask_stats"] = bio_mask_stats
            adata.uns["cluster_rounds"] = rounds
    except Exception as e:
        LOGGER.warning("Failed to store round annotation linkage/mask stats: %s", e)

    LOGGER.info(
        "Annotation done for round '%s' using cluster_key='%s'. "
        "Wrote: cell='%s', cluster='%s', pretty='%s' (+ aliases). celltypist_ok=%s",
        round_id,
        cluster_key,
        cell_key,
        cluster_ct_key,
        pretty_key,
        bool(celltypist_ok),
    )

    return {
        "round_id": str(round_id),
        "cluster_key": str(cluster_key),
        "celltypist_cell_key": str(cell_key),
        "celltypist_cluster_key": str(cluster_ct_key),
        "pretty_cluster_key": str(pretty_key),
    }



def _celltypist_entropy_margin_mask(
    prob_matrix: pd.DataFrame,
    *,
    entropy_abs_limit: float,
    entropy_quantile: float,
    margin_min: float,
) -> tuple[np.ndarray, dict]:
    """
    good = (H <= max(H_abs, H_q)) AND (margin >= margin_min)
    """
    P = prob_matrix.to_numpy(dtype=np.float64, copy=False)
    n = P.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool), {"n_cells": 0}

    eps = 1e-12
    P_clip = np.clip(P, eps, 1.0)
    entropy = -np.sum(P_clip * np.log(P_clip), axis=1)

    top2 = np.partition(P, kth=-2, axis=1)[:, -2:]
    p1 = np.max(top2, axis=1)
    p2 = np.min(top2, axis=1)
    margin = p1 - p2

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
    if mask.shape[0] != n_obs:
        stats["disabled_reason"] = "mask_length_mismatch"
        return None, stats

    return mask, stats


# -------------------------------------------------------------------------
# Biological metrics (used in sweep; clustering-oriented)
# -------------------------------------------------------------------------
def _compute_bio_homogeneity(
    labels: np.ndarray,
    bio_labels: np.ndarray,
) -> float:
    df = pd.DataFrame({"cl": labels, "bio": bio_labels})
    groups = df.groupby("cl")
    homs: List[float] = []
    for _, g in groups:
        vc = g["bio"].value_counts()
        if vc.empty:
            continue
        homs.append(float(vc.iloc[0] / len(g)))
    return float(np.mean(homs)) if homs else 0.0


def _compute_bio_fragmentation(
    labels: np.ndarray,
    bio_labels: np.ndarray,
    frac_thr: float = 0.15,
) -> float:
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
    ari_adjacent: Dict[Tuple[float, float], float] = {}
    sorted_res = sorted(resolutions)
    for r1, r2 in zip(sorted_res[:-1], sorted_res[1:]):
        ari = adjusted_rand_score(labels_per_resolution[r1], labels_per_resolution[r2])
        ari_adjacent[(r1, r2)] = float(ari)
    return ari_adjacent


def _compute_smoothed_stability(
    resolutions: Sequence[float],
    ari_adjacent: Dict[Tuple[float, float], float],
) -> Dict[float, float]:
    sorted_res = sorted(resolutions)
    stab: Dict[float, float] = {}
    for i, r in enumerate(sorted_res):
        terms: List[float] = []
        if i > 0:
            r_prev = sorted_res[i - 1]
            if (r_prev, r) in ari_adjacent:
                terms.append(ari_adjacent[(r_prev, r)])
        if i < len(sorted_res) - 1:
            r_next = sorted_res[i + 1]
            if (r, r_next) in ari_adjacent:
                terms.append(ari_adjacent[(r, r_next)])
        stab[r] = float(np.mean(terms)) if terms else 0.0
    return stab


def _detect_plateaus(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
    stability: Dict[float, float],
) -> List[Plateau]:
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
                plateaus.append(Plateau(resolutions=current_segment.copy(), mean_stability=mean_stab))
            current_segment = []

    if len(current_segment) >= config.min_plateau_len:
        mean_stab = float(np.mean([stability[x] for x in current_segment]))
        plateaus.append(Plateau(resolutions=current_segment.copy(), mean_stability=mean_stab))

    return plateaus


def _normalize_scores(d: Dict[float, float]) -> Dict[float, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax == vmin:
        return {k: 0.0 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}


def compute_tiny_cluster_penalty(cluster_sizes: np.ndarray, tiny_threshold: int) -> float:
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

    return float(0.5 * (penalty_cluster_fraction + penalty_cell_fraction))


def select_best_resolution(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
) -> ResolutionSelectionResult:
    ari_adjacent = metrics.ari_adjacent or _compute_ari_adjacent(
        resolutions=metrics.resolutions,
        labels_per_resolution=metrics.labels_per_resolution,
    )
    stability = _compute_smoothed_stability(metrics.resolutions, ari_adjacent)

    sil_norm = _normalize_scores(metrics.silhouette)
    tiny_penalty = {
        float(r): compute_tiny_cluster_penalty(metrics.cluster_sizes[float(r)], config.tiny_cluster_size)
        for r in metrics.resolutions
    }
    tiny_norm = _normalize_scores(tiny_penalty)
    stab_norm = _normalize_scores(stability)

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

    sorted_res = sorted(float(r) for r in metrics.resolutions)
    interior = sorted_res[1:-1]

    min_stab_ok = 0.60
    feasible = [r for r in interior if stability.get(r, 0.0) >= min_stab_ok]

    if use_bio and metrics.n_bio_labels:
        max_clusters = 2.5 * metrics.n_bio_labels
        feasible = [r for r in feasible if metrics.cluster_counts.get(r, 0) <= max_clusters]

    SearchSet = feasible if feasible else interior

    def pick_parsimonious(cands, eps=0.03):
        if not cands:
            return None
        best = max(cands, key=lambda r: all_scores.get(r, -np.inf))
        best_val = all_scores[best]
        near = [r for r in cands if all_scores.get(r, -np.inf) >= (1 - eps) * best_val]
        return min(near) if near else best

    plateaus = _detect_plateaus(metrics, config, stability)
    if plateaus:
        best_plateau = max(plateaus, key=lambda p: (p.mean_stability, len(p.resolutions)))
        plateau_res = [float(r) for r in best_plateau.resolutions]
        plateau_res = [r for r in plateau_res if r in SearchSet]

        bio_plateau = None
        if use_bio and plateau_res:
            bioari_vals = [bioari_norm.get(r, np.nan) for r in plateau_res]
            if np.isfinite(bioari_vals).all():
                spread = max(bioari_vals) - min(bioari_vals)
                if spread < 0.05:
                    bio_plateau = plateau_res

        cands = bio_plateau if bio_plateau else plateau_res
        best = pick_parsimonious(cands)
        return ResolutionSelectionResult(
            best_resolution=float(best),
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            bio_homogeneity=metrics.bio_homogeneity,
            bio_fragmentation=metrics.bio_fragmentation,
            bio_ari=metrics.bio_ari,
        )

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
    best = knee if knee is not None else pick_parsimonious(SearchSet)

    return ResolutionSelectionResult(
        best_resolution=float(best),
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

    n_bio_labels_masked: Optional[int] = None
    if use_bio:
        n_bio_labels_masked = int(pd.unique(celltypist_labels[bio_mask]).size)

    for res in resolutions:
        res_f = float(res)

        # -----------------------------
        # Cleaner, human-facing logging
        # -----------------------------
        LOGGER.info("Running Leiden clustering at resolution %.2f", res_f)

        key = f"{cfg.label_key}_{res_f:.2f}"
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
        pen = sil - cfg.penalty_alpha * n_clusters

        silhouette_scores.append(sil)
        penalized_scores.append(pen)

        # -----------------------------
        # One-line quantitative summary
        # -----------------------------
        LOGGER.info(
            "  → %d clusters | silhouette=%.3f | penalized=%.3f | min/med/max size=%d/%d/%d",
            n_clusters,
            sil,
            pen,
            int(sizes.min()),
            int(np.median(sizes)),
            int(sizes.max()),
        )

        if use_bio:
            m = bio_mask
            if m.shape[0] != labels.shape[0]:
                raise ValueError("bio_mask length does not match number of cells.")

            labels_m = labels[m]
            bio_m = celltypist_labels[m]

            if labels_m.size >= 2 and np.unique(labels_m).size >= 2:
                bh = float(_compute_bio_homogeneity(labels_m, bio_m))
                bf = float(_compute_bio_fragmentation(labels_m, bio_m))
                ba = float(adjusted_rand_score(labels_m, bio_m))

                bio_hom[res_f] = bh
                bio_frag[res_f] = bf
                bio_ari[res_f] = ba

                LOGGER.info(
                    "    bio metrics: homogeneity=%.3f | fragmentation=%.3f | bio-ARI=%.3f",
                    bh,
                    bf,
                    ba,
                )

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

    sweep: Dict[str, object] = {
        "resolutions": np.array(res_list, dtype=float),
        "silhouette_scores": silhouette_scores,
        "n_clusters": n_clusters_list,
        "penalized_scores": penalized_scores,
        "composite_scores": [selection.scores[r] for r in res_list],
        "stability_scores": [selection.stability[r] for r in res_list],
        "tiny_cluster_penalty": [selection.tiny_cluster_penalty[r] for r in res_list],
        "cluster_sizes": cluster_sizes,
        "plateaus": [{"resolutions": p.resolutions, "mean_stability": p.mean_stability} for p in selection.plateaus],
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

    clusterings_str: Dict[str, np.ndarray] = {_res_key(r): labs for r, labs in clusterings_float.items()}
    return best_res, sweep, clusterings_str



def _subsampling_stability(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    best_res: float,
) -> List[float]:
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

    stability_aris: List[float] = []
    for i in range(cfg.stability_repeats):
        rng_i = np.random.default_rng(cfg.random_state + i)
        n_sub = int(round(cfg.subsample_frac * adata.n_obs))
        cells = rng_i.choice(adata.obs_names.to_numpy(), size=n_sub, replace=False)
        sub = adata[cells].copy()

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

    return stability_aris


def _apply_final_clustering(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    best_res: float,
) -> None:
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


def _final_real_silhouette_qc(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    figdir: Path,
    *,
    cluster_key: str,
    round_id: str | None = None,
) -> Optional[float]:
    if cluster_key not in adata.obs:
        LOGGER.warning("final_real_silhouette_qc: cluster_key '%s' not in adata.obs; skipping.", cluster_key)
        return None

    if round_id is None:
        rid = adata.uns.get("active_cluster_round", None)
        round_id = str(rid) if rid else None

    labels = adata.obs[cluster_key].to_numpy()
    if np.unique(labels).size < 2:
        LOGGER.warning("final_real_silhouette_qc: <2 clusters; skipping.")
        return None

    X = adata.obsm[embedding_key]
    sil_values = silhouette_samples(X, labels, metric="euclidean")
    sil_mean = float(np.mean(sil_values))

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


# -------------------------------------------------------------------------
# Public: Biology Informed Structural Clustering (BISC) — ROUNDS-ONLY
# -------------------------------------------------------------------------
def run_BISC(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    embedding_key: str,
    celltypist_labels: Optional[np.ndarray],
    celltypist_proba: Optional[pd.DataFrame],
    round_suffix: str = "BISC",
) -> ad.AnnData:
    _ensure_cluster_rounds(adata)

    if "neighbors" not in adata.uns:
        LOGGER.info("BISC: neighbors not found; computing neighbors using embedding_key=%r", embedding_key)
        sc.pp.neighbors(adata, use_rep=embedding_key)

    if "X_umap" not in adata.obsm:
        LOGGER.info("BISC: UMAP not found; computing UMAP.")
        sc.tl.umap(adata)

    bio_mask, bio_mask_stats = _maybe_build_bio_mask(cfg, celltypist_proba, adata.n_obs)

    best_res, sweep, _clusterings = _resolution_sweep(
        adata,
        cfg,
        embedding_key,
        celltypist_labels=celltypist_labels,
        bio_mask=bio_mask,
    )

    sweep = dict(sweep) if isinstance(sweep, dict) else {}
    sweep["bio_mask_stats"] = bio_mask_stats

    stability_aris = _subsampling_stability(adata, cfg, embedding_key, best_res)

    _apply_final_clustering(adata, cfg, best_res)

    idx = _next_round_index(adata)
    round_id = _make_round_id(idx, round_suffix)

    labels_obs_key = f"{cfg.label_key}__{round_id}"
    adata.obs[labels_obs_key] = adata.obs[cfg.label_key].astype(str).astype("category")

    try:
        if f"{cfg.label_key}_colors" in adata.uns:
            adata.uns[f"{labels_obs_key}_colors"] = list(adata.uns[f"{cfg.label_key}_colors"])
    except Exception:
        pass

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
        cluster_id_map=identity_map,
        cluster_renumbering=identity_renumbering,
        compacting=None,
        cache_labels=False,
    )

    set_active_round(adata, round_id, publish_decoupler=False)

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

    res_list = [float(r) for r in rinfo.get("sweep", {}).get("resolutions", [])]
    sil_list = rinfo.get("sweep", {}).get("silhouette_scores", []) or []
    n_list = rinfo.get("sweep", {}).get("n_clusters", []) or []
    pen_list = rinfo.get("sweep", {}).get("penalized_scores", []) or []

    rinfo.setdefault("diagnostics", {})
    rinfo["diagnostics"]["tested_resolutions"] = res_list
    rinfo["diagnostics"]["silhouette_centroid"] = {_res_key(r): float(s) for r, s in zip(res_list, sil_list)}
    rinfo["diagnostics"]["cluster_counts"] = {_res_key(r): int(n) for r, n in zip(res_list, n_list)}
    rinfo["diagnostics"]["penalized_scores"] = {_res_key(r): float(p) for r, p in zip(res_list, pen_list)}

    comp_list = sweep.get("composite_scores", None)
    stab_list = sweep.get("stability_scores", None)
    tiny_list = sweep.get("tiny_cluster_penalty", None)

    if isinstance(comp_list, list) and len(comp_list) == len(res_list):
        rinfo["diagnostics"]["composite_scores"] = {_res_key(r): float(v) for r, v in zip(res_list, comp_list)}
    if isinstance(stab_list, list) and len(stab_list) == len(res_list):
        rinfo["diagnostics"]["resolution_stability"] = {_res_key(r): float(v) for r, v in zip(res_list, stab_list)}
    if isinstance(tiny_list, list) and len(tiny_list) == len(res_list):
        rinfo["diagnostics"]["tiny_cluster_penalty"] = {_res_key(r): float(v) for r, v in zip(res_list, tiny_list)}

    rounds[round_id] = rinfo
    adata.uns["cluster_rounds"] = rounds

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
