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
from . import plot_utils, ct_utils

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
    if not isinstance(adata.uns.get("cluster_round_order", None), list):
        adata.uns["cluster_round_order"] = []


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


def _cluster_order_by_size(labels: pd.Series) -> list[str]:
    """
    Return cluster ids ordered by:
      1) descending size
      2) stable tie-break by cluster id (string)
    """
    s = labels.astype(str)
    vc = s.value_counts(dropna=False)
    df = pd.DataFrame({"cluster": vc.index.astype(str), "n": vc.values.astype(int)})
    df["cluster_sort"] = df["cluster"].astype(str)
    df = df.sort_values(["n", "cluster_sort"], ascending=[False, True], kind="mergesort")
    return df["cluster"].astype(str).tolist()


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

    if not isinstance(adata.uns.get("cluster_round_order", None), list):
        adata.uns["cluster_round_order"] = []
    if round_id not in adata.uns["cluster_round_order"]:
        adata.uns["cluster_round_order"].append(round_id)

    adata.uns["active_cluster_round"] = round_id


def _create_shallow_round_from_parent(
    adata: ad.AnnData,
    *,
    parent_round_id: str,
    round_name: str,
    new_round_id: str | None = None,
    round_type: str,
    kind: str,
    notes: str | None = None,
    set_active: bool = True,
    cluster_key: str | None = None,
    labels_obs_key: str | None = None,
    best_resolution: float | None = None,
    sweep: dict | None = None,
    cfg_snapshot: dict | None = None,
    cluster_id_map: dict[str, str] | None = None,
    cluster_renumbering: dict[str, str] | None = None,
    compacting: dict | None = None,
    inherit_fields: Sequence[str] = (
        "annotation",
        "decoupler",
        "qc",
        "stability",
        "diagnostics",
        "inputs",
        "bio_mask",
        "cluster_order",
        "cluster_display_map",
        "cluster_sizes",
    ),
) -> str:
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or parent_round_id not in rounds:
        raise KeyError(f"Parent round {parent_round_id!r} not found in adata.uns['cluster_rounds'].")

    parent = rounds[parent_round_id]
    if not isinstance(parent, dict):
        raise TypeError(f"Parent round {parent_round_id!r} must be a dict.")

    cluster_key_use = str(cluster_key or parent.get("cluster_key", "leiden"))
    labels_obs_key_use = str(labels_obs_key or parent.get("labels_obs_key", cluster_key_use))

    if labels_obs_key_use not in adata.obs:
        raise KeyError(f"labels_obs_key '{labels_obs_key_use}' not found in adata.obs.")

    if new_round_id is None:
        idx = _next_round_index(adata)
        new_round_id = _make_round_id(idx, round_name)

    prev_active_round = adata.uns.get("active_cluster_round", None)
    _register_round(
        adata,
        round_id=str(new_round_id),
        parent_round_id=str(parent_round_id),
        cluster_key=cluster_key_use,
        labels_obs_key=labels_obs_key_use,
        kind=str(kind),
        best_resolution=parent.get("best_resolution", None) if best_resolution is None else best_resolution,
        sweep=parent.get("sweep", None) if sweep is None else sweep,
        cfg_snapshot=parent.get("cfg", None) if cfg_snapshot is None else cfg_snapshot,
        notes=notes,
        cluster_id_map=parent.get("cluster_id_map", None) if cluster_id_map is None else cluster_id_map,
        cluster_renumbering=(
            parent.get("cluster_renumbering", None)
            if cluster_renumbering is None
            else cluster_renumbering
        ),
        compacting=parent.get("compacting", None) if compacting is None else compacting,
        cache_labels=False,
    )

    rounds = adata.uns.get("cluster_rounds", {})
    if isinstance(rounds, dict) and new_round_id in rounds and isinstance(rounds[new_round_id], dict):
        rnew = rounds[new_round_id]
        for field in inherit_fields:
            if field in parent:
                try:
                    value = parent[field]
                    if isinstance(value, dict):
                        rnew[field] = dict(value)
                    elif isinstance(value, list):
                        rnew[field] = list(value)
                    else:
                        rnew[field] = value
                except Exception:
                    pass
        rnew["round_type"] = str(round_type)
        rounds[new_round_id] = rnew
        adata.uns["cluster_rounds"] = rounds

    if set_active:
        set_active_round(adata, str(new_round_id), publish_decoupler=False)
    else:
        adata.uns["active_cluster_round"] = prev_active_round

    return str(new_round_id)


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


def rebuild_round_from_label_parts(
    adata: ad.AnnData,
    *,
    round_id: str,
    label_parts: pd.Series,
    round_type: str | None = None,
    metadata_key: str | None = None,
    metadata_value: dict[str, Any] | None = None,
    annotation_updates: dict[str, Any] | None = None,
    set_active: bool = True,
) -> None:
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"Round {round_id!r} not found in adata.uns['cluster_rounds'].")

    rinfo = rounds[round_id]
    cluster_key = str(rinfo.get("cluster_key", "leiden"))
    labels_obs_key = str(rinfo.get("labels_obs_key", f"{cluster_key}__{round_id}"))
    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"

    parts = label_parts.reindex(adata.obs_names).fillna("Unknown").astype(str)
    label_order = _cluster_order_by_size(parts)
    raw_cluster_order = [str(i) for i in range(len(label_order))]
    label_to_raw = {label: raw_id for raw_id, label in zip(raw_cluster_order, label_order)}
    raw_to_ccode = {raw_id: f"C{i:02d}" for i, raw_id in enumerate(raw_cluster_order)}

    raw_labels = parts.map(label_to_raw)
    pretty_labels = raw_labels.map(
        lambda raw_id: f"{raw_to_ccode[str(raw_id)]}: {label_order[int(str(raw_id))]}"
    )
    display_map = {
        raw_id: f"{raw_to_ccode[raw_id]}: {label_order[int(raw_id)]}"
        for raw_id in raw_cluster_order
    }
    cluster_sizes = {
        raw_id: int((raw_labels == raw_id).sum())
        for raw_id in raw_cluster_order
    }

    adata.obs[labels_obs_key] = pd.Categorical(raw_labels.astype(str), categories=raw_cluster_order)
    adata.obs[pretty_key] = pd.Categorical(
        pretty_labels.astype(str),
        categories=[display_map[raw_id] for raw_id in raw_cluster_order],
    )
    if set_active:
        adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]

    try:
        from scanpy.plotting.palettes import default_102

        colors = list(default_102[: len(raw_cluster_order)])
        adata.uns[f"{pretty_key}_colors"] = colors
        if set_active:
            adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = colors
    except Exception as e:
        LOGGER.warning("Could not set pretty-label palette for round rebuild: %s", e)

    rinfo["labels_obs_key"] = labels_obs_key
    if round_type is not None:
        rinfo["round_type"] = str(round_type)
    rinfo["cluster_sizes"] = cluster_sizes
    rinfo["cluster_order"] = list(raw_cluster_order)
    rinfo["cluster_display_map"] = dict(display_map)
    rinfo["cluster_id_map"] = {raw_id: raw_id for raw_id in raw_cluster_order}
    rinfo["cluster_renumbering"] = {raw_id: raw_id for raw_id in raw_cluster_order}
    rinfo["decoupler"] = {}
    rinfo["qc"] = {}
    rinfo["stability"] = {}
    rinfo["diagnostics"] = {}
    rinfo.setdefault("annotation", {})
    rinfo["annotation"]["pretty_cluster_key"] = pretty_key
    rinfo["annotation"]["cluster_key_used"] = labels_obs_key
    if isinstance(annotation_updates, dict):
        rinfo["annotation"].update(annotation_updates)
    if metadata_key is not None and metadata_value is not None:
        rinfo[str(metadata_key)] = dict(metadata_value)
    rounds[round_id] = rinfo
    adata.uns["cluster_rounds"] = rounds
    if set_active:
        set_active_round(adata, round_id, publish_decoupler=False)


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
            for layer in ("counts_cb", "counts_raw"):
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
            bio_mask, bio_mask_stats = ct_utils.build_entropy_margin_mask(
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

    mask, mstats = ct_utils.build_entropy_margin_mask(
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
    # Archive-only carrier for legacy benchmark reconstruction; the selector ignores it.
    penalized: Optional[Dict[float, float]] = None
    bio_homogeneity: Optional[Dict[float, float]] = None
    bio_fragmentation: Optional[Dict[float, float]] = None
    bio_ari: Optional[Dict[float, float]] = None
    n_bio_labels: Optional[int] = None


@dataclass
class ResolutionSelectionConfig:
    stability_threshold: float = 0.85
    min_plateau_len: int = 3
    min_cluster_size: int = 20
    tiny_cluster_size: int = 20
    w_stab: float = 0.50
    w_sil: float = 0.35
    w_tiny: float = 0.15
    w_hom: float = 0.0
    w_frag: float = 0.0
    w_bioari: float = 0.0
    use_bio: bool = False


_BISC_MIN_FEASIBLE_STABILITY = 0.60
_BISC_PARSIMONY_TOLERANCE = 0.03
_BISC_MAX_CLUSTERS_PER_BIOLOGICAL_LABEL = 2.5
_BISC_ABSOLUTE_MINIMUM_CLUSTER_SIZE = 5
_BISC_PLATEAU_SUPPORT_FRACTION = 0.50
_BISC_SELECTOR_VERSION = "raw_edge_persistence_v3"


def _bisc_fixed_rule_snapshot() -> Dict[str, float | int | str]:
    return {
        "selector_version": _BISC_SELECTOR_VERSION,
        "minimum_feasible_stability": _BISC_MIN_FEASIBLE_STABILITY,
        "plateau_support_fraction": _BISC_PLATEAU_SUPPORT_FRACTION,
        "parsimony_tolerance": _BISC_PARSIMONY_TOLERANCE,
        "max_clusters_per_biological_label": _BISC_MAX_CLUSTERS_PER_BIOLOGICAL_LABEL,
        "absolute_minimum_cluster_size": _BISC_ABSOLUTE_MINIMUM_CLUSTER_SIZE,
    }


@dataclass
class Plateau:
    resolutions: List[float]
    mean_stability: float
    internal_floor: Optional[float] = None
    internal_peak: Optional[float] = None
    boundary_level: Optional[float] = None
    prominence: Optional[float] = None
    representative_resolution: Optional[float] = None
    representative_score: Optional[float] = None
    reproducibility_mean: Optional[float] = None
    reproducibility_min: Optional[float] = None
    internal_edge_persistence_mean: Optional[float] = None
    internal_edge_persistence_min: Optional[float] = None
    boundary_persistence_mean: Optional[float] = None
    boundary_persistence_min: Optional[float] = None
    persistence_score: Optional[float] = None
    selected: bool = False


@dataclass
class ResolutionSelectionResult:
    best_resolution: float
    scores: Dict[float, float]
    stability: Dict[float, float]
    tiny_cluster_penalty: Dict[float, float]
    plateaus: List[Plateau]
    structural_scores: Dict[float, float]
    selected_plateau_index: Optional[int]
    alternative_plateau_index: Optional[int]
    selection_mode: str
    confidence: str
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


def _plateau_support_level(config: ResolutionSelectionConfig) -> float:
    support = _BISC_MIN_FEASIBLE_STABILITY + _BISC_PLATEAU_SUPPORT_FRACTION * (
        float(config.stability_threshold) - _BISC_MIN_FEASIBLE_STABILITY
    )
    return min(float(config.stability_threshold), float(support))


def _detect_plateau_intervals(
    edges: Sequence[float],
    config: ResolutionSelectionConfig,
) -> List[Tuple[int, int]]:
    """Return disjoint rescued edge intervals around strong-edge cores."""
    edge_values = [float(value) for value in edges]
    minimum_edges = max(1, int(config.min_plateau_len) - 1)
    support_level = _plateau_support_level(config)

    cores: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index in range(len(edge_values) + 1):
        strong = (
            index < len(edge_values)
            and edge_values[index] >= float(config.stability_threshold)
        )
        if strong and start is None:
            start = index
        if start is None or strong:
            continue
        cores.append((start, index - 1))
        start = None

    intervals: List[Tuple[int, int]] = []
    for core_start, core_end in cores:
        start, end = core_start, core_end
        while end - start + 1 < minimum_edges:
            neighbours: List[Tuple[float, int]] = []
            if start > 0 and edge_values[start - 1] >= support_level:
                neighbours.append((edge_values[start - 1], start - 1))
            if end + 1 < len(edge_values) and edge_values[end + 1] >= support_level:
                neighbours.append((edge_values[end + 1], end + 1))
            if not neighbours:
                break
            _, chosen = max(neighbours, key=lambda item: (item[0], -item[1]))
            if chosen < start:
                start = chosen
            else:
                end = chosen
        if end - start + 1 >= minimum_edges:
            intervals.append((start, end))

    merged: List[Tuple[int, int]] = []
    for start, end in sorted(set(intervals)):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _detect_plateaus(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
    stability: Dict[float, float],
) -> List[Plateau]:
    del stability
    sorted_res = sorted(metrics.resolutions)
    if len(sorted_res) < 2:
        return []
    ari_adjacent = metrics.ari_adjacent or _compute_ari_adjacent(
        sorted_res, metrics.labels_per_resolution
    )
    edges = [
        float(ari_adjacent[(left, right)])
        for left, right in zip(sorted_res[:-1], sorted_res[1:])
    ]
    intervals = _detect_plateau_intervals(edges, config)

    plateaus: List[Plateau] = []
    for start, end in intervals:
        internal = np.asarray(edges[start : end + 1], dtype=float)
        boundaries: List[float] = []
        if start > 0:
            boundaries.append(edges[start - 1])
        if end + 1 < len(edges):
            boundaries.append(edges[end + 1])
        boundary = max(boundaries) if boundaries else _BISC_MIN_FEASIBLE_STABILITY
        floor = float(np.min(internal))
        plateaus.append(
            Plateau(
                resolutions=[float(r) for r in sorted_res[start : end + 2]],
                mean_stability=float(np.mean(internal)),
                internal_floor=floor,
                internal_peak=float(np.max(internal)),
                boundary_level=float(boundary),
                prominence=float(floor - boundary),
            )
        )
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


def _annotate_plateau_persistence(
    plateaus: Sequence[Plateau],
    sorted_resolutions: Sequence[float],
    config: ResolutionSelectionConfig,
    plateau_reproducibility: Dict[float, Sequence[float]],
    adjacent_reproducibility: Dict[Tuple[float, float], Sequence[float]],
) -> None:
    """Attach fixed-partition, internal-edge, and boundary persistence metrics."""
    sorted_res = [float(value) for value in sorted_resolutions]
    edge_keys = list(zip(sorted_res[:-1], sorted_res[1:]))
    missing_edges = [key for key in edge_keys if key not in adjacent_reproducibility]
    if missing_edges:
        raise ValueError(
            "Missing subsampling results for adjacent BISC edges: "
            f"{missing_edges}"
        )
    repeat_counts = {len(adjacent_reproducibility[key]) for key in edge_keys}
    if not repeat_counts or len(repeat_counts) != 1 or next(iter(repeat_counts)) == 0:
        raise ValueError("Adjacent-edge subsampling results must have equal nonzero lengths")
    n_repeats = next(iter(repeat_counts))
    support_level = _plateau_support_level(config)

    for plateau in plateaus:
        probe = float(plateau.representative_resolution)
        probe_values = [
            float(value) for value in plateau_reproducibility.get(probe, [])
        ]
        if len(probe_values) != n_repeats:
            raise ValueError(
                f"Expected {n_repeats} fixed-resolution subsampling results for "
                f"plateau probe {probe:.3f}; found {len(probe_values)}"
            )
        plateau.reproducibility_mean = float(np.mean(probe_values))
        plateau.reproducibility_min = float(np.min(probe_values))

        start = sorted_res.index(float(plateau.resolutions[0]))
        end = sorted_res.index(float(plateau.resolutions[-1]))
        internal_indices = list(range(start, end))

        internal_by_repeat: List[float] = []
        boundary_by_repeat: List[float] = []
        for repeat in range(n_repeats):
            internal_values = [
                float(adjacent_reproducibility[edge_keys[index]][repeat])
                for index in internal_indices
            ]
            internal_by_repeat.append(
                float(all(value >= support_level for value in internal_values))
            )
            boundary_checks: List[bool] = []
            if start > 0:
                boundary_checks.append(
                    float(adjacent_reproducibility[edge_keys[start - 1]][repeat])
                    < internal_values[0]
                )
            if end < len(sorted_res) - 1:
                boundary_checks.append(
                    float(adjacent_reproducibility[edge_keys[end]][repeat])
                    < internal_values[-1]
                )
            boundary_by_repeat.append(
                float(np.mean(boundary_checks)) if boundary_checks else 1.0
            )

        plateau.internal_edge_persistence_mean = float(np.mean(internal_by_repeat))
        plateau.internal_edge_persistence_min = float(np.min(internal_by_repeat))
        plateau.boundary_persistence_mean = float(np.mean(boundary_by_repeat))
        plateau.boundary_persistence_min = float(np.min(boundary_by_repeat))
        plateau.persistence_score = float(
            min(
                plateau.reproducibility_mean,
                plateau.internal_edge_persistence_mean,
                plateau.boundary_persistence_mean,
            )
        )


def select_best_resolution(
    metrics: ResolutionMetrics,
    config: ResolutionSelectionConfig,
    plateau_reproducibility: Optional[Dict[float, Sequence[float]]] = None,
    adjacent_reproducibility: Optional[
        Dict[Tuple[float, float], Sequence[float]]
    ] = None,
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

    structural_scores = {
        float(r): float(
            config.w_stab * stab_norm.get(r, 0.0)
            + config.w_sil * sil_norm.get(r, 0.0)
            + config.w_tiny * tiny_norm.get(r, 0.0)
        )
        for r in metrics.resolutions
    }

    def composite(r: float) -> float:
        s = structural_scores[r]
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
    if not interior:
        raise ValueError(
            "BISC resolution selection requires at least one interior candidate; "
            "provide at least three tested resolutions."
        )

    size_safe = [
        r
        for r in interior
        if (
            metrics.cluster_sizes[r].size == 0
            or int(np.min(metrics.cluster_sizes[r]))
            >= _BISC_ABSOLUTE_MINIMUM_CLUSTER_SIZE
        )
        and (
            metrics.cluster_sizes[r].size > 0
            and float(np.median(metrics.cluster_sizes[r])) >= config.min_cluster_size
        )
    ]
    if not size_safe:
        raise ValueError(
            "No interior BISC resolution satisfies the minimum cluster-size safeguards"
        )
    structurally_feasible = [
        r
        for r in size_safe
        if stability.get(r, 0.0) >= _BISC_MIN_FEASIBLE_STABILITY
    ]
    structural_candidates = structurally_feasible if structurally_feasible else size_safe

    def apply_biological_cluster_limit(candidates: Sequence[float]) -> List[float]:
        retained = [float(r) for r in candidates]
        if not (use_bio and metrics.n_bio_labels):
            return retained
        max_clusters = _BISC_MAX_CLUSTERS_PER_BIOLOGICAL_LABEL * metrics.n_bio_labels
        capped = [
            r for r in retained if metrics.cluster_counts.get(r, 0) <= max_clusters
        ]
        return capped if capped else retained

    def pick_parsimonious(cands, eps=_BISC_PARSIMONY_TOLERANCE):
        if not cands:
            return None
        best = max(cands, key=lambda r: all_scores.get(r, -np.inf))
        best_val = all_scores[best]
        near = [r for r in cands if all_scores.get(r, -np.inf) >= (1 - eps) * best_val]
        return (
            min(near, key=lambda r: (metrics.cluster_counts[r], r))
            if near
            else best
        )

    plateaus = _detect_plateaus(metrics, config, stability)
    search_set = set(structural_candidates)
    feasible_plateaus = [
        (
            plateau,
            [float(r) for r in plateau.resolutions if float(r) in search_set],
        )
        for plateau in plateaus
    ]
    feasible_plateaus = [item for item in feasible_plateaus if item[1]]

    if feasible_plateaus:
        global_stability_values = np.asarray(
            [stability[r] for r in sorted_res], dtype=float
        )
        lower = float(np.min(global_stability_values))
        span = float(np.max(global_stability_values) - lower)
        global_norm = _normalize_scores(stability)
        ari_adjacent = metrics.ari_adjacent or _compute_ari_adjacent(
            sorted_res, metrics.labels_per_resolution
        )

        for plateau, candidates in feasible_plateaus:
            local_scores: Dict[float, float] = {}
            plateau_set = set(plateau.resolutions)
            for resolution in candidates:
                index = sorted_res.index(resolution)
                terms: List[float] = []
                if index > 0 and sorted_res[index - 1] in plateau_set:
                    terms.append(ari_adjacent[(sorted_res[index - 1], resolution)])
                if index + 1 < len(sorted_res) and sorted_res[index + 1] in plateau_set:
                    terms.append(ari_adjacent[(resolution, sorted_res[index + 1])])
                local_stability = float(np.mean(terms)) if terms else stability[resolution]
                local_norm = (
                    0.0
                    if np.isclose(span, 0.0)
                    else float(np.clip((local_stability - lower) / span, 0.0, 1.0))
                )
                structural_score = structural_scores[resolution]
                local_scores[resolution] = (
                    float(
                        structural_score
                        - config.w_stab * global_norm[resolution]
                        + config.w_stab * local_norm
                    )
                    if np.isfinite(structural_score)
                    else local_norm
                )
            best_probe_score = max(local_scores.values())
            exact_best = [
                resolution
                for resolution, score in local_scores.items()
                if np.isclose(score, best_probe_score, rtol=1e-12, atol=1e-12)
            ]
            probe = min(
                exact_best,
                key=lambda r: (metrics.cluster_counts[r], r),
            )
            plateau.representative_resolution = float(probe)
            plateau.representative_score = float(local_scores[probe])

        if plateau_reproducibility is not None:
            for plateau, _ in feasible_plateaus:
                probe = float(plateau.representative_resolution)
                values = [float(value) for value in plateau_reproducibility.get(probe, [])]
                if not values:
                    raise ValueError(
                        f"Missing fixed-resolution subsampling results for plateau probe {probe:.3f}"
                    )
                plateau.reproducibility_mean = float(np.mean(values))
                plateau.reproducibility_min = float(np.min(values))
            if adjacent_reproducibility is not None:
                _annotate_plateau_persistence(
                    [plateau for plateau, _ in feasible_plateaus],
                    sorted_res,
                    config,
                    plateau_reproducibility,
                    adjacent_reproducibility,
                )
            best_persistence = max(
                float(
                    plateau.persistence_score
                    if plateau.persistence_score is not None
                    else plateau.reproducibility_mean
                )
                for plateau, _ in feasible_plateaus
            )
            best_plateaus = [
                item
                for item in feasible_plateaus
                if np.isclose(
                    float(
                        item[0].persistence_score
                        if item[0].persistence_score is not None
                        else item[0].reproducibility_mean
                    ),
                    best_persistence,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ]
            selected_plateau, plateau_res = min(
                best_plateaus,
                key=lambda item: (
                    metrics.cluster_counts[float(item[0].representative_resolution)],
                    float(item[0].representative_resolution),
                ),
            )
            selection_mode = (
                "plateau_persistence_subsampling"
                if adjacent_reproducibility is not None
                else "plateau_probe_subsampling"
            )
        else:
            selected_plateau, plateau_res = max(
                feasible_plateaus,
                key=lambda item: (item[0].mean_stability, len(item[0].resolutions)),
            )
            selection_mode = "structural_preselection"
        selected_plateau.selected = True
        alternatives = [
            item for item in feasible_plateaus if item[0] is not selected_plateau
        ]
        if plateau_reproducibility is not None:
            alternative = (
                max(
                    alternatives,
                    key=lambda item: (
                        float(
                            item[0].persistence_score
                            if item[0].persistence_score is not None
                            else item[0].reproducibility_mean
                        ),
                        -metrics.cluster_counts[
                            float(item[0].representative_resolution)
                        ],
                        -float(item[0].representative_resolution),
                    ),
                )
                if alternatives
                else None
            )
        else:
            alternative = (
                max(
                    alternatives,
                    key=lambda item: (
                        item[0].mean_stability,
                        len(item[0].resolutions),
                    ),
                )
                if alternatives
                else None
            )
        final_candidates = apply_biological_cluster_limit(plateau_res)
        best = pick_parsimonious(final_candidates)
        confidence = "multiscale" if alternatives else "clear"
        if (
            selected_plateau.persistence_score is not None
            and selected_plateau.persistence_score < _BISC_MIN_FEASIBLE_STABILITY
        ):
            confidence = "unstable"
        return ResolutionSelectionResult(
            best_resolution=float(best),
            scores=all_scores,
            stability=stability,
            tiny_cluster_penalty=tiny_penalty,
            plateaus=plateaus,
            structural_scores=structural_scores,
            selected_plateau_index=plateaus.index(selected_plateau),
            alternative_plateau_index=(
                plateaus.index(alternative[0]) if alternative is not None else None
            ),
            selection_mode=selection_mode,
            confidence=confidence,
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

    fallback_candidates = apply_biological_cluster_limit(structural_candidates)
    knee = stability_knee(fallback_candidates)
    best = knee if knee is not None else pick_parsimonious(fallback_candidates)

    return ResolutionSelectionResult(
        best_resolution=float(best),
        scores=all_scores,
        stability=stability,
        tiny_cluster_penalty=tiny_penalty,
        plateaus=plateaus,
        structural_scores=structural_scores,
        selected_plateau_index=None,
        alternative_plateau_index=None,
        selection_mode="stability_knee_fallback" if knee is not None else "composite_fallback",
        confidence="weak",
        bio_homogeneity=metrics.bio_homogeneity,
        bio_fragmentation=metrics.bio_fragmentation,
        bio_ari=metrics.bio_ari,
    )


def _subsampling_resolution_stability(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    labels_per_resolution: Dict[float, np.ndarray],
    candidate_resolutions: Sequence[float],
) -> Tuple[
    Dict[float, List[float]],
    Dict[Tuple[float, float], List[float]],
]:
    candidates = sorted({float(r) for r in candidate_resolutions})
    if not candidates:
        return {}, {}
    missing = [r for r in candidates if r not in labels_per_resolution]
    if missing:
        raise ValueError(f"Missing full-data labels for BISC probes: {missing}")

    embedding = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    n_sub = int(round(cfg.subsample_frac * adata.n_obs))
    if not 2 <= n_sub <= adata.n_obs:
        raise ValueError(
            "BISC subsampling requires at least two retained cells"
        )
    results: Dict[float, List[float]] = {r: [] for r in candidates}
    edge_results: Dict[Tuple[float, float], List[float]] = {
        edge: [] for edge in zip(candidates[:-1], candidates[1:])
    }
    LOGGER.info(
        "Evaluating %d BISC resolutions and their boundaries across %d %.0f%% cell subsamples",
        len(candidates),
        cfg.stability_repeats,
        100.0 * cfg.subsample_frac,
    )
    for repeat in range(cfg.stability_repeats):
        seed = cfg.random_state + repeat
        rng = np.random.default_rng(seed)
        positions = rng.choice(adata.n_obs, size=n_sub, replace=False)
        sub = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names[positions].copy()))
        sub.obsm[embedding_key] = embedding[positions]
        sc.pp.neighbors(sub, use_rep=embedding_key, random_state=seed)
        repeat_labels: Dict[float, np.ndarray] = {}
        for candidate_index, resolution in enumerate(candidates):
            key = f"{cfg.label_key}_probe_{repeat}_{candidate_index}"
            sc.tl.leiden(
                sub,
                resolution=resolution,
                key_added=key,
                random_state=seed,
                flavor="igraph",
            )
            labels = sub.obs[key].to_numpy()
            repeat_labels[resolution] = labels
            ari = adjusted_rand_score(
                labels_per_resolution[resolution][positions],
                labels,
            )
            results[resolution].append(float(ari))
        for left, right in zip(candidates[:-1], candidates[1:]):
            edge_results[(left, right)].append(
                float(adjusted_rand_score(repeat_labels[left], repeat_labels[right]))
            )
    return results, edge_results


def _subsampling_candidate_stability(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    labels_per_resolution: Dict[float, np.ndarray],
    candidate_resolutions: Sequence[float],
) -> Dict[float, List[float]]:
    """Compatibility wrapper returning fixed-resolution reproducibility only."""
    candidate_results, _ = _subsampling_resolution_stability(
        adata,
        cfg,
        embedding_key,
        labels_per_resolution,
        candidate_resolutions,
    )
    return candidate_results


# -------------------------------------------------------------------------
# Resolution sweep and adjacent-resolution stability
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
        silhouette_scores.append(sil)

        # -----------------------------
        # One-line quantitative summary
        # -----------------------------
        LOGGER.info(
            "  → %d clusters | centroid separation=%.3f | min/med/max size=%d/%d/%d",
            n_clusters,
            sil,
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

    preselection = select_best_resolution(metrics, sel_cfg)
    candidate_subsampling_ari, edge_subsampling_ari = (
        _subsampling_resolution_stability(
            adata,
            cfg,
            embedding_key,
            clusterings_float,
            res_list,
        )
    )
    probe_resolutions = {
        float(plateau.representative_resolution)
        for plateau in preselection.plateaus
        if plateau.representative_resolution is not None
    }
    plateau_probe_stability = {
        resolution: candidate_subsampling_ari[resolution]
        for resolution in probe_resolutions
    }
    selection = select_best_resolution(
        metrics,
        sel_cfg,
        plateau_reproducibility=(
            plateau_probe_stability if plateau_probe_stability else None
        ),
        adjacent_reproducibility=(
            edge_subsampling_ari if plateau_probe_stability else None
        ),
    )
    best_res = float(selection.best_resolution)

    ari_adjacent = metrics.ari_adjacent or _compute_ari_adjacent(
        res_list, clusterings_float
    )

    selected_plateau = (
        selection.plateaus[selection.selected_plateau_index]
        if selection.selected_plateau_index is not None
        else None
    )
    alternative_plateau = (
        selection.plateaus[selection.alternative_plateau_index]
        if selection.alternative_plateau_index is not None
        else None
    )
    selected_probe = (
        float(selected_plateau.representative_resolution)
        if selected_plateau is not None
        else None
    )
    alternative_probe = (
        float(alternative_plateau.representative_resolution)
        if alternative_plateau is not None
        else None
    )
    support_level = _plateau_support_level(sel_cfg)
    edge_persistence = []
    for left, right in zip(res_list[:-1], res_list[1:]):
        full_ari = float(ari_adjacent[(left, right)])
        values = [float(value) for value in edge_subsampling_ari[(left, right)]]
        if full_ari >= float(sel_cfg.stability_threshold):
            full_state = "strong"
        elif full_ari >= support_level:
            full_state = "support"
        else:
            full_state = "separator"
        edge_persistence.append(
            {
                "left_resolution": float(left),
                "right_resolution": float(right),
                "full_data_ari": full_ari,
                "full_data_state": full_state,
                "strong_probability": float(
                    np.mean(
                        np.asarray(values, dtype=float)
                        >= float(sel_cfg.stability_threshold)
                    )
                ),
                "support_probability": float(
                    np.mean(np.asarray(values, dtype=float) >= support_level)
                ),
                "state_retention_probability": float(
                    np.mean(np.asarray(values, dtype=float) >= support_level)
                    if full_state != "separator"
                    else np.mean(np.asarray(values, dtype=float) < support_level)
                ),
            }
        )
    sweep: Dict[str, object] = {
        "resolutions": np.array(res_list, dtype=float),
        "silhouette_scores": silhouette_scores,
        "n_clusters": n_clusters_list,
        "composite_scores": [selection.scores[r] for r in res_list],
        "structural_scores": [selection.structural_scores[r] for r in res_list],
        "stability_scores": [selection.stability[r] for r in res_list],
        "adjacent_ari": [
            float(ari_adjacent[(left, right)])
            for left, right in zip(res_list[:-1], res_list[1:])
        ],
        "tiny_cluster_penalty": [selection.tiny_cluster_penalty[r] for r in res_list],
        "cluster_sizes": cluster_sizes,
        "plateaus": [asdict(plateau) for plateau in selection.plateaus],
        "selection": {
            "mode": selection.selection_mode,
            "confidence": selection.confidence,
            "selected_plateau_index": selection.selected_plateau_index,
            "alternative_plateau_index": selection.alternative_plateau_index,
            "best_resolution": best_res,
            "best_n_clusters": int(metrics.cluster_counts[best_res]),
            "final_score": float(selection.scores[best_res]),
            "selected_probe_resolution": selected_probe,
            "selected_probe_n_clusters": (
                int(metrics.cluster_counts[selected_probe])
                if selected_probe is not None
                else None
            ),
            "alternative_probe_resolution": alternative_probe,
            "alternative_probe_n_clusters": (
                int(metrics.cluster_counts[alternative_probe])
                if alternative_probe is not None
                else None
            ),
            "probe_reproducibility_gap": (
                float(selected_plateau.reproducibility_mean)
                - float(alternative_plateau.reproducibility_mean)
                if selected_plateau is not None
                and alternative_plateau is not None
                and selected_plateau.reproducibility_mean is not None
                and alternative_plateau.reproducibility_mean is not None
                else None
            ),
            "plateau_persistence_gap": (
                float(selected_plateau.persistence_score)
                - float(alternative_plateau.persistence_score)
                if selected_plateau is not None
                and alternative_plateau is not None
                and selected_plateau.persistence_score is not None
                and alternative_plateau.persistence_score is not None
                else None
            ),
            "probe_cluster_count_gap": (
                int(metrics.cluster_counts[alternative_probe])
                - int(metrics.cluster_counts[selected_probe])
                if selected_probe is not None and alternative_probe is not None
                else None
            ),
        },
        "plateau_probe_subsampling_ari": {
            _res_key(resolution): [float(value) for value in values]
            for resolution, values in plateau_probe_stability.items()
        },
        "resolution_subsampling_ari": {
            _res_key(resolution): [float(value) for value in values]
            for resolution, values in candidate_subsampling_ari.items()
        },
        "edge_subsampling_ari": {
            f"{_res_key(left)}|{_res_key(right)}": [
                float(value) for value in values
            ]
            for (left, right), values in edge_subsampling_ari.items()
        },
        "edge_persistence": edge_persistence,
        "selection_config": asdict(sel_cfg),
        "selection_rules": _bisc_fixed_rule_snapshot(),
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
    LOGGER.info(
        "Computing reference clustering for post-selection subsampling reproducibility "
        "at resolution %.3f",
        best_res,
    )
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
    make_figures: Optional[bool] = None,
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

    do_figures = cfg.make_figures if make_figures is None else bool(make_figures)
    if do_figures:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sil_values, bins=40, color="steelblue", alpha=0.85)
        ax.axvline(sil_mean, color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Silhouette value")
        ax.set_ylabel("Number of cells")
        title_key = f"{cluster_key}" + (f" [{round_id}]" if round_id else "")
        ax.set_title(f"Final clustering: true silhouette ({title_key}, mean={sil_mean:.3f})")
        fig.tight_layout()
        plot_utils.record_plot_artifact("final_real_silhouette", figdir, fig)

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
    make_figures: bool = True,
) -> ad.AnnData:
    _ensure_cluster_rounds(adata)

    if "neighbors" not in adata.uns:
        LOGGER.info("BISC: neighbors not found; computing neighbors using embedding_key=%r", embedding_key)
        sc.pp.neighbors(adata, use_rep=embedding_key)

    if make_figures and "X_umap" not in adata.obsm:
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

    resolution_stability = sweep.get("resolution_subsampling_ari", {})
    resolution_stability = (
        resolution_stability if isinstance(resolution_stability, dict) else {}
    )
    probe_stability = sweep.get("plateau_probe_subsampling_ari", {})
    probe_stability = probe_stability if isinstance(probe_stability, dict) else {}
    retained_final_stability = resolution_stability.get(_res_key(best_res), [])
    if retained_final_stability:
        stability_aris = [float(value) for value in retained_final_stability]
        LOGGER.info(
            "Reusing resolution-sweep subsampling results for final resolution %.3f",
            best_res,
        )
    else:
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
            "adjacent_ari": sweep.get("adjacent_ari", None),
            "plateaus": sweep.get("plateaus", None),
            "selection": sweep.get("selection", {}),
            "plateau_probe_subsampling_ari": sweep.get(
                "plateau_probe_subsampling_ari", {}
            ),
            "resolution_subsampling_ari": sweep.get(
                "resolution_subsampling_ari", {}
            ),
            "edge_subsampling_ari": sweep.get("edge_subsampling_ari", {}),
            "edge_persistence": sweep.get("edge_persistence", []),
            "selection_config": sweep.get("selection_config", {}),
            "selection_rules": sweep.get("selection_rules", {}),
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
    rinfo["stability"]["subsampling_role"] = "final_partition_diagnostic"
    rinfo["stability"]["plateau_probe_subsampling_ari"] = probe_stability
    rinfo["stability"]["plateau_probe_selection_role"] = "cross_plateau_selection"

    res_list = [float(r) for r in rinfo.get("sweep", {}).get("resolutions", [])]
    sil_list = rinfo.get("sweep", {}).get("silhouette_scores", []) or []
    n_list = rinfo.get("sweep", {}).get("n_clusters", []) or []

    rinfo.setdefault("diagnostics", {})
    rinfo["diagnostics"]["tested_resolutions"] = res_list
    rinfo["diagnostics"]["silhouette_centroid"] = {_res_key(r): float(s) for r, s in zip(res_list, sil_list)}
    rinfo["diagnostics"]["cluster_counts"] = {_res_key(r): int(n) for r, n in zip(res_list, n_list)}

    comp_list = sweep.get("composite_scores", None)
    structural_list = sweep.get("structural_scores", None)
    stab_list = sweep.get("stability_scores", None)
    tiny_list = sweep.get("tiny_cluster_penalty", None)

    if isinstance(comp_list, list) and len(comp_list) == len(res_list):
        rinfo["diagnostics"]["composite_scores"] = {_res_key(r): float(v) for r, v in zip(res_list, comp_list)}
    if isinstance(structural_list, list) and len(structural_list) == len(res_list):
        rinfo["diagnostics"]["structural_scores"] = {
            _res_key(r): float(v) for r, v in zip(res_list, structural_list)
        }
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
        make_figures=make_figures,
    )

    LOGGER.info(
        "BISC complete: best_res=%.3f stored as round '%s' using cluster_key='%s'",
        float(best_res),
        round_id,
        cfg.label_key,
    )
    return adata


def create_manual_rename_round(
    adata: ad.AnnData,
    *,
    mapping: dict[str, str],
    parent_round_id: str | None = None,
    new_round_id: str | None = None,
    round_name: str = "manual_rename",
    notes: str | None = None,
    collapse_same_labels: bool = False,
    update_existing_round: bool = False,
    set_active: bool = True,
) -> str:
    _ensure_cluster_rounds(adata)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a non-empty dict of Cnn -> new label strings.")

    mapping_norm: dict[str, str] = {}
    for k, v in mapping.items():
        if k is None or v is None:
            raise ValueError("mapping contains None keys or values.")
        kk = str(k).strip()
        vv = str(v).strip()
        if not kk or not vv:
            raise ValueError("mapping contains empty keys or values.")
        mapping_norm[kk] = vv

    import re
    bad_keys = [k for k in mapping_norm.keys() if re.fullmatch(r"C\d+", k) is None]
    if bad_keys:
        raise ValueError(f"mapping keys must be strict 'Cnn' format (e.g., C03). Bad keys: {bad_keys}")

    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict):
        raise KeyError("adata.uns['cluster_rounds'] is missing or invalid.")

    existing_round = None
    if update_existing_round:
        if new_round_id is None:
            raise ValueError("update_existing_round=True requires new_round_id.")
        existing_round = rounds.get(str(new_round_id), None)
        if not isinstance(existing_round, dict):
            raise KeyError(f"Target round {new_round_id!r} not found in adata.uns['cluster_rounds'].")
        if str(existing_round.get("round_type", "")) != "manual_rename":
            raise ValueError(f"target round {new_round_id!r} is not a manual_rename round.")
        stored_parent_round_id = None
        manual_rename_payload = existing_round.get("manual_rename", None)
        if isinstance(manual_rename_payload, dict):
            stored_parent_round_id = manual_rename_payload.get("parent_round_id", None)
        if stored_parent_round_id is None:
            stored_parent_round_id = existing_round.get("parent_round_id", None)
        if parent_round_id is None and stored_parent_round_id is not None:
            parent_round_id = str(stored_parent_round_id)

    if parent_round_id is None:
        rid0 = adata.uns.get("active_cluster_round", None)
        parent_round_id = str(rid0) if rid0 else None
    if parent_round_id is None:
        raise KeyError("No parent_round_id provided and adata.uns['active_cluster_round'] is None.")
    if parent_round_id not in rounds:
        raise KeyError(f"Parent round {parent_round_id!r} not found in adata.uns['cluster_rounds'].")

    parent = rounds[parent_round_id]
    labels_obs_key = parent.get("labels_obs_key", None)
    if not labels_obs_key or str(labels_obs_key) not in adata.obs:
        raise KeyError("Parent round missing labels_obs_key or it is not present in adata.obs.")
    labels_obs_key = str(labels_obs_key)

    parent_pretty_key = None
    ann = parent.get("annotation", {}) if isinstance(parent.get("annotation", {}), dict) else {}
    if isinstance(ann, dict):
        parent_pretty_key = ann.get("pretty_cluster_key", None)
    if parent_pretty_key and str(parent_pretty_key) in adata.obs:
        parent_pretty_key = str(parent_pretty_key)
    else:
        fallback = f"{CLUSTER_LABEL_KEY}__{parent_round_id}"
        if fallback in adata.obs:
            parent_pretty_key = fallback
        elif CLUSTER_LABEL_KEY in adata.obs:
            parent_pretty_key = CLUSTER_LABEL_KEY
        else:
            parent_pretty_key = None

    clust_vals = adata.obs[labels_obs_key].astype(str)

    cluster_order = parent.get("cluster_order", None)
    if not isinstance(cluster_order, list) or not cluster_order:
        cluster_order = _cluster_order_by_size(clust_vals)
    cluster_order = [str(c) for c in cluster_order]

    ord_map = {c: f"C{i:02d}" for i, c in enumerate(cluster_order)}

    parent_label_part: dict[str, str] = {}
    if parent_pretty_key and parent_pretty_key in adata.obs:
        tmp = pd.DataFrame(
            {
                "cluster": clust_vals.to_numpy(),
                "pretty": adata.obs[parent_pretty_key].astype(str).to_numpy(),
            },
            index=adata.obs_names,
        )
        for c, g in tmp.groupby("cluster", sort=False):
            val = str(g["pretty"].iloc[0])
            if ": " in val:
                _, lbl = val.split(": ", 1)
                parent_label_part[str(c)] = lbl
            else:
                parent_label_part[str(c)] = val

    parent_display_labels: dict[str, str] = {}
    for c in cluster_order:
        ccode = ord_map.get(str(c), "C??")
        base = parent_label_part.get(str(c), "Unknown")
        parent_display_labels[str(c)] = f"{ccode}: {base}"

    display_label_by_ccode = {v.split(": ", 1)[0]: v for v in parent_display_labels.values() if ": " in v}
    missing_ccodes = [k for k in mapping_norm.keys() if k not in display_label_by_ccode]
    if missing_ccodes:
        raise ValueError(f"mapping keys not found in parent round: {missing_ccodes}")

    label_part_by_ccode = {cc: lab for cc, lab in mapping_norm.items()}

    def _pretty_for_cluster(c: str) -> str:
        ccode = ord_map.get(str(c), "C??")
        base = parent_label_part.get(str(c), "Unknown")
        if ccode in label_part_by_ccode:
            base = label_part_by_ccode[ccode]
        return f"{ccode}: {base}"

    if new_round_id is None:
        idx = _next_round_index(adata)
        new_round_id = _make_round_id(idx, round_name)
    elif not update_existing_round:
        new_round_id = str(new_round_id)

    pretty_key = f"{CLUSTER_LABEL_KEY}__{new_round_id}"
    if pretty_key in adata.obs and not update_existing_round:
        raise ValueError(f"pretty_key '{pretty_key}' already exists in adata.obs.")

    labels_obs_key_new = labels_obs_key
    identity_map = {str(c): str(c) for c in pd.unique(clust_vals.astype(str))}
    cluster_renumbering = {str(c): str(c) for c in sorted(set(identity_map.values()))}
    if collapse_same_labels:
        if update_existing_round and isinstance(existing_round, dict):
            labels_obs_key_existing = existing_round.get("labels_obs_key", None)
            if labels_obs_key_existing:
                labels_obs_key_new = str(labels_obs_key_existing)
            else:
                labels_obs_key_new = f"{parent.get('cluster_key', 'leiden')}__{new_round_id}"
        else:
            labels_obs_key_new = f"{parent.get('cluster_key', 'leiden')}__{new_round_id}"
        if labels_obs_key_new in adata.obs and not update_existing_round:
            raise ValueError(f"labels_obs_key '{labels_obs_key_new}' already exists in adata.obs.")
        adata.obs[labels_obs_key_new] = adata.obs[labels_obs_key].astype(str).astype("category")
        pretty_series = clust_vals.map(lambda c: _pretty_for_cluster(str(c)))
        label_parts = pretty_series.astype(str).map(
            lambda val: val.split(": ", 1)[1] if ": " in val else str(val)
        )
    elif not update_existing_round:
        pretty_series = clust_vals.map(lambda c: _pretty_for_cluster(str(c)))
        pretty_categories = [_pretty_for_cluster(str(c)) for c in cluster_order]
        adata.obs[pretty_key] = pd.Categorical(
            pretty_series.astype(str),
            categories=pretty_categories,
            ordered=False,
        )
        if set_active:
            adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]

        try:
            from scanpy.plotting.palettes import default_102
            cats_pretty = list(adata.obs[pretty_key].cat.categories)
            adata.uns[f"{pretty_key}_colors"] = list(default_102[: len(cats_pretty)])
            if set_active:
                adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = adata.uns[f"{pretty_key}_colors"]
        except Exception as e:
            LOGGER.warning("Could not set pretty-label palette for manual rename: %s", e)

    if update_existing_round:
        new_round_id = str(new_round_id)
    else:
        new_round_id = _create_shallow_round_from_parent(
            adata,
            parent_round_id=str(parent_round_id),
            round_name=round_name,
            new_round_id=new_round_id,
            round_type="manual_rename",
            kind="MANUAL_RENAME",
            notes=notes,
            set_active=False,
            cluster_key=str(parent.get("cluster_key", CLUSTER_LABEL_KEY)),
            labels_obs_key=labels_obs_key_new,
            best_resolution=None,
            sweep=None,
            cfg_snapshot=None,
            cluster_id_map=identity_map,
            cluster_renumbering=cluster_renumbering,
            compacting={},
            inherit_fields=(),
        )

    round_notes = notes
    if round_notes is None and isinstance(existing_round, dict):
        round_notes = existing_round.get("notes", None)

    rounds = adata.uns.get("cluster_rounds", {})
    if isinstance(rounds, dict) and new_round_id in rounds and isinstance(rounds[new_round_id], dict):
        rounds[new_round_id]["parent_round_id"] = str(parent_round_id)
        rounds[new_round_id]["round_type"] = "manual_rename"
        rounds[new_round_id]["kind"] = "MANUAL_RENAME"
        rounds[new_round_id]["cluster_key"] = str(parent.get("cluster_key", CLUSTER_LABEL_KEY))
        rounds[new_round_id]["labels_obs_key"] = str(labels_obs_key_new)
        rounds[new_round_id]["notes"] = round_notes
        rounds[new_round_id].setdefault("annotation", {})
        rounds[new_round_id]["annotation"].update(
            {
                "pretty_cluster_key": pretty_key,
                "cluster_key_used": str(labels_obs_key_new),
            }
        )
        if isinstance(ann, dict):
            for k in ("celltypist_cell_key", "celltypist_cluster_key"):
                if k in ann:
                    rounds[new_round_id]["annotation"][k] = ann[k]
        rounds[new_round_id]["manual_rename"] = {
            "parent_round_id": str(parent_round_id),
            "mapping": dict(mapping_norm),
            "collapse_same_labels": bool(collapse_same_labels),
        }
        adata.uns["cluster_rounds"] = rounds

    if collapse_same_labels:
        annotation_updates = {}
        if isinstance(ann, dict):
            for k in ("celltypist_cell_key", "celltypist_cluster_key"):
                if k in ann:
                    annotation_updates[k] = ann[k]
        rebuild_round_from_label_parts(
            adata,
            round_id=str(new_round_id),
            label_parts=label_parts,
            round_type="manual_rename",
            metadata_key="manual_rename",
            metadata_value={
                "parent_round_id": str(parent_round_id),
                "mapping": dict(mapping_norm),
                "collapse_same_labels": True,
            },
            annotation_updates=annotation_updates,
            set_active=set_active,
        )
    elif isinstance(rounds, dict) and new_round_id in rounds and isinstance(rounds[new_round_id], dict):
        pretty_series = clust_vals.map(lambda c: _pretty_for_cluster(str(c)))
        pretty_categories = [_pretty_for_cluster(str(c)) for c in cluster_order]
        adata.obs[pretty_key] = pd.Categorical(
            pretty_series.astype(str),
            categories=pretty_categories,
            ordered=False,
        )
        if set_active:
            adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]

        try:
            from scanpy.plotting.palettes import default_102

            cats_pretty = list(adata.obs[pretty_key].cat.categories)
            adata.uns[f"{pretty_key}_colors"] = list(default_102[: len(cats_pretty)])
            if set_active:
                adata.uns[f"{CLUSTER_LABEL_KEY}_colors"] = adata.uns[f"{pretty_key}_colors"]
        except Exception as e:
            LOGGER.warning("Could not set pretty-label palette for manual rename: %s", e)

        rounds[new_round_id]["cluster_order"] = list(map(str, cluster_order))
        rounds[new_round_id]["cluster_display_map"] = {
            str(c): _pretty_for_cluster(str(c)) for c in cluster_order
        }
        adata.uns["cluster_rounds"] = rounds

    if set_active and not collapse_same_labels:
        set_active_round(adata, str(new_round_id), publish_decoupler=False)

    return str(new_round_id)
