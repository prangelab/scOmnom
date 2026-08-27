from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .io_utils import get_celltypist_model

LOGGER = logging.getLogger(__name__)


def get_celltypist_outputs(
    adata: ad.AnnData,
    label_key: str,
    *,
    proba_key: str = "celltypist_proba",
    proba_cols_key: str = "celltypist_proba_columns",
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], dict]:
    labels = None
    proba = None

    if label_key in adata.obs:
        s = adata.obs[label_key]
        if s.shape[0] == adata.n_obs:
            labels = s.to_numpy()

    proba_arr = adata.obsm.get(proba_key, None)
    proba_cols = adata.uns.get(proba_cols_key, None)

    if proba_arr is not None and proba_cols is not None:
        try:
            arr = np.asarray(proba_arr)
            if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] == len(proba_cols):
                cols = [str(c) for c in proba_cols]
                proba = pd.DataFrame(arr, index=adata.obs_names, columns=cols)
        except Exception:
            proba = None

    stored_meta = adata.uns.get("celltypist_meta", {})
    if not isinstance(stored_meta, dict):
        stored_meta = {}

    meta = {
        "labels_ok": labels is not None,
        "proba_ok": proba is not None,
        "model_name": stored_meta.get("model_name", None),
    }
    return labels, proba, meta


def store_celltypist_outputs(
    adata: ad.AnnData,
    label_key: str,
    labels: Optional[np.ndarray],
    proba: Optional[pd.DataFrame],
    *,
    model_name: Optional[str] = None,
    proba_key: str = "celltypist_proba",
    proba_cols_key: str = "celltypist_proba_columns",
) -> None:
    if labels is not None:
        adata.obs[label_key] = pd.Series(labels, index=adata.obs_names).astype(str).astype("category")

    if proba is not None and not proba.empty:
        try:
            pm = proba.loc[adata.obs_names]
        except Exception:
            pm = proba.reindex(adata.obs_names)
        adata.obsm[proba_key] = pm.to_numpy()
        adata.uns[proba_cols_key] = list(pm.columns.astype(str))

    adata.uns["celltypist_meta"] = {
        "model_name": (None if model_name is None else str(model_name)),
    }


def ensure_celltypist(
    adata: ad.AnnData,
    cfg,
    *,
    reuse: bool = True,
    store: bool = True,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], dict]:
    label_key = str(getattr(cfg, "celltypist_label_key", "celltypist_label"))
    proba_key = "celltypist_proba"
    proba_cols_key = "celltypist_proba_columns"
    requested_model = getattr(cfg, "celltypist_model", None)

    meta = {"reused": False, "requested_model": requested_model}

    if requested_model is None:
        LOGGER.info("No CellTypist model provided; skipping CellTypist.")
        return None, None, meta

    if reuse:
        labels, proba, meta = get_celltypist_outputs(adata, label_key)
        meta["reused"] = False
        meta["requested_model"] = requested_model
        stored_model = meta.get("model_name", None)
        # Reuse only when labels and a valid probability matrix are both present.
        # Otherwise recompute to avoid stale/inconsistent mask inputs on subsetted objects.
        if labels is not None and proba is not None:
            if stored_model == requested_model:
                meta["reused"] = True
                return labels, proba, meta
            if stored_model is None:
                LOGGER.info(
                    "CellTypist cached outputs found but lack model metadata; recomputing for requested model %r.",
                    requested_model,
                )
            else:
                LOGGER.info(
                    "CellTypist cached outputs were generated with model %r, but %r was requested; recomputing.",
                    stored_model,
                    requested_model,
                )
        if labels is not None or proba is not None:
            LOGGER.info(
                "CellTypist reuse payload incomplete/invalid (labels_ok=%s, proba_ok=%s); recomputing.",
                bool(labels is not None),
                bool(proba is not None),
            )
            # Clear stale payloads before recompute so downstream cannot pick up mismatched arrays/columns.
            try:
                if proba_key in adata.obsm:
                    del adata.obsm[proba_key]
            except Exception:
                pass
            try:
                if proba_cols_key in adata.uns:
                    del adata.uns[proba_cols_key]
            except Exception:
                pass
            try:
                if "celltypist_meta" in adata.uns:
                    del adata.uns["celltypist_meta"]
            except Exception:
                pass

    try:
        LOGGER.info("Running CellTypist precompute (predictions + probabilities).")

        picked_layer: Optional[str] = None
        X_src = None

        for layer in ("counts_cb", "counts_raw"):
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
                "Using adata.X as-is (no normalize_total/log1p)."
            )
            adata_ct = ad.AnnData(
                X=adata.X,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata_ct.obs_names = adata.obs_names.copy()
            adata_ct.var_names = adata.var_names.copy()

        model_path = get_celltypist_model(requested_model)

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
            return None, None, meta

        prob_matrix = preds.probability_matrix
        if not isinstance(prob_matrix, pd.DataFrame) or prob_matrix.empty:
            LOGGER.warning("CellTypist returned no/empty probability_matrix; returning labels only.")
            if store:
                store_celltypist_outputs(adata, label_key, labels, None, model_name=requested_model)
            return labels, None, meta

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

        if store:
            store_celltypist_outputs(adata, label_key, labels, prob_matrix, model_name=requested_model)

        return labels, prob_matrix, meta

    except Exception as e:
        LOGGER.warning(
            "CellTypist precompute failed: %s. Proceeding without CellTypist outputs.",
            e,
        )
        return None, None, meta


def build_entropy_margin_mask(
    prob_matrix: pd.DataFrame,
    *,
    entropy_abs_limit: float,
    entropy_quantile: float,
    margin_min: float,
) -> tuple[np.ndarray, dict]:
    P = prob_matrix.to_numpy(dtype=np.float64, copy=False)
    n = P.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool), {"n_cells": 0}
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError("CellTypist confidence masking requires at least two probability columns.")
    if not np.isfinite(P).all():
        raise ValueError("CellTypist probability matrix contains non-finite values.")
    if (P < 0.0).any() or (P > 1.0).any():
        raise ValueError("CellTypist probability values must lie in [0, 1].")
    if not (0.0 < float(entropy_quantile) <= 1.0):
        raise ValueError("entropy_quantile must lie in (0, 1].")
    if float(entropy_abs_limit) < 0.0:
        raise ValueError("entropy_abs_limit must be non-negative.")
    if not (0.0 <= float(margin_min) <= 1.0):
        raise ValueError("margin_min must lie in [0, 1].")

    row_sums = P.sum(axis=1)
    if (row_sums <= 0.0).any():
        raise ValueError("CellTypist probability matrix contains rows with no positive score.")

    eps = 1e-12
    # CellTypist applies independent logistic transforms to one-vs-rest scores;
    # normalize only for Shannon entropy while retaining raw score margins.
    P_normalized = P / row_sums[:, None]
    P_clip = np.clip(P_normalized, eps, 1.0)
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
        "kept_frac": float(mask.mean()) if n > 0 else 0.0,
        "entropy_abs_limit": float(H_abs),
        "entropy_quantile": float(entropy_quantile),
        "entropy_q_value": float(H_q),
        "entropy_cut_used": float(H_cut),
        "entropy_cut_rule": "max_baseline_or_quantile",
        "entropy_probability_normalization": "row_sum",
        "probability_row_sum_min": float(row_sums.min()),
        "probability_row_sum_median": float(np.median(row_sums)),
        "probability_row_sum_max": float(row_sums.max()),
        "margin_min": float(margin_min),
    }
    return mask, stats


def summarize_cluster_celltypist_labels(
    clusters: pd.Series,
    labels: pd.Series,
    confidence_mask: np.ndarray,
    *,
    celltypist_ok: bool,
    min_confident_cells: int,
    min_confident_fraction: float,
    min_label_purity: float,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Assign auditable cluster labels from confidence-masked cell predictions."""
    if len(clusters) != len(labels) or len(clusters) != len(confidence_mask):
        raise ValueError("clusters, labels, and confidence_mask must have equal length.")
    if int(min_confident_cells) < 0:
        raise ValueError("min_confident_cells must be non-negative.")
    if not (0.0 <= float(min_confident_fraction) <= 1.0):
        raise ValueError("min_confident_fraction must lie in [0, 1].")
    if not (0.0 <= float(min_label_purity) <= 1.0):
        raise ValueError("min_label_purity must lie in [0, 1].")

    frame = pd.DataFrame(
        {
            "cluster": clusters.astype(str).to_numpy(),
            "label": labels.astype(str).to_numpy(),
            "confident": np.asarray(confidence_mask, dtype=bool),
        },
        index=clusters.index,
    )
    assignments: dict[str, str] = {}
    rows: list[dict[str, object]] = []

    for cluster, group in frame.groupby("cluster", sort=False):
        confident = group.loc[group["confident"]]
        n_total = int(group.shape[0])
        n_confident = int(confident.shape[0])
        confident_fraction = float(n_confident / n_total) if n_total else 0.0
        valid_labels = confident.loc[
            ~confident["label"].str.strip().str.lower().isin({"", "unknown", "nan", "none"}),
            "label",
        ]
        counts = valid_labels.value_counts()
        winner = "Unknown"
        winner_count = 0
        runner_up = ""
        runner_up_count = 0
        if not counts.empty:
            ranked = sorted(
                ((str(label), int(count)) for label, count in counts.items()),
                key=lambda item: (-item[1], item[0]),
            )
            winner, winner_count = ranked[0]
            if len(ranked) > 1:
                runner_up, runner_up_count = ranked[1]

        winner_fraction = float(winner_count / n_confident) if n_confident else 0.0
        runner_up_fraction = float(runner_up_count / n_confident) if n_confident else 0.0
        reason = "assigned"
        if not celltypist_ok:
            reason = "celltypist_unavailable"
        elif n_confident < int(min_confident_cells):
            reason = "insufficient_confident_cells"
        elif confident_fraction < float(min_confident_fraction):
            reason = "insufficient_confident_fraction"
        elif winner_count == 0:
            reason = "no_confident_labels"
        elif winner_fraction <= float(min_label_purity):
            reason = "insufficient_label_purity"

        assigned = winner if reason == "assigned" else "Unknown"
        assignments[str(cluster)] = assigned
        rows.append(
            {
                "cluster": str(cluster),
                "n_total": n_total,
                "n_confident": n_confident,
                "confident_fraction": confident_fraction,
                "winning_label": winner,
                "winning_count": winner_count,
                "winning_fraction": winner_fraction,
                "runner_up_label": runner_up,
                "runner_up_count": runner_up_count,
                "runner_up_fraction": runner_up_fraction,
                "assigned_label": assigned,
                "status": reason,
            }
        )

    audit = pd.DataFrame(rows)
    return assignments, audit
