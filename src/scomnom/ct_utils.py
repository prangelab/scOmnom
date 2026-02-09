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

    meta = {
        "labels_ok": labels is not None,
        "proba_ok": proba is not None,
    }
    return labels, proba, meta


def store_celltypist_outputs(
    adata: ad.AnnData,
    label_key: str,
    labels: Optional[np.ndarray],
    proba: Optional[pd.DataFrame],
    *,
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


def ensure_celltypist(
    adata: ad.AnnData,
    cfg,
    *,
    reuse: bool = True,
    store: bool = True,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], dict]:
    label_key = str(getattr(cfg, "celltypist_label_key", "celltypist_label"))

    if reuse:
        labels, proba, meta = get_celltypist_outputs(adata, label_key)
        if labels is not None or proba is not None:
            meta["reused"] = True
            return labels, proba, meta

    meta = {"reused": False}

    if getattr(cfg, "celltypist_model", None) is None:
        LOGGER.info("No CellTypist model provided; skipping CellTypist.")
        return None, None, meta

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
            return None, None, meta

        prob_matrix = preds.probability_matrix
        if not isinstance(prob_matrix, pd.DataFrame) or prob_matrix.empty:
            LOGGER.warning("CellTypist returned no/empty probability_matrix; returning labels only.")
            if store:
                store_celltypist_outputs(adata, label_key, labels, None)
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
            store_celltypist_outputs(adata, label_key, labels, prob_matrix)

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
        "kept_frac": float(mask.mean()) if n > 0 else 0.0,
        "entropy_abs_limit": float(H_abs),
        "entropy_quantile": float(entropy_quantile),
        "entropy_q_value": float(H_q),
        "entropy_cut_used": float(H_cut),
        "margin_min": float(margin_min),
    }
    return mask, stats
