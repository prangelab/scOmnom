from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from . import io_utils
from .io_utils import resolve_msigdb_gene_sets
from .clustering_utils import _ensure_cluster_rounds  # avoid duplication

LOGGER = logging.getLogger(__name__)

# Canonical pretty cluster label column name (your module uses this)
CLUSTER_LABEL_KEY = "cluster_label"

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
            # Fallback path (kept for safety)
            LOGGER.info("Running CellTypist on main AnnData for annotation (fallback).")
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

            # write cell-level labels
            if isinstance(cell_level_labels, (pd.Series, pd.DataFrame)):
                s = cell_level_labels.squeeze()
                adata.obs[cell_key] = s.astype(str).astype("category")
            else:
                adata.obs[cell_key] = pd.Series(np.asarray(cell_level_labels).ravel(), index=adata.obs_names).astype(str).astype("category")

            # probability matrix if available
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
    # --------------------------------------------------------------
    # Make stable ordinal prefix from sorted cluster ids (string)
    try:
        cats = sorted(pd.unique(clust_vals.astype(str)).astype(str).tolist())
    except Exception:
        cats = sorted(set(map(str, clust_vals.tolist())))

    ord_map = {c: f"C{i:02d}" for i, c in enumerate(cats)}

    maj = adata.obs[cluster_ct_key].astype(str).fillna("Unknown")
    pretty_labels = clust_vals.map(lambda c: f"{ord_map.get(str(c), 'C??')}: {maj.loc[maj.index[0]]}" if False else None)  # placeholder

    # Build pretty label per cell: "C00: <majority>" where majority is looked up by cluster id
    # (this avoids embedding raw cluster ids; it’s stable and readable)
    cl_to_maj = {str(k): str(v) for k, v in majority_map.items()}
    pretty = clust_vals.map(lambda c: f"{ord_map.get(str(c), 'C??')}: {cl_to_maj.get(str(c), 'Unknown')}")
    adata.obs[pretty_key] = pretty.astype("category")
    adata.obs[CLUSTER_LABEL_KEY] = adata.obs[pretty_key]  # alias to latest round

    # Palette for round-scoped pretty labels + alias
    try:
        from scanpy.plotting.palettes import default_102
        cats_pretty = adata.obs[pretty_key].cat.categories
        adata.uns[f"{pretty_key}_colors"] = list(default_102[: len(cats_pretty)])
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
                    "cluster_key_used": str(cluster_key),
                    "pretty_label_masked": True,
                    "pretty_label_min_masked_cells": int(min_masked_cells),
                    "pretty_label_min_masked_frac": float(min_masked_frac),
                    "celltypist_ok": bool(celltypist_ok),
                }
            )
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


# -------------------------------------------------------------------------
# Pseudobulk store (round-scoped)
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
        raise KeyError(
            f"Malformed pseudobulk store at adata.uns[{store_key!r}]; missing: {sorted(missing)}"
        )

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

    Storage format:
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
    """
    from scipy import sparse

    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in adata.obs")

    agg = (agg or "mean").lower().strip()
    if agg not in {"mean", "median"}:
        agg = "mean"

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

    expr_mat = np.stack([expr_cols[c] for c in expr_cols.keys()], axis=1).astype(
        np.float32, copy=False
    )
    clusters_out = np.array(list(expr_cols.keys()), dtype=object)

    adata.uns[store_key] = {
        "level": "cluster",
        "cluster_key": cluster_key,
        "agg": agg,
        "use_raw_like": bool(use_raw_like),
        "prefer_layers": list(prefer_layers),
        "genes": np.asarray(genes, dtype=object),
        "clusters": clusters_out,
        "expr": expr_mat,
    }

    LOGGER.info(
        "Stored pseudobulk in adata.uns[%r]: %d genes × %d clusters.",
        store_key,
        expr_mat.shape[0],
        expr_mat.shape[1],
    )


# -------------------------------------------------------------------------
# MSigDB utilities (activity_by_gmt)
# -------------------------------------------------------------------------
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
      - feature_meta: df indexed by term OR has column 'term', with gmt_col giving family.
    If feature_meta not provided, falls back to prefix inference.
    """
    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        return {}

    A = activity.copy()
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)

    if feature_meta is not None and isinstance(feature_meta, pd.DataFrame) and not feature_meta.empty:
        fm = feature_meta.copy()
        if "term" in fm.columns:
            try:
                fm = fm.set_index("term", drop=False)
            except Exception:
                pass
        fm.index = fm.index.astype(str)

        if gmt_col not in fm.columns:
            prefixes = pd.Series(A.columns, index=A.columns).map(_infer_msigdb_gmt)
        else:
            prefixes = pd.Series(A.columns, index=A.columns).map(
                lambda c: fm.loc[c, gmt_col] if c in fm.index else np.nan
            )
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

    if (
        "activity_by_gmt" in ms
        and isinstance(ms["activity_by_gmt"], dict)
        and ms["activity_by_gmt"]
    ):
        return

    activity = ms.get(activity_key, None)
    feature_meta = ms.get(feature_meta_key, None)

    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        raise KeyError(
            "Cannot build activity_by_gmt: round['decoupler']['msigdb']['activity'] missing/empty."
        )

    if feature_meta is not None and isinstance(feature_meta, dict):
        feature_meta = pd.DataFrame(feature_meta)

    by_gmt = msigdb_activity_by_gmt_from_activity_and_meta(
        activity, feature_meta=feature_meta, gmt_col=gmt_col
    )
    if not by_gmt:
        raise ValueError("Failed to build MSigDB activity_by_gmt (no GMT blocks produced).")

    ms["activity_by_gmt"] = by_gmt


# -------------------------------------------------------------------------
# Decoupler runner (shared low-level helper)
# -------------------------------------------------------------------------
def _dc_run_method(
    *,
    method: str,
    mat: pd.DataFrame,
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
    min_n: int = 5,
    consensus_methods: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run a decoupler method on an expression matrix + network and return activities.

    Returns: sources x samples (i.e., regulators/pathways x clusters).
    """
    import decoupler as dc

    if mat is None or not isinstance(mat, pd.DataFrame) or mat.empty:
        raise ValueError("decoupler: 'mat' must be a non-empty pandas DataFrame.")
    if net is None or not isinstance(net, pd.DataFrame) or net.empty:
        raise ValueError("decoupler: 'net' must be a non-empty pandas DataFrame.")

    method = (method or "consensus").lower().strip()

    for col in (source, target):
        if col not in net.columns:
            raise KeyError(f"decoupler: net missing required column '{col}'")

    net_use = net.copy()
    net_use[source] = net_use[source].astype(str)
    net_use[target] = net_use[target].astype(str)

    if source != "source":
        net_use = net_use.rename(columns={source: "source"})
    if target != "target":
        net_use = net_use.rename(columns={target: "target"})
    if weight is None or weight not in net_use.columns:
        net_use["weight"] = 1.0
    elif weight != "weight":
        net_use = net_use.rename(columns={weight: "weight"})

    # Normalize expression orientation: samples x genes
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

    tgt_counts = net_use.groupby("source")["target"].nunique()
    keep_sources = tgt_counts[tgt_counts >= int(min_n)].index.astype(str).tolist()
    net_use = net_use[net_use["source"].isin(keep_sources)].copy()
    if net_use.empty:
        raise ValueError(f"decoupler: after min_n={min_n} filtering, net is empty.")

    try:
        available = {m.lower() for m in dc.mt.show_methods()}
    except Exception:
        available = {m.lower() for m in dir(dc.mt) if not m.startswith("_")}

    def _run_one(m: str) -> pd.DataFrame:
        m = (m or "").lower().strip()
        if m not in available:
            raise ValueError(
                f"decoupler: method '{m}' not available. "
                f"Use decoupler.mt.show_methods() to inspect available methods."
            )

        func = getattr(dc.mt, m)
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
            raise RuntimeError(
                "decoupler consensus: all constituent methods failed: " + " | ".join(errs)
            )

        common_sources = set(ests[0].index)
        common_samples = set(ests[0].columns)
        for e in ests[1:]:
            common_sources &= set(e.index)
            common_samples &= set(e.columns)

        if not common_sources or not common_samples:
            raise RuntimeError("decoupler consensus: no common sources/samples across methods.")

        common_sources = sorted(common_sources)
        common_samples = sorted(common_samples)

        stack = np.stack(
            [e.loc[common_sources, common_samples].to_numpy() for e in ests], axis=0
        )
        mean_est = np.nanmean(stack, axis=0)

        return pd.DataFrame(mean_est, index=common_sources, columns=common_samples)

    return _run_one(method)


# -------------------------------------------------------------------------
# Round storage + publishing
# -------------------------------------------------------------------------
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
            del adata.uns["pseudobulk"]

    for res in resources:
        payload = dec.get(res, None)
        if isinstance(payload, dict) and "activity" in payload:
            adata.uns[res] = payload
            try:
                adata.uns[res].setdefault("config", {})
                adata.uns[res]["config"]["cluster_display_map"] = dec.get("cluster_display_map", None)
                adata.uns[res]["config"]["cluster_display_labels"] = dec.get(
                    "cluster_display_labels", None
                )
            except Exception:
                pass
        else:
            if clear_missing and res in adata.uns:
                del adata.uns[res]


def _round_cluster_display_map(
    adata: ad.AnnData,
    *,
    round_id: str,
    labels_obs_key: str,
    cluster_label_key: str = CLUSTER_LABEL_KEY,
) -> dict[str, str]:
    """
    Build {cluster_id -> display_label} for a round.

    Prefers round-scoped pretty labels column: f"{cluster_label_key}__{round_id}".
    Falls back to identity mapping.
    """
    if labels_obs_key not in adata.obs:
        raise KeyError(f"labels_obs_key '{labels_obs_key}' not found in adata.obs")

    pretty_key = f"{cluster_label_key}__{round_id}"
    if pretty_key not in adata.obs:
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

    m = (
        tmp.groupby("cluster_id", observed=True)["pretty"]
        .agg(lambda x: str(x.iloc[0]) if len(x) else "Unknown")
        .astype(str)
        .to_dict()
    )
    return {str(k): str(v) for k, v in m.items()}


# -------------------------------------------------------------------------
# Resource-specific runners
# -------------------------------------------------------------------------
def _run_msigdb(
    adata: ad.AnnData,
    cfg,
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "msigdb",
) -> None:
    rid = _get_round_id_default(adata, round_id)

    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)  # genes x clusters
    except Exception as e:
        LOGGER.warning("MSigDB: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("MSigDB: pseudobulk expr is empty; skipping.")
        return

    gene_sets = getattr(cfg, "msigdb_gene_sets", None) or ["HALLMARK", "REACTOME"]

    try:
        gmt_files, used_keywords, msigdb_release = resolve_msigdb_gene_sets(gene_sets)
    except Exception as e:
        LOGGER.warning("MSigDB: failed to resolve gene sets: %s", e)
        return
    if not gmt_files:
        LOGGER.warning("MSigDB: no gene set files resolved; skipping.")
        return

    method = (
        getattr(cfg, "msigdb_method", None)
        or getattr(cfg, "decoupler_method", None)
        or "consensus"
    )
    min_n = int(getattr(cfg, "msigdb_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)

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

    try:
        feature_meta = pd.DataFrame(
            {"term": activity.columns.astype(str), "gmt": [_infer_msigdb_gmt(c) for c in activity.columns.astype(str)]}
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
        "activity_by_gmt": activity_by_gmt,
        "feature_meta": feature_meta,
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
    adata: ad.AnnData,
    cfg,
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "progeny",
) -> None:
    import decoupler as dc

    rid = _get_round_id_default(adata, round_id)

    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)
    except Exception as e:
        LOGGER.warning("PROGENy: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("PROGENy: pseudobulk expr is empty; skipping.")
        return

    method = (
        getattr(cfg, "progeny_method", None)
        or getattr(cfg, "decoupler_method", None)
        or "consensus"
    )
    min_n = int(getattr(cfg, "progeny_min_n_targets", getattr(cfg, "decoupler_min_n_targets", 5)) or 5)
    top_n = int(getattr(cfg, "progeny_top_n", 100) or 100)
    organism = getattr(cfg, "progeny_organism", "human") or "human"

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
        LOGGER.warning("PROGENy: no usable weight column found; skipping.")
        return

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
        activity = est.T
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
    adata: ad.AnnData,
    cfg,
    *,
    store_key: str = "pseudobulk",
    round_id: str | None = None,
    out_resource: str = "dorothea",
) -> None:
    import decoupler as dc

    rid = _get_round_id_default(adata, round_id)

    try:
        expr = _get_cluster_pseudobulk_df(adata, store_key=store_key)
    except Exception as e:
        LOGGER.warning("DoRothEA: missing/invalid pseudobulk store '%s'; skipping. (%s)", store_key, e)
        return
    if expr.empty:
        LOGGER.warning("DoRothEA: pseudobulk expr is empty; skipping.")
        return

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
        LOGGER.warning("DoRothEA: no usable weight column found; skipping.")
        return

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
        activity = est.T
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


# -------------------------------------------------------------------------
# Public orchestrator: round-native decoupler run
# -------------------------------------------------------------------------
def run_decoupler_for_round(
    adata: ad.AnnData,
    cfg,
    *,
    round_id: str | None = None,
    publish_to_top_level_if_active: bool = True,
) -> None:
    """
    Round-native decoupler orchestrator:

    1) Computes round-scoped pseudobulk store: adata.uns[f"pseudobulk__{round_id}"]
       IMPORTANT: pseudobulk is computed on the round-stable cluster ids
       stored in round["labels_obs_key"].

    2) Runs enabled decoupler resources and stores them in:
         adata.uns["cluster_rounds"][round_id]["decoupler"][resource]

    3) Records pointers + plotting display metadata in the round:
         round["decoupler"]["pseudobulk_store_key"] = ...
         round["decoupler"]["cluster_display_map"] = {cluster_id -> pretty_label}
         round["decoupler"]["cluster_display_labels"] = [...] aligned to pseudobulk cluster order

    4) If round_id is active and publish_to_top_level_if_active=True, publishes
       the round's decoupler payloads to top-level adata.uns["msigdb"/"progeny"/"dorothea"/"pseudobulk"].
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

    # 1) Pseudobulk
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

    rinfo["decoupler"]["pseudobulk_store_key"] = pb_key

    # 2) Display metadata
    display_map = _round_cluster_display_map(
        adata,
        round_id=rid,
        labels_obs_key=labels_obs_key,
        cluster_label_key=CLUSTER_LABEL_KEY,
    )
    rinfo["decoupler"]["cluster_display_map"] = dict(display_map)

    try:
        pb_store = adata.uns.get(pb_key, None)
        clusters = []
        if isinstance(pb_store, dict) and "clusters" in pb_store:
            clusters = list(np.asarray(pb_store["clusters"], dtype=object))
        rinfo["decoupler"]["cluster_display_labels"] = (
            [display_map.get(str(c), str(c)) for c in clusters] if clusters else None
        )
    except Exception:
        rinfo["decoupler"]["cluster_display_labels"] = None

    rounds[rid] = rinfo
    adata.uns["cluster_rounds"] = rounds

    # 3) Resources
    _run_msigdb(adata, cfg, store_key=pb_key, round_id=rid, out_resource="msigdb")

    if getattr(cfg, "run_dorothea", True):
        _run_dorothea(adata, cfg, store_key=pb_key, round_id=rid, out_resource="dorothea")

    if getattr(cfg, "run_progeny", True):
        _run_progeny(adata, cfg, store_key=pb_key, round_id=rid, out_resource="progeny")

    # 4) Publish if active
    active = adata.uns.get("active_cluster_round", None)
    if publish_to_top_level_if_active and (active is not None) and (str(active) == rid):
        _publish_decoupler_from_round_to_top_level(
            adata,
            round_id=rid,
            resources=("msigdb", "progeny", "dorothea"),
            publish_pseudobulk=True,
            clear_missing=True,
        )

        # Also publish display hints into each top-level payload (best-effort)
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
