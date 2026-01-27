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
    prefer_layers: tuple[str, ...] = ("counts_cb", "counts_raw"),
    store_key: str = "pseudobulk",
) -> None:
    """
    Compute and store cluster-level pseudobulk expression (genes x clusters) in adata.uns[store_key].

    Ordering policy (IMPORTANT / stable):
      - Clusters are ordered by descending cluster size (Leiden-style),
        with deterministic tie-break by cluster id (string).

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
    import numpy as np
    import pandas as pd

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
        else:
            LOGGER.info("Pseudobulk requested raw-like but none found; using adata.X.")

    # Cluster labels per cell (string)
    cl = adata.obs[cluster_key].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # Stable Leiden-style ordering: size desc, tie-break by cluster id
    # ------------------------------------------------------------------
    vc = pd.Series(cl).value_counts(dropna=False)  # desc by default
    # deterministic tie-break
    size_df = pd.DataFrame({"cluster": vc.index.astype(str), "n": vc.values.astype(int)})
    size_df["cluster_sort"] = size_df["cluster"].astype(str)
    size_df = size_df.sort_values(["n", "cluster_sort"], ascending=[False, True], kind="mergesort")
    clusters = size_df["cluster"].astype(str).tolist()

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

    # Preallocate columns in stable order
    expr_list: list[np.ndarray] = []
    clusters_out: list[str] = []

    for c in clusters:
        idx = np.where(cl == c)[0]
        if idx.size == 0:
            continue

        if sparse.issparse(X_csr):
            sub = X_csr[idx, :]
            if agg == "mean":
                vec = np.asarray(sub.mean(axis=0)).ravel()
            else:
                # median on sparse -> densify for that subset (clusters are usually not huge)
                vec = np.median(sub.toarray(), axis=0)
        else:
            sub = X_csr[idx, :]
            vec = sub.mean(axis=0) if agg == "mean" else np.median(sub, axis=0)

        expr_list.append(np.asarray(vec, dtype=np.float32))
        clusters_out.append(str(c))

    if not expr_list:
        raise RuntimeError("Pseudobulk: no clusters produced aggregated expression.")

    expr_mat = np.stack(expr_list, axis=1).astype(np.float32, copy=False)
    clusters_out_arr = np.asarray(clusters_out, dtype=object)
    genes_arr = np.asarray(genes, dtype=object)

    adata.uns[store_key] = {
        "level": "cluster",
        "cluster_key": cluster_key,
        "agg": agg,
        "use_raw_like": bool(use_raw_like),
        "prefer_layers": list(prefer_layers),
        "genes": genes_arr,
        "clusters": clusters_out_arr,
        "expr": expr_mat,
        # extra helpful metadata (non-breaking)
        "cluster_sizes": {str(k): int(v) for k, v in zip(size_df["cluster"].astype(str), size_df["n"].astype(int))},
        "cluster_order_policy": "size_desc_then_cluster_id",
    }

    LOGGER.info(
        "Stored pseudobulk in adata.uns[%r]: %d genes × %d clusters.",
        store_key,
        expr_mat.shape[0],
        expr_mat.shape[1],
    )

import copy
import numpy as np
import pandas as pd

def clone_decoupler_from_parent_round(
    adata: ad.AnnData,
    *,
    parent_round_id: str,
    child_round_id: str,
    publish_to_top_level_if_active: bool = True,
) -> None:
    _ensure_cluster_rounds(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if parent_round_id not in rounds or child_round_id not in rounds:
        raise KeyError("clone_decoupler: parent/child round not found")

    parent = rounds[parent_round_id]
    child = rounds[child_round_id]

    # need parent decoupler payloads
    pdec = parent.get("decoupler", {})
    if not isinstance(pdec, dict) or not pdec:
        raise KeyError("clone_decoupler: parent round has no decoupler dict")

    # mapping old->new from child's compacting reverse_map
    comp = child.get("compacting", {}) if isinstance(child.get("compacting", {}), dict) else {}
    rev = comp.get("reverse_map", None)
    if not isinstance(rev, dict) or not rev:
        raise KeyError("clone_decoupler: child round missing compacting.reverse_map")

    old_to_new = {}
    for new_id, olds in rev.items():
        if not isinstance(olds, (list, tuple)) or len(olds) != 1:
            # this clone helper is meant for no-merge compaction only
            continue
        old_to_new[str(olds[0])] = str(new_id)

    if not old_to_new:
        raise ValueError("clone_decoupler: could not construct old->new mapping")

    labels_obs_key = child.get("labels_obs_key", None)
    if not labels_obs_key or str(labels_obs_key) not in adata.obs:
        raise KeyError("clone_decoupler: child labels_obs_key missing/not in obs")
    labels_obs_key = str(labels_obs_key)

    # ------------------------------------------------------------
    # 1) pseudobulk: CLONE from parent (do not recompute)
    # ------------------------------------------------------------
    pb_key_child = f"pseudobulk__{child_round_id}"
    pb_key_parent = f"pseudobulk__{parent_round_id}"

    if pb_key_parent in adata.uns:
        adata.uns[pb_key_child] = copy.deepcopy(adata.uns[pb_key_parent])
    else:
        # fallback: recompute (but prefer counts_cb and fix agg lookup)
        agg = "mean"
        cfg0 = pdec.get("config", {})
        if isinstance(cfg0, dict):
            agg = str(cfg0.get("decoupler_pseudobulk_agg", agg) or agg)

        _store_cluster_pseudobulk(
            adata,
            cluster_key=labels_obs_key,
            agg=agg,
            use_raw_like=True,
            prefer_layers=("counts_cb", "counts_raw"),  # <-- prefer cb first
            store_key=pb_key_child,
        )

    child.setdefault("decoupler", {})
    child["decoupler"]["pseudobulk_store_key"] = pb_key_child

    # ------------------------------------------------------------
    # 2) clone each resource activity and reindex clusters
    # ------------------------------------------------------------
    def _reindex_activity(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x.index = x.index.astype(str).map(lambda c: old_to_new.get(str(c), str(c)))
        return x

    for res in ("msigdb", "progeny", "dorothea"):
        payload = pdec.get(res, None)
        if not isinstance(payload, dict) or "activity" not in payload:
            continue

        new_payload = dict(payload)
        act = payload["activity"]
        if isinstance(act, pd.DataFrame):
            new_payload["activity"] = _reindex_activity(act)

        if res == "msigdb":
            abg = payload.get("activity_by_gmt", None)
            if isinstance(abg, dict):
                new_abg = {}
                for gmt, df in abg.items():
                    if isinstance(df, pd.DataFrame):
                        new_abg[str(gmt)] = _reindex_activity(df)
                new_payload["activity_by_gmt"] = new_abg

        cfg0 = new_payload.get("config", {})
        if isinstance(cfg0, dict):
            cfg1 = dict(cfg0)
            cfg1["round_id"] = str(child_round_id)
            cfg1["cloned_from_round_id"] = str(parent_round_id)
            new_payload["config"] = cfg1

        _round_put_decoupler(adata, round_id=child_round_id, resource=res, payload=new_payload)

    # ------------------------------------------------------------
    # 3) rebuild display metadata for child
    # ------------------------------------------------------------
    display_map = _round_cluster_display_map(
        adata,
        round_id=child_round_id,
        labels_obs_key=labels_obs_key,
        cluster_label_key=CLUSTER_LABEL_KEY,
    )
    child["decoupler"]["cluster_display_map"] = dict(display_map)

    clusters = child.get("cluster_order", None)
    if isinstance(clusters, (list, tuple)) and clusters:
        clusters = [str(x) for x in clusters]
    else:
        pb_store = adata.uns.get(pb_key_child, None)
        clusters = [str(x) for x in np.asarray(pb_store.get("clusters", []), dtype=object).tolist()] if isinstance(pb_store, dict) else []

    child["decoupler"]["cluster_display_labels"] = [display_map.get(c, c) for c in clusters] if clusters else None
    child["decoupler"]["cluster_order"] = clusters if clusters else None

    rounds[child_round_id] = child
    adata.uns["cluster_rounds"] = rounds

    # ------------------------------------------------------------
    # 4) publish if active
    # ------------------------------------------------------------
    active = adata.uns.get("active_cluster_round", None)
    if publish_to_top_level_if_active and active is not None and str(active) == str(child_round_id):
        _publish_decoupler_from_round_to_top_level(adata, round_id=child_round_id)

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
        prefer_layers=("counts_cb", "counts_raw"),
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

    # Authoritative cluster order for plots (prefer round["cluster_order"] if present)
    clusters: list[str] = []
    try:
        # Prefer round-level stable order (size-sorted Leiden-style)
        co = rinfo.get("cluster_order", None)
        if isinstance(co, (list, tuple)) and co:
            clusters = [str(x) for x in co]
        else:
            # Fallback: pseudobulk-reported order
            pb_store = adata.uns.get(pb_key, None)
            if isinstance(pb_store, dict) and "clusters" in pb_store:
                clusters = [str(x) for x in np.asarray(pb_store["clusters"], dtype=object).tolist()]
    except Exception:
        clusters = []

    rinfo["decoupler"]["cluster_display_labels"] = (
        [display_map.get(str(c), str(c)) for c in clusters] if clusters else None
    )
    rinfo["decoupler"]["cluster_order"] = clusters if clusters else None

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
