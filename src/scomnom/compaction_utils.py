from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from scomnom.annotation_utils import ensure_round_msigdb_activity_by_gmt
from .clustering_utils import _ensure_cluster_rounds, _register_round

LOGGER = logging.getLogger(__name__)


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
        comp: list[str] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in nbr[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def _clique_components(nodes: List[str], pass_edges_set: set[Tuple[str, str]]) -> List[List[str]]:
    """
    Greedy clique cover:
    - repeatedly pick highest-degree remaining node as seed
    - grow a clique by adding candidates connected to all clique members
    """
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
    adata: ad.AnnData,
    round_snapshot: Dict[str, Any],
    celltypist_obs_key: str,
    min_cells: int = 0,
    zscore_scope: str = "global",
    similarity_metric: str = "cosine_zscore",
    grouping: str = "connected_components",
    thr_progeny: float = 0.98,
    thr_dorothea: float = 0.98,
    thr_msigdb_default: float = 0.98,
    thr_msigdb_by_gmt: Optional[Dict[str, float]] = None,
    msigdb_required: bool = True,
) -> CompactionOutputs:
    """
    Compaction decision engine.

    HARD GUARANTEE:
      - Compaction is confined within CellTypist label groups.
      - Clusters assigned to UNKNOWN/Unknown are NEVER compacted (always singletons).
    """
    import numpy as np
    import pandas as pd

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

    # ------------------------------------------------------------------
    # z-score strategy
    # ------------------------------------------------------------------
    zscope = (zscore_scope or "global").strip().lower()
    if zscope not in {"global", "within_celltypist_label"}:
        LOGGER.warning("Compaction: unknown zscore_scope=%r; falling back to 'global'.", zscore_scope)
        zscope = "global"

    WITHIN_LABEL_MIN_CLUSTERS = 4

    prog_z_global = _zscore_cols(prog)
    doro_z_global = _zscore_cols(doro)
    msig_z_global = {g: _zscore_cols(df) for g, df in msig_by_gmt.items()}

    # ------------------------------------------------------------------
    # Determine cluster grouping label (CellTypist majority per cluster)
    # ------------------------------------------------------------------
    def _normalize_ct_label(x: Any) -> str:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "na"}:
            return "UNKNOWN"
        # treat any "unknown" spelling/casing as UNKNOWN
        if s.strip().lower() == "unknown":
            return "UNKNOWN"
        return s

    ann = round_snapshot.get("annotation", {})
    ann = ann if isinstance(ann, dict) else {}
    ct_cluster_key = ann.get("celltypist_cluster_key", None)

    # If there is a cluster-level CT column, take MAJORITY vote (robust to NaNs/mixed)
    if isinstance(ct_cluster_key, str) and ct_cluster_key in adata.obs:
        tmp = pd.DataFrame(
            {
                "cluster": cluster_per_cell.values,
                "ct_cluster": adata.obs[ct_cluster_key].map(_normalize_ct_label).values,
            }
        )
        majority_ct: Dict[str, str] = (
            tmp.groupby("cluster", observed=True)["ct_cluster"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "UNKNOWN")
            .astype(str)
            .to_dict()
        )
        LOGGER.info(
            "Compaction: using cluster-level CT key %r for gating (majority vote).",
            str(ct_cluster_key),
        )
    else:
        tmp = pd.DataFrame(
            {
                "cluster": cluster_per_cell.values,
                "ct": adata.obs[celltypist_obs_key].map(_normalize_ct_label).values,
            }
        )
        majority_ct = (
            tmp.groupby("cluster", observed=True)["ct"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "UNKNOWN")
            .astype(str)
            .to_dict()
        )
        LOGGER.info(
            "Compaction: using cell-level CT key %r for gating (majority vote).",
            str(celltypist_obs_key),
        )

    label_to_clusters: Dict[str, List[str]] = {}
    for cl in all_clusters:
        if min_cells and int(cluster_sizes.get(cl, 0)) < int(min_cells):
            continue
        lab = _normalize_ct_label(majority_ct.get(cl, "UNKNOWN"))
        label_to_clusters.setdefault(lab, []).append(cl)

    # ------------------------------------------------------------------
    # HARD POLICY: UNKNOWN clusters are NEVER compacted (always singleton).
    # Remove UNKNOWN group from compaction graph construction.
    # Those clusters will be re-added as singleton components via "missing".
    # ------------------------------------------------------------------
    if "UNKNOWN" in label_to_clusters:
        n_unknown = len(label_to_clusters.get("UNKNOWN", []))
        LOGGER.info(
            "Compaction: excluding %d UNKNOWN clusters from compaction (policy: never compact Unknown).",
            int(n_unknown),
        )
        label_to_clusters.pop("UNKNOWN", None)

    # HARD sanity check: each cluster appears at most once in label_to_clusters
    all_seen = [c for xs in label_to_clusters.values() for c in xs]
    if len(all_seen) != len(set(all_seen)):
        raise RuntimeError("Compaction bug: a cluster appears in multiple CellTypist label groups")

    counts = {k: len(v) for k, v in label_to_clusters.items()}
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    LOGGER.info("Compaction: label_to_clusters top groups (label, n_clusters): %s", top)

    # ------------------------------------------------------------------
    # Thresholds + MSigDB majority + MSigDB top-K union
    # ------------------------------------------------------------------
    TOPK_MSIGDB = 25
    ADAPT_Q = 0.90

    FLOOR_PROG = 0.70
    FLOOR_DORO = 0.60

    MSIGDB_FLOOR_BY_GMT = {"HALLMARK": 0.60, "REACTOME": 0.45}
    FLOOR_MSIG_DEFAULT = 0.50
    MSIGDB_MAJORITY_FRAC = 0.67

    def _safe_quantile(vals: List[float], q: float, default: float) -> float:
        v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            return float(default)
        return float(np.quantile(v, q))

    def _cosine_topk_union(v_a: np.ndarray, v_b: np.ndarray, k: int = TOPK_MSIGDB) -> float:
        a = np.asarray(v_a, dtype=float)
        b = np.asarray(v_b, dtype=float)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        kk = int(min(int(k), int(a.size)))
        if kk <= 0:
            return 0.0
        ia = np.argpartition(np.abs(a), -kk)[-kk:]
        ib = np.argpartition(np.abs(b), -kk)[-kk:]
        idx = np.unique(np.concatenate([ia, ib], axis=0))
        if idx.size == 0:
            return 0.0
        return _cosine(a[idx], b[idx])

    def _cap_for_gmt(gmt: str) -> float:
        if gmt in thr_msigdb_by_gmt:
            return float(thr_msigdb_by_gmt[gmt])
        return float(thr_msigdb_default)

    def _floor_for_gmt(gmt: str) -> float:
        g = str(gmt).upper()
        return float(MSIGDB_FLOOR_BY_GMT.get(g, FLOOR_MSIG_DEFAULT))

    def _msig_required_passes(n_gmts: int, frac: float) -> int:
        if n_gmts <= 0:
            return 0
        if n_gmts == 1:
            return 1
        if n_gmts == 2:
            return 1
        need = int(np.ceil(float(frac) * float(n_gmts)))
        return int(max(1, min(n_gmts, need)))

    edge_rows: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    pass_edges_by_label: Dict[str, List[Tuple[str, str]]] = {}
    adaptive_thresholds_by_label: Dict[str, Dict[str, Any]] = {}

    def _effective_z_views_for_label(
        clusters_for_label: List[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], str]:
        c = [str(x) for x in clusters_for_label]
        if zscope == "within_celltypist_label" and len(c) >= WITHIN_LABEL_MIN_CLUSTERS:
            prog_z_loc = _zscore_cols(prog.loc[c])
            doro_z_loc = _zscore_cols(doro.loc[c])
            msig_z_loc = {g: _zscore_cols(msig_by_gmt[g].loc[c]) for g in msig_by_gmt.keys()}
            return prog_z_loc, doro_z_loc, msig_z_loc, "within_celltypist_label"

        return (
            prog_z_global.loc[c],
            doro_z_global.loc[c],
            {g: msig_z_global[g].loc[c] for g in msig_by_gmt.keys()},
            "global",
        )

    for ct_label, clusters in sorted(label_to_clusters.items(), key=lambda kv: kv[0]):
        clusters = sorted(map(str, clusters))
        if len(clusters) < 2:
            continue

        prog_z, doro_z, msig_z, effective_scope = _effective_z_views_for_label(clusters)

        prog_sims: List[float] = []
        doro_sims: List[float] = []
        msig_sims_by_gmt: Dict[str, List[float]] = {g: [] for g in msig_z.keys()}

        for i in range(len(clusters)):
            a = clusters[i]
            va_prog = prog_z.loc[a].to_numpy()
            va_doro = doro_z.loc[a].to_numpy()
            for j in range(i + 1, len(clusters)):
                b = clusters[j]
                prog_sims.append(_cosine(va_prog, prog_z.loc[b].to_numpy()))
                doro_sims.append(_cosine(va_doro, doro_z.loc[b].to_numpy()))
                for gmt, dfz in msig_z.items():
                    ms = _cosine_topk_union(dfz.loc[a].to_numpy(), dfz.loc[b].to_numpy(), k=TOPK_MSIGDB)
                    msig_sims_by_gmt[gmt].append(float(ms))

        q_prog = _safe_quantile(prog_sims, ADAPT_Q, default=FLOOR_PROG)
        q_doro = _safe_quantile(doro_sims, ADAPT_Q, default=FLOOR_DORO)

        thr_prog_ct = max(FLOOR_PROG, float(q_prog))
        thr_doro_ct = max(FLOOR_DORO, float(q_doro))

        if thr_progeny is not None:
            thr_prog_ct = min(float(thr_progeny), float(thr_prog_ct))
        if thr_dorothea is not None:
            thr_doro_ct = min(float(thr_dorothea), float(thr_doro_ct))

        thr_msig_ct: Dict[str, float] = {}
        for gmt, sims in msig_sims_by_gmt.items():
            floor_g = _floor_for_gmt(gmt)
            q_m = _safe_quantile(sims, ADAPT_Q, default=floor_g)
            thr_eff = max(float(floor_g), float(q_m))
            thr_eff = min(float(_cap_for_gmt(gmt)), float(thr_eff))
            thr_msig_ct[str(gmt)] = float(thr_eff)

        adaptive_thresholds_by_label[str(ct_label)] = {
            "thr_progeny": float(thr_prog_ct),
            "thr_dorothea": float(thr_doro_ct),
            "thr_msigdb_by_gmt": {str(g): float(t) for g, t in thr_msig_ct.items()},
            "adaptive_quantile": float(ADAPT_Q),
            "msigdb_topk": int(TOPK_MSIGDB),
            "msigdb_majority_frac": float(MSIGDB_MAJORITY_FRAC),
            "zscore_scope_arg": str(zscore_scope),
            "zscore_scope_effective": str(effective_scope),
            "within_label_min_clusters_for_zscore": int(WITHIN_LABEL_MIN_CLUSTERS),
        }

        passed_edges: List[Tuple[str, str]] = []

        for i in range(len(clusters)):
            a = clusters[i]
            va_prog = prog_z.loc[a].to_numpy()
            va_doro = doro_z.loc[a].to_numpy()
            for j in range(i + 1, len(clusters)):
                b = clusters[j]

                sim_prog = _cosine(va_prog, prog_z.loc[b].to_numpy())
                sim_doro = _cosine(va_doro, doro_z.loc[b].to_numpy())

                pass_prog = bool(sim_prog >= float(thr_prog_ct))
                pass_doro = bool(sim_doro >= float(thr_doro_ct))

                msig_pass_flags: Dict[str, bool] = {}
                msig_fail_gmts: List[str] = []
                msig_sims: Dict[str, float] = {}

                for gmt, dfz in msig_z.items():
                    s = _cosine_topk_union(dfz.loc[a].to_numpy(), dfz.loc[b].to_numpy(), k=TOPK_MSIGDB)
                    msig_sims[gmt] = float(s)
                    thr_g = float(
                        thr_msig_ct.get(
                            str(gmt),
                            min(_cap_for_gmt(str(gmt)), max(_floor_for_gmt(str(gmt)), 0.0)),
                        )
                    )
                    ok = bool(s >= thr_g)
                    msig_pass_flags[gmt] = ok
                    if not ok:
                        msig_fail_gmts.append(str(gmt))

                if msigdb_required:
                    n_gmts = int(len(msig_z))
                    if n_gmts <= 0:
                        pass_msigdb = False
                        n_pass = 0
                        need = 0
                    else:
                        n_pass = int(sum(bool(v) for v in msig_pass_flags.values()))
                        need = _msig_required_passes(n_gmts, MSIGDB_MAJORITY_FRAC)
                        pass_msigdb = bool(n_pass >= need)
                else:
                    pass_msigdb = True
                    n_pass = int(sum(bool(v) for v in msig_pass_flags.values()))
                    need = _msig_required_passes(int(len(msig_z)), MSIGDB_MAJORITY_FRAC)

                pass_all = bool(pass_prog and pass_doro and pass_msigdb)
                if pass_all:
                    passed_edges.append((a, b))

                row: Dict[str, Any] = {
                    "celltypist_label": ct_label,
                    "a": a,
                    "b": b,
                    "n_a": int(cluster_sizes.get(a, 0)),
                    "n_b": int(cluster_sizes.get(b, 0)),
                    "sim_progeny": float(sim_prog),
                    "thr_progeny": float(thr_prog_ct),
                    "pass_progeny": bool(pass_prog),
                    "sim_dorothea": float(sim_doro),
                    "thr_dorothea": float(thr_doro_ct),
                    "pass_dorothea": bool(pass_doro),
                    "msigdb_topk": int(TOPK_MSIGDB),
                    "msigdb_majority_frac": float(MSIGDB_MAJORITY_FRAC),
                    "msigdb_majority_need": int(need),
                    "msigdb_majority_passed": int(n_pass),
                    "pass_msigdb_majority": bool(pass_msigdb),
                    "fail_msigdb_gmts": ",".join(msig_fail_gmts) if msig_fail_gmts else "",
                    "pass_all": bool(pass_all),
                    "similarity_metric": f"{similarity_metric}+msigdb_topk_union_k{TOPK_MSIGDB}+adaptive_q{ADAPT_Q}",
                    "zscore_scope": str(zscore_scope),
                    "zscore_scope_effective": str(effective_scope),
                    "grouping": grouping,
                }

                for gmt, s in msig_sims.items():
                    thr_g = float(
                        thr_msig_ct.get(
                            str(gmt),
                            min(_cap_for_gmt(str(gmt)), max(_floor_for_gmt(str(gmt)), 0.0)),
                        )
                    )
                    row[f"sim_msigdb__{gmt}"] = float(s)
                    row[f"thr_msigdb__{gmt}"] = float(thr_g)
                    row[f"pass_msigdb__{gmt}"] = bool(s >= thr_g)

                row["pass_msigdb_all_gmt"] = bool(pass_msigdb)
                edge_rows.append(row)

        pass_edges_by_label[ct_label] = passed_edges
        adjacency[ct_label] = passed_edges

    edges_df = pd.DataFrame(edge_rows)

    decision_log: List[Dict[str, Any]] = []
    all_components: List[List[str]] = []

    for ct_label, clusters in sorted(label_to_clusters.items(), key=lambda kv: kv[0]):
        clusters = sorted(map(str, clusters))
        if not clusters:
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
                    "reason": (
                        "adaptive multiview agreement within CellTypist label "
                        "(progeny + dorothea + msigdb_majority(topK))"
                    ),
                    "grouping": grouping,
                    "adaptive": adaptive_thresholds_by_label.get(str(ct_label), None),
                }
            )
        all_components.extend([list(c) for c in comps])

    # Any cluster not covered yet becomes its own singleton (this includes UNKNOWN clusters we excluded above)
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



# =============================================================================
# Round creation helpers
# =============================================================================
def _apply_cluster_id_map_to_obs(
    adata: ad.AnnData,
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
    cfg,
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
    thr_progeny: float = 0.98,
    thr_dorothea: float = 0.98,
    thr_msigdb_default: float = 0.98,
    thr_msigdb_by_gmt: dict[str, float] | None = None,
    msigdb_required: bool = True,
) -> None:
    """
    Creates a new COMPACTED round derived from a parent round.

    HARD GUARANTEE:
      - UNKNOWN clusters are never compacted (forced).
    """
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

    if msigdb_required:
        ensure_round_msigdb_activity_by_gmt(parent)

    # Force policy: Unknown clusters are never compacted.
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
    )

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

    compacting_payload = {
        "parent_round_id": str(parent_round_id),
        "within_celltypist_label_only": True,
        "celltypist_gating_key": str(celltypist_obs_key),
        "unknown_policy": "never_compact_unknown",  # NEW explicit audit field
        "similarity_metric": "cosine_zscore",
        "params": {
            "min_cells": int(min_cells),
            "zscore_scope": str(zscore_scope),
            "grouping": str(grouping),
            "msigdb_required": bool(msigdb_required),
        },
        "thresholds": {
            "thr_progeny": float(thr_progeny),
            "thr_dorothea": float(thr_dorothea),
            "thr_msigdb_default": float(thr_msigdb_default),
            "thr_msigdb_by_gmt": dict(thr_msigdb_by_gmt or {}),
        },
        "pairwise": {
            "edges": outputs.edges,
            "adjacency": outputs.adjacency,
        },
        "components": None,
        "decision_log": outputs.decision_log,
        "reverse_map": outputs.reverse_map,
    }

    cfg_snapshot = None
    try:
        if hasattr(cfg, "__dataclass_fields__"):
            from dataclasses import asdict
            cfg_snapshot = asdict(cfg)
    except Exception:
        cfg_snapshot = None

    _register_round(
        adata,
        round_id=new_round_id,
        parent_round_id=parent_round_id,
        cluster_key=str(parent_cluster_key),
        labels_obs_key=str(labels_obs_key_new),
        kind="COMPACTED",
        best_resolution=None,
        sweep=None,
        cfg_snapshot=cfg_snapshot,
        notes=notes,
        cluster_id_map=dict(outputs.cluster_id_map),
        cluster_renumbering=dict(outputs.cluster_renumbering),
        compacting=compacting_payload,
        cache_labels=False,
    )
