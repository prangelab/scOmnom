from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scomnom.annotation_utils import ensure_round_msigdb_activity_by_gmt
from .clustering_utils import _create_shallow_round_from_parent, _ensure_cluster_rounds

LOGGER = logging.getLogger(__name__)

FLOOR_PROGENY = 0.70
FLOOR_DOROTHEA = 0.60
FLOOR_TRANSCRIPTOMIC = 0.90
MSIGDB_FLOOR_BY_GMT = {"HALLMARK": 0.60, "REACTOME": 0.45}
FLOOR_MSIGDB_DEFAULT = 0.50
MSIGDB_MAJORITY_FRAC = 0.67
MSIGDB_TOPK = 25
ADAPTIVE_MIN_GROUP_SIZE = 4
MIN_VARIABLE_FEATURES = 2
DEFAULT_TRANSCRIPTOMIC_N_FEATURES = 2000
STATE_NORMALIZATION_TARGET = 10000.0
STATE_LOG2FC_PSEUDOCOUNT = 0.1
STATE_MIN_EXPRESSED_FRACTION = 0.05
STATE_TECHNICAL_PREFIXES = ("MT-", "RPS", "RPL", "MTRNR")
STATE_TECHNICAL_GENES = {"MALAT1"}
STATE_LOOSE_LOG2FC_THRESHOLD = 0.75
STATE_LOOSE_DETECTION_DELTA_THRESHOLD = 0.15
STATE_STRICT_LOG2FC_THRESHOLD = 1.50
STATE_STRICT_DETECTION_DELTA_THRESHOLD = 0.25
DEFAULT_STATE_LOG2FC_THRESHOLD = 1.00
DEFAULT_STATE_DETECTION_DELTA_THRESHOLD = 0.20
DEFAULT_STATE_MAX_FRACTION = 0.02
COMPACTION_METHOD_IDENTITY = "multiview_all_pairs_with_state_divergence_veto"


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
    cluster_eligibility: pd.DataFrame
    group_membership: pd.DataFrame
    thresholds_by_label: pd.DataFrame
    view_audit: pd.DataFrame
    transcriptomic_provenance: Dict[str, Any]


@dataclass
class _StateDivergenceEvidence:
    expression: pd.DataFrame
    detection: pd.DataFrame
    technical_mask: np.ndarray


def _normalize_celltypist_label(value: Any) -> str:
    label = str(value).strip()
    if not label or label.lower() in {"unknown", "nan", "none", "null", "na"}:
        return "UNKNOWN"
    return label


def _prepare_activity_view(
    name: str,
    value: Any,
    *,
    all_clusters: list[str],
    required: bool,
    min_variable_features: int = MIN_VARIABLE_FEATURES,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    audit: dict[str, Any] = {
        "view": str(name),
        "required": bool(required),
        "status": "available",
        "n_input_features": 0,
        "n_variable_features": 0,
        "n_complete_clusters": 0,
        "incomplete_clusters": "",
        "dropped_constant_features": "",
    }
    if value is None or not isinstance(value, pd.DataFrame) or value.empty:
        audit["status"] = "missing"
        if required:
            raise ValueError(f"Compaction requires non-empty {name} activity.")
        return None, audit

    frame = value.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Compaction {name} activity has duplicate cluster rows.")
    if frame.columns.has_duplicates:
        raise ValueError(f"Compaction {name} activity has duplicate feature columns.")

    numeric = frame.apply(pd.to_numeric, errors="coerce").reindex(index=all_clusters)
    audit["n_input_features"] = int(numeric.shape[1])
    finite_rows = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    complete_clusters = numeric.index[finite_rows].astype(str).tolist()
    incomplete_clusters = numeric.index[~finite_rows].astype(str).tolist()
    audit["n_complete_clusters"] = int(len(complete_clusters))
    audit["incomplete_clusters"] = ",".join(incomplete_clusters)

    if len(complete_clusters) < 2:
        audit["status"] = "insufficient_complete_clusters"
        if required:
            raise ValueError(
                f"Compaction {name} activity has fewer than two clusters with complete finite evidence."
            )
        return None, audit

    complete = numeric.loc[complete_clusters]
    standard_deviation = complete.std(axis=0)
    variable_columns = standard_deviation.index[
        np.isfinite(standard_deviation.to_numpy(dtype=float))
        & (standard_deviation.to_numpy(dtype=float) > 0.0)
    ].astype(str).tolist()
    dropped = [column for column in numeric.columns if str(column) not in set(variable_columns)]
    audit["n_variable_features"] = int(len(variable_columns))
    audit["dropped_constant_features"] = ",".join(map(str, dropped))

    if len(variable_columns) < int(min_variable_features):
        audit["status"] = "insufficient_variable_features"
        if required:
            raise ValueError(
                f"Compaction {name} activity has {len(variable_columns)} variable features; "
                f"at least {int(min_variable_features)} are required."
            )
        return None, audit

    numeric = numeric.loc[:, variable_columns]
    zscores = pd.DataFrame(
        np.nan,
        index=numeric.index,
        columns=numeric.columns,
        dtype=float,
    )
    means = complete.loc[:, variable_columns].mean(axis=0)
    scales = complete.loc[:, variable_columns].std(axis=0)
    zscores.loc[complete_clusters] = (complete.loc[:, variable_columns] - means) / scales
    return zscores, audit


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or a.size != b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.size != b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    a_centered = a - float(a.mean())
    b_centered = b - float(b.mean())
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denominator)


def _prepare_transcriptomic_view(
    adata: ad.AnnData,
    *,
    cluster_per_cell: pd.Series,
    all_clusters: list[str],
    source: str = "auto",
    n_features: int = DEFAULT_TRANSCRIPTOMIC_N_FEATURES,
) -> tuple[pd.DataFrame, _StateDivergenceEvidence, dict[str, Any], dict[str, Any]]:
    requested_source = str(source).strip()
    normalized_source = requested_source.lower()
    valid_sources = {"auto", "counts_cb", "counts_raw", "x"}
    if normalized_source not in valid_sources:
        raise ValueError("transcriptomic_source must be one of: auto, counts_cb, counts_raw, X")
    if int(n_features) < MIN_VARIABLE_FEATURES:
        raise ValueError(
            f"transcriptomic_n_features must be at least {MIN_VARIABLE_FEATURES}"
        )

    if normalized_source == "auto":
        if "counts_cb" in adata.layers:
            resolved_source = "counts_cb"
        elif "counts_raw" in adata.layers:
            resolved_source = "counts_raw"
        else:
            raise ValueError(
                "Compaction requires authoritative count evidence in counts_cb or counts_raw "
                "when transcriptomic_source='auto'. Set transcriptomic_source='X' explicitly "
                "only when adata.X contains nonnegative counts."
            )
    elif normalized_source == "x":
        resolved_source = "X"
    else:
        resolved_source = normalized_source

    if resolved_source == "X":
        matrix = adata.X
    else:
        if resolved_source not in adata.layers:
            raise KeyError(f"Compaction transcriptomic source {resolved_source!r} is unavailable.")
        matrix = adata.layers[resolved_source]
    aggregation = "cluster_sum_target_10000"

    if matrix.shape != adata.shape:
        raise ValueError("Compaction transcriptomic matrix is not aligned to adata.")
    if adata.var_names.astype(str).duplicated().any():
        raise ValueError("Compaction transcriptomic features must have unique names.")

    if sparse.issparse(matrix):
        count_matrix = matrix.tocsr().astype(np.float64)
    else:
        count_matrix = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    count_matrix.eliminate_zeros()
    if count_matrix.data.size and (
        not np.isfinite(count_matrix.data).all() or count_matrix.data.min() < 0.0
    ):
        raise ValueError(
            f"Compaction count source {resolved_source!r} contains non-finite or negative values."
        )

    cluster_index = {cluster: index for index, cluster in enumerate(all_clusters)}
    codes = cluster_per_cell.map(cluster_index).to_numpy(dtype=int)
    membership = sparse.csr_matrix(
        (
            np.ones(adata.n_obs, dtype=float),
            (codes, np.arange(adata.n_obs, dtype=int)),
        ),
        shape=(len(all_clusters), adata.n_obs),
    )
    aggregated = membership @ count_matrix
    if sparse.issparse(aggregated):
        aggregated = aggregated.toarray()
    else:
        aggregated = np.asarray(aggregated, dtype=float)

    detected = count_matrix.copy()
    detected.data = np.ones_like(detected.data, dtype=np.float64)
    detection_counts = membership @ detected
    if sparse.issparse(detection_counts):
        detection_counts = detection_counts.toarray()
    else:
        detection_counts = np.asarray(detection_counts, dtype=float)

    cluster_sizes = np.bincount(codes, minlength=len(all_clusters)).astype(float)
    library_sizes = aggregated.sum(axis=1)
    if (library_sizes <= 0.0).any():
        empty = [all_clusters[index] for index in np.flatnonzero(library_sizes <= 0.0)]
        raise ValueError(f"Compaction transcriptomic pseudobulks are empty for clusters: {empty}")
    normalized_expression = (
        aggregated / library_sizes[:, None] * STATE_NORMALIZATION_TARGET
    )
    detection_fraction = detection_counts / cluster_sizes[:, None]
    values = np.log1p(normalized_expression)

    finite_features = np.isfinite(values).all(axis=0)
    variances = np.var(values, axis=0)
    variable_features = finite_features & np.isfinite(variances) & (variances > 0.0)
    variable_indices = np.flatnonzero(variable_features)
    if variable_indices.size < MIN_VARIABLE_FEATURES:
        raise ValueError(
            "Compaction transcriptomic view has fewer than two complete variable features."
        )
    ranked = variable_indices[
        np.argsort(-variances[variable_indices], kind="stable")
    ]
    selected = ranked[: min(int(n_features), int(ranked.size))]
    selected_names = adata.var_names.astype(str).to_numpy()[selected].tolist()
    selected_hash = hashlib.sha256("\n".join(selected_names).encode("utf-8")).hexdigest()
    frame = pd.DataFrame(
        values[:, selected],
        index=all_clusters,
        columns=selected_names,
        dtype=float,
    )
    gene_names = adata.var_names.astype(str)
    upper_gene_names = gene_names.str.upper()
    technical_mask = np.asarray(
        [
            name in STATE_TECHNICAL_GENES
            or any(name.startswith(prefix) for prefix in STATE_TECHNICAL_PREFIXES)
            for name in upper_gene_names
        ],
        dtype=bool,
    )
    state_evidence = _StateDivergenceEvidence(
        expression=pd.DataFrame(
            normalized_expression,
            index=all_clusters,
            columns=gene_names,
            dtype=float,
        ),
        detection=pd.DataFrame(
            detection_fraction,
            index=all_clusters,
            columns=gene_names,
            dtype=float,
        ),
        technical_mask=technical_mask,
    )
    audit = {
        "view": "Transcriptome",
        "required": True,
        "status": "available",
        "decision_role": "state_divergence_veto",
        "pearson_role": "diagnostic_only",
        "source": resolved_source,
        "aggregation": aggregation,
        "feature_selection": "top_variance_across_parent_cluster_pseudobulks",
        "n_input_features": int(adata.n_vars),
        "n_variable_features": int(variable_indices.size),
        "n_selected_features": int(selected.size),
        "n_complete_clusters": int(len(all_clusters)),
        "incomplete_clusters": "",
        "selected_feature_sha256": selected_hash,
    }
    provenance = {
        "requested_source": requested_source,
        "resolved_source": resolved_source,
        "aggregation": aggregation,
        "normalization_target_sum": STATE_NORMALIZATION_TARGET,
        "state_log2fc_pseudocount": STATE_LOG2FC_PSEUDOCOUNT,
        "state_min_expressed_fraction": STATE_MIN_EXPRESSED_FRACTION,
        "state_technical_prefixes": list(STATE_TECHNICAL_PREFIXES),
        "state_technical_genes": sorted(STATE_TECHNICAL_GENES),
        "feature_selection": "top_variance_across_parent_cluster_pseudobulks",
        "n_input_features": int(adata.n_vars),
        "n_variable_features": int(variable_indices.size),
        "n_selected_features": int(selected.size),
        "selected_feature_sha256": selected_hash,
        "selected_features": selected_names,
    }
    return frame, state_evidence, audit, provenance


def _state_divergence_metrics(
    evidence: _StateDivergenceEvidence,
    *,
    cluster_a: str,
    cluster_b: str,
    log2fc_threshold: float,
    detection_delta_threshold: float,
) -> dict[str, Any]:
    expression_a = evidence.expression.loc[cluster_a].to_numpy(dtype=float)
    expression_b = evidence.expression.loc[cluster_b].to_numpy(dtype=float)
    detection_a = evidence.detection.loc[cluster_a].to_numpy(dtype=float)
    detection_b = evidence.detection.loc[cluster_b].to_numpy(dtype=float)
    log2fc = np.log2(
        (expression_a + STATE_LOG2FC_PSEUDOCOUNT)
        / (expression_b + STATE_LOG2FC_PSEUDOCOUNT)
    )
    detection_delta = np.abs(detection_a - detection_b)
    eligible = (
        np.isfinite(log2fc)
        & np.isfinite(detection_delta)
        & (np.maximum(detection_a, detection_b) >= STATE_MIN_EXPRESSED_FRACTION)
        & ~evidence.technical_mask
    )
    n_eligible = int(eligible.sum())
    if n_eligible < MIN_VARIABLE_FEATURES:
        raise ValueError(
            f"Compaction pair {cluster_a!r}/{cluster_b!r} has fewer than two expressed "
            "non-technical genes for the state-divergence veto."
        )

    abs_log2fc = np.abs(log2fc)

    def summarize(log2fc_cutoff: float, detection_cutoff: float) -> tuple[int, float]:
        affected = eligible & (abs_log2fc >= log2fc_cutoff) & (
            detection_delta >= detection_cutoff
        )
        count = int(affected.sum())
        return count, float(count / n_eligible)

    n_loose, fraction_loose = summarize(
        STATE_LOOSE_LOG2FC_THRESHOLD,
        STATE_LOOSE_DETECTION_DELTA_THRESHOLD,
    )
    n_medium, fraction_medium = summarize(log2fc_threshold, detection_delta_threshold)
    n_strict, fraction_strict = summarize(
        STATE_STRICT_LOG2FC_THRESHOLD,
        STATE_STRICT_DETECTION_DELTA_THRESHOLD,
    )
    return {
        "n_state_genes_eligible": n_eligible,
        "n_state_genes_loose": n_loose,
        "fraction_state_genes_loose": fraction_loose,
        "n_state_genes_medium": n_medium,
        "fraction_state_genes_medium": fraction_medium,
        "n_state_genes_strict": n_strict,
        "fraction_state_genes_strict": fraction_strict,
    }


def _cosine_topk_union(a: np.ndarray, b: np.ndarray, *, k: int = MSIGDB_TOPK) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or a.size != b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    n_select = min(int(k), int(a.size))
    if n_select <= 0:
        return float("nan")
    top_a = np.argpartition(np.abs(a), -n_select)[-n_select:]
    top_b = np.argpartition(np.abs(b), -n_select)[-n_select:]
    selected = np.unique(np.concatenate([top_a, top_b]))
    return _cosine(a[selected], b[selected])


def _safe_quantile(values: list[float], quantile: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, float(quantile)))


def _msigdb_required_passes(n_blocks: int) -> int:
    if n_blocks <= 0:
        return 0
    return int(max(1, min(n_blocks, np.ceil(MSIGDB_MAJORITY_FRAC * n_blocks))))


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


def _complete_link_components(nodes: List[str], pass_edges_set: set[Tuple[str, str]]) -> List[List[str]]:
    """
    Deterministic all-pairs clique cover:
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
    components: List[List[str]] = []

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

        components.append(sorted(clique))
        remaining -= clique

    return components


def compact_clusters_by_multiview_agreement(
    *,
    adata: ad.AnnData,
    round_snapshot: Dict[str, Any],
    celltypist_obs_key: str,
    min_cells: int = 0,
    zscore_scope: str = "global",
    grouping: str = "complete_link",
    progeny_threshold_cap: float = 0.98,
    dorothea_threshold_cap: float = 0.98,
    msigdb_threshold_cap: float = 0.98,
    msigdb_threshold_cap_by_gmt: Optional[Dict[str, float]] = None,
    transcriptomic_source: str = "auto",
    transcriptomic_n_features: int = DEFAULT_TRANSCRIPTOMIC_N_FEATURES,
    transcriptomic_threshold_cap: float = 0.99,
    state_divergence_log2fc_threshold: float = DEFAULT_STATE_LOG2FC_THRESHOLD,
    state_divergence_detection_delta_threshold: float = DEFAULT_STATE_DETECTION_DELTA_THRESHOLD,
    state_divergence_max_fraction: float = DEFAULT_STATE_MAX_FRACTION,
    adaptive_quantile: float = 0.90,
    msigdb_required: bool = True,
) -> CompactionOutputs:
    """Return conservative compaction groups from trusted multiview evidence."""
    if str(zscore_scope).strip().lower() != "global":
        raise ValueError("Compaction zscore_scope must be 'global'.")
    grouping = str(grouping).strip().lower()
    if grouping == "clique":
        grouping = "complete_link"
    if grouping not in {"complete_link", "connected_components"}:
        raise ValueError("grouping must be 'complete_link' or legacy 'connected_components'")
    if not 0.0 < float(adaptive_quantile) <= 1.0:
        raise ValueError("adaptive_quantile must be in (0, 1]")
    if not FLOOR_PROGENY <= float(progeny_threshold_cap) <= 1.0:
        raise ValueError(f"progeny_threshold_cap must be in [{FLOOR_PROGENY}, 1]")
    if not FLOOR_DOROTHEA <= float(dorothea_threshold_cap) <= 1.0:
        raise ValueError(f"dorothea_threshold_cap must be in [{FLOOR_DOROTHEA}, 1]")
    if not FLOOR_TRANSCRIPTOMIC <= float(transcriptomic_threshold_cap) <= 1.0:
        raise ValueError(
            f"transcriptomic_threshold_cap must be in [{FLOOR_TRANSCRIPTOMIC}, 1]"
        )
    if float(state_divergence_log2fc_threshold) <= 0.0:
        raise ValueError("state_divergence_log2fc_threshold must be > 0")
    if not 0.0 <= float(state_divergence_detection_delta_threshold) <= 1.0:
        raise ValueError("state_divergence_detection_delta_threshold must be in [0, 1]")
    if not 0.0 <= float(state_divergence_max_fraction) <= 1.0:
        raise ValueError("state_divergence_max_fraction must be in [0, 1]")
    minimum_default_msigdb_cap = max(FLOOR_MSIGDB_DEFAULT, *MSIGDB_FLOOR_BY_GMT.values())
    if not minimum_default_msigdb_cap <= float(msigdb_threshold_cap) <= 1.0:
        raise ValueError(f"msigdb_threshold_cap must be in [{minimum_default_msigdb_cap}, 1]")

    msigdb_threshold_cap_by_gmt = dict(msigdb_threshold_cap_by_gmt or {})
    for gmt, value in msigdb_threshold_cap_by_gmt.items():
        floor = MSIGDB_FLOOR_BY_GMT.get(str(gmt).upper(), FLOOR_MSIGDB_DEFAULT)
        if not floor <= float(value) <= 1.0:
            raise ValueError(f"MSigDB cap for {gmt!r} must be in [{floor}, 1]")

    labels_obs_key = round_snapshot.get("labels_obs_key")
    if not labels_obs_key or labels_obs_key not in adata.obs:
        raise KeyError("round_snapshot['labels_obs_key'] missing or not in adata.obs")
    cluster_per_cell = adata.obs[str(labels_obs_key)].astype(str)
    cluster_sizes = cluster_per_cell.value_counts()
    all_clusters = sorted(cluster_sizes.index.astype(str).tolist())
    transcriptome, state_evidence, transcriptome_audit, transcriptomic_provenance = (
        _prepare_transcriptomic_view(
            adata,
            cluster_per_cell=cluster_per_cell,
            all_clusters=all_clusters,
            source=transcriptomic_source,
            n_features=transcriptomic_n_features,
        )
    )
    transcriptomic_provenance.update(
        {
            "decision_rule": "one_sided_state_divergence_veto",
            "pearson_role": "diagnostic_only",
            "state_log2fc_threshold": float(state_divergence_log2fc_threshold),
            "state_detection_delta_threshold": float(
                state_divergence_detection_delta_threshold
            ),
            "state_max_affected_fraction": float(state_divergence_max_fraction),
            "state_boundary_policy": "affected_fraction_lte_threshold_passes",
            "diagnostic_envelopes": {
                "loose": {
                    "abs_log2fc": STATE_LOOSE_LOG2FC_THRESHOLD,
                    "detection_delta": STATE_LOOSE_DETECTION_DELTA_THRESHOLD,
                },
                "strict": {
                    "abs_log2fc": STATE_STRICT_LOG2FC_THRESHOLD,
                    "detection_delta": STATE_STRICT_DETECTION_DELTA_THRESHOLD,
                },
            },
        }
    )

    annotation = round_snapshot.get("annotation", {})
    if not isinstance(annotation, dict):
        raise TypeError("round_snapshot['annotation'] must be a dictionary")
    celltypist_cluster_key = annotation.get("celltypist_cluster_key")
    if not isinstance(celltypist_cluster_key, str) or celltypist_cluster_key not in adata.obs:
        raise KeyError("Compaction requires the round-specific CellTypist cluster label column.")
    audit = annotation.get("celltypist_cluster_label_audit")
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        raise KeyError("Compaction requires annotation['celltypist_cluster_label_audit'].")
    required_audit_columns = {
        "cluster", "n_total", "n_confident", "confident_fraction", "winning_label",
        "winning_fraction", "runner_up_fraction", "assigned_label", "status",
    }
    missing_audit_columns = sorted(required_audit_columns - set(audit.columns))
    if missing_audit_columns:
        raise ValueError(f"CellTypist cluster audit is missing columns: {missing_audit_columns}")
    if audit["cluster"].astype(str).duplicated().any():
        raise ValueError("CellTypist cluster audit contains duplicate cluster rows.")
    audit_by_cluster = audit.assign(cluster=audit["cluster"].astype(str)).set_index("cluster")

    decoupler = round_snapshot.get("decoupler", {})
    if not isinstance(decoupler, dict):
        raise TypeError("round_snapshot['decoupler'] must be a dictionary")
    progeny, progeny_audit = _prepare_activity_view(
        "PROGENy", decoupler.get("progeny", {}).get("activity"),
        all_clusters=all_clusters, required=True,
    )
    dorothea, dorothea_audit = _prepare_activity_view(
        "DoRothEA", decoupler.get("dorothea", {}).get("activity"),
        all_clusters=all_clusters, required=True,
    )
    assert progeny is not None and dorothea is not None

    raw_msigdb = decoupler.get("msigdb", {}).get("activity_by_gmt")
    if raw_msigdb is None:
        raw_msigdb = {}
    if not isinstance(raw_msigdb, dict):
        raise TypeError("MSigDB activity_by_gmt must be a dictionary when present.")
    msigdb: dict[str, pd.DataFrame] = {}
    view_audits = [transcriptome_audit, progeny_audit, dorothea_audit]
    for gmt, value in sorted(raw_msigdb.items(), key=lambda item: str(item[0])):
        prepared, block_audit = _prepare_activity_view(
            f"MSigDB:{gmt}", value, all_clusters=all_clusters, required=msigdb_required,
        )
        view_audits.append(block_audit)
        if prepared is not None:
            msigdb[str(gmt)] = prepared
    if msigdb_required and not msigdb:
        raise ValueError("msigdb_required=True but no valid MSigDB activity blocks are available.")

    complete_by_view: dict[str, set[str]] = {
        "Transcriptome": set(transcriptome.dropna(axis=0, how="any").index.astype(str)),
        "PROGENy": set(progeny.dropna(axis=0, how="any").index.astype(str)),
        "DoRothEA": set(dorothea.dropna(axis=0, how="any").index.astype(str)),
    }
    for gmt, frame in msigdb.items():
        complete_by_view[f"MSigDB:{gmt}"] = set(frame.dropna(axis=0, how="any").index.astype(str))

    eligibility_rows: list[dict[str, Any]] = []
    label_to_clusters: dict[str, list[str]] = {}
    for cluster in all_clusters:
        reasons: list[str] = []
        audit_row = audit_by_cluster.loc[cluster] if cluster in audit_by_cluster.index else None
        assigned_label = "UNKNOWN"
        if audit_row is None:
            reasons.append("missing_celltypist_audit")
            audit_values = {key: np.nan for key in required_audit_columns - {"cluster"}}
        else:
            audit_values = audit_row.to_dict()
            assigned_label = _normalize_celltypist_label(audit_values.get("assigned_label"))
            audit_n_total = pd.to_numeric(audit_values.get("n_total"), errors="coerce")
            if not np.isfinite(audit_n_total) or int(audit_n_total) != int(cluster_sizes.get(cluster, 0)):
                reasons.append("celltypist_audit_size_mismatch")
            if str(audit_values.get("status")) != "assigned":
                reasons.append(f"celltypist_{audit_values.get('status')}")
            if assigned_label == "UNKNOWN":
                reasons.append("celltypist_unknown")

        cluster_labels = adata.obs.loc[cluster_per_cell == cluster, celltypist_cluster_key]
        observed_labels = {
            _normalize_celltypist_label(value) for value in cluster_labels.astype(object).tolist()
        }
        if len(observed_labels) != 1 or assigned_label not in observed_labels:
            reasons.append("celltypist_cluster_label_mismatch")
        if int(min_cells) > 0 and int(cluster_sizes.get(cluster, 0)) < int(min_cells):
            reasons.append("below_compact_min_cells")
        for view_name in ("Transcriptome", "PROGENy", "DoRothEA"):
            if cluster not in complete_by_view[view_name]:
                reasons.append(f"missing_{view_name.lower()}_activity")
        if msigdb_required:
            for gmt in msigdb:
                if cluster not in complete_by_view[f"MSigDB:{gmt}"]:
                    reasons.append(f"missing_msigdb_{gmt}_activity")

        reasons = sorted(set(reasons))
        eligible = not reasons
        row = {
            "cluster": cluster,
            "n_cells": int(cluster_sizes.get(cluster, 0)),
            "assigned_label": assigned_label,
            "n_confident": audit_values.get("n_confident", np.nan),
            "confident_fraction": audit_values.get("confident_fraction", np.nan),
            "winning_label": audit_values.get("winning_label", ""),
            "winning_fraction": audit_values.get("winning_fraction", np.nan),
            "runner_up_fraction": audit_values.get("runner_up_fraction", np.nan),
            "celltypist_status": audit_values.get("status", "missing"),
            "required_activity_complete": not any("activity" in reason for reason in reasons),
            "eligible": bool(eligible),
            "exclusion_reasons": ";".join(reasons),
        }
        for view_name, complete_clusters in complete_by_view.items():
            row[f"complete__{view_name}"] = cluster in complete_clusters
        eligibility_rows.append(row)
        if eligible:
            label_to_clusters.setdefault(assigned_label, []).append(cluster)

    cluster_eligibility = pd.DataFrame(eligibility_rows)
    eligibility_by_cluster = cluster_eligibility.set_index("cluster")
    edge_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    adjacency: dict[str, list[tuple[str, str]]] = {}
    components: list[list[str]] = []
    decision_log: list[dict[str, Any]] = []

    def floor_for_gmt(gmt: str) -> float:
        return float(MSIGDB_FLOOR_BY_GMT.get(str(gmt).upper(), FLOOR_MSIGDB_DEFAULT))

    def cap_for_gmt(gmt: str) -> float:
        return float(msigdb_threshold_cap_by_gmt.get(str(gmt), msigdb_threshold_cap))

    for celltypist_label, clusters in sorted(label_to_clusters.items()):
        clusters = sorted(clusters)
        if len(clusters) < 2:
            components.extend([[cluster] for cluster in clusters])
            adjacency[celltypist_label] = []
            continue

        pair_indices = [(a, b) for i, a in enumerate(clusters) for b in clusters[i + 1:]]
        similarities: dict[str, dict[tuple[str, str], float]] = {
            "Transcriptome": {
                pair: _pearson(
                    transcriptome.loc[pair[0]].to_numpy(),
                    transcriptome.loc[pair[1]].to_numpy(),
                )
                for pair in pair_indices
            },
            "PROGENy": {
                pair: _cosine(progeny.loc[pair[0]].to_numpy(), progeny.loc[pair[1]].to_numpy())
                for pair in pair_indices
            },
            "DoRothEA": {
                pair: _cosine(dorothea.loc[pair[0]].to_numpy(), dorothea.loc[pair[1]].to_numpy())
                for pair in pair_indices
            },
        }
        for gmt, frame in msigdb.items():
            similarities[f"MSigDB:{gmt}"] = {
                pair: _cosine_topk_union(
                    frame.loc[pair[0]].to_numpy(), frame.loc[pair[1]].to_numpy()
                )
                for pair in pair_indices
            }

        adaptive = len(clusters) >= ADAPTIVE_MIN_GROUP_SIZE
        view_thresholds: dict[str, dict[str, float | bool]] = {}
        threshold_specs = {
            "Transcriptome": (FLOOR_TRANSCRIPTOMIC, float(transcriptomic_threshold_cap)),
            "PROGENy": (FLOOR_PROGENY, float(progeny_threshold_cap)),
            "DoRothEA": (FLOOR_DOROTHEA, float(dorothea_threshold_cap)),
            **{
                f"MSigDB:{gmt}": (floor_for_gmt(gmt), cap_for_gmt(gmt))
                for gmt in msigdb
            },
        }
        for view_name, (floor, cap) in threshold_specs.items():
            relative = (
                _safe_quantile(list(similarities[view_name].values()), adaptive_quantile)
                if adaptive else float("nan")
            )
            uncapped = max(float(floor), float(relative)) if np.isfinite(relative) else float(floor)
            effective = min(float(cap), uncapped)
            view_thresholds[view_name] = {
                "floor": float(floor), "relative": float(relative),
                "cap": float(cap), "effective": float(effective), "adaptive": bool(adaptive),
            }
            threshold_rows.append({
                "celltypist_label": celltypist_label,
                "n_clusters": len(clusters),
                "view": view_name,
                "decision_role": (
                    "diagnostic_only" if view_name == "Transcriptome" else "required"
                ),
                "floor": float(floor),
                "adaptive_used": bool(adaptive),
                "adaptive_quantile": float(adaptive_quantile) if adaptive else np.nan,
                "adaptive_value": float(relative),
                "cap": float(cap),
                "effective_threshold": float(effective),
            })

        passed_edges: list[tuple[str, str]] = []
        for a, b in pair_indices:
            state_metrics = _state_divergence_metrics(
                state_evidence,
                cluster_a=a,
                cluster_b=b,
                log2fc_threshold=float(state_divergence_log2fc_threshold),
                detection_delta_threshold=float(
                    state_divergence_detection_delta_threshold
                ),
            )
            pass_transcriptome = (
                similarities["Transcriptome"][(a, b)]
                >= view_thresholds["Transcriptome"]["effective"]
            )
            pass_progeny = similarities["PROGENy"][(a, b)] >= view_thresholds["PROGENy"]["effective"]
            pass_dorothea = similarities["DoRothEA"][(a, b)] >= view_thresholds["DoRothEA"]["effective"]
            msigdb_passes = {
                gmt: similarities[f"MSigDB:{gmt}"][(a, b)]
                >= view_thresholds[f"MSigDB:{gmt}"]["effective"]
                for gmt in msigdb
            }
            n_msigdb_passed = sum(msigdb_passes.values())
            n_msigdb_required = _msigdb_required_passes(len(msigdb))
            pass_msigdb = n_msigdb_passed >= n_msigdb_required if msigdb_required else True
            pass_state_divergence = bool(
                state_metrics["fraction_state_genes_medium"]
                <= float(state_divergence_max_fraction)
            )
            state_fraction = float(state_metrics["fraction_state_genes_medium"])
            if float(state_divergence_max_fraction) > 0.0:
                state_decision_margin = (
                    float(state_divergence_max_fraction) - state_fraction
                ) / float(state_divergence_max_fraction)
            else:
                state_decision_margin = 0.0 if state_fraction == 0.0 else -state_fraction
            pass_all = bool(
                pass_state_divergence and pass_progeny and pass_dorothea and pass_msigdb
            )
            if pass_all:
                passed_edges.append((a, b))

            required_margins = [
                state_decision_margin,
                similarities["PROGENy"][(a, b)] - float(view_thresholds["PROGENy"]["effective"]),
                similarities["DoRothEA"][(a, b)] - float(view_thresholds["DoRothEA"]["effective"]),
            ]
            msigdb_margins = sorted(
                (
                    similarities[f"MSigDB:{gmt}"][(a, b)]
                    - float(view_thresholds[f"MSigDB:{gmt}"]["effective"])
                    for gmt in msigdb
                ),
                reverse=True,
            )
            msigdb_decision_margin = float("nan")
            if msigdb_required:
                msigdb_decision_margin = float(msigdb_margins[n_msigdb_required - 1])
                required_margins.append(msigdb_decision_margin)
            row: dict[str, Any] = {
                "celltypist_label": celltypist_label,
                "a": a, "b": b,
                "n_a": int(cluster_sizes[a]), "n_b": int(cluster_sizes[b]),
                "confident_fraction_a": eligibility_by_cluster.loc[a, "confident_fraction"],
                "confident_fraction_b": eligibility_by_cluster.loc[b, "confident_fraction"],
                "winning_fraction_a": eligibility_by_cluster.loc[a, "winning_fraction"],
                "winning_fraction_b": eligibility_by_cluster.loc[b, "winning_fraction"],
                "sim_transcriptome": similarities["Transcriptome"][(a, b)],
                "floor_transcriptome": FLOOR_TRANSCRIPTOMIC,
                "cap_transcriptome": float(transcriptomic_threshold_cap),
                "threshold_transcriptome": view_thresholds["Transcriptome"]["effective"],
                "pass_transcriptome": bool(pass_transcriptome),
                "pass_transcriptome_diagnostic": bool(pass_transcriptome),
                "transcriptome_decision_role": "diagnostic_only",
                **state_metrics,
                "state_divergence_log2fc_threshold": float(
                    state_divergence_log2fc_threshold
                ),
                "state_divergence_detection_delta_threshold": float(
                    state_divergence_detection_delta_threshold
                ),
                "state_divergence_max_fraction": float(state_divergence_max_fraction),
                "state_divergence_veto": not pass_state_divergence,
                "pass_state_divergence": pass_state_divergence,
                "state_divergence_decision_margin": state_decision_margin,
                "sim_progeny": similarities["PROGENy"][(a, b)],
                "floor_progeny": FLOOR_PROGENY,
                "cap_progeny": float(progeny_threshold_cap),
                "threshold_progeny": view_thresholds["PROGENy"]["effective"],
                "pass_progeny": bool(pass_progeny),
                "sim_dorothea": similarities["DoRothEA"][(a, b)],
                "floor_dorothea": FLOOR_DOROTHEA,
                "cap_dorothea": float(dorothea_threshold_cap),
                "threshold_dorothea": view_thresholds["DoRothEA"]["effective"],
                "pass_dorothea": bool(pass_dorothea),
                "msigdb_required": bool(msigdb_required),
                "msigdb_majority_fraction": MSIGDB_MAJORITY_FRAC,
                "msigdb_majority_needed": n_msigdb_required,
                "msigdb_majority_passed": n_msigdb_passed,
                "msigdb_decision_margin": msigdb_decision_margin,
                "pass_msigdb": bool(pass_msigdb),
                "pass_all": pass_all,
                "decision_margin": float(min(required_margins)),
                "grouping": grouping,
            }
            for gmt, passed in msigdb_passes.items():
                view_name = f"MSigDB:{gmt}"
                row[f"sim_msigdb__{gmt}"] = similarities[view_name][(a, b)]
                row[f"floor_msigdb__{gmt}"] = view_thresholds[view_name]["floor"]
                row[f"cap_msigdb__{gmt}"] = view_thresholds[view_name]["cap"]
                row[f"threshold_msigdb__{gmt}"] = view_thresholds[view_name]["effective"]
                row[f"pass_msigdb__{gmt}"] = bool(passed)
            edge_rows.append(row)

        adjacency[celltypist_label] = passed_edges
        if grouping == "connected_components":
            label_components = _connected_components(clusters, passed_edges)
        else:
            canonical_edges = {(min(a, b), max(a, b)) for a, b in passed_edges}
            label_components = _complete_link_components(clusters, canonical_edges)
        components.extend(label_components)
        for component in label_components:
            if len(component) > 1:
                decision_log.append({
                    "celltypist_label": celltypist_label,
                    "members": list(component),
                    "n_members": len(component),
                    "reason": (
                        "required state-divergence and activity-view agreement within a trusted "
                        "CellTypist label; Pearson concordance retained as a diagnostic"
                    ),
                    "grouping": grouping,
                })

    covered = {cluster for component in components for cluster in component}
    components.extend([[cluster] for cluster in all_clusters if cluster not in covered])

    def component_size(component: list[str]) -> int:
        return int(sum(int(cluster_sizes[cluster]) for cluster in component))

    sorted_components = sorted(components, key=lambda item: (-component_size(item), item))
    cluster_id_map: dict[str, str] = {}
    reverse_map: dict[str, list[str]] = {}
    membership_rows: list[dict[str, Any]] = []
    eligibility_lookup = cluster_eligibility.set_index("cluster")
    for index, members in enumerate(sorted_components):
        compacted_cluster = f"C{index:02d}"
        reverse_map[compacted_cluster] = list(members)
        for cluster in members:
            cluster_id_map[cluster] = compacted_cluster
            membership_rows.append({
                "compacted_cluster": compacted_cluster,
                "parent_cluster": cluster,
                "parent_n_cells": int(cluster_sizes[cluster]),
                "compacted_n_cells": component_size(members),
                "n_parent_clusters": len(members),
                "did_merge": len(members) > 1,
                "celltypist_label": eligibility_lookup.loc[cluster, "assigned_label"],
                "parent_eligible": bool(eligibility_lookup.loc[cluster, "eligible"]),
                "parent_exclusion_reasons": eligibility_lookup.loc[cluster, "exclusion_reasons"],
            })

    return CompactionOutputs(
        components=[list(members) for members in reverse_map.values()],
        cluster_id_map=cluster_id_map,
        reverse_map=reverse_map,
        cluster_renumbering={cluster: cluster for cluster in reverse_map},
        edges=pd.DataFrame(edge_rows),
        adjacency=adjacency,
        decision_log=decision_log,
        cluster_eligibility=cluster_eligibility,
        group_membership=pd.DataFrame(membership_rows),
        thresholds_by_label=pd.DataFrame(threshold_rows),
        view_audit=pd.DataFrame(view_audits),
        transcriptomic_provenance=transcriptomic_provenance,
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
    min_cells: int = 0,
    zscore_scope: str = "global",
    grouping: str = "complete_link",
    progeny_threshold_cap: float = 0.98,
    dorothea_threshold_cap: float = 0.98,
    msigdb_threshold_cap: float = 0.98,
    msigdb_threshold_cap_by_gmt: dict[str, float] | None = None,
    transcriptomic_source: str = "auto",
    transcriptomic_n_features: int = DEFAULT_TRANSCRIPTOMIC_N_FEATURES,
    transcriptomic_threshold_cap: float = 0.99,
    state_divergence_log2fc_threshold: float = DEFAULT_STATE_LOG2FC_THRESHOLD,
    state_divergence_detection_delta_threshold: float = DEFAULT_STATE_DETECTION_DELTA_THRESHOLD,
    state_divergence_max_fraction: float = DEFAULT_STATE_MAX_FRACTION,
    adaptive_quantile: float = 0.90,
    msigdb_required: bool = True,
) -> None:
    """Create and activate a fully audited compacted child round."""
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

    try:
        ensure_round_msigdb_activity_by_gmt(parent)
    except (KeyError, TypeError, ValueError) as exc:
        if msigdb_required:
            raise
        LOGGER.info("Compaction: optional MSigDB GMT activity is unavailable: %s", exc)

    outputs = compact_clusters_by_multiview_agreement(
        adata=adata,
        round_snapshot=parent,
        celltypist_obs_key=celltypist_obs_key,
        min_cells=min_cells,
        zscore_scope=zscore_scope,
        grouping=grouping,
        progeny_threshold_cap=progeny_threshold_cap,
        dorothea_threshold_cap=dorothea_threshold_cap,
        msigdb_threshold_cap=msigdb_threshold_cap,
        msigdb_threshold_cap_by_gmt=msigdb_threshold_cap_by_gmt,
        transcriptomic_source=transcriptomic_source,
        transcriptomic_n_features=transcriptomic_n_features,
        transcriptomic_threshold_cap=transcriptomic_threshold_cap,
        state_divergence_log2fc_threshold=state_divergence_log2fc_threshold,
        state_divergence_detection_delta_threshold=(
            state_divergence_detection_delta_threshold
        ),
        state_divergence_max_fraction=state_divergence_max_fraction,
        adaptive_quantile=adaptive_quantile,
        msigdb_required=msigdb_required,
    )

    did_merge = any(len(members) > 1 for members in outputs.reverse_map.values())

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

    decoupler = parent.get("decoupler", {})
    upstream_provenance = {}
    if isinstance(decoupler, dict):
        for view_name in ("progeny", "dorothea", "msigdb"):
            payload = decoupler.get(view_name, {})
            if isinstance(payload, dict):
                upstream_provenance[view_name] = dict(payload.get("method_provenance", {}))

    compacting_payload = {
        "method_identity": COMPACTION_METHOD_IDENTITY,
        "parent_round_id": str(parent_round_id),
        "within_celltypist_label_only": True,
        "celltypist_cell_key": str(celltypist_obs_key),
        "celltypist_cluster_key": parent.get("annotation", {}).get("celltypist_cluster_key"),
        "unknown_policy": "singleton",
        "ineligible_policy": "singleton",
        "no_op_policy": "retain_active_compacted_child",
        "similarity_metric": "cosine_after_global_feature_zscore",
        "msigdb_similarity_metric": f"top_{MSIGDB_TOPK}_union_cosine_after_global_feature_zscore",
        "transcriptomic_decision_rule": "one_sided_state_divergence_veto",
        "transcriptomic_similarity_metric": "pearson_cluster_pseudobulk_diagnostic_only",
        "transcriptomic_provenance": outputs.transcriptomic_provenance,
        "params": {
            "min_cells": int(min_cells),
            "zscore_scope": str(zscore_scope),
            "grouping": "complete_link" if str(grouping) == "clique" else str(grouping),
            "adaptive_min_group_size": ADAPTIVE_MIN_GROUP_SIZE,
            "adaptive_quantile": float(adaptive_quantile),
            "transcriptomic_source": str(transcriptomic_source),
            "transcriptomic_n_features": int(transcriptomic_n_features),
            "state_divergence_log2fc_threshold": float(
                state_divergence_log2fc_threshold
            ),
            "state_divergence_detection_delta_threshold": float(
                state_divergence_detection_delta_threshold
            ),
            "state_divergence_max_fraction": float(state_divergence_max_fraction),
            "msigdb_required": bool(msigdb_required),
            "msigdb_majority_fraction": MSIGDB_MAJORITY_FRAC,
        },
        "threshold_policy": {
            "progeny_floor": FLOOR_PROGENY,
            "dorothea_floor": FLOOR_DOROTHEA,
            "transcriptomic_floor": FLOOR_TRANSCRIPTOMIC,
            "msigdb_floor_default": FLOOR_MSIGDB_DEFAULT,
            "msigdb_floor_by_gmt": dict(MSIGDB_FLOOR_BY_GMT),
            "progeny_cap": float(progeny_threshold_cap),
            "dorothea_cap": float(dorothea_threshold_cap),
            "transcriptomic_cap": float(transcriptomic_threshold_cap),
            "transcriptomic_pearson_role": "diagnostic_only",
            "state_divergence_log2fc_threshold": float(
                state_divergence_log2fc_threshold
            ),
            "state_divergence_detection_delta_threshold": float(
                state_divergence_detection_delta_threshold
            ),
            "state_divergence_max_fraction": float(state_divergence_max_fraction),
            "state_divergence_boundary_policy": "affected_fraction_lte_threshold_passes",
            "msigdb_cap_default": float(msigdb_threshold_cap),
            "msigdb_cap_by_gmt": dict(msigdb_threshold_cap_by_gmt or {}),
        },
        "upstream_method_provenance": upstream_provenance,
        "view_audit": outputs.view_audit,
        "cluster_eligibility": outputs.cluster_eligibility,
        "thresholds_by_label": outputs.thresholds_by_label,
        "pairwise": {
            "edges": outputs.edges,
            "adjacency": outputs.adjacency,
        },
        "components": outputs.components,
        "group_membership": outputs.group_membership,
        "decision_log": outputs.decision_log,
        "did_merge": bool(did_merge),
        "reverse_map": outputs.reverse_map,
    }

    cfg_snapshot = None
    try:
        if hasattr(cfg, "model_dump"):
            cfg_snapshot = cfg.model_dump(mode="json")
        elif hasattr(cfg, "__dataclass_fields__"):
            from dataclasses import asdict

            cfg_snapshot = asdict(cfg)
    except Exception:
        cfg_snapshot = None

    _create_shallow_round_from_parent(
        adata,
        parent_round_id=str(parent_round_id),
        round_name=str(new_round_id).split("_", 1)[1] if "_" in str(new_round_id) else "compacted",
        new_round_id=str(new_round_id),
        round_type="compaction",
        kind="COMPACTED",
        notes=notes,
        set_active=True,
        cluster_key=str(parent_cluster_key),
        labels_obs_key=str(labels_obs_key_new),
        best_resolution=None,
        sweep=None,
        cfg_snapshot=cfg_snapshot,
        cluster_id_map=dict(outputs.cluster_id_map),
        cluster_renumbering=dict(outputs.cluster_renumbering),
        compacting=compacting_payload,
        inherit_fields=(),
    )


def export_compaction_audit_tables(
    adata: ad.AnnData,
    *,
    round_id: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the stored compaction review tables without recomputing decisions."""
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise KeyError(f"Compaction round {round_id!r} was not found.")
    compacting = rounds[round_id].get("compacting", {})
    if not isinstance(compacting, dict):
        raise KeyError(f"Round {round_id!r} has no compaction payload.")

    tables = {
        "view_audit": compacting.get("view_audit"),
        "cluster_eligibility": compacting.get("cluster_eligibility"),
        "thresholds_by_label": compacting.get("thresholds_by_label"),
        "pairwise_evidence": compacting.get("pairwise", {}).get("edges"),
        "group_membership": compacting.get("group_membership"),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame() if table is None else pd.DataFrame(table)
        path = target / f"{name}.tsv"
        table.to_csv(path, sep="\t", index=False)
        written[name] = path
    return written
