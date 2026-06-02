# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
import importlib
import inspect
import json
import os
import configparser
import re
import sys
import threading
import time
import traceback
import gc
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import stats as sstats

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .logging_utils import init_logging
from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
    _set_blas_threads,
    compute_markers_celllevel,
    de_cluster_vs_rest_pseudobulk,
    pydeseq2_supports_interaction_by_name,
)
from .composition_utils import (
    _resolve_active_cluster_key,
    prepare_counts_and_metadata,
    _choose_reference_most_stable,
    _validate_min_samples_per_level,
    run_sccoda_model,
    run_glm_composition,
    run_clr_mannwhitney,
    run_graph_da,
    _standardize_composition_results,
    _build_composition_consensus_summary,
    _MIN_GLM_SAMPLES_PER_LEVEL,
)
from .annotation_utils import (
    run_decoupler_for_round,
    _publish_decoupler_from_round_to_top_level,
    _apply_gene_filters_to_var_names,
    _parse_gene_filter_entry,
    _normalize_gene_filter_expr,
    _prepare_decoupler_grouping,
    clear_top_level_decoupler_state,
)

LOGGER = logging.getLogger(__name__)
_SCCODA_LOCK = threading.Lock()
_LIANA_DEFAULT_AGGREGATE_METHODS = ("cellphonedb", "natmi", "sca", "logfc")
_CELLCHATDB_INTERACTION_ANNOTATIONS = (
    Path(__file__).resolve().parent / "resources" / "cellchatdb_interaction_annotations.tsv"
)
_NICHENET_R_HELPER = (
    Path(__file__).resolve().parent / "resources" / "run_nichenet_sender_focused.R"
)
_MEBOCOST_GIT_SPEC = "git+https://github.com/kaifuchenlab/MEBOCOST.git"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NICHENET_R_LIB_DIR = (
    Path.home() / "Library" / "Caches" / "scOmnom" / "r-libs" / "nichenet"
    if sys.platform == "darwin"
    else Path.home() / ".cache" / "scOmnom" / "r-libs" / "nichenet"
)
_NICHENET_R_BOOTSTRAP_CRAN_PACKAGES = ("DiceKriging", "emoa", "fdrtool", "mlrMBO")
_MEBOCOST_CACHE_ROOT = (
    Path.home() / "Library" / "Caches" / "scOmnom" / "mebocost"
    if sys.platform == "darwin"
    else Path.home() / ".cache" / "scOmnom" / "mebocost"
)
_MEBOCOST_RESOURCE_REPO = _MEBOCOST_CACHE_ROOT / "MEBOCOST"
_MEBOCOST_RESOURCE_CONFIG = _MEBOCOST_CACHE_ROOT / "mebocost.conf"

# -----------------------------------------------------------------------------
# Internal policy guard
# -----------------------------------------------------------------------------
_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK = 6
_MIN_SAMPLES_PER_LEVEL_COMPOSITION = 2


def _matplotlib_backend_supports_threaded_plotting() -> bool:
    try:
        backend = str(plot_utils.mpl.get_backend() or "").strip().lower()
    except Exception:
        return False
    if not backend:
        return False
    safe = (
        "agg",
        "pdf",
        "svg",
        "ps",
        "pgf",
        "cairo",
    )
    return any(token in backend for token in safe) and "macosx" not in backend


def _match_cluster_color(label: str, color_map: dict[str, str]) -> Optional[str]:
    lab = str(label)
    if lab in color_map:
        return color_map[lab]
    token = plot_utils._extract_cnn_token(lab)
    if token in color_map:
        return color_map[token]
    if token:
        # Support maps keyed by verbose labels like "C02: Hepatocytes".
        for key, color in color_map.items():
            if plot_utils._extract_cnn_token(str(key)) == token:
                return color
    digits = re.sub(r"[^0-9]", "", lab)
    if digits:
        try:
            num = int(digits)
        except ValueError:
            num = None
        if num is not None:
            candidates = [digits, str(num), f"C{num}", f"C{num:02d}"]
            for cand in candidates:
                if cand in color_map:
                    return color_map[cand]
    return None


def _resolve_cluster_colors(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    labels: Sequence[str],
    round_id: Optional[str],
) -> list:
    keys = [str(cluster_key)]
    if round_id:
        pretty_key = f"cluster_label__{round_id}"
        if pretty_key in adata.obs:
            keys.append(pretty_key)
    if "cluster_label" in adata.obs:
        keys.append("cluster_label")

    color_map: dict[str, str] = {}
    for key in keys:
        cm = plot_utils._cluster_color_map(adata, key)
        if cm:
            color_map = cm
            break

    labels = [str(l) for l in labels]
    colors: list[Optional[str]] = []
    if color_map:
        for lab in labels:
            colors.append(_match_cluster_color(lab, color_map))

    if not colors or any(c is None for c in colors):
        import matplotlib.pyplot as plt

        tab20 = plt.colormaps["tab20"]
        tab20b = plt.colormaps["tab20b"]
        unique_labels = list(dict.fromkeys(labels))
        palette = [
            tab20(i / 20) if i < 20 else tab20b((i - 20) / 20)
            for i in range(len(unique_labels))
        ]
        palette_map = dict(zip(unique_labels, palette))
        if not colors:
            return [palette_map[l] for l in labels]
        filled = []
        for lab, col in zip(labels, colors):
            filled.append(col if col is not None else palette_map[lab])
        return filled
    return colors  # type: ignore[return-value]


def _prune_uns_de(adata: ad.AnnData, store_key: str, *, top_n: int = 50, decoupler_top_n: int = 20) -> None:
    """
    Reduce memory footprint of adata.uns[store_key] by keeping summaries
    and dropping large per-gene result tables.
    """
    if store_key not in adata.uns or not isinstance(adata.uns.get(store_key), dict):
        return

    block = adata.uns.get(store_key, {})
    top_n = int(top_n)
    decoupler_top_n = int(decoupler_top_n)

    def _top_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()
        n = int(max(1, n))
        d = df.copy()
        cols = list(d.columns)
        if "gene" in cols:
            d["gene"] = d["gene"].astype(str)
        if "cell_wilcoxon_padj" in cols:
            lfc_col = "cell_wilcoxon_logfc" if "cell_wilcoxon_logfc" in cols else None
            if lfc_col:
                d["_abs_lfc"] = d[lfc_col].abs()
                d = d.sort_values(by=["cell_wilcoxon_padj", "_abs_lfc"], ascending=[True, False], kind="mergesort")
                d = d.drop(columns=["_abs_lfc"], errors="ignore")
            else:
                d = d.sort_values(by=["cell_wilcoxon_padj"], ascending=[True], kind="mergesort")
        elif "padj" in cols:
            lfc_col = "log2FoldChange" if "log2FoldChange" in cols else None
            if lfc_col:
                d["_abs_lfc"] = d[lfc_col].abs()
                d = d.sort_values(by=["padj", "_abs_lfc"], ascending=[True, False], kind="mergesort")
                d = d.drop(columns=["_abs_lfc"], errors="ignore")
            else:
                d = d.sort_values(by=["padj"], ascending=[True], kind="mergesort")
        elif "pval" in cols:
            d = d.sort_values(by=["pval"], ascending=[True], kind="mergesort")
        elif "score" in cols:
            d = d.sort_values(by=["score"], ascending=[False], kind="mergesort")
        elif "cell_wilcoxon_score" in cols:
            d = d.sort_values(by=["cell_wilcoxon_score"], ascending=[False], kind="mergesort")
        d = d.head(n)
        keep_cols = [
            "gene",
            "cell_wilcoxon_logfc",
            "cell_wilcoxon_padj",
            "cell_wilcoxon_pval",
            "cell_wilcoxon_score",
            "cell_logreg_coef",
            "cell_logreg_score",
            "cell_wilcoxon_pts",
            "cell_wilcoxon_pts_rest",
            "pb_log2fc",
            "log2FoldChange",
            "padj",
            "pval",
            "score",
            "stat",
        ]
        keep = [c for c in keep_cols if c in d.columns]
        if keep:
            d = d.loc[:, keep]
        return d

    def _top_activity(activity: pd.DataFrame, n: int) -> pd.DataFrame:
        if activity is None or getattr(activity, "empty", True):
            return pd.DataFrame()
        n = int(max(1, n))
        try:
            scores = activity.abs().mean(axis=0)
            top_cols = scores.sort_values(ascending=False).head(n).index.tolist()
        except Exception:
            top_cols = list(activity.columns[:n])
        return activity.loc[:, top_cols].copy()

    def _top_enrichment_results(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()
        d = df.copy()
        if "cluster" in d.columns:
            d["cluster"] = d["cluster"].astype(str)
        if "pathway" in d.columns:
            d["pathway"] = d["pathway"].astype(str)
        rank_col = None
        for cand in ("decoupler_score", "NES", "ES"):
            if cand in d.columns:
                rank_col = cand
                break
        if rank_col is not None:
            d["_abs_rank"] = pd.to_numeric(d[rank_col], errors="coerce").abs()
        sort_cols: list[str] = []
        ascending: list[bool] = []
        if "cluster" in d.columns:
            sort_cols.append("cluster")
            ascending.append(True)
        if "padj" in d.columns:
            sort_cols.append("padj")
            ascending.append(True)
        if "_abs_rank" in d.columns:
            sort_cols.append("_abs_rank")
            ascending.append(False)
        if "pathway" in d.columns:
            sort_cols.append("pathway")
            ascending.append(True)
        if sort_cols:
            d = d.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        if "cluster" in d.columns:
            d = d.groupby("cluster", sort=False, as_index=False).head(int(max(1, n)))
        else:
            d = d.head(int(max(1, n)))
        keep_cols = [
            "cluster",
            "pathway",
            "decoupler_score",
            "NES",
            "ES",
            "padj",
            "pval",
            "leading_edge_n",
            "leading_edge_preview",
            "direction",
            "decoupler_direction",
            "gsea_direction",
            "gsea_sig",
            "sign_concordant",
            "supported_by_both",
            "joint_rank",
        ]
        keep = [c for c in keep_cols if c in d.columns]
        if "_abs_rank" in d.columns:
            d = d.drop(columns=["_abs_rank"], errors="ignore")
        return d.loc[:, keep].copy() if keep else d.copy()

    # Cluster-vs-rest pseudobulk
    pb_cvr = block.get("pseudobulk_cluster_vs_rest", None)
    if isinstance(pb_cvr, dict):
        if "results" in pb_cvr:
            results = pb_cvr.get("results", {})
            top = {}
            if isinstance(results, dict):
                for cl, df in results.items():
                    if isinstance(df, pd.DataFrame):
                        top[str(cl)] = _top_df(df, top_n)
            if top:
                pb_cvr["top_genes"] = top
            pb_cvr.pop("results", None)

    # Within-cluster pseudobulk multi
    pb_within = block.get("pseudobulk_condition_within_group_multi", None)
    if isinstance(pb_within, dict):
        # keep metadata, drop per-contrast results tables
        keys = list(pb_within.keys())
        for k in keys:
            if isinstance(pb_within.get(k), dict) and "results" in pb_within[k]:
                res = pb_within[k].get("results", None)
                if isinstance(res, pd.DataFrame):
                    pb_within[k]["top_genes"] = _top_df(res, top_n)
                pb_within[k].pop("results", None)

    # Contrast-conditional (cell-level within-cluster)
    cc = block.get("contrast_conditional", None)
    if isinstance(cc, dict):
        if "results" in cc:
            results = cc.get("results", {})
            top = {}
            if isinstance(results, dict):
                for cl, per_contrast in results.items():
                    if not isinstance(per_contrast, dict):
                        continue
                    for pair_key, tables in per_contrast.items():
                        if not isinstance(tables, dict):
                            continue
                        df = tables.get("combined", None)
                        if df is None or getattr(df, "empty", True):
                            df = tables.get("wilcoxon", None)
                        if isinstance(df, pd.DataFrame):
                            top.setdefault(str(cl), {})[str(pair_key)] = _top_df(df, top_n)
            if top:
                cc["top_genes"] = top
            cc.pop("results", None)

    cc_multi = block.get("contrast_conditional_multi", None)
    if isinstance(cc_multi, dict):
        for ck, cc_block in list(cc_multi.items()):
            if not isinstance(cc_block, dict):
                continue
            if "results" in cc_block:
                results = cc_block.get("results", {})
                top = {}
                if isinstance(results, dict):
                    for cl, per_contrast in results.items():
                        if not isinstance(per_contrast, dict):
                            continue
                        for pair_key, tables in per_contrast.items():
                            if not isinstance(tables, dict):
                                continue
                            df = tables.get("combined", None)
                            if df is None or getattr(df, "empty", True):
                                df = tables.get("wilcoxon", None)
                            if isinstance(df, pd.DataFrame):
                                top.setdefault(str(cl), {})[str(pair_key)] = _top_df(df, top_n)
                if top:
                    cc_block["top_genes"] = top
                cc_block.pop("results", None)

    # DE-decoupler payloads
    de_dec = block.get("de_decoupler", None)
    if isinstance(de_dec, dict):
        for ck, per_contrast in list(de_dec.items()):
            if not isinstance(per_contrast, dict):
                continue
            for contrast, payload_by_source in list(per_contrast.items()):
                if not isinstance(payload_by_source, dict):
                    continue
                for source, payload in list(payload_by_source.items()):
                    if not isinstance(payload, dict):
                        continue
                    nets = payload.get("nets", None)
                    if not isinstance(nets, dict):
                        continue
                    for net_name, net_payload in list(nets.items()):
                        if not isinstance(net_payload, dict):
                            continue
                        results = net_payload.get("results", None)
                        if isinstance(results, pd.DataFrame):
                            net_payload["results_top"] = _top_enrichment_results(results, decoupler_top_n)
                            summary = {
                                "n_rows": int(len(results)),
                            }
                            if "cluster" in results.columns:
                                summary["n_clusters"] = int(results["cluster"].astype(str).nunique(dropna=True))
                            if "pathway" in results.columns:
                                summary["n_pathways"] = int(results["pathway"].astype(str).nunique(dropna=True))
                            if "supported_by_both" in results.columns:
                                summary["n_supported_by_both"] = int(pd.Series(results["supported_by_both"]).fillna(False).astype(bool).sum())
                            if "sign_concordant" in results.columns:
                                summary["n_sign_concordant"] = int(pd.Series(results["sign_concordant"]).fillna(False).astype(bool).sum())
                            net_payload["summary"] = summary
                        activity = net_payload.get("activity", None)
                        if isinstance(activity, pd.DataFrame):
                            net_payload["activity_top"] = _top_activity(activity, decoupler_top_n)
                        net_payload.pop("activity", None)
                        net_payload.pop("results", None)
                        net_payload.pop("activity_by_gmt", None)
                        net_payload.pop("feature_meta", None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _choose_counts_layer(
    adata: ad.AnnData,
    *,
    candidates: Sequence[str],
    allow_X_counts: bool,
) -> Optional[str]:
    """
    Choose counts layer by priority. Returns:
      - layer name if found
      - None if falling back to adata.X is allowed
    Raises if nothing found and allow_X_counts=False.
    """
    for layer in candidates:
        if layer in adata.layers:
            return str(layer)

    if allow_X_counts:
        return None

    raise RuntimeError(
        "markers_and_de (pseudobulk): no candidate counts layer found in adata.layers "
        f"(candidates={list(candidates)!r}) and allow_X_counts=False"
    )


def _n_unique_samples(adata: ad.AnnData, sample_key: str) -> int:
    return int(adata.obs[str(sample_key)].astype(str).nunique(dropna=True))


def _safe_combo_token(x: object) -> str:
    s = str(x)
    s = s.replace("/", "_").replace(":", "_")
    return s


def _run_namespace_for_round(
    adata: ad.AnnData,
    *,
    prefix: str,
    round_id: Optional[str],
) -> str:
    rid = round_id
    if rid is None:
        rid0 = adata.uns.get("active_cluster_round", None)
        rid = str(rid0) if rid0 else None
    if rid:
        return f"{str(prefix)}_{_safe_combo_token(str(rid))}"
    return str(prefix)


def _snapshot_top_level_decoupler_state(
    adata: ad.AnnData,
    *,
    keys: Sequence[str] = ("msigdb", "progeny", "dorothea", "pseudobulk"),
) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key in keys:
        if key in adata.uns:
            snapshot[str(key)] = adata.uns[str(key)]
    return snapshot


def _restore_top_level_decoupler_state(
    adata: ad.AnnData,
    snapshot: Mapping[str, object],
    *,
    keys: Sequence[str] = ("msigdb", "progeny", "dorothea", "pseudobulk"),
) -> None:
    for key in keys:
        if key in snapshot:
            adata.uns[str(key)] = snapshot[str(key)]
        else:
            adata.uns.pop(str(key), None)


def _resolve_condition_key(adata: ad.AnnData, key: str) -> str:
    raw = str(key).strip()
    if ":" not in raw:
        return raw

    parts = [p.strip() for p in raw.split(":") if p.strip()]
    if len(parts) < 2:
        return raw

    missing = [p for p in parts if p not in adata.obs]
    if missing:
        raise RuntimeError(f"condition_key parts not in adata.obs: {missing}")

    combo_key = ".".join(parts)
    if combo_key not in adata.obs:
        comp = _normalize_levels(adata.obs[parts[0]]).map(_safe_combo_token)
        for p in parts[1:]:
            comp = comp + "." + _normalize_levels(adata.obs[p]).map(_safe_combo_token)
        adata.obs[combo_key] = comp
        LOGGER.info("created composite condition_key=%r from %s", combo_key, parts)

    return combo_key


def _parse_condition_key_spec(adata: ad.AnnData, key: str) -> tuple[str, Optional[tuple[str, str]], Optional[tuple[str, str]]]:
    raw = str(key).strip()
    if "^" in raw:
        parts = [p.strip() for p in raw.split("^") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid interaction condition key {raw!r}. Use 'A^B'.")
        return f"{parts[0]}^{parts[1]}", None, (parts[0], parts[1])
    if "@" not in raw:
        return _resolve_condition_key(adata, raw), None, None

    parts = [p.strip() for p in raw.split("@") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid within-B condition key {raw!r}. Use 'A@B'.")
    composite = _resolve_condition_key(adata, f"{parts[0]}:{parts[1]}")
    return composite, (parts[0], parts[1]), None


def _normalize_levels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def _normalize_pair(s: str) -> tuple[str, str]:
    s = str(s).strip()
    if "_vs_" in s:
        a, b = s.split("_vs_", 1)
    elif " vs " in s:
        a, b = s.split(" vs ", 1)
    else:
        raise ValueError(f"Invalid contrast spec {s!r}. Use 'A_vs_B'.")
    a = a.strip()
    b = b.strip()
    if not a or not b:
        raise ValueError(f"Invalid contrast spec {s!r}. Use 'A_vs_B'.")
    if a == b:
        raise ValueError(f"Invalid contrast spec {s!r}: A and B must differ.")
    return a, b


def _pairs_within_group(
    adata: ad.AnnData,
    *,
    groupby: str,
    group_value: str,
    a_key: str,
    b_key: str,
    ref_a: Optional[str] = None,
) -> list[tuple[str, str]]:
    if a_key not in adata.obs or b_key not in adata.obs:
        raise RuntimeError(f"within-cluster: A|B keys not found in adata.obs: {a_key!r}, {b_key!r}")

    g_mask = adata.obs[str(groupby)].astype(str).to_numpy() == str(group_value)
    if not np.any(g_mask):
        return []

    a_vals = _normalize_levels(adata.obs[a_key].loc[g_mask])
    b_vals = _normalize_levels(adata.obs[b_key].loc[g_mask])

    pairs: list[tuple[str, str]] = []
    for b_level in sorted(pd.unique(b_vals).tolist()):
        a_subset = a_vals.loc[b_vals == b_level]
        a_levels = sorted(pd.unique(a_subset).tolist())
        if len(a_levels) < 2:
            continue
        a_counts = a_subset.astype(str).value_counts().to_dict()
        effective_ref = str(ref_a) if ref_a and ref_a in a_levels else _pick_fallback_ref(a_levels, a_counts)
        if effective_ref and effective_ref in a_levels:
            for a_other in a_levels:
                if a_other == effective_ref:
                    continue
                a_label = f"{_safe_combo_token(a_other)}.{_safe_combo_token(b_level)}"
                b_label = f"{_safe_combo_token(effective_ref)}.{_safe_combo_token(b_level)}"
                pairs.append((a_label, b_label))
        else:
            for A, B in _select_pairs(a_levels, None):
                a_label = f"{_safe_combo_token(A)}.{_safe_combo_token(b_level)}"
                b_label = f"{_safe_combo_token(B)}.{_safe_combo_token(b_level)}"
                pairs.append((a_label, b_label))
    return pairs


def _global_ref_level(adata: ad.AnnData, key: str) -> Optional[str]:
    if key not in adata.obs:
        return None
    vc = adata.obs[str(key)].astype(str).value_counts()
    if vc.empty:
        return None
    return str(vc.index[0])


def _pick_fallback_ref(levels: Sequence[str], counts_by_level: Optional[Mapping[str, int]]) -> Optional[str]:
    if not levels:
        return None
    if not counts_by_level:
        return str(levels[0])
    return str(
        max(
            levels,
            key=lambda lv: (int(counts_by_level.get(str(lv), 0)), str(lv)),
        )
    )


def _select_pairs(levels: Sequence[str], requested: Sequence[str] | None) -> list[tuple[str, str]]:
    lv = [str(x) for x in levels]
    all_pairs = [(lv[i], lv[j]) for i in range(len(lv)) for j in range(i + 1, len(lv))]
    if not requested:
        return all_pairs
    req_pairs = [_normalize_pair(x) for x in requested]
    lv_set = set(lv)
    out = []
    for a, b in req_pairs:
        if a not in lv_set or b not in lv_set:
            raise ValueError(f"Requested contrast {a}_vs_{b} not in levels={sorted(lv_set)}")
        out.append((a, b))
    return out


def _select_pairs_with_ref(
    levels: list[str],
    *,
    ref_level: Optional[str],
    requested: Optional[Sequence[str]],
    counts_by_level: Optional[Mapping[str, int]] = None,
) -> list[tuple[str, str]]:
    if requested:
        return _select_pairs(levels, requested)
    if ref_level and ref_level in levels:
        return [(str(lv), str(ref_level)) for lv in levels if str(lv) != str(ref_level)]
    fallback = _pick_fallback_ref(levels, counts_by_level)
    if fallback and fallback in levels:
        return [(str(lv), str(fallback)) for lv in levels if str(lv) != str(fallback)]
    return _select_pairs(levels, None)


def _select_stat_col(df: pd.DataFrame, preferred: str, fallbacks: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    if preferred in df.columns:
        return str(preferred)
    for c in fallbacks:
        if c in df.columns:
            return str(c)
    return None


def _build_stats_matrix_from_tables(
    tables: dict[str, pd.DataFrame],
    *,
    preferred_col: str,
    fallback_cols: Sequence[str],
) -> pd.DataFrame:
    series_by_cluster: dict[str, pd.Series] = {}
    for cl, df in tables.items():
        if df is None or getattr(df, "empty", True):
            continue
        col = _select_stat_col(df, preferred_col, fallback_cols)
        if col is None:
            continue
        if "gene" in df.columns:
            genes = df["gene"].astype(str)
        else:
            genes = df.index.astype(str)
        vals = pd.to_numeric(df[col], errors="coerce")
        s = pd.Series(vals.to_numpy(), index=genes.to_numpy(), name=str(cl))
        s = s.dropna()
        if s.empty:
            continue
        s = s.groupby(level=0).mean()
        series_by_cluster[str(cl)] = s

    if not series_by_cluster:
        return pd.DataFrame()

    out = pd.DataFrame(series_by_cluster).fillna(0.0)
    return out


def _collect_pseudobulk_de_tables(
    adata: ad.AnnData,
    *,
    store_key: str,
    condition_key: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    block = adata.uns.get(store_key, {})
    cond_multi = block.get("pseudobulk_condition_within_group_multi", {})
    cond_single = block.get("pseudobulk_condition_within_group", {})
    candidates = []
    if isinstance(cond_multi, dict) and cond_multi:
        candidates.append(cond_multi)
    if isinstance(cond_single, dict) and cond_single:
        candidates.append(cond_single)
    if not candidates:
        return {}

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for cond in candidates:
        for _, payload in cond.items():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("condition_key", "")) != str(condition_key):
                continue
            df = payload.get("results", None)
            if df is None:
                continue
            cl = str(payload.get("group_value", "")) or "cluster"
            if bool(payload.get("interaction", False)):
                contrast = "interaction"
            else:
                test = payload.get("test", None)
                ref = payload.get("reference", None)
                if test is None or ref is None:
                    continue
                contrast = f"{test}_vs_{ref}"
            out.setdefault(str(contrast), {})[str(cl)] = df
    return out


def _collect_cell_contrast_tables(
    adata: ad.AnnData,
    *,
    store_key: str,
    contrast_key: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    block = adata.uns.get(store_key, {})
    multi = block.get("contrast_conditional_multi", {})
    if isinstance(multi, dict) and str(contrast_key) in multi:
        cc = multi.get(str(contrast_key), {})
    else:
        cc = block.get("contrast_conditional", {})
        if isinstance(cc, dict):
            cc_key = cc.get("contrast_key", None)
            if cc_key is not None and str(cc_key) != str(contrast_key):
                return {}

    results = cc.get("results", {}) if isinstance(cc, dict) else {}
    if not isinstance(results, dict) or not results:
        return {}

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for cl, per_contrast in results.items():
        if not isinstance(per_contrast, dict):
            continue
        for pair_key, tables in per_contrast.items():
            if not isinstance(tables, dict):
                continue
            df = tables.get("combined", None)
            if df is None or getattr(df, "empty", True):
                df = tables.get("wilcoxon", None)
            if df is None:
                continue
            out.setdefault(str(pair_key), {})[str(cl)] = df
    return out


def _write_settings(out_dir: Path, name: str, lines: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _resolve_stable_groupby_and_display_map(
    adata: ad.AnnData,
    *,
    groupby: Optional[str],
    round_id: Optional[str],
    label_source: str,
) -> tuple[str, dict[str, str]]:
    if groupby:
        if groupby not in adata.obs:
            raise KeyError(f"groupby={groupby!r} not in adata.obs")
        return str(groupby), {}

    rid = round_id
    if rid is None:
        rid0 = adata.uns.get("active_cluster_round", None)
        rid = str(rid0) if rid0 else None

    rounds = adata.uns.get("cluster_rounds", {})
    if not rid or not isinstance(rounds, dict) or rid not in rounds:
        raise RuntimeError(
            "markers-and-de: active cluster round not resolved. "
            f"Resolved round_id={rid!r}, active_round={adata.uns.get('active_cluster_round', None)!r}."
        )

    rinfo = rounds[rid]
    labels_obs_key = rinfo.get("labels_obs_key", None)
    if not labels_obs_key or str(labels_obs_key) not in adata.obs:
        raise RuntimeError(
            f"markers-and-de: labels_obs_key not found in adata.obs for round_id={rid!r}."
        )

    label_source_l = str(label_source or "").lower().strip()
    display_map = {}
    if label_source_l == "pretty":
        disp = rinfo.get("cluster_display_map", None)
        if isinstance(disp, dict):
            display_map = {str(k): str(v) for k, v in disp.items()}

    return str(labels_obs_key), display_map


def _prepare_display_groupby(
    adata: ad.AnnData,
    *,
    stable_key: str,
    display_map: dict[str, str],
) -> tuple[Optional[str], Optional[str]]:
    if not display_map:
        return None, None

    labels = adata.obs[stable_key].astype(str)
    display = labels.map(lambda x: display_map.get(str(x), str(x)))

    if display.nunique(dropna=False) != labels.nunique(dropna=False):
        LOGGER.warning(
            "Display labels are not unique; skipping display groupby for plotting."
        )
        return None, None

    base = f"{stable_key}__display"
    key = base
    i = 1
    while key in adata.obs:
        key = f"{base}_{i}"
        i += 1

    labels = adata.obs[stable_key].astype(str)
    seen = set(labels.tolist())
    ordered_display = [str(display_map[k]) for k in display_map.keys() if str(k) in seen]
    if not ordered_display:
        ordered_display = list(dict.fromkeys(display.astype(str).tolist()))
    else:
        for v in display.astype(str).tolist():
            if v not in ordered_display:
                ordered_display.append(v)

    adata.obs[key] = pd.Categorical(display.astype(str), categories=ordered_display, ordered=False)

    color_map = plot_utils._cluster_color_map(adata, stable_key)
    if color_map:
        reverse = {}
        for k, v in display_map.items():
            reverse[str(v)] = str(k)
        cats = list(adata.obs[key].cat.categories)
        colors = []
        for c in cats:
            stable = reverse.get(str(c), str(c))
            colors.append(color_map.get(stable, "#333333"))
        adata.uns[f"{key}_colors"] = list(colors)
        return key, f"{key}_colors"

    return key, None


# -----------------------------------------------------------------------------
# Orchestrator 1: cluster-vs-rest
# -----------------------------------------------------------------------------
def run_cluster_vs_rest(cfg) -> ad.AnnData:
    """
    Cluster-vs-rest orchestrator.

    Runs:
      - cell-level markers (scanpy rank_genes_groups) if cfg.run in {"cell","both"}
      - pseudobulk cluster-vs-rest DE (DESeq2) if cfg.run in {"pseudobulk","both"}
        and guard passes (>= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK unique samples)

    Always:
      - saves dataset
      - writes only the tables that correspond to what actually ran
      - plots only what actually ran
      - records provenance without referencing undefined locals

    Notes:
      - This orchestrator does NOT handle within-cluster contrasts; that is split out.
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (cluster-vs-rest)...")
    if not bool(getattr(cfg, "prune_uns_de", True)):
        LOGGER.warning(
            "prune_uns_de is disabled; adata.uns may become very large. "
            "Use --prune-uns-de to keep only summaries and top genes."
        )

    # ----------------------------
    # I/O + figure setup
    # ----------------------------
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("within-cluster: output_dir=%s", str(output_dir))
    LOGGER.info("within-cluster: input_path=%s", str(getattr(cfg, "input_path")))

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # ----------------------------
    # Resolve stable groupby + display labels
    # ----------------------------
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )

    sample_key = (
        getattr(cfg, "sample_key", None)
        or getattr(cfg, "batch_key", None)
        or adata.uns.get("batch_key")
    )
    if sample_key is None:
        raise RuntimeError("cluster-vs-rest: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")
    if str(sample_key) not in adata.obs:
        raise RuntimeError(f"cluster-vs-rest: sample_key={sample_key!r} not found in adata.obs")

    LOGGER.info("cluster-vs-rest: groupby=%r, sample_key=%r", str(groupby), str(sample_key))

    # ----------------------------
    # Mode decoding
    # ----------------------------
    mode = str(getattr(cfg, "run", "both")).lower()
    run_cell_requested = mode in ("cell", "both")
    run_pb_requested = mode in ("pseudobulk", "both")

    regenerate_figures = bool(getattr(cfg, "regenerate_figures", False))
    if regenerate_figures and not bool(getattr(cfg, "make_figures", True)):
        raise RuntimeError("regenerate_figures=True requires make_figures=True.")

    # ----------------------------
    # Predefine locals (avoid NameErrors)
    # ----------------------------
    positive_only = bool(getattr(cfg, "positive_only", True))

    markers_key: Optional[str] = str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon"))
    layer_candidates = list(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw")))
    counts_layer_used: Optional[str] = None

    n_samples_total = _n_unique_samples(adata, str(sample_key))
    run_pseudobulk = False

    pb_spec: Optional[PseudobulkSpec] = None
    pb_opts: Optional[PseudobulkDEOptions] = None

    if regenerate_figures:
        mk = str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon"))
        has_cell = mk in adata.uns and isinstance(adata.uns.get(mk), dict)
        sk = str(getattr(cfg, "store_key", "scomnom_de"))
        pb = adata.uns.get(sk, {}).get("pseudobulk_cluster_vs_rest", {})
        has_pb = False
        if isinstance(pb, dict):
            res = pb.get("results", {})
            top = pb.get("top_genes", {})
            summary = pb.get("summary", None)
            has_pb = (
                (isinstance(res, dict) and bool(res))
                or (isinstance(top, dict) and bool(top))
                or (isinstance(summary, pd.DataFrame) and not summary.empty)
            )

        if not has_cell and not has_pb:
            raise RuntimeError("regenerate_figures: no cell-level or pseudobulk results found in adata.uns.")
        if run_cell_requested and not has_cell:
            LOGGER.warning("regenerate_figures: no cell-level results found; skipping cell-level plots.")
            run_cell_requested = False
        if run_pb_requested and not has_pb:
            LOGGER.warning("regenerate_figures: no pseudobulk results found; skipping pseudobulk plots.")
            run_pb_requested = False

    run_cell = run_cell_requested and not regenerate_figures

    # ----------------------------
    # 1) Cell-level markers (cluster-vs-rest)
    # ----------------------------
    if run_cell:
        markers_spec = CellLevelMarkerSpec(
            method=str(getattr(cfg, "markers_method", "wilcoxon")).lower(),
            n_genes=int(getattr(cfg, "markers_n_genes", 100)),
            use_raw=bool(getattr(cfg, "markers_use_raw", False)),
            layer=getattr(cfg, "markers_layer", None),
            rankby_abs=bool(getattr(cfg, "markers_rankby_abs", True)),
            max_cells_per_group=int(getattr(cfg, "markers_downsample_max_per_group", 2000)),
            random_state=int(getattr(cfg, "random_state", 42)),
            min_pct=float(getattr(cfg, "min_pct", 0.25)),
            min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
        )

        markers_key = str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon"))
        compute_markers_celllevel(
            adata,
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            spec=markers_spec,
            key_added=markers_key,
            store=True,
        )
    else:
        if regenerate_figures and run_cell_requested:
            LOGGER.info("cluster-vs-rest: regenerate_figures=True; skipping cell-level computation.")
        else:
            LOGGER.info("cluster-vs-rest: skipping cell-level markers (run=%r).", mode)

    # ----------------------------
    # 2) Pseudobulk cluster-vs-rest DE (guarded)
    # ----------------------------
    if run_pb_requested and not regenerate_figures:
        run_pseudobulk = n_samples_total >= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK
        if not run_pseudobulk:
            LOGGER.warning(
                "cluster-vs-rest: pseudobulk requested but disabled: only %d unique samples in %r (< %d).",
                n_samples_total,
                str(sample_key),
                _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK,
            )
        else:
            counts_layer_used = _choose_counts_layer(
                adata,
                candidates=layer_candidates,
                allow_X_counts=bool(getattr(cfg, "allow_X_counts", True)),
            )

            pb_spec = PseudobulkSpec(
                sample_key=str(sample_key),
                counts_layer=counts_layer_used,  # None -> uses adata.X
            )

            pb_opts = PseudobulkDEOptions(
                min_cells_per_sample_group=int(getattr(cfg, "min_cells_target", 20)),
                min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
                alpha=float(getattr(cfg, "alpha", 0.05)),
                shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
                min_total_counts=int(getattr(cfg, "pb_min_total_counts", 100)),
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
                min_counts_per_lib=int(getattr(cfg, "pb_min_counts_per_lib", 5)),
                min_lib_pct=float(getattr(cfg, "pb_min_lib_pct", 0.0)),
                covariates=tuple(getattr(cfg, "pb_covariates", ())),
            )

            LOGGER.info(
                "cluster-vs-rest: pseudobulk DE (sample_key=%r, counts_layer=%r, min_cells_per_sample_group=%d).",
                pb_spec.sample_key,
                pb_spec.counts_layer,
                pb_opts.min_cells_per_sample_group,
            )

            _ = de_cluster_vs_rest_pseudobulk(
                adata,
                groupby=str(groupby),
                round_id=getattr(cfg, "round_id", None),
                spec=pb_spec,
                opts=pb_opts,
                store_key=str(getattr(cfg, "store_key", "scomnom_de")),
                store=True,
                n_jobs=int(getattr(cfg, "n_jobs", 1)),
            )
    else:
        if regenerate_figures and run_pb_requested:
            LOGGER.info("cluster-vs-rest: regenerate_figures=True; skipping pseudobulk computation.")
        else:
            LOGGER.info("cluster-vs-rest: skipping pseudobulk (run=%r).", mode)

    # ----------------------------
    # Provenance (safe)
    # ----------------------------
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "mode": "cluster-vs-rest",
            "groupby": str(groupby),
            "sample_key": str(sample_key),
            "run": mode,
            "cell_ran": bool(run_cell),
            "pseudobulk_requested": bool(run_pb_requested),
            "pseudobulk_enabled": bool(run_pseudobulk),
            "pseudobulk_guard_min_total_samples": int(_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK),
            "pseudobulk_n_unique_samples": int(n_samples_total),
            "markers_key": str(markers_key) if markers_key else None,
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer_used,
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "positive_only_markers": bool(positive_only),
        }
    )

    # ----------------------------
    # Exports (only what ran)
    # ----------------------------
    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    run_namespace = _run_namespace_for_round(
        adata,
        prefix="markers",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    marker_cell_dir = results_dir / "tables" / run_round / "cell_based"
    marker_pb_dir = results_dir / "tables" / run_round / "pseudobulk_based"

    if run_cell and markers_key and not regenerate_figures:
        io_utils.export_rank_genes_groups_tables(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
            display_map=display_map,
            prefix="celllevel_markers",
            tables_root=marker_cell_dir,
        )
        io_utils.export_rank_genes_groups_excel(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
            display_map=display_map,
            filename="celllevel_markers.xlsx",
            max_genes=int(getattr(cfg, "markers_n_genes", 100)),
            tables_root=marker_cell_dir,
        )
        _write_settings(
            marker_cell_dir,
            "de_settings.txt",
            [
                "mode=cluster-vs-rest",
                "engine=cell",
                f"groupby={groupby}",
                f"sample_key={sample_key}",
                f"markers_method={getattr(cfg, 'markers_method', None)}",
                f"markers_n_genes={getattr(cfg, 'markers_n_genes', None)}",
                f"markers_rankby_abs={getattr(cfg, 'markers_rankby_abs', None)}",
                f"markers_use_raw={getattr(cfg, 'markers_use_raw', None)}",
                f"markers_layer={getattr(cfg, 'markers_layer', None)}",
                f"markers_downsample_threshold={getattr(cfg, 'markers_downsample_threshold', None)}",
                f"markers_downsample_max_per_group={getattr(cfg, 'markers_downsample_max_per_group', None)}",
                f"min_pct={getattr(cfg, 'min_pct', None)}",
                f"min_diff_pct={getattr(cfg, 'min_diff_pct', None)}",
                "design_formula=NA (cell-level)",
            ],
        )
    else:
        LOGGER.info("cluster-vs-rest: skipping cell-level exports (cell markers did not run).")

    store_key = str(getattr(cfg, "store_key", "scomnom_de"))

    if run_pseudobulk and not regenerate_figures:
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            display_map=display_map,
            groupby=str(groupby),
            condition_key=None,
            tables_root=marker_pb_dir,
        )
        io_utils.export_pseudobulk_cluster_vs_rest_excel(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            display_map=display_map,
            tables_root=marker_pb_dir,
        )
        covariates = tuple(getattr(cfg, "pb_covariates", ()))
        design_terms = ["sample", *covariates, "binary_cluster"]
        _write_settings(
            marker_pb_dir,
            "de_settings.txt",
            [
                "mode=cluster-vs-rest",
                "engine=pseudobulk",
                f"groupby={groupby}",
                f"sample_key={sample_key}",
                f"counts_layer={counts_layer_used}",
                f"min_cells_per_sample_group={getattr(cfg, 'min_cells_target', None)}",
                f"min_samples_per_level={getattr(cfg, 'min_samples_per_level', None)}",
                f"alpha={getattr(cfg, 'alpha', None)}",
                f"shrink_lfc={getattr(cfg, 'shrink_lfc', None)}",
                f"min_total_counts={getattr(cfg, 'pb_min_total_counts', None)}",
                f"min_counts_per_lib={getattr(cfg, 'pb_min_counts_per_lib', None)}",
                f"min_lib_pct={getattr(cfg, 'pb_min_lib_pct', None)}",
                f"min_pct={getattr(cfg, 'min_pct', None)}",
                "min_diff_pct=NA (unused for pseudobulk)",
                f"positive_only={getattr(cfg, 'positive_only', None)}",
                f"pb_max_genes={getattr(cfg, 'pb_max_genes', None)}",
                f"pb_covariates={covariates}",
                f"design_formula={' + '.join(design_terms)}",
                "contrast=binary_cluster: target vs rest",
            ],
        )
    else:
        LOGGER.info("cluster-vs-rest: skipping pseudobulk exports (pseudobulk disabled or not requested).")

    # ----------------------------
    # Plotting (only what ran)
    # ----------------------------
    if bool(getattr(cfg, "make_figures", True)):
        LOGGER.info("cluster-vs-rest: constructing figures...")
        from . import de_plot_utils
        display_key = None
        display_colors_key = None
        try:
            display_key, display_colors_key = _prepare_display_groupby(
                adata,
                stable_key=str(groupby),
                display_map=display_map,
            )

            alpha = float(getattr(cfg, "alpha", 0.05))
            lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
            top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
            top_n_genes = int(getattr(cfg, "plot_top_n_per_cluster", 9))
            dotplot_top_n_genes = int(getattr(cfg, "plot_dotplot_top_n_genes", 15))
            use_raw = bool(getattr(cfg, "plot_use_raw", False))
            layer = getattr(cfg, "plot_layer", None)
            ncols = int(getattr(cfg, "plot_umap_ncols", 3))

            if run_cell_requested and markers_key:
                de_plot_utils.plot_marker_genes_ranksum(
                    adata,
                    groupby=str(groupby),
                    display_groupby=str(display_key) if display_key else None,
                    display_map=display_map,
                    markers_key=str(markers_key),
                    alpha=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    top_n_genes=top_n_genes,
                    dotplot_top_n_genes=dotplot_top_n_genes,
                    use_raw=use_raw,
                    layer=layer,
                    umap_ncols=ncols,
                    plot_gene_filter=getattr(cfg, "plot_gene_filter", ()),
                )
            else:
                LOGGER.info("cluster-vs-rest: skipping cell-level marker plots (no markers computed).")

            if run_pb_requested:
                de_plot_utils.plot_marker_genes_pseudobulk(
                    adata,
                    groupby=str(groupby),
                    display_groupby=str(display_key) if display_key else None,
                    display_map=display_map,
                    store_key=store_key,
                    alpha=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    top_n_genes=top_n_genes,
                    dotplot_top_n_genes=dotplot_top_n_genes,
                    use_raw=use_raw,
                    layer=layer,
                    umap_ncols=ncols,
                    plot_gene_filter=getattr(cfg, "plot_gene_filter", ()),
                )
            else:
                LOGGER.info("cluster-vs-rest: skipping pseudobulk plots (pseudobulk disabled or not requested).")

            try:
                for fmt in getattr(cfg, "figure_formats", ["png", "pdf"]):
                    reporting.generate_markers_report(
                        fig_root=figdir,
                        fmt=fmt,
                        cfg=cfg,
                        version=__version__,
                        adata=adata,
                    )
                LOGGER.info("Wrote markers report.")
            except Exception as e:
                LOGGER.warning("Failed to generate markers report: %s", e)
        finally:
            if display_key is not None:
                adata.obs.drop(columns=[display_key], inplace=True, errors="ignore")
            if display_colors_key is not None:
                adata.uns.pop(display_colors_key, None)

    # ----------------------------
    # Save dataset
    # ----------------------------
    if regenerate_figures:
        LOGGER.info("cluster-vs-rest: regenerate_figures=True; skipping dataset save.")
    else:
        if bool(getattr(cfg, "prune_uns_de", True)):
            _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))

        out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
        LOGGER.info("within-cluster: saving outputs...")
        LOGGER.info("Saving dataset → %s", out_zarr)
        io_utils.save_dataset(adata, out_zarr, fmt="zarr")

        if bool(getattr(cfg, "save_h5ad", False)):
            out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
            LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
            io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (cluster-vs-rest).")
    return adata


def run_enrichment(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (enrichment)...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    rounds = adata.uns.get("cluster_rounds", {})
    target_round_id = getattr(cfg, "round_id", None)
    if target_round_id is None:
        active_round_id = adata.uns.get("active_cluster_round", None)
        target_round_id = str(active_round_id) if active_round_id is not None else None
    if not target_round_id or not isinstance(rounds, dict) or str(target_round_id) not in rounds:
        raise RuntimeError(
            "enrichment: target round not resolved. "
            f"Resolved round_id={target_round_id!r}, active_round={adata.uns.get('active_cluster_round', None)!r}."
        )
    target_round_id = str(target_round_id)

    regenerate_figures = bool(getattr(cfg, "regenerate_figures", False))
    if regenerate_figures:
        round_dec = rounds[target_round_id].get("decoupler", {})
        if not isinstance(round_dec, dict) or not round_dec:
            raise RuntimeError(
                f"enrichment: no stored decoupler payload found for round_id={target_round_id!r}."
            )
    else:
        run_decoupler_for_round(adata, cfg, round_id=target_round_id)

    if bool(getattr(cfg, "make_figures", True)):
        run_namespace = _run_namespace_for_round(
            adata,
            prefix="enrichment",
            round_id=target_round_id,
        )
        run_round = str(plot_utils.get_run_subdir(run_namespace))
        figdir_round = Path(run_round)

        snapshot = _snapshot_top_level_decoupler_state(adata)
        try:
            _publish_decoupler_from_round_to_top_level(
                adata,
                round_id=target_round_id,
                resources=("msigdb", "progeny", "dorothea"),
                publish_pseudobulk=True,
                clear_missing=True,
            )
            artifacts = []
            if "msigdb" in adata.uns:
                artifacts.extend(
                    plot_utils.plot_decoupler_all_styles(
                        adata,
                        net_key="msigdb",
                        net_name="MSigDB",
                        figdir=figdir_round,
                        heatmap_top_k=30,
                        bar_top_n=15,
                        bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                        bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                        bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                        dotplot_top_k=25,
                    )
                )
            if "progeny" in adata.uns:
                artifacts.extend(
                    plot_utils.plot_decoupler_all_styles(
                        adata,
                        net_key="progeny",
                        net_name="PROGENy",
                        figdir=figdir_round,
                        heatmap_top_k=30,
                        bar_top_n=15,
                        bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                        bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                        bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                        dotplot_top_k=25,
                    )
                )
            if "dorothea" in adata.uns:
                artifacts.extend(
                    plot_utils.plot_decoupler_all_styles(
                        adata,
                        net_key="dorothea",
                        net_name="DoRothEA",
                        figdir=figdir_round,
                        heatmap_top_k=30,
                        bar_top_n=15,
                        bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                        bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                        bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                        dotplot_top_k=25,
                    )
            )
            plot_utils.persist_plot_artifacts(artifacts)
        finally:
            _restore_top_level_decoupler_state(adata, snapshot)

        try:
            LOGGER.info("enrichment: generating enrichment report...")
            for fmt in getattr(cfg, "figure_formats", ["png", "pdf"]):
                reporting.generate_enrichment_cluster_report(
                    fig_root=figdir,
                    fmt=fmt,
                    cfg=cfg,
                    version=__version__,
                    run_dir=figdir / str(fmt).lower().lstrip(".") / run_round,
                )
            LOGGER.info("Wrote enrichment report.")
        except Exception as e:
            LOGGER.warning("Failed to generate enrichment report: %s", e)

    if regenerate_figures:
        LOGGER.info("enrichment: regenerate_figures=True; skipping dataset save.")
    else:
        clear_top_level_decoupler_state(adata)
        if bool(getattr(cfg, "prune_uns_de", True)):
            _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))
        gc.collect()
        out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.enrichment")) + ".zarr")
        LOGGER.info("Saving dataset → %s", out_zarr)
        io_utils.save_dataset(adata, out_zarr, fmt="zarr")

        if bool(getattr(cfg, "save_h5ad", False)):
            out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.enrichment")) + ".h5ad")
            LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
            io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (enrichment).")
    return adata


def run_enrichment_cluster(cfg) -> ad.AnnData:
    return run_enrichment(cfg)


def _sanitize_module_name(name: str) -> str:
    token = io_utils.sanitize_identifier(str(name).strip(), allow_spaces=False)
    return token or "module"


def _load_modules_from_gmt(path: Path) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                continue
            module_name = str(parts[0]).strip()
            genes = [str(g).strip() for g in parts[2:] if str(g).strip()]
            if module_name and genes:
                modules.setdefault(module_name, [])
                modules[module_name].extend(genes)
    return modules


def _load_modules_from_delimited(path: Path) -> dict[str, list[str]]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if df.empty:
        return {}
    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    module_col = cols_lower.get("module") or cols_lower.get("set") or cols_lower.get("signature")
    gene_col = cols_lower.get("gene") or cols_lower.get("genes") or cols_lower.get("symbol")

    modules: dict[str, list[str]] = {}
    if module_col and gene_col:
        for row in df[[module_col, gene_col]].itertuples(index=False):
            module_name = str(row[0]).strip()
            gene = str(row[1]).strip()
            if module_name and gene:
                modules.setdefault(module_name, []).append(gene)
        return modules

    if df.shape[1] >= 2:
        first, second = list(df.columns[:2])
        for row in df[[first, second]].itertuples(index=False):
            module_name = str(row[0]).strip()
            gene = str(row[1]).strip()
            if module_name and gene:
                modules.setdefault(module_name, []).append(gene)
        if modules:
            return modules

    genes = [str(x).strip() for x in df.iloc[:, 0].tolist() if str(x).strip()]
    if genes:
        return {path.stem: genes}
    return {}


def _load_modules_from_txt(path: Path) -> dict[str, list[str]]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not genes:
        return {}
    return {path.stem: genes}


def _load_module_definitions(module_files: Sequence[str | Path]) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for raw_path in module_files:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"module-score: module file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".gmt":
            loaded = _load_modules_from_gmt(path)
        elif suffix in {".tsv", ".csv"}:
            loaded = _load_modules_from_delimited(path)
        elif suffix in {".txt", ".list"}:
            loaded = _load_modules_from_txt(path)
        else:
            raise RuntimeError(
                f"module-score: unsupported module file format for {path.name!r}. "
                "Use .gmt, .tsv, .csv, or single-module .txt."
            )

        for module_name, genes in loaded.items():
            bucket = modules.setdefault(str(module_name).strip(), [])
            bucket.extend(str(g).strip() for g in genes if str(g).strip())

    cleaned: dict[str, list[str]] = {}
    for module_name, genes in modules.items():
        deduped = list(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
        if deduped:
            cleaned[str(module_name)] = deduped
    if not cleaned:
        raise RuntimeError("module-score: no module definitions could be loaded from the supplied files.")
    return cleaned


def _module_score_gene_pool(
    adata: ad.AnnData,
    *,
    use_raw: bool,
) -> list[str]:
    if use_raw and adata.raw is not None:
        gene_index = adata.raw.var_names.astype(str)
    else:
        gene_index = adata.var_names.astype(str)
    return list(gene_index.astype(str))


def _score_modules_scanpy(
    adata: ad.AnnData,
    *,
    modules: Mapping[str, list[str]],
    gene_pool: Sequence[str],
    target_round_id: str,
    module_set_token: str,
    use_raw: bool,
    layer: str | None,
    ctrl_size: int,
    n_bins: int,
    random_state: int,
) -> tuple[list[str], dict[str, str], list[dict[str, object]]]:
    import scanpy as sc

    gene_pool_set = set(gene_pool)
    score_keys: list[str] = []
    score_key_to_module: dict[str, str] = {}
    module_records: list[dict[str, object]] = []

    for module_name, genes in modules.items():
        retained = [str(g) for g in genes if str(g) in gene_pool_set]
        record: dict[str, object] = {
            "module": str(module_name),
            "n_genes_input": int(len(genes)),
            "n_genes_retained": int(len(retained)),
            "genes_retained": ",".join(retained),
        }
        if not retained:
            LOGGER.warning("module-score: module %r retained no genes after filtering; skipping.", str(module_name))
            module_records.append(record)
            continue

        score_key = (
            f"module_score__{_sanitize_module_name(target_round_id)}__"
            f"{module_set_token}__{_sanitize_module_name(module_name)}"
        )
        sc.tl.score_genes(
            adata,
            gene_list=retained,
            score_name=score_key,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            random_state=random_state,
            gene_pool=list(gene_pool),
            use_raw=use_raw,
            layer=layer,
            copy=False,
        )
        adata.obs[score_key] = pd.to_numeric(adata.obs[score_key], errors="coerce").astype(np.float32)
        score_keys.append(score_key)
        score_key_to_module[str(score_key)] = str(module_name)
        record["score_key"] = str(score_key)
        module_records.append(record)

    return score_keys, score_key_to_module, module_records


def _score_modules_aucell(
    adata: ad.AnnData,
    *,
    modules: Mapping[str, list[str]],
    gene_pool: Sequence[str],
    target_round_id: str,
    module_set_token: str,
    use_raw: bool,
    layer: str | None,
) -> tuple[list[str], dict[str, str], list[dict[str, object]]]:
    import decoupler as dc

    gene_pool_set = set(gene_pool)
    module_records: list[dict[str, object]] = []
    net_rows: list[dict[str, object]] = []
    module_names: list[str] = []

    for module_name, genes in modules.items():
        retained = [str(g) for g in genes if str(g) in gene_pool_set]
        record: dict[str, object] = {
            "module": str(module_name),
            "n_genes_input": int(len(genes)),
            "n_genes_retained": int(len(retained)),
            "genes_retained": ",".join(retained),
        }
        if not retained:
            LOGGER.warning("module-score: module %r retained no genes after filtering; skipping.", str(module_name))
            module_records.append(record)
            continue

        module_names.append(str(module_name))
        for gene in retained:
            net_rows.append({"source": str(module_name), "target": str(gene)})
        module_records.append(record)

    if not net_rows:
        return [], {}, module_records

    net = pd.DataFrame(net_rows, columns=["source", "target"])
    dc.mt.aucell(
        adata,
        net=net,
        tmin=1,
        raw=bool(use_raw),
        layer=str(layer) if layer else None,
        verbose=False,
    )
    scores = adata.obsm.get("score_aucell", None)
    if scores is None or not isinstance(scores, pd.DataFrame) or scores.empty:
        raise RuntimeError("module-score: AUCell did not produce a valid score matrix.")

    score_keys: list[str] = []
    score_key_to_module: dict[str, str] = {}
    for module_name in module_names:
        if str(module_name) not in scores.columns:
            LOGGER.warning("module-score: AUCell output missing module %r; skipping.", str(module_name))
            continue
        score_key = (
            f"module_score__{_sanitize_module_name(target_round_id)}__"
            f"{module_set_token}__{_sanitize_module_name(module_name)}"
        )
        adata.obs[score_key] = pd.to_numeric(scores[str(module_name)], errors="coerce").astype(np.float32)
        score_keys.append(score_key)
        score_key_to_module[str(score_key)] = str(module_name)

    for record in module_records:
        module_name = str(record.get("module", ""))
        score_key = next((k for k, v in score_key_to_module.items() if v == module_name), None)
        if score_key is not None:
            record["score_key"] = str(score_key)

    adata.obsm.pop("score_aucell", None)
    adata.obsm.pop("padj_aucell", None)
    return score_keys, score_key_to_module, module_records


def _zscore_module_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    centered = summary - summary.mean(axis=0)
    scale = summary.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = centered.divide(scale, axis=1).fillna(0.0)
    return z.astype(np.float32, copy=False)


def _store_module_score_payload(
    adata: ad.AnnData,
    *,
    round_id: str,
    module_set_name: str,
    payload: Mapping[str, object],
) -> None:
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or round_id not in rounds:
        raise RuntimeError(f"module-score: round_id={round_id!r} missing from adata.uns['cluster_rounds'].")
    round_info = rounds[str(round_id)]
    module_block = round_info.setdefault("module_scores", {})
    if not isinstance(module_block, dict):
        module_block = {}
        round_info["module_scores"] = module_block
    module_block[str(module_set_name)] = dict(payload)


def _compute_module_score_on_adata(
    adata: ad.AnnData,
    cfg,
) -> tuple[dict[str, object], list[str], str]:
    rounds = adata.uns.get("cluster_rounds", {})
    target_round_id = getattr(cfg, "round_id", None)
    if target_round_id is None:
        active_round_id = adata.uns.get("active_cluster_round", None)
        target_round_id = str(active_round_id) if active_round_id is not None else None
    if not target_round_id or not isinstance(rounds, dict) or str(target_round_id) not in rounds:
        raise RuntimeError(
            "module-score: target round not resolved. "
            f"Resolved round_id={target_round_id!r}, active_round={adata.uns.get('active_cluster_round', None)!r}."
        )
    target_round_id = str(target_round_id)

    method = str(getattr(cfg, "module_score_method", "scanpy") or "scanpy").strip().lower()
    if method not in {"scanpy", "aucell"}:
        raise RuntimeError(f"module-score: unsupported module_score_method={method!r}")

    use_raw = bool(getattr(cfg, "module_score_use_raw", False))
    layer = getattr(cfg, "module_score_layer", None)
    if use_raw and layer:
        raise RuntimeError("module-score: cannot use both module_score_use_raw and module_score_layer.")
    if use_raw and adata.raw is None:
        raise RuntimeError("module-score: module_score_use_raw=True but adata.raw is not available.")
    if layer and str(layer) not in adata.layers:
        raise RuntimeError(f"module-score: module_score_layer={layer!r} not found in adata.layers.")

    module_set_name = str(getattr(cfg, "module_set_name", None) or "module_score").strip()
    module_set_token = _sanitize_module_name(module_set_name)
    modules = _load_module_definitions(getattr(cfg, "module_files", ()))

    gene_pool = _module_score_gene_pool(adata, use_raw=use_raw)
    if not gene_pool:
        raise RuntimeError("module-score: no genes are available in the selected expression source.")

    round_info = rounds[target_round_id]
    labels_obs_key = str(round_info.get("labels_obs_key") or round_info.get("cluster_key") or "leiden")
    grouping = _prepare_decoupler_grouping(
        adata,
        round_id=target_round_id,
        labels_obs_key=labels_obs_key,
        condition_key=getattr(cfg, "condition_key", None),
    )
    group_key = str(grouping["group_key"])
    cleanup_key = grouping.get("cleanup_key", None)
    display_map = dict(grouping.get("display_map", {}) or {})
    display_order = grouping.get("display_order", None)

    ctrl_size = int(getattr(cfg, "module_score_ctrl_size", 50))
    n_bins = int(getattr(cfg, "module_score_n_bins", 25))
    random_state = int(getattr(cfg, "module_score_random_state", 0))
    if method == "scanpy":
        score_keys, score_key_to_module, module_records = _score_modules_scanpy(
            adata,
            modules=modules,
            gene_pool=gene_pool,
            target_round_id=target_round_id,
            module_set_token=module_set_token,
            use_raw=use_raw,
            layer=str(layer) if layer else None,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            random_state=random_state,
        )
    else:
        score_keys, score_key_to_module, module_records = _score_modules_aucell(
            adata,
            modules=modules,
            gene_pool=gene_pool,
            target_round_id=target_round_id,
            module_set_token=module_set_token,
            use_raw=use_raw,
            layer=str(layer) if layer else None,
        )

    if not score_keys:
        raise RuntimeError("module-score: no modules retained any genes after filtering.")

    score_frame = adata.obs[[group_key] + score_keys].copy()
    summary_mean = score_frame.groupby(group_key, observed=False)[score_keys].mean()
    summary_median = score_frame.groupby(group_key, observed=False)[score_keys].median()
    n_cells = score_frame.groupby(group_key, observed=False).size().rename("n_cells").to_frame()

    if display_map:
        summary_mean.index = [display_map.get(str(x), str(x)) for x in summary_mean.index]
        summary_median.index = [display_map.get(str(x), str(x)) for x in summary_median.index]
        n_cells.index = [display_map.get(str(x), str(x)) for x in n_cells.index]

    if display_order:
        order = [str(x) for x in display_order if str(x) in summary_mean.index]
        if order:
            summary_mean = summary_mean.reindex(order)
            summary_median = summary_median.reindex(order)
            n_cells = n_cells.reindex(order)

    summary_mean = summary_mean.rename(columns=score_key_to_module).astype(np.float32)
    summary_median = summary_median.rename(columns=score_key_to_module).astype(np.float32)
    summary_mean_z = _zscore_module_summary(summary_mean)

    module_meta = pd.DataFrame(module_records)
    payload = {
        "module_set_name": module_set_name,
        "method": method,
        "round_id": target_round_id,
        "condition_key": grouping.get("condition_key", None),
        "module_files": [str(Path(p)) for p in getattr(cfg, "module_files", ())],
        "module_score_use_raw": bool(use_raw),
        "module_score_layer": str(layer) if layer else None,
        "module_score_ctrl_size": int(ctrl_size),
        "module_score_n_bins": int(n_bins),
        "module_score_random_state": int(random_state),
        "score_keys": list(score_keys),
        "score_key_to_module": dict(score_key_to_module),
        "group_display_map": display_map,
        "group_display_order": list(display_order) if isinstance(display_order, list) else display_order,
        "module_meta": module_meta,
        "summary_mean": summary_mean,
        "summary_median": summary_median,
        "summary_mean_z": summary_mean_z,
        "n_cells": n_cells,
    }
    _store_module_score_payload(
        adata,
        round_id=target_round_id,
        module_set_name=module_set_name,
        payload=payload,
    )
    if cleanup_key:
        adata.obs.drop(columns=[str(cleanup_key)], inplace=True, errors="ignore")

    return payload, score_keys, target_round_id


def run_module_score(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (module-score)...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    payload, score_keys, target_round_id = _compute_module_score_on_adata(adata, cfg)

    run_namespace = _run_namespace_for_round(
        adata,
        prefix=f"module_score_{_sanitize_module_name(str(payload['module_set_name']))}",
        round_id=target_round_id,
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round
    tables_root.mkdir(parents=True, exist_ok=True)
    module_meta = payload["module_meta"]
    summary_mean = payload["summary_mean"]
    summary_median = payload["summary_median"]
    summary_mean_z = payload["summary_mean_z"]
    n_cells = payload["n_cells"]
    module_meta.to_csv(tables_root / "module_meta.tsv", sep="\t", index=False)
    summary_mean.to_csv(tables_root / "module_score_summary_mean.tsv", sep="\t")
    summary_median.to_csv(tables_root / "module_score_summary_median.tsv", sep="\t")
    summary_mean_z.to_csv(tables_root / "module_score_summary_mean_z.tsv", sep="\t")
    n_cells.to_csv(tables_root / "module_score_group_sizes.tsv", sep="\t")
    _write_settings(
        tables_root,
        "__settings.txt",
        [
            f"round_id={target_round_id}",
            f"module_set_name={payload['module_set_name']}",
            f"module_score_method={payload['method']}",
            f"module_score_use_raw={payload['module_score_use_raw']}",
            f"module_score_layer={payload['module_score_layer']}",
            f"condition_key={payload['condition_key']}",
            f"module_files={payload['module_files']}",
        ],
    )

    if bool(getattr(cfg, "make_figures", True)):
        artifacts = []
        artifacts.extend(
            plot_utils.plot_module_score_summary_heatmap(
                payload["summary_mean_z"],
                figdir=Path(run_round),
                stem="module_score_summary_mean_z",
                title=f"Module score summary ({payload['module_set_name']})",
            )
        )
        max_umaps = int(getattr(cfg, "module_score_max_umaps", 12))
        umap_keys = list(score_keys[:max(0, max_umaps)])
        if umap_keys:
            artifacts.extend(
                plot_utils.umap_by(
                    adata,
                    umap_keys,
                    figdir=Path(run_round) / "umaps",
                    stem="module_score",
                )
            )
        plot_utils.persist_plot_artifacts(artifacts)

        try:
            LOGGER.info("module-score: generating report...")
            for fmt in getattr(cfg, "figure_formats", ["png", "pdf"]):
                reporting.generate_module_score_report(
                    fig_root=figdir,
                    fmt=fmt,
                    cfg=cfg,
                    version=__version__,
                    adata=adata,
                    run_dir=figdir / str(fmt).lower().lstrip(".") / run_round,
                )
            LOGGER.info("Wrote module-score report.")
        except Exception as e:
            LOGGER.warning("Failed to generate module-score report: %s", e)

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.module_score")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.module_score")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info(
        "Finished markers-and-de (module-score); modules=%d retained=%d.",
        int(len(payload["module_meta"])),
        int(len(score_keys)),
    )
    return adata


def _build_gene_meta_from_de_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for df in tables.values():
        if df is None or getattr(df, "empty", True):
            continue
        meta = df.copy()
        if "gene" in meta.columns:
            meta["gene"] = meta["gene"].astype(str)
        else:
            meta.insert(0, "gene", meta.index.astype(str))
        meta = meta.drop_duplicates(subset=["gene"], keep="first")
        frames.append(meta)

    if not frames:
        return pd.DataFrame(columns=["gene"]).set_index("gene", drop=False)

    meta = pd.concat(frames, axis=0, ignore_index=True)
    meta["gene"] = meta["gene"].astype(str)
    meta = meta.drop_duplicates(subset=["gene"], keep="first")
    return meta.set_index("gene", drop=False)


def _apply_gene_filters_to_de_stats(
    stats: pd.DataFrame,
    *,
    gene_meta: pd.DataFrame,
    gene_filter: Optional[Sequence[str]],
    resource_name: str,
) -> tuple[pd.DataFrame, Optional[dict[str, object]]]:
    if stats.empty or not gene_filter:
        return stats, None

    exprs: list[str] = []
    for raw in gene_filter:
        parsed = _parse_gene_filter_entry(raw)
        if parsed:
            exprs.append(parsed)
    if not exprs:
        return stats, None

    meta = gene_meta.copy()
    if "gene" not in meta.columns:
        meta["gene"] = meta.index.astype(str)
    meta = meta.reindex(stats.index.astype(str))

    keep = pd.Series(True, index=meta.index)
    for expr_raw in exprs:
        expr_norm = _normalize_gene_filter_expr(str(expr_raw))
        try:
            matched = meta.query(expr_norm, engine="python")
            keep &= meta.index.isin(matched.index)
        except Exception as e:
            raise RuntimeError(
                f"{resource_name}: gene_filter failed for expr={str(expr_raw)!r}: {e}"
            ) from e

    kept_genes = keep[keep].index.astype(str).tolist()
    filtered = stats.loc[kept_genes].copy()
    info = {
        "gene_filter": tuple(str(x) for x in exprs),
        "n_genes_input": int(stats.shape[0]),
        "n_genes_retained": int(filtered.shape[0]),
    }
    LOGGER.info(
        "%s: gene_filter retained %d/%d genes.",
        resource_name,
        int(filtered.shape[0]),
        int(stats.shape[0]),
    )
    return filtered, info


def _collect_pseudobulk_de_tables_from_dir(
    input_dir: Path,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    out: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for cond_dir in sorted(input_dir.glob("condition_within_cluster__*")):
        if not cond_dir.is_dir():
            continue
        condition_key = cond_dir.name.removeprefix("condition_within_cluster__")
        for csv_path in sorted(cond_dir.glob("condition_within_cluster__*.csv")):
            if csv_path.name == "__summary.csv":
                continue
            stem = csv_path.stem.removeprefix("condition_within_cluster__")
            if "__" in stem:
                cluster_label, contrast = stem.split("__", 1)
            else:
                cluster_label, contrast = stem, "contrast"
            out.setdefault(str(condition_key), {}).setdefault(str(contrast), {})[str(cluster_label)] = pd.read_csv(csv_path)
    return out


def _collect_cell_contrast_tables_from_dir(
    input_dir: Path,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    raw: dict[str, dict[str, dict[str, dict[str, pd.DataFrame]]]] = {}
    for pair_dir in sorted(input_dir.glob("*_DE")):
        if not pair_dir.is_dir():
            continue
        prefix = pair_dir.name[:-3] if pair_dir.name.endswith("_DE") else pair_dir.name
        for csv_path in sorted(pair_dir.rglob("cluster__*__*.csv")):
            stem = csv_path.stem
            if not stem.startswith("cluster__"):
                continue
            rest = stem.removeprefix("cluster__")
            if "__" not in rest:
                continue
            cluster_label, remainder = rest.split("__", 1)
            if "__" not in remainder:
                continue
            pair_key, kind = remainder.rsplit("__", 1)
            contrast_key = prefix
            suffix = f"_{pair_key}"
            if contrast_key.endswith(suffix):
                contrast_key = contrast_key[: -len(suffix)]
            raw.setdefault(str(contrast_key), {}).setdefault(str(pair_key), {}).setdefault(str(cluster_label), {})[str(kind)] = pd.read_csv(csv_path)

    out: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for contrast_key, per_pair in raw.items():
        for pair_key, per_cluster in per_pair.items():
            chosen: dict[str, pd.DataFrame] = {}
            for cluster_label, per_kind in per_cluster.items():
                for kind in ("combined", "wilcoxon", "logreg", "pseudobulk_effect"):
                    df = per_kind.get(kind)
                    if df is not None and not getattr(df, "empty", True):
                        chosen[str(cluster_label)] = df
                        break
            if chosen:
                out.setdefault(str(contrast_key), {})[str(pair_key)] = chosen
    return out


def _write_de_enrichment_tables(
    payloads: Mapping[str, dict],
    *,
    tables_root: Path,
) -> None:
    tables_root.mkdir(parents=True, exist_ok=True)
    for net_name, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        net_dir = tables_root / io_utils.sanitize_identifier(str(net_name), allow_spaces=False)
        net_dir.mkdir(parents=True, exist_ok=True)
        activity = payload.get("activity")
        if isinstance(activity, pd.DataFrame) and not activity.empty:
            activity.to_csv(net_dir / "activity.tsv", sep="\t")
        results = payload.get("results")
        if isinstance(results, pd.DataFrame) and not results.empty:
            results.to_csv(net_dir / "results.tsv", sep="\t", index=False)
        activity_by_gmt = payload.get("activity_by_gmt")
        if isinstance(activity_by_gmt, dict):
            for gmt_key, gmt_df in activity_by_gmt.items():
                if isinstance(gmt_df, pd.DataFrame) and not gmt_df.empty:
                    gmt_name = io_utils.sanitize_identifier(str(gmt_key), allow_spaces=False)
                    gmt_df.to_csv(net_dir / f"activity_{gmt_name}.tsv", sep="\t")


def _de_enrichment_rel_base(
    *,
    source: str,
    condition_key: str,
    contrast: str,
) -> Path:
    if str(source) == "cell":
        return Path(str(condition_key)) / str(contrast)
    if str(contrast) == "interaction" or "^" in str(condition_key):
        return Path(f"{condition_key}__interaction") / "interaction"
    return Path(str(condition_key)) / str(contrast)


def _export_de_enrichment_tables_from_uns(
    adata: ad.AnnData,
    *,
    store_key: str,
    de_cell_dir: Path,
    de_pb_dir: Path,
) -> None:
    de_block = adata.uns.get(store_key, {}).get("de_decoupler", {})
    if not isinstance(de_block, dict) or not de_block:
        return
    for condition_key, per_contrast in de_block.items():
        if not isinstance(per_contrast, dict):
            continue
        for contrast, payload_by_source in per_contrast.items():
            if not isinstance(payload_by_source, dict):
                continue
            for source, payload in payload_by_source.items():
                if not isinstance(payload, dict):
                    continue
                payloads = payload.get("nets", {})
                if not isinstance(payloads, dict) or not payloads:
                    continue
                rel_base = _de_enrichment_rel_base(
                    source=str(source),
                    condition_key=str(condition_key),
                    contrast=str(contrast),
                )
                tables_root = (de_cell_dir if str(source) == "cell" else de_pb_dir) / rel_base
                _write_de_enrichment_tables(payloads, tables_root=tables_root)


def _load_de_enrichment_payload_from_tables(
    payload: dict,
    *,
    net_name: str,
    tables_root: Path,
) -> dict:
    if not isinstance(payload, dict):
        return {}
    full_payload = dict(payload)
    needs_results = "results" not in full_payload
    needs_activity = "activity" not in full_payload
    if not needs_results and not needs_activity:
        return full_payload

    net_dir = Path(tables_root) / io_utils.sanitize_identifier(str(net_name), allow_spaces=False)
    if not net_dir.exists():
        return full_payload

    activity_path = net_dir / "activity.tsv"
    results_path = net_dir / "results.tsv"
    if needs_activity and activity_path.exists():
        try:
            full_payload["activity"] = pd.read_csv(activity_path, sep="\t", index_col=0)
        except Exception:
            LOGGER.warning("Failed to reload DE enrichment activity from %s", str(activity_path))
    if needs_results and results_path.exists():
        try:
            full_payload["results"] = pd.read_csv(results_path, sep="\t")
        except Exception:
            LOGGER.warning("Failed to reload DE enrichment results from %s", str(results_path))
    if str(net_name).lower().strip() == "msigdb" and "activity_by_gmt" not in full_payload:
        by_gmt: dict[str, pd.DataFrame] = {}
        for gmt_path in sorted(net_dir.glob("activity_*.tsv")):
            if gmt_path.name == "activity.tsv":
                continue
            key = gmt_path.stem[len("activity_") :]
            try:
                by_gmt[str(key).upper()] = pd.read_csv(gmt_path, sep="\t", index_col=0)
            except Exception:
                LOGGER.warning("Failed to reload MSigDB activity split from %s", str(gmt_path))
        if by_gmt:
            full_payload["activity_by_gmt"] = by_gmt
    return full_payload


def _compute_de_enrichment_from_dir(
    input_dir: Path,
    *,
    cfg,
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    from .annotation_utils import (
        _run_msigdb_from_stats,
        _run_msigdb_gsea_from_stats,
        _merge_msigdb_decoupler_and_gsea,
        _run_progeny_from_stats,
        _run_dorothea_from_stats,
    )

    pb_tables = _collect_pseudobulk_de_tables_from_dir(input_dir)
    cell_tables = _collect_cell_contrast_tables_from_dir(input_dir)
    if not pb_tables and not cell_tables:
        raise RuntimeError(f"enrichment de: no DE CSV tables found under {input_dir}")

    de_source = str(getattr(cfg, "de_decoupler_source", "auto") or "auto").lower()
    if de_source not in ("auto", "all", "pseudobulk", "cell", "none"):
        raise RuntimeError(f"enrichment de: invalid de_decoupler_source={de_source!r}")
    if de_source == "none":
        return {}

    stat_col = str(getattr(cfg, "de_decoupler_stat_col", "stat") or "stat")
    results: dict[str, dict[str, dict[str, dict[str, object]]]] = {}

    condition_keys = sorted(set(pb_tables.keys()) | set(cell_tables.keys()))
    for condition_key in condition_keys:
        sources: list[tuple[str, dict[str, dict[str, pd.DataFrame]]]] = []
        if de_source in ("auto", "all", "pseudobulk") and condition_key in pb_tables:
            sources.append(("pseudobulk", pb_tables[condition_key]))
        if de_source in ("auto", "all", "cell") and condition_key in cell_tables:
            sources.append(("cell", cell_tables[condition_key]))

        for source, tables_by_contrast in sources:
            for contrast, tables in tables_by_contrast.items():
                stats = _build_stats_matrix_from_tables(
                    tables,
                    preferred_col=stat_col,
                    fallback_cols=("log2FoldChange", "cell_wilcoxon_score", "cell_wilcoxon_logfc", "cell_logreg_coef"),
                )
                if stats is None or stats.empty:
                    continue

                gene_meta = _build_gene_meta_from_de_tables(tables)
                stats, gene_filter_info = _apply_gene_filters_to_de_stats(
                    stats,
                    gene_meta=gene_meta,
                    gene_filter=getattr(cfg, "gene_filter", ()),
                    resource_name=f"enrichment de [{source}:{condition_key}:{contrast}]",
                )
                if stats.empty:
                    LOGGER.info(
                        "enrichment de: gene_filter removed all genes for source=%r condition_key=%r contrast=%r; skipping.",
                        str(source),
                        str(condition_key),
                        str(contrast),
                    )
                    continue

                input_label = f"{source}:{condition_key}:{contrast}:{stat_col}"
                payloads: dict[str, dict] = {}
                LOGGER.info(
                    "DE enrichment: running MSigDB decoupler source=%r condition_key=%r contrast=%r stat_col=%r",
                    str(source),
                    str(condition_key),
                    str(contrast),
                    str(stat_col),
                )
                msigdb_payload = _run_msigdb_from_stats(stats, cfg, input_label=input_label)
                if msigdb_payload is not None:
                    payloads["msigdb"] = msigdb_payload
                if bool(getattr(cfg, "run_gsea", True)):
                    LOGGER.info(
                        "DE enrichment: running MSigDB GSEA source=%r condition_key=%r contrast=%r stat_col=%r",
                        str(source),
                        str(condition_key),
                        str(contrast),
                        str(stat_col),
                    )
                    gsea_payload = _run_msigdb_gsea_from_stats(stats, cfg, input_label=input_label)
                    if gsea_payload is not None:
                        payloads["msigdb_gsea"] = gsea_payload
                        if msigdb_payload is not None:
                            joint_payload = _merge_msigdb_decoupler_and_gsea(
                                decoupler_payload=msigdb_payload,
                                gsea_payload=gsea_payload,
                                alpha=float(getattr(cfg, "joint_enrichment_alpha", 0.05)),
                                leading_edge_top_n=int(getattr(cfg, "joint_enrichment_leading_edge_top_n", 8)),
                            )
                            if joint_payload is not None:
                                payloads["msigdb_joint"] = joint_payload
                if bool(getattr(cfg, "run_progeny", True)):
                    prog = _run_progeny_from_stats(stats, cfg, input_label=input_label)
                    if prog is not None:
                        payloads["progeny"] = prog
                if bool(getattr(cfg, "run_dorothea", True)):
                    doro = _run_dorothea_from_stats(stats, cfg, input_label=input_label)
                    if doro is not None:
                        payloads["dorothea"] = doro
                if not payloads:
                    continue

                results.setdefault(str(condition_key), {}).setdefault(str(contrast), {})[str(source)] = {
                    "source": str(source),
                    "condition_key": str(condition_key),
                    "contrast": str(contrast),
                    "stat_col": str(stat_col),
                    "gene_filter": gene_filter_info,
                    "nets": payloads,
                }

    return results


def run_enrichment_de(cfg) -> None:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (DE enrichment)...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    input_dir = Path(getattr(cfg, "input_dir", None) or getattr(cfg, "input_path"))
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"enrichment de: input_dir not found or not a directory: {input_dir}")

    from . import de_plot_utils

    de_source = str(getattr(cfg, "de_decoupler_source", "auto") or "auto").lower()
    results = _compute_de_enrichment_from_dir(input_dir, cfg=cfg)
    if not results:
        LOGGER.info("enrichment de: no enrichment payloads produced; nothing to write.")
        return None

    run_key = io_utils.sanitize_identifier(
        str(getattr(cfg, "output_name", f"{input_dir.name}.enrichment_de")).replace(".", "_"),
        allow_spaces=False,
    )
    run_round = str(plot_utils.get_run_subdir(run_key))
    tables_root = output_dir / "tables" / run_round
    stat_col = str(getattr(cfg, "de_decoupler_stat_col", "stat") or "stat")
    total_runs = 0

    if bool(getattr(cfg, "make_figures", True)):
        figdir_round = Path(run_round)
    else:
        figdir_round = None

    _write_settings(
        tables_root,
        "__settings.txt",
        [
            f"input_dir={input_dir}",
            f"de_decoupler_source={de_source}",
            f"de_decoupler_stat_col={stat_col}",
            f"gene_filter={list(getattr(cfg, 'gene_filter', ()))}",
        ],
    )

    for condition_key, per_contrast in results.items():
        for contrast, payload_by_source in per_contrast.items():
            for source, payload in payload_by_source.items():
                payloads = payload.get("nets", {})
                gene_filter_info = payload.get("gene_filter", None)
                total_runs += 1
                if str(source) == "cell":
                    base = Path("cell_level_DE") / str(condition_key) / str(contrast)
                else:
                    if str(contrast) == "interaction" or "^" in str(condition_key):
                        base = Path("pseudobulk_DE") / f"{condition_key}__interaction" / "interaction"
                    else:
                        base = Path("pseudobulk_DE") / str(condition_key) / str(contrast)

                run_tables_dir = tables_root / base
                _write_de_enrichment_tables(payloads, tables_root=run_tables_dir)
                if gene_filter_info:
                    _write_settings(
                        run_tables_dir,
                        "__settings.txt",
                        [
                            f"source={source}",
                            f"condition_key={condition_key}",
                            f"contrast={contrast}",
                            f"stat_col={stat_col}",
                            f"gene_filter={list(gene_filter_info['gene_filter'])}",
                            f"gene_filter_n_genes_input={int(gene_filter_info['n_genes_input'])}",
                            f"gene_filter_n_genes_retained={int(gene_filter_info['n_genes_retained'])}",
                        ],
                    )

                if figdir_round is not None:
                    pos_label = None
                    neg_label = None
                    if "_vs_" in str(contrast):
                        parts = str(contrast).split("_vs_", 1)
                        if len(parts) == 2:
                            pos_label, neg_label = parts[0], parts[1]

                    for net_name, net_payload in payloads.items():
                        if str(net_name) == "msigdb_gsea":
                            artifacts = de_plot_utils.plot_de_gsea_payload(
                                net_payload,
                                figdir=base / "msigdb_gsea",
                                title_prefix=f"{condition_key} {contrast}",
                                top_n=int(getattr(cfg, "joint_enrichment_top_n", 20)),
                            )
                        elif str(net_name) == "msigdb_joint":
                            artifacts = de_plot_utils.plot_de_msigdb_joint_payload(
                                net_payload,
                                figdir=base / "msigdb_joint",
                                title_prefix=f"{condition_key} {contrast}",
                                top_n=int(getattr(cfg, "joint_enrichment_top_n", 20)),
                                require_gsea_sig=bool(getattr(cfg, "joint_enrichment_require_gsea_sig", True)),
                            )
                        else:
                            artifacts = de_plot_utils.plot_de_decoupler_payload(
                                net_payload,
                                net_name=str(net_name),
                                figdir=base,
                                heatmap_top_k=int(getattr(cfg, "plot_max_genes_total", 80)),
                                bar_top_n=10,
                                bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                                bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                                bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                                dotplot_top_k=25,
                                title_prefix=f"{condition_key} {contrast}",
                                pos_label=pos_label,
                                neg_label=neg_label,
                            )
                        plot_utils.persist_plot_artifacts(artifacts)

    if total_runs == 0:
        raise RuntimeError(f"enrichment de: no enrichment payloads were produced from DE tables in {input_dir}")

    if figdir_round is not None:
        try:
            LOGGER.info("enrichment de: generating report...")
            for fmt in getattr(cfg, "figure_formats", ["png", "pdf"]):
                reporting.generate_enrichment_de_report(
                    fig_root=figdir,
                    fmt=fmt,
                    cfg=cfg,
                    version=__version__,
                    run_dir=figdir / str(fmt).lower().lstrip(".") / run_round,
                )
            LOGGER.info("Wrote enrichment de report.")
        except Exception as e:
            LOGGER.warning("Failed to generate enrichment de report: %s", e)

    LOGGER.info("Finished markers-and-de (DE enrichment); runs=%d", int(total_runs))
    return None


def _import_liana_module() -> Any:
    try:
        import liana as li
    except ImportError as exc:
        raise RuntimeError(
            "markers-and-de ccc liana requires the Python package `liana`. "
            "Install it in the scOmnom environment before running this command."
        ) from exc
    return li


def _liana_method_specs(li: Any) -> dict[str, dict[str, Any]]:
    method_mod = getattr(li, "method")
    return {
        "cellphonedb": {
            "callable": getattr(method_mod, "cellphonedb"),
            "score_col": "lr_means",
            "score_ascending": False,
            "specificity_col": "cellphone_pvals",
            "specificity_ascending": True,
        },
        "connectome": {
            "callable": getattr(method_mod, "connectome"),
            "score_col": "scaled_weight",
            "score_ascending": False,
            "specificity_col": None,
            "specificity_ascending": True,
        },
        "natmi": {
            "callable": getattr(method_mod, "natmi"),
            "score_col": "spec_weight",
            "score_ascending": False,
            "specificity_col": None,
            "specificity_ascending": True,
        },
        "sca": {
            "callable": getattr(method_mod, "singlecellsignalr"),
            "score_col": "lrscore",
            "score_ascending": False,
            "specificity_col": None,
            "specificity_ascending": True,
        },
        "logfc": {
            "callable": getattr(method_mod, "logfc"),
            "score_col": "lr_logfc",
            "score_ascending": False,
            "specificity_col": None,
            "specificity_ascending": True,
        },
    }


def _make_liana_rank_aggregate(li: Any, method_names: Sequence[str]) -> Any:
    selected = [m for m in method_names if str(m) != "rank_aggregate"]
    specs = _liana_method_specs(li)
    if not selected:
        selected = list(_LIANA_DEFAULT_AGGREGATE_METHODS)
    methods = [specs[str(name)]["callable"] for name in selected]
    return getattr(li, "mt").AggregateClass(getattr(li, "mt").aggregate_meta, methods=methods)


def _sort_liana_results(
    df: pd.DataFrame,
    *,
    score_col: Optional[str],
    score_ascending: bool,
    specificity_col: Optional[str],
    specificity_ascending: bool,
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    out = df.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if specificity_col and specificity_col in out.columns:
        out[specificity_col] = pd.to_numeric(out[specificity_col], errors="coerce")
        sort_cols.append(str(specificity_col))
        ascending.append(bool(specificity_ascending))
    if score_col and score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
        sort_cols.append(str(score_col))
        ascending.append(bool(score_ascending))
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return out.reset_index(drop=True)


def _summarize_liana_source_targets(
    df: pd.DataFrame,
    *,
    score_col: Optional[str],
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["source", "target", "n_interactions"])
    if "source" not in df.columns or "target" not in df.columns:
        return pd.DataFrame(columns=["source", "target", "n_interactions"])

    work = df.copy()
    work["source"] = work["source"].astype(str)
    work["target"] = work["target"].astype(str)
    agg = (
        work.groupby(["source", "target"], observed=False)
        .size()
        .rename("n_interactions")
        .reset_index()
    )
    if score_col and score_col in work.columns:
        score = (
            work.groupby(["source", "target"], observed=False)[score_col]
            .mean()
            .rename("mean_score")
            .reset_index()
        )
        agg = agg.merge(score, on=["source", "target"], how="left")
    return agg.sort_values(["n_interactions", "source", "target"], ascending=[False, True, True], kind="mergesort")


def _liana_plot_label(value: object, display_map: Mapping[str, str]) -> str:
    raw = str(value)
    pretty = str(display_map.get(raw, raw))
    token = plot_utils._extract_cnn_token(pretty)
    return str(token or raw)


def _prepare_liana_plot_df(df: pd.DataFrame, *, display_map: Mapping[str, str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if getattr(df, "empty", True):
        return df.copy()
    out = df.copy()
    for col in ("source", "target"):
        if col in out.columns:
            out[col] = out[col].astype(str).map(lambda x: _liana_plot_label(x, display_map))
    return out


def _liana_family_label(value: object, display_map: Mapping[str, str]) -> str:
    raw = str(value)
    pretty = str(display_map.get(raw, raw)).strip()
    pretty = re.sub(r"^\s*C\d+\s*[:\-]?\s*", "", pretty)
    return pretty or raw


def _liana_signal_family(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Unknown"
    token = re.split(r"[_:;|]", raw, maxsplit=1)[0].strip()
    m = re.match(r"([A-Za-z-]+)", token)
    if m:
        family = m.group(1).rstrip("-")
        if family:
            return family
    return token or raw


def _normalize_cellchat_lr_token(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = raw.replace(" ", "")
    raw = raw.strip("()")
    raw = raw.replace("+", "_")
    raw = raw.replace("-", "_")
    raw = raw.replace("/", "_")
    raw = raw.replace(":", "_")
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw.upper()


def _parse_cellchat_interaction_name_2(value: object) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw or " - " not in raw:
        return None
    ligand, receptor = raw.split(" - ", 1)
    ligand_key = _normalize_cellchat_lr_token(ligand)
    receptor_key = _normalize_cellchat_lr_token(receptor)
    if not ligand_key or not receptor_key:
        return None
    return ligand_key, receptor_key


@lru_cache(maxsize=1)
def _load_cellchatdb_pathway_lookup() -> dict[tuple[str, str], tuple[str, str]]:
    if not _CELLCHATDB_INTERACTION_ANNOTATIONS.exists():
        LOGGER.warning(
            "CellChatDB interaction annotations not found at %s; using LIANA route-family heuristic only.",
            _CELLCHATDB_INTERACTION_ANNOTATIONS,
        )
        return {}

    df = pd.read_csv(_CELLCHATDB_INTERACTION_ANNOTATIONS, sep="\t", dtype=str).fillna("")
    by_key: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in df.itertuples(index=False):
        pathway_name = str(getattr(row, "pathway_name", "")).strip()
        annotation = str(getattr(row, "annotation", "")).strip()
        if not pathway_name:
            continue
        keys: set[tuple[str, str]] = set()
        ligand_key = _normalize_cellchat_lr_token(getattr(row, "ligand", ""))
        receptor_key = _normalize_cellchat_lr_token(getattr(row, "receptor", ""))
        if ligand_key and receptor_key:
            keys.add((ligand_key, receptor_key))
        parsed_key = _parse_cellchat_interaction_name_2(getattr(row, "interaction_name_2", ""))
        if parsed_key is not None:
            keys.add(parsed_key)
        for key in keys:
            by_key.setdefault(key, set()).add((pathway_name, annotation))

    resolved: dict[tuple[str, str], tuple[str, str]] = {}
    for key, hits in by_key.items():
        resolved[key] = sorted(hits, key=lambda x: (x[0], x[1]))[0]
    return resolved


def _lookup_cellchat_route_family(ligand: object, receptor: object) -> tuple[str, str] | None:
    ligand_key = _normalize_cellchat_lr_token(ligand)
    receptor_key = _normalize_cellchat_lr_token(receptor)
    if not ligand_key or not receptor_key:
        return None
    return _load_cellchatdb_pathway_lookup().get((ligand_key, receptor_key))


def _lookup_cellchat_route_info(ligand: object, receptor: object) -> dict[str, str]:
    hit = _lookup_cellchat_route_family(ligand, receptor)
    if hit is None:
        return {"pathway_name": "", "annotation": ""}
    pathway_name, annotation = hit
    return {
        "pathway_name": str(pathway_name or ""),
        "annotation": str(annotation or ""),
    }


def _liana_route_family_heuristic(ligand: object, receptor: object) -> str:
    ligand = str(ligand or "").strip()
    receptor = str(receptor or "").strip()

    if (
        receptor.startswith("TLR")
        or receptor.startswith("C5AR")
        or receptor.startswith("C3AR")
        or receptor in {"CD93", "LILRB2", "LILRB3"}
        or ligand in {"HMGB1", "CD14", "IRAK4", "HSPA4"}
    ):
        return "Innate sensing / complement"
    if (
        receptor.startswith("ITGA")
        or receptor.startswith("ITGB")
        or receptor in {"CD44", "SDC2", "PLAUR", "PTPRJ", "CADM1", "APLP2", "ACKR3"}
        or ligand in {"ADAM9", "ADAM10", "CXCL12", "PDGFB", "ITGB3BP", "F13A1", "TIMP2", "TLN1"}
    ):
        return "ECM / adhesion"
    if (
        receptor.startswith("EGFR")
        or receptor.startswith("ERBB")
        or receptor.startswith("FGFR")
        or receptor in {"MET", "NRP1", "ASGR1", "ASGR2", "LDLR", "INSR", "IGF1R", "IL6R"}
        or ligand in {"HGF", "FGF13", "GRN", "S100A4", "PTPN6", "LRIG2", "HP", "TGFA", "IGF1", "ANXA1", "NAMPT", "ARF1", "GNAI2"}
    ):
        return "Growth factor / receptor handling"
    if receptor.startswith("NOTCH") or ligand == "MAML2":
        return "Notch / juxtacrine"
    if (
        receptor in {"LRP1", "ABCA1", "AR", "SORT1", "SORL1", "VLDLR"}
        or ligand in {"APOA1", "A2M", "PSAP", "LRPAP1", "ACTR2", "PLTP"}
    ):
        return "Scavenger / metabolic handling"
    if (
        receptor.startswith("FCGR")
        or receptor.startswith("KLR")
        or receptor in {"PTPRK", "B2M_FCGRT", "PILRB", "CD81", "CD99L2", "TYRO3"}
        or ligand in {"LGALS9", "ALB", "B2M", "CD99", "GAS6", "PROS1", "CD200R1"}
    ):
        return "Immune recognition"
    if (
        receptor.startswith("PLXN")
        or receptor.startswith("EPH")
        or ligand.startswith("EFN")
        or ligand in {"AFDN", "ANG", "FARP2", "RTN4"}
    ):
        return "Guidance / interface"
    if receptor.startswith("IL6R") or ligand in {"ADAM10", "ADAM17"}:
        return "Protease / cytokine interface"
    return "Other"


def _liana_route_family(ligand: object, receptor: object) -> str:
    cellchat_hit = _lookup_cellchat_route_family(ligand, receptor)
    if cellchat_hit is not None:
        return str(cellchat_hit[0])
    return _liana_route_family_heuristic(ligand, receptor)


def _prepare_liana_family_plot_df(df: pd.DataFrame, *, display_map: Mapping[str, str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if getattr(df, "empty", True):
        return df.copy()
    out = _prepare_liana_plot_df(df, display_map=display_map)
    if "source" in df.columns:
        out["source_family"] = df["source"].astype(str).map(lambda x: _liana_family_label(x, display_map))
    if "target" in df.columns:
        out["target_family"] = df["target"].astype(str).map(lambda x: _liana_family_label(x, display_map))
    if "ligand_complex" in df.columns:
        out["ligand_family"] = df["ligand_complex"].astype(str).map(_liana_signal_family)
    if "receptor_complex" in df.columns:
        out["receptor_family"] = df["receptor_complex"].astype(str).map(_liana_signal_family)
    if {"ligand_complex", "receptor_complex"}.issubset(df.columns):
        route_info = [
            _lookup_cellchat_route_info(lig, rec)
            for lig, rec in zip(df["ligand_complex"], df["receptor_complex"])
        ]
        out["route_annotation"] = [x.get("annotation", "") for x in route_info]
        out["route_family"] = [
            _liana_route_family(lig, rec)
            for lig, rec in zip(df["ligand_complex"], df["receptor_complex"])
        ]
    return out


def _summarize_liana_route_families(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["source", "route_family", "n_interactions"])
    if not {"source", "ligand_complex", "receptor_complex"}.issubset(df.columns):
        return pd.DataFrame(columns=["source", "route_family", "n_interactions"])
    work = df.copy()
    work["source"] = work["source"].astype(str)
    work["route_family"] = [
        _liana_route_family(lig, rec)
        for lig, rec in zip(work["ligand_complex"], work["receptor_complex"])
    ]
    out = (
        work.groupby(["source", "route_family"], observed=False)
        .size()
        .rename("n_interactions")
        .reset_index()
        .sort_values(["source", "n_interactions", "route_family"], ascending=[True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def _resolve_liana_cluster_dataset_levels(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    dataset_key: str,
) -> dict[str, str]:
    if str(cluster_key) not in adata.obs or str(dataset_key) not in adata.obs:
        raise RuntimeError(
            f"ccc liana: cluster_key={cluster_key!r} or dataset_key={dataset_key!r} not found in adata.obs"
        )
    tab = pd.crosstab(
        adata.obs[str(cluster_key)].astype(str),
        adata.obs[str(dataset_key)].astype(str),
        dropna=False,
    )
    out: dict[str, str] = {}
    for cluster in tab.index.astype(str):
        counts = tab.loc[cluster]
        counts = counts[counts > 0]
        if counts.empty:
            continue
        if len(counts.index) > 1:
            LOGGER.warning(
                "ccc liana: cluster %r spans multiple %s levels %s; using dominant level %r for cross-tissue filtering.",
                str(cluster),
                str(dataset_key),
                list(counts.index.astype(str)),
                str(counts.sort_values(ascending=False).index[0]),
            )
        out[str(cluster)] = str(counts.sort_values(ascending=False).index[0])
    return out


def _filter_liana_results_cross_tissue(
    df: pd.DataFrame,
    *,
    cluster_dataset_map: Mapping[str, str],
    source_levels: Sequence[str],
    target_levels: Sequence[str],
    signal_scope: str,
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame() if df is None else df.copy()
    if not {"source", "target"}.issubset(df.columns):
        return df.copy()

    out = df.copy()
    out["source_dataset_level"] = out["source"].astype(str).map(lambda x: cluster_dataset_map.get(str(x), ""))
    out["target_dataset_level"] = out["target"].astype(str).map(lambda x: cluster_dataset_map.get(str(x), ""))
    source_allowed = {str(x) for x in source_levels if str(x)}
    target_allowed = {str(x) for x in target_levels if str(x)}
    mask = (
        out["source_dataset_level"].astype(str).isin(source_allowed)
        & out["target_dataset_level"].astype(str).isin(target_allowed)
    )
    out = out.loc[mask].copy()
    if out.empty:
        return out

    if str(signal_scope).strip().lower() == "secreted" and {"ligand_complex", "receptor_complex"}.issubset(out.columns):
        route_info = [
            _lookup_cellchat_route_info(lig, rec)
            for lig, rec in zip(out["ligand_complex"], out["receptor_complex"])
        ]
        out["route_annotation"] = [x.get("annotation", "") for x in route_info]
        out = out[out["route_annotation"].astype(str) == "Secreted Signaling"].copy()
    return out.reset_index(drop=True)


def _resolve_cluster_request(requested: str, display_map: Mapping[str, str]) -> str:
    raw = str(requested).strip()
    if not raw:
        raise RuntimeError("Empty cluster label request.")
    if raw in display_map:
        return raw
    requested_token = plot_utils._extract_cnn_token(raw)
    for cluster_id, label in display_map.items():
        label_str = str(label)
        if raw == label_str:
            return str(cluster_id)
        label_token = plot_utils._extract_cnn_token(label_str)
        if requested_token and requested_token == label_token:
            return str(cluster_id)
    raise RuntimeError(f"Requested cluster {requested!r} could not be matched to the active round labels.")


def _get_matrix_slice(
    adata: ad.AnnData,
    *,
    layer: Optional[str],
    use_raw: bool,
    mask: np.ndarray,
):
    if layer:
        matrix = adata.layers[str(layer)]
    elif use_raw:
        if adata.raw is None:
            raise RuntimeError("Requested use_raw=True but adata.raw is not initialized.")
        matrix = adata.raw.X
    else:
        matrix = adata.X
    return matrix[mask]


def _compute_expressed_genes(
    adata: ad.AnnData,
    *,
    mask: np.ndarray,
    layer: Optional[str],
    use_raw: bool,
    min_fraction: float,
) -> list[str]:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if int(mask.sum()) == 0:
        return []
    matrix = _get_matrix_slice(adata, layer=layer, use_raw=use_raw, mask=mask)
    if sp.issparse(matrix):
        detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    else:
        detected = np.asarray(matrix > 0).sum(axis=0)
    frac = detected / float(mask.sum())
    return [str(g) for g, keep in zip(adata.var_names.astype(str), frac >= float(min_fraction)) if bool(keep)]


def _load_gene_list_file(path: Path) -> list[str]:
    genes: list[str] = []
    for line in Path(path).read_text().splitlines():
        gene = str(line).strip()
        if gene:
            genes.append(gene)
    return sorted(dict.fromkeys(genes))


def _extract_receiver_de_geneset(
    adata_receiver: ad.AnnData,
    *,
    compare_key: str,
    condition_oi: str,
    condition_reference: str,
    min_logfc: float,
    padj_threshold: float,
) -> tuple[list[str], list[str], pd.DataFrame]:
    import scanpy as sc

    if str(compare_key) not in adata_receiver.obs:
        raise RuntimeError(f"NicheNet compare_key={compare_key!r} not found in receiver obs.")
    work = adata_receiver.copy()
    work.obs[str(compare_key)] = _normalize_levels(work.obs[str(compare_key)]).astype(str)
    sc.tl.rank_genes_groups(
        work,
        groupby=str(compare_key),
        groups=[str(condition_oi)],
        reference=str(condition_reference),
        method="wilcoxon",
        use_raw=False,
        pts=False,
    )
    de_df = sc.get.rank_genes_groups_df(work, group=str(condition_oi))
    if "names" in de_df.columns:
        de_df = de_df.rename(columns={"names": "gene"})
    de_df["gene"] = de_df["gene"].astype(str)
    bg_genes = sorted(dict.fromkeys(work.var_names.astype(str).tolist()))
    keep = pd.Series(True, index=de_df.index)
    if "logfoldchanges" in de_df.columns:
        keep &= de_df["logfoldchanges"].fillna(-np.inf) >= float(min_logfc)
    if "pvals_adj" in de_df.columns:
        keep &= de_df["pvals_adj"].fillna(np.inf) <= float(padj_threshold)
    geneset = sorted(dict.fromkeys(de_df.loc[keep, "gene"].astype(str).tolist()))
    return geneset, bg_genes, de_df


def _nichenet_condition_spec_token(raw_spec: str) -> str:
    token = str(raw_spec).strip()
    token = token.replace("@", "_at_").replace(":", "_and_").replace("^", "_x_")
    return _safe_combo_token(token)


def _build_nichenet_condition_run_specs(adata: ad.AnnData, cfg) -> list[dict[str, Any]]:
    cond_keys = [str(x).strip() for x in (getattr(cfg, "ccc_condition_keys", ()) or ()) if str(x).strip()]
    if not cond_keys:
        ck = getattr(cfg, "ccc_condition_key", None)
        if ck:
            cond_keys = [str(ck).strip()]
    condition_values = tuple(str(x) for x in (getattr(cfg, "ccc_condition_values", ()) or ()) if str(x))
    compare_levels = tuple(str(x) for x in (getattr(cfg, "ccc_compare_levels", ()) or ()) if str(x))
    gene_list_file = getattr(cfg, "nichenet_gene_list_file", None)

    if gene_list_file and not cond_keys:
        return [
            {
                "run_id": "global",
                "run_label": "global",
                "adata": adata,
                "compare_key": None,
                "condition_oi": None,
                "condition_reference": None,
                "condition_spec": None,
                "context_key": None,
                "context_value": None,
                "tables_rel": Path("."),
                "figs_rel": Path("."),
            }
        ]

    if not cond_keys:
        raise RuntimeError("NicheNet requires either a gene list file or at least one condition spec.")
    if len(compare_levels) != 2:
        raise RuntimeError("NicheNet receiver-DE mode requires exactly two compare levels.")

    run_specs: list[dict[str, Any]] = []
    condition_oi, condition_reference = str(compare_levels[0]), str(compare_levels[1])
    for raw_spec in cond_keys:
        spec_token = _nichenet_condition_spec_token(raw_spec)
        spec_root = Path(f"condition__{spec_token}")
        if "@" in raw_spec:
            parts = [p.strip() for p in raw_spec.split("@") if p.strip()]
            if len(parts) != 2:
                raise RuntimeError(f"ccc nichenet: invalid A@B condition specification {raw_spec!r}.")
            a_key = _resolve_condition_key(adata, parts[0])
            b_key = _resolve_condition_key(adata, parts[1])
            b_series = _normalize_levels(adata.obs[b_key])
            wanted_b = sorted(set(condition_values) if condition_values else set(b_series.unique().tolist()))
            for b_level in wanted_b:
                mask_b = b_series.astype(str).to_numpy() == str(b_level)
                if not np.any(mask_b):
                    continue
                adata_b = adata[mask_b].copy()
                a_levels = set(_normalize_levels(adata_b.obs[a_key]).astype(str).unique().tolist())
                if str(condition_oi) not in a_levels or str(condition_reference) not in a_levels:
                    LOGGER.warning(
                        "ccc nichenet: skipping %s=%r because compare levels %r and %r are not both present.",
                        str(b_key),
                        str(b_level),
                        str(condition_oi),
                        str(condition_reference),
                    )
                    continue
                context_token = f"{_safe_combo_token(b_key)}={_safe_combo_token(b_level)}"
                run_specs.append(
                    {
                        "run_id": f"{raw_spec}::{context_token}::{condition_oi}_vs_{condition_reference}",
                        "run_label": f"{condition_oi}_vs_{condition_reference}",
                        "adata": adata_b,
                        "compare_key": str(a_key),
                        "condition_oi": str(condition_oi),
                        "condition_reference": str(condition_reference),
                        "condition_spec": raw_spec,
                        "context_key": str(b_key),
                        "context_value": str(b_level),
                        "tables_rel": spec_root / context_token,
                        "figs_rel": spec_root / context_token,
                    }
                )
        else:
            compare_key = _resolve_condition_key(adata, raw_spec)
            levels = set(_normalize_levels(adata.obs[compare_key]).astype(str).unique().tolist())
            if str(condition_oi) not in levels or str(condition_reference) not in levels:
                LOGGER.warning(
                    "ccc nichenet: skipping spec=%r because compare levels %r and %r are not both present.",
                    raw_spec,
                    str(condition_oi),
                    str(condition_reference),
                )
                continue
            run_specs.append(
                {
                    "run_id": f"{raw_spec}::{condition_oi}_vs_{condition_reference}",
                    "run_label": f"{condition_oi}_vs_{condition_reference}",
                    "adata": adata,
                    "compare_key": str(compare_key),
                    "condition_oi": str(condition_oi),
                    "condition_reference": str(condition_reference),
                    "condition_spec": raw_spec,
                    "context_key": None,
                    "context_value": None,
                    "tables_rel": spec_root,
                    "figs_rel": spec_root,
                }
            )
    return run_specs


def _run_nichenet_sender_focused(
    *,
    sender_expressed_genes: Sequence[str],
    receiver_expressed_genes: Sequence[str],
    geneset_oi: Sequence[str],
    background_genes: Sequence[str],
    top_n_ligands: int,
    top_n_targets: int,
    organism: str,
    signal_scope: str,
    r_env: Optional[Mapping[str, str]] = None,
) -> dict[str, pd.DataFrame]:
    rscript = _resolve_rscript()
    if not rscript:
        raise RuntimeError("NicheNet requires `Rscript` on PATH.")
    if not _NICHENET_R_HELPER.exists():
        raise RuntimeError(f"NicheNet helper script not found at {_NICHENET_R_HELPER}.")

    with tempfile.TemporaryDirectory(prefix="scomnom_nichenet_") as tmpdir:
        tmp = Path(tmpdir)
        sender_path = tmp / "sender_expressed_genes.txt"
        receiver_path = tmp / "receiver_expressed_genes.txt"
        geneset_path = tmp / "geneset_oi.txt"
        bg_path = tmp / "background_genes.txt"
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        sender_path.write_text("\n".join(str(x) for x in sender_expressed_genes))
        receiver_path.write_text("\n".join(str(x) for x in receiver_expressed_genes))
        geneset_path.write_text("\n".join(str(x) for x in geneset_oi))
        bg_path.write_text("\n".join(str(x) for x in background_genes))
        cfg_path = tmp / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "sender_expressed_genes_file": str(sender_path),
                    "receiver_expressed_genes_file": str(receiver_path),
                    "geneset_file": str(geneset_path),
                    "background_genes_file": str(bg_path),
                    "output_dir": str(out_dir),
                    "top_n_ligands": int(top_n_ligands),
                    "top_n_targets": int(top_n_targets),
                    "organism": str(organism),
                    "signal_scope": str(signal_scope),
                }
            )
        )
        res = subprocess.run(
            [str(rscript), str(_NICHENET_R_HELPER), str(cfg_path)],
            check=False,
            capture_output=True,
            text=True,
            env=dict(r_env) if r_env is not None else None,
        )
        if res.returncode != 0:
            stderr = str(res.stderr or "").strip()
            stdout = str(res.stdout or "").strip()
            raise RuntimeError(
                "NicheNet R helper failed."
                + (f"\nSTDERR:\n{stderr}" if stderr else "")
                + (f"\nSTDOUT:\n{stdout}" if stdout else "")
            )
        outputs = {
            "ligand_activity": out_dir / "ligand_activity.tsv",
            "ligand_target_links": out_dir / "ligand_target_links.tsv",
            "ligand_receptor_links": out_dir / "ligand_receptor_links.tsv",
            "potential_ligands": out_dir / "potential_ligands.tsv",
        }
        result: dict[str, pd.DataFrame] = {}
        for key, path in outputs.items():
            result[key] = pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()
        return result


def _import_mebocost_api(*, install_missing: bool) -> Any:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

    def _resolve_loaded_api() -> Any:
        candidates = [
            ("mebocost.mebocost", None),
            ("mebocost", None),
            ("mebocost", "mebocost"),
            ("MEBOCOST.mebocost", None),
            ("MEBOCOST", None),
            ("MEBOCOST", "mebocost"),
        ]
        errors: list[str] = []
        for module_name, attr_name in candidates:
            try:
                mod = importlib.import_module(module_name)
            except Exception as e:
                errors.append(f"{module_name}: {type(e).__name__}: {e}")
                continue
            api = getattr(mod, attr_name, None) if attr_name else mod
            if api is not None and hasattr(api, "create_obj"):
                return api
        return errors

    api_or_errors = _resolve_loaded_api()
    if not isinstance(api_or_errors, list):
        return api_or_errors
    import_errors = api_or_errors

    api = None
    if api is not None:
        return api

    install_hint = f"{sys.executable} -m pip install '{_MEBOCOST_GIT_SPEC}'"
    if not install_missing:
        raise RuntimeError(
            "markers-and-de ccc mebocost requires the Python package `MEBOCOST`.\n"
            "Install it into the active environment, or rerun with `--install-missing-python-deps`.\n"
            f"Suggested install command:\n{install_hint}"
            + (f"\nImport attempts:\n" + "\n".join(import_errors) if import_errors else "")
        )

    res = subprocess.run(
        [sys.executable, "-m", "pip", "install", _MEBOCOST_GIT_SPEC],
        check=False,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        stderr = str(res.stderr or "").strip()
        stdout = str(res.stdout or "").strip()
        raise RuntimeError(
            "Automatic MEBOCOST dependency installation failed.\n"
            f"Suggested install command:\n{install_hint}"
            + (f"\nSTDERR:\n{stderr}" if stderr else "")
            + (f"\nSTDOUT:\n{stdout}" if stdout else "")
        )

    api_or_errors = _resolve_loaded_api()
    if isinstance(api_or_errors, list):
        raise RuntimeError(
            "Automatic MEBOCOST dependency installation completed without a detectable MEBOCOST import.\n"
            f"Suggested install command:\n{install_hint}"
            + (f"\nImport attempts:\n" + "\n".join(api_or_errors) if api_or_errors else "")
        )
    api = api_or_errors
    LOGGER.warning("Installed `MEBOCOST` into the active Python environment.")
    return api


def _mebocost_resource_file_map(repo_dir: Path) -> dict[str, Path]:
    data_dir = repo_dir / "data"
    human_dir = data_dir / "mebocost_db" / "human"
    mouse_dir = data_dir / "mebocost_db" / "mouse"

    def _resolve_sensor_file(base_dir: Path, patterns: Sequence[str]) -> Path:
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(base_dir.glob(pattern)))
        if matches:
            return matches[-1]
        fallback = base_dir / patterns[0]
        return fallback

    return {
        "hmdb_info_path": data_dir / "mebocost_db" / "common" / "metabolite_annotation_HMDB_summary.tsv",
        "scfea_info_path": data_dir / "scFEA" / "Human_M168_information.symbols.csv",
        "compass_rxt_ann_path": data_dir / "Compass" / "rxn_md.csv",
        "compass_met_ann_path": data_dir / "Compass" / "met_md.csv",
        "human_met_enzyme_path": human_dir / "metabolite_associated_gene_reaction_HMDB_summary.tsv",
        "human_met_sensor_path": _resolve_sensor_file(
            human_dir,
            ("human_met_sensor_update_*.tsv", "met_sen_*.tsv"),
        ),
        "mouse_met_enzyme_path": mouse_dir / "metabolite_associated_gene_reaction_HMDB_summary_mouse.tsv",
        "mouse_met_sensor_path": _resolve_sensor_file(
            mouse_dir,
            ("mouse_met_sensor_update_*.tsv", "mouse_met_sen_*.tsv"),
        ),
    }


def _write_mebocost_resource_config(repo_dir: Path, config_path: Path) -> Path:
    files = _mebocost_resource_file_map(repo_dir)
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "MEBOCOST resource bootstrap is incomplete; required files are missing.\n"
            + "\n".join(missing)
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = "\n".join(
        [
            "[common]",
            f"hmdb_info_path = {files['hmdb_info_path']}",
            f"scfea_info_path = {files['scfea_info_path']}",
            f"compass_rxt_ann_path = {files['compass_rxt_ann_path']}",
            f"compass_met_ann_path = {files['compass_met_ann_path']}",
            "",
            "[human]",
            f"met_enzyme_path = {files['human_met_enzyme_path']}",
            f"met_sensor_path = {files['human_met_sensor_path']}",
            "",
            "[mouse]",
            f"met_enzyme_path = {files['mouse_met_enzyme_path']}",
            f"met_sensor_path = {files['mouse_met_sensor_path']}",
            "",
        ]
    )
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _read_mebocost_resource_config(config_path: Path | str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(str(config_path))
    out: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        out[str(section)] = {str(k): str(v) for k, v in parser.items(section)}
    return out


@lru_cache(maxsize=4)
def _load_mebocost_annotation_tables(config_path: str, species: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _read_mebocost_resource_config(config_path)
    common = config.get("common", {})
    species_key = str(species).strip().lower()
    species_cfg = config.get(species_key, {})
    hmdb_path = common.get("hmdb_info_path", "")
    sensor_path = species_cfg.get("met_sensor_path", "")
    hmdb = pd.read_csv(hmdb_path, sep="\t") if hmdb_path else pd.DataFrame()
    sensor = pd.read_csv(sensor_path, sep="\t") if sensor_path else pd.DataFrame()
    return hmdb, sensor


def _ensure_mebocost_resource_config(*, install_missing: bool) -> Path:
    repo_dir = _MEBOCOST_RESOURCE_REPO
    config_path = _MEBOCOST_RESOURCE_CONFIG
    file_map = _mebocost_resource_file_map(repo_dir)
    if config_path.exists() and all(path.exists() for path in file_map.values()):
        return config_path

    clone_hint = (
        f"tmpdir=$(mktemp -d) && git clone --depth 1 https://github.com/kaifuchenlab/MEBOCOST.git "
        f"\"$tmpdir/MEBOCOST\" && mkdir -p \"{_MEBOCOST_CACHE_ROOT}\" && "
        f"rm -rf \"{repo_dir}\" && mv \"$tmpdir/MEBOCOST\" \"{repo_dir}\""
    )
    if not install_missing:
        raise RuntimeError(
            "markers-and-de ccc mebocost requires the upstream MEBOCOST resource database and config files.\n"
            "Install them into the local cache by rerunning with `--install-missing-python-deps`, or bootstrap them manually.\n"
            f"Suggested bootstrap command:\n{clone_hint}"
        )

    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError(
            "Automatic MEBOCOST resource bootstrap failed because `git` is not available on PATH.\n"
            f"Suggested bootstrap command:\n{clone_hint}"
        )
    _MEBOCOST_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        tmp_root = Path(tempfile.mkdtemp(prefix="scomnom_mebocost_"))
        clone_dir = tmp_root / "MEBOCOST"
        try:
            res = subprocess.run(
                [git_bin, "clone", "--depth", "1", "https://github.com/kaifuchenlab/MEBOCOST.git", str(clone_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                stderr = str(res.stderr or "").strip()
                stdout = str(res.stdout or "").strip()
                raise RuntimeError(
                    "Automatic MEBOCOST resource bootstrap failed.\n"
                    f"Suggested bootstrap command:\n{clone_hint}"
                    + (f"\nSTDERR:\n{stderr}" if stderr else "")
                    + (f"\nSTDOUT:\n{stdout}" if stdout else "")
                )
            shutil.move(str(clone_dir), str(repo_dir))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    config_path = _write_mebocost_resource_config(repo_dir, config_path)
    LOGGER.warning("Bootstrapped MEBOCOST resources into local cache: %s", str(repo_dir))
    return config_path


def _call_with_supported_kwargs(func, **kwargs):
    try:
        sig = inspect.signature(func)
    except Exception:
        return func(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**accepted)


def _standardize_mebocost_commu_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    out = df.copy()
    rename_map: dict[str, str] = {}
    candidates = {
        "source": ("sender", "Sender", "cell_type_sender", "Cell_Type_Sender"),
        "target": ("receiver", "Receiver", "cell_type_receiver", "Cell_Type_Receiver"),
        "metabolite": ("metabolite", "Metabolite"),
        "sensor": ("sensor", "Sensor", "receptor", "Receptor"),
        "enzyme": ("enzyme", "Enzyme"),
        "commu_score": ("commu_score", "Commu_Score", "score", "Score"),
        "pval": ("pval", "Pvalue", "p_value", "fdr", "FDR", "permutation_test_fdr"),
    }
    for dest, options in candidates.items():
        for option in options:
            if option in out.columns:
                rename_map[option] = dest
                break
    out = out.rename(columns=rename_map)
    for col in ("source", "target", "metabolite", "sensor", "enzyme"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ("commu_score", "pval"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _summarize_mebocost_source_target(commu_res: pd.DataFrame) -> pd.DataFrame:
    if commu_res is None or getattr(commu_res, "empty", True):
        return pd.DataFrame(columns=["source", "target", "n_events", "mean_score"])
    if not {"source", "target"}.issubset(commu_res.columns):
        return pd.DataFrame(columns=["source", "target", "n_events", "mean_score"])
    work = commu_res.copy()
    group = work.groupby(["source", "target"], observed=False)
    out = group.size().rename("n_events").reset_index()
    if "commu_score" in work.columns:
        mean_score = (
            group["commu_score"]
            .mean()
            .rename("mean_score")
            .reset_index()
        )
        out = out.merge(mean_score, on=["source", "target"], how="left")
    else:
        out["mean_score"] = np.nan
    out = out.sort_values(
        ["n_events", "mean_score", "source", "target"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _prepare_mebocost_plot_df(
    df: pd.DataFrame,
    *,
    display_map: Mapping[str, str],
    valid_group_tokens: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    valid_tokens = {str(x) for x in (valid_group_tokens or ()) if str(x)}

    def _resolve_plot_group(value: object) -> str:
        raw = str(value).strip()
        if not raw:
            return raw
        if raw in valid_tokens:
            return raw
        raw_token = plot_utils._extract_cnn_token(raw)
        if raw_token and raw_token in valid_tokens:
            return raw_token
        return _resolve_cluster_request(raw, display_map)

    if "metabolite" in out.columns:
        met_raw = out["metabolite"].astype(str)
        met_name = (
            out["Metabolite_Name"].astype(str)
            if "Metabolite_Name" in out.columns
            else pd.Series([""] * len(out), index=out.index, dtype=object)
        )
        out["metabolite_label"] = [
            f"{name} [{raw}]" if name and name != "nan" and name != raw else raw
            for raw, name in zip(met_raw, met_name)
        ]
    if "sensor" in out.columns:
        sensor_raw = out["sensor"].astype(str)
        sensor_name = (
            out["sensor_protein_name"].astype(str)
            if "sensor_protein_name" in out.columns
            else pd.Series([""] * len(out), index=out.index, dtype=object)
        )
        out["sensor_label"] = [
            f"{raw}"
            if not name or name == "nan"
            else f"{raw}"
            for raw, name in zip(sensor_raw, sensor_name)
        ]
    for axis in ("source", "target"):
        token_col = f"{axis}_token"
        label_col = f"{axis}_label"
        if token_col in out.columns:
            tokens = out[token_col].map(_resolve_plot_group)
        elif axis in out.columns:
            tokens = out[axis].map(_resolve_plot_group)
        else:
            continue
        out[token_col] = tokens.astype(str)
        if axis in out.columns:
            out[f"{axis}_id"] = out[axis].astype(str)
        else:
            out[f"{axis}_id"] = out[token_col].astype(str)
        if label_col in out.columns:
            labels = out[label_col].astype(str)
            missing_mask = labels.str.strip().eq("") | labels.str.lower().eq("nan")
            labels = labels.where(~missing_mask, out[token_col].map(lambda x: str(display_map.get(str(x), str(x)))))
            out[label_col] = labels
        else:
            out[label_col] = out[token_col].map(lambda x: str(display_map.get(str(x), str(x))))
    return out


def _normalize_join_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _annotate_mebocost_commu_table(
    df: pd.DataFrame,
    *,
    config_path: Path,
    species: str,
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "HMDB_ID" not in out.columns:
        out["HMDB_ID"] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
    hmdb_missing = out["HMDB_ID"].isna() | (out["HMDB_ID"].astype(str).str.strip() == "") | (out["HMDB_ID"].astype(str).str.lower() == "nan")
    if "metabolite" in out.columns:
        metabolite_as_id = out["metabolite"].astype(str).str.strip()
        hmdb_like = metabolite_as_id.str.match(r"^HMDB\d+$", na=False)
        out.loc[hmdb_missing & hmdb_like, "HMDB_ID"] = metabolite_as_id.loc[hmdb_missing & hmdb_like]
    if "sensor" in out.columns and "sensor_gene" not in out.columns:
        out["sensor_gene"] = out["sensor"].astype(str)
    if "Annotation" in out.columns and "sensor_annotation" not in out.columns:
        out["sensor_annotation"] = out["Annotation"].astype(str)
    hmdb, sensor = _load_mebocost_annotation_tables(str(config_path), str(species))

    if not hmdb.empty:
        hmdb_keep = [
            "HMDB_ID",
            "metabolite",
            "kingdom",
            "super_class",
            "class",
            "sub_class",
            "BioLocation_Summary",
            "Subcellular",
            "Kegg_ID",
            "associated_gene",
        ]
        hmdb_sub = hmdb[[c for c in hmdb_keep if c in hmdb.columns]].copy()
        if "HMDB_ID" in out.columns and "HMDB_ID" in hmdb_sub.columns:
            out = out.merge(hmdb_sub.drop_duplicates(subset=["HMDB_ID"]), on="HMDB_ID", how="left")
        elif "metabolite" in out.columns and "metabolite" in hmdb_sub.columns:
            left = out.assign(_met_key=_normalize_join_text(out["metabolite"]))
            right = hmdb_sub.assign(_met_key=_normalize_join_text(hmdb_sub["metabolite"]))
            out = left.merge(
                right.drop_duplicates(subset=["_met_key"]),
                on="_met_key",
                how="left",
                suffixes=("", "_hmdb"),
            ).drop(columns=["_met_key"])
            if "metabolite_hmdb" in out.columns:
                out = out.drop(columns=["metabolite_hmdb"])
        if "metabolite_x" in out.columns:
            out = out.rename(columns={"metabolite_x": "metabolite"})
        if "metabolite_y" in out.columns:
            out = out.drop(columns=["metabolite_y"])

    if not sensor.empty:
        sensor_keep = [
            "HMDB_ID",
            "standard_metName",
            "Gene_name",
            "Protein_name",
            "Annotation",
            "Evidence",
        ]
        sensor_sub = sensor[[c for c in sensor_keep if c in sensor.columns]].copy()
        sensor_sub = sensor_sub.rename(
            columns={
                "Gene_name": "sensor_gene",
                "Protein_name": "sensor_protein_name",
                "Annotation": "sensor_annotation",
                "Evidence": "sensor_evidence",
                "standard_metName": "sensor_metabolite_name",
            }
        )
        left = out.copy()
        if "HMDB_ID" in left.columns and "HMDB_ID" in sensor_sub.columns:
            left["_hmdb_key"] = left["HMDB_ID"].astype(str)
            sensor_sub["_hmdb_key"] = sensor_sub["HMDB_ID"].astype(str)
        else:
            left["_hmdb_key"] = ""
            sensor_sub["_hmdb_key"] = ""
        left["_sensor_key"] = _normalize_join_text(left["sensor_gene"]) if "sensor_gene" in left.columns else ""
        sensor_sub["_sensor_key"] = _normalize_join_text(sensor_sub["sensor_gene"])
        if "metabolite" in left.columns:
            left["_met_key"] = _normalize_join_text(left["metabolite"])
        else:
            left["_met_key"] = ""
        sensor_sub["_met_key"] = _normalize_join_text(sensor_sub["sensor_metabolite_name"]) if "sensor_metabolite_name" in sensor_sub.columns else ""

        primary = left.merge(
            sensor_sub.drop_duplicates(subset=["_hmdb_key", "_sensor_key"]),
            on=["_hmdb_key", "_sensor_key"],
            how="left",
            suffixes=("", "_sensordb"),
        )
        need_fallback = primary["sensor_annotation"].isna() if "sensor_annotation" in primary.columns else pd.Series(False, index=primary.index)
        if need_fallback.any():
            fallback = left.loc[need_fallback].merge(
                sensor_sub.drop_duplicates(subset=["_met_key", "_sensor_key"]),
                on=["_met_key", "_sensor_key"],
                how="left",
                suffixes=("", "_sensordb"),
            )
            for col in ("sensor_protein_name", "sensor_annotation", "sensor_evidence", "HMDB_ID_sensordb", "sensor_metabolite_name"):
                if col in fallback.columns and col in primary.columns:
                    primary.loc[need_fallback, col] = fallback[col].to_numpy()
        out = primary.drop(columns=[c for c in ("_hmdb_key", "_sensor_key", "_met_key") if c in primary.columns])
        if "sensor_annotation" in out.columns and "Annotation" in out.columns:
            out["sensor_annotation"] = out["sensor_annotation"].where(out["sensor_annotation"].notna(), out["Annotation"])

    return out


def _summarize_mebocost_annotation(
    commu_res: pd.DataFrame,
    annotation_col: str,
    *,
    focus_col: str | None = None,
) -> pd.DataFrame:
    cols = [str(annotation_col)]
    if focus_col:
        cols.insert(0, str(focus_col))
    base_cols = [c for c in cols if c in (commu_res.columns if commu_res is not None else [])]
    if commu_res is None or getattr(commu_res, "empty", True) or len(base_cols) != len(cols):
        out_cols = ([] if not focus_col else [str(focus_col)]) + [str(annotation_col), "n_events", "mean_score"]
        return pd.DataFrame(columns=out_cols)
    work = commu_res.copy()
    work[str(annotation_col)] = work[str(annotation_col)].fillna("Unannotated").astype(str)
    if focus_col:
        work[str(focus_col)] = work[str(focus_col)].astype(str)
    group = work.groupby(cols, observed=False)
    out = group.size().rename("n_events").reset_index()
    if "commu_score" in work.columns:
        mean_score = group["commu_score"].mean().rename("mean_score").reset_index()
        out = out.merge(mean_score, on=cols, how="left")
    else:
        out["mean_score"] = np.nan
    sort_cols = ["n_events", "mean_score"] + cols
    asc = [False, False] + [True] * len(cols)
    out = out.sort_values(sort_cols, ascending=asc, kind="mergesort").reset_index(drop=True)
    return out


def _read_mebocost_candidate_events(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"ccc mebocost paired-rescore: candidate event file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def _read_liana_candidate_events(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"ccc liana paired-rescore: candidate event file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def _split_liana_complex_genes(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "nan":
        return []
    genes = [part.strip() for part in raw.split("_") if part.strip()]
    return sorted(dict.fromkeys(genes))


def _normalize_liana_candidate_events(
    candidate_df: pd.DataFrame,
    *,
    display_map: Mapping[str, str],
    valid_group_tokens: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidate_df is None or getattr(candidate_df, "empty", True):
        raise RuntimeError("ccc liana paired-rescore: candidate event table is empty.")
    required = {"source", "target", "ligand_complex", "receptor_complex"}
    missing = [col for col in required if col not in candidate_df.columns]
    if missing:
        raise RuntimeError(
            "ccc liana paired-rescore: candidate events require columns: "
            + ", ".join(sorted(required))
            + f". Missing: {', '.join(missing)}"
        )
    out = candidate_df.copy()
    valid_tokens = {str(x) for x in (valid_group_tokens or ()) if str(x)}

    def _resolve_event_group(value: object) -> str:
        raw = str(value).strip()
        if not raw:
            raise RuntimeError("ccc liana paired-rescore: empty candidate cluster label.")
        if raw in valid_tokens:
            return raw
        raw_token = plot_utils._extract_cnn_token(raw)
        if raw_token and raw_token in valid_tokens:
            return raw_token
        return _resolve_cluster_request(raw, display_map)

    if "source_token" in out.columns:
        out["source_token"] = out["source_token"].map(_resolve_event_group)
    else:
        out["source_token"] = out["source"].map(_resolve_event_group)
    if "target_token" in out.columns:
        out["target_token"] = out["target_token"].map(_resolve_event_group)
    else:
        out["target_token"] = out["target"].map(_resolve_event_group)
    out["source_label"] = out["source_token"].map(lambda x: str(display_map.get(str(x), str(x))))
    out["target_label"] = out["target_token"].map(lambda x: str(display_map.get(str(x), str(x))))
    if "source" not in out.columns:
        out["source"] = out["source_token"].astype(str)
    if "target" not in out.columns:
        out["target"] = out["target_token"].astype(str)
    if "route_family" not in out.columns:
        out["route_family"] = [
            _liana_route_family(lig, rec)
            for lig, rec in zip(out["ligand_complex"], out["receptor_complex"], strict=False)
        ]
    out["branch_pair"] = out["source_label"].astype(str) + " -> " + out["target_label"].astype(str)
    return out


def _filter_liana_candidate_events(candidate_df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = candidate_df.copy()
    for col_name, cfg_name in (
        ("source_token", "liana_source_filter"),
        ("target_token", "liana_target_filter"),
        ("ligand_complex", "liana_ligand_filter"),
        ("receptor_complex", "liana_receptor_filter"),
        ("route_family", "liana_route_family_filter"),
    ):
        allowed = {str(x) for x in (getattr(cfg, cfg_name, ()) or ()) if str(x)}
        if allowed and col_name in out.columns:
            out = out[out[col_name].astype(str).isin(allowed)].copy()

    sort_col = None
    ascending = False
    for candidate_col in (
        "magnitude_rank",
        "specificity_rank",
        "rank",
        "lr_means",
        "lrscore",
        "score",
        "magnitude",
    ):
        if candidate_col in out.columns:
            sort_col = candidate_col
            ascending = "rank" in candidate_col.lower()
            break
    dedupe_cols = ["source_token", "target_token", "ligand_complex", "receptor_complex"]
    if sort_col is not None:
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=[c for c in dedupe_cols if c in out.columns], keep="first").reset_index(drop=True)
    max_edges = int(getattr(cfg, "liana_max_edges", 200))
    if max_edges > 0:
        out = out.head(max_edges).copy()
    if out.empty:
        raise RuntimeError("ccc liana paired-rescore: no candidate events remain after filtering.")
    return out


def _mean_complex_expression(
    matrix,
    *,
    row_idx: np.ndarray,
    gene_idx: Sequence[int],
    values_logged: bool,
) -> tuple[float, float]:
    if row_idx.size == 0 or len(gene_idx) == 0:
        return float("nan"), float("nan")
    sub = matrix[row_idx][:, list(gene_idx)]
    if sp.issparse(sub):
        detect = np.asarray((sub > 0).mean(axis=0)).ravel().astype(float)
        work = sub.copy().tocsr()
        if not values_logged:
            work.data = np.log1p(work.data)
        means = np.asarray(work.mean(axis=0)).ravel().astype(float)
    else:
        arr = np.asarray(sub, dtype=np.float64)
        detect = np.mean(arr > 0, axis=0, dtype=np.float64)
        if not values_logged:
            arr = np.log1p(arr)
        means = np.mean(arr, axis=0, dtype=np.float64)
    return float(np.mean(means)), float(np.mean(detect))


def _score_liana_paired_edges(
    adata: ad.AnnData,
    *,
    candidate_df: pd.DataFrame,
    groupby: str,
    pairing_key: str,
    condition_cols: Sequence[str],
    dataset_key: Optional[str],
    source_levels: Sequence[str],
    target_levels: Sequence[str],
    layer: Optional[str],
    values_logged: bool,
    min_sender_cells: int,
    min_receiver_cells: int,
) -> pd.DataFrame:
    if pairing_key not in adata.obs:
        raise RuntimeError(f"ccc liana paired-rescore: pairing_key={pairing_key!r} not found in adata.obs.")
    matrix = adata.layers[layer] if layer else adata.X
    gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
    rows: list[dict[str, Any]] = []
    source_level_set = {str(x) for x in source_levels if str(x)}
    target_level_set = {str(x) for x in target_levels if str(x)}

    for sample_id, sample_cells in adata.obs.groupby(str(pairing_key), observed=False).groups.items():
        sample_pos = adata.obs_names.get_indexer(list(sample_cells))
        if sample_pos.size == 0:
            continue
        sample_obs = adata.obs.iloc[sample_pos].copy()
        sample_clusters = sample_obs[str(groupby)].astype(str)
        sample_datasets = sample_obs[str(dataset_key)].astype(str) if dataset_key else None
        sample_counts = sample_clusters.value_counts()
        meta_row: dict[str, Any] = {"sample_id": str(sample_id)}
        for col in condition_cols:
            if col in sample_obs.columns:
                vals = sample_obs[col].astype(str).dropna().unique().tolist()
                meta_row[str(col)] = vals[0] if vals else ""

        for _, event in candidate_df.iterrows():
            row = dict(meta_row)
            row.update(event.to_dict())
            source_token = str(event["source_token"])
            target_token = str(event["target_token"])
            source_mask = sample_clusters.to_numpy() == source_token
            target_mask = sample_clusters.to_numpy() == target_token
            if sample_datasets is not None:
                source_allowed = {str(event.get("source_dataset_level", ""))} if str(event.get("source_dataset_level", "")).strip() else source_level_set
                target_allowed = {str(event.get("target_dataset_level", ""))} if str(event.get("target_dataset_level", "")).strip() else target_level_set
                if source_allowed:
                    source_mask &= sample_datasets.isin(list(source_allowed)).to_numpy()
                if target_allowed:
                    target_mask &= sample_datasets.isin(list(target_allowed)).to_numpy()

            sender_n = int(np.sum(source_mask))
            receiver_n = int(np.sum(target_mask))
            row["sample_id"] = str(sample_id)
            row["sender_n_cells"] = sender_n
            row["receiver_n_cells"] = receiver_n
            row["ligand_expr"] = np.nan
            row["ligand_prop"] = np.nan
            row["receptor_expr"] = np.nan
            row["receptor_prop"] = np.nan
            row["edge_score"] = np.nan
            row["edge_score_log1p"] = np.nan
            row["missing_reason"] = ""
            prior_col = next(
                (col for col in ("magnitude_rank", "specificity_rank", "rank", "lr_means", "lrscore", "score", "magnitude") if col in row),
                None,
            )
            row["candidate_prior_score"] = row.get(prior_col, np.nan) if prior_col else np.nan

            if sender_n < int(min_sender_cells):
                row["missing_reason"] = "too_few_sender_cells"
                rows.append(row)
                continue
            if receiver_n < int(min_receiver_cells):
                row["missing_reason"] = "too_few_receiver_cells"
                rows.append(row)
                continue

            ligand_genes = [gene for gene in _split_liana_complex_genes(event["ligand_complex"]) if gene in gene_to_idx]
            receptor_genes = [gene for gene in _split_liana_complex_genes(event["receptor_complex"]) if gene in gene_to_idx]
            row["n_ligand_genes_total"] = len(_split_liana_complex_genes(event["ligand_complex"]))
            row["n_receptor_genes_total"] = len(_split_liana_complex_genes(event["receptor_complex"]))
            row["n_ligand_genes_detected"] = len(ligand_genes)
            row["n_receptor_genes_detected"] = len(receptor_genes)
            if not ligand_genes:
                row["missing_reason"] = "ligand_genes_absent"
                rows.append(row)
                continue
            if not receptor_genes:
                row["missing_reason"] = "receptor_genes_absent"
                rows.append(row)
                continue

            ligand_expr, ligand_prop = _mean_complex_expression(
                matrix,
                row_idx=sample_pos[source_mask],
                gene_idx=[gene_to_idx[g] for g in ligand_genes],
                values_logged=values_logged,
            )
            receptor_expr, receptor_prop = _mean_complex_expression(
                matrix,
                row_idx=sample_pos[target_mask],
                gene_idx=[gene_to_idx[g] for g in receptor_genes],
                values_logged=values_logged,
            )
            row["ligand_expr"] = ligand_expr
            row["ligand_prop"] = ligand_prop
            row["receptor_expr"] = receptor_expr
            row["receptor_prop"] = receptor_prop
            if not np.isfinite(ligand_expr):
                row["missing_reason"] = "ligand_score_missing"
                rows.append(row)
                continue
            if not np.isfinite(receptor_expr):
                row["missing_reason"] = "receptor_score_missing"
                rows.append(row)
                continue
            edge_score = float(np.sqrt(max(ligand_expr, 0.0) * max(receptor_expr, 0.0)))
            row["edge_score"] = edge_score
            row["edge_score_log1p"] = float(np.log1p(edge_score))
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_liana_paired_routes(event_scores: pd.DataFrame) -> pd.DataFrame:
    if event_scores is None or getattr(event_scores, "empty", True):
        return pd.DataFrame()
    group_cols = ["sample_id"]
    for col in (
        "source_token",
        "target_token",
        "source_label",
        "target_label",
        "branch_pair",
        "route_family",
        "sex",
        "MASLD",
        "timepoint",
    ):
        if col in event_scores.columns:
            group_cols.append(col)
    out = (
        event_scores.groupby(group_cols, observed=False)
        .agg(
            n_edges_scored=("edge_score", lambda s: int(s.notna().sum())),
            n_edges_missing=("edge_score", lambda s: int(s.isna().sum())),
            mean_edge_score=("edge_score", "mean"),
            median_edge_score=("edge_score", "median"),
            sum_edge_score=("edge_score", "sum"),
            mean_ligand_expr=("ligand_expr", "mean"),
            mean_receptor_expr=("receptor_expr", "mean"),
            mean_edge_score_log1p=("edge_score_log1p", "mean"),
        )
        .reset_index()
    )
    return out


def _summarize_paired_missingness(
    scores_df: pd.DataFrame,
    *,
    group_by: Sequence[str],
    score_col: str,
    primary_condition_key: Optional[str],
) -> pd.DataFrame:
    if scores_df is None or getattr(scores_df, "empty", True):
        return pd.DataFrame()
    df = scores_df.copy()
    cols = [str(c) for c in group_by if str(c) in df.columns]
    if primary_condition_key and str(primary_condition_key) in df.columns:
        cols.append(str(primary_condition_key))
    if not cols or score_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "missing_reason" in work.columns:
        work["missing_reason"] = work["missing_reason"].fillna("").astype(str)
    else:
        work["missing_reason"] = ""
    work["_is_scored"] = pd.to_numeric(work[score_col], errors="coerce").notna()
    grouped = work.groupby(cols, observed=False)
    out = grouped.size().rename("n_total").reset_index()
    out = out.merge(grouped["_is_scored"].sum().rename("n_scored").reset_index(), on=cols, how="left")
    out["n_scored"] = pd.to_numeric(out["n_scored"], errors="coerce").fillna(0).astype(int)
    out["n_missing"] = out["n_total"].astype(int) - out["n_scored"]
    for reason in sorted({str(x).strip() for x in work["missing_reason"].unique().tolist() if str(x).strip()}):
        token = re.sub(r"[^A-Za-z0-9]+", "_", reason).strip("_").lower() or "missing"
        col = f"n_{token}"
        tmp = (
            work[work["missing_reason"] == reason]
            .groupby(cols, observed=False)
            .size()
            .rename(col)
            .reset_index()
        )
        out = out.merge(tmp, on=cols, how="left")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    sort_cols = ["n_scored", "n_missing"] + cols
    asc = [False, False] + [True] * len(cols)
    out = out.sort_values(sort_cols, ascending=asc, kind="mergesort").reset_index(drop=True)
    return out


def _summarize_liana_paired_effects(
    scores_df: pd.DataFrame,
    *,
    primary_condition_key: Optional[str],
    condition_cols: Sequence[str],
    compare_levels: Sequence[str],
    value_col: str,
    group_by: Sequence[str],
    median_support_cols: Sequence[str] = (),
    min_scored_donors_per_group: int = 1,
) -> pd.DataFrame:
    if (
        scores_df is None
        or getattr(scores_df, "empty", True)
        or not primary_condition_key
        or primary_condition_key not in scores_df.columns
        or value_col not in scores_df.columns
    ):
        return pd.DataFrame()
    df = scores_df.copy()
    df = df[df[value_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    requested_levels = [str(x) for x in compare_levels if str(x)]
    levels = [str(x) for x in df[primary_condition_key].dropna().astype(str).unique().tolist()]
    if requested_levels:
        levels = [lv for lv in levels if lv in set(requested_levels)]
    if len(levels) < 2:
        return pd.DataFrame()
    pairs = _select_pairs(levels, None)
    context_cols = [str(c) for c in condition_cols if str(c) != str(primary_condition_key) and str(c) in df.columns]
    rows: list[dict[str, Any]] = []
    context_groups = [([], df)] if not context_cols else list(df.groupby(context_cols, observed=False))
    for context_key, context_df in context_groups:
        context_map: dict[str, Any] = {}
        if context_cols:
            if not isinstance(context_key, tuple):
                context_key = (context_key,)
            context_map = {col: val for col, val in zip(context_cols, context_key, strict=False)}
        for pair_a, pair_b in pairs:
            left = context_df[context_df[primary_condition_key].astype(str) == str(pair_a)]
            right = context_df[context_df[primary_condition_key].astype(str) == str(pair_b)]
            merged = pd.concat([left, right], ignore_index=True)
            for keys, sub in merged.groupby(list(group_by), observed=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = {col: val for col, val in zip(group_by, keys, strict=False)}
                row.update(context_map)
                x = sub.loc[sub[primary_condition_key].astype(str) == str(pair_a), value_col].dropna().to_numpy(dtype=float)
                y = sub.loc[sub[primary_condition_key].astype(str) == str(pair_b), value_col].dropna().to_numpy(dtype=float)
                row["contrast"] = f"{pair_a}_vs_{pair_b}"
                row["group_a"] = str(pair_a)
                row["group_b"] = str(pair_b)
                row["n_group_a"] = int(x.size)
                row["n_group_b"] = int(y.size)
                row["mean_group_a"] = float(np.mean(x)) if x.size else np.nan
                row["mean_group_b"] = float(np.mean(y)) if y.size else np.nan
                row["median_group_a"] = float(np.median(x)) if x.size else np.nan
                row["median_group_b"] = float(np.median(y)) if y.size else np.nan
                row["min_scored_donors_per_group"] = int(max(1, min_scored_donors_per_group))
                if x.size < int(max(1, min_scored_donors_per_group)) or y.size < int(max(1, min_scored_donors_per_group)):
                    row["cliffs_delta"] = np.nan
                    row["mannwhitney_pval"] = np.nan
                    row["insufficient_scored_donors"] = True
                    for support_col in median_support_cols:
                        if support_col in sub.columns:
                            row[f"{support_col}_median"] = float(pd.to_numeric(sub[support_col], errors="coerce").median())
                    rows.append(row)
                    continue
                row["cliffs_delta"] = _cliffs_delta(x, y)
                if x.size and y.size:
                    row["mannwhitney_pval"] = float(sstats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
                else:
                    row["mannwhitney_pval"] = np.nan
                row["insufficient_scored_donors"] = False
                for support_col in median_support_cols:
                    if support_col in sub.columns:
                        row[f"{support_col}_median"] = float(pd.to_numeric(sub[support_col], errors="coerce").median())
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["insufficient_scored_donors"].astype(bool) == False].copy()
    if out.empty:
        return out
    pvals = pd.to_numeric(out["mannwhitney_pval"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    fdr = np.full_like(pvals, np.nan, dtype=float)
    if valid.any():
        order = np.argsort(pvals[valid])
        ranked = pvals[valid][order]
        m = float(len(ranked))
        adj = np.minimum.accumulate((ranked * m / (np.arange(len(ranked)) + 1))[::-1])[::-1]
        fdr_vals = np.empty_like(ranked)
        fdr_vals[order] = np.clip(adj, 0.0, 1.0)
        fdr[valid] = fdr_vals
    out["fdr"] = fdr
    return out


def _resolve_condition_filter_context(
    adata: ad.AnnData,
    *,
    condition_spec: Optional[str],
    condition_values: Sequence[str],
) -> tuple[ad.AnnData, Optional[str], list[str], str]:
    if not condition_spec:
        return adata, None, [], "all"
    spec = str(condition_spec).strip()
    values = [str(x).strip() for x in condition_values if str(x).strip()]
    if "@" not in spec:
        if spec not in adata.obs:
            raise RuntimeError(f"ccc mebocost paired-rescore: condition_key={spec!r} not found in adata.obs.")
        if values:
            mask = _normalize_levels(adata.obs[spec]).astype(str).isin(values)
            return adata[mask].copy(), spec, [spec], f"{spec}=" + ",".join(values)
        return adata, spec, [spec], spec
    a_key, b_key = [str(x).strip() for x in spec.split("@", 1)]
    if a_key not in adata.obs or b_key not in adata.obs:
        raise RuntimeError(f"ccc mebocost paired-rescore: condition spec {spec!r} requires both columns in adata.obs.")
    if values:
        mask = _normalize_levels(adata.obs[b_key]).astype(str).isin(values)
        return adata[mask].copy(), a_key, [a_key, b_key], f"{a_key}@{b_key}=" + ",".join(values)
    return adata, a_key, [a_key, b_key], spec


def _normalize_mebocost_candidate_events(
    candidate_df: pd.DataFrame,
    *,
    display_map: Mapping[str, str],
    valid_group_tokens: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidate_df is None or getattr(candidate_df, "empty", True):
        raise RuntimeError("ccc mebocost paired-rescore: candidate event table is empty.")
    out = candidate_df.copy()
    valid_tokens = {str(x) for x in (valid_group_tokens or ()) if str(x)}

    def _resolve_event_group(value: object) -> str:
        raw = str(value).strip()
        if not raw:
            raise RuntimeError("ccc mebocost paired-rescore: empty candidate cluster label.")
        if raw in valid_tokens:
            return raw
        raw_token = plot_utils._extract_cnn_token(raw)
        if raw_token and raw_token in valid_tokens:
            return raw_token
        return _resolve_cluster_request(raw, display_map)

    if "source_token" not in out.columns:
        if "source" not in out.columns:
            raise RuntimeError("ccc mebocost paired-rescore: candidate events require source or source_token.")
        out["source_token"] = out["source"].map(_resolve_event_group)
    else:
        out["source_token"] = out["source_token"].map(_resolve_event_group)
    if "target_token" not in out.columns:
        if "target" not in out.columns:
            raise RuntimeError("ccc mebocost paired-rescore: candidate events require target or target_token.")
        out["target_token"] = out["target"].map(_resolve_event_group)
    else:
        out["target_token"] = out["target_token"].map(_resolve_event_group)
    if "source" not in out.columns:
        out["source"] = out["source_token"].astype(str)
    if "target" not in out.columns:
        out["target"] = out["target_token"].astype(str)
    if "sensor_gene" not in out.columns:
        if "sensor" in out.columns:
            out["sensor_gene"] = out["sensor"].astype(str)
        else:
            raise RuntimeError("ccc mebocost paired-rescore: candidate events require sensor or sensor_gene.")
    if "sensor" not in out.columns:
        out["sensor"] = out["sensor_gene"].astype(str)
    if "HMDB_ID" not in out.columns:
        if "metabolite" in out.columns:
            met = out["metabolite"].astype(str).str.strip()
            out["HMDB_ID"] = met.where(met.str.match(r"^HMDB\d+$", na=False), np.nan)
        else:
            raise RuntimeError("ccc mebocost paired-rescore: candidate events require HMDB_ID or metabolite.")
    if "metabolite" not in out.columns:
        out["metabolite"] = out["HMDB_ID"].astype(str)
    if "Metabolite_Name" not in out.columns:
        out["Metabolite_Name"] = out.get("metabolite", pd.Series([""] * len(out), index=out.index))
    out["source_token"] = out["source_token"].astype(str)
    out["target_token"] = out["target_token"].astype(str)
    out["source_label"] = out["source_token"].map(lambda x: str(display_map.get(str(x), str(x))))
    out["target_label"] = out["target_token"].map(lambda x: str(display_map.get(str(x), str(x))))
    return out


def _filter_mebocost_candidate_events(candidate_df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = candidate_df.copy()
    for col_name, cfg_name in (
        ("source_token", "mebocost_source_filter"),
        ("target_token", "mebocost_target_filter"),
        ("HMDB_ID", "mebocost_metabolite_filter"),
        ("sensor_gene", "mebocost_sensor_filter"),
        ("super_class", "mebocost_superclass_filter"),
        ("class", "mebocost_class_filter"),
        ("sub_class", "mebocost_subclass_filter"),
    ):
        allowed = {str(x) for x in (getattr(cfg, cfg_name, ()) or ()) if str(x)}
        if allowed and col_name in out.columns:
            out = out[out[col_name].astype(str).isin(allowed)].copy()
    sort_col = None
    for candidate_col in ("Norm_Commu_Score", "commu_score", "Commu_Score", "candidate_prior_score"):
        if candidate_col in out.columns:
            sort_col = candidate_col
            break
    dedupe_cols = ["source_token", "target_token", "sensor_gene"]
    if "HMDB_ID" in out.columns:
        dedupe_cols.insert(2, "HMDB_ID")
    else:
        dedupe_cols.insert(2, "metabolite")
    if sort_col is not None:
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, ascending=False, kind="mergesort")
    out = out.drop_duplicates(subset=[c for c in dedupe_cols if c in out.columns], keep="first").reset_index(drop=True)
    max_events = int(getattr(cfg, "mebocost_max_events", 200))
    if max_events > 0:
        out = out.head(max_events).copy()
    if out.empty:
        raise RuntimeError("ccc mebocost paired-rescore: no candidate events remain after filtering.")
    return out


def _make_mebocost_object_for_sample(
    adata_sample: ad.AnnData,
    *,
    groupby: str,
    organism: str,
    config_path: Path,
    mebocost_api,
    layer: Optional[str],
) -> Any:
    adata_mebo = adata_sample.copy()
    if layer:
        if str(layer) not in adata_mebo.layers:
            raise RuntimeError(f"ccc mebocost paired-rescore: layer={layer!r} not found in donor subset.")
        adata_mebo.X = adata_mebo.layers[str(layer)].copy()
    create_obj = getattr(mebocost_api, "create_obj", None)
    if create_obj is None:
        raise RuntimeError("ccc mebocost paired-rescore: imported MEBOCOST API does not expose create_obj().")
    mebo_obj = _call_with_supported_kwargs(
        create_obj,
        adata=adata_mebo,
        group_col=str(groupby),
        species=str(organism).strip().lower(),
        config_path=str(config_path),
    )
    mebo_obj._load_config_()
    mebo_obj._avg_by_group_()
    mebo_obj._get_gene_exp_()
    mebo_obj.estimator()
    mebo_obj._avg_met_group_()
    mebo_obj._check_aboundance_()
    return mebo_obj


def _dense_scalar_from_group_matrix(matrix, row_index: Sequence[Any], col_index: Sequence[Any], row_key: str, col_key: str) -> float:
    row_lookup = {str(x): i for i, x in enumerate(row_index)}
    col_lookup = {str(x): i for i, x in enumerate(col_index)}
    if str(row_key) not in row_lookup or str(col_key) not in col_lookup:
        return float("nan")
    value = matrix[row_lookup[str(row_key)], col_lookup[str(col_key)]]
    if sp.issparse(value):
        value = value.toarray()
    return float(np.asarray(value).squeeze())


def _split_associated_genes(value: object) -> list[str]:
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []
    genes: list[str] = []
    for chunk in re.split(r"[;,]", raw):
        gene = str(chunk).strip()
        if gene:
            genes.append(gene)
    return sorted(dict.fromkeys(genes))


def _proxy_metabolite_score_for_event(
    adata_sample: ad.AnnData,
    *,
    source_token: str,
    source_mask: np.ndarray,
    associated_genes: Sequence[str],
    layer: Optional[str],
) -> tuple[float, float, int, int, str]:
    genes_total = len(associated_genes)
    gene_to_idx = {str(g): i for i, g in enumerate(adata_sample.var_names.astype(str))}
    detected = [g for g in associated_genes if g in gene_to_idx]
    genes_detected = len(detected)
    if genes_detected == 0:
        return float("nan"), float("nan"), genes_total, genes_detected, "no_associated_genes_detected"
    matrix = adata_sample.layers[layer] if layer else adata_sample.X
    sub = matrix[source_mask][:, [gene_to_idx[g] for g in detected]]
    if sp.issparse(sub):
        expr = np.asarray(sub.mean(axis=1)).ravel()
        score = float(np.asarray(sub.mean()).squeeze())
        prop = float(np.mean(np.asarray((sub > 0).sum(axis=1)).ravel() > 0))
    else:
        expr = np.asarray(sub.mean(axis=1)).ravel()
        score = float(np.asarray(sub.mean()).squeeze())
        prop = float(np.mean(np.any(sub > 0, axis=1)))
    return score, prop, genes_total, genes_detected, ""


def _score_mebocost_paired_events(
    adata: ad.AnnData,
    *,
    candidate_df: pd.DataFrame,
    groupby: str,
    pairing_key: str,
    condition_cols: Sequence[str],
    dataset_key: Optional[str],
    source_levels: Sequence[str],
    target_levels: Sequence[str],
    organism: str,
    config_path: Path,
    mebocost_api,
    layer: Optional[str],
    score_method: str,
    min_sender_cells: int,
    min_receiver_cells: int,
) -> pd.DataFrame:
    if pairing_key not in adata.obs:
        raise RuntimeError(f"ccc mebocost paired-rescore: pairing_key={pairing_key!r} not found in adata.obs.")
    source_allowed = {str(x) for x in source_levels if str(x)}
    target_allowed = {str(x) for x in target_levels if str(x)}
    rows: list[dict[str, Any]] = []

    for sample_id, sample_cells in adata.obs.groupby(str(pairing_key), observed=False).groups.items():
        idx = np.asarray(list(sample_cells))
        adata_sample = adata[idx].copy()
        meta_row: dict[str, Any] = {str(pairing_key): str(sample_id)}
        for col in condition_cols:
            if col in adata_sample.obs:
                vals = adata_sample.obs[col].astype(str).dropna().unique().tolist()
                meta_row[str(col)] = vals[0] if vals else ""
        sample_counts = adata_sample.obs[str(groupby)].astype(str).value_counts()

        mebo_obj = None
        if score_method == "mebocost-metabolite-sensor":
            mebo_obj = _make_mebocost_object_for_sample(
                adata_sample,
                groupby=str(groupby),
                organism=str(organism),
                config_path=config_path,
                mebocost_api=mebocost_api,
                layer=layer,
            )

        for _, event in candidate_df.iterrows():
            row = dict(meta_row)
            row.update(event.to_dict())
            source_token = str(event["source_token"])
            target_token = str(event["target_token"])
            hmdb_id = str(event.get("HMDB_ID", ""))
            sensor_gene = str(event.get("sensor_gene", event.get("sensor", "")))
            row["sample_id"] = str(sample_id)
            row["source"] = source_token
            row["target"] = target_token
            row["source_token"] = source_token
            row["target_token"] = target_token
            row["sensor_gene"] = sensor_gene
            row["source_dataset_level"] = row.get("source_dataset_level", source_allowed and next(iter(source_allowed)) or "")
            row["target_dataset_level"] = row.get("target_dataset_level", target_allowed and next(iter(target_allowed)) or "")
            row["sender_n_cells"] = int(sample_counts.get(source_token, 0))
            row["receiver_n_cells"] = int(sample_counts.get(target_token, 0))
            row["score_method"] = score_method
            row["candidate_prior_score"] = row.get("Norm_Commu_Score", row.get("commu_score", row.get("Commu_Score", np.nan)))
            row["candidate_prior_pval"] = row.get("pval", row.get("permutation_test_pval", row.get("permutation_test_fdr", np.nan)))
            row["sender_metabolite_score"] = np.nan
            row["sender_metabolite_prop"] = np.nan
            row["receiver_sensor_score"] = np.nan
            row["receiver_sensor_prop"] = np.nan
            row["paired_commu_score"] = np.nan
            row["paired_commu_score_log1p"] = np.nan
            row["missing_reason"] = ""
            if row["sender_n_cells"] < int(min_sender_cells):
                row["missing_reason"] = "too_few_sender_cells"
                rows.append(row)
                continue
            if row["receiver_n_cells"] < int(min_receiver_cells):
                row["missing_reason"] = "too_few_receiver_cells"
                rows.append(row)
                continue

            if score_method == "mebocost-metabolite-sensor":
                row["sender_metabolite_score"] = _dense_scalar_from_group_matrix(
                    mebo_obj.avg_met,
                    mebo_obj.avg_met_indexer,
                    mebo_obj.avg_met_columns,
                    hmdb_id,
                    source_token,
                )
                row["receiver_sensor_score"] = _dense_scalar_from_group_matrix(
                    mebo_obj.avg_exp,
                    mebo_obj.avg_exp_indexer,
                    mebo_obj.avg_exp_columns,
                    sensor_gene,
                    target_token,
                )
                if source_token in mebo_obj.met_prop.index and hmdb_id in mebo_obj.met_prop.columns:
                    row["sender_metabolite_prop"] = float(mebo_obj.met_prop.loc[source_token, hmdb_id])
                if target_token in mebo_obj.exp_prop.index and sensor_gene in mebo_obj.exp_prop.columns:
                    row["receiver_sensor_prop"] = float(mebo_obj.exp_prop.loc[target_token, sensor_gene])
                if pd.isna(row["sender_metabolite_score"]):
                    row["missing_reason"] = "metabolite_not_estimated"
                elif pd.isna(row["receiver_sensor_score"]):
                    row["missing_reason"] = "sensor_gene_absent"
            else:
                source_mask = adata_sample.obs[str(groupby)].astype(str).to_numpy() == str(source_token)
                genes = _split_associated_genes(row.get("associated_gene", ""))
                score, prop, n_total, n_detected, reason = _proxy_metabolite_score_for_event(
                    adata_sample,
                    source_token=source_token,
                    source_mask=source_mask,
                    associated_genes=genes,
                    layer=layer,
                )
                row["n_associated_genes_total"] = int(n_total)
                row["n_associated_genes_detected"] = int(n_detected)
                row["sender_metabolite_score"] = score
                row["sender_metabolite_prop"] = prop
                receiver_mask = adata_sample.obs[str(groupby)].astype(str).to_numpy() == str(target_token)
                receiver_genes = {str(g): i for i, g in enumerate(adata_sample.var_names.astype(str))}
                if sensor_gene in receiver_genes:
                    mat = adata_sample.layers[layer] if layer else adata_sample.X
                    sub = mat[receiver_mask][:, receiver_genes[sensor_gene]]
                    if sp.issparse(sub):
                        arr = np.asarray(sub).ravel()
                    else:
                        arr = np.asarray(sub).ravel()
                    row["receiver_sensor_score"] = float(np.mean(arr))
                    row["receiver_sensor_prop"] = float(np.mean(arr > 0))
                else:
                    reason = reason or "sensor_gene_absent"
                row["missing_reason"] = reason

            if not row["missing_reason"]:
                sender_score = max(float(row["sender_metabolite_score"]), 0.0)
                receiver_score = max(float(row["receiver_sensor_score"]), 0.0)
                paired = float(np.sqrt(sender_score * receiver_score))
                row["paired_commu_score"] = paired
                row["paired_commu_score_log1p"] = float(np.log1p(paired))
            rows.append(row)
    return pd.DataFrame(rows)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.mean(np.sign(x[:, None] - y[None, :])))


def _summarize_mebocost_paired_routes(event_scores: pd.DataFrame) -> pd.DataFrame:
    if event_scores is None or getattr(event_scores, "empty", True):
        return pd.DataFrame()
    df = event_scores.copy()
    if "super_class" in df.columns:
        df["route"] = df["super_class"].fillna("Unannotated").astype(str)
    else:
        df["route"] = "all_events"
    group_cols = ["sample_id"]
    for col in ("source_token", "target_token", "source_label", "target_label", "route", "sex", "MASLD"):
        if col in df.columns:
            group_cols.append(col)
    score_ok = df["paired_commu_score"].notna()
    out = (
        df.groupby(group_cols, observed=False)
        .agg(
            n_events_scored=("paired_commu_score", lambda s: int(s.notna().sum())),
            n_events_missing=("paired_commu_score", lambda s: int(s.isna().sum())),
            mean_paired_commu_score=("paired_commu_score", "mean"),
            median_paired_commu_score=("paired_commu_score", "median"),
            sum_paired_commu_score=("paired_commu_score", "sum"),
            mean_paired_commu_score_log1p=("paired_commu_score_log1p", "mean"),
        )
        .reset_index()
    )
    return out


def _summarize_mebocost_paired_group_effects(
    scores_df: pd.DataFrame,
    *,
    primary_condition_key: Optional[str],
    min_scored_donors_per_group: int = 1,
) -> pd.DataFrame:
    if scores_df is None or getattr(scores_df, "empty", True) or not primary_condition_key or primary_condition_key not in scores_df.columns:
        return pd.DataFrame()
    df = scores_df.copy()
    df = df[df["paired_commu_score"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    levels = [str(x) for x in df[primary_condition_key].dropna().astype(str).unique().tolist()]
    if len(levels) < 2:
        return pd.DataFrame()
    pairs = _select_pairs(levels, None)
    context_cols = [c for c in ("MASLD", "timepoint") if c in df.columns and c != primary_condition_key]
    group_by = ["source_token", "target_token", "HMDB_ID", "sensor_gene"]
    for col in ("Metabolite_Name", "super_class", "sensor_annotation"):
        if col in df.columns:
            group_by.append(col)
    rows: list[dict[str, Any]] = []
    context_groups = [([], df)] if not context_cols else list(df.groupby(context_cols, observed=False))
    for context_key, context_df in context_groups:
        context_map = {}
        if context_cols:
            if not isinstance(context_key, tuple):
                context_key = (context_key,)
            context_map = {col: val for col, val in zip(context_cols, context_key)}
        for pair_a, pair_b in pairs:
            left = context_df[context_df[primary_condition_key].astype(str) == str(pair_a)]
            right = context_df[context_df[primary_condition_key].astype(str) == str(pair_b)]
            merged = pd.concat([left, right], ignore_index=True)
            for keys, sub in merged.groupby(group_by, observed=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = {col: val for col, val in zip(group_by, keys)}
                row.update(context_map)
                x = sub.loc[sub[primary_condition_key].astype(str) == str(pair_a), "paired_commu_score"].dropna().to_numpy(dtype=float)
                y = sub.loc[sub[primary_condition_key].astype(str) == str(pair_b), "paired_commu_score"].dropna().to_numpy(dtype=float)
                row["contrast"] = f"{pair_a}_vs_{pair_b}"
                row["group_a"] = str(pair_a)
                row["group_b"] = str(pair_b)
                row["n_group_a"] = int(x.size)
                row["n_group_b"] = int(y.size)
                row["mean_group_a"] = float(np.mean(x)) if x.size else np.nan
                row["mean_group_b"] = float(np.mean(y)) if y.size else np.nan
                row["median_group_a"] = float(np.median(x)) if x.size else np.nan
                row["median_group_b"] = float(np.median(y)) if y.size else np.nan
                row["min_scored_donors_per_group"] = int(max(1, min_scored_donors_per_group))
                if x.size < int(max(1, min_scored_donors_per_group)) or y.size < int(max(1, min_scored_donors_per_group)):
                    row["cliffs_delta"] = np.nan
                    row["mannwhitney_pval"] = np.nan
                    row["insufficient_scored_donors"] = True
                    rows.append(row)
                    continue
                row["cliffs_delta"] = _cliffs_delta(x, y)
                if x.size and y.size:
                    row["mannwhitney_pval"] = float(sstats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
                else:
                    row["mannwhitney_pval"] = np.nan
                row["insufficient_scored_donors"] = False
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["insufficient_scored_donors"].astype(bool) == False].copy()
    if out.empty:
        return out
    pvals = pd.to_numeric(out["mannwhitney_pval"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    fdr = np.full_like(pvals, np.nan, dtype=float)
    if valid.any():
        order = np.argsort(pvals[valid])
        ranked = pvals[valid][order]
        m = float(len(ranked))
        adj = np.minimum.accumulate((ranked * m / (np.arange(len(ranked)) + 1))[::-1])[::-1]
        fdr_vals = np.empty_like(ranked)
        fdr_vals[order] = np.clip(adj, 0.0, 1.0)
        fdr[valid] = fdr_vals
    out["fdr"] = fdr
    return out


def _nichenet_r_env(r_lib_dir: Path, *, install_only: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    r_lib = str(r_lib_dir)
    env["R_LIBS_USER"] = r_lib
    if install_only:
        env["R_LIBS"] = r_lib
        env.pop("R_LIBS_SITE", None)
    else:
        env.pop("R_LIBS", None)
    return env


def _resolve_rscript() -> Optional[str]:
    env_rscript = Path(sys.prefix) / "bin" / "Rscript"
    if env_rscript.exists():
        return str(env_rscript)
    return shutil.which("Rscript")


def _resolve_r_binary() -> Optional[str]:
    env_r = Path(sys.prefix) / "bin" / "R"
    if env_r.exists():
        return str(env_r)
    return shutil.which("R")


def _nichenet_install_command_hint(r_lib_dir: Path) -> str:
    r_lib = str(r_lib_dir)
    r_bin = _resolve_r_binary() or "R"
    cran_pkgs_expr = ", ".join(f'"{pkg}"' for pkg in _NICHENET_R_BOOTSTRAP_CRAN_PACKAGES)
    return (
        "tmpdir=$(mktemp -d) && "
        + "R_LIBS_USER="
        + r_lib
        + " R_LIBS="
        + r_lib
        + " "
        + (_resolve_rscript() or "Rscript")
        + " -e 'dir.create(Sys.getenv(\"R_LIBS_USER\"), recursive=TRUE, showWarnings=FALSE); .libPaths(c(Sys.getenv(\"R_LIBS_USER\"), .Library, .Library.site)); pkgs <- c("
        + cran_pkgs_expr
        + "); missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing_pkgs)) install.packages(missing_pkgs, repos=\"https://cloud.r-project.org\", lib=Sys.getenv(\"R_LIBS_USER\"))' && "
        + "git clone --depth 1 https://github.com/saeyslab/nichenetr.git \"$tmpdir/nichenetr\" && "
        + "mkdir -p "
        + r_lib
        + " && "
        + "R_LIBS_USER="
        + r_lib
        + " R_LIBS="
        + r_lib
        + " "
        + r_bin
        + " CMD INSTALL --no-test-load -l "
        + r_lib
        + " \"$tmpdir/nichenetr\""
    )


def _ensure_nichenet_r_runtime(*, install_missing: bool) -> Path:
    rscript = _resolve_rscript()
    r_bin = _resolve_r_binary()
    if not rscript:
        raise RuntimeError("NicheNet requires `Rscript` on PATH.")
    if not r_bin:
        raise RuntimeError("NicheNet requires `R` on PATH.")

    r_lib_dir = _NICHENET_R_LIB_DIR
    r_lib_dir.mkdir(parents=True, exist_ok=True)
    runtime_env = _nichenet_r_env(r_lib_dir, install_only=False)
    check_expr = (
        'dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE, showWarnings=FALSE); '
        '.libPaths(c(Sys.getenv("R_LIBS_USER"), .Library, .Library.site)); '
        'quit(save="no", status=if (requireNamespace("nichenetr", quietly=TRUE)) 0 else 1)'
    )
    check_res = subprocess.run(
        [str(rscript), "-e", check_expr],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env,
    )
    if check_res.returncode == 0:
        return r_lib_dir

    manual_hint = _nichenet_install_command_hint(r_lib_dir)
    if not install_missing:
        raise RuntimeError(
            "NicheNet requires the R package `nichenetr` in scOmnom's project-local R library.\n"
            f"Expected library: {r_lib_dir}\n"
            "Re-run with `--install-missing-r-deps` to bootstrap it automatically, or install it manually with:\n"
            f"{manual_hint}"
        )

    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError(
            "Automatic NicheNet dependency installation requires `git` on PATH.\n"
            f"Project-local R library: {r_lib_dir}\n"
            "You can retry manually with:\n"
            f"{manual_hint}"
        )
    cran_pkgs_expr = ", ".join(f'"{pkg}"' for pkg in _NICHENET_R_BOOTSTRAP_CRAN_PACKAGES)
    cran_bootstrap_expr = (
        'dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE, showWarnings=FALSE); '
        '.libPaths(c(Sys.getenv("R_LIBS_USER"), .Library, .Library.site)); '
        f'pkgs <- c({cran_pkgs_expr}); '
        'missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; '
        'if (length(missing_pkgs)) install.packages(missing_pkgs, repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER")); '
        'still_missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; '
        'if (length(still_missing)) { write(paste(still_missing, collapse=", "), stderr()); quit(save="no", status=1) }'
    )
    cran_bootstrap_res = subprocess.run(
        [str(rscript), "-e", cran_bootstrap_expr],
        check=False,
        capture_output=True,
        text=True,
        env=install_env,
    )
    if cran_bootstrap_res.returncode != 0:
        stderr = str(cran_bootstrap_res.stderr or "").strip()
        stdout = str(cran_bootstrap_res.stdout or "").strip()
        raise RuntimeError(
            "Automatic NicheNet dependency installation failed while bootstrapping required CRAN packages.\n"
            f"Project-local R library: {r_lib_dir}\n"
            "You can retry manually with:\n"
            f"{manual_hint}"
            + (f"\nSTDERR:\n{stderr}" if stderr else "")
            + (f"\nSTDOUT:\n{stdout}" if stdout else "")
        )
    with tempfile.TemporaryDirectory(prefix="scomnom_nichenetr_src_") as tmpdir:
        src_dir = Path(tmpdir) / "nichenetr"
        clone_res = subprocess.run(
            [str(git_bin), "clone", "--depth", "1", "https://github.com/saeyslab/nichenetr.git", str(src_dir)],
            check=False,
            capture_output=True,
            text=True,
            env=install_env,
        )
        if clone_res.returncode != 0:
            stderr = str(clone_res.stderr or "").strip()
            stdout = str(clone_res.stdout or "").strip()
            raise RuntimeError(
                "Automatic NicheNet source download failed.\n"
                f"Project-local R library: {r_lib_dir}\n"
                "You can retry manually with:\n"
                f"{manual_hint}"
                + (f"\nSTDERR:\n{stderr}" if stderr else "")
                + (f"\nSTDOUT:\n{stdout}" if stdout else "")
            )
        install_res = subprocess.run(
            [str(r_bin), "CMD", "INSTALL", "--no-test-load", "-l", str(r_lib_dir), str(src_dir)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    if install_res.returncode != 0:
        stderr = str(install_res.stderr or "").strip()
        stdout = str(install_res.stdout or "").strip()
        raise RuntimeError(
            "Automatic NicheNet dependency installation failed.\n"
            f"Project-local R library: {r_lib_dir}\n"
            "You can retry manually with:\n"
            f"{manual_hint}"
            + (f"\nSTDERR:\n{stderr}" if stderr else "")
            + (f"\nSTDOUT:\n{stdout}" if stdout else "")
        )

    verify_res = subprocess.run(
        [str(rscript), "-e", check_expr],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env,
    )
    if verify_res.returncode != 0:
        install_stderr = str(install_res.stderr or "").strip()
        install_stdout = str(install_res.stdout or "").strip()
        verify_stderr = str(verify_res.stderr or "").strip()
        verify_stdout = str(verify_res.stdout or "").strip()
        raise RuntimeError(
            "Automatic NicheNet dependency installation completed without a detectable `nichenetr` import.\n"
            f"Project-local R library: {r_lib_dir}\n"
            "Please inspect the installation manually and retry."
            + (f"\nINSTALL STDERR:\n{install_stderr}" if install_stderr else "")
            + (f"\nINSTALL STDOUT:\n{install_stdout}" if install_stdout else "")
            + (f"\nVERIFY STDERR:\n{verify_stderr}" if verify_stderr else "")
            + (f"\nVERIFY STDOUT:\n{verify_stdout}" if verify_stdout else "")
        )
    LOGGER.warning("Installed `nichenetr` into project-local R library: %s", r_lib_dir)
    return r_lib_dir


def _liana_plot_color_map(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    display_map: Mapping[str, str],
    round_id: Optional[str],
    raw_labels: Sequence[str],
) -> dict[str, Any]:
    if not raw_labels:
        return {}
    colors = _resolve_cluster_colors(
        adata,
        cluster_key=str(cluster_key),
        labels=[str(x) for x in raw_labels],
        round_id=round_id,
    )
    out: dict[str, Any] = {}
    for raw, color in zip(raw_labels, colors):
        out[_liana_plot_label(raw, display_map)] = color
    return out


def _write_liana_settings(
    out_dir: Path,
    *,
    condition_label: str,
    condition_key: Optional[str],
    condition_value: Optional[str],
    methods: Sequence[str],
    aggregated_methods: Sequence[str],
    resource: str,
    groupby: str,
    use_raw: bool,
    layer: Optional[str],
    input_mode: str,
    lognorm_target_sum: Optional[float],
    expr_prop: float,
    n_perms: Optional[int],
    cross_tissue_mode: bool = False,
    dataset_key: Optional[str] = None,
    source_levels: Sequence[str] = (),
    target_levels: Sequence[str] = (),
    signal_scope: str = "all",
) -> None:
    _write_settings(
        out_dir,
        "__settings.txt",
        [
            f"condition_label={condition_label}",
            f"condition_key={condition_key}",
            f"condition_value={condition_value}",
            f"methods={list(methods)}",
            f"aggregated_methods={list(aggregated_methods)}",
            f"resource={resource}",
            f"groupby={groupby}",
            f"use_raw={use_raw}",
            f"layer={layer}",
            f"input_mode={input_mode}",
            f"lognorm_target_sum={lognorm_target_sum}",
            f"expr_prop={expr_prop}",
            f"n_perms={n_perms}",
            f"cross_tissue_mode={cross_tissue_mode}",
            f"dataset_key={dataset_key}",
            f"source_levels={list(source_levels)}",
            f"target_levels={list(target_levels)}",
            f"signal_scope={signal_scope}",
        ],
    )


def _call_liana_safely(method_callable: Any, adata_in: ad.AnnData, **kwargs) -> pd.DataFrame:
    implicit_mod_warn = getattr(ad, "ImplicitModificationWarning", UserWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The dtype argument is deprecated and will be removed.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Use uns .* AnnData\.uns_keys is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The default of observed=False is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Trying to modify attribute `?\.obs`? of view, initializing view as actual\.",
            category=implicit_mod_warn,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Setting element `?\.layers\['scaled'\]`? of view, initializing view as actual\.",
            category=implicit_mod_warn,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Setting element `?\.layers\['normcounts'\]`? of view, initializing view as actual\.",
            category=implicit_mod_warn,
        )
        return method_callable(adata_in, **kwargs)


def _effective_liana_use_raw(adata: ad.AnnData, *, requested_use_raw: bool, layer: Optional[str]) -> bool:
    if layer is not None:
        return False
    if not requested_use_raw:
        return False
    if getattr(adata, "raw", None) is None:
        LOGGER.warning(
            "ccc liana: requested use_raw=True but adata.raw is not initialized; falling back to adata.X."
        )
        return False
    return True


def _build_liana_lognorm_layer(
    adata: ad.AnnData,
    *,
    target_sum: float,
) -> str:
    source_layer = None
    for candidate in ("counts_cb", "counts_raw"):
        if candidate in adata.layers:
            source_layer = str(candidate)
            break
    if source_layer is None:
        raise RuntimeError(
            "ccc liana: input_mode=lognorm requires adata.layers['counts_cb'] or adata.layers['counts_raw']."
        )
    target_layer = f"lognorm_{source_layer}"
    if target_layer in adata.layers:
        LOGGER.info("ccc liana: reusing adata.layers[%r] as LIANA input.", target_layer)
        return target_layer

    if target_sum <= 0:
        raise RuntimeError("ccc liana: liana_lognorm_target_sum must be > 0.")

    matrix = adata.layers[source_layer]
    LOGGER.info(
        "ccc liana: building adata.layers[%r] from adata.layers[%r] with normalize_total(target_sum=%s)+log1p.",
        target_layer,
        source_layer,
        float(target_sum),
    )
    if sp.issparse(matrix):
        work = matrix.copy().tocsr()
        if work.dtype != np.float32:
            work = work.astype(np.float32)
        totals = np.asarray(work.sum(axis=1)).ravel().astype(np.float64, copy=False)
        scale = np.ones_like(totals, dtype=np.float32)
        mask = totals > 0
        scale[mask] = (float(target_sum) / totals[mask]).astype(np.float32, copy=False)
        for row_idx in np.flatnonzero(mask):
            start = int(work.indptr[row_idx])
            end = int(work.indptr[row_idx + 1])
            if end > start:
                work.data[start:end] *= scale[row_idx]
        np.log1p(work.data, out=work.data)
        adata.layers[target_layer] = work
    else:
        LOGGER.warning(
            "ccc liana: source layer %r is dense; building %r will allocate a dense float32 copy.",
            source_layer,
            target_layer,
        )
        work = np.asarray(matrix, dtype=np.float32).copy()
        totals = work.sum(axis=1, dtype=np.float64)
        scale = np.ones(work.shape[0], dtype=np.float32)
        mask = totals > 0
        scale[mask] = (float(target_sum) / totals[mask]).astype(np.float32, copy=False)
        work *= scale[:, None]
        np.log1p(work, out=work)
        adata.layers[target_layer] = work
    return target_layer


def _effective_mebocost_layer(
    adata: ad.AnnData,
    *,
    input_mode: str,
    lognorm_target_sum: float,
) -> Optional[str]:
    mode = str(input_mode).strip().lower()
    if mode == "lognorm":
        return _build_liana_lognorm_layer(adata, target_sum=float(lognorm_target_sum))
    for preferred in ("counts_cb", "counts_raw"):
        if preferred in adata.layers:
            LOGGER.info("ccc mebocost: using adata.layers[%r] as MEBOCOST input.", preferred)
            return str(preferred)
    LOGGER.info("ccc mebocost: no counts_cb/counts_raw layer found; using adata.X as MEBOCOST input.")
    return None


def _effective_liana_layer(
    adata: ad.AnnData,
    *,
    requested_use_raw: bool,
    layer: Optional[str],
    input_mode: str,
    lognorm_target_sum: float,
) -> Optional[str]:
    if layer is not None:
        return str(layer)
    if requested_use_raw:
        return None
    if str(input_mode).strip().lower() == "lognorm":
        return _build_liana_lognorm_layer(adata, target_sum=float(lognorm_target_sum))
    for preferred in ("counts_cb", "counts_raw"):
        if preferred in adata.layers:
            LOGGER.info("ccc liana: using adata.layers[%r] as LIANA input.", preferred)
            return str(preferred)
    LOGGER.info("ccc liana: no counts_cb/counts_raw layer found; using adata.X as LIANA input.")
    return None


def _liana_condition_spec_token(raw_spec: str) -> str:
    token = str(raw_spec).strip()
    token = token.replace("@", "_at_").replace(":", "_and_").replace("^", "_x_")
    return _safe_combo_token(token)


def _build_liana_condition_run_specs(
    adata: ad.AnnData,
    cfg,
    *,
    log_prefix: str = "ccc liana",
) -> list[dict[str, Any]]:
    cond_keys = [str(x).strip() for x in (getattr(cfg, "ccc_condition_keys", ()) or ()) if str(x).strip()]
    if not cond_keys:
        ck = getattr(cfg, "ccc_condition_key", None)
        if ck:
            cond_keys = [str(ck).strip()]
    condition_values = tuple(str(x) for x in (getattr(cfg, "ccc_condition_values", ()) or ()) if str(x))
    compare_levels = tuple(str(x) for x in (getattr(cfg, "ccc_compare_levels", ()) or ()) if str(x))

    if not cond_keys:
        return [
            {
                "run_id": "global",
                "run_label": "global",
                "adata": adata,
                "condition_key": None,
                "condition_value": None,
                "condition_spec": None,
                "condition_spec_token": "global",
                "context_key": None,
                "context_value": None,
                "context_label": None,
                "tables_rel": Path("."),
                "figs_rel": Path("."),
                "compare_rel": None,
                "compare_bucket": None,
                "compare_title": None,
            }
        ]

    run_specs: list[dict[str, Any]] = []
    for raw_spec in cond_keys:
        spec_token = _liana_condition_spec_token(raw_spec)
        spec_root = Path(f"condition__{spec_token}")
        if "@" in raw_spec:
            parts = [p.strip() for p in raw_spec.split("@") if p.strip()]
            if len(parts) != 2:
                raise RuntimeError(f"{log_prefix}: invalid A@B condition specification {raw_spec!r}.")
            a_key = _resolve_condition_key(adata, parts[0])
            b_key = _resolve_condition_key(adata, parts[1])
            if a_key not in adata.obs or b_key not in adata.obs:
                raise RuntimeError(f"{log_prefix}: condition keys not found in adata.obs: {[a_key, b_key]}")
            b_series = _normalize_levels(adata.obs[b_key])
            wanted_b = sorted(set(condition_values) if condition_values else set(b_series.unique().tolist()))
            for b_level in wanted_b:
                mask_b = b_series.astype(str).to_numpy() == str(b_level)
                if not np.any(mask_b):
                    LOGGER.warning("%s: skipping empty within-level %r for %s", log_prefix, str(b_level), str(b_key))
                    continue
                adata_b = adata[mask_b].copy()
                a_series = _normalize_levels(adata_b.obs[a_key])
                wanted_a = sorted(set(compare_levels) if compare_levels else set(a_series.unique().tolist()))
                context_label = f"{b_key}={b_level}"
                context_token = f"{_safe_combo_token(b_key)}={_safe_combo_token(b_level)}"
                compare_rel = spec_root / context_token / "compare"
                compare_title = f"{a_key} within {b_key}={b_level}"
                for a_level in wanted_a:
                    mask_a = a_series.astype(str).to_numpy() == str(a_level)
                    if not np.any(mask_a):
                        LOGGER.warning(
                            "%s: skipping empty compare level %r for %s within %s=%r",
                            log_prefix,
                            str(a_level),
                            str(a_key),
                            str(b_key),
                            str(b_level),
                        )
                        continue
                    LOGGER.info(
                        "%s: expanded %r -> condition_key=%r within %s=%r at level=%r (n_cells=%d).",
                        log_prefix,
                        raw_spec,
                        str(a_key),
                        str(b_key),
                        str(b_level),
                        str(a_level),
                        int(mask_a.sum()),
                    )
                    run_specs.append(
                        {
                            "run_id": f"{raw_spec}::{context_label}::{a_level}",
                            "run_label": str(a_level),
                            "adata": adata_b[mask_a].copy(),
                            "condition_key": str(a_key),
                            "condition_value": str(a_level),
                            "condition_spec": raw_spec,
                            "condition_spec_token": spec_token,
                            "context_key": str(b_key),
                            "context_value": str(b_level),
                            "context_label": context_label,
                            "tables_rel": spec_root / context_token / _safe_combo_token(a_level),
                            "figs_rel": spec_root / context_token / _safe_combo_token(a_level),
                            "compare_rel": compare_rel,
                            "compare_bucket": f"{spec_token}::{context_token}",
                            "compare_title": compare_title,
                        }
                    )
        else:
            condition_key = _resolve_condition_key(adata, raw_spec)
            if condition_key not in adata.obs:
                raise RuntimeError(f"{log_prefix}: condition_key={condition_key!r} not found in adata.obs")
            cond_series = _normalize_levels(adata.obs[condition_key])
            wanted = sorted(set(compare_levels) if compare_levels else set(condition_values) if condition_values else set(cond_series.unique().tolist()))
            compare_rel = spec_root / "compare"
            for condition_value in wanted:
                mask = cond_series.astype(str).to_numpy() == str(condition_value)
                if not np.any(mask):
                    LOGGER.warning("%s: skipping empty condition_value=%r for spec=%r", log_prefix, str(condition_value), raw_spec)
                    continue
                run_specs.append(
                    {
                        "run_id": f"{raw_spec}::{condition_value}",
                        "run_label": str(condition_value),
                        "adata": adata[mask].copy(),
                        "condition_key": str(condition_key),
                        "condition_value": str(condition_value),
                        "condition_spec": raw_spec,
                        "condition_spec_token": spec_token,
                        "context_key": None,
                        "context_value": None,
                        "context_label": None,
                        "tables_rel": spec_root / _safe_combo_token(condition_value),
                        "figs_rel": spec_root / _safe_combo_token(condition_value),
                        "compare_rel": compare_rel,
                        "compare_bucket": str(spec_token),
                        "compare_title": str(condition_key),
                    }
                )
    return run_specs


def run_liana_ccc(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (ccc liana)...")

    li = _import_liana_module()
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )
    if str(groupby) not in adata.obs:
        raise RuntimeError(f"ccc liana: groupby={groupby!r} not found in adata.obs")
    dataset_key = getattr(cfg, "ccc_dataset_key", None)
    source_levels = tuple(str(x) for x in (getattr(cfg, "ccc_source_levels", ()) or ()) if str(x))
    target_levels = tuple(str(x) for x in (getattr(cfg, "ccc_target_levels", ()) or ()) if str(x))
    signal_scope = str(getattr(cfg, "ccc_signal_scope", "all") or "all").strip().lower()
    cross_tissue_mode = bool(dataset_key)
    cluster_dataset_map: dict[str, str] = {}
    if cross_tissue_mode:
        if not source_levels or not target_levels:
            raise RuntimeError("ccc liana: cross-tissue mode requires at least one source level and one target level.")
        cluster_dataset_map = _resolve_liana_cluster_dataset_levels(
            adata,
            cluster_key=str(groupby),
            dataset_key=str(dataset_key),
        )

    methods = tuple(str(x).strip().lower() for x in (getattr(cfg, "liana_methods", ()) or ()) if str(x).strip())
    if not methods:
        methods = ("rank_aggregate",)
    non_aggregate_methods = tuple(m for m in methods if m != "rank_aggregate")
    method_specs = _liana_method_specs(li)
    bad_methods = [m for m in non_aggregate_methods if m not in method_specs]
    if bad_methods:
        raise RuntimeError(f"ccc liana: unsupported methods requested: {bad_methods}")
    aggregate_methods = non_aggregate_methods or _LIANA_DEFAULT_AGGREGATE_METHODS
    if "connectome" in aggregate_methods:
        LOGGER.warning(
            "ccc liana: aggregate methods include 'connectome', which triggers scaling and may densify sparse input. "
            "This can be memory-intensive on large datasets."
        )

    if bool(getattr(cfg, "liana_use_raw", False)) and getattr(cfg, "liana_layer", None):
        raise RuntimeError("ccc liana: cannot use both liana_use_raw=True and liana_layer.")
    requested_use_raw = bool(getattr(cfg, "liana_use_raw", False))
    requested_layer = getattr(cfg, "liana_layer", None)
    liana_input_mode = str(getattr(cfg, "liana_input_mode", "counts")).strip().lower()
    if liana_input_mode not in {"counts", "lognorm"}:
        raise RuntimeError("ccc liana: liana_input_mode must be one of {'counts', 'lognorm'}.")
    liana_lognorm_target_sum = float(getattr(cfg, "liana_lognorm_target_sum", 1e4))
    if requested_use_raw and liana_input_mode != "counts":
        raise RuntimeError("ccc liana: liana_use_raw=True is only compatible with liana_input_mode='counts'.")
    if cross_tissue_mode and liana_input_mode == "counts" and not requested_use_raw and requested_layer is None:
        LOGGER.warning(
            "ccc liana: cross-tissue mode is running on raw count-like input (%s). "
            "If source datasets differ strongly in sequencing depth or chemistry, LIANA rankings can be biased by detection depth. "
            "Consider rerunning with --input-mode lognorm for a depth-normalized expression layer.",
            "counts_cb/counts_raw",
        )
    liana_layer = _effective_liana_layer(
        adata,
        requested_use_raw=requested_use_raw,
        layer=requested_layer,
        input_mode=liana_input_mode,
        lognorm_target_sum=liana_lognorm_target_sum,
    )
    global_use_raw = _effective_liana_use_raw(
        adata,
        requested_use_raw=requested_use_raw,
        layer=liana_layer,
    )

    run_specs = _build_liana_condition_run_specs(adata, cfg)

    run_namespace = _run_namespace_for_round(
        adata,
        prefix="ccc_liana",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round
    tables_root.mkdir(parents=True, exist_ok=True)

    adata.uns.setdefault("markers_and_de", {})
    ccc_block = adata.uns["markers_and_de"].setdefault("ccc", {})
    liana_block = ccc_block.setdefault("liana", {})
    runs_store = liana_block.setdefault("runs", {})
    comparison_buckets: dict[str, dict[str, Any]] = {}

    rank_callable = _make_liana_rank_aggregate(li, methods)
    aggregate_score_col = "magnitude_rank"
    aggregate_specificity_col = "specificity_rank"

    for run_spec in run_specs:
        condition_label = str(run_spec["run_label"])
        adata_run = run_spec["adata"]
        condition_key = run_spec["condition_key"]
        condition_value = run_spec["condition_value"]
        use_raw = _effective_liana_use_raw(
            adata_run,
            requested_use_raw=global_use_raw,
            layer=liana_layer,
        )
        cond_fig_rel = Path(run_spec["figs_rel"])
        run_tables_dir = tables_root / Path(run_spec["tables_rel"])
        run_tables_dir.mkdir(parents=True, exist_ok=True)

        per_method_results: dict[str, pd.DataFrame] = {}
        if "rank_aggregate" in methods:
            agg_key = "rank_aggregate"
            agg_df = _call_liana_safely(
                rank_callable,
                adata_run,
                groupby=str(groupby),
                resource_name=str(getattr(cfg, "liana_resource", "consensus")),
                expr_prop=float(getattr(cfg, "liana_expr_prop", 0.1)),
                use_raw=use_raw,
                layer=liana_layer,
                n_perms=getattr(cfg, "liana_n_perms", None),
                seed=int(getattr(cfg, "liana_seed", 42)),
                n_jobs=int(getattr(cfg, "n_jobs", 1)),
                return_all_lrs=bool(getattr(cfg, "liana_return_all_lrs", False)),
                inplace=False,
                verbose=False,
            )
            per_method_results[agg_key] = _sort_liana_results(
                agg_df,
                score_col=aggregate_score_col,
                score_ascending=True,
                specificity_col=aggregate_specificity_col,
                specificity_ascending=True,
            )
            if cross_tissue_mode:
                per_method_results[agg_key] = _filter_liana_results_cross_tissue(
                    per_method_results[agg_key],
                    cluster_dataset_map=cluster_dataset_map,
                    source_levels=source_levels,
                    target_levels=target_levels,
                    signal_scope=signal_scope,
                )

        for method_name in non_aggregate_methods:
            spec = method_specs[str(method_name)]
            res_df = _call_liana_safely(
                spec["callable"],
                adata_run,
                groupby=str(groupby),
                resource_name=str(getattr(cfg, "liana_resource", "consensus")),
                expr_prop=float(getattr(cfg, "liana_expr_prop", 0.1)),
                use_raw=use_raw,
                layer=liana_layer,
                n_perms=getattr(cfg, "liana_n_perms", None),
                seed=int(getattr(cfg, "liana_seed", 42)),
                n_jobs=int(getattr(cfg, "n_jobs", 1)),
                return_all_lrs=bool(getattr(cfg, "liana_return_all_lrs", False)),
                inplace=False,
                verbose=False,
            )
            per_method_results[str(method_name)] = _sort_liana_results(
                res_df,
                score_col=spec["score_col"],
                score_ascending=bool(spec["score_ascending"]),
                specificity_col=spec["specificity_col"],
                specificity_ascending=bool(spec["specificity_ascending"]),
            )
            if cross_tissue_mode:
                per_method_results[str(method_name)] = _filter_liana_results_cross_tissue(
                    per_method_results[str(method_name)],
                    cluster_dataset_map=cluster_dataset_map,
                    source_levels=source_levels,
                    target_levels=target_levels,
                    signal_scope=signal_scope,
                )

        for method_name, df in per_method_results.items():
            df.to_csv(run_tables_dir / f"liana_{method_name}.tsv", sep="\t", index=False)

        primary_method = "rank_aggregate" if "rank_aggregate" in per_method_results else next(iter(per_method_results.keys()))
        primary_df = per_method_results.get(primary_method, pd.DataFrame())
        if primary_df is None or primary_df.empty:
            LOGGER.warning("ccc liana: no interactions returned for condition=%r", str(condition_label))
            primary_top = pd.DataFrame()
            source_target_summary = pd.DataFrame(columns=["source", "target", "n_interactions"])
            route_family_summary = pd.DataFrame(columns=["source", "route_family", "n_interactions"])
        else:
            primary_top = primary_df.head(int(getattr(cfg, "liana_top_n", 250))).copy()
            score_col = aggregate_score_col if primary_method == "rank_aggregate" else method_specs[primary_method]["score_col"]
            source_target_summary = _summarize_liana_source_targets(primary_top, score_col=score_col)
            route_family_summary = _summarize_liana_route_families(primary_top)
            primary_top.to_csv(run_tables_dir / f"liana_{primary_method}_top.tsv", sep="\t", index=False)
            source_target_summary.to_csv(run_tables_dir / "source_target_summary.tsv", sep="\t", index=False)
            route_family_summary.to_csv(run_tables_dir / "route_family_summary.tsv", sep="\t", index=False)

            if bool(getattr(cfg, "make_figures", True)):
                plot_top_n = int(getattr(cfg, "liana_plot_top_n", 60))
                primary_score_col = aggregate_score_col if primary_method == "rank_aggregate" else method_specs[primary_method]["score_col"]
                primary_score_ascending = True if primary_method == "rank_aggregate" else bool(method_specs[primary_method]["score_ascending"])
                plot_primary_df = _prepare_liana_plot_df(primary_df, display_map=display_map)
                plot_primary_family_df = _prepare_liana_family_plot_df(primary_df, display_map=display_map)
                plot_source_target_summary = _prepare_liana_plot_df(source_target_summary, display_map=display_map)
                plot_primary_df["run_label"] = str(condition_label)
                plot_primary_family_df["run_label"] = str(condition_label)
                raw_labels = sorted(
                    set(primary_df["source"].astype(str)).union(set(primary_df["target"].astype(str)))
                ) if not primary_df.empty else []
                plot_color_map = _liana_plot_color_map(
                    adata,
                    cluster_key=str(groupby),
                    display_map=display_map,
                    round_id=getattr(cfg, "round_id", None),
                    raw_labels=raw_labels,
                )
                artifacts = []
                artifacts.extend(
                    plot_utils.plot_liana_source_target_heatmap(
                        plot_source_target_summary,
                        figdir=cond_fig_rel,
                        stem="liana_source_target_heatmap",
                        title=f"{'LIANA cross-tissue signaling' if cross_tissue_mode else 'LIANA source-target summary'} ({condition_label})",
                    )
                )
                if "mean_score" in plot_source_target_summary.columns:
                    artifacts.extend(
                        plot_utils.plot_liana_source_target_heatmap(
                            plot_source_target_summary,
                            figdir=cond_fig_rel,
                            stem="liana_source_target_mean_score_heatmap",
                            title=f"{'LIANA cross-tissue mean score' if cross_tissue_mode else 'LIANA source-target mean score'} ({condition_label})",
                            value_col="mean_score",
                            cmap="mako",
                        )
                    )
                artifacts.extend(
                    plot_utils.plot_liana_send_receive_summary(
                        plot_source_target_summary,
                        figdir=cond_fig_rel,
                        stem="liana_send_receive_n_interactions",
                        title=f"{'LIANA cross-tissue send/receive counts' if cross_tissue_mode else 'LIANA send/receive counts'} ({condition_label})",
                        value_col="n_interactions",
                    )
                )
                if "mean_score" in plot_source_target_summary.columns:
                    artifacts.extend(
                        plot_utils.plot_liana_send_receive_summary(
                            plot_source_target_summary,
                            figdir=cond_fig_rel,
                            stem="liana_send_receive_mean_score",
                            title=f"{'LIANA cross-tissue send/receive mean score' if cross_tissue_mode else 'LIANA send/receive mean score'} ({condition_label})",
                            value_col="mean_score",
                        )
                    )
                artifacts.extend(
                    plot_utils.plot_liana_circos(
                        plot_primary_df,
                        figdir=cond_fig_rel,
                        stem="liana_circos_n_interactions",
                        title=f"{'LIANA cross-tissue circos' if cross_tissue_mode else 'LIANA circos'} ({condition_label})",
                        value_col=None,
                        node_color_map=plot_color_map,
                    )
                )
                if primary_score_col in plot_primary_df.columns:
                    artifacts.extend(
                        plot_utils.plot_liana_circos(
                            plot_primary_df,
                            figdir=cond_fig_rel,
                            stem="liana_circos_mean_score",
                            title=f"{'LIANA cross-tissue circos mean score' if cross_tissue_mode else 'LIANA circos mean score'} ({condition_label})",
                            value_col=primary_score_col,
                            node_color_map=plot_color_map,
                            inverse_score=primary_score_ascending,
                        )
                    )
                artifacts.extend(
                    plot_utils.plot_liana_top_interactions(
                        plot_primary_df,
                        figdir=cond_fig_rel,
                        stem=f"liana_top_{primary_method}",
                        title=f"{'Top LIANA cross-tissue interactions' if cross_tissue_mode else 'Top LIANA interactions'} ({condition_label})",
                        top_n=plot_top_n,
                        score_col=primary_score_col,
                        ascending=primary_score_ascending,
                    )
                )
                artifacts.extend(
                    plot_utils.plot_liana_top_interactions_by_family(
                        plot_primary_family_df,
                        figdir=cond_fig_rel,
                        stem=f"liana_top_{primary_method}_route_families",
                        title=f"{'Top LIANA cross-tissue route families' if cross_tissue_mode else 'Top LIANA route families'} ({condition_label})",
                        top_n=min(plot_top_n, 12),
                        family_col="route_family",
                    )
                )
                artifacts.extend(
                    plot_utils.plot_liana_top_interactions_by_target_cluster(
                        plot_primary_df,
                        figdir=cond_fig_rel,
                        stem=f"liana_top_{primary_method}_target_clusters",
                        title=f"{'Top LIANA cross-tissue target clusters' if cross_tissue_mode else 'Top LIANA target clusters'} ({condition_label})",
                        top_n=min(plot_top_n, 12),
                    )
                )
                plot_utils.persist_plot_artifacts(artifacts)
                compare_bucket = run_spec.get("compare_bucket")
                if compare_bucket:
                    bucket = comparison_buckets.setdefault(
                        str(compare_bucket),
                        {
                            "compare_rel": Path(run_spec["compare_rel"]),
                            "title": run_spec.get("compare_title"),
                            "summaries": [],
                            "primary": [],
                            "primary_family": [],
                        },
                    )
                    if not plot_source_target_summary.empty:
                        plot_compare_df = plot_source_target_summary.copy()
                        plot_compare_df["run_label"] = str(condition_label)
                        bucket["summaries"].append(plot_compare_df)
                    if not plot_primary_df.empty:
                        bucket["primary"].append(plot_primary_df.copy())
                    if not plot_primary_family_df.empty:
                        bucket["primary_family"].append(plot_primary_family_df.copy())

        _write_liana_settings(
            run_tables_dir,
            condition_label=str(condition_label),
            condition_key=condition_key,
            condition_value=condition_value,
            methods=methods,
            aggregated_methods=aggregate_methods,
            resource=str(getattr(cfg, "liana_resource", "consensus")),
            groupby=str(groupby),
            use_raw=use_raw,
            layer=liana_layer,
            input_mode=liana_input_mode,
            lognorm_target_sum=(liana_lognorm_target_sum if liana_input_mode == "lognorm" else None),
            expr_prop=float(getattr(cfg, "liana_expr_prop", 0.1)),
            n_perms=getattr(cfg, "liana_n_perms", None),
            cross_tissue_mode=bool(cross_tissue_mode),
            dataset_key=str(dataset_key) if dataset_key else None,
            source_levels=source_levels,
            target_levels=target_levels,
            signal_scope=str(signal_scope),
        )

        runs_store[str(run_spec["run_id"])] = {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "round_id": getattr(cfg, "round_id", None),
            "groupby": str(groupby),
            "display_map": dict(display_map),
            "condition_key": condition_key,
            "condition_value": condition_value,
            "condition_spec": run_spec.get("condition_spec"),
            "context_key": run_spec.get("context_key"),
            "context_value": run_spec.get("context_value"),
            "cross_tissue_mode": bool(cross_tissue_mode),
            "dataset_key": str(dataset_key) if dataset_key else None,
            "source_levels": list(source_levels),
            "target_levels": list(target_levels),
            "signal_scope": str(signal_scope),
            "resource": str(getattr(cfg, "liana_resource", "consensus")),
            "methods": list(methods),
            "aggregated_methods": list(aggregate_methods),
            "input_mode": liana_input_mode,
            "lognorm_target_sum": liana_lognorm_target_sum if liana_input_mode == "lognorm" else None,
            "primary_method": str(primary_method),
            "top_interactions": primary_top,
            "source_target_summary": source_target_summary,
            "route_family_summary": route_family_summary,
            "n_interactions": int(len(primary_df)) if primary_df is not None else 0,
        }

    if bool(getattr(cfg, "make_figures", True)):
        for bucket in comparison_buckets.values():
            combined_summary = pd.concat(bucket["summaries"], ignore_index=True) if bucket["summaries"] else pd.DataFrame()
            if combined_summary.empty or combined_summary["run_label"].astype(str).nunique() < 2:
                continue
            comparison_artifacts = []
            title_suffix = f" [{bucket['title']}]" if bucket.get("title") else ""
            comparison_artifacts.extend(
                plot_utils.plot_liana_condition_heatmap_grid(
                    combined_summary,
                    figdir=Path(bucket["compare_rel"]),
                    stem="liana_source_target_compare_heatmap",
                    title=f"LIANA source-target summary by run{title_suffix}",
                    value_col="n_interactions",
                )
            )
            if "mean_score" in combined_summary.columns:
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_heatmap_grid(
                        combined_summary,
                        figdir=Path(bucket["compare_rel"]),
                        stem="liana_source_target_compare_mean_score_heatmap",
                        title=f"LIANA source-target mean score by run{title_suffix}",
                        value_col="mean_score",
                        cmap="mako",
                    )
                )
            primary_method_name = None
            for run_payload in runs_store.values():
                primary_method_name = str(run_payload.get("primary_method", "") or "")
                if primary_method_name:
                    break
            if primary_method_name and bucket["primary"]:
                combined_primary = pd.concat(bucket["primary"], ignore_index=True)
                primary_score_col = aggregate_score_col if primary_method_name == "rank_aggregate" else method_specs[primary_method_name]["score_col"]
                primary_score_ascending = True if primary_method_name == "rank_aggregate" else bool(method_specs[primary_method_name]["score_ascending"])
                raw_labels = sorted(
                    set(combined_primary["source"].astype(str)).union(set(combined_primary["target"].astype(str)))
                )
                compare_color_map = _liana_plot_color_map(
                    adata,
                    cluster_key=str(groupby),
                    display_map=display_map,
                    round_id=getattr(cfg, "round_id", None),
                    raw_labels=raw_labels,
                )
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_circos_grid(
                        combined_primary,
                        figdir=Path(bucket["compare_rel"]),
                        stem="liana_circos_by_condition",
                        title=f"LIANA circos by condition{title_suffix}",
                        node_color_map=compare_color_map,
                    )
                )
                if primary_score_col in combined_primary.columns:
                    comparison_artifacts.extend(
                        plot_utils.plot_liana_condition_circos_grid(
                            combined_primary,
                            figdir=Path(bucket["compare_rel"]),
                            stem="liana_circos_mean_score_by_condition",
                            title=f"LIANA circos mean score by condition{title_suffix}",
                            value_col=primary_score_col,
                            node_color_map=compare_color_map,
                            inverse_score=primary_score_ascending,
                        )
                    )
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_split_top_interactions(
                        combined_primary,
                        figdir=Path(bucket["compare_rel"]),
                        stem=f"liana_top_{primary_method_name}_by_condition",
                        title=f"LIANA condition-split top interactions{title_suffix}",
                        top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 6),
                        score_col=primary_score_col,
                        ascending=primary_score_ascending,
                    )
                )
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_split_target_clusters(
                        combined_primary,
                        figdir=Path(bucket["compare_rel"]),
                        stem=f"liana_top_{primary_method_name}_target_clusters_by_condition",
                        title=f"LIANA condition-split target clusters{title_suffix}",
                        top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 12),
                    )
                )
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_split_target_clusters(
                        combined_primary,
                        figdir=Path(bucket["compare_rel"]),
                        stem=f"liana_top_{primary_method_name}_target_cluster_share_by_condition",
                        title="LIANA condition-split target cluster share",
                        top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 12),
                        normalize=True,
                        x_label="target share",
                    )
                )
                if primary_score_col in combined_primary.columns:
                    comparison_artifacts.extend(
                        plot_utils.plot_liana_condition_split_target_clusters(
                            combined_primary,
                            figdir=Path(bucket["compare_rel"]),
                            stem=f"liana_top_{primary_method_name}_target_cluster_mean_score_by_condition",
                            title="LIANA condition-split target cluster mean score",
                            top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 12),
                            value_col=primary_score_col,
                            x_label="mean score",
                        )
                    )
            comparison_artifacts.extend(
                plot_utils.plot_liana_condition_alluvial_grid(
                    combined_summary,
                    figdir=Path(bucket["compare_rel"]),
                    stem="liana_source_target_alluvial_by_condition",
                    title=f"LIANA source-target alluvial by condition{title_suffix}",
                    value_col="n_interactions",
                    top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 10),
                )
            )
            if "mean_score" in combined_summary.columns:
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_alluvial_grid(
                        combined_summary,
                        figdir=Path(bucket["compare_rel"]),
                        stem="liana_source_target_mean_score_alluvial_by_condition",
                        title=f"LIANA source-target mean score alluvial by condition{title_suffix}",
                        value_col="mean_score",
                        top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 10),
                    )
                )
            if primary_method_name and bucket["primary_family"]:
                combined_primary_family = pd.concat(bucket["primary_family"], ignore_index=True)
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_split_family_counts(
                        combined_primary_family,
                        figdir=Path(bucket["compare_rel"]),
                        stem=f"liana_top_{primary_method_name}_route_families_by_condition",
                        title=f"LIANA condition-split route families{title_suffix}",
                        top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 12),
                        family_col="route_family",
                    )
                )
            plot_utils.persist_plot_artifacts(comparison_artifacts)

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_liana")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_liana")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (ccc liana).")
    return adata


def run_nichenet_ccc(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (ccc nichenet)...")
    nichenet_r_lib_dir = _ensure_nichenet_r_runtime(
        install_missing=bool(getattr(cfg, "nichenet_install_missing_r_deps", False))
    )
    nichenet_r_env = _nichenet_r_env(nichenet_r_lib_dir)

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )
    if str(groupby) not in adata.obs:
        raise RuntimeError(f"ccc nichenet: groupby={groupby!r} not found in adata.obs")

    receiver_cluster_spec = str(getattr(cfg, "nichenet_receiver_cluster", "all") or "all").strip()
    all_receiver_clusters = sorted(dict.fromkeys(adata.obs[str(groupby)].astype(str).tolist()))
    if receiver_cluster_spec.lower() == "all":
        receiver_clusters = tuple(all_receiver_clusters)
    else:
        receiver_clusters = (_resolve_cluster_request(receiver_cluster_spec, display_map),)
    requested_sender_clusters = tuple(
        _resolve_cluster_request(str(x), display_map)
        for x in (getattr(cfg, "nichenet_sender_clusters", ()) or ())
        if str(x).strip()
    )
    dataset_key = getattr(cfg, "ccc_dataset_key", None)
    source_levels = tuple(str(x) for x in (getattr(cfg, "ccc_source_levels", ()) or ()) if str(x))
    target_levels = tuple(str(x) for x in (getattr(cfg, "ccc_target_levels", ()) or ()) if str(x))
    signal_scope = str(getattr(cfg, "ccc_signal_scope", "all") or "all").strip().lower()
    cross_tissue_mode = bool(dataset_key)

    requested_layer = _effective_liana_layer(
        adata,
        requested_use_raw=False,
        layer=None,
        input_mode=str(getattr(cfg, "nichenet_input_mode", "counts")),
        lognorm_target_sum=float(getattr(cfg, "nichenet_lognorm_target_sum", 1e4)),
    )
    expressed_use_raw = _effective_liana_use_raw(
        adata,
        requested_use_raw=False,
        layer=requested_layer,
    )
    min_fraction = float(getattr(cfg, "nichenet_expression_pct", 0.10))
    run_specs = _build_nichenet_condition_run_specs(adata, cfg)

    run_namespace = _run_namespace_for_round(
        adata,
        prefix="ccc_nichenet",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round
    tables_root.mkdir(parents=True, exist_ok=True)

    adata.uns.setdefault("markers_and_de", {})
    ccc_block = adata.uns["markers_and_de"].setdefault("ccc", {})
    nichenet_block = ccc_block.setdefault("nichenet", {})
    runs_store = nichenet_block.setdefault("runs", {})

    for run_spec in run_specs:
        adata_run = run_spec["adata"]
        run_tables_dir = tables_root / Path(run_spec["tables_rel"])
        run_tables_dir.mkdir(parents=True, exist_ok=True)
        run_fig_rel = Path(run_spec["figs_rel"])
        compare_key = run_spec.get("compare_key")
        condition_oi = run_spec.get("condition_oi")
        condition_reference = run_spec.get("condition_reference")

        cluster_series = adata_run.obs[str(groupby)].astype(str)
        for receiver_cluster in receiver_clusters:
            receiver_token = _safe_combo_token(receiver_cluster)
            receiver_tables_dir = run_tables_dir / f"receiver__{receiver_token}"
            receiver_tables_dir.mkdir(parents=True, exist_ok=True)
            receiver_fig_rel = run_fig_rel / f"receiver__{receiver_token}"

            receiver_mask = cluster_series.to_numpy() == str(receiver_cluster)
            sender_mask = cluster_series.to_numpy() != str(receiver_cluster)
            if requested_sender_clusters:
                sender_mask &= cluster_series.isin(list(requested_sender_clusters)).to_numpy()
            if cross_tissue_mode:
                if str(dataset_key) not in adata_run.obs:
                    raise RuntimeError(f"ccc nichenet: dataset_key={dataset_key!r} not found in adata.obs")
                dataset_series = adata_run.obs[str(dataset_key)].astype(str)
                sender_mask &= dataset_series.isin(list(source_levels)).to_numpy()
                receiver_mask &= dataset_series.isin(list(target_levels)).to_numpy()
            if int(receiver_mask.sum()) == 0:
                LOGGER.warning(
                    "ccc nichenet: skipping run=%r receiver=%r because no receiver cells matched.",
                    str(run_spec["run_id"]),
                    str(receiver_cluster),
                )
                continue
            if int(sender_mask.sum()) == 0:
                LOGGER.warning(
                    "ccc nichenet: skipping run=%r receiver=%r because no sender cells matched.",
                    str(run_spec["run_id"]),
                    str(receiver_cluster),
                )
                continue

            sender_expressed_genes = _compute_expressed_genes(
                adata_run,
                mask=sender_mask,
                layer=requested_layer,
                use_raw=expressed_use_raw,
                min_fraction=min_fraction,
            )
            receiver_expressed_genes = _compute_expressed_genes(
                adata_run,
                mask=receiver_mask,
                layer=requested_layer,
                use_raw=expressed_use_raw,
                min_fraction=min_fraction,
            )
            if getattr(cfg, "nichenet_gene_list_file", None):
                geneset_oi = _load_gene_list_file(Path(getattr(cfg, "nichenet_gene_list_file")))
                background_genes = sorted(dict.fromkeys(receiver_expressed_genes))
                receiver_de_df = pd.DataFrame(columns=["gene", "logfoldchanges", "pvals_adj"])
            else:
                receiver_adata = adata_run[receiver_mask].copy()
                geneset_oi, background_genes, receiver_de_df = _extract_receiver_de_geneset(
                    receiver_adata,
                    compare_key=str(compare_key),
                    condition_oi=str(condition_oi),
                    condition_reference=str(condition_reference),
                    min_logfc=float(getattr(cfg, "nichenet_min_logfc", 0.25)),
                    padj_threshold=float(getattr(cfg, "nichenet_padj_threshold", 0.05)),
                )
                background_genes = sorted(set(background_genes).intersection(receiver_expressed_genes))
            geneset_oi = sorted(set(geneset_oi).intersection(background_genes))
            if not geneset_oi:
                LOGGER.warning(
                    "ccc nichenet: skipping run=%r receiver=%r because the receiver gene set is empty after filtering.",
                    str(run_spec["run_id"]),
                    str(receiver_cluster),
                )
                continue

            nichenet_res = _run_nichenet_sender_focused(
                sender_expressed_genes=sender_expressed_genes,
                receiver_expressed_genes=receiver_expressed_genes,
                geneset_oi=geneset_oi,
                background_genes=background_genes,
                top_n_ligands=int(getattr(cfg, "nichenet_top_n_ligands", 30)),
                top_n_targets=int(getattr(cfg, "nichenet_top_n_targets", 200)),
                organism=str(getattr(cfg, "nichenet_organism", "human")),
                signal_scope=signal_scope,
                r_env=nichenet_r_env,
            )
            ligand_activity = nichenet_res.get("ligand_activity", pd.DataFrame())
            ligand_target_links = nichenet_res.get("ligand_target_links", pd.DataFrame())
            ligand_receptor_links = nichenet_res.get("ligand_receptor_links", pd.DataFrame())
            potential_ligands = nichenet_res.get("potential_ligands", pd.DataFrame())

            ligand_activity.to_csv(receiver_tables_dir / "nichenet_ligand_activity.tsv", sep="\t", index=False)
            ligand_target_links.to_csv(receiver_tables_dir / "nichenet_ligand_target_links.tsv", sep="\t", index=False)
            ligand_receptor_links.to_csv(receiver_tables_dir / "nichenet_ligand_receptor_links.tsv", sep="\t", index=False)
            potential_ligands.to_csv(receiver_tables_dir / "nichenet_potential_ligands.tsv", sep="\t", index=False)
            receiver_de_df.to_csv(receiver_tables_dir / "nichenet_receiver_de.tsv", sep="\t", index=False)

            settings_lines = [
                f"receiver_cluster\t{receiver_cluster}",
                f"condition_spec\t{run_spec.get('condition_spec')}",
                f"compare_key\t{compare_key}",
                f"condition_oi\t{condition_oi}",
                f"condition_reference\t{condition_reference}",
                f"cross_tissue_mode\t{cross_tissue_mode}",
                f"dataset_key\t{dataset_key}",
                f"source_levels\t{','.join(source_levels)}",
                f"target_levels\t{','.join(target_levels)}",
                f"signal_scope\t{signal_scope}",
                f"project_r_lib\t{nichenet_r_lib_dir}",
                f"expression_pct\t{min_fraction}",
                f"n_sender_expressed_genes\t{len(sender_expressed_genes)}",
                f"n_receiver_expressed_genes\t{len(receiver_expressed_genes)}",
                f"n_geneset_oi\t{len(geneset_oi)}",
            ]
            (receiver_tables_dir / "nichenet_settings.tsv").write_text("\n".join(settings_lines) + "\n")

            if bool(getattr(cfg, "make_figures", True)):
                artifacts = []
                artifacts.extend(
                    plot_utils.plot_nichenet_top_ligands(
                        ligand_activity,
                        figdir=receiver_fig_rel,
                        stem="nichenet_top_ligands",
                        title=f"NicheNet top ligands ({run_spec['run_label']}) [{_liana_plot_label(receiver_cluster, display_map)}]",
                        top_n=int(getattr(cfg, "nichenet_top_n_ligands", 30)),
                    )
                )
                artifacts.extend(
                    plot_utils.plot_nichenet_ligand_target_heatmap(
                        ligand_target_links,
                        figdir=receiver_fig_rel,
                        stem="nichenet_ligand_target_heatmap",
                        title=f"NicheNet ligand-target links ({run_spec['run_label']}) [{_liana_plot_label(receiver_cluster, display_map)}]",
                        top_n_ligands=min(int(getattr(cfg, "nichenet_top_n_ligands", 30)), 15),
                        top_n_targets=min(int(getattr(cfg, "nichenet_top_n_targets", 200)), 40),
                    )
                )
                plot_utils.persist_plot_artifacts(artifacts)

            run_id = f"{run_spec['run_id']}::receiver={receiver_cluster}"
            runs_store[str(run_id)] = {
                "version": __version__,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "round_id": getattr(cfg, "round_id", None),
                "groupby": str(groupby),
                "receiver_cluster": str(receiver_cluster),
                "receiver_cluster_label": _liana_plot_label(receiver_cluster, display_map),
                "receiver_cluster_mode": "all" if receiver_cluster_spec.lower() == "all" else "single",
                "display_map": dict(display_map),
                "condition_spec": run_spec.get("condition_spec"),
                "context_key": run_spec.get("context_key"),
                "context_value": run_spec.get("context_value"),
                "compare_key": compare_key,
                "condition_oi": condition_oi,
                "condition_reference": condition_reference,
                "cross_tissue_mode": bool(cross_tissue_mode),
                "dataset_key": str(dataset_key) if dataset_key else None,
                "source_levels": list(source_levels),
                "target_levels": list(target_levels),
                "signal_scope": str(signal_scope),
                "project_r_lib": str(nichenet_r_lib_dir),
                "ligand_activity": ligand_activity,
                "ligand_target_links": ligand_target_links,
                "ligand_receptor_links": ligand_receptor_links,
                "potential_ligands": potential_ligands,
                "receiver_de": receiver_de_df,
            }

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_nichenet")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")
    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_nichenet")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
    LOGGER.info("Finished markers-and-de (ccc nichenet).")
    return adata


def run_mebocost_ccc(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (ccc mebocost)...")
    install_missing = bool(getattr(cfg, "mebocost_install_missing_python_deps", False))
    mebocost_api = _import_mebocost_api(
        install_missing=install_missing
    )
    mebocost_config_path = _ensure_mebocost_resource_config(install_missing=install_missing)

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )
    if str(groupby) not in adata.obs:
        raise RuntimeError(f"ccc mebocost: groupby={groupby!r} not found in adata.obs")

    dataset_key = getattr(cfg, "ccc_dataset_key", None)
    source_levels = tuple(str(x) for x in (getattr(cfg, "ccc_source_levels", ()) or ()) if str(x))
    target_levels = tuple(str(x) for x in (getattr(cfg, "ccc_target_levels", ()) or ()) if str(x))
    cross_tissue_mode = bool(dataset_key)
    if cross_tissue_mode:
        if not source_levels or not target_levels:
            raise RuntimeError("ccc mebocost: cross-tissue mode requires at least one source level and one target level.")
        cluster_dataset_map = _resolve_liana_cluster_dataset_levels(
            adata,
            cluster_key=str(groupby),
            dataset_key=str(dataset_key),
        )
    else:
        cluster_dataset_map = {}
    mebocost_input_mode = str(getattr(cfg, "mebocost_input_mode", "counts")).strip().lower()
    if mebocost_input_mode not in {"counts", "lognorm"}:
        raise RuntimeError("ccc mebocost: mebocost_input_mode must be one of {'counts', 'lognorm'}.")
    mebocost_lognorm_target_sum = float(getattr(cfg, "mebocost_lognorm_target_sum", 1e4))
    if cross_tissue_mode and mebocost_input_mode == "counts":
        LOGGER.warning(
            "ccc mebocost: cross-tissue mode is running on raw count-like input (counts_cb/counts_raw). "
            "If source datasets differ strongly in sequencing depth or chemistry, MEBOCOST event ranking can be biased by detection depth. "
            "Consider rerunning with --input-mode lognorm for a depth-normalized expression layer."
        )
    mebocost_layer = _effective_mebocost_layer(
        adata,
        input_mode=mebocost_input_mode,
        lognorm_target_sum=mebocost_lognorm_target_sum,
    )

    run_specs = _build_liana_condition_run_specs(adata, cfg, log_prefix="ccc mebocost")
    run_namespace = _run_namespace_for_round(
        adata,
        prefix="ccc_mebocost",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round
    tables_root.mkdir(parents=True, exist_ok=True)

    adata.uns.setdefault("markers_and_de", {})
    ccc_block = adata.uns["markers_and_de"].setdefault("ccc", {})
    mebocost_block = ccc_block.setdefault("mebocost", {})
    runs_store = mebocost_block.setdefault("runs", {})
    comparison_buckets: dict[str, dict[str, Any]] = {}

    for run_spec in run_specs:
        adata_run = run_spec["adata"].copy()
        run_id = str(run_spec["run_id"])
        condition_label = str(run_spec["run_label"])
        run_tables_dir = tables_root / Path(run_spec["tables_rel"])
        run_tables_dir.mkdir(parents=True, exist_ok=True)
        run_fig_rel = Path(run_spec["figs_rel"])
        adata_mebo = adata_run.copy()
        if mebocost_layer is not None:
            adata_mebo.X = adata_mebo.layers[mebocost_layer].copy()

        create_obj = getattr(mebocost_api, "create_obj", None)
        if create_obj is None:
            raise RuntimeError("ccc mebocost: imported MEBOCOST API does not expose create_obj().")
        mebo_obj = _call_with_supported_kwargs(
            create_obj,
            adata=adata_mebo,
            group_col=str(groupby),
            species=str(getattr(cfg, "mebocost_organism", "human")).strip().lower(),
            config_path=str(mebocost_config_path),
        )
        infer_fn = getattr(mebo_obj, "infer_commu", None)
        if infer_fn is None:
            raise RuntimeError("ccc mebocost: MEBOCOST object does not expose infer_commu().")

        infer_res = _call_with_supported_kwargs(
            infer_fn,
            n_shuffle=int(getattr(cfg, "mebocost_n_shuffle", 1000)),
            n_permutations=int(getattr(cfg, "mebocost_n_shuffle", 1000)),
            seed=int(getattr(cfg, "mebocost_seed", 42)),
            Return=True,
            save_permuation=False,
            save_permutation=False,
            min_cell_number=int(getattr(cfg, "mebocost_min_cell_number", 10)),
            pval_method="permutation_test_fdr",
            pval_cutoff=float(getattr(cfg, "mebocost_pval_cutoff", 0.05)),
        )

        commu_res_raw = getattr(mebo_obj, "commu_res", None)
        if not isinstance(commu_res_raw, pd.DataFrame) and isinstance(infer_res, pd.DataFrame):
            commu_res_raw = infer_res
        if not isinstance(commu_res_raw, pd.DataFrame):
            commu_res_raw = pd.DataFrame()

        commu_res = _standardize_mebocost_commu_table(commu_res_raw)
        if cross_tissue_mode:
            commu_res = _filter_liana_results_cross_tissue(
                commu_res,
                cluster_dataset_map=cluster_dataset_map,
                source_levels=source_levels,
                target_levels=target_levels,
                signal_scope="all",
            )
        commu_res = _annotate_mebocost_commu_table(
            commu_res,
            config_path=mebocost_config_path,
            species=str(getattr(cfg, "mebocost_organism", "human")).strip().lower(),
        )
        if "pval" in commu_res.columns:
            sig_res = commu_res.loc[commu_res["pval"].fillna(np.inf) <= float(getattr(cfg, "mebocost_pval_cutoff", 0.05))].copy()
        else:
            sig_res = commu_res.copy()
        plot_base = sig_res if not sig_res.empty else commu_res
        summary = _summarize_mebocost_source_target(plot_base)
        metabolite_superclass_summary = _summarize_mebocost_annotation(plot_base, "super_class")
        sensor_annotation_summary = _summarize_mebocost_annotation(plot_base, "sensor_annotation")
        metabolite_superclass_by_source = _summarize_mebocost_annotation(plot_base, "super_class", focus_col="source")
        metabolite_superclass_by_target = _summarize_mebocost_annotation(plot_base, "super_class", focus_col="target")
        sensor_annotation_by_source = _summarize_mebocost_annotation(plot_base, "sensor_annotation", focus_col="source")
        sensor_annotation_by_target = _summarize_mebocost_annotation(plot_base, "sensor_annotation", focus_col="target")
        commu_res = _prepare_mebocost_plot_df(commu_res, display_map=display_map)
        sig_res = _prepare_mebocost_plot_df(sig_res, display_map=display_map)
        plot_base = _prepare_mebocost_plot_df(plot_base, display_map=display_map)
        summary = _prepare_mebocost_plot_df(summary, display_map=display_map)
        metabolite_superclass_summary = _prepare_mebocost_plot_df(metabolite_superclass_summary, display_map=display_map)
        sensor_annotation_summary = _prepare_mebocost_plot_df(sensor_annotation_summary, display_map=display_map)
        metabolite_superclass_by_source = _prepare_mebocost_plot_df(metabolite_superclass_by_source, display_map=display_map)
        metabolite_superclass_by_target = _prepare_mebocost_plot_df(metabolite_superclass_by_target, display_map=display_map)
        sensor_annotation_by_source = _prepare_mebocost_plot_df(sensor_annotation_by_source, display_map=display_map)
        sensor_annotation_by_target = _prepare_mebocost_plot_df(sensor_annotation_by_target, display_map=display_map)

        commu_res.to_csv(run_tables_dir / "mebocost_commu_res.tsv", sep="\t", index=False)
        sig_res.to_csv(run_tables_dir / "mebocost_sig_res.tsv", sep="\t", index=False)
        summary.to_csv(run_tables_dir / "mebocost_source_target_summary.tsv", sep="\t", index=False)
        metabolite_superclass_summary.to_csv(run_tables_dir / "mebocost_metabolite_superclass_summary.tsv", sep="\t", index=False)
        sensor_annotation_summary.to_csv(run_tables_dir / "mebocost_sensor_annotation_summary.tsv", sep="\t", index=False)
        metabolite_superclass_by_source.to_csv(run_tables_dir / "mebocost_metabolite_superclass_by_source.tsv", sep="\t", index=False)
        metabolite_superclass_by_target.to_csv(run_tables_dir / "mebocost_metabolite_superclass_by_target.tsv", sep="\t", index=False)
        sensor_annotation_by_source.to_csv(run_tables_dir / "mebocost_sensor_annotation_by_source.tsv", sep="\t", index=False)
        sensor_annotation_by_target.to_csv(run_tables_dir / "mebocost_sensor_annotation_by_target.tsv", sep="\t", index=False)

        settings_lines = [
            f"condition_spec\t{run_spec.get('condition_spec')}",
            f"cross_tissue_mode\t{cross_tissue_mode}",
            f"dataset_key\t{dataset_key}",
            f"source_levels\t{','.join(source_levels)}",
            f"target_levels\t{','.join(target_levels)}",
            f"organism\t{getattr(cfg, 'mebocost_organism', 'human')}",
            f"input_mode\t{mebocost_input_mode}",
            f"lognorm_target_sum\t{mebocost_lognorm_target_sum if mebocost_input_mode == 'lognorm' else ''}",
            f"expression_layer\t{mebocost_layer}",
            f"n_shuffle\t{int(getattr(cfg, 'mebocost_n_shuffle', 1000))}",
            f"min_cell_number\t{int(getattr(cfg, 'mebocost_min_cell_number', 10))}",
            f"pval_cutoff\t{float(getattr(cfg, 'mebocost_pval_cutoff', 0.05))}",
        ]
        (run_tables_dir / "mebocost_settings.tsv").write_text("\n".join(settings_lines) + "\n")

        if bool(getattr(cfg, "make_figures", True)):
            artifacts = []
            artifacts.extend(
                plot_utils.plot_mebocost_top_events(
                    plot_base,
                    figdir=run_fig_rel,
                    stem="mebocost_top_events",
                    title=f"{'MEBOCOST cross-tissue top metabolite-sensor events' if cross_tissue_mode else 'MEBOCOST top metabolite-sensor events'} ({condition_label})",
                    top_n=int(getattr(cfg, "mebocost_plot_top_n", 40)),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_annotation_summary_bars(
                    plot_base,
                    figdir=run_fig_rel,
                    stem="mebocost_metabolite_superclass_summary",
                    annotation_col="super_class",
                    title=f"{'MEBOCOST metabolite super-classes' if not cross_tissue_mode else 'MEBOCOST cross-tissue metabolite super-classes'} ({condition_label})",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_annotation_summary_bars(
                    plot_base,
                    figdir=run_fig_rel,
                    stem="mebocost_sensor_annotation_summary",
                    annotation_col="sensor_annotation",
                    title=f"{'MEBOCOST sensor annotation classes' if not cross_tissue_mode else 'MEBOCOST cross-tissue sensor annotation classes'} ({condition_label})",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_source_target_heatmap(
                    summary,
                    figdir=run_fig_rel,
                    stem="mebocost_source_target_heatmap",
                    title=f"{'MEBOCOST cross-tissue event count heatmap' if cross_tissue_mode else 'MEBOCOST event count heatmap'} ({condition_label})",
                    value_col="n_events",
                )
            )
            if "mean_score" in summary.columns and summary["mean_score"].notna().any():
                artifacts.extend(
                    plot_utils.plot_mebocost_source_target_heatmap(
                        summary,
                        figdir=run_fig_rel,
                        stem="mebocost_source_target_mean_score_heatmap",
                        title=f"{'MEBOCOST cross-tissue mean score heatmap' if cross_tissue_mode else 'MEBOCOST mean score heatmap'} ({condition_label})",
                        value_col="mean_score",
                        cmap="mako",
                    )
                )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_partner_bars(
                    summary,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_top_target_clusters_by_source",
                    focus="source",
                    title_prefix="MEBOCOST top targets by event count",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_partner_bars(
                    summary,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_top_source_clusters_by_target",
                    focus="target",
                    title_prefix="MEBOCOST top senders by event count",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            if "mean_score" in summary.columns and summary["mean_score"].notna().any():
                artifacts.extend(
                    plot_utils.plot_mebocost_cluster_partner_bars(
                        summary,
                        figdir=run_fig_rel,
                        stem_prefix="mebocost_top_target_clusters_by_source_mean_score",
                        focus="source",
                        title_prefix="MEBOCOST top targets by mean score",
                        value_col="mean_score",
                        top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                    )
                )
                artifacts.extend(
                    plot_utils.plot_mebocost_cluster_partner_bars(
                        summary,
                        figdir=run_fig_rel,
                        stem_prefix="mebocost_top_source_clusters_by_target_mean_score",
                        focus="target",
                        title_prefix="MEBOCOST top senders by mean score",
                        value_col="mean_score",
                        top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_annotation_bars(
                    metabolite_superclass_by_source,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_metabolite_superclass_by_source",
                    focus="source",
                    annotation_col="super_class",
                    title_prefix="MEBOCOST metabolite super-classes",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_annotation_bars(
                    metabolite_superclass_by_target,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_metabolite_superclass_by_target",
                    focus="target",
                    annotation_col="super_class",
                    title_prefix="MEBOCOST metabolite super-classes",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_annotation_bars(
                    sensor_annotation_by_source,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_sensor_annotation_by_source",
                    focus="source",
                    annotation_col="sensor_annotation",
                    title_prefix="MEBOCOST sensor annotations",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_annotation_bars(
                    sensor_annotation_by_target,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_sensor_annotation_by_target",
                    focus="target",
                    annotation_col="sensor_annotation",
                    title_prefix="MEBOCOST sensor annotations",
                    value_col="n_events",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_event_bars(
                    plot_base,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_top_events_by_source",
                    focus="source",
                    title_prefix="MEBOCOST",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            artifacts.extend(
                plot_utils.plot_mebocost_cluster_event_bars(
                    plot_base,
                    figdir=run_fig_rel,
                    stem_prefix="mebocost_top_events_by_target",
                    focus="target",
                    title_prefix="MEBOCOST",
                    top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 12),
                )
            )
            plot_utils.persist_plot_artifacts(artifacts)
            compare_bucket = run_spec.get("compare_bucket")
            if compare_bucket:
                bucket = comparison_buckets.setdefault(
                    str(compare_bucket),
                    {
                        "compare_rel": Path(run_spec["compare_rel"]),
                        "title": run_spec.get("compare_title"),
                        "summaries": [],
                        "events": [],
                        "met_source": [],
                        "met_target": [],
                        "sensor_source": [],
                        "sensor_target": [],
                    },
                )
                if not summary.empty:
                    tmp = summary.copy()
                    tmp["run_label"] = str(condition_label)
                    bucket["summaries"].append(tmp)
                if not plot_base.empty:
                    tmp = plot_base.copy()
                    tmp["run_label"] = str(condition_label)
                    bucket["events"].append(tmp)
                for key, df_summary in (
                    ("met_source", metabolite_superclass_by_source),
                    ("met_target", metabolite_superclass_by_target),
                    ("sensor_source", sensor_annotation_by_source),
                    ("sensor_target", sensor_annotation_by_target),
                ):
                    if not df_summary.empty:
                        tmp = df_summary.copy()
                        tmp["run_label"] = str(condition_label)
                        bucket[key].append(tmp)

        runs_store[run_id] = {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "round_id": getattr(cfg, "round_id", None),
            "groupby": str(groupby),
            "display_map": dict(display_map),
            "condition_spec": run_spec.get("condition_spec"),
            "context_key": run_spec.get("context_key"),
            "context_value": run_spec.get("context_value"),
            "condition_label": condition_label,
            "cross_tissue_mode": bool(cross_tissue_mode),
            "dataset_key": str(dataset_key) if dataset_key else None,
            "source_levels": list(source_levels),
            "target_levels": list(target_levels),
            "organism": str(getattr(cfg, "mebocost_organism", "human")),
            "input_mode": mebocost_input_mode,
            "lognorm_target_sum": mebocost_lognorm_target_sum if mebocost_input_mode == "lognorm" else None,
            "expression_layer": mebocost_layer,
            "commu_res": commu_res,
            "sig_res": sig_res,
            "source_target_summary": summary,
            "metabolite_superclass_summary": metabolite_superclass_summary,
            "sensor_annotation_summary": sensor_annotation_summary,
            "metabolite_superclass_by_source": metabolite_superclass_by_source,
            "metabolite_superclass_by_target": metabolite_superclass_by_target,
            "sensor_annotation_by_source": sensor_annotation_by_source,
            "sensor_annotation_by_target": sensor_annotation_by_target,
        }

    if bool(getattr(cfg, "make_figures", True)):
        for bucket in comparison_buckets.values():
            title_suffix = f" [{bucket['title']}]" if bucket.get("title") else ""
            comparison_artifacts = []
            combined_summary = pd.concat(bucket["summaries"], ignore_index=True) if bucket["summaries"] else pd.DataFrame()
            if not combined_summary.empty and combined_summary["run_label"].astype(str).nunique() >= 2:
                comparison_artifacts.extend(
                    plot_utils.plot_liana_condition_heatmap_grid(
                        combined_summary,
                        figdir=Path(bucket["compare_rel"]),
                        stem="mebocost_source_target_compare_heatmap",
                        title=f"MEBOCOST source-target event count by run{title_suffix}",
                        value_col="n_events",
                        cmap="viridis",
                    )
                )
                if "mean_score" in combined_summary.columns and combined_summary["mean_score"].notna().any():
                    comparison_artifacts.extend(
                        plot_utils.plot_liana_condition_heatmap_grid(
                            combined_summary,
                            figdir=Path(bucket["compare_rel"]),
                            stem="mebocost_source_target_compare_mean_score_heatmap",
                            title=f"MEBOCOST source-target mean score by run{title_suffix}",
                            value_col="mean_score",
                            cmap="mako",
                        )
                    )
            combined_events = pd.concat(bucket["events"], ignore_index=True) if bucket["events"] else pd.DataFrame()
            if not combined_events.empty and combined_events["run_label"].astype(str).nunique() >= 2:
                comparison_artifacts.extend(
                    plot_utils.plot_mebocost_condition_split_events(
                        combined_events,
                        figdir=Path(bucket["compare_rel"]),
                        focus="source",
                        stem="mebocost_top_events_by_source_by_condition",
                        title=f"MEBOCOST condition-split top events by source{title_suffix}",
                        top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 6),
                    )
                )
                comparison_artifacts.extend(
                    plot_utils.plot_mebocost_condition_split_events(
                        combined_events,
                        figdir=Path(bucket["compare_rel"]),
                        focus="target",
                        stem="mebocost_top_events_by_target_by_condition",
                        title=f"MEBOCOST condition-split top events by target{title_suffix}",
                        top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 6),
                    )
                )
            for payload, focus, annotation_col, stem, title in (
                ("met_source", "source", "super_class", "mebocost_metabolite_superclass_by_source_by_condition", "MEBOCOST condition-split metabolite super-classes by source"),
                ("met_target", "target", "super_class", "mebocost_metabolite_superclass_by_target_by_condition", "MEBOCOST condition-split metabolite super-classes by target"),
                ("sensor_source", "source", "sensor_annotation", "mebocost_sensor_annotation_by_source_by_condition", "MEBOCOST condition-split sensor annotations by source"),
                ("sensor_target", "target", "sensor_annotation", "mebocost_sensor_annotation_by_target_by_condition", "MEBOCOST condition-split sensor annotations by target"),
            ):
                combined_annotation = pd.concat(bucket[payload], ignore_index=True) if bucket[payload] else pd.DataFrame()
                if combined_annotation.empty or combined_annotation["run_label"].astype(str).nunique() < 2:
                    continue
                comparison_artifacts.extend(
                    plot_utils.plot_mebocost_condition_split_annotations(
                        combined_annotation,
                        figdir=Path(bucket["compare_rel"]),
                        focus=focus,
                        annotation_col=annotation_col,
                        stem=stem,
                        title=f"{title}{title_suffix}",
                        value_col="n_events",
                        top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 8),
                        x_label="n_events",
                    )
                )
            plot_utils.persist_plot_artifacts(comparison_artifacts)

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_mebocost")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")
    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_mebocost")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
    LOGGER.info("Finished markers-and-de (ccc mebocost).")
    return adata


def run_liana_paired_rescore(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (ccc liana paired-rescore)...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )
    if str(groupby) not in adata.obs:
        raise RuntimeError(f"ccc liana paired-rescore: groupby={groupby!r} not found in adata.obs")

    liana_input_mode = str(getattr(cfg, "liana_input_mode", "counts")).strip().lower()
    if liana_input_mode not in {"counts", "lognorm"}:
        raise RuntimeError("ccc liana paired-rescore: liana_input_mode must be one of {'counts', 'lognorm'}.")
    requested_layer = _effective_liana_layer(
        adata,
        requested_use_raw=False,
        layer=None,
        input_mode=liana_input_mode,
        lognorm_target_sum=float(getattr(cfg, "liana_lognorm_target_sum", 1e4)),
    )
    pairing_key = str(getattr(cfg, "liana_pairing_key", None) or "sample_id").strip()
    filtered_adata, primary_condition_key, condition_cols, condition_label = _resolve_condition_filter_context(
        adata,
        condition_spec=getattr(cfg, "ccc_condition_key", None),
        condition_values=tuple(str(x) for x in (getattr(cfg, "ccc_condition_values", ()) or ()) if str(x)),
    )
    dataset_key = getattr(cfg, "ccc_dataset_key", None)
    source_levels = tuple(str(x) for x in (getattr(cfg, "ccc_source_levels", ()) or ()) if str(x))
    target_levels = tuple(str(x) for x in (getattr(cfg, "ccc_target_levels", ()) or ()) if str(x))
    cross_tissue_mode = bool(dataset_key)

    if cross_tissue_mode and liana_input_mode == "counts":
        LOGGER.warning(
            "ccc liana paired-rescore: cross-tissue mode is running on raw count-like input (counts_cb/counts_raw). "
            "If source datasets differ strongly in sequencing depth or chemistry, donor-level LIANA rescoring can be biased by detection depth. "
            "Consider rerunning with --input-mode lognorm for a depth-normalized expression layer."
        )
    values_logged = requested_layer is not None and str(requested_layer).startswith("lognorm_")

    candidate_df = _read_liana_candidate_events(Path(getattr(cfg, "liana_candidate_events")))
    candidate_df = _normalize_liana_candidate_events(
        candidate_df,
        display_map=display_map,
        valid_group_tokens=sorted(filtered_adata.obs[str(groupby)].astype(str).unique().tolist()),
    )
    if cross_tissue_mode:
        if not source_levels or not target_levels:
            raise RuntimeError("ccc liana paired-rescore: cross-tissue mode requires at least one source level and one target level.")
        source_allowed = {str(x) for x in source_levels}
        target_allowed = {str(x) for x in target_levels}
        if "source_dataset_level" in candidate_df.columns:
            candidate_df = candidate_df[candidate_df["source_dataset_level"].astype(str).isin(source_allowed)].copy()
        if "target_dataset_level" in candidate_df.columns:
            candidate_df = candidate_df[candidate_df["target_dataset_level"].astype(str).isin(target_allowed)].copy()
    candidate_df = _filter_liana_candidate_events(candidate_df, cfg)

    edge_scores = _score_liana_paired_edges(
        filtered_adata,
        candidate_df=candidate_df,
        groupby=str(groupby),
        pairing_key=pairing_key,
        condition_cols=condition_cols,
        dataset_key=str(dataset_key) if dataset_key else None,
        source_levels=source_levels,
        target_levels=target_levels,
        layer=requested_layer,
        values_logged=values_logged,
        min_sender_cells=int(getattr(cfg, "liana_min_sender_cells", 5)),
        min_receiver_cells=int(getattr(cfg, "liana_min_receiver_cells", 5)),
    )
    route_scores = _summarize_liana_paired_routes(edge_scores)
    edge_missingness = _summarize_paired_missingness(
        edge_scores,
        group_by=("source_token", "target_token", "source_label", "target_label", "branch_pair", "route_family", "ligand_complex", "receptor_complex"),
        score_col="edge_score",
        primary_condition_key=primary_condition_key,
    )
    route_missingness = _summarize_paired_missingness(
        route_scores,
        group_by=("source_token", "target_token", "source_label", "target_label", "branch_pair", "route_family"),
        score_col="mean_edge_score",
        primary_condition_key=primary_condition_key,
    )
    compare_levels = tuple(str(x) for x in (getattr(cfg, "ccc_compare_levels", ()) or ()) if str(x))
    edge_effects = _summarize_liana_paired_effects(
        edge_scores,
        primary_condition_key=primary_condition_key,
        condition_cols=condition_cols,
        compare_levels=compare_levels,
        value_col="edge_score",
        group_by=(
            "source_token",
            "target_token",
            "source_label",
            "target_label",
            "branch_pair",
            "route_family",
            "ligand_complex",
            "receptor_complex",
        ),
        median_support_cols=("sender_n_cells", "receiver_n_cells"),
        min_scored_donors_per_group=int(getattr(cfg, "liana_min_scored_donors_per_group", 3)),
    )
    route_effects = _summarize_liana_paired_effects(
        route_scores,
        primary_condition_key=primary_condition_key,
        condition_cols=condition_cols,
        compare_levels=compare_levels,
        value_col="mean_edge_score",
        group_by=("source_token", "target_token", "source_label", "target_label", "branch_pair", "route_family"),
        median_support_cols=("n_edges_scored",),
        min_scored_donors_per_group=int(getattr(cfg, "liana_min_scored_donors_per_group", 3)),
    )

    run_namespace = _run_namespace_for_round(
        adata,
        prefix="ccc_liana",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round / "paired_rescore" / _safe_combo_token(condition_label)
    tables_root.mkdir(parents=True, exist_ok=True)
    edge_scores.to_csv(tables_root / "liana_paired_lr_edge_scores.tsv", sep="\t", index=False)
    route_scores.to_csv(tables_root / "liana_paired_route_scores.tsv", sep="\t", index=False)
    edge_missingness.to_csv(tables_root / "liana_paired_lr_edge_missingness.tsv", sep="\t", index=False)
    route_missingness.to_csv(tables_root / "liana_paired_route_missingness.tsv", sep="\t", index=False)
    edge_effects.to_csv(tables_root / "liana_paired_lr_edge_effects.tsv", sep="\t", index=False)
    route_effects.to_csv(tables_root / "liana_paired_route_effects.tsv", sep="\t", index=False)
    if edge_effects.empty and route_effects.empty:
        n_scored = int(pd.to_numeric(edge_scores.get("edge_score", pd.Series(dtype=float)), errors="coerce").notna().sum())
        LOGGER.warning(
            "ccc liana paired-rescore: no group effects were produced for %s. Scored donor-level edges=%d. "
            "This usually reflects donor-level sparsity or scored-donor thresholds rather than absence of biology. "
            "Inspect liana_paired_lr_edge_missingness.tsv and consider relaxing --min-sender-cells, --min-receiver-cells, or --min-scored-donors-per-group.",
            str(condition_label),
            n_scored,
        )

    if bool(getattr(cfg, "make_figures", False)):
        fig_rel = Path("paired_rescore") / _safe_combo_token(condition_label)
        artifacts = []
        artifacts.extend(
            plot_utils.plot_liana_paired_route_dotplot(
                route_effects,
                figdir=fig_rel,
                stem_prefix="liana_paired_route_dotplot",
                title_prefix="LIANA paired route effects",
                top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 20),
            )
        )
        artifacts.extend(
            plot_utils.plot_liana_paired_edge_strip(
                edge_effects,
                figdir=fig_rel,
                stem_prefix="liana_paired_lr_edge_strip",
                title_prefix="LIANA paired LR edge effects",
                top_n=min(int(getattr(cfg, "liana_plot_top_n", 60)), 16),
            )
        )
        plot_utils.persist_plot_artifacts(artifacts)

    settings_lines = [
        f"input_path\t{getattr(cfg, 'input_path')}",
        f"candidate_events\t{getattr(cfg, 'liana_candidate_events')}",
        f"groupby\t{groupby}",
        f"pairing_key\t{pairing_key}",
        f"condition_key\t{getattr(cfg, 'ccc_condition_key', None)}",
        f"condition_values\t{','.join(str(x) for x in getattr(cfg, 'ccc_condition_values', ()) or ())}",
        f"compare_levels\t{','.join(compare_levels)}",
        f"dataset_key\t{dataset_key}",
        f"source_levels\t{','.join(source_levels)}",
        f"target_levels\t{','.join(target_levels)}",
        f"input_mode\t{liana_input_mode}",
        f"lognorm_target_sum\t{getattr(cfg, 'liana_lognorm_target_sum', 1e4) if liana_input_mode == 'lognorm' else ''}",
        f"expression_layer\t{requested_layer}",
        "edge_score_formula\tsqrt(ligand_expr * receptor_expr)",
        f"min_sender_cells\t{int(getattr(cfg, 'liana_min_sender_cells', 5))}",
        f"min_receiver_cells\t{int(getattr(cfg, 'liana_min_receiver_cells', 5))}",
        f"min_scored_donors_per_group\t{int(getattr(cfg, 'liana_min_scored_donors_per_group', 3))}",
    ]
    (tables_root / "liana_paired_settings.tsv").write_text("\n".join(settings_lines) + "\n")

    adata.uns.setdefault("markers_and_de", {})
    ccc_block = adata.uns["markers_and_de"].setdefault("ccc", {})
    paired_block = ccc_block.setdefault("liana_paired_rescore", {})
    paired_block[str(condition_label)] = {
        "edge_scores": edge_scores,
        "route_scores": route_scores,
        "edge_missingness": edge_missingness,
        "route_missingness": route_missingness,
        "edge_effects": edge_effects,
        "route_effects": route_effects,
        "candidate_events": candidate_df,
        "input_mode": liana_input_mode,
        "expression_layer": requested_layer,
        "pairing_key": pairing_key,
    }
    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_liana_paired")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")
    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_liana_paired")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
    LOGGER.info("Finished markers-and-de (ccc liana paired-rescore).")
    return adata


def run_mebocost_paired_rescore(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (ccc mebocost paired-rescore)...")
    install_missing = bool(getattr(cfg, "mebocost_install_missing_python_deps", False))
    mebocost_api = _import_mebocost_api(install_missing=install_missing)
    mebocost_config_path = _ensure_mebocost_resource_config(install_missing=install_missing)

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )
    if str(groupby) not in adata.obs:
        raise RuntimeError(f"ccc mebocost paired-rescore: groupby={groupby!r} not found in adata.obs")

    pairing_key = str(getattr(cfg, "mebocost_pairing_key", None) or "sample_id").strip()
    filtered_adata, primary_condition_key, condition_cols, condition_label = _resolve_condition_filter_context(
        adata,
        condition_spec=getattr(cfg, "ccc_condition_key", None),
        condition_values=tuple(str(x) for x in (getattr(cfg, "ccc_condition_values", ()) or ()) if str(x)),
    )
    dataset_key = getattr(cfg, "ccc_dataset_key", None)
    source_levels = tuple(str(x) for x in (getattr(cfg, "ccc_source_levels", ()) or ()) if str(x))
    target_levels = tuple(str(x) for x in (getattr(cfg, "ccc_target_levels", ()) or ()) if str(x))
    cross_tissue_mode = bool(dataset_key)

    mebocost_input_mode = str(getattr(cfg, "mebocost_input_mode", "counts")).strip().lower()
    if mebocost_input_mode not in {"counts", "lognorm"}:
        raise RuntimeError("ccc mebocost paired-rescore: mebocost_input_mode must be one of {'counts', 'lognorm'}.")
    if cross_tissue_mode and mebocost_input_mode == "counts":
        LOGGER.warning(
            "ccc mebocost paired-rescore: cross-tissue mode is running on raw count-like input (%s). "
            "If source datasets differ strongly in sequencing depth or chemistry, donor-level MEBOCOST rescoring can be biased by detection depth. "
            "Consider rerunning with --input-mode lognorm for a depth-normalized expression layer.",
            "counts_cb/counts_raw",
        )
    mebocost_layer = _effective_mebocost_layer(
        filtered_adata,
        input_mode=mebocost_input_mode,
        lognorm_target_sum=float(getattr(cfg, "mebocost_lognorm_target_sum", 1e4)),
    )

    candidate_df = _read_mebocost_candidate_events(Path(getattr(cfg, "mebocost_candidate_events")))
    candidate_df = _normalize_mebocost_candidate_events(
        candidate_df,
        display_map=display_map,
        valid_group_tokens=sorted(filtered_adata.obs[str(groupby)].astype(str).unique().tolist()),
    )
    if cross_tissue_mode:
        if not source_levels or not target_levels:
            raise RuntimeError("ccc mebocost paired-rescore: cross-tissue mode requires at least one source level and one target level.")
        source_allowed = {str(x) for x in source_levels}
        target_allowed = {str(x) for x in target_levels}
        if "source_dataset_level" in candidate_df.columns:
            candidate_df = candidate_df[candidate_df["source_dataset_level"].astype(str).isin(source_allowed)].copy()
        if "target_dataset_level" in candidate_df.columns:
            candidate_df = candidate_df[candidate_df["target_dataset_level"].astype(str).isin(target_allowed)].copy()
    candidate_df = _annotate_mebocost_commu_table(
        candidate_df,
        config_path=mebocost_config_path,
        species=str(getattr(cfg, "mebocost_organism", "human")).strip().lower(),
    )
    candidate_df = _prepare_mebocost_plot_df(
        candidate_df,
        display_map=display_map,
        valid_group_tokens=sorted(filtered_adata.obs[str(groupby)].astype(str).unique().tolist()),
    )
    candidate_df = _filter_mebocost_candidate_events(candidate_df, cfg)

    event_scores = _score_mebocost_paired_events(
        filtered_adata,
        candidate_df=candidate_df,
        groupby=str(groupby),
        pairing_key=pairing_key,
        condition_cols=condition_cols,
        dataset_key=str(dataset_key) if dataset_key else None,
        source_levels=source_levels,
        target_levels=target_levels,
        organism=str(getattr(cfg, "mebocost_organism", "human")).strip().lower(),
        config_path=mebocost_config_path,
        mebocost_api=mebocost_api,
        layer=mebocost_layer,
        score_method=str(getattr(cfg, "mebocost_score_method", "mebocost-metabolite-sensor")).strip().lower(),
        min_sender_cells=int(getattr(cfg, "mebocost_min_sender_cells", 5)),
        min_receiver_cells=int(getattr(cfg, "mebocost_min_receiver_cells", 5)),
    )
    route_scores = _summarize_mebocost_paired_routes(event_scores)
    event_missingness = _summarize_paired_missingness(
        event_scores,
        group_by=("source_token", "target_token", "source_label", "target_label", "HMDB_ID", "sensor_gene"),
        score_col="paired_commu_score",
        primary_condition_key=primary_condition_key,
    )
    route_missingness = _summarize_paired_missingness(
        route_scores,
        group_by=("source_token", "target_token", "source_label", "target_label", "route"),
        score_col="mean_paired_commu_score",
        primary_condition_key=primary_condition_key,
    )
    group_effects = _summarize_mebocost_paired_group_effects(
        event_scores,
        primary_condition_key=primary_condition_key,
        min_scored_donors_per_group=int(getattr(cfg, "mebocost_min_scored_donors_per_group", 3)),
    )

    run_namespace = _run_namespace_for_round(
        adata,
        prefix="ccc_mebocost",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    tables_root = output_dir / "tables" / run_round / "paired_rescore" / _safe_combo_token(condition_label)
    tables_root.mkdir(parents=True, exist_ok=True)
    event_scores.to_csv(tables_root / "mebocost_paired_event_scores.tsv", sep="\t", index=False)
    route_scores.to_csv(tables_root / "mebocost_paired_route_scores.tsv", sep="\t", index=False)
    event_missingness.to_csv(tables_root / "mebocost_paired_event_missingness.tsv", sep="\t", index=False)
    route_missingness.to_csv(tables_root / "mebocost_paired_route_missingness.tsv", sep="\t", index=False)
    group_effects.to_csv(tables_root / "mebocost_paired_group_effects.tsv", sep="\t", index=False)
    if group_effects.empty:
        n_scored = int(pd.to_numeric(event_scores.get("paired_commu_score", pd.Series(dtype=float)), errors="coerce").notna().sum())
        LOGGER.warning(
            "ccc mebocost paired-rescore: no group effects were produced for %s. Scored donor-level events=%d. "
            "This usually reflects donor-level sparsity or scored-donor thresholds rather than absence of biology. "
            "Inspect mebocost_paired_event_missingness.tsv and consider relaxing --min-sender-cells, --min-receiver-cells, or --min-scored-donors-per-group.",
            str(condition_label),
            n_scored,
        )
    if bool(getattr(cfg, "make_figures", False)):
        fig_rel = Path("paired_rescore") / _safe_combo_token(condition_label)
        artifacts = []
        artifacts.extend(
            plot_utils.plot_mebocost_paired_route_summary(
                route_scores,
                figdir=fig_rel,
                stem="mebocost_paired_route_summary",
                title=f"MEBOCOST paired route summary [{condition_label}]",
                top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 15),
            )
        )
        artifacts.extend(
            plot_utils.plot_mebocost_paired_route_heatmap(
                route_scores,
                figdir=fig_rel,
                stem="mebocost_paired_route_heatmap",
                title=f"MEBOCOST paired route heatmap [{condition_label}]",
                top_n_routes=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 20),
            )
        )
        artifacts.extend(
            plot_utils.plot_mebocost_paired_group_effects(
                group_effects,
                figdir=fig_rel,
                stem_prefix="mebocost_paired_group_effects",
                title_prefix="MEBOCOST paired group effects",
                top_n=min(int(getattr(cfg, "mebocost_plot_top_n", 40)), 15),
            )
        )
        plot_utils.persist_plot_artifacts(artifacts)

    settings_lines = [
        f"input_path\t{getattr(cfg, 'input_path')}",
        f"candidate_events\t{getattr(cfg, 'mebocost_candidate_events')}",
        f"groupby\t{groupby}",
        f"pairing_key\t{pairing_key}",
        f"condition_key\t{getattr(cfg, 'ccc_condition_key', None)}",
        f"condition_values\t{','.join(str(x) for x in getattr(cfg, 'ccc_condition_values', ()) or ())}",
        f"dataset_key\t{dataset_key}",
        f"source_levels\t{','.join(source_levels)}",
        f"target_levels\t{','.join(target_levels)}",
        f"organism\t{getattr(cfg, 'mebocost_organism', 'human')}",
        f"input_mode\t{mebocost_input_mode}",
        f"lognorm_target_sum\t{getattr(cfg, 'mebocost_lognorm_target_sum', 1e4) if mebocost_input_mode == 'lognorm' else ''}",
        f"expression_layer\t{mebocost_layer}",
        f"score_method\t{getattr(cfg, 'mebocost_score_method', 'mebocost-metabolite-sensor')}",
        "communication_score_formula\tsqrt(sender_metabolite_score * receiver_sensor_score)",
        f"min_sender_cells\t{int(getattr(cfg, 'mebocost_min_sender_cells', 5))}",
        f"min_receiver_cells\t{int(getattr(cfg, 'mebocost_min_receiver_cells', 5))}",
        f"min_scored_donors_per_group\t{int(getattr(cfg, 'mebocost_min_scored_donors_per_group', 3))}",
    ]
    (tables_root / "mebocost_paired_settings.tsv").write_text("\n".join(settings_lines) + "\n")

    adata.uns.setdefault("markers_and_de", {})
    ccc_block = adata.uns["markers_and_de"].setdefault("ccc", {})
    mebo_block = ccc_block.setdefault("mebocost_paired_rescore", {})
    mebo_block[str(condition_label)] = {
        "event_scores": event_scores,
        "route_scores": route_scores,
        "event_missingness": event_missingness,
        "route_missingness": route_missingness,
        "group_effects": group_effects,
        "candidate_events": candidate_df,
        "score_method": str(getattr(cfg, "mebocost_score_method", "mebocost-metabolite-sensor")).strip().lower(),
        "input_mode": mebocost_input_mode,
        "expression_layer": mebocost_layer,
        "pairing_key": pairing_key,
    }
    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_mebocost_paired")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")
    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.ccc_mebocost_paired")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
    LOGGER.info("Finished markers-and-de (ccc mebocost paired-rescore).")
    return adata


def run_composition(cfg) -> ad.AnnData:
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (composition)...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    sample_key = (
        getattr(cfg, "sample_key", None)
        or getattr(cfg, "batch_key", None)
        or adata.uns.get("batch_key")
        or "sample_id"
    )
    if str(sample_key) not in adata.obs:
        raise RuntimeError(f"composition: sample_key={sample_key!r} not found in adata.obs")

    condition_keys = list(getattr(cfg, "condition_keys", ()) or [])
    if not condition_keys:
        fallback = getattr(cfg, "condition_key", None)
        if fallback:
            condition_keys = [str(fallback)]
    if not condition_keys:
        raise RuntimeError("composition: condition_keys is required.")

    regenerate_figures = bool(getattr(cfg, "regenerate_figures", False))
    if regenerate_figures and not bool(getattr(cfg, "make_figures", True)):
        raise RuntimeError("regenerate_figures=True requires make_figures=True.")

    expanded_conditions: list[tuple[str, Optional[np.ndarray], str]] = []
    for raw_key in condition_keys:
        raw = str(raw_key).strip()
        if "@" in raw:
            parts = [p.strip() for p in raw.split("@") if p.strip()]
            if len(parts) != 2:
                raise RuntimeError(f"composition: invalid within-B condition key {raw!r}. Use 'A@B'.")
            a_key = _resolve_condition_key(adata, parts[0])
            b_key = _resolve_condition_key(adata, parts[1])
            if str(a_key) not in adata.obs or str(b_key) not in adata.obs:
                raise RuntimeError(f"composition: condition_keys not found in adata.obs: {[a_key, b_key]}")
            b_vals = _normalize_levels(adata.obs[str(b_key)])
            for b_level in sorted(pd.unique(b_vals).tolist()):
                mask = b_vals.astype(str).to_numpy() == str(b_level)
                label = f"{a_key}@{b_key}={b_level}"
                expanded_conditions.append((str(a_key), mask, str(label)))
                LOGGER.info(
                    "composition: expanded %r -> condition_key=%r within %s=%r (n_cells=%d).",
                    raw,
                    str(a_key),
                    str(b_key),
                    str(b_level),
                    int(mask.sum()),
                )
            continue
        ck = _resolve_condition_key(adata, raw)
        if str(ck) not in adata.obs:
            raise RuntimeError(f"composition: condition_keys not found in adata.obs: {ck!r}")
        expanded_conditions.append((str(ck), None, str(ck)))

    cluster_key = _resolve_active_cluster_key(adata, round_id=getattr(cfg, "round_id", None))
    covariates = tuple(getattr(cfg, "composition_covariates", ()) or ())
    min_mean_prop = float(getattr(cfg, "composition_min_mean_prop", 0.01))

    methods = [str(m).lower() for m in (getattr(cfg, "composition_methods", ()) or ())]
    if not methods:
        methods = ["sccoda", "glm", "clr", "graph"]
    primary_method = methods[0]
    alpha = float(getattr(cfg, "composition_alpha", 0.05))

    n_iterations = int(getattr(cfg, "composition_n_iterations", 10000) or 10000)
    n_warmup = int(getattr(cfg, "composition_n_warmup", max(1000, n_iterations // 10)) or max(1000, n_iterations // 10))
    run_namespace = _run_namespace_for_round(
        adata,
        prefix="da",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))

    from .de_utils import _set_blas_threads

    total_cpus = int(getattr(cfg, "n_jobs", 1))
    total_tasks = len(expanded_conditions)
    max_workers = min(int(total_cpus), max(1, total_tasks))
    LOGGER.info(
        "composition: parallel run (tasks=%d, workers=%d, total_cpus=%d).",
        int(total_tasks),
        int(max_workers),
        int(total_cpus),
    )
    _set_blas_threads(1, force=True)

    def _run_condition(condition_key: str, restrict_mask: Optional[np.ndarray], condition_label: str) -> dict:
        try:
            counts, metadata = prepare_counts_and_metadata(
                adata,
                cluster_key=cluster_key,
                sample_key=str(sample_key),
                condition_key=str(condition_key),
                covariates=covariates,
                restrict_mask=restrict_mask,
            )
            try:
                _validate_min_samples_per_level(
                    metadata,
                    condition_key=str(condition_key),
                    min_samples=_MIN_SAMPLES_PER_LEVEL_COMPOSITION,
                )
            except Exception as e:
                return {
                    "status": "skip",
                    "reason": str(e),
                    "condition_key": str(condition_key),
                    "condition_label": str(condition_label),
                }

            reference = str(getattr(cfg, "composition_reference", "most_stable"))
            if reference.lower() == "most_stable":
                reference = _choose_reference_most_stable(counts, min_mean_prop=min_mean_prop)

            def _run_method(tag: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
                graph_meta = None
                adata_sub = adata
                if restrict_mask is not None:
                    adata_sub = adata[restrict_mask].copy()
                if tag == "sccoda":
                    with _SCCODA_LOCK:
                        res = run_sccoda_model(
                            adata_sub,
                            cluster_key=cluster_key,
                            sample_key=str(sample_key),
                            condition_key=str(condition_key),
                            covariates=covariates,
                            reference_cell_type=reference,
                            fdr=alpha,
                            num_samples=n_iterations,
                            num_warmup=n_warmup,
                        )
                elif tag == "glm":
                    res = run_glm_composition(
                        counts,
                        metadata,
                        condition_key=str(condition_key),
                        covariates=covariates,
                        reference_level=None,
                    )
                elif tag == "clr":
                    res = run_clr_mannwhitney(
                        counts,
                        metadata,
                        condition_key=str(condition_key),
                    )
                    if not res.empty:
                        res = res.assign(effect=res["log2fc_test_vs_ref"])
                elif tag == "graph":
                    res, graph_meta = run_graph_da(
                        adata_sub,
                        cluster_key=cluster_key,
                        sample_key=str(sample_key),
                        condition_key=str(condition_key),
                        covariates=covariates,
                        embedding_key="X_integrated",
                        n_seeds=int(getattr(cfg, "composition_graph_n_seeds", 2000)),
                        k_ref=int(getattr(cfg, "composition_graph_k_ref", 30)),
                        max_k=int(getattr(cfg, "composition_graph_max_k", 200)),
                        min_size=int(getattr(cfg, "composition_graph_min_size", 20)),
                        random_state=int(getattr(cfg, "composition_graph_random_state", 42)),
                        min_nonzero_samples_per_level=int(
                            getattr(cfg, "composition_graph_min_nonzero_samples_per_level", 3)
                        ),
                        n_permutations=int(getattr(cfg, "composition_graph_n_permutations", 0)),
                        effect_shrink_k=float(getattr(cfg, "composition_graph_effect_shrink_k", 10.0)),
                    )
                    if graph_meta is not None and not graph_meta.empty and isinstance(res, pd.DataFrame) and not res.empty:
                        if "cluster" in res.columns:
                            graph_meta = graph_meta.set_index("neighborhood")
                            res = res.merge(graph_meta, left_on="cluster", right_index=True, how="left")
                else:
                    raise RuntimeError(f"composition: unsupported method={tag!r}")

                res = _standardize_composition_results(
                    res,
                    backend=tag,
                    condition_key=str(condition_key),
                )
                return res, graph_meta

            results_by_method: dict[str, pd.DataFrame] = {}
            graph_meta_global: Optional[pd.DataFrame] = None
            for method in methods:
                res_df, meta_df = _run_method(method)
                results_by_method[method] = res_df
                if method == "graph" and meta_df is not None and not meta_df.empty:
                    graph_meta_global = meta_df

            consensus = _build_composition_consensus_summary(
                results_by_method,
                alpha=alpha,
                condition_key=str(condition_key),
            )
            return {
                "status": "ok",
                "condition_key": str(condition_key),
                "condition_label": str(condition_label),
                "counts": counts,
                "metadata": metadata,
                "results_by_method": results_by_method,
                "consensus": consensus,
                "graph_meta_global": graph_meta_global,
                "reference": reference,
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "status": "error",
                "reason": str(e),
                "traceback": tb,
                "condition_key": str(condition_key),
                "condition_label": str(condition_label),
            }

    def _write_outputs(
        payload: dict,
        *,
        write_tables: bool = True,
        write_settings: bool = True,
        write_figures: bool = True,
    ) -> None:
        condition_key = str(payload["condition_key"])
        condition_label = str(payload["condition_label"])
        counts = payload["counts"]
        metadata = payload["metadata"]
        results_by_method = payload["results_by_method"]
        consensus = payload["consensus"]
        graph_meta_global = payload.get("graph_meta_global")
        reference = str(payload["reference"])

        adata.uns.setdefault("markers_and_de", {})
        comp_block = adata.uns["markers_and_de"].setdefault("composition", {})
        comp_block.setdefault("schema", "multi")
        runs = comp_block.setdefault("runs", {})
        runs[str(condition_label)] = {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "methods": list(methods),
            "primary_method": primary_method,
            "reference": reference,
            "round_id": getattr(cfg, "round_id", None),
            "cluster_key": cluster_key,
            "sample_key": str(sample_key),
            "condition_key": str(condition_key),
            "condition_label": str(condition_label),
            "covariates": list(covariates),
            "alpha": alpha,
            "min_mean_prop": min_mean_prop,
            "n_samples": int(counts.shape[0]),
            "n_clusters": int(counts.shape[1]),
            "results_by_method": results_by_method,
            "consensus": consensus,
            "graph_meta_global": graph_meta_global,
            "counts": counts,
            "metadata": metadata,
        }

        cond_tag = _safe_combo_token(str(condition_label))
        results_dir = output_dir / "tables" / run_round / cond_tag
        results_dir.mkdir(parents=True, exist_ok=True)
        fig_subdir = Path(cond_tag)

        def _cnnize_df(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or getattr(df, "empty", False):
                return df
            df2 = df.copy()
            if "cluster" in df2.columns:
                df2["cluster"] = df2["cluster"].astype(str).map(
                    lambda x: io_utils._cnn_label_for_group(x, None)
                )
            if df2.index.name == "cluster":
                df2.index = pd.Index(
                    [io_utils._cnn_label_for_group(str(x), None) for x in df2.index],
                    name="cluster",
                )
            return df2

        def _build_graphda_diagnostics_df() -> pd.DataFrame:
            gdf = results_by_method.get("graph", pd.DataFrame())
            if gdf is None or gdf.empty:
                return pd.DataFrame()
            d = gdf.copy()
            cluster_col = "cluster_label" if "cluster_label" in d.columns else "cluster"
            if cluster_col not in d.columns:
                return pd.DataFrame()
            d[cluster_col] = d[cluster_col].astype(str)
            d["pval"] = pd.to_numeric(d.get("pval", np.nan), errors="coerce")
            d["fdr"] = pd.to_numeric(d.get("fdr", np.nan), errors="coerce")
            d["effect"] = pd.to_numeric(d.get("effect", np.nan), errors="coerce")

            fdr_ok = d["fdr"].notna()
            d["is_sig"] = np.where(fdr_ok, d["fdr"] <= float(alpha), d["pval"] <= float(alpha))
            agg = (
                d.groupby(cluster_col, observed=False)
                .agg(
                    n_test_rows=("is_sig", "size"),
                    n_sig_fdr=("is_sig", "sum"),
                    min_pval=("pval", "min"),
                    min_fdr=("fdr", "min"),
                    median_pval=("pval", "median"),
                    median_fdr=("fdr", "median"),
                    median_abs_effect=("effect", lambda x: np.nanmedian(np.abs(pd.to_numeric(x, errors="coerce")))),
                )
                .reset_index()
                .rename(columns={cluster_col: "cluster"})
            )

            if graph_meta_global is not None and not graph_meta_global.empty:
                m = graph_meta_global.copy()
                if "cluster_label" in m.columns:
                    m["cluster"] = m["cluster_label"].astype(str)
                elif "cluster" in m.columns:
                    m["cluster"] = m["cluster"].astype(str)
                else:
                    m["cluster"] = "NA"
                tested = pd.to_numeric(m.get("tested", False), errors="coerce").fillna(0.0).astype(bool)
                m["tested"] = tested
                m_agg = (
                    m.groupby("cluster", observed=False)
                    .agg(
                        n_total_neighborhoods=("cluster", "size"),
                        n_tested_neighborhoods=("tested", "sum"),
                    )
                    .reset_index()
                )
                agg = agg.merge(m_agg, on="cluster", how="outer")

            for col in ("n_test_rows", "n_sig_fdr", "n_total_neighborhoods", "n_tested_neighborhoods"):
                if col in agg.columns:
                    agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0).astype(int)
            if "n_test_rows" in agg.columns:
                denom = agg["n_test_rows"].replace(0, np.nan)
                agg["frac_sig_rows"] = (agg["n_sig_fdr"] / denom).fillna(0.0)
            if "n_total_neighborhoods" in agg.columns and "n_tested_neighborhoods" in agg.columns:
                denom = agg["n_total_neighborhoods"].replace(0, np.nan)
                agg["frac_tested_neighborhoods"] = (agg["n_tested_neighborhoods"] / denom).fillna(0.0)

            return agg.sort_values("cluster").reset_index(drop=True)

        graph_diag_df = _build_graphda_diagnostics_df()

        if write_tables:
            for method, df in results_by_method.items():
                if isinstance(df, pd.DataFrame):
                    df_out = _cnnize_df(df)
                    df_out.to_csv(results_dir / f"composition_global_{method}.tsv", sep="	")
            if isinstance(consensus, pd.DataFrame) and not consensus.empty:
                consensus_out = _cnnize_df(consensus)
                consensus_out.to_csv(results_dir / "composition_consensus.tsv", sep="	", index=False)
            if graph_meta_global is not None and not graph_meta_global.empty:
                graph_out = _cnnize_df(graph_meta_global)
                graph_out.to_csv(results_dir / "composition_graph_neighborhoods.tsv", sep="\t", index=False)
            if not graph_diag_df.empty:
                graph_diag_out = _cnnize_df(graph_diag_df)
                graph_diag_out.to_csv(results_dir / "graphda_diagnostics.tsv", sep="\t", index=False)

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            if graph_meta_global is not None and not graph_meta_global.empty:
                try:
                    artifacts = []
                    artifacts.extend(
                        plot_utils.plot_graphda_summaries(
                            results_by_method.get("graph", pd.DataFrame()),
                            graph_meta_global,
                            fig_subdir,
                            alpha=alpha,
                            all_clusters=counts.columns.astype(str).tolist(),
                        )
                    )
                    if not graph_diag_df.empty:
                        artifacts.extend(
                            plot_utils.plot_graphda_diagnostics(
                                results_by_method.get("graph", pd.DataFrame()),
                                graph_diag_df,
                                fig_subdir,
                                alpha=alpha,
                            )
                        )
                    plot_utils.persist_plot_artifacts(artifacts)
                except Exception:
                    LOGGER.exception("composition: failed to plot GraphDA summary")

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            for method in methods:
                df = results_by_method.get(method, pd.DataFrame())
                if df is None or df.empty:
                    continue
                try:
                    artifacts = []
                    if method in ("glm", "clr"):
                        artifacts.extend(plot_utils.plot_composition_volcano(method, df, fig_subdir, alpha=alpha))
                    if method == "sccoda":
                        if "cluster" in df.columns:
                            labels = df["cluster"].astype(str)
                        else:
                            labels = df.index.astype(str)
                        colors = _resolve_cluster_colors(
                            adata,
                            cluster_key=cluster_key,
                            labels=labels,
                            round_id=getattr(cfg, "round_id", None),
                        )
                        color_map = {str(label): color for label, color in zip(labels, colors)}
                        artifacts.extend(plot_utils.plot_sccoda_effects_top(df, color_map, fig_subdir, alpha=alpha))
                    plot_utils.persist_plot_artifacts(artifacts)
                except Exception:
                    LOGGER.exception("composition: method plot failed for %s", method)

        if write_settings:
            _write_settings(
                results_dir,
                "composition_settings.txt",
                [
                    f"methods={list(methods)}",
                    f"primary_method={primary_method}",
                    f"reference={reference}",
                    f"round_id={getattr(cfg, 'round_id', None)}",
                    f"cluster_key={cluster_key}",
                    f"sample_key={sample_key}",
                    f"condition_key={condition_key}",
                    f"covariates={list(covariates)}",
                    f"alpha={alpha}",
                    f"min_mean_prop={min_mean_prop}",
                    f"graph_n_seeds={getattr(cfg, 'composition_graph_n_seeds', None)}",
                    f"graph_k_ref={getattr(cfg, 'composition_graph_k_ref', None)}",
                    f"graph_max_k={getattr(cfg, 'composition_graph_max_k', None)}",
                    f"graph_min_size={getattr(cfg, 'composition_graph_min_size', None)}",
                    f"graph_random_state={getattr(cfg, 'composition_graph_random_state', None)}",
                    f"graph_min_nonzero_samples_per_level={getattr(cfg, 'composition_graph_min_nonzero_samples_per_level', None)}",
                    f"graph_n_permutations_deprecated={getattr(cfg, 'composition_graph_n_permutations', None)}",
                    f"graph_effect_shrink_k={getattr(cfg, 'composition_graph_effect_shrink_k', None)}",
                    f"glm_min_samples_per_level={_MIN_GLM_SAMPLES_PER_LEVEL}",
                    f"glm_min_levels=3",
                ],
            )

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            try:
                def _effect_sig_mask(df_eff: pd.DataFrame, labels_eff: pd.Index, alpha_eff: float) -> np.ndarray:
                    if df_eff is None or df_eff.empty:
                        return np.zeros(len(labels_eff), dtype=bool)
                    dtmp = df_eff.copy()
                    if "cluster" in dtmp.columns:
                        dtmp["cluster"] = dtmp["cluster"].astype(str)
                    else:
                        dtmp["cluster"] = dtmp.index.astype(str)
                    grp = dtmp.groupby("cluster", dropna=False)
                    labels_series = pd.Series(labels_eff.astype(str), index=np.arange(len(labels_eff)))
                    if "fdr" in dtmp.columns:
                        s = grp["fdr"].min()
                        mapped = pd.to_numeric(labels_series.map(s), errors="coerce")
                        return mapped.le(float(alpha_eff)).fillna(False).to_numpy()
                    if "pval" in dtmp.columns:
                        s = grp["pval"].min()
                        mapped = pd.to_numeric(labels_series.map(s), errors="coerce")
                        return mapped.le(float(alpha_eff)).fillna(False).to_numpy()
                    if "inclusion_prob" in dtmp.columns:
                        s = grp["inclusion_prob"].max()
                        mapped = pd.to_numeric(labels_series.map(s), errors="coerce")
                        return mapped.ge(float(1.0 - alpha_eff)).fillna(False).to_numpy()
                    if "Inclusion probability" in dtmp.columns:
                        s = grp["Inclusion probability"].max()
                        mapped = pd.to_numeric(labels_series.map(s), errors="coerce")
                        return mapped.ge(float(1.0 - alpha_eff)).fillna(False).to_numpy()
                    return np.zeros(len(labels_eff), dtype=bool)

                global_df = results_by_method.get(primary_method)
                if isinstance(global_df, pd.DataFrame) and not global_df.empty:
                    plotted = False
                    plotted_name = "composition_effects_global"
                    if primary_method == "sccoda":
                        if "Expected Sample log2-fold change" in global_df.columns:
                            vals = pd.to_numeric(global_df["Expected Sample log2-fold change"], errors="coerce")
                        elif "Expected Sample" in global_df.columns:
                            vals = pd.to_numeric(global_df["Expected Sample"], errors="coerce")
                        elif "Final Parameter" in global_df.columns:
                            vals = pd.to_numeric(global_df["Final Parameter"], errors="coerce")
                        else:
                            vals = pd.Series(dtype=float)
                        if "cluster" in global_df.columns:
                            labels = global_df["cluster"].astype(str)
                        else:
                            labels = global_df.index.astype(str)
                        if not vals.isna().all():
                            labels = pd.Index(labels).astype(str)
                            colors = _resolve_cluster_colors(
                                adata,
                                cluster_key=cluster_key,
                                labels=labels,
                                round_id=getattr(cfg, "round_id", None),
                            )
                            artifacts = plot_utils.plot_composition_effects_global(
                                vals.to_numpy(),
                                labels,
                                colors,
                                fig_subdir,
                                plot_name="composition_effects_global_sccoda",
                                title="Composition effects (global, scCODA)",
                                sig_mask=_effect_sig_mask(global_df, labels, alpha),
                                alpha=alpha,
                            )
                            plot_utils.persist_plot_artifacts(artifacts)
                            plotted_name = "composition_effects_global_sccoda"
                            plotted = True
                    elif "effect" in global_df.columns:
                        vals = pd.to_numeric(global_df["effect"], errors="coerce")
                        if "cluster" in global_df.columns:
                            labels = global_df["cluster"].astype(str)
                        else:
                            labels = global_df.index.astype(str)
                        if not vals.isna().all():
                            labels = pd.Index(labels).astype(str)
                            colors = _resolve_cluster_colors(
                                adata,
                                cluster_key=cluster_key,
                                labels=labels,
                                round_id=getattr(cfg, "round_id", None),
                            )
                            artifacts = plot_utils.plot_composition_effects_global(
                                vals.to_numpy(),
                                labels,
                                colors,
                                fig_subdir,
                                plot_name="composition_effects_global",
                                title="Composition effects (global)",
                                sig_mask=_effect_sig_mask(global_df, labels, alpha),
                                alpha=alpha,
                            )
                            plot_utils.persist_plot_artifacts(artifacts)
                            plotted = True
                    elif primary_method == "glm" and "coef" in global_df.columns:
                        vals = global_df.groupby("cluster")["coef"].mean()
                        labels = vals.index.astype(str)
                        if not vals.isna().all():
                            labels = pd.Index(labels).astype(str)
                            colors = _resolve_cluster_colors(
                                adata,
                                cluster_key=cluster_key,
                                labels=labels,
                                round_id=getattr(cfg, "round_id", None),
                            )
                            artifacts = plot_utils.plot_composition_effects_global(
                                vals.values,
                                labels,
                                colors,
                                fig_subdir,
                                plot_name="composition_effects_global",
                                title="Composition effects (global)",
                                sig_mask=_effect_sig_mask(global_df, labels, alpha),
                                alpha=alpha,
                            )
                            plot_utils.persist_plot_artifacts(artifacts)
                            plotted = True
                    if plotted:
                        LOGGER.info("Saved plot: %s/%s", fig_subdir, plotted_name)

                clr_df = results_by_method.get("clr")
                if isinstance(clr_df, pd.DataFrame) and not clr_df.empty and "effect" in clr_df.columns:
                    clr_vals = pd.to_numeric(clr_df["effect"], errors="coerce")
                    if "cluster" in clr_df.columns:
                        clr_labels = clr_df["cluster"].astype(str)
                    else:
                        clr_labels = clr_df.index.astype(str)
                    if not clr_vals.isna().all():
                        clr_labels = pd.Index(clr_labels).astype(str)
                        clr_colors = _resolve_cluster_colors(
                            adata,
                            cluster_key=cluster_key,
                            labels=clr_labels,
                            round_id=getattr(cfg, "round_id", None),
                        )
                        artifacts = plot_utils.plot_composition_effects_global(
                            clr_vals.to_numpy(),
                            clr_labels,
                            clr_colors,
                            fig_subdir,
                            plot_name="composition_effects_global_clr",
                            title="Composition effects (global, CLR)",
                            sig_mask=_effect_sig_mask(clr_df, clr_labels, alpha),
                            alpha=alpha,
                        )
                        plot_utils.persist_plot_artifacts(artifacts)
                        LOGGER.info("Saved plot: %s/%s", fig_subdir, "composition_effects_global_clr")
            except Exception:
                LOGGER.exception("composition: failed to generate plots")

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            try:
                totals = counts.sum(axis=1).replace(0, np.nan)
                props = counts.div(totals, axis=0)
                if props.empty:
                    LOGGER.warning("composition: no proportions available for plotting")
                    return
                props_plot = props.copy()
                props_plot.columns = props_plot.columns.astype(str)
                cluster_order = props_plot.mean(axis=0).sort_values(ascending=False).index.tolist()
                colors = _resolve_cluster_colors(
                    adata,
                    cluster_key=cluster_key,
                    labels=cluster_order,
                    round_id=getattr(cfg, "round_id", None),
                )
                artifacts = plot_utils.plot_composition_stacks(
                    counts,
                    metadata,
                    condition_key=str(condition_key),
                    cluster_order=cluster_order,
                    colors=colors,
                    figdir=fig_subdir,
                    consensus=consensus if isinstance(consensus, pd.DataFrame) else None,
                    alpha=float(alpha),
                )
                plot_utils.persist_plot_artifacts(artifacts)
            except Exception:
                LOGGER.exception("composition: failed to plot composition stacks")

    def _load_payload_from_store(condition_label: str) -> dict:
        comp_block = adata.uns.get("markers_and_de", {}).get("composition", {})
        runs = comp_block.get("runs", {})
        payload = runs.get(str(condition_label))
        if not isinstance(payload, dict):
            raise RuntimeError(f"regenerate_figures: missing composition run for {condition_label!r}.")
        required = ("results_by_method", "consensus", "counts", "metadata")
        for key in required:
            if payload.get(key, None) is None:
                raise RuntimeError(
                    "regenerate_figures: missing %r for condition_label=%r in adata.uns['markers_and_de']['composition']."
                    % (str(key), str(condition_label))
                )
        return {
            "status": "ok",
            "condition_key": payload.get("condition_key", ""),
            "condition_label": condition_label,
            "counts": payload.get("counts"),
            "metadata": payload.get("metadata"),
            "results_by_method": payload.get("results_by_method", {}),
            "consensus": payload.get("consensus"),
            "graph_meta_global": payload.get("graph_meta_global"),
            "reference": payload.get("reference", ""),
        }

    if regenerate_figures:
        for _, _, label in expanded_conditions:
            payload = _load_payload_from_store(label)
            _write_outputs(payload, write_tables=False, write_settings=False, write_figures=True)
    else:
        done = 0
        t0 = time.perf_counter()
        if expanded_conditions and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

            heartbeat_s = 60.0
            next_heartbeat = time.perf_counter() + heartbeat_s
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(_run_condition, ck, mask, label): (ck, label)
                    for ck, mask, label in expanded_conditions
                }
                pending = set(futs.keys())
                start_by_fut = {f: time.perf_counter() for f in pending}
                while pending:
                    now = time.perf_counter()
                    done_set, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    if not done_set:
                        if now >= next_heartbeat:
                            LOGGER.info(
                                "composition: heartbeat (done=%d/%d, pending=%d).",
                                int(done),
                                int(total_tasks),
                                int(len(pending)),
                            )
                            next_heartbeat = now + heartbeat_s
                        continue
                    for fut in done_set:
                        ck, label = futs.get(fut, ("", ""))
                        t_elapsed = now - start_by_fut.get(fut, now)
                        try:
                            payload = fut.result()
                        except Exception:
                            LOGGER.exception("composition: task failed (condition_label=%s).", str(label))
                            done += 1
                            continue
                        status = payload.get("status")
                        if status == "skip":
                            LOGGER.warning(
                                "composition: skipping condition_key=%r (%s)",
                                payload.get("condition_key"),
                                payload.get("reason"),
                            )
                            done += 1
                        elif status == "error":
                            LOGGER.error(
                                "composition: failed condition_key=%r (%s)",
                                payload.get("condition_key"),
                                payload.get("reason"),
                            )
                            tb = payload.get("traceback")
                            if tb:
                                LOGGER.error("composition: traceback\n%s", tb)
                            done += 1
                        else:
                            _write_outputs(payload)
                            done += 1
                        elapsed = time.perf_counter() - t0
                        eta_s = (elapsed / max(1, done)) * (total_tasks - done)
                        LOGGER.info(
                            "composition: progress %d/%d (elapsed=%.1fs, eta=%.1fs).",
                            int(done),
                            int(total_tasks),
                            elapsed,
                            eta_s,
                        )
                        LOGGER.info(
                            "composition: task finished (condition_key=%s, condition_label=%s, task_s=%.1f).",
                            str(ck),
                            str(label),
                            float(t_elapsed),
                        )
        else:
            for ck, mask, label in expanded_conditions:
                payload = _run_condition(ck, mask, label)
                if payload.get("status") == "skip":
                    LOGGER.warning(
                        "composition: skipping condition_key=%r (%s)",
                        payload.get("condition_key"),
                        payload.get("reason"),
                    )
                    continue
                if payload.get("status") == "error":
                    LOGGER.error(
                        "composition: failed condition_key=%r (%s)",
                        payload.get("condition_key"),
                        payload.get("reason"),
                    )
                    tb = payload.get("traceback")
                    if tb:
                        LOGGER.error("composition: traceback\n%s", tb)
                    continue
                _write_outputs(payload)

    if regenerate_figures:
        LOGGER.info("composition: regenerate_figures=True; skipping dataset save.")
    else:
        out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.da")) + ".zarr")
        LOGGER.info("Saving dataset → %s", out_zarr)
        io_utils.save_dataset(adata, out_zarr, fmt="zarr")

        if bool(getattr(cfg, "save_h5ad", False)):
            out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.da")) + ".h5ad")
            LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
            io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (composition).")
    return adata


def run_within_cluster(cfg) -> ad.AnnData:
    """
    Within-cluster contrasts orchestrator.

    Runs (depending on cfg.run):
      - cell-level within-cluster contrasts via contrast_conditional_markers
        if cfg.run in {"cell","both"}  (NO pseudobulk guard)
      - pseudobulk within-cluster DE (DESeq2) if cfg.run in {"pseudobulk","both"}
        and guard passes (>= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK unique samples)

    Always:
      - saves dataset
      - writes only the tables that correspond to what actually ran
      - plots only what actually ran
      - records provenance without referencing undefined locals
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers-and-de (within-cluster)...")
    if not bool(getattr(cfg, "prune_uns_de", True)):
        LOGGER.warning(
            "prune_uns_de is disabled; adata.uns may become very large. "
            "Use --prune-uns-de to keep only summaries and top genes."
        )

    # ----------------------------
    # I/O + figure setup
    # ----------------------------
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    run_namespace = _run_namespace_for_round(
        adata,
        prefix="de",
        round_id=getattr(cfg, "round_id", None),
    )
    run_round = str(plot_utils.get_run_subdir(run_namespace))
    de_cell_dir = results_dir / "tables" / run_round / "cell_based"
    de_pb_dir = results_dir / "tables" / run_round / "pseudobulk_based"

    # ----------------------------
    # Resolve stable groupby + display labels
    # ----------------------------
    groupby, display_map = _resolve_stable_groupby_and_display_map(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        label_source=str(getattr(cfg, "label_source", "pretty")),
    )

    sample_key = (
        getattr(cfg, "sample_key", None)
        or getattr(cfg, "batch_key", None)
        or adata.uns.get("batch_key")
    )
    if sample_key is None:
        raise RuntimeError("within-cluster: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")
    if str(sample_key) not in adata.obs:
        raise RuntimeError(f"within-cluster: sample_key={sample_key!r} not found in adata.obs")

    # ----------------------------
    # Condition keys (required)
    # ----------------------------
    condition_keys = list(getattr(cfg, "condition_keys", ())) or []
    if not condition_keys:
        ck = getattr(cfg, "condition_key", None)
        if ck:
            condition_keys = [str(ck)]
    if not condition_keys:
        raise RuntimeError("within-cluster: condition_key or condition_keys is required.")

    condition_specs: list[tuple[str, Optional[tuple[str, str]], Optional[tuple[str, str]]]] = []
    within_by_key: dict[str, Optional[tuple[str, str]]] = {}
    interaction_by_key: dict[str, Optional[tuple[str, str]]] = {}
    ref_by_key: dict[str, Optional[str]] = {}
    ref_by_factor: dict[str, Optional[str]] = {}
    for k in condition_keys:
        ck, within, interaction = _parse_condition_key_spec(adata, k)
        if ck in within_by_key and within_by_key[ck] != within:
            raise RuntimeError(f"within-cluster: conflicting condition_key specs for {ck!r}")
        if ck in interaction_by_key and interaction_by_key[ck] != interaction:
            raise RuntimeError(f"within-cluster: conflicting condition_key specs for {ck!r}")
        within_by_key[ck] = within
        interaction_by_key[ck] = interaction
        condition_specs.append((ck, within, interaction))
        if interaction:
            a_key, b_key = interaction
            ref_by_factor.setdefault(str(a_key), _global_ref_level(adata, str(a_key)))
            ref_by_factor.setdefault(str(b_key), _global_ref_level(adata, str(b_key)))
        elif within:
            a_key, _ = within
            ref_by_factor.setdefault(str(a_key), _global_ref_level(adata, str(a_key)))
        else:
            ref_by_key.setdefault(str(ck), _global_ref_level(adata, str(ck)))
    condition_keys = [ck for ck, _, _ in condition_specs]
    for k in condition_keys:
        if interaction_by_key.get(k):
            continue
        try:
            if k in adata.obs and not pd.api.types.is_categorical_dtype(adata.obs[k]):
                adata.obs[k] = adata.obs[k].astype("category")
        except Exception:
            pass

    # Optional explicit contrasts (e.g. ["M_vs_F"])
    condition_contrasts = list(getattr(cfg, "condition_contrasts", [])) or None

    LOGGER.info(
        "within-cluster: groupby=%r, condition_keys=%r, sample_key=%r, contrasts=%r",
        str(groupby),
        condition_keys,
        str(sample_key),
        condition_contrasts,
    )

    regenerate_figures = bool(getattr(cfg, "regenerate_figures", False))
    if regenerate_figures and not bool(getattr(cfg, "make_figures", True)):
        raise RuntimeError("regenerate_figures=True requires make_figures=True.")

    gene_filter = tuple(getattr(cfg, "gene_filter", ()) or ())
    gene_var_mask: Optional[np.ndarray] = None
    gene_names_for_de: Optional[list[str]] = None
    gene_filter_info: Optional[dict[str, object]] = None
    if not regenerate_figures and gene_filter:
        gene_var_mask, gene_filter_info = _apply_gene_filters_to_var_names(
            adata,
            gene_filter=gene_filter,
            resource_name="DE",
        )
        if int(gene_filter_info["n_genes_retained"]) == 0:
            raise RuntimeError("within-cluster: gene_filter removed all genes; aborting DE.")
        gene_names_for_de = adata.var_names[gene_var_mask].astype(str).tolist()

    # ----------------------------
    # Mode decoding
    # ----------------------------
    mode = str(getattr(cfg, "run", "both")).lower()
    run_cell_requested = mode in ("cell", "both")
    run_pb_requested = mode in ("pseudobulk", "both")

    # ----------------------------
    # Predefine locals (avoid NameErrors)
    # ----------------------------
    positive_only = bool(getattr(cfg, "positive_only", True))
    layer_candidates = list(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw")))
    counts_layer_used: Optional[str] = None

    n_samples_total = _n_unique_samples(adata, str(sample_key))
    pseudobulk_enabled = n_samples_total >= _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK

    all_groups = pd.Index(pd.unique(adata.obs[str(groupby)].astype(str))).sort_values()
    target_groups_cfg = [str(x) for x in (getattr(cfg, "target_groups", ()) or ())]
    target_groups_cfg = [x for x in target_groups_cfg if x]
    if target_groups_cfg:
        target_set = set(target_groups_cfg)
        selected = [str(g) for g in all_groups.astype(str).tolist() if str(g) in target_set]
        missing = sorted(target_set.difference(set(all_groups.astype(str).tolist())))
        if missing:
            LOGGER.warning(
                "within-cluster: target_groups not found in %r and will be ignored: %s",
                str(groupby),
                ", ".join(missing),
            )
        if not selected:
            raise RuntimeError(
                f"within-cluster: none of target_groups matched {groupby!r}; aborting."
            )
        selected_groups = pd.Index(selected)
    else:
        selected_groups = pd.Index(all_groups.astype(str).tolist())
    LOGGER.info(
        "within-cluster: target group selection (%d/%d groups).",
        int(len(selected_groups)),
        int(len(all_groups)),
    )

    pb_spec: Optional[PseudobulkSpec] = None
    pb_opts: Optional[PseudobulkDEOptions] = None

    store_key = str(getattr(cfg, "store_key", "scomnom_de"))

    if regenerate_figures:
        block = adata.uns.get(store_key, {})
        if run_cell_requested:
            cc_multi = block.get("contrast_conditional_multi", {})
            if not isinstance(cc_multi, dict) or not cc_multi:
                raise RuntimeError(f"regenerate_figures: contrast_conditional_multi missing in adata.uns[{store_key!r}].")
            for condition_key in condition_keys:
                contrast_key = str(getattr(cfg, "contrast_key", None) or condition_key)
                if str(contrast_key) not in cc_multi:
                    raise RuntimeError(
                        "regenerate_figures: contrast_key=%r missing in adata.uns[%r]['contrast_conditional_multi']."
                        % (str(contrast_key), str(store_key))
                    )
        if run_pb_requested:
            pb_block = block.get("pseudobulk_condition_within_group_multi", {})
            if not pb_block:
                pb_block = block.get("pseudobulk_condition_within_group", {})
            if not isinstance(pb_block, dict) or not pb_block:
                raise RuntimeError(
                    f"regenerate_figures: pseudobulk_condition_within_group missing in adata.uns[{store_key!r}]."
                )
            for condition_key in condition_keys:
                has_key = False
                for payload in pb_block.values():
                    if not isinstance(payload, dict):
                        continue
                    if str(payload.get("condition_key", "")) == str(condition_key):
                        has_key = True
                        break
                if not has_key:
                    raise RuntimeError(
                        "regenerate_figures: no pseudobulk entries for condition_key=%r in adata.uns[%r]."
                        % (str(condition_key), str(store_key))
                    )

    # ----------------------------
    # 1) Pseudobulk within-cluster DE (guarded)
    # ----------------------------
    ran_pseudobulk = False
    if run_pb_requested and not regenerate_figures:
        if not pseudobulk_enabled:
            LOGGER.warning(
                "within-cluster: pseudobulk requested but disabled: only %d unique samples in %r (< %d).",
                n_samples_total,
                str(sample_key),
                _MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK,
            )
        else:
            counts_layer_used = _choose_counts_layer(
                adata,
                candidates=layer_candidates,
                allow_X_counts=bool(getattr(cfg, "allow_X_counts", True)),
            )

            pb_spec = PseudobulkSpec(
                sample_key=str(sample_key),
                counts_layer=counts_layer_used,  # None -> uses adata.X
            )

            pb_opts = PseudobulkDEOptions(
                min_cells_per_sample_group=int(getattr(cfg, "min_cells_condition", 20)),
                min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
                alpha=float(getattr(cfg, "alpha", 0.05)),
                shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
                min_total_counts=int(getattr(cfg, "pb_min_total_counts", 100)),
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
                min_counts_per_lib=int(getattr(cfg, "pb_min_counts_per_lib", 5)),
                min_lib_pct=float(getattr(cfg, "pb_min_lib_pct", 0.0)),
                covariates=tuple(getattr(cfg, "pb_covariates", ())),
            )

            groups = selected_groups

            from .de_utils import (
                de_condition_within_group_pseudobulk_multi,
                de_condition_within_group_pseudobulk_interaction,
                _normalize_pair,
                _select_pairs,
            )

            tasks: list[dict[str, str]] = []
            tasks_by_key: dict[str, int] = {}
            groups_with_tasks_by_key: dict[str, set[str]] = {}

            interaction_supported = pydeseq2_supports_interaction_by_name()
            interaction_warned = False
            for cond_key in condition_keys:
                interaction_parts = interaction_by_key.get(str(cond_key))
                if interaction_parts is None and str(cond_key) not in adata.obs:
                    raise RuntimeError(f"within-cluster: condition_key={cond_key!r} not found in adata.obs")

                if interaction_parts and not interaction_supported:
                    if not interaction_warned:
                        LOGGER.warning(
                            "within-cluster: interaction requested but PyDESeq2 does not support interaction "
                            "contrasts by name in this environment; using contrast-vector fallback."
                        )
                        interaction_warned = True

                LOGGER.info(
                    "within-cluster: pseudobulk DE across %d groups (groupby=%r) for condition_key=%r",
                    len(groups),
                    str(groupby),
                    str(cond_key),
                )

                cond_norm = None
                if interaction_parts is None:
                    cond_norm = adata.obs[str(cond_key)].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                within_parts = within_by_key.get(str(cond_key))
                if within_parts and condition_contrasts:
                    LOGGER.info(
                        "within-cluster: condition_key=%r uses A@B shorthand; ignoring explicit contrasts.",
                        str(cond_key),
                    )
                for g in groups:
                    g_mask = adata.obs[str(groupby)].astype(str).to_numpy() == str(g)
                    if interaction_parts:
                        ref_a = ref_by_factor.get(str(interaction_parts[0]))
                        ref_b = ref_by_factor.get(str(interaction_parts[1]))
                        tasks.append(
                            {
                                "kind": "interaction",
                                "cond_key": str(cond_key),
                                "group": str(g),
                                "a_key": str(interaction_parts[0]),
                                "b_key": str(interaction_parts[1]),
                                "ref_a": str(ref_a) if ref_a is not None else "",
                                "ref_b": str(ref_b) if ref_b is not None else "",
                            }
                        )
                        tasks_by_key[str(cond_key)] = tasks_by_key.get(str(cond_key), 0) + 1
                        groups_with_tasks_by_key.setdefault(str(cond_key), set()).add(str(g))
                        continue
                    if within_parts:
                        A_key, B_key = within_parts
                        pairs = _pairs_within_group(
                            adata,
                            groupby=str(groupby),
                            group_value=str(g),
                            a_key=str(A_key),
                            b_key=str(B_key),
                            ref_a=ref_by_factor.get(str(A_key)),
                        )
                    else:
                        levels = pd.Index(pd.unique(cond_norm.loc[g_mask])).sort_values().tolist()
                        if len(levels) < 2:
                            continue
                        counts_by_level = cond_norm.loc[g_mask].astype(str).value_counts().to_dict()
                        pairs = _select_pairs_with_ref(
                            levels,
                            ref_level=ref_by_key.get(str(cond_key)),
                            requested=condition_contrasts,
                            counts_by_level=counts_by_level,
                        )
                    for A, B in pairs:
                        tasks.append(
                            {
                                "kind": "pair",
                                "cond_key": str(cond_key),
                                "group": str(g),
                                "A": str(A),
                                "B": str(B),
                            }
                        )
                        tasks_by_key[str(cond_key)] = tasks_by_key.get(str(cond_key), 0) + 1
                        groups_with_tasks_by_key.setdefault(str(cond_key), set()).add(str(g))

            for cond_key in condition_keys:
                total_pairs = int(tasks_by_key.get(str(cond_key), 0))
                groups_with_tasks = len(groups_with_tasks_by_key.get(str(cond_key), set()))
                LOGGER.info(
                    "within-cluster: pseudobulk task build summary (condition_key=%r, groups_skipped=%d, total_pairs=%d).",
                    str(cond_key),
                    int(len(groups) - groups_with_tasks),
                    int(total_pairs),
                )

            total = int(len(tasks))
            total_cpus = int(getattr(cfg, "n_jobs", 1))
            worker_cap = int(getattr(cfg, "max_workers", 16) or 16)
            requested_workers = min(int(total_cpus), max(1, total), int(max(1, worker_cap)))
            max_workers = 1
            if requested_workers > 1:
                LOGGER.warning(
                    "within-cluster: pseudobulk requested workers=%d, but PyDESeq2 fits in thread pools "
                    "have shown native instability here; forcing serial execution for stability.",
                    int(requested_workers),
                )
            LOGGER.info(
                "within-cluster: pseudobulk execution mode (tasks=%d, workers=%d, requested_workers=%d, total_cpus=%d, worker_cap=%d).",
                int(total),
                int(max_workers),
                int(requested_workers),
                int(total_cpus),
                int(max(1, worker_cap)),
            )

            entries: list[tuple[str, dict]] = []

            def _run_task(task: dict[str, str]):
                t_task_start = time.perf_counter()
                if task["kind"] == "interaction":
                    res, meta = de_condition_within_group_pseudobulk_interaction(
                        adata,
                        group_value=str(task["group"]),
                        groupby=str(groupby),
                        round_id=getattr(cfg, "round_id", None),
                        factor_a=str(task["a_key"]),
                        factor_b=str(task["b_key"]),
                        ref_a=str(task.get("ref_a", "")) if task.get("ref_a") else None,
                        ref_b=str(task.get("ref_b", "")) if task.get("ref_b") else None,
                        spec=pb_spec,
                        opts=pb_opts,
                        gene_names=gene_names_for_de,
                        store_key=None,
                        store=False,
                        n_cpus=1,
                    )
                    t_task_end = time.perf_counter()
                    return "interaction", str(task["cond_key"]), str(task["group"]), res, meta, float(t_task_end - t_task_start)
                res = de_condition_within_group_pseudobulk_multi(
                    adata,
                    group_value=str(task["group"]),
                    groupby=str(groupby),
                    round_id=getattr(cfg, "round_id", None),
                    condition_key=str(task["cond_key"]),
                    spec=pb_spec,
                    opts=pb_opts,
                    gene_names=gene_names_for_de,
                    contrasts=[f"{task['A']}_vs_{task['B']}"],
                    store_key=None,
                    store=False,
                    n_cpus=1,
                )
                t_task_end = time.perf_counter()
                return "pair", str(task["cond_key"]), str(task["group"]), res, None, float(t_task_end - t_task_start)

            done = 0
            t0 = time.perf_counter()
            if tasks and max_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

                heartbeat_s = 60.0
                slow_task_warn_s = 30 * 60.0
                next_heartbeat = time.perf_counter() + heartbeat_s
                _set_blas_threads(1, force=True)
                gc_every_n = max(8, int(max_workers))
                LOGGER.info(
                    "within-cluster: pseudobulk memory-hygiene settings (scheduler=dynamic_refill, gc_every_n=%d).",
                    int(gc_every_n),
                )

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    task_iter = iter(tasks)
                    futs: dict = {}
                    pending = set()
                    submitted_by_fut: dict = {}

                    for _ in range(min(int(max_workers), len(tasks))):
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            break
                        fut = ex.submit(_run_task, task)
                        futs[fut] = task
                        pending.add(fut)
                        submitted_by_fut[fut] = time.perf_counter()

                    while pending:
                        now = time.perf_counter()
                        done_set, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                        if not done_set:
                            if now >= next_heartbeat:
                                if pending:
                                    longest_fut = max(pending, key=lambda f: now - submitted_by_fut.get(f, now))
                                    longest_age = now - submitted_by_fut.get(longest_fut, now)
                                    longest_task = futs.get(longest_fut, {})
                                    LOGGER.info(
                                        "within-cluster: pseudobulk longest-pending task (wall_age=%.1fs, kind=%s, group=%s, condition_key=%s, A=%s, B=%s, factor_a=%s, factor_b=%s).",
                                        float(longest_age),
                                        str(longest_task.get("kind")),
                                        str(longest_task.get("group")),
                                        str(longest_task.get("cond_key")),
                                        str(longest_task.get("A", "")),
                                        str(longest_task.get("B", "")),
                                        str(longest_task.get("a_key", "")),
                                        str(longest_task.get("b_key", "")),
                                    )
                                    if float(longest_age) >= float(slow_task_warn_s):
                                        LOGGER.warning(
                                            "within-cluster: pseudobulk slow pending task (wall_age=%.1fs, kind=%s, group=%s, condition_key=%s, A=%s, B=%s, factor_a=%s, factor_b=%s).",
                                            float(longest_age),
                                            str(longest_task.get("kind")),
                                            str(longest_task.get("group")),
                                            str(longest_task.get("cond_key")),
                                            str(longest_task.get("A", "")),
                                            str(longest_task.get("B", "")),
                                            str(longest_task.get("a_key", "")),
                                            str(longest_task.get("b_key", "")),
                                        )
                                LOGGER.info(
                                    "within-cluster: pseudobulk heartbeat (done=%d/%d, pending=%d).",
                                    int(done),
                                    int(total),
                                    int(len(pending)),
                                )
                                next_heartbeat = now + heartbeat_s
                            continue
                        for fut in done_set:
                            done_now = time.perf_counter()
                            kind, cond_key, gval, res, meta, run_s = fut.result()
                            task_info = futs.get(fut, {})
                            submitted_at = submitted_by_fut.get(fut, done_now)
                            wall_s = max(0.0, float(done_now - submitted_at))
                            queue_s = max(0.0, float(wall_s - float(run_s)))
                            if kind == "interaction":
                                key = f"{groupby}={gval}::{cond_key}::interaction"
                                payload = {
                                    "group_key": str(groupby),
                                    "group_value": str(gval),
                                    "condition_key": str(cond_key),
                                    "test": None,
                                    "reference": None,
                                    "interaction": True,
                                    "factor_a": meta.get("factor_a") if meta else None,
                                    "factor_b": meta.get("factor_b") if meta else None,
                                    "ref_a": meta.get("ref_a") if meta else None,
                                    "ref_b": meta.get("ref_b") if meta else None,
                                    "level_a": meta.get("level_a") if meta else None,
                                    "level_b": meta.get("level_b") if meta else None,
                                    "coef_name": meta.get("coef_name") if meta else None,
                                    "results": res,
                                    "options": {
                                        "min_cells_per_sample_group": int(pb_opts.min_cells_per_sample_group),
                                        "min_samples_per_level": int(pb_opts.min_samples_per_level),
                                        "alpha": float(pb_opts.alpha),
                                        "shrink_lfc": bool(pb_opts.shrink_lfc),
                                        "min_total_counts": int(getattr(pb_opts, "min_total_counts", 0)),
                                        "min_counts_per_lib": int(getattr(pb_opts, "min_counts_per_lib", 0)),
                                    },
                                }
                                entries.append((key, payload))
                            else:
                                for contrast_key, df in res.items():
                                    A2, B2 = _normalize_pair(contrast_key)
                                    key = f"{groupby}={gval}::{cond_key}::{A2}_vs_{B2}"
                                    payload = {
                                        "group_key": str(groupby),
                                        "group_value": str(gval),
                                        "condition_key": str(cond_key),
                                        "test": str(A2),
                                        "reference": str(B2),
                                        "results": df,
                                        "options": {
                                            "min_cells_per_sample_group": int(pb_opts.min_cells_per_sample_group),
                                            "min_samples_per_level": int(pb_opts.min_samples_per_level),
                                            "alpha": float(pb_opts.alpha),
                                            "shrink_lfc": bool(pb_opts.shrink_lfc),
                                            "min_total_counts": int(getattr(pb_opts, "min_total_counts", 0)),
                                            "min_counts_per_lib": int(getattr(pb_opts, "min_counts_per_lib", 0)),
                                        },
                                    }
                                    LOGGER.info(
                                        "within-cluster: pseudobulk stored contrast=%r test=%r reference=%r group=%r condition_key=%r",
                                        str(contrast_key),
                                        str(A2),
                                        str(B2),
                                        str(gval),
                                        str(cond_key),
                                    )
                                    entries.append((key, payload))
                            done += 1
                            elapsed = time.perf_counter() - t0
                            eta_s = (elapsed / max(1, done)) * (total - done)
                            LOGGER.info(
                                "within-cluster: pseudobulk progress %d/%d (elapsed=%.1fs, eta=%.1fs).",
                                int(done),
                                int(total),
                                elapsed,
                                eta_s,
                            )
                            LOGGER.info(
                                "within-cluster: pseudobulk task finished (kind=%s, group=%s, condition_key=%s, A=%s, B=%s, factor_a=%s, factor_b=%s, run_s=%.1f, queue_s=%.1f, wall_s=%.1f).",
                                str(task_info.get("kind")),
                                str(task_info.get("group")),
                                str(task_info.get("cond_key")),
                                str(task_info.get("A", "")),
                                str(task_info.get("B", "")),
                                str(task_info.get("a_key", "")),
                                str(task_info.get("b_key", "")),
                                float(run_s),
                                float(queue_s),
                                float(wall_s),
                            )

                            futs.pop(fut, None)
                            submitted_by_fut.pop(fut, None)
                            del fut

                            try:
                                task = next(task_iter)
                                fut2 = ex.submit(_run_task, task)
                                futs[fut2] = task
                                pending.add(fut2)
                                submitted_by_fut[fut2] = time.perf_counter()
                            except StopIteration:
                                pass

                            if done % int(gc_every_n) == 0:
                                gc.collect()
                gc.collect()
            else:
                for task in tasks:
                    kind, cond_key, gval, res, meta, run_s = _run_task(task)
                    if kind == "interaction":
                        key = f"{groupby}={gval}::{cond_key}::interaction"
                        payload = {
                            "group_key": str(groupby),
                            "group_value": str(gval),
                            "condition_key": str(cond_key),
                            "test": None,
                            "reference": None,
                            "interaction": True,
                            "factor_a": meta.get("factor_a") if meta else None,
                            "factor_b": meta.get("factor_b") if meta else None,
                            "ref_a": meta.get("ref_a") if meta else None,
                            "ref_b": meta.get("ref_b") if meta else None,
                            "level_a": meta.get("level_a") if meta else None,
                            "level_b": meta.get("level_b") if meta else None,
                            "coef_name": meta.get("coef_name") if meta else None,
                            "results": res,
                            "options": {
                                "min_cells_per_sample_group": int(pb_opts.min_cells_per_sample_group),
                                "min_samples_per_level": int(pb_opts.min_samples_per_level),
                                "alpha": float(pb_opts.alpha),
                                "shrink_lfc": bool(pb_opts.shrink_lfc),
                                "min_total_counts": int(getattr(pb_opts, "min_total_counts", 0)),
                                "min_counts_per_lib": int(getattr(pb_opts, "min_counts_per_lib", 0)),
                            },
                        }
                        entries.append((key, payload))
                    else:
                        for contrast_key, df in res.items():
                            A2, B2 = _normalize_pair(contrast_key)
                            key = f"{groupby}={gval}::{cond_key}::{A2}_vs_{B2}"
                            payload = {
                                "group_key": str(groupby),
                                "group_value": str(gval),
                                "condition_key": str(cond_key),
                                "test": str(A2),
                                "reference": str(B2),
                                "results": df,
                                "options": {
                                    "min_cells_per_sample_group": int(pb_opts.min_cells_per_sample_group),
                                    "min_samples_per_level": int(pb_opts.min_samples_per_level),
                                    "alpha": float(pb_opts.alpha),
                                    "shrink_lfc": bool(pb_opts.shrink_lfc),
                                    "min_total_counts": int(getattr(pb_opts, "min_total_counts", 0)),
                                    "min_counts_per_lib": int(getattr(pb_opts, "min_counts_per_lib", 0)),
                                },
                            }
                            LOGGER.info(
                                "within-cluster: pseudobulk stored contrast=%r test=%r reference=%r group=%r condition_key=%r",
                                str(contrast_key),
                                str(A2),
                                str(B2),
                                str(gval),
                                str(cond_key),
                            )
                            entries.append((key, payload))
                    done += 1
                    elapsed = time.perf_counter() - t0
                    eta_s = (elapsed / max(1, done)) * (total - done)
                    LOGGER.info(
                        "within-cluster: pseudobulk progress %d/%d (elapsed=%.1fs, eta=%.1fs).",
                        int(done),
                        int(total),
                        elapsed,
                        eta_s,
                    )
                    LOGGER.info(
                        "within-cluster: pseudobulk task finished (kind=%s, group=%s, condition_key=%s, A=%s, B=%s, factor_a=%s, factor_b=%s, run_s=%.1f, queue_s=%.1f, wall_s=%.1f).",
                        str(task.get("kind", "")),
                        str(task.get("group", "")),
                        str(task.get("cond_key", "")),
                        str(task.get("A", "")),
                        str(task.get("B", "")),
                        str(task.get("a_key", "")),
                        str(task.get("b_key", "")),
                        float(run_s),
                        0.0,
                        float(run_s),
                    )

            LOGGER.info(
                "within-cluster: pseudobulk tasks complete (done=%d/%d).",
                int(done),
                int(total),
            )

            if store_key:
                adata.uns.setdefault(store_key, {})
                adata.uns[store_key].setdefault("pseudobulk_condition_within_group_multi", {})
                for key, payload in entries:
                    adata.uns[store_key]["pseudobulk_condition_within_group_multi"][key] = payload

            ran_pseudobulk = True
    else:
        if regenerate_figures and run_pb_requested:
            LOGGER.info("within-cluster: regenerate_figures=True; skipping pseudobulk computation.")
        LOGGER.info("within-cluster: skipping pseudobulk (run=%r).", mode)

    # ----------------------------
    # 2) Cell-level within-cluster contrasts (NO pseudobulk guard)
    # ----------------------------
    ran_cell_contrast = False
    if run_cell_requested and not regenerate_figures:
        from .de_utils import ContrastConditionalSpec

        total_cpus = int(getattr(cfg, "n_jobs", 1))
        specs = []
        for condition_key in condition_keys:
            if interaction_by_key.get(str(condition_key)):
                LOGGER.info(
                    "within-cluster: skipping cell-level contrasts for interaction condition_key=%r.",
                    str(condition_key),
                )
                continue
            if str(condition_key) not in adata.obs:
                raise RuntimeError(f"within-cluster: condition_key={condition_key!r} not found in adata.obs")
            contrast_key = getattr(cfg, "contrast_key", None) or str(condition_key)
            within_parts = within_by_key.get(str(condition_key))
            ref_level = ref_by_key.get(str(condition_key))
            if within_parts:
                ref_level = ref_by_factor.get(str(within_parts[0]))
            specs.append(
                ContrastConditionalSpec(
                    contrast_key=str(contrast_key),
                    within_parts=within_parts,
                    ref_level=ref_level,
                    methods=tuple(getattr(cfg, "contrast_methods", ("wilcoxon", "logreg"))),
                    min_cells_per_level_in_cluster=int(getattr(cfg, "contrast_min_cells_per_level", 50)),
                    max_cells_per_level_in_cluster=int(getattr(cfg, "contrast_max_cells_per_level", 2000)),
                    min_total_counts=int(getattr(cfg, "contrast_min_total_counts", 10)),
                    pseudocount=float(getattr(cfg, "contrast_pseudocount", 1.0)),
                    cl_alpha=float(getattr(cfg, "contrast_cl_alpha", 0.05)),
                    cl_min_abs_logfc=float(getattr(cfg, "contrast_cl_min_abs_logfc", 0.25)),
                    lr_min_abs_coef=float(getattr(cfg, "contrast_lr_min_abs_coef", 0.25)),
                    pb_min_abs_log2fc=float(getattr(cfg, "contrast_pb_min_abs_log2fc", 0.5)),
                    random_state=int(getattr(cfg, "random_state", 42)),
                    min_pct=float(getattr(cfg, "min_pct", 0.25)),
                    min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                )
            )

        from .de_utils import contrast_conditional_markers_multi
        worker_cap = int(getattr(cfg, "max_workers", 16) or 16)
        cell_workers = min(int(total_cpus), int(max(1, worker_cap)))
        LOGGER.info(
            "within-cluster: cell-level worker selection (workers=%d, total_cpus=%d, worker_cap=%d).",
            int(cell_workers),
            int(total_cpus),
            int(max(1, worker_cap)),
        )
        results_by_key, summaries_by_key = contrast_conditional_markers_multi(
            adata,
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            specs=specs,
            pb_spec=pb_spec,
            gene_var_mask=gene_var_mask,
            n_jobs=cell_workers,
            cluster_values=selected_groups.astype(str).tolist(),
        )

        for condition_key in condition_keys:
            if interaction_by_key.get(str(condition_key)):
                continue
            contrast_key = getattr(cfg, "contrast_key", None) or str(condition_key)
            out_local = results_by_key.get(str(contrast_key), {})
            summary_local = summaries_by_key.get(str(contrast_key), pd.DataFrame())

            if store_key:
                adata.uns.setdefault(store_key, {})
                adata.uns[store_key].setdefault("contrast_conditional", {})
                adata.uns[store_key]["contrast_conditional"]["group_key"] = str(groupby)
                adata.uns[store_key]["contrast_conditional"]["contrast_key"] = str(contrast_key)
                counts_layer = pb_spec.counts_layer if pb_spec is not None else None
                adata.uns[store_key]["contrast_conditional"]["counts_layer"] = str(counts_layer) if counts_layer else None
                spec_match = next((s for s in specs if str(s.contrast_key) == str(contrast_key)), None)
                if spec_match is not None:
                    adata.uns[store_key]["contrast_conditional"]["spec"] = {**spec_match.__dict__, "methods": list(spec_match.methods)}
                adata.uns[store_key]["contrast_conditional"]["pseudobulk_effect_included"] = bool(pb_spec is not None)
                adata.uns[store_key]["contrast_conditional"]["summary"] = summary_local
                adata.uns[store_key]["contrast_conditional"]["results"] = out_local

                from copy import deepcopy
                adata.uns[store_key].setdefault("contrast_conditional_multi", {})
                adata.uns[store_key]["contrast_conditional_multi"][str(contrast_key)] = deepcopy(
                    adata.uns[store_key]["contrast_conditional"]
                )

            io_utils.export_contrast_conditional_markers_tables(
                adata,
                output_dir=output_dir,
                store_key=store_key,
                display_map=display_map,
                tables_root=de_cell_dir,
                filename=None,
                contrast_key=str(contrast_key),
            )

            settings_name = "de_settings.txt"
            if len(condition_keys) > 1:
                settings_name = f"de_settings__{io_utils.sanitize_identifier(condition_key, allow_spaces=False)}.txt"

            _write_settings(
                de_cell_dir,
                settings_name,
                [
                    "mode=within-cluster",
                    "engine=cell",
                    f"groupby={groupby}",
                    f"target_groups={list(getattr(cfg, 'target_groups', ()) or ())}",
                    f"contrast_key={contrast_key}",
                    f"contrast_methods={tuple(getattr(cfg, 'contrast_methods', ())) }",
                    f"max_workers={getattr(cfg, 'max_workers', None)}",
                    f"contrast_min_cells_per_level={getattr(cfg, 'contrast_min_cells_per_level', None)}",
                    f"contrast_max_cells_per_level={getattr(cfg, 'contrast_max_cells_per_level', None)}",
                    f"contrast_min_total_counts={getattr(cfg, 'contrast_min_total_counts', None)}",
                    f"contrast_pseudocount={getattr(cfg, 'contrast_pseudocount', None)}",
                    f"contrast_cl_alpha={getattr(cfg, 'contrast_cl_alpha', None)}",
                    f"contrast_cl_min_abs_logfc={getattr(cfg, 'contrast_cl_min_abs_logfc', None)}",
                    f"contrast_lr_min_abs_coef={getattr(cfg, 'contrast_lr_min_abs_coef', None)}",
                    f"contrast_pb_min_abs_log2fc={getattr(cfg, 'contrast_pb_min_abs_log2fc', None)}",
                    f"min_pct={getattr(cfg, 'min_pct', None)}",
                    f"min_diff_pct={getattr(cfg, 'min_diff_pct', None)}",
                    f"gene_filter={list(gene_filter)}",
                    f"gene_filter_n_genes_input={int(gene_filter_info['n_genes_input']) if gene_filter_info else int(adata.n_vars)}",
                    f"gene_filter_n_genes_retained={int(gene_filter_info['n_genes_retained']) if gene_filter_info else int(adata.n_vars)}",
                    f"de_decoupler_source={getattr(cfg, 'de_decoupler_source', None)}",
                    f"de_decoupler_stat_col={getattr(cfg, 'de_decoupler_stat_col', None)}",
                    f"decoupler_method={getattr(cfg, 'decoupler_method', None)}",
                    f"decoupler_consensus_methods={getattr(cfg, 'decoupler_consensus_methods', None)}",
                    f"decoupler_min_n_targets={getattr(cfg, 'decoupler_min_n_targets', None)}",
                    f"msigdb_gene_sets={getattr(cfg, 'msigdb_gene_sets', None)}",
                    f"msigdb_method={getattr(cfg, 'msigdb_method', None)}",
                    f"msigdb_min_n_targets={getattr(cfg, 'msigdb_min_n_targets', None)}",
                    f"run_progeny={getattr(cfg, 'run_progeny', None)}",
                    f"progeny_method={getattr(cfg, 'progeny_method', None)}",
                    f"progeny_min_n_targets={getattr(cfg, 'progeny_min_n_targets', None)}",
                    f"progeny_top_n={getattr(cfg, 'progeny_top_n', None)}",
                    f"progeny_organism={getattr(cfg, 'progeny_organism', None)}",
                    f"run_dorothea={getattr(cfg, 'run_dorothea', None)}",
                    f"dorothea_method={getattr(cfg, 'dorothea_method', None)}",
                    f"dorothea_min_n_targets={getattr(cfg, 'dorothea_min_n_targets', None)}",
                    f"dorothea_confidence={getattr(cfg, 'dorothea_confidence', None)}",
                    f"dorothea_organism={getattr(cfg, 'dorothea_organism', None)}",
                    "design_formula=NA (cell-level)",
                ],
            )
        ran_cell_contrast = True
    else:
        if regenerate_figures and run_cell_requested:
            LOGGER.info("within-cluster: regenerate_figures=True; skipping cell-level contrast computation.")
        LOGGER.info("within-cluster: skipping cell-level within-cluster contrasts (run=%r).", mode)

    # ----------------------------
    # 3) DE-based decoupler (pathways/TF from DE stats)
    # ----------------------------
    de_decoupler_ran = False
    de_source = str(getattr(cfg, "de_decoupler_source", "auto") or "auto").lower()
    if de_source not in ("auto", "all", "pseudobulk", "cell", "none"):
        raise RuntimeError(f"within-cluster: invalid de_decoupler_source={de_source!r}")

    if de_source != "none" and not regenerate_figures:
        from .annotation_utils import (
            _run_msigdb_from_stats,
            _run_msigdb_gsea_from_stats,
            _merge_msigdb_decoupler_and_gsea,
            _run_progeny_from_stats,
            _run_dorothea_from_stats,
        )

        stat_col = str(getattr(cfg, "de_decoupler_stat_col", "stat") or "stat")

        for condition_key in condition_keys:
            pb_tables = _collect_pseudobulk_de_tables(
                adata,
                store_key=store_key,
                condition_key=str(condition_key),
            )
            cell_tables = _collect_cell_contrast_tables(
                adata,
                store_key=store_key,
                contrast_key=str(condition_key),
            )

            sources: list[tuple[str, dict[str, dict[str, pd.DataFrame]]]] = []
            if de_source == "auto":
                if pb_tables:
                    sources.append(("pseudobulk", pb_tables))
                if cell_tables:
                    sources.append(("cell", cell_tables))
            elif de_source == "all":
                if pb_tables:
                    sources.append(("pseudobulk", pb_tables))
                if cell_tables:
                    sources.append(("cell", cell_tables))
            elif de_source == "pseudobulk":
                if pb_tables:
                    sources = [("pseudobulk", pb_tables)]
            elif de_source == "cell":
                if cell_tables:
                    sources = [("cell", cell_tables)]

            if not sources:
                LOGGER.info(
                    "DE-decoupler: no DE tables found for condition_key=%r (skipping).",
                    str(condition_key),
                )
                continue

            for source, tables_by_contrast in sources:
                for contrast, tables in tables_by_contrast.items():
                    LOGGER.info(
                        "DE-decoupler: running source=%r condition_key=%r contrast=%r stat_col=%r",
                        str(source),
                        str(condition_key),
                        str(contrast),
                        str(stat_col),
                    )
                    stats = _build_stats_matrix_from_tables(
                        tables,
                        preferred_col=stat_col,
                        fallback_cols=("log2FoldChange", "cell_wilcoxon_score", "cell_wilcoxon_logfc", "cell_logreg_coef"),
                    )
                    if stats is None or stats.empty:
                        continue

                    input_label = f"{source}:{condition_key}:{contrast}:{stat_col}"
                    payloads = {}

                    LOGGER.info(
                        "DE enrichment: running MSigDB decoupler source=%r condition_key=%r contrast=%r stat_col=%r",
                        str(source),
                        str(condition_key),
                        str(contrast),
                        str(stat_col),
                    )
                    msigdb_payload = _run_msigdb_from_stats(stats, cfg, input_label=input_label)
                    if msigdb_payload is not None:
                        payloads["msigdb"] = msigdb_payload
                    if bool(getattr(cfg, "run_gsea", True)):
                        LOGGER.info(
                            "DE enrichment: running MSigDB GSEA source=%r condition_key=%r contrast=%r stat_col=%r",
                            str(source),
                            str(condition_key),
                            str(contrast),
                            str(stat_col),
                        )
                        gsea_payload = _run_msigdb_gsea_from_stats(stats, cfg, input_label=input_label)
                        if gsea_payload is not None:
                            payloads["msigdb_gsea"] = gsea_payload
                            if msigdb_payload is not None:
                                joint_payload = _merge_msigdb_decoupler_and_gsea(
                                    decoupler_payload=msigdb_payload,
                                    gsea_payload=gsea_payload,
                                    alpha=float(getattr(cfg, "joint_enrichment_alpha", 0.05)),
                                    leading_edge_top_n=int(getattr(cfg, "joint_enrichment_leading_edge_top_n", 8)),
                                )
                                if joint_payload is not None:
                                    payloads["msigdb_joint"] = joint_payload

                    if bool(getattr(cfg, "run_progeny", True)):
                        prog = _run_progeny_from_stats(stats, cfg, input_label=input_label)
                        if prog is not None:
                            payloads["progeny"] = prog

                    if bool(getattr(cfg, "run_dorothea", True)):
                        doro = _run_dorothea_from_stats(stats, cfg, input_label=input_label)
                        if doro is not None:
                            payloads["dorothea"] = doro

                    if not payloads:
                        continue

                    adata.uns.setdefault(store_key, {})
                    adata.uns[store_key].setdefault("de_decoupler", {})
                    adata.uns[store_key]["de_decoupler"].setdefault(str(condition_key), {})
                    adata.uns[store_key]["de_decoupler"][str(condition_key)].setdefault(str(contrast), {})
                    adata.uns[store_key]["de_decoupler"][str(condition_key)][str(contrast)][str(source)] = {
                        "source": source,
                        "stat_col": stat_col,
                        "nets": payloads,
                    }

                    de_decoupler_ran = True
    else:
        if regenerate_figures and de_source != "none":
            LOGGER.info("within-cluster: regenerate_figures=True; skipping DE-decoupler computation.")
        else:
            LOGGER.info("within-cluster: skipping DE-decoupler (disabled).")

    # ----------------------------
    # Provenance (safe)
    # ----------------------------
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "mode": "within-cluster",
            "groupby": str(groupby),
            "condition_key": str(condition_keys[-1]) if condition_keys else None,
            "condition_keys": list(condition_keys),
            "condition_contrasts": list(condition_contrasts) if condition_contrasts else None,
            "sample_key": str(sample_key),
            "run": mode,
            "cell_requested": bool(run_cell_requested),
            "cell_ran": bool(ran_cell_contrast),
            "pseudobulk_requested": bool(run_pb_requested),
            "pseudobulk_enabled": bool(pseudobulk_enabled and run_pb_requested),
            "pseudobulk_ran": bool(ran_pseudobulk),
            "pseudobulk_guard_min_total_samples": int(_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK),
            "pseudobulk_n_unique_samples": int(n_samples_total),
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer_used,
            "gene_filter": list(gene_filter),
            "gene_filter_n_genes_input": int(gene_filter_info["n_genes_input"]) if gene_filter_info else int(adata.n_vars),
            "gene_filter_n_genes_retained": int(gene_filter_info["n_genes_retained"]) if gene_filter_info else int(adata.n_vars),
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "positive_only_markers": bool(positive_only),
            "de_decoupler_ran": bool(de_decoupler_ran),
        }
    )

    # ----------------------------
    # Pseudobulk exports (only if actually ran)
    # ----------------------------
    if ran_pseudobulk and not regenerate_figures:
        covariates = tuple(getattr(cfg, "pb_covariates", ()))

        for condition_key in condition_keys:
            io_utils.export_pseudobulk_de_tables(
                adata,
                output_dir=output_dir,
                store_key=store_key,
                display_map=display_map,
                groupby=str(groupby),
                condition_key=str(condition_key),
                tables_root=de_pb_dir,
            )
            io_utils.export_pseudobulk_condition_within_cluster_excel(
                adata,
                output_dir=output_dir,
                store_key=store_key,
                condition_key=str(condition_key),
                display_map=display_map,
                tables_root=de_pb_dir,
            )
            design_terms = ["sample", *covariates, str(condition_key)]
            interaction_parts = interaction_by_key.get(str(condition_key))
            if interaction_parts:
                design_terms = ["sample", *covariates, str(interaction_parts[0]), str(interaction_parts[1]), f"{interaction_parts[0]}:{interaction_parts[1]}"]

            settings_name = "de_settings.txt"
            if len(condition_keys) > 1:
                settings_name = f"de_settings__{io_utils.sanitize_identifier(condition_key, allow_spaces=False)}.txt"

            _write_settings(
                de_pb_dir,
                settings_name,
                [
                    "mode=within-cluster",
                    "engine=pseudobulk",
                    f"groupby={groupby}",
                    f"condition_key={condition_key}",
                    f"target_groups={list(getattr(cfg, 'target_groups', ()) or ())}",
                    f"sample_key={sample_key}",
                    f"counts_layer={counts_layer_used}",
                    f"min_cells_per_sample_group={getattr(cfg, 'min_cells_condition', None)}",
                    f"min_samples_per_level={getattr(cfg, 'min_samples_per_level', None)}",
                    f"max_workers={getattr(cfg, 'max_workers', None)}",
                    f"gene_filter={list(gene_filter)}",
                    f"gene_filter_n_genes_input={int(gene_filter_info['n_genes_input']) if gene_filter_info else int(adata.n_vars)}",
                    f"gene_filter_n_genes_retained={int(gene_filter_info['n_genes_retained']) if gene_filter_info else int(adata.n_vars)}",
                    f"alpha={getattr(cfg, 'alpha', None)}",
                    f"shrink_lfc={getattr(cfg, 'shrink_lfc', None)}",
                    f"min_total_counts={getattr(cfg, 'pb_min_total_counts', None)}",
                    f"min_counts_per_lib={getattr(cfg, 'pb_min_counts_per_lib', None)}",
                    f"min_lib_pct={getattr(cfg, 'pb_min_lib_pct', None)}",
                    f"min_pct={getattr(cfg, 'min_pct', None)}",
                    "min_diff_pct=NA (unused for pseudobulk)",
                    f"positive_only={getattr(cfg, 'positive_only', None)}",
                    f"de_decoupler_source={getattr(cfg, 'de_decoupler_source', None)}",
                    f"de_decoupler_stat_col={getattr(cfg, 'de_decoupler_stat_col', None)}",
                    f"decoupler_method={getattr(cfg, 'decoupler_method', None)}",
                    f"decoupler_consensus_methods={getattr(cfg, 'decoupler_consensus_methods', None)}",
                    f"decoupler_min_n_targets={getattr(cfg, 'decoupler_min_n_targets', None)}",
                    f"msigdb_gene_sets={getattr(cfg, 'msigdb_gene_sets', None)}",
                    f"msigdb_method={getattr(cfg, 'msigdb_method', None)}",
                    f"msigdb_min_n_targets={getattr(cfg, 'msigdb_min_n_targets', None)}",
                    f"run_progeny={getattr(cfg, 'run_progeny', None)}",
                    f"progeny_method={getattr(cfg, 'progeny_method', None)}",
                    f"progeny_min_n_targets={getattr(cfg, 'progeny_min_n_targets', None)}",
                    f"progeny_top_n={getattr(cfg, 'progeny_top_n', None)}",
                    f"progeny_organism={getattr(cfg, 'progeny_organism', None)}",
                    f"run_dorothea={getattr(cfg, 'run_dorothea', None)}",
                    f"dorothea_method={getattr(cfg, 'dorothea_method', None)}",
                    f"dorothea_min_n_targets={getattr(cfg, 'dorothea_min_n_targets', None)}",
                    f"dorothea_confidence={getattr(cfg, 'dorothea_confidence', None)}",
                    f"dorothea_organism={getattr(cfg, 'dorothea_organism', None)}",
                    f"pb_max_genes={getattr(cfg, 'pb_max_genes', None)}",
                    f"pb_covariates={covariates}",
                    f"design_formula={' + '.join(design_terms)}",
                ],
            )
    else:
        LOGGER.info("within-cluster: skipping pseudobulk exports (not run).")

    if de_decoupler_ran and not regenerate_figures:
        _export_de_enrichment_tables_from_uns(
            adata,
            store_key=store_key,
            de_cell_dir=de_cell_dir / "de_enrichment",
            de_pb_dir=de_pb_dir / "de_enrichment",
        )

    if regenerate_figures:
        ran_pseudobulk = bool(run_pb_requested)
        ran_cell_contrast = bool(run_cell_requested)

    # ----------------------------
    # Plotting (only what ran)
    # ----------------------------
    if bool(getattr(cfg, "make_figures", True)):
        from . import de_plot_utils

        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
        top_n_genes = int(getattr(cfg, "plot_top_n_per_cluster", 9))
        dotplot_top_n_genes = int(getattr(cfg, "plot_dotplot_top_n_genes", 15))
        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)
        plot_ann_keys = list(getattr(cfg, "plot_sample_annotation_keys", ()) or condition_keys)

        try:
            de_plot_utils.prepare_condition_color_registry(
                adata,
                condition_keys=condition_keys,
                annotation_keys=plot_ann_keys,
            )
            if condition_keys:
                LOGGER.info("within-cluster: plotting UMAPs for condition keys...")
                artifacts = de_plot_utils.plot_condition_umaps(
                    adata,
                    groupby=str(groupby),
                    condition_keys=condition_keys,
                )
                plot_utils.persist_plot_artifacts(artifacts)

            # Condition plots only if pseudobulk ran
            if ran_pseudobulk:
                LOGGER.info("within-cluster: plotting pseudobulk condition figures...")
                for condition_key in condition_keys:
                    artifacts = de_plot_utils.plot_condition_within_cluster_all(
                        adata,
                        cluster_key=str(groupby),
                        condition_key=str(condition_key),
                        store_key=store_key,
                        alpha=alpha,
                        lfc_thresh=lfc_thresh,
                        top_label_n=top_label_n,
                        dotplot_top_n=dotplot_top_n_genes,
                        violin_top_n=top_n_genes,
                        heatmap_top_n=dotplot_top_n_genes,
                        use_raw=use_raw,
                        layer=layer,
                        sample_key=sample_key,
                        plot_gene_filter=getattr(cfg, "plot_gene_filter", ()),
                        annotation_keys=plot_ann_keys,
                    )
                    plot_utils.persist_plot_artifacts(artifacts)
            else:
                LOGGER.info("within-cluster: skipping condition plots (pseudobulk not run).")

            if ran_cell_contrast:
                LOGGER.info("within-cluster: plotting cell-level contrast figures...")
                for condition_key in condition_keys:
                    artifacts = de_plot_utils.plot_contrast_conditional_markers_multi(
                        adata,
                        groupby=str(groupby),
                        display_map=display_map,
                        contrast_key=str(getattr(cfg, "contrast_key", None) or condition_key),
                        store_key=store_key,
                        alpha=alpha,
                        lfc_thresh=lfc_thresh,
                        top_label_n=top_label_n,
                        top_n_genes=top_n_genes,
                        dotplot_top_n_genes=dotplot_top_n_genes,
                        use_raw=use_raw,
                        layer=layer,
                        sample_key=sample_key,
                        plot_gene_filter=getattr(cfg, "plot_gene_filter", ()),
                        annotation_keys=plot_ann_keys,
                    )
                    plot_utils.persist_plot_artifacts(artifacts)

            # DE-based decoupler plots (if available)
            if de_source != "none":
                de_block = adata.uns.get(store_key, {}).get("de_decoupler", {})
                if isinstance(de_block, dict) and de_block:
                    LOGGER.info("within-cluster: plotting DE-decoupler figures...")
                    pb_cond = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
                    pb_cond_multi = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group_multi", {})
                    plot_tasks: list[tuple[dict, str, Path, Path, str, Optional[str], Optional[str]]] = []
                    for condition_key, per_contrast in de_block.items():
                        if not isinstance(per_contrast, dict):
                            continue
                        for contrast, payload_by_source in per_contrast.items():
                            if not isinstance(payload_by_source, dict):
                                continue
                            for source, payload in payload_by_source.items():
                                if not isinstance(payload, dict):
                                    continue
                                nets = payload.get("nets", {})
                                if not isinstance(nets, dict) or not nets:
                                    continue

                                rel_base = _de_enrichment_rel_base(
                                    source=str(source),
                                    condition_key=str(condition_key),
                                    contrast=str(contrast),
                                )
                                if str(source) == "cell":
                                    base = Path("cell_level_DE") / rel_base
                                    tables_base = de_cell_dir / "de_enrichment" / rel_base
                                else:
                                    base = Path("pseudobulk_DE") / rel_base
                                    tables_base = de_pb_dir / "de_enrichment" / rel_base

                                pos_label = None
                                neg_label = None
                                if "_vs_" in str(contrast):
                                    parts = str(contrast).split("_vs_", 1)
                                    if len(parts) == 2:
                                        pos_label, neg_label = parts[0], parts[1]
                                elif str(contrast) == "interaction":
                                    payloads = []
                                    if isinstance(pb_cond, dict) and pb_cond:
                                        payloads.extend(list(pb_cond.values()))
                                    if isinstance(pb_cond_multi, dict) and pb_cond_multi:
                                        payloads.extend(list(pb_cond_multi.values()))
                                    for pb in payloads:
                                        if not isinstance(pb, dict):
                                            continue
                                        if str(pb.get("condition_key", "")) != str(condition_key):
                                            continue
                                        if not bool(pb.get("interaction", False)):
                                            continue
                                        level_a = pb.get("level_a", None)
                                        level_b = pb.get("level_b", None)
                                        ref_a = pb.get("ref_a", None)
                                        ref_b = pb.get("ref_b", None)
                                        if level_a and level_b and ref_a and ref_b:
                                            pos_label = f"{level_a}.{level_b}"
                                            neg_label = f"{ref_a}.{ref_b}"
                                            break

                                for net_name, net_payload in nets.items():
                                    plot_tasks.append(
                                        (
                                            net_payload,
                                            str(net_name),
                                            base,
                                            tables_base,
                                            f"{condition_key} {contrast}",
                                            pos_label,
                                            neg_label,
                                        )
                                    )
                    if plot_tasks:
                        requested_workers = int(getattr(cfg, "n_jobs", 1) or 1)
                        max_workers = min(max(1, requested_workers), 8, len(plot_tasks))
                        threaded_ok = _matplotlib_backend_supports_threaded_plotting()
                        if threaded_ok and max_workers > 1:
                            LOGGER.info(
                                "within-cluster: plotting DE-decoupler figures in parallel (tasks=%d, workers=%d, backend=%s).",
                                len(plot_tasks),
                                max_workers,
                                str(plot_utils.mpl.get_backend()),
                            )
                        else:
                            LOGGER.info(
                                "within-cluster: plotting DE-decoupler figures serially (tasks=%d, backend=%s).",
                                len(plot_tasks),
                                str(plot_utils.mpl.get_backend()),
                            )

                        def _plot_de_decoupler_task(task: tuple[dict, str, Path, Path, str, Optional[str], Optional[str]]):
                            net_payload, net_name, base, tables_base, title_prefix, pos_label, neg_label = task
                            payload_full = _load_de_enrichment_payload_from_tables(
                                net_payload,
                                net_name=net_name,
                                tables_root=tables_base,
                            )
                            if str(net_name) == "msigdb_gsea":
                                return de_plot_utils.plot_de_gsea_payload(
                                    payload_full,
                                    figdir=base / "msigdb_gsea",
                                    title_prefix=title_prefix,
                                    top_n=int(getattr(cfg, "joint_enrichment_top_n", 20)),
                                )
                            if str(net_name) == "msigdb_joint":
                                return de_plot_utils.plot_de_msigdb_joint_payload(
                                    payload_full,
                                    figdir=base / "msigdb_joint",
                                    title_prefix=title_prefix,
                                    top_n=int(getattr(cfg, "joint_enrichment_top_n", 20)),
                                    require_gsea_sig=bool(getattr(cfg, "joint_enrichment_require_gsea_sig", True)),
                                )
                            return de_plot_utils.plot_de_decoupler_payload(
                                    payload_full,
                                    net_name=net_name,
                                    figdir=base,
                                    heatmap_top_k=int(getattr(cfg, "plot_max_genes_total", 80)),
                                    bar_top_n=int(top_n_genes),
                                    bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                                    bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                                    bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                                    dotplot_top_k=int(dotplot_top_n_genes),
                                    title_prefix=title_prefix,
                                    pos_label=pos_label,
                                    neg_label=neg_label,
                                )

                        done = 0
                        failures = 0
                        if threaded_ok and max_workers > 1:
                            from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

                            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                                task_iter = iter(plot_tasks)
                                futs: dict = {}
                                pending = set()

                                for _ in range(min(int(max_workers), len(plot_tasks))):
                                    try:
                                        task = next(task_iter)
                                    except StopIteration:
                                        break
                                    fut = ex.submit(_plot_de_decoupler_task, task)
                                    futs[fut] = task
                                    pending.add(fut)

                                while pending:
                                    done_set, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                                    if not done_set:
                                        continue
                                    for fut in done_set:
                                        task = futs.pop(fut)
                                        done += 1
                                        try:
                                            artifacts = fut.result()
                                            plot_utils.persist_plot_artifacts(artifacts)
                                            del artifacts
                                            gc.collect()
                                        except Exception as e:
                                            failures += 1
                                            LOGGER.exception(
                                                "within-cluster: DE-decoupler plotting task failed (net=%s, figdir=%s, title=%s). (%s)",
                                                task[1],
                                                str(task[2]),
                                                task[4],
                                                e,
                                            )
                                        try:
                                            next_task = next(task_iter)
                                        except StopIteration:
                                            next_task = None
                                        if next_task is not None:
                                            next_fut = ex.submit(_plot_de_decoupler_task, next_task)
                                            futs[next_fut] = next_task
                                            pending.add(next_fut)
                                        if done == len(plot_tasks) or done % 10 == 0:
                                            LOGGER.info(
                                                "within-cluster: DE-decoupler plotting progress %d/%d (failed=%d).",
                                                done,
                                                len(plot_tasks),
                                                failures,
                                            )
                        else:
                            for task in plot_tasks:
                                done += 1
                                try:
                                    artifacts = _plot_de_decoupler_task(task)
                                    plot_utils.persist_plot_artifacts(artifacts)
                                    del artifacts
                                    gc.collect()
                                except Exception as e:
                                    failures += 1
                                    LOGGER.exception(
                                        "within-cluster: DE-decoupler plotting task failed (net=%s, figdir=%s, title=%s). (%s)",
                                        task[1],
                                        str(task[2]),
                                        task[4],
                                        e,
                                    )
                                if done == len(plot_tasks) or done % 10 == 0:
                                    LOGGER.info(
                                        "within-cluster: DE-decoupler plotting progress %d/%d (failed=%d).",
                                        done,
                                        len(plot_tasks),
                                        failures,
                                    )

            try:
                LOGGER.info("within-cluster: generating DE report...")
                for fmt in getattr(cfg, "figure_formats", ["png", "pdf"]):
                    reporting.generate_de_report(
                        fig_root=figdir,
                        fmt=fmt,
                        cfg=cfg,
                        version=__version__,
                        adata=adata,
                    )
                LOGGER.info("Wrote DE report.")
            except Exception as e:
                LOGGER.warning("Failed to generate DE report: %s", e)
        except Exception as e:
            LOGGER.exception("within-cluster: plotting failed; continuing to save outputs. (%s)", e)

    # ----------------------------
    # Save dataset
    # ----------------------------
    if regenerate_figures:
        LOGGER.info("within-cluster: regenerate_figures=True; skipping dataset save.")
    else:
        if bool(getattr(cfg, "prune_uns_de", True)):
            _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))

        out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
        LOGGER.info("Saving dataset → %s", out_zarr)
        io_utils.save_dataset(adata, out_zarr, fmt="zarr")

        if bool(getattr(cfg, "save_h5ad", False)):
            out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
            LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
            io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (within-cluster).")
    return adata
    install_env = _nichenet_r_env(r_lib_dir, install_only=True)
