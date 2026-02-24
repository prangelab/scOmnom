# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .logging_utils import init_logging
from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
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

LOGGER = logging.getLogger(__name__)
_SCCODA_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# Internal policy guard
# -----------------------------------------------------------------------------
_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK = 6
_MIN_SAMPLES_PER_LEVEL_COMPOSITION = 2


def _match_cluster_color(label: str, color_map: dict[str, str]) -> Optional[str]:
    lab = str(label)
    if lab in color_map:
        return color_map[lab]
    token = plot_utils._extract_cnn_token(lab)
    if token in color_map:
        return color_map[token]
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
                        activity = net_payload.get("activity", None)
                        if isinstance(activity, pd.DataFrame):
                            net_payload["activity_top"] = _top_activity(activity, decoupler_top_n)
                        net_payload.pop("activity", None)
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

    adata.obs[key] = pd.Categorical(display.astype(str))

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
        res = pb.get("results", {}) if isinstance(pb, dict) else {}
        has_pb = isinstance(res, dict) and bool(res)

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
    run_round = plot_utils.get_run_round_tag("markers")
    marker_cell_dir = results_dir / "tables" / f"marker_tables_{run_round}" / "cell_based"
    marker_pb_dir = results_dir / "tables" / f"marker_tables_{run_round}" / "pseudobulk_based"

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
    run_round = plot_utils.get_run_round_tag("DA")

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
                        n_seeds=int(getattr(cfg, "composition_graph_n_seeds", 1000)),
                        k_ref=int(getattr(cfg, "composition_graph_k_ref", 50)),
                        max_k=int(getattr(cfg, "composition_graph_max_k", 200)),
                        min_size=int(getattr(cfg, "composition_graph_min_size", 20)),
                        random_state=int(getattr(cfg, "composition_graph_random_state", 42)),
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
        results_dir = output_dir / "tables" / f"DA_tables_{run_round}" / cond_tag
        results_dir.mkdir(parents=True, exist_ok=True)
        fig_subdir = Path("DA") / cond_tag

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

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            if graph_meta_global is not None and not graph_meta_global.empty:
                try:
                    plot_utils.plot_graphda_summaries(
                        results_by_method.get("graph", pd.DataFrame()),
                        graph_meta_global,
                        fig_subdir,
                        alpha=alpha,
                    )
                except Exception:
                    LOGGER.exception("composition: failed to plot GraphDA summary")

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            for method in methods:
                df = results_by_method.get(method, pd.DataFrame())
                if df is None or df.empty:
                    continue
                try:
                    if method in ("glm", "clr"):
                        plot_utils.plot_composition_volcano(method, df, fig_subdir)
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
                        color_map = {str(label): str(color) for label, color in zip(labels, colors)}
                        plot_utils.plot_sccoda_effects_top(df, color_map, fig_subdir)
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
                    f"glm_min_samples_per_level={_MIN_GLM_SAMPLES_PER_LEVEL}",
                    f"glm_min_levels=3",
                ],
            )

        if write_figures and bool(getattr(cfg, "make_figures", True)):
            try:
                global_df = results_by_method.get(primary_method)
                if isinstance(global_df, pd.DataFrame) and not global_df.empty:
                    plotted = False
                    if "effect" in global_df.columns:
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
                            plot_utils.plot_composition_effects_global(vals.to_numpy(), labels, colors, fig_subdir)
                            plotted = True
                    elif primary_method == "sccoda" and "Final Parameter" in global_df.columns:
                        vals = pd.to_numeric(global_df["Final Parameter"], errors="coerce")
                        labels = global_df.index.astype(str)
                        if not vals.isna().all():
                            labels = pd.Index(labels).astype(str)
                            colors = _resolve_cluster_colors(
                                adata,
                                cluster_key=cluster_key,
                                labels=labels,
                                round_id=getattr(cfg, "round_id", None),
                            )
                            plot_utils.plot_composition_effects_global(vals.to_numpy(), labels, colors, fig_subdir)
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
                            plot_utils.plot_composition_effects_global(vals.values, labels, colors, fig_subdir)
                            plotted = True
                    if plotted:
                        LOGGER.info("Saved plot: %s/%s", fig_subdir, "composition_effects_global")
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
                plot_utils.plot_composition_stacks(
                    counts,
                    metadata,
                    condition_key=str(condition_key),
                    cluster_order=cluster_order,
                    colors=colors,
                    figdir=fig_subdir,
                    consensus=consensus if isinstance(consensus, pd.DataFrame) else None,
                    alpha=float(alpha),
                )
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
    run_round = plot_utils.get_run_round_tag("DE")
    de_cell_dir = results_dir / "tables" / f"DE_tables_{run_round}" / "cell_based"
    de_pb_dir = results_dir / "tables" / f"DE_tables_{run_round}" / "pseudobulk_based"

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

            groups = pd.Index(pd.unique(adata.obs[str(groupby)].astype(str))).sort_values()

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
            max_workers = min(int(total_cpus), max(1, total))
            LOGGER.info(
                "within-cluster: pseudobulk parallel run (tasks=%d, workers=%d, total_cpus=%d).",
                int(total),
                int(max_workers),
                int(total_cpus),
            )

            entries: list[tuple[str, dict]] = []

            def _run_task(task: dict[str, str]):
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
                        store_key=None,
                        store=False,
                        n_cpus=1,
                    )
                    return "interaction", str(task["cond_key"]), str(task["group"]), res, meta
                res = de_condition_within_group_pseudobulk_multi(
                    adata,
                    group_value=str(task["group"]),
                    groupby=str(groupby),
                    round_id=getattr(cfg, "round_id", None),
                    condition_key=str(task["cond_key"]),
                    spec=pb_spec,
                    opts=pb_opts,
                    contrasts=[f"{task['A']}_vs_{task['B']}"],
                    store_key=None,
                    store=False,
                    n_cpus=1,
                )
                return "pair", str(task["cond_key"]), str(task["group"]), res, None

            done = 0
            t0 = time.perf_counter()
            if tasks and max_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

                heartbeat_s = 60.0
                slow_task_warn_s = 30 * 60.0
                next_heartbeat = time.perf_counter() + heartbeat_s

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_run_task, task): task for task in tasks}
                    pending = set(futs.keys())
                    start_by_fut = {f: time.perf_counter() for f in pending}
                    while pending:
                        now = time.perf_counter()
                        done_set, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                        if not done_set:
                            if now >= next_heartbeat:
                                if pending:
                                    longest_fut = max(pending, key=lambda f: now - start_by_fut.get(f, now))
                                    longest_age = now - start_by_fut.get(longest_fut, now)
                                    longest_task = futs.get(longest_fut, {})
                                    LOGGER.info(
                                        "within-cluster: pseudobulk longest-running task (age=%.1fs, kind=%s, group=%s, condition_key=%s, A=%s, B=%s).",
                                        float(longest_age),
                                        str(longest_task.get("kind")),
                                        str(longest_task.get("group")),
                                        str(longest_task.get("cond_key")),
                                        str(longest_task.get("A", "")),
                                        str(longest_task.get("B", "")),
                                    )
                                    if float(longest_age) >= float(slow_task_warn_s):
                                        LOGGER.warning(
                                            "within-cluster: pseudobulk slow task (age=%.1fs, kind=%s, group=%s, condition_key=%s, A=%s, B=%s).",
                                            float(longest_age),
                                            str(longest_task.get("kind")),
                                            str(longest_task.get("group")),
                                            str(longest_task.get("cond_key")),
                                            str(longest_task.get("A", "")),
                                            str(longest_task.get("B", "")),
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
                            kind, cond_key, gval, res, meta = fut.result()
                            task_info = futs.get(fut, {})
                            t_elapsed = now - start_by_fut.get(fut, now)
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
                                "within-cluster: pseudobulk task finished (kind=%s, group=%s, condition_key=%s, A=%s, B=%s, task_s=%.1f).",
                                str(task_info.get("kind")),
                                str(task_info.get("group")),
                                str(task_info.get("cond_key")),
                                str(task_info.get("A", "")),
                                str(task_info.get("B", "")),
                                float(t_elapsed),
                            )
            else:
                for task in tasks:
                    kind, cond_key, gval, res, meta = _run_task(task)
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
        results_by_key, summaries_by_key = contrast_conditional_markers_multi(
            adata,
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            specs=specs,
            pb_spec=pb_spec,
            n_jobs=total_cpus,
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
                    f"contrast_key={contrast_key}",
                    f"contrast_methods={tuple(getattr(cfg, 'contrast_methods', ())) }",
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
                        fallback_cols=("log2FoldChange", "cell_wilcoxon_logfc", "cell_logreg_coef"),
                    )
                    if stats is None or stats.empty:
                        continue

                    input_label = f"{source}:{condition_key}:{contrast}:{stat_col}"
                    payloads = {}

                    msigdb_payload = _run_msigdb_from_stats(stats, cfg, input_label=input_label)
                    if msigdb_payload is not None:
                        payloads["msigdb"] = msigdb_payload

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
                    f"sample_key={sample_key}",
                    f"counts_layer={counts_layer_used}",
                    f"min_cells_per_sample_group={getattr(cfg, 'min_cells_condition', None)}",
                    f"min_samples_per_level={getattr(cfg, 'min_samples_per_level', None)}",
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

        try:
            if condition_keys:
                LOGGER.info("within-cluster: plotting UMAPs for condition keys...")
                de_plot_utils.plot_condition_umaps(
                    adata,
                    groupby=str(groupby),
                    condition_keys=condition_keys,
                )

            # Condition plots only if pseudobulk ran
            if ran_pseudobulk:
                LOGGER.info("within-cluster: plotting pseudobulk condition figures...")
                plot_ann_keys = list(getattr(cfg, "plot_sample_annotation_keys", ()) or condition_keys)
                for condition_key in condition_keys:
                    de_plot_utils.plot_condition_within_cluster_all(
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
            else:
                LOGGER.info("within-cluster: skipping condition plots (pseudobulk not run).")

            if ran_cell_contrast:
                LOGGER.info("within-cluster: plotting cell-level contrast figures...")
                plot_ann_keys = list(getattr(cfg, "plot_sample_annotation_keys", ()) or condition_keys)
                for condition_key in condition_keys:
                    de_plot_utils.plot_contrast_conditional_markers_multi(
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

            # DE-based decoupler plots (if available)
            if de_source != "none":
                de_block = adata.uns.get(store_key, {}).get("de_decoupler", {})
                if isinstance(de_block, dict) and de_block:
                    LOGGER.info("within-cluster: plotting DE-decoupler figures...")
                    pb_cond = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
                    pb_cond_multi = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group_multi", {})
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

                                base = Path("DE")
                                if str(source) == "cell":
                                    base = base / "cell_level_DE" / str(condition_key) / str(contrast)
                                else:
                                    if str(contrast) == "interaction" or "^" in str(condition_key):
                                        base = (
                                            base
                                            / "pseudobulk_DE"
                                            / f"{condition_key}__interaction"
                                            / "interaction"
                                        )
                                    else:
                                        base = base / "pseudobulk_DE" / str(condition_key) / str(contrast)

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
                                    de_plot_utils.plot_de_decoupler_payload(
                                        net_payload,
                                        net_name=str(net_name),
                                        figdir=base,
                                        heatmap_top_k=int(getattr(cfg, "plot_max_genes_total", 80)),
                                        bar_top_n=int(top_n_genes),
                                        bar_top_n_up=getattr(cfg, "decoupler_bar_top_n_up", None),
                                        bar_top_n_down=getattr(cfg, "decoupler_bar_top_n_down", None),
                                        bar_split_signed=bool(getattr(cfg, "decoupler_bar_split_signed", True)),
                                        dotplot_top_k=int(dotplot_top_n_genes),
                                        title_prefix=f"{condition_key} {contrast}",
                                        pos_label=pos_label,
                                        neg_label=neg_label,
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
