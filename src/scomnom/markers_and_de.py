# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import pandas as pd

from scomnom import __version__
from . import io_utils, plot_utils
from .logging_utils import init_logging
from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
    compute_markers_celllevel,
    de_cluster_vs_rest_pseudobulk,
    resolve_group_key,
)

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal policy guard
# -----------------------------------------------------------------------------
_MIN_TOTAL_SAMPLES_FOR_PSEUDOBULK = 6


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
        if "cl_padj" in cols:
            lfc_col = "cl_logfc" if "cl_logfc" in cols else None
            if lfc_col:
                d["_abs_lfc"] = d[lfc_col].abs()
                d = d.sort_values(by=["cl_padj", "_abs_lfc"], ascending=[True, False], kind="mergesort")
                d = d.drop(columns=["_abs_lfc"], errors="ignore")
            else:
                d = d.sort_values(by=["cl_padj"], ascending=[True], kind="mergesort")
        elif "padj" in cols:
            lfc_col = "log2FoldChange" if "log2FoldChange" in cols else ("cl_logfc" if "cl_logfc" in cols else None)
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
        elif "cl_score" in cols:
            d = d.sort_values(by=["cl_score"], ascending=[False], kind="mergesort")
        d = d.head(n)
        keep_cols = [
            "gene",
            "cl_logfc",
            "cl_padj",
            "cl_pval",
            "cl_score",
            "lr_coef",
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
        raise RuntimeError(f"within-cluster: condition_key parts not in adata.obs: {missing}")

    combo_key = ".".join(parts)
    if combo_key not in adata.obs:
        comp = adata.obs[parts[0]].astype(str).map(_safe_combo_token)
        for p in parts[1:]:
            comp = comp + "." + adata.obs[p].astype(str).map(_safe_combo_token)
        adata.obs[combo_key] = comp
        LOGGER.info("within-cluster: created composite condition_key=%r from %s", combo_key, parts)

    return combo_key


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
    cond = block.get("pseudobulk_condition_within_group_multi", {})
    if not isinstance(cond, dict) or not cond:
        return {}

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for _, payload in cond.items():
        if not isinstance(payload, dict):
            continue
        if str(payload.get("condition_key", "")) != str(condition_key):
            continue
        test = payload.get("test", None)
        ref = payload.get("reference", None)
        if test is None or ref is None:
            continue
        contrast = f"{test}_vs_{ref}"
        cl = str(payload.get("group_value", "")) or "cluster"
        df = payload.get("results", None)
        if df is None:
            continue
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

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # ----------------------------
    # Resolve groupby + sample_key
    # ----------------------------
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
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
    run_cell = mode in ("cell", "both")
    run_pb_requested = mode in ("pseudobulk", "both")

    # ----------------------------
    # Predefine locals (avoid NameErrors)
    # ----------------------------
    positive_only = bool(getattr(cfg, "positive_only", True))

    markers_key: Optional[str] = None
    layer_candidates = list(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw")))
    counts_layer_used: Optional[str] = None

    n_samples_total = _n_unique_samples(adata, str(sample_key))
    run_pseudobulk = False

    pb_spec: Optional[PseudobulkSpec] = None
    pb_opts: Optional[PseudobulkDEOptions] = None

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
        LOGGER.info("cluster-vs-rest: skipping cell-level markers (run=%r).", mode)

    # ----------------------------
    # 2) Pseudobulk cluster-vs-rest DE (guarded)
    # ----------------------------
    if run_pb_requested:
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
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
                min_counts_per_lib=int(getattr(cfg, "pb_min_counts_per_lib", 0)),
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

    if run_cell and markers_key:
        io_utils.export_rank_genes_groups_tables(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
            prefix="celllevel_markers",
            tables_root=marker_cell_dir,
        )
        io_utils.export_rank_genes_groups_excel(
            adata,
            key_added=str(markers_key),
            output_dir=output_dir,
            groupby=str(groupby),
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

    if run_pseudobulk:
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=store_key,
            groupby=str(groupby),
            condition_key=None,
            tables_root=marker_pb_dir,
        )
        io_utils.export_pseudobulk_cluster_vs_rest_excel(
            adata,
            output_dir=output_dir,
            store_key=store_key,
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
        from . import de_plot_utils

        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))
        top_n_genes = int(getattr(cfg, "plot_top_n_per_cluster", 9))
        dotplot_top_n_genes = int(getattr(cfg, "plot_dotplot_top_n_genes", 15))
        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)
        de_source = str(getattr(cfg, "de_decoupler_source", "auto") or "auto").lower()
        ncols = int(getattr(cfg, "plot_umap_ncols", 3))

        if run_cell and markers_key:
            de_plot_utils.plot_marker_genes_ranksum(
                adata,
                groupby=str(groupby),
                markers_key=str(markers_key),
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                top_n_genes=top_n_genes,
                dotplot_top_n_genes=dotplot_top_n_genes,
                use_raw=use_raw,
                layer=layer,
                umap_ncols=ncols,
            )
        else:
            LOGGER.info("cluster-vs-rest: skipping cell-level marker plots (no markers computed).")

        if run_pseudobulk:
            de_plot_utils.plot_marker_genes_pseudobulk(
                adata,
                groupby=str(groupby),
                store_key=store_key,
                alpha=alpha,
                lfc_thresh=lfc_thresh,
                top_label_n=top_label_n,
                top_n_genes=top_n_genes,
                dotplot_top_n_genes=dotplot_top_n_genes,
                use_raw=use_raw,
                layer=layer,
                umap_ncols=ncols,
            )
        else:
            LOGGER.info("cluster-vs-rest: skipping pseudobulk plots (pseudobulk disabled or not requested).")

    # ----------------------------
    # Save dataset
    # ----------------------------
    if bool(getattr(cfg, "prune_uns_de", True)):
        _prune_uns_de(adata, store_key=str(getattr(cfg, "store_key", "scomnom_de")))

    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers-and-de (cluster-vs-rest).")
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
    de_cell_dir = results_dir / "tables" / f"de_tables_{run_round}" / "cell_based"
    de_pb_dir = results_dir / "tables" / f"de_tables_{run_round}" / "pseudobulk_based"

    # ----------------------------
    # Resolve groupby + sample_key
    # ----------------------------
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
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

    condition_keys = [_resolve_condition_key(adata, k) for k in condition_keys]
    for k in condition_keys:
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

    # ----------------------------
    # 1) Pseudobulk within-cluster DE (guarded)
    # ----------------------------
    ran_pseudobulk = False
    if run_pb_requested:
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
                min_pct=float(getattr(cfg, "min_pct", 0.25)),
                min_diff_pct=float(getattr(cfg, "min_diff_pct", 0.25)),
                positive_only=bool(getattr(cfg, "positive_only", True)),
                max_genes=getattr(cfg, "pb_max_genes", None),
                min_counts_per_lib=int(getattr(cfg, "pb_min_counts_per_lib", 0)),
                min_lib_pct=float(getattr(cfg, "pb_min_lib_pct", 0.0)),
                covariates=tuple(getattr(cfg, "pb_covariates", ())),
            )

            groups = pd.Index(pd.unique(adata.obs[str(groupby)].astype(str))).sort_values()

            from .de_utils import de_condition_within_group_pseudobulk_multi

            for condition_key in condition_keys:
                if str(condition_key) not in adata.obs:
                    raise RuntimeError(f"within-cluster: condition_key={condition_key!r} not found in adata.obs")

                LOGGER.info(
                    "within-cluster: pseudobulk DE across %d groups (groupby=%r) for condition_key=%r",
                    len(groups),
                    str(groupby),
                    str(condition_key),
                )

                for g in groups:
                    _ = de_condition_within_group_pseudobulk_multi(
                        adata,
                        group_value=str(g),
                        groupby=str(groupby),
                        round_id=getattr(cfg, "round_id", None),
                        condition_key=str(condition_key),
                        spec=pb_spec,
                        opts=pb_opts,
                        contrasts=condition_contrasts,
                        store_key=store_key,
                        store=True,
                        n_cpus=int(getattr(cfg, "n_jobs", 1)),
                    )

                ran_pseudobulk = True
    else:
        LOGGER.info("within-cluster: skipping pseudobulk (run=%r).", mode)

    # ----------------------------
    # 2) Cell-level within-cluster contrasts (NO pseudobulk guard)
    # ----------------------------
    ran_cell_contrast = False
    if run_cell_requested:
        from .de_utils import ContrastConditionalSpec, contrast_conditional_markers

        for condition_key in condition_keys:
            if str(condition_key) not in adata.obs:
                raise RuntimeError(f"within-cluster: condition_key={condition_key!r} not found in adata.obs")

            # What defines the levels to contrast at cell-level:
            # prefer explicit cfg.contrast_key; else condition_key.
            contrast_key = getattr(cfg, "contrast_key", None) or str(condition_key)

            cc_spec = ContrastConditionalSpec(
                contrast_key=str(contrast_key),
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

            _ = contrast_conditional_markers(
                adata,
                groupby=str(groupby),
                round_id=getattr(cfg, "round_id", None),
                spec=cc_spec,
                pb_spec=pb_spec,  # may be None; function should tolerate (cell-only mode)
                store_key=store_key,
                store=True,
            )

            # Persist per-contrast_key results in a multi-store block
            if store_key in adata.uns and isinstance(adata.uns.get(store_key), dict):
                cc_block = adata.uns[store_key].get("contrast_conditional", None)
                if isinstance(cc_block, dict):
                    from copy import deepcopy
                    adata.uns[store_key].setdefault("contrast_conditional_multi", {})
                    adata.uns[store_key]["contrast_conditional_multi"][str(contrast_key)] = deepcopy(cc_block)

            # tables for contrast-conditional markers
            io_utils.export_contrast_conditional_markers_tables(
                adata,
                output_dir=output_dir,
                store_key=store_key,
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
        LOGGER.info("within-cluster: skipping cell-level within-cluster contrasts (run=%r).", mode)

    # ----------------------------
    # 3) DE-based decoupler (pathways/TF from DE stats)
    # ----------------------------
    de_decoupler_ran = False
    de_source = str(getattr(cfg, "de_decoupler_source", "auto") or "auto").lower()
    if de_source not in ("auto", "all", "pseudobulk", "cell", "none"):
        raise RuntimeError(f"within-cluster: invalid de_decoupler_source={de_source!r}")

    if de_source != "none":
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
                    sources = [("pseudobulk", pb_tables)]
                elif cell_tables:
                    sources = [("cell", cell_tables)]
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
                        fallback_cols=("log2FoldChange", "cl_logfc", "lr_coef"),
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
    if ran_pseudobulk:
        covariates = tuple(getattr(cfg, "pb_covariates", ()))

        for condition_key in condition_keys:
            io_utils.export_pseudobulk_de_tables(
                adata,
                output_dir=output_dir,
                store_key=store_key,
                groupby=str(groupby),
                condition_key=str(condition_key),
                tables_root=de_pb_dir,
            )
            io_utils.export_pseudobulk_condition_within_cluster_excel(
                adata,
                output_dir=output_dir,
                store_key=store_key,
                condition_key=str(condition_key),
                tables_root=de_pb_dir,
            )
            design_terms = ["sample", *covariates, str(condition_key)]

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

        # Condition plots only if pseudobulk ran
        if ran_pseudobulk:
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
                )
        else:
            LOGGER.info("within-cluster: skipping condition plots (pseudobulk not run).")

        if ran_cell_contrast:
            for condition_key in condition_keys:
                de_plot_utils.plot_contrast_conditional_markers_multi(
                    adata,
                    groupby=str(groupby),
                    contrast_key=str(getattr(cfg, "contrast_key", None) or condition_key),
                    store_key=store_key,
                    alpha=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    top_n_genes=top_n_genes,
                    dotplot_top_n_genes=dotplot_top_n_genes,
                    use_raw=use_raw,
                    layer=layer,
                )

        # DE-based decoupler plots (if available)
        if de_source != "none":
            de_block = adata.uns.get(store_key, {}).get("de_decoupler", {})
            if isinstance(de_block, dict) and de_block:
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
                                base = base / "pseudobulk_DE" / str(condition_key) / str(contrast)

                            for net_name, net_payload in nets.items():
                                de_plot_utils.plot_de_decoupler_payload(
                                    net_payload,
                                    net_name=str(net_name),
                                    figdir=base,
                                    heatmap_top_k=int(getattr(cfg, "plot_max_genes_total", 80)),
                                    bar_top_n=int(top_n_genes),
                                    dotplot_top_k=int(dotplot_top_n_genes),
                                    title_prefix=f"{condition_key} {contrast}",
                                )

    # ----------------------------
    # Save dataset
    # ----------------------------
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
