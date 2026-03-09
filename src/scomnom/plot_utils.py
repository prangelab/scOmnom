from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Sequence, Mapping, Iterable, List, Any
import ast

import logging
from sklearn.metrics import silhouette_samples

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import re

import textwrap
import seaborn as sns

LOGGER = logging.getLogger(__name__)


@dataclass
class PlotArtifact:
    stem: str
    figdir: Path
    fig: Any = None
    savefig_kwargs: dict | None = None


_ACTIVE_PLOT_CAPTURE: list[PlotArtifact] | None = None
_ARTIFACT_COLLECTION_STACK: list[list[PlotArtifact]] = []
_KEEP_PLOTS_OPEN_DEPTH: int = 0


@contextmanager
def capture_plot_artifacts():
    global _ACTIVE_PLOT_CAPTURE
    prev = _ACTIVE_PLOT_CAPTURE
    bucket: list[PlotArtifact] = []
    _ACTIVE_PLOT_CAPTURE = bucket
    try:
        yield bucket
    finally:
        _ACTIVE_PLOT_CAPTURE = prev


@contextmanager
def keep_plots_open():
    global _KEEP_PLOTS_OPEN_DEPTH
    _KEEP_PLOTS_OPEN_DEPTH += 1
    try:
        yield
    finally:
        _KEEP_PLOTS_OPEN_DEPTH = max(0, _KEEP_PLOTS_OPEN_DEPTH - 1)


def close_plot(fig=None) -> None:
    if _KEEP_PLOTS_OPEN_DEPTH > 0:
        return
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def persist_plot_artifacts(artifacts: Iterable[PlotArtifact]) -> None:
    for artifact in artifacts:
        save_multi(
            stem=str(artifact.stem),
            figdir=Path(artifact.figdir),
            fig=artifact.fig,
            savefig_kwargs=artifact.savefig_kwargs,
        )


def collect_plot_artifacts(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        _ARTIFACT_COLLECTION_STACK.append([])
        result = None
        try:
            result = func(*args, **kwargs)
            bucket = list(_ARTIFACT_COLLECTION_STACK[-1])
        finally:
            _ARTIFACT_COLLECTION_STACK.pop()
        if isinstance(result, PlotArtifact):
            return [result]
        if isinstance(result, list) and all(isinstance(x, PlotArtifact) for x in result):
            return result
        return bucket

    return _wrapped


def record_plot_artifact(stem: str, figdir: Path, fig=None, *, savefig_kwargs: dict | None = None) -> PlotArtifact:
    if fig is None:
        try:
            fig = plt.gcf()
        except Exception:
            fig = None
    artifact = PlotArtifact(
        stem=stem,
        figdir=Path(figdir),
        fig=fig,
        savefig_kwargs=savefig_kwargs,
    )
    if _ACTIVE_PLOT_CAPTURE is not None:
        _ACTIVE_PLOT_CAPTURE.append(artifact)
    if _ARTIFACT_COLLECTION_STACK:
        for bucket in _ARTIFACT_COLLECTION_STACK:
            bucket.append(artifact)
    return artifact


def _extract_cnn_token(label: str) -> str:
    s = str(label or "")
    m = re.search(r"\b(C\d+)\b", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(C\d+)", s)
    if m2:
        return m2.group(1)
    return s.split()[0][:8] if s.strip() else "C?"


def _normalize_color(value) -> str | tuple:
    if isinstance(value, str):
        s = value.strip()
        # Accept legacy stringified tuples from prior color-map serialization.
        if s.startswith("(") and s.endswith(")"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return tuple(float(x) for x in parsed)
            except Exception:
                pass
        return s
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return tuple(float(x) for x in value)
        except Exception:
            return str(value)
    return str(value)


def _cluster_color_map(adata, label_key: str) -> dict[str, str | tuple]:
    palette = adata.uns.get(f"{label_key}_colors", None)
    if not palette:
        return {}
    if isinstance(palette, dict):
        return {str(k): _normalize_color(v) for k, v in palette.items()}
    if hasattr(palette, "to_dict"):
        try:
            return {str(k): _normalize_color(v) for k, v in palette.to_dict().items()}
        except Exception:
            pass
    try:
        cats = list(adata.obs[label_key].cat.categories)
    except Exception:
        cats = list(pd.Series(adata.obs[label_key].astype(str)).unique())
    return {str(cat): _normalize_color(palette[i % len(palette)]) for i, cat in enumerate(cats)}


def _add_outside_legend(
    ax,
    labels: list[str],
    colors: list[str],
    title: str,
    *,
    max_rows: int = 12,
    face: str = "full",
) -> None:
    if not labels:
        return
    ncol = max(1, math.ceil(len(labels) / max_rows))
    fig = ax.figure
    fig.subplots_adjust(right=0.80)
    if face == "none":
        handles = [
            mpl.patches.Patch(facecolor="none", edgecolor=c, linewidth=1.2, label=l)
            for l, c in zip(labels, colors)
        ]
    else:
        handles = [
            mpl.patches.Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=l)
            for l, c in zip(labels, colors)
        ]
    ax.legend(
        handles=handles,
        title=title,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        ncol=ncol,
    )


@collect_plot_artifacts
def plot_graphda_summaries(
    graph_df: pd.DataFrame,
    graph_meta: pd.DataFrame | None,
    figdir: Path,
    *,
    alpha: float = 0.05,
    all_clusters: Sequence[str] | None = None,
) -> None:
    """
    Plot GraphDA summary panels and save via record_plot_artifact().
    """
    if graph_df is None or graph_df.empty or "effect" not in graph_df.columns:
        return

    gdf = graph_df.copy()
    if graph_meta is not None and not graph_meta.empty:
        if "cluster_label" not in gdf.columns and "cluster_label" in graph_meta.columns:
            graph_meta_indexed = graph_meta.set_index("neighborhood")
            gdf = gdf.merge(graph_meta_indexed[["cluster_label"]], left_on="cluster", right_index=True, how="left")
    if "cluster_label" not in gdf.columns:
        gdf["cluster_label"] = "NA"
    if "fdr" in gdf.columns:
        gdf = gdf.sort_values("fdr")
    elif "pval" in gdf.columns:
        gdf = gdf.sort_values("pval")
    gdf = gdf.assign(_abs_effect=pd.to_numeric(gdf["effect"], errors="coerce").abs())
    gdf = gdf.sort_values("_abs_effect", ascending=False)
    top = gdf.head(25)

    fig, ax = plt.subplots(figsize=(8, 4))
    if "cluster_label" in top.columns:
        labels = top["cluster_label"].astype(str)
    elif "cluster" in top.columns:
        labels = top["cluster"].astype(str)
    else:
        labels = top.index.astype(str)
    labels = pd.Index(labels).astype(str)
    colors = ["#6b7aa1"] * len(labels)
    x = np.arange(len(labels))
    y = pd.to_numeric(top["effect"], errors="coerce").to_numpy()
    ax.bar(x, y, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Effect")
    ax.set_title("GraphDA top neighborhoods")
    ax.tick_params(axis="x", labelrotation=45)
    if "level_ref" in top.columns and "level_test" in top.columns:
        ref_label = str(top["level_ref"].iloc[0])
        test_label = str(top["level_test"].iloc[0])
        ax.text(
            0.98,
            0.02,
            f"+ = {test_label}, - = {ref_label}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    ax.grid(False)
    if "fdr" in top.columns:
        sig_top = pd.to_numeric(top["fdr"], errors="coerce") <= float(alpha)
    elif "pval" in top.columns:
        sig_top = pd.to_numeric(top["pval"], errors="coerce") <= float(alpha)
    else:
        sig_top = pd.Series(False, index=top.index)
    for xi, yi, is_sig in zip(x, y, sig_top.to_numpy()):
        if bool(is_sig):
            va = "bottom" if yi >= 0 else "top"
            yoff = 0.04 if yi >= 0 else -0.04
            ax.text(xi, yi + yoff, "*", ha="center", va=va, fontsize=10, color="#d62728")
    record_plot_artifact("graphda_top_neighborhoods", figdir, fig=fig)
    close_plot(fig)

    cluster_universe: list[str] = []
    if all_clusters is not None:
        cluster_universe = [str(c) for c in pd.Index(all_clusters).astype(str).tolist()]
    if "cluster_label" in gdf.columns:
        gdf["cluster_label"] = gdf["cluster_label"].astype(str)
        idx = gdf.groupby("cluster_label")["_abs_effect"].idxmax()
        top_by_cluster = gdf.loc[idx].copy().sort_values("cluster_label")
    else:
        top_by_cluster = pd.DataFrame()
    if cluster_universe:
        if top_by_cluster.empty:
            top_by_cluster = pd.DataFrame({"cluster_label": cluster_universe, "effect": 0.0})
            top_by_cluster["has_neighborhood"] = False
        else:
            top_by_cluster = top_by_cluster.set_index("cluster_label")
            top_by_cluster = top_by_cluster.reindex(cluster_universe)
            top_by_cluster["has_neighborhood"] = ~top_by_cluster["effect"].isna()
            top_by_cluster["effect"] = pd.to_numeric(top_by_cluster["effect"], errors="coerce").fillna(0.0)
            top_by_cluster = top_by_cluster.reset_index().rename(columns={"index": "cluster_label"})
    if not top_by_cluster.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top_by_cluster))))
        labels = top_by_cluster["cluster_label"].astype(str)
        x = np.arange(len(labels))
        y = pd.to_numeric(top_by_cluster["effect"], errors="coerce").to_numpy()
        has_nh = top_by_cluster.get("has_neighborhood", pd.Series(True, index=top_by_cluster.index)).astype(bool).to_numpy()
        bar_colors = ["#6b7aa1" if h else "#c8cedd" for h in has_nh]
        ax.bar(x, y, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Effect")
        ax.set_title("GraphDA top neighborhood per cluster")
        if "level_ref" in top_by_cluster.columns and "level_test" in top_by_cluster.columns:
            ref_label = str(top_by_cluster["level_ref"].iloc[0])
            test_label = str(top_by_cluster["level_test"].iloc[0])
            ax.text(
                0.98,
                0.02,
                f"+ = {test_label}, - = {ref_label}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
        ax.grid(False)
        if "fdr" in top_by_cluster.columns:
            sig_by = pd.to_numeric(top_by_cluster["fdr"], errors="coerce") <= float(alpha)
        elif "pval" in top_by_cluster.columns:
            sig_by = pd.to_numeric(top_by_cluster["pval"], errors="coerce") <= float(alpha)
        else:
            sig_by = pd.Series(False, index=top_by_cluster.index)
        for xi, yi, is_sig, has_nh_i in zip(x, y, sig_by.to_numpy(), has_nh):
            if bool(is_sig) and bool(has_nh_i):
                va = "bottom" if yi >= 0 else "top"
                yoff = 0.04 if yi >= 0 else -0.04
                ax.text(xi, yi + yoff, "*", ha="center", va=va, fontsize=10, color="#d62728")
        legend_handles = [
            mpl.patches.Patch(facecolor="#6b7aa1", edgecolor="black", linewidth=0.4, label="Has neighborhoods"),
            mpl.patches.Patch(facecolor="#c8cedd", edgecolor="black", linewidth=0.4, label="No neighborhoods"),
            mpl.lines.Line2D([], [], marker="None", linestyle="None", label="* significant"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
        plt.tight_layout()
        record_plot_artifact("graphda_top_by_cluster", figdir, fig=fig)
        close_plot(fig)

    labels = gdf["cluster_label"].astype(str)
    if cluster_universe:
        order = pd.Index(cluster_universe)
    else:
        order = pd.Index(pd.unique(labels))
    y_pos = {lab: i for i, lab in enumerate(order)}
    x = pd.to_numeric(gdf["effect"], errors="coerce")
    y = labels.map(y_pos).astype(float)
    jitter = (np.random.default_rng(0).random(len(y)) - 0.5) * 0.4
    yj = y + jitter

    if "fdr" in gdf.columns:
        sig = gdf["fdr"].astype(float) <= float(alpha)
    elif "pval" in gdf.columns:
        sig = gdf["pval"].astype(float) <= float(alpha)
    else:
        sig = pd.Series(False, index=gdf.index)

    sig_color = "#d62728"
    ns_color = "#b0b0b0"
    colors = [sig_color if s else ns_color for s in sig]
    alphas = [0.9 if s else 0.35 for s in sig]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(order))))
    ax.scatter(x, yj, s=16, c=colors, alpha=alphas, edgecolors="none")
    sig_legend_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#d62728", label="Significant", markersize=5),
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#b0b0b0", label="Not significant", markersize=5),
    ]
    if cluster_universe:
        present = set(labels.tolist())
        missing = [c for c in cluster_universe if c not in present]
        if missing:
            xm = np.zeros(len(missing), dtype=float)
            ym = np.array([y_pos[c] for c in missing], dtype=float)
            ax.scatter(
                xm,
                ym,
                s=18,
                c="#9aa0aa",
                alpha=0.9,
                marker="x",
                linewidths=0.9,
                label="No tested neighborhoods",
            )
            sig_legend_handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="None",
                    color="#9aa0aa",
                    label="No tested neighborhoods",
                    markersize=6,
                )
            )
    ax.legend(handles=sig_legend_handles, loc="lower right", frameon=True, fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Effect (log2 fold-change)")
    ax.set_ylabel("Cluster")
    ax.set_title("GraphDA effects by cluster")
    ax.grid(False)
    record_plot_artifact("graphda_effects_by_cluster", figdir, fig=fig)
    close_plot(fig)


@collect_plot_artifacts
def plot_graphda_diagnostics(
    graph_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    figdir: Path,
    *,
    alpha: float = 0.05,
) -> None:
    """
    Plot GraphDA diagnostics and save via record_plot_artifact().
    """
    if graph_df is None or graph_df.empty:
        return

    gdf = graph_df.copy()
    p = pd.to_numeric(gdf.get("pval", np.nan), errors="coerce")
    q = pd.to_numeric(gdf.get("fdr", np.nan), errors="coerce")
    sig = (q <= float(alpha)) if np.isfinite(q).any() else (p <= float(alpha))

    # QC 1: p-value vs FDR to inspect multiplicity impact.
    mask = np.isfinite(p) & np.isfinite(q)
    if mask.any():
        x = -np.log10(np.clip(p[mask].to_numpy(dtype=float), 1e-300, 1.0))
        y = -np.log10(np.clip(q[mask].to_numpy(dtype=float), 1e-300, 1.0))
        sig_m = np.asarray(sig[mask].fillna(False), dtype=bool)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            x,
            y,
            s=14,
            c=np.where(sig_m, "#d62728", "#9e9e9e"),
            alpha=np.where(sig_m, 0.9, 0.5),
            edgecolors="none",
        )
        thr = -np.log10(float(alpha))
        ax.axhline(thr, color="#d62728", linestyle=":", linewidth=1.0)
        ax.set_xlabel("-log10(p-value)")
        ax.set_ylabel("-log10(FDR)")
        ax.set_title("GraphDA QC: p-value vs FDR")
        handles = [
            mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#d62728", label="Significant", markersize=5),
            mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#9e9e9e", label="Not significant", markersize=5),
            mpl.lines.Line2D([], [], linestyle=":", color="#d62728", label=f"FDR={alpha:g}"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
        ax.grid(False)
        record_plot_artifact("graphda_qc_pval_vs_fdr", figdir, fig=fig)
        close_plot(fig)

    # QC 2: per-cluster tested vs significant counts.
    if diag_df is not None and not diag_df.empty and "cluster" in diag_df.columns:
        d = diag_df.copy()
        d["cluster"] = d["cluster"].astype(str)
        d["n_tested_neighborhoods"] = pd.to_numeric(d.get("n_tested_neighborhoods", 0), errors="coerce").fillna(0.0)
        d["n_sig_fdr"] = pd.to_numeric(d.get("n_sig_fdr", 0), errors="coerce").fillna(0.0)
        d = d.sort_values(["n_tested_neighborhoods", "cluster"], ascending=[False, True])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(d))))
        y = np.arange(len(d))
        ax.barh(y, d["n_tested_neighborhoods"].to_numpy(), color="#6b7aa1", alpha=0.75, label="Tested neighborhoods")
        ax.barh(y, d["n_sig_fdr"].to_numpy(), color="#d62728", alpha=0.9, label="Significant (FDR)")
        ax.set_yticks(y)
        ax.set_yticklabels(d["cluster"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("GraphDA QC: tested vs significant by cluster")
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(False)
        record_plot_artifact("graphda_qc_cluster_power", figdir, fig=fig)
        close_plot(fig)


@collect_plot_artifacts
def plot_composition_volcano(method: str, df: pd.DataFrame, figdir: Path, *, alpha: float = 0.05) -> None:
    """
    Plot a volcano for a composition method and save via record_plot_artifact().
    """
    if df is None or df.empty:
        return
    if "effect" in df.columns:
        x = pd.to_numeric(df["effect"], errors="coerce")
    else:
        return
    if "pval" in df.columns:
        y_raw = pd.to_numeric(df["pval"], errors="coerce")
        y_source = "pval"
    elif "fdr" in df.columns:
        y_raw = pd.to_numeric(df["fdr"], errors="coerce")
        y_source = "FDR"
    else:
        return
    y = -np.log10(y_raw)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sig = pd.Series(False, index=df.index)
    if "fdr" in df.columns:
        sig = pd.to_numeric(df["fdr"], errors="coerce") <= float(alpha)
        sig_label = f"FDR<={alpha:g}"
    elif "pval" in df.columns:
        sig = pd.to_numeric(df["pval"], errors="coerce") <= float(alpha)
        sig_label = f"pval<={alpha:g}"
    else:
        sig_label = "significant"
    sig = sig[mask]
    pt_colors = np.where(sig.to_numpy(), "#d62728", "#7a7a7a")
    pt_alpha = np.where(sig.to_numpy(), 0.9, 0.65)
    ax.scatter(x, y, s=16, c=pt_colors, alpha=pt_alpha, edgecolors="none")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    y_thr = -np.log10(float(alpha))
    if np.isfinite(y_thr):
        ax.axhline(y_thr, color="#d62728", linestyle=":", linewidth=1)
    ax.set_xlabel("Effect (log2 scale)")
    ax.set_ylabel(f"-log10({y_source})")
    ax.set_title(f"{str(method).upper()} volcano")
    legend_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#d62728", label=f"Significant ({sig_label})", markersize=5),
        mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#7a7a7a", label="Not significant", markersize=5),
        mpl.lines.Line2D([], [], color="#d62728", linestyle=":", label=f"{y_source}={alpha:g}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    ax.grid(False)
    record_plot_artifact(f"{str(method)}_volcano", figdir, fig=fig)
    close_plot(fig)


@collect_plot_artifacts
def plot_sccoda_effects_top(
    df: pd.DataFrame,
    color_map: Mapping[str, Any],
    figdir: Path,
    *,
    alpha: float = 0.05,
) -> None:
    """
    Plot top scCODA effects and save via record_plot_artifact().
    """
    if df is None or df.empty:
        return
    top = df.copy()
    if "fdr" in top.columns:
        top = top.sort_values("fdr")
    top = top.head(25)
    if top.empty:
        return

    if "effect" in top.columns:
        eff = pd.to_numeric(top["effect"], errors="coerce")
    elif "Final Parameter" in top.columns:
        eff = pd.to_numeric(top["Final Parameter"], errors="coerce")
    else:
        return

    if "ci_low" in top.columns and "ci_high" in top.columns:
        ci_low = pd.to_numeric(top["ci_low"], errors="coerce")
        ci_high = pd.to_numeric(top["ci_high"], errors="coerce")
    elif "HDI 3%" in top.columns and "HDI 97%" in top.columns:
        ci_low = pd.to_numeric(top["HDI 3%"], errors="coerce")
        ci_high = pd.to_numeric(top["HDI 97%"], errors="coerce")
    elif "SD" in top.columns:
        sd = pd.to_numeric(top["SD"], errors="coerce")
        ci_low = eff - sd
        ci_high = eff + sd
    else:
        ci_low = None
        ci_high = None

    if "cluster" in top.columns:
        labels = top["cluster"].astype(str)
    else:
        labels = top.index.astype(str)
    labels = pd.Index(labels).astype(str)
    colors = [_normalize_color(color_map.get(str(label), "#6b7aa1")) for label in labels]
    if "fdr" in top.columns:
        sig = pd.to_numeric(top["fdr"], errors="coerce") <= float(alpha)
    elif "pval" in top.columns:
        sig = pd.to_numeric(top["pval"], errors="coerce") <= float(alpha)
    elif "inclusion_prob" in top.columns:
        sig = pd.to_numeric(top["inclusion_prob"], errors="coerce") >= float(1.0 - alpha)
    elif "Inclusion probability" in top.columns:
        sig = pd.to_numeric(top["Inclusion probability"], errors="coerce") >= float(1.0 - alpha)
    else:
        sig = pd.Series(False, index=top.index)
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(labels))))
    if ci_low is not None and ci_high is not None:
        ax.hlines(y_pos, ci_low, ci_high, color="#999999", linewidth=1.5)
    ax.scatter(eff, y_pos, s=30, color=colors, zorder=3)
    for x_i, y_i, is_sig in zip(eff.to_numpy(), y_pos, sig.to_numpy()):
        if bool(is_sig):
            ax.text(x_i, y_i, " *", va="center", ha="left", fontsize=10, color="#d62728")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect")
    ax.set_title("scCODA effects (top)")
    legend_handles = [mpl.lines.Line2D([], [], marker="None", linestyle="None", label=f"* significant (alpha={alpha:g})")]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)
    ax.grid(False)
    plt.tight_layout()
    record_plot_artifact("sccoda_effects_top", figdir, fig=fig)
    close_plot(fig)


@collect_plot_artifacts
def plot_composition_effects_global(
    values: np.ndarray,
    labels: Sequence[str],
    colors: Sequence[str],
    figdir: Path,
    *,
    plot_name: str = "composition_effects_global",
    title: str = "Composition effects (global)",
    sig_mask: Sequence[bool] | None = None,
    alpha: float = 0.05,
) -> None:
    """
    Plot global composition effects and save via record_plot_artifact().
    """
    labels = pd.Index(labels).astype(str)
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    if vals.isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    vals_np = vals.to_numpy()
    ax.bar(x, vals_np, color=list(colors))
    if sig_mask is not None:
        sig_arr = np.array([bool(v) for v in sig_mask], dtype=bool)
        if sig_arr.size != vals_np.size:
            sig_arr = np.resize(sig_arr, vals_np.size)
        for xi, yi, is_sig in zip(x, vals_np, sig_arr):
            if is_sig and np.isfinite(yi):
                va = "bottom" if yi >= 0 else "top"
                yoff = 0.02 * max(1.0, float(np.nanmax(np.abs(vals_np))))
                ax.text(xi, yi + (yoff if yi >= 0 else -yoff), "*", ha="center", va=va, fontsize=10, color="#d62728")
        if sig_arr.any():
            legend_handles = [mpl.lines.Line2D([], [], marker="None", linestyle="None", label=f"* significant (alpha={alpha:g})")]
            ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    if np.nanmax(np.abs(vals_np)) == 0:
        ax.scatter(x, vals_np, color=list(colors), s=30, zorder=3)
        ax.set_ylim(-0.05, 0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Effect")
    ax.set_title(str(title))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(False)
    record_plot_artifact(str(plot_name), figdir, fig=fig)
    close_plot(fig)


@collect_plot_artifacts
def plot_composition_stacks(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_key: str,
    cluster_order: Sequence[str],
    colors: Sequence[str],
    figdir: Path,
    consensus: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> None:
    """
    Plot stacked composition summaries and save via record_plot_artifact().
    """
    totals = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(totals, axis=0)
    if props.empty:
        LOGGER.warning("composition: no proportions available for plotting")
        return
    props_plot = props.copy()
    props_plot.columns = props_plot.columns.astype(str)
    cluster_order = list(cluster_order)
    colors = list(colors)

    cond_levels = sorted(metadata[str(condition_key)].astype(str).dropna().unique().tolist())
    if not cond_levels:
        LOGGER.warning("composition: no condition levels available for plotting")
        return

    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(cond_levels)), 4))
    for j, cond in enumerate(cond_levels):
        mask = metadata.loc[metadata.index, str(condition_key)].astype(str) == str(cond)
        if mask.sum() == 0:
            mean_props = pd.Series(0.0, index=cluster_order)
        else:
            mean_props = props_plot.loc[mask].mean(axis=0).reindex(cluster_order)
        bottom = 0.0
        for idx, cl in enumerate(cluster_order):
            val = mean_props[cl]
            ax.bar(j, val, bottom=bottom, color=colors[idx], edgecolor="white", linewidth=0.3)
            bottom += val
    ax.set_xticks(range(len(cond_levels)))
    labels = [
        f"{cond}\n(n={(metadata[str(condition_key)].astype(str) == str(cond)).sum()})"
        for cond in cond_levels
    ]
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean proportion")
    ax.set_title("Cell Type Composition (100% stacked)")
    ax.grid(False)
    _add_outside_legend(
        ax,
        [str(c) for c in cluster_order],
        list(colors),
        "Cluster",
        max_rows=16,
    )
    plt.tight_layout()
    record_plot_artifact("composition_stacked_bar_100", figdir, fig=fig)
    close_plot(fig)

    sig_clusters = set()
    if consensus is not None and isinstance(consensus, pd.DataFrame) and not consensus.empty:
        try:
            if "high_confidence_da" in consensus.columns:
                sig_clusters = set(
                    consensus.loc[consensus["high_confidence_da"].fillna(False), "cluster"].astype(str).tolist()
                )
            else:
                sig_clusters = set(consensus.loc[consensus["n_sig"] > 0, "cluster"].astype(str).tolist())
        except Exception:
            sig_clusters = set()

    if len(cond_levels) == 2:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(cluster_order))))
        left = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[0]].mean()
        right = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[1]].mean()
        left = left.reindex(cluster_order)
        right = right.reindex(cluster_order)
        y = np.arange(len(cluster_order))
        ax.barh(y, left.values, color=colors, edgecolor="white", linewidth=0.3)
        ax.barh(y, right.values, left=left.values, color=colors, alpha=0.6, edgecolor="white", linewidth=0.3)
        totals = left.values + right.values
        if sig_clusters:
            for i, cl in enumerate(cluster_order):
                if str(cl) in sig_clusters:
                    ax.text(
                        totals[i] + 0.01,
                        y[i],
                        "*",
                        va="center",
                        ha="left",
                        fontsize=12,
                        color="#111111",
                    )
        ax.set_yticks(y)
        ax.set_yticklabels(cluster_order)
        ax.set_xlabel("Mean proportion")
        ax.set_title("Cell Type Composition (stacked comparison)")
        ax.text(
            0.98,
            0.98,
            f"Solid={cond_levels[0]}  Light={cond_levels[1]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#444444",
        )
        ax.grid(False)
        _add_outside_legend(
            ax,
            [str(c) for c in cluster_order],
            list(colors),
            "Cluster",
            max_rows=16,
        )
        plt.tight_layout()
        record_plot_artifact("composition_stacked_comparison", figdir, fig=fig)
        close_plot(fig)
    else:
        LOGGER.warning(
            "composition: stacked comparison plot skipped (requires exactly 2 condition levels, found %d).",
            len(cond_levels),
        )

    if len(cond_levels) == 2:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(cluster_order))))
        left = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[0]].mean()
        right = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[1]].mean()
        left = left.reindex(cluster_order)
        right = right.reindex(cluster_order)
        y = np.arange(len(cluster_order))
        for idx, cl in enumerate(cluster_order):
            y0 = idx
            x0 = float(left[cl])
            x1 = float(right[cl])
            ax.annotate(
                "",
                xy=(x1, y0),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors[idx],
                    lw=2,
                    alpha=0.85,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=10,
                ),
            )
            ax.scatter([x0], [y0], color=colors[idx], s=28, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(cluster_order)
        ax.set_xlabel("Mean proportion")
        ax.set_title("Cell Type Composition Flow")
        ax.text(
            0.98,
            0.98,
            f"Start(dot): {cond_levels[0]}   End(arrow): {cond_levels[1]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#444444",
        )
        ax.grid(False)
        _add_outside_legend(
            ax,
            [str(c) for c in cluster_order],
            list(colors),
            "Cluster",
            max_rows=16,
        )
        plt.tight_layout()
        record_plot_artifact("composition_flow", figdir, fig=fig)
        close_plot(fig)
    else:
        LOGGER.warning(
            "composition: flow plot skipped (requires exactly 2 condition levels, found %d).",
            len(cond_levels),
        )

    if len(cond_levels) == 2:
        from matplotlib.patches import Polygon

        fig, ax = plt.subplots(figsize=(8, 5))
        left = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[0]].mean()
        right = props_plot.loc[metadata[str(condition_key)].astype(str) == cond_levels[1]].mean()
        left = left.reindex(cluster_order).fillna(0.0)
        right = right.reindex(cluster_order).fillna(0.0)

        left_bottom = left.cumsum().shift(fill_value=0.0)
        right_bottom = right.cumsum().shift(fill_value=0.0)

        x_left = 0.0
        x_right = 1.0
        bar_width = 0.28
        left_edge = x_left + bar_width / 2
        right_edge = x_right - bar_width / 2

        for idx, cl in enumerate(cluster_order):
            y0_l = left_bottom[cl]
            y1_l = y0_l + left[cl]
            y0_r = right_bottom[cl]
            y1_r = y0_r + right[cl]
            poly = Polygon(
                [
                    (left_edge, y0_l),
                    (left_edge, y1_l),
                    (right_edge, y1_r),
                    (right_edge, y0_r),
                ],
                closed=True,
                facecolor=colors[idx],
                edgecolor="none",
                alpha=0.55,
            )
            ax.add_patch(poly)

        for idx, cl in enumerate(cluster_order):
            ax.bar(
                x_left,
                left[cl],
                bottom=left_bottom[cl],
                color=colors[idx],
                width=bar_width,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.bar(
                x_right,
                right[cl],
                bottom=right_bottom[cl],
                color=colors[idx],
                width=bar_width,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(
            [
                f"{cond_levels[0]}\n(n={(metadata[str(condition_key)].astype(str) == cond_levels[0]).sum()})",
                f"{cond_levels[1]}\n(n={(metadata[str(condition_key)].astype(str) == cond_levels[1]).sum()})",
            ]
        )
        ax.set_ylabel("Mean proportion")
        ax.set_title("Cell Type Composition Alluvial")
        ax.grid(False)
        _add_outside_legend(
            ax,
            [str(c) for c in cluster_order],
            list(colors),
            "Cluster",
            max_rows=16,
        )
        plt.tight_layout()
        record_plot_artifact("composition_alluvial", figdir, fig=fig)
        close_plot(fig)
    else:
        LOGGER.warning(
            "composition: alluvial plot skipped (requires exactly 2 condition levels, found %d).",
            len(cond_levels),
        )

# -------------------------------------------------------------------------
# Global styling
# -------------------------------------------------------------------------
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.linewidth"] = 0.6
mpl.rcParams["axes.edgecolor"] = "#555555"

mpl.rcParams["xtick.color"] = "#333333"
mpl.rcParams["ytick.color"] = "#333333"
mpl.rcParams["xtick.alignment"] = "right"

mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.constrained_layout.use"] = True

mpl.rcParams["figure.subplot.bottom"] = 0.25
mpl.rcParams["figure.subplot.right"] = 0.9

FIGURE_FORMATS = ["png", "pdf"]
ROOT_FIGDIR: Path | None = None
RUN_FIG_SUBDIR: Path | None = None
RUN_KEY: str | None = None


# -------------------------------------------------------------------------
# Setup + saving
# -------------------------------------------------------------------------
def set_figure_formats(formats: Sequence[str]) -> None:
    global FIGURE_FORMATS
    FIGURE_FORMATS = list(formats)


def setup_scanpy_figs(figdir: Path, formats: Sequence[str] | None = None) -> None:
    """
    Configure Scanpy and global figure settings for scOmnom.

    figdir is the base root where per-format folders live, typically:
      <output_dir>/figures
    """
    global ROOT_FIGDIR, RUN_FIG_SUBDIR, RUN_KEY

    figdir = Path(figdir)
    ROOT_FIGDIR = figdir.resolve()

    if formats is not None:
        set_figure_formats(formats)

    # Reset run routing; it will be inferred lazily from first save_multi call.
    RUN_FIG_SUBDIR = None
    RUN_KEY = None

    sc.settings.figdir = ROOT_FIGDIR
    sc.settings.autoshow = False
    sc.settings.autosave = False

    sc.settings.set_figure_params(
        dpi=300,
        dpi_save=300,
        facecolor="white",
        frameon=False,
        vector_friendly=False,
        fontsize=10,
        figsize=(6, 5),
        format=FIGURE_FORMATS[0],
    )

    ROOT_FIGDIR.mkdir(parents=True, exist_ok=True)


def get_run_subdir(run_key: str | None = None) -> Path:
    """
    Ensure and return the current run subdir used by record_plot_artifact().

    If not set yet, it initializes the run folder using the provided run_key
    (or the first inferred key if available) and pre-creates per-format dirs.
    """
    global ROOT_FIGDIR, RUN_FIG_SUBDIR, RUN_KEY

    if ROOT_FIGDIR is None:
        raise RuntimeError("ROOT_FIGDIR is not set. Call setup_scanpy_figs() first.")

    if RUN_FIG_SUBDIR is not None:
        if run_key is not None and RUN_KEY is not None and str(run_key) != str(RUN_KEY):
            LOGGER.warning(
                "get_run_subdir: run_key=%r does not match active RUN_KEY=%r; using existing run subdir.",
                str(run_key),
                str(RUN_KEY),
            )
        return RUN_FIG_SUBDIR

    if run_key is not None:
        RUN_KEY = str(run_key)
    if RUN_KEY is None:
        RUN_KEY = "figures"

    RUN_FIG_SUBDIR = _next_round_subdir(
        root_figdir=ROOT_FIGDIR,
        formats=FIGURE_FORMATS,
        run_name=RUN_KEY,
    )

    for ext in FIGURE_FORMATS:
        (ROOT_FIGDIR / ext / RUN_FIG_SUBDIR).mkdir(parents=True, exist_ok=True)

    return RUN_FIG_SUBDIR


def get_run_round_tag(run_key: str | None = None) -> str:
    """
    Return the round tag (e.g., "round3") for the current figure run.
    Ensures the run subdir is initialized.
    """
    run_subdir = get_run_subdir(run_key)
    m = re.search(r"_round(\d+)$", str(run_subdir))
    if not m:
        return "round1"
    return f"round{m.group(1)}"


def save_multi(stem: str, figdir: Path, fig=None, *, savefig_kwargs: dict | None = None) -> None:
    """
    Save the current matplotlib figure (or a provided figure) to multiple formats.

    Output layout (Option B):
      ROOT_FIGDIR/<ext>/<RUN_FIG_SUBDIR>/<rel_figdir>/<stem>.<ext>

    Where RUN_FIG_SUBDIR is inferred from the first path component of `figdir`,
    e.g. integration -> integration_roundN.
    The filename stem is ALWAYS sanitized to be filesystem-safe.
    """
    import re
    import matplotlib.pyplot as plt
    global ROOT_FIGDIR, RUN_FIG_SUBDIR, RUN_KEY, _ACTIVE_PLOT_CAPTURE

    # --------------------------------------------------
    # Sanitize filename stem
    # --------------------------------------------------
    def _safe_stem(s: str, max_len: int = 180) -> str:
        if s is None:
            return "figure"

        s = str(s)

        # Kill path separators first
        s = s.replace("/", "_").replace("\\", "_")

        # Replace other common offenders
        s = s.replace(":", "_")

        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # Replace anything not filesystem-safe
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

        # Collapse multiple underscores
        s = re.sub(r"_+", "_", s)

        # Trim junk from ends
        s = s.strip("._-")

        if not s:
            s = "figure"

        # Clamp length to avoid OS/path limits
        if len(s) > max_len:
            s = s[:max_len].rstrip("._-")

        return s

    stem = _safe_stem(stem)

    figdir = Path(figdir)

    if _ACTIVE_PLOT_CAPTURE is not None:
        fig_obj = fig
        if fig_obj is None:
            try:
                fig_obj = plt.gcf()
            except Exception:
                fig_obj = None
        _ACTIVE_PLOT_CAPTURE.append(
            PlotArtifact(
                stem=stem,
                figdir=figdir,
                fig=fig_obj,
                savefig_kwargs=savefig_kwargs,
            )
        )
        return

    if ROOT_FIGDIR is None:
        raise RuntimeError("ROOT_FIGDIR is not set. Call setup_scanpy_figs() first.")

    # --------------------------------------------------
    # Activate provided figure if any
    # --------------------------------------------------
    if fig is not None:
        plt.figure(fig.number)

    # --------------------------------------------------
    # Save in all configured formats
    # --------------------------------------------------
    # Lazily infer run folder from the first save call
    if RUN_FIG_SUBDIR is None:
        RUN_KEY = _infer_run_key(figdir)
        RUN_FIG_SUBDIR = _next_round_subdir(
            root_figdir=ROOT_FIGDIR,
            formats=FIGURE_FORMATS,
            run_name=RUN_KEY,
        )
        LOGGER.info(
            "Figure run root: %s/<ext>/%s/",
            ROOT_FIGDIR,
            RUN_FIG_SUBDIR,
        )

        # Precreate per-format run dirs
        for ext in FIGURE_FORMATS:
            (ROOT_FIGDIR / ext / RUN_FIG_SUBDIR).mkdir(parents=True, exist_ok=True)

    # Always compute rel_figdir (avoid duplicate integration/integration)
    rel_figdir = figdir
    if RUN_KEY and rel_figdir.parts and rel_figdir.parts[0] == RUN_KEY:
        rel_figdir = Path(*rel_figdir.parts[1:])  # may become "."

    kwargs = dict(dpi=300)
    if savefig_kwargs:
        kwargs.update(savefig_kwargs)

    # --------------------------------------------------
    # Save in all configured formats
    # --------------------------------------------------
    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / RUN_FIG_SUBDIR / rel_figdir
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{stem}.{ext}"
        LOGGER.debug("Saving figure: %s", outfile)

        if fig is not None:
            # avoid pyplot global state
            fig.savefig(outfile, **kwargs)
        else:
            plt.savefig(outfile, **kwargs)

    # --------------------------------------------------
    # Close safely
    # --------------------------------------------------
    if fig is not None:
        close_plot(fig)  # closes exactly that figure
    else:
        close_plot()  # closes current figure


@collect_plot_artifacts
def save_umap_multi(
    stem: str,
    figdir: Path,
    fig: mpl.figure.Figure,
    *,
    pad_inches: float = 0.25,
    tight: bool = True,
    right: float | None = 0.78,
) -> None:
    """
    UMAP-only saver that prevents Scanpy legends / annotations from being clipped.
    Delegates actual saving to record_plot_artifact().
    """
    if right is not None:
        try:
            fig.subplots_adjust(right=right)
        except Exception:
            pass

    savefig_kwargs = {}
    if tight:
        savefig_kwargs["bbox_inches"] = "tight"
        savefig_kwargs["pad_inches"] = pad_inches

    record_plot_artifact(
        stem=stem,
        figdir=figdir,
        fig=fig,
        savefig_kwargs=savefig_kwargs,
    )


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _sanitize_uns_colors(adata: ad.AnnData, key: str) -> None:
    """
    Ensure adata.uns[f"{key}_colors"] is a clean list of valid matplotlib colors
    with length == number of categories in adata.obs[key].

    Fixes common corruption patterns (dicts with '__type__', bad entries).
    If uns colors are unusable, regenerates a palette.
    """
    import matplotlib.colors as mcolors
    from scanpy.plotting.palettes import default_102

    colors_key = f"{key}_colors"
    if key not in adata.obs:
        return

    # Determine expected length (categories)
    try:
        cats = adata.obs[key].astype("category").cat.categories
        n_cats = int(len(cats))
    except Exception:
        # if not categorical, don't try to manage palette
        return

    raw = adata.uns.get(colors_key, None)

    # Normalize various shapes into a list[str]
    colors: list[str] | None = None
    if isinstance(raw, list):
        colors = [str(x) for x in raw]
    elif isinstance(raw, dict):
        # common: {"__type__": ..., "0": "#...", ...}
        # keep numeric-ish keys only, ordered
        items: list[tuple[int, str]] = []
        for k, v in raw.items():
            if str(k).startswith("__"):
                continue
            try:
                idx = int(k)
            except Exception:
                continue
            items.append((idx, str(v)))
        if items:
            items.sort(key=lambda t: t[0])
            colors = [v for _, v in items]

    # Validate
    def _is_valid_color(c: str) -> bool:
        try:
            return bool(mcolors.is_color_like(c))
        except Exception:
            return False

    ok = (
        colors is not None
        and len(colors) >= n_cats
        and all(_is_valid_color(c) for c in colors[:n_cats])
    )

    if ok:
        adata.uns[colors_key] = colors[:n_cats]
        return

    # Regenerate palette
    adata.uns[colors_key] = list(default_102[:n_cats])


def _next_round_subdir(root_figdir: Path, formats: Sequence[str], run_name: str) -> Path:
    """
    Pick next <run_name>_roundN by scanning *only* that module's folders:
      <root_figdir>/<fmt>/<run_name>_roundN

    Other modules' folders (e.g. qc_round*) are ignored.
    """
    root_figdir = Path(root_figdir)
    rx = re.compile(rf"^{re.escape(run_name)}_round(\d+)$")

    existing: set[int] = set()

    for fmt in formats:
        fmt_dir = root_figdir / fmt
        if not fmt_dir.exists():
            continue

        for p in fmt_dir.iterdir():
            if not p.is_dir():
                continue
            m = rx.match(p.name)
            if m:
                existing.add(int(m.group(1)))

    n = 1
    while n in existing:
        n += 1

    return Path(f"{run_name}_round{n}")


def _infer_run_key(figdir: Path) -> str:
    """
    Infer module/run key from the first path component of figdir.
    Examples:
      Path("integration") -> "integration"
      Path("integration/umaps") -> "integration"
      Path("QC_plots/qc_metrics") -> "QC_plots"
    """
    figdir = Path(figdir)
    parts = figdir.parts
    if not parts:
        return "figures"
    return str(parts[0])


def _is_categorical_series(s: pd.Series) -> bool:
    return (
        pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    )


def _umap_figsize_for_key(adata: ad.AnnData, key: str) -> tuple[float, float]:
    """
    Choose a wide enough figure size so large categorical legends don't squash the UMAP.
    """
    base_w, base_h = 6.5, 5.5
    if key not in adata.obs:
        return base_w, base_h

    s = adata.obs[key]
    if _is_categorical_series(s):
        try:
            n = int(s.astype("category").cat.categories.size)
        except Exception:
            n = int(s.nunique(dropna=True))

        # widen with number of categories, but cap to keep things sane
        w = min(22.0, max(base_w, 7.5 + 0.35 * min(n, 50)))
        # a bit taller if legend becomes multi-row
        h = min(10.0, max(base_h, 5.5 + 0.06 * max(0, n - 20)))
        return float(w), float(h)

    return base_w, base_h


def _tune_umap_legend(fig: plt.Figure, n_cats: int) -> None:
    """
    Scanpy draws the legend on the main axis; move it to a right margin area and
    split into columns if it's tall.
    """
    if not fig.axes:
        return
    ax = fig.axes[0]
    leg = ax.get_legend()
    if leg is None:
        return

    # Choose columns to reduce legend height
    if n_cats <= 18:
        ncol = 1
    elif n_cats <= 40:
        ncol = 2
    else:
        ncol = 3

    try:
        leg.set_ncols(ncol)
    except Exception:
        try:
            leg._ncols = ncol  # older mpl
        except Exception:
            pass

    # Put legend in the right margin, vertically centered
    leg.set_bbox_to_anchor((1.02, 0.5))
    leg._loc = 6  # "center left" (works across mpl versions)
    try:
        leg.set_loc("center left")
    except Exception:
        pass


def _nice_gmt_name(gmt_path_or_name: str) -> str | None:
    """
    Try to convert an MSigDB GMT filename into a nice label.
    Returns None if we can't confidently infer.
    Examples:
      h.all.v2025.1.Hs.symbols.gmt -> HALLMARK
      c2.cp.reactome.v2025.1.Hs.symbols.gmt -> REACTOME
    """
    s = str(gmt_path_or_name)
    base = Path(s).name.lower()

    # hallmark collection
    if base.startswith("h.all."):
        return "HALLMARK"

    # common curated libs
    if "reactome" in base:
        return "REACTOME"
    if "kegg" in base:
        return "KEGG"
    if "wikipathways" in base:
        return "WIKIPATHWAYS"
    if "biocarta" in base:
        return "BIOCARTA"

    # If you later add more, do it here.
    return None


def _finalize_categorical_x(
    fig,
    ax,
    *,
    rotate: float = 45,
    ha: str = "right",
    bottom: float = 0.30,
    right: float = 0.98,
    left: float = 0.10,
    top: float = 0.92,
):
    """
    For plots with many categorical x tick labels:
    - rotate labels
    - reserve margins so labels don't get clipped
    """
    try:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha=ha)
    except Exception:
        pass

    try:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    except Exception:
        pass


def _clean_axes(ax):
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_alpha(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def _reserve_bottom_for_xticklabels(
    fig,
    ax,
    *,
    rotation: float = 45,
    fontsize: float | None = None,
    ha: str = "right",
    extra_bottom: float = 0.03,
    max_bottom: float = 0.72,
):
    """
    Rotate x tick labels and reserve enough bottom margin so they don't clip.

    Heuristic: bottom margin increases with the longest label length.
    """
    labels = [t.get_text() for t in ax.get_xticklabels()]
    max_len = max((len(s) for s in labels), default=0)

    # Base bottom + length-dependent bump (works well for long cluster names)
    bottom = 0.26 + 0.0085 * max_len + extra_bottom
    bottom = float(min(max_bottom, bottom))

    try:
        for t in ax.get_xticklabels():
            t.set_rotation(rotation)
            t.set_ha(ha)
            if fontsize is not None:
                t.set_fontsize(fontsize)
    except Exception:
        pass

    # Critical: manual adjust AFTER any tight_layout / plotting
    try:
        fig.subplots_adjust(bottom=bottom)
    except Exception:
        pass


def _ensure_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _res_key(r: float | str) -> str:
    """
    Canonical string key for a resolution: always '%.3f'.

    All resolution-indexed mappings in scOmnom are expected
    to use this representation, e.g. '0.200', '0.400', ...
    """
    return f"{float(r):.3f}"


def _sorted_resolutions(resolutions: Iterable[float | str]) -> list[float]:
    """Return sorted float list of resolutions."""
    return sorted(float(r) for r in resolutions)


def _extract_series(
        resolutions: Sequence[float | str],
        values: Mapping[str, Any],
        default: float = np.nan,
) -> np.ndarray:
    """
    Given resolutions and a mapping keyed by canonical '%.3f' strings,
    return a numpy array of values aligned to sorted resolutions.

    If a resolution key is missing, `default` is used.
    """
    res_sorted = _sorted_resolutions(resolutions)
    out: List[float] = []
    for r in res_sorted:
        key = _res_key(r)
        if key not in values:
            out.append(float(default))
            continue
        try:
            out.append(float(values[key]))
        except Exception:
            out.append(float(default))
    return np.array(out, dtype=float)


def _normalize_array(x: np.ndarray) -> np.ndarray:
    """Min-max normalize 1D array to [0,1], ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x)
    vmin = float(np.nanmin(x[mask]))
    vmax = float(np.nanmax(x[mask]))
    if vmax == vmin:
        out = np.zeros_like(x)
        out[mask] = 0.0
        return out
    out = np.zeros_like(x)
    out[mask] = (x[mask] - vmin) / (vmax - vmin)
    return out


def _plateau_spans(plateaus: Sequence[Mapping[str, object]]) -> list[tuple[float, float]]:
    """
    Convert plateau definitions (with 'resolutions' lists) to x-span pairs.
    Assumes plateau['resolutions'] contains float-like entries.
    """
    spans: list[tuple[float, float]] = []
    for p in plateaus or []:
        rs = [float(x) for x in p.get("resolutions", [])]
        if not rs:
            continue
        spans.append((min(rs), max(rs)))
    return spans


# -------------------------------------------------------------------------
# SCANPY WRAPPERS (QC)
# -------------------------------------------------------------------------
def _violin_with_points(
        adata: ad.AnnData,
        metric: str,
        *,
        groupby: str,
        horizontal: bool,
        ax=None,
        point_alpha: float = 0.08,
        point_size: float = 3.0,
        max_points: int | None = None,
):
    """
    Draw per-cell scatter points BEHIND a Scanpy violin plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    obs = adata.obs
    groups = obs[groupby].astype("category")
    y = obs[metric].to_numpy()

    # Optional downsampling (very large datasets)
    idx = np.arange(len(y))
    if max_points is not None and len(idx) > max_points:
        idx = np.random.choice(idx, max_points, replace=False)

    groups = groups.iloc[idx]
    y = y[idx]

    cats = groups.cat.categories

    # --- scatter FIRST (behind violin) ---
    for i, cat in enumerate(cats):
        mask = groups == cat
        y_cat = y[mask]

        jitter = np.random.normal(0, 0.04, size=len(y_cat))

        if horizontal:
            # X = category, Y = metric
            x = i + jitter
            ax.scatter(
                x,
                y_cat,
                s=point_size,
                alpha=point_alpha,
                color="black",
                rasterized=True,
                zorder=1,
            )
        else:
            # Rotated: X = metric, Y = category
            y_pos = i + jitter
            ax.scatter(
                y_cat,
                y_pos,
                s=point_size,
                alpha=point_alpha,
                color="black",
                rasterized=True,
                zorder=1,
            )


@collect_plot_artifacts
def qc_scatter(adata, groupby: str, cfg):
    figdir = Path("QC_plots") / "qc_scatter"

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False,
    )
    _clean_axes(plt.gca())
    record_plot_artifact("QC_scatter_mt", figdir)


@collect_plot_artifacts
def hvgs_and_pca_plots(adata, max_pcs_plot: int, cfg):
    figdir = Path("QC_plots") / "overview"

    sc.pl.highly_variable_genes(adata, show=False)
    record_plot_artifact("QC_highly_variable_genes", figdir)

    sc.pl.pca_variance_ratio(adata, n_pcs=max_pcs_plot, log=True, show=False)
    record_plot_artifact("QC_pca_variance_ratio", figdir)


@collect_plot_artifacts
def umap_by(adata, keys, figdir: Path | None = None, stem: str | None = None):
    """
    Plot UMAP colored by one or more keys.
    Fixes squishing by allocating enough width for large legends.
    """
    import matplotlib.pyplot as plt

    if isinstance(keys, str):
        keys = [keys]

    if stem is None:
        name = f"QC_umap_{'_'.join(keys)}"
    else:
        name = stem

    if figdir is None:
        figdir = ROOT_FIGDIR

    # Plot each key separately so we can size per-legend and save cleanly.
    for key in keys:
        if key not in adata.obs:
            LOGGER.warning("umap_by: key '%s' not in adata.obs; skipping.", key)
            continue

        # Decide legend behavior
        s = adata.obs[key]
        is_cat = _is_categorical_series(s)
        if is_cat:
            try:
                n_cats = int(s.astype("category").cat.categories.size)
            except Exception:
                n_cats = int(s.nunique(dropna=True))
        else:
            n_cats = 0

        fig_w, fig_h = _umap_figsize_for_key(adata, key)

        # Create figure/axes explicitly (avoid Scanpy passing figsize to scatter)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass

        sc.pl.umap(
            adata,
            color=key,
            use_raw=False,
            show=False,
            ax=ax,  # <- key change
            legend_loc=("right margin" if is_cat else None),
            legend_fontsize=(10 if is_cat else None),
        )

        if is_cat and n_cats > 0:
            _tune_umap_legend(fig, n_cats)
            fig.subplots_adjust(left=0.06, right=0.72, top=0.92, bottom=0.08)
        else:
            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)

        record_plot_artifact(f"{name}_{key}", figdir, fig)
        close_plot(fig)


# -------------------------------------------------------------------------
# Cell-calling elbow/knee plot
# -------------------------------------------------------------------------
@collect_plot_artifacts
def plot_elbow_knee(
        adata,
        figpath_stem: str,
        figdir: Path,
        title: str = "Barcode Rank UMI Knee Plot",
):
    from kneed import KneeLocator

    if "counts_raw" in adata.layers:
        total = np.asarray(adata.layers["counts_raw"].sum(axis=1)).ravel()
    else:
        total = np.asarray(adata.X.sum(axis=1)).ravel()

    sorted_idx = np.argsort(total)[::-1]
    sorted_counts = total[sorted_idx]
    ranks = np.arange(1, len(sorted_counts) + 1)

    kl = KneeLocator(ranks, sorted_counts, curve="convex", direction="decreasing")
    knee_rank = kl.elbow

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ranks, sorted_counts, lw=1, color="steelblue")

    if knee_rank is not None:
        knee_val = sorted_counts[knee_rank - 1]
        ax.axvline(knee_rank, color="red", linestyle="--", lw=0.8)
        ax.axhline(knee_val, color="red", linestyle="--", lw=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Barcode rank")
    ax.set_ylabel("Total UMI counts")
    ax.set_title(title)

    _clean_axes(ax)
    fig.tight_layout()

    record_plot_artifact(figpath_stem, figdir)


# -------------------------------------------------------------------------
# Final cell-counts plot
# -------------------------------------------------------------------------
@collect_plot_artifacts
def plot_final_cell_counts(adata, cfg) -> None:
    """Plot final per-sample cell counts with a mean line and summary box."""
    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key not in adata.obs:
        LOGGER.warning("batch_key '%s' not found in adata.obs; skipping plot.", batch_key)
        return

    counts = adata.obs[batch_key].value_counts().sort_index()
    mean_cells = counts.mean()
    total_cells = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")

    ax.axhline(mean_cells, linestyle="--", color="#1f4e79", linewidth=1.0)
    _clean_axes(ax)

    ax.set_ylabel("Cell count")
    ax.set_title("Final cell counts per sample")

    summary_text = f"Total cells: {total_cells:,}\nMean per sample: {mean_cells:,.0f}"
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    figdir_qc = Path("QC_plots") / "overview"
    record_plot_artifact(stem="final_cell_counts", figdir=figdir_qc)

    close_plot(fig)


# -------------------------------------------------------------------------
# MT histogram
# -------------------------------------------------------------------------
@collect_plot_artifacts
def plot_mt_histogram(adata, cfg, suffix: str):
    figdir_qc = Path("QC_plots") / "qc_metrics"

    fig, ax = plt.subplots(figsize=(5, 4))
    _clean_axes(ax)

    ax.hist(adata.obs["pct_counts_mt"], bins=50, color="steelblue", alpha=0.85)

    ax.set_xlabel("Percent mitochondrial counts")
    ax.set_ylabel("Number of cells")
    ax.set_title("Distribution of mitochondrial content")

    fig.tight_layout()
    record_plot_artifact(f"{suffix}_QC_hist_pct_mt", figdir_qc)


# -------------------------------------------------------------------------
# QC plots: pre + post filter
# -------------------------------------------------------------------------
@collect_plot_artifacts
def run_qc_plots_pre_filter_df(qc_df: pd.DataFrame, cfg) -> None:
    """
    Pre-filter QC plots using a lightweight DataFrame instead of a giant AnnData.

    Expects qc_df with columns:
      - 'sample'
      - 'total_counts'
      - 'n_genes_by_counts'
      - 'pct_counts_mt'
    """
    if not cfg.make_figures:
        return

    qc_df = qc_df.copy()
    qc_df.index = qc_df.index.astype(str)

    qc_adata = ad.AnnData(obs=qc_df)
    qc_adata.obs[cfg.batch_key] = qc_df["sample"].values

    qc_violin_panels(qc_adata, cfg, "prefilter")
    qc_scatter_panels(qc_adata, cfg, "prefilter")
    plot_mt_histogram(qc_adata, cfg, "prefilter")
    plot_hist_n_genes(qc_adata, cfg, "prefilter")
    plot_hist_total_counts(qc_adata, cfg, "prefilter")


@collect_plot_artifacts
def run_qc_plots_postfilter(adata, cfg):
    """
    Run post-filter QC plots on:
      1) raw counts  (layers["counts_raw"])
      2) CellBender counts (layers["counts_cb"], if present)

    QC metrics are computed transiently per pass to avoid overwriting.
    """
    from scanpy import settings as sc_settings
    from .load_and_filter import compute_qc_metrics

    if not cfg.make_figures:
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_qc_pass(adata, X, label):
        """
        Run QC plots with X temporarily set to given matrix.
        Restores obs, var, X, and qc_metrics afterwards.
        """
        # Backup state
        obs_backup = adata.obs.copy()
        var_backup = adata.var.copy()
        X_backup = adata.X
        qc_uns_backup = adata.uns.get("qc_metrics", None)

        try:
            adata.X = X

            compute_qc_metrics(adata, cfg)

            qc_violin_panels(adata, cfg, f"postfilter_{label}")
            qc_scatter_panels(adata, cfg, f"postfilter_{label}")
            plot_mt_histogram(adata, cfg, f"postfilter_{label}")
            plot_hist_n_genes(adata, cfg, f"postfilter_{label}")
            plot_hist_total_counts(adata, cfg, f"postfilter_{label}")

        finally:
            # Restore exact original state
            adata.X = X_backup
            adata.obs = obs_backup.copy()
            adata.var = var_backup.copy()

            if qc_uns_backup is None:
                adata.uns.pop("qc_metrics", None)
            else:
                adata.uns["qc_metrics"] = qc_uns_backup

    # --------------------------------------------------------------
    # 1. Raw counts QC (canonical)
    # --------------------------------------------------------------
    if "counts_raw" in adata.layers:
        _run_qc_pass(
            adata,
            adata.layers["counts_raw"],
            label="raw",
        )

    # --------------------------------------------------------------
    # 2. CellBender QC (diagnostic)
    # --------------------------------------------------------------
    if "counts_cb" in adata.layers:
        _run_qc_pass(
            adata,
            adata.layers["counts_cb"],
            label="cb",
        )


@collect_plot_artifacts
def plot_hist_total_counts(adata, cfg, stage: str):
    """
    Histogram of total_counts for prefilter / postfilter.
    stage = 'prefilter' or 'postfilter'
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not cfg.make_figures:
        return

    figdir_qc = Path("QC_plots") / "qc_metrics"

    plt.figure(figsize=(6, 4))
    sns.histplot(
        adata.obs["total_counts"],
        bins=60,
        kde=False,
        color="darkorange",
    )
    plt.xlabel("Total UMI counts")
    plt.ylabel("Cell count")
    plt.title(f"total_counts ({stage})")
    _clean_axes(plt.gca())

    record_plot_artifact(f"{stage}_QC_hist_total_counts", figdir_qc)
    close_plot()


@collect_plot_artifacts
def plot_hist_n_genes(adata, cfg, stage: str):
    """
    Histogram of n_genes_by_counts for prefilter / postfilter.
    stage = 'prefilter' or 'postfilter'
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not cfg.make_figures:
        return

    figdir_qc = Path("QC_plots") / "qc_metrics"

    plt.figure(figsize=(6, 4))
    sns.histplot(
        adata.obs["n_genes_by_counts"],
        bins=60,
        kde=False,
        color="steelblue",
    )
    plt.xlabel("Number of genes detected")
    plt.ylabel("Cell count")
    plt.title(f"n_genes_by_counts ({stage})")
    _clean_axes(plt.gca())

    record_plot_artifact(f"{stage}_QC_hist_n_genes", figdir_qc)
    close_plot()


def qc_violin_panels(adata, cfg, stage: str):
    """
    Three-panel QC violin summary:
      1) n_genes_by_counts
      2) total_counts
      3) pct_counts_mt

    Layout automatically switches:
      - <= 25 samples → violins on X-axis (samples as categories)
      - > 25 samples → violins stacked on Y-axis for readability
    """
    import matplotlib.pyplot as plt
    import scanpy as sc
    from scanpy import settings as sc_settings

    if not cfg.make_figures:
        return

    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key not in adata.obs:
        LOGGER.warning("batch_key '%s' missing in adata.obs; skipping QC violin panels", batch_key)
        return

    figdir_qc = Path("QC_plots") / "qc_metrics"

    # Decide layout
    n_samples = adata.obs[batch_key].nunique()
    horizontal = n_samples <= 25  # True → normal violins on X; False → rotate

    metrics = [
        ("n_genes_by_counts", "QC_violin_genes"),
        ("total_counts", "QC_violin_counts"),
        ("pct_counts_mt", "QC_violin_mt"),
        ("pct_counts_ribo", "QC_violin_ribo"),
        ("pct_counts_hb", "QC_violin_hb"),
    ]

    for metric, stem in metrics:
        if metric not in adata.obs:
            LOGGER.warning("Metric '%s' missing in adata.obs; skipping", metric)
            continue

        plt.figure(figsize=(10, 6) if horizontal else (12, 10))
        ax = plt.gca()

        # ---- scatter points FIRST (behind violins) ----
        _violin_with_points(
            adata,
            metric,
            groupby=batch_key,
            horizontal=horizontal,
            ax=ax,
            point_alpha=0.08,
            point_size=3.0,
            max_points=200_000,  # safety for huge datasets
        )

        # ---- violin on top ----
        sc.pl.violin(
            adata,
            metric,
            groupby=batch_key,
            rotation=45 if horizontal else 0,
            show=False,
            stripplot=False,
            ax=ax,
        )

        _clean_axes(ax)
        ax.set_title(f"{metric} ({stage})")

        # --- reserve space for long sample names ---
        n_cats = adata.obs[batch_key].astype("category").cat.categories.size

        # heuristic: more categories → more bottom margin
        bottom = 0.28 if n_cats <= 20 else 0.36 if n_cats <= 40 else 0.45

        fig = ax.figure

        if not horizontal:
            # categories are y ticklabels, give them room
            fig.subplots_adjust(left=0.35, right=0.98, top=0.92, bottom=0.10)

        _finalize_categorical_x(
            fig,
            ax,
            rotate=45 if horizontal else 0,  # when rotated layout, ticks are on y anyway
            ha="right",
            bottom=bottom,
        )

        record_plot_artifact(f"{stem}_{stage}", figdir_qc)
        close_plot()


@collect_plot_artifacts
def qc_scatter_panels(adata, cfg, stage: str):
    """
    Additional scatter QC plots:
      - total_counts vs n_genes_by_counts (colored by pct_counts_mt)
      - total_counts vs pct_counts_mt
    stage = 'prefilter' or 'postfilter'
    """

    if not cfg.make_figures:
        return

    figdir = Path("QC_plots") / "qc_scatter"

    # --------------------------------------------------------------
    # Scatter 1: Complexity plot (colored by mt%)
    # --------------------------------------------------------------
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False,
    )
    _clean_axes(plt.gca())
    record_plot_artifact(f"QC_complexity_{stage}", figdir)
    close_plot()

    # --------------------------------------------------------------
    # Scatter 2: total_counts vs pct_counts_mt
    # --------------------------------------------------------------
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
    )
    _clean_axes(plt.gca())
    record_plot_artifact(f"QC_scatter_mt_{stage}", figdir)
    close_plot()

@collect_plot_artifacts
def plot_cellbender_effects(
        adata: ad.AnnData,
        *,
        batch_key: str | None,
        figdir: Path,
) -> None:
    """
    Diagnostic plots comparing raw vs CellBender counts
    for the same retained cells.

    Requires:
      - adata.layers["counts_raw"]
      - adata.layers["counts_cb"]

    Outputs to:
      figures/<FMT>/QC_plots/cellbender/
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if "counts_raw" not in adata.layers or "counts_cb" not in adata.layers:
        LOGGER.info("Skipping CellBender effect plots (raw or cb counts missing).")
        return

    X_raw = adata.layers["counts_raw"]
    X_cb = adata.layers["counts_cb"]

    # ------------------------------------------------------------------
    # Per-cell aggregates (OOM-safe)
    # ------------------------------------------------------------------
    raw_cell = np.asarray(X_raw.sum(axis=1)).ravel()
    cb_cell = np.asarray(X_cb.sum(axis=1)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        removed_frac_cell = (raw_cell - cb_cell) / raw_cell
        removed_frac_cell[~np.isfinite(removed_frac_cell)] = 0.0

    # ------------------------------------------------------------------
    # 1. Histogram: per-cell removed fraction
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(removed_frac_cell, bins=50, color="steelblue", alpha=0.85)
    ax.set_xlabel("Fraction of counts removed by CellBender")
    ax.set_ylabel("Cells")
    ax.set_title("CellBender background removal (per cell)")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("cellbender_removed_fraction_hist", figdir, fig)
    close_plot(fig)

    # ------------------------------------------------------------------
    # 2. Per-sample removed fraction (if batch_key present)
    # ------------------------------------------------------------------
    if batch_key is not None and batch_key in adata.obs:
        groups = adata.obs[batch_key].astype("category")
        data = [
            removed_frac_cell[groups == g]
            for g in groups.cat.categories
        ]

        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(data)), 4))
        ax.violinplot(data, showmeans=False, showextrema=False)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(groups.cat.categories, rotation=45, ha="right")
        ax.set_ylabel("Fraction removed")
        ax.set_title("CellBender background removal per sample")
        _clean_axes(ax)
        fig.tight_layout()
        record_plot_artifact("cellbender_removed_fraction_per_sample", figdir, fig)
        close_plot(fig)

    # ------------------------------------------------------------------
    # Per-gene aggregates (OOM-safe)
    # ------------------------------------------------------------------
    raw_gene = np.asarray(X_raw.sum(axis=0)).ravel()
    cb_gene = np.asarray(X_cb.sum(axis=0)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        removed_frac_gene = (raw_gene - cb_gene) / raw_gene
        removed_frac_gene[~np.isfinite(removed_frac_gene)] = 0.0

    # ------------------------------------------------------------------
    # 3. Per-gene raw vs CB scatter (log–log)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        np.log10(raw_gene + 1),
        np.log10(cb_gene + 1),
        s=4,
        alpha=0.3,
        rasterized=True,
    )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="black", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("log10(raw gene counts + 1)")
    ax.set_ylabel("log10(CellBender gene counts + 1)")
    ax.set_title("Gene-level counts: raw vs CellBender")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("cellbender_gene_raw_vs_cb", figdir, fig)
    close_plot(fig)

    # ------------------------------------------------------------------
    # 4. Per-gene removed fraction vs expression
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        np.log10(raw_gene + 1),
        removed_frac_gene,
        s=4,
        alpha=0.3,
        rasterized=True,
    )
    ax.set_xlabel("log10(raw gene counts + 1)")
    ax.set_ylabel("Fraction removed")
    ax.set_title("Gene-level background removal")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("cellbender_gene_removed_fraction", figdir, fig)
    close_plot(fig)

    # ------------------------------------------------------------------
    # 5. Per-cell removed fraction vs library size, colored by %mt
    # ------------------------------------------------------------------
    if "pct_counts_mt" in adata.obs:
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(
            np.log10(raw_cell + 1),
            removed_frac_cell,
            c=adata.obs["pct_counts_mt"],
            cmap="viridis",
            s=6,
            alpha=0.4,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("% mitochondrial counts")

        ax.set_xlabel("log10(total raw counts + 1)")
        ax.set_ylabel("Fraction removed")
        ax.set_title("CellBender effect vs library size")
        _clean_axes(ax)
        fig.tight_layout()
        record_plot_artifact("cellbender_removed_fraction_vs_library_mt", figdir, fig)
        close_plot(fig)

    LOGGER.info("Generated CellBender effect QC plots.")


@collect_plot_artifacts
def plot_qc_filter_stack(
    adata,
    *,
    batch_key: str = "sample_id",
    figdir: Path,
    fname: str = "qc_filter_effects_stacked",
) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if "qc_filter_stats" not in adata.uns:
        raise KeyError("qc_filter_stats not found in adata.uns")

    df = adata.uns["qc_filter_stats"].copy()

    # --------------------------------------------------
    # Keep only per-sample cell filters
    # --------------------------------------------------
    df = df[
        (df["scope"] == "cell")
        & (df["batch"] != "ALL")
    ]

    if df.empty:
        raise ValueError("No per-sample QC filter stats available")

    # --------------------------------------------------
    # Determine filter order (as applied)
    # --------------------------------------------------
    filter_order = (
        df.groupby("filter")["n_before"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # --------------------------------------------------
    # Reconstruct fraction relative to ORIGINAL cell count
    # --------------------------------------------------
    rows = []

    for batch, g in df.groupby("batch"):
        g = g.set_index("filter").loc[filter_order]

        n0 = g["n_before"].iloc[0]  # original cells
        for filt, row in g.iterrows():
            rows.append(
                {
                    "batch": batch,
                    "filter": filt,
                    "frac_removed_total": row["n_removed"] / n0,
                }
            )

    plot_df = pd.DataFrame(rows)

    # Pivot: batch × filter
    plot_df = (
        plot_df
        .pivot(index="batch", columns="filter", values="frac_removed_total")
        .fillna(0.0)
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * plot_df.shape[0]), 4)
    )

    bottom = np.zeros(plot_df.shape[0])

    for filt in plot_df.columns:
        vals = plot_df[filt].values
        ax.bar(
            plot_df.index,
            vals,
            bottom=bottom,
            label=filt,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += vals

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of original cells removed")
    ax.set_title("QC filtering effects per sample (100% stacked)")
    ax.legend(
        title="Filter",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    _clean_axes(ax)

    fig.tight_layout()
    record_plot_artifact("QC Filter effects", figdir, fig)
    close_plot(fig)


@collect_plot_artifacts
def doublet_plots(
    adata: ad.AnnData,
    *,
    batch_key: str,
    figdir: Path,
) -> None:
    """
    SOLO doublet QC plots.

    Must be called AFTER SOLO prediction (doublet_score + predicted_doublet)
    but BEFORE cleanup. Thresholds are inferred per sample from rate-based
    calling and visualized for diagnostics only.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)

    required = {"doublet_score", "predicted_doublet", batch_key}
    if not required.issubset(adata.obs.columns):
        LOGGER.warning("Skipping doublet plots; missing required columns.")
        return

    scores = adata.obs["doublet_score"].to_numpy()
    is_doublet = adata.obs["predicted_doublet"].astype(bool)

    # --------------------------------------------------
    # Compute per-sample inferred thresholds
    # --------------------------------------------------
    thresholds: dict[str, float] = {}
    for sample, obs in adata.obs.groupby(batch_key, observed=True):
        called = obs["predicted_doublet"].astype(bool)
        if called.any():
            thresholds[sample] = float(
                obs.loc[called, "doublet_score"].min()
            )

    def _draw_thresholds(ax, *, vertical: bool):
        for thr in thresholds.values():
            if vertical:
                ax.axvline(
                    thr,
                    color="red",
                    lw=0.6,
                    alpha=0.25,
                    zorder=1,
                )
            else:
                ax.axhline(
                    thr,
                    color="red",
                    lw=0.6,
                    alpha=0.25,
                    zorder=1,
                )

    # ==================================================
    # 1. Doublet score histogram
    # ==================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=50, color="steelblue", alpha=0.85)
    _draw_thresholds(ax, vertical=True)

    ax.set_xlabel("Doublet score")
    ax.set_ylabel("Cells")
    ax.set_title("SOLO doublet score distribution")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("doublet_score_hist", figdir, fig)
    close_plot(fig)

    # ==================================================
    # 2. Per-sample inferred doublet score threshold
    # ==================================================
    thr_series = (
        adata.obs[batch_key]
        .map(thresholds)
        .dropna()
        .groupby(adata.obs[batch_key], observed=True)
        .first()
        .sort_values(ascending=False)
    )

    thr_series = thr_series.astype(float)

    if len(thr_series) > 0:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.5 * len(thr_series)), 4)
        )
        thr_series.plot.bar(
            ax=ax,
            color="firebrick",
            edgecolor="black",
        )
        ax.set_ylabel("Inferred doublet score threshold")
        ax.set_title("Per-sample inferred SOLO threshold")
        ax.set_ylim(
            0,
            max(thr_series.max() * 1.2, thr_series.max() + 0.05),
        )
        plt.xticks(rotation=45, ha="right")
        _clean_axes(ax)
        fig.tight_layout()
        record_plot_artifact("doublet_inferred_threshold_per_sample", figdir, fig)
        close_plot(fig)

    # ==================================================
    # 3. Per-sample observed doublet fraction
    # ==================================================
    frac = (
        is_doublet.astype(int)
        .groupby(adata.obs[batch_key], observed=True)
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(frac)), 4))
    frac.plot.bar(ax=ax, color="firebrick", edgecolor="black")
    ax.set_ylabel("Fraction doublets")
    ax.set_title("Observed doublet fraction per sample")
    ax.set_ylim(0, max(0.05, frac.max() * 1.2))
    plt.xticks(rotation=45, ha="right")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("doublet_fraction_per_sample", figdir, fig)
    close_plot(fig)

    # ==================================================
    # 4. Doublet score vs library size
    # ==================================================
    X = adata.layers.get("counts_raw", adata.X)
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        total_counts,
        scores,
        s=5,
        alpha=0.3,
        rasterized=True,
    )
    _draw_thresholds(ax, vertical=False)

    ax.set_xlabel("Total UMI counts (raw)")
    ax.set_ylabel("Doublet score")
    ax.set_title("Doublet score vs library size")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("doublet_score_vs_total_counts", figdir, fig)
    close_plot(fig)

    # ==================================================
    # 5. Violin: doublet score per sample
    # ==================================================
    samples = list(frac.index)
    data = [
        adata.obs.loc[adata.obs[batch_key] == s, "doublet_score"]
        for s in samples
    ]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(samples)), 4))
    ax.violinplot(
        data,
        showmeans=False,
        showextrema=False,
        widths=0.8,
    )

    half_width = 0.25
    for i, sample in enumerate(samples, start=1):
        thr = thresholds.get(sample)
        if thr is None:
            continue
        ax.hlines(
            y=thr,
            xmin=i - half_width,
            xmax=i + half_width,
            color="red",
            lw=1.2,
            alpha=0.7,
            zorder=3,
        )

    ax.set_xticks(range(1, len(samples) + 1))
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("Doublet score")
    ax.set_title("Doublet score distribution per sample")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("doublet_score_violin_per_sample", figdir, fig)
    close_plot(fig)

    # ==================================================
    # 6. ECDF of doublet scores
    # ==================================================
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, lw=1.5)
    _draw_thresholds(ax, vertical=True)

    ax.set_xlabel("Doublet score")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("ECDF of doublet scores")
    _clean_axes(ax)
    fig.tight_layout()
    record_plot_artifact("doublet_score_ecdf", figdir, fig)
    close_plot(fig)

    LOGGER.info("Generated SOLO doublet QC plots (per-sample inferred thresholds).")


@collect_plot_artifacts
def umap_plots(
    adata: ad.AnnData,
    *,
    batch_key: str,
    figdir: Path,
    cluster_key: str = "leiden",
) -> None:
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)

    if "X_umap" not in adata.obsm:
        LOGGER.warning("Skipping UMAP plots: X_umap not found.")
        return

    # 1) batch
    if batch_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=batch_key,
            show=False,
            return_fig=True,
            legend_loc="right margin",   # <- consistent, outside axes
        )
        save_umap_multi("umap_batch", figdir, fig, right=0.78)
    else:
        LOGGER.warning("Batch key '%s' not found in adata.obs", batch_key)

    # 2) clusters
    if cluster_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=cluster_key,
            show=False,
            return_fig=True,
            legend_loc="right margin",   # <- NOT "on data" (avoids messy labels)
        )
        save_umap_multi(f"umap_{cluster_key}", figdir, fig, right=0.78)
    else:
        LOGGER.warning("Cluster key '%s' not found in adata.obs", cluster_key)

    LOGGER.info("Generated UMAP plots (batch + %s)", cluster_key)


@collect_plot_artifacts
def umap_by_two_legend_styles(
    adata,
    *,
    key: str,
    figdir,
    stem: str,
    title: str | None = None,
) -> None:
    """
    Emit two UMAPs for a categorical key:
      - fulllegend: standard right-side legend with full labels
      - shortlegend: no legend; Cnn overlaid at cluster centroids

    Respects Scanpy colors in adata.uns[f"{key}_colors"] if present.
    Uses save_umap_multi(...) like the rest of your plotting.
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import scanpy as sc
    from pathlib import Path

    base = Path(figdir)

    if "X_umap" not in adata.obsm:
        raise KeyError("umap_by_two_legend_styles: adata.obsm['X_umap'] missing.")
    if key not in adata.obs:
        raise KeyError(f"umap_by_two_legend_styles: key={key!r} missing from adata.obs.")

    X = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs[key].astype(str).to_numpy()
    ttl = title if title is not None else key

    def _extract_cnn(x: str) -> str:
        s = str(x or "")
        m = re.search(r"\b(C\d+)\b", s)
        if m:
            return m.group(1)
        m2 = re.search(r"(C\d+)", s)
        if m2:
            return m2.group(1)
        return s.split()[0][:8] if s.strip() else "C?"

    def _unique_stable(arr: np.ndarray) -> list[str]:
        seen = set()
        out: list[str] = []
        for v in arr.tolist():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _annotate_cnn(ax) -> None:
        if X.ndim != 2 or X.shape[1] != 2:
            return
        for lab in _unique_stable(labels):
            m = labels == lab
            if not np.any(m):
                continue
            cx = float(np.median(X[m, 0]))
            cy = float(np.median(X[m, 1]))
            ax.text(
                cx,
                cy,
                _extract_cnn(lab),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65),
                zorder=10,
            )

    # 1) full legend
    fig_full = sc.pl.umap(
        adata,
        color=key,
        title=ttl,
        show=False,
        return_fig=True,
        legend_loc="right margin",
    )
    save_umap_multi(f"{stem}__fulllegend", figdir=base, fig=fig_full)

    # 2) short legend (no legend; Cnn on clusters)
    fig_short = sc.pl.umap(
        adata,
        color=key,
        title=ttl,
        show=False,
        return_fig=True,
        legend_loc=None,
    )
    ax = fig_short.axes[0] if getattr(fig_short, "axes", None) else plt.gca()
    _annotate_cnn(ax)
    save_umap_multi(f"{stem}__shortlegend", figdir=base, fig=fig_short)


# -------------------------------------------------------------------------
# scIB-style results table
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path


@collect_plot_artifacts
def plot_scib_results_table(scaled: pd.DataFrame, *, stem: str = "scIB_results_table") -> None:
    df = scaled.copy()

    # ------------------------------------------------------------------
    # 1. Clean and sort
    # ------------------------------------------------------------------
    df = df.loc[~df.index.str.contains("Metric", case=False, na=False)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Total", ascending=False)

    all_cols = df.columns.tolist()

    # ------------------------------------------------------------------
    # 2. Define categories
    # ------------------------------------------------------------------
    agg_metrics = [c for c in ["Batch correction", "Bio conservation", "Total"] if c in all_cols]
    batch_metrics = [
        c for c in ["iLISI", "KBET", "Graph connectivity", "PCR comparison", "Silhouette batch"]
        if c in all_cols
    ]
    bio_metrics = [c for c in all_cols if c not in agg_metrics + batch_metrics]

    ordered_cols = bio_metrics + batch_metrics + agg_metrics
    df = df[ordered_cols]

    # ------------------------------------------------------------------
    # 3. X-position mapping (tighter gaps, centered dividers)
    # ------------------------------------------------------------------
    cell_w, cell_h = 1.15, 0.85
    agg_gap = 0.6

    x_positions = {}
    x_curr = 0
    section_boundaries = []

    for i, col in enumerate(ordered_cols):
        if i > 0:
            prev_col = ordered_cols[i - 1]
            if (
                (prev_col in bio_metrics and col in batch_metrics)
                or (prev_col in batch_metrics and col in agg_metrics)
            ):
                # divider exactly centered in the gap
                section_boundaries.append(x_curr + agg_gap / 2)
                x_curr += agg_gap

        x_positions[col] = x_curr
        x_curr += 1

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(cell_w * (x_curr + 1), cell_h * (n_rows + 4)))

    # ------------------------------------------------------------------
    # 4. Draw data
    # ------------------------------------------------------------------
    for i, (method_name, row) in enumerate(df.iterrows()):
        y_coord = i + 0.5
        for col in df.columns:
            x_c = x_positions[col]
            val = row[col]
            if np.isnan(val):
                continue

            if col in agg_metrics:
                ax.barh(
                    y_coord,
                    val,
                    left=x_c,
                    height=cell_h * 0.55,
                    color="#a6bddb",
                    align="center",
                    zorder=2,
                )
                ax.text(
                    x_c + 0.05,
                    y_coord,
                    f"{val:.2f}",
                    va="center",
                    ha="left",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_c + 0.5,
                    y_coord,
                    s=1100,
                    c=[plt.cm.viridis(val)],
                    edgecolors="none",
                    zorder=2,
                )
                ax.text(
                    x_c + 0.5,
                    y_coord,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if val < 0.3 or val > 0.8 else "black",
                    zorder=3,
                )

    # ------------------------------------------------------------------
    # 5. Section dividers (shorter, centered)
    # ------------------------------------------------------------------
    for boundary in section_boundaries:
        ax.plot(
            [boundary, boundary],
            [0.2, n_rows - 0.2],
            color="gray",
            linestyle="-",
            linewidth=2.0,
            alpha=0.6,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # 6. Headers (lowered)
    # ------------------------------------------------------------------
    def draw_header(cols, title, y_top=-1.4, y_sub=-0.35):
        if not cols:
            return
        pos = [x_positions[c] for c in cols]
        center = (min(pos) + max(pos) + 1) / 2

        ax.text(
            center,
            y_top,
            title,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=15,
        )

        for c in cols:
            display_name = c.replace(" ", "\n")
            ax.text(
                x_positions[c] + 0.5,
                y_sub,
                display_name,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="medium",
            )

    draw_header(bio_metrics, "Biological Conservation")
    draw_header(batch_metrics, "Batch Correction")
    draw_header(agg_metrics, "Aggregate Scores")

    # ------------------------------------------------------------------
    # 7. Final styling
    # ------------------------------------------------------------------
    ax.set_ylim(-2.4, n_rows)
    ax.set_xlim(0, x_curr)
    ax.invert_yaxis()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # Row labels with baseline annotation
    new_labels = []
    for idx in df.index:
        idx_s = str(idx)
        if "unintegrated" in idx_s.lower():
            new_labels.append(f"{idx_s}\n(baseline)")
        elif "bbknn" in idx_s.lower():
            new_labels.append(f"{idx_s}\n(graph baseline)")
        else:
            new_labels.append(idx_s)

    ax.set_yticklabels(new_labels, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    record_plot_artifact(stem, Path("integration"), fig=fig)


# -------------------------------------------------------------------------
# CLUSTERING RESOLUTION / STABILITY PLOTS
# -------------------------------------------------------------------------
@collect_plot_artifacts
def plot_clustering_resolution_sweep(
        resolutions: np.ndarray,
        silhouette_scores: List[float],
        n_clusters: List[int],
        penalized_scores: List[float],
        figdir: Path,
) -> None:
    """Plot silhouette, #clusters, and penalized score across resolutions."""
    resolutions = np.array([float(r) for r in resolutions], dtype=float)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    ax = axs[0]
    _clean_axes(ax)
    ax.plot(resolutions, silhouette_scores, marker="o")
    ax.set_title("Silhouette score")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")

    ax = axs[1]
    _clean_axes(ax)
    ax.plot(resolutions, n_clusters, marker="o")
    ax.set_title("Number of clusters")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Clusters")

    ax = axs[2]
    _clean_axes(ax)
    ax.plot(resolutions, penalized_scores, marker="o")
    ax.set_title("Penalized score\n(silhouette - α·N)")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")

    fig.tight_layout()
    record_plot_artifact("clustering_resolution_sweep", figdir)


@collect_plot_artifacts
def plot_clustering_stability_ari(
        stability_aris: List[float],
        figdir: Path,
) -> None:
    """Line plot of ARI vs repetition for subsampling stability."""
    if not stability_aris:
        return

    repeats = np.arange(1, len(stability_aris) + 1)
    mean_ari = float(np.mean(stability_aris))

    fig, ax = plt.subplots(figsize=(5, 4))
    _clean_axes(ax)
    ax.plot(repeats, stability_aris, marker="o", label="ARI")
    ax.axhline(mean_ari, color="red", linestyle="--", label=f"Mean ARI = {mean_ari:.3f}")
    ax.set_title("Subsampling stability (ARI)")
    ax.set_xlabel("Repeat")
    ax.set_ylabel("ARI with full data")
    ax.legend(frameon=False)

    fig.tight_layout()
    record_plot_artifact("clustering_stability_ari", figdir)

#------------------------------------------
# Plot UMAPs
#------------------------------------------
@collect_plot_artifacts
def plot_cluster_umaps(
    adata,
    label_key: str,
    batch_key: str,
    figdir: Path,
) -> None:

    _sanitize_uns_colors(adata, label_key)
    if batch_key is not None and batch_key in adata.obs:
        _sanitize_uns_colors(adata, batch_key)

    fig = sc.pl.umap(adata, color=[label_key], show=False, return_fig=True, legend_loc="right margin")
    save_umap_multi(f"cluster_umap_{label_key}", figdir, fig, right=0.78)

    if batch_key in adata.obs:
        fig = sc.pl.umap(adata, color=[batch_key], show=False, return_fig=True, legend_loc="right margin")
        save_umap_multi(f"cluster_umap_{batch_key}", figdir, fig, right=0.78)

        fig = sc.pl.umap(
            adata,
            color=[batch_key, label_key],
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"cluster_umap_{batch_key}_and_{label_key}", figdir, fig, right=0.78)



@collect_plot_artifacts
def plot_integration_umaps(
    adata,
    embedding_keys,
    batch_key: str,
    color: str,
    *,
    selected_embedding: str | None = None,
    comparison_max_cells: int = 100000,
) -> None:
    """
    Plot UMAPs for unintegrated + integrated embeddings.

    Core behavior ONLY:
      - For each embedding in embedding_keys:
          * recompute neighbors + UMAP (or BBKNN graph)
          * save single UMAP: integration/umap_<emb>.png
          * save side-by-side vs Unintegrated (if available and emb != Unintegrated):
              integration/umap_<emb>_vs_Unintegrated.png
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import scanpy as sc
    import logging
    import time

    LOGGER = logging.getLogger(__name__)
    base = Path("integration")

    n_obs = int(getattr(adata, "n_obs", 0) or 0)
    do_comparison = n_obs <= comparison_max_cells
    LOGGER.info(
        "Integration UMAPs: n_obs=%d, comparison=%s (cutoff=%d)",
        n_obs,
        "on" if do_comparison else "off",
        comparison_max_cells,
    )

    umap_unintegrated = None
    if "Unintegrated" in adata.obsm:
        LOGGER.info("Integration UMAPs: start Unintegrated")
        t0 = time.time()
        try:
            sc.pp.neighbors(adata, use_rep="Unintegrated")
            sc.tl.umap(adata)
            umap_unintegrated = adata.obsm["X_umap"].copy()

            fig = sc.pl.umap(
                adata,
                color=color,
                title="Unintegrated",
                show=False,
                return_fig=True,
            )
            save_umap_multi("umap_Unintegrated", figdir=base, fig=fig)
        except Exception as e:
            LOGGER.warning("Failed to plot UMAP for Unintegrated: %s", e)
        finally:
            LOGGER.info("Integration UMAPs: done Unintegrated (%.1fs)", time.time() - t0)
    else:
        LOGGER.warning("Embedding 'Unintegrated' missing; skipping its UMAP")

    for emb in embedding_keys:
        if emb == "Unintegrated":
            continue
        if emb not in adata.obsm and emb != "BBKNN":
            LOGGER.warning("Embedding '%s' missing; skipping", emb)
            continue

        LOGGER.info("Integration UMAPs: start %s", emb)
        t0 = time.time()
        try:
            if emb == "BBKNN":
                import bbknn

                bbknn.bbknn(adata, batch_key=batch_key, use_rep="X_pca")
            else:
                sc.pp.neighbors(adata, use_rep=emb)

            sc.tl.umap(adata)
            umap_current = adata.obsm["X_umap"].copy()

            fig = sc.pl.umap(
                adata,
                color=color,
                title=emb,
                show=False,
                return_fig=True,
            )
            save_umap_multi(f"umap_{emb}", figdir=base, fig=fig)

            if do_comparison and umap_unintegrated is not None:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                adata.obsm["X_umap"] = umap_current
                sc.pl.umap(
                    adata,
                    color=color,
                    ax=axs[0],
                    show=False,
                    title=emb,
                    legend_loc=None,
                )

                adata.obsm["X_umap"] = umap_unintegrated
                sc.pl.umap(
                    adata,
                    color=color,
                    ax=axs[1],
                    show=False,
                    title="Unintegrated",
                    legend_loc="on data",
                )

                adata.obsm["X_umap"] = umap_current
                save_umap_multi(
                    f"umap_{emb}_vs_Unintegrated",
                    figdir=base,
                    fig=fig,
                    right=0.92,
                )

        except Exception as e:
            LOGGER.warning("Failed to plot UMAP for %s: %s", emb, e)
        finally:
            LOGGER.info("Integration UMAPs: done %s (%.1fs)", emb, time.time() - t0)

    if selected_embedding:
        try:
            LOGGER.info("Integration UMAPs: restore selected embedding '%s'", selected_embedding)
            if selected_embedding == "BBKNN":
                import bbknn

                bbknn.bbknn(adata, batch_key=batch_key, use_rep="X_pca")
            else:
                sc.pp.neighbors(adata, use_rep=selected_embedding)
            sc.tl.umap(adata)
        except Exception as e:
            LOGGER.warning("Failed to restore selected embedding '%s': %s", selected_embedding, e)

@collect_plot_artifacts
def plot_annotated_run_umaps(
    adata,
    *,
    batch_key: str,
    final_label_key: str,
    round_id: str,
    figdir="integration",
) -> None:
    """
    Emit 58annotated-run UMAP plots:

      1) Pre  (full legend; full pretty labels)
      2) Post (full legend; full pretty labels)
      3) Pre  (short legend; Cnn labels on clusters; no legend)
      4) Post (short legend; Cnn labels on clusters; no legend)
      5) Pre vs Post (short legend; 2-panel; Cnn labels; no legend)
      6) Pre batchkey
      7) Post batchkey
      8) Pre vs Post batchkey

    Requires:
      - adata.obsm["X_umap__pre_annotated_run"] exists (pre)
      - adata.obsm["X_umap"] exists (post/active)

    Uses save_umap_multi(...) saver (as in your existing plot_utils).
    """
    from pathlib import Path
    import logging
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import scanpy as sc

    LOGGER = logging.getLogger(__name__)
    base = Path(figdir)

    # ----------------------------
    # Validate inputs
    # ----------------------------
    if "X_umap" not in adata.obsm:
        raise KeyError("plot_annotated_run_umaps: adata.obsm['X_umap'] missing (post/active UMAP).")
    if "X_umap__pre_annotated_run" not in adata.obsm:
        raise KeyError("plot_annotated_run_umaps: adata.obsm['X_umap__pre_annotated_run'] missing (pre UMAP stash).")
    if final_label_key not in adata.obs:
        raise KeyError(f"plot_annotated_run_umaps: final_label_key={final_label_key!r} missing from adata.obs.")
    if batch_key not in adata.obs:
        raise KeyError(f"plot_annotated_run_umaps: batch_key={batch_key!r} missing from adata.obs.")

    rid = str(round_id)

    pre_umap = np.asarray(adata.obsm["X_umap__pre_annotated_run"])
    post_umap = np.asarray(adata.obsm["X_umap"])

    # ----------------------------
    # Helpers
    # ----------------------------
    def _sanitize_tag(s: str) -> str:
        s = str(s or "").strip()
        s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
        return s or "NA"

    tag = _sanitize_tag(rid)

    def _extract_cnn(x: str) -> str:
        """
        Extract the Cnn token from a pretty label string.
        Examples:
          "C07 Astrocytes" -> "C07"
          "Astrocytes C07" -> "C07"
        Fallback: first token, truncated.
        """
        s = str(x or "")
        m = re.search(r"\b(C\d+)\b", s)
        if m:
            return m.group(1)
        # fallback: maybe it starts like "C07_..."
        m2 = re.search(r"(C\d+)", s)
        if m2:
            return m2.group(1)
        return s.split()[0][:8] if s.strip() else "C?"

    def _annotate_cluster_short_ids(ax, umap_xy: np.ndarray, labels: np.ndarray) -> None:
        """
        Place Cnn label at per-cluster centroid in the provided UMAP coords.
        """
        labs = np.asarray(labels).astype(str)
        xy = np.asarray(umap_xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            return

        uniq = pd_unique_stable(labs)
        for lab in uniq:
            m = labs == lab
            if not np.any(m):
                continue
            cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
            ax.text(
                float(cx),
                float(cy),
                _extract_cnn(lab),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65),
                zorder=10,
            )

    def pd_unique_stable(arr: np.ndarray) -> list[str]:
        # stable unique without pandas dependency
        seen = set()
        out = []
        for x in arr.tolist():
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _tmp_with_umap(X_umap: np.ndarray):
        tmp = adata.copy()
        tmp.obsm["X_umap"] = np.asarray(X_umap)
        return tmp

    title_pre = f"Pre scANVI annotated integration on {rid}"
    title_post = f"Post scANVI annotated integration on {rid}"

    # ----------------------------
    # 1) PRE (full legend; full labels)
    # ----------------------------
    try:
        tmp_pre = _tmp_with_umap(pre_umap)
        fig_pre_full = sc.pl.umap(
            tmp_pre,
            color=final_label_key,
            title=title_pre,
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"umap_pre__{tag}__fulllegend", figdir=base, fig=fig_pre_full)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (pre/fulllegend) failed: %s", e)

    # ----------------------------
    # 2) POST (full legend; full labels)
    # ----------------------------
    try:
        tmp_post = _tmp_with_umap(post_umap)
        fig_post_full = sc.pl.umap(
            tmp_post,
            color=final_label_key,
            title=title_post,
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"umap_post__{tag}__fulllegend", figdir=base, fig=fig_post_full)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (post/fulllegend) failed: %s", e)

    # ----------------------------
    # 3) PRE (short legend; Cnn labels; no legend)
    # ----------------------------
    try:
        tmp_pre = _tmp_with_umap(pre_umap)
        fig_pre_short = sc.pl.umap(
            tmp_pre,
            color=final_label_key,
            title=title_pre,
            show=False,
            return_fig=True,
            legend_loc=None,
        )
        ax = fig_pre_short.axes[0] if fig_pre_short.axes else plt.gca()
        _annotate_cluster_short_ids(ax, pre_umap, tmp_pre.obs[final_label_key].astype(str).to_numpy())
        save_umap_multi(f"umap_pre__{tag}__shortlegend", figdir=base, fig=fig_pre_short)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (pre/shortlegend) failed: %s", e)

    # ----------------------------
    # 4) POST (short legend; Cnn labels; no legend)
    # ----------------------------
    try:
        tmp_post = _tmp_with_umap(post_umap)
        fig_post_short = sc.pl.umap(
            tmp_post,
            color=final_label_key,
            title=title_post,
            show=False,
            return_fig=True,
            legend_loc=None,
        )
        ax = fig_post_short.axes[0] if fig_post_short.axes else plt.gca()
        _annotate_cluster_short_ids(ax, post_umap, tmp_post.obs[final_label_key].astype(str).to_numpy())
        save_umap_multi(f"umap_post__{tag}__shortlegend", figdir=base, fig=fig_post_short)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (post/shortlegend) failed: %s", e)

    # ----------------------------
    # 5) PRE vs POST (2-panel; short legend; Cnn labels; no legend)
    # ----------------------------
    try:
        # Make a temp object just to reuse sc.pl.umap on specific axes
        tmp = adata.copy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Left: PRE
        tmp.obsm["X_umap"] = pre_umap
        sc.pl.umap(
            tmp,
            color=final_label_key,
            ax=axs[0],
            show=False,
            title=title_pre,
            legend_loc=None,
        )
        _annotate_cluster_short_ids(
            axs[0],
            pre_umap,
            tmp.obs[final_label_key].astype(str).to_numpy(),
        )

        # Right: POST
        tmp.obsm["X_umap"] = post_umap
        sc.pl.umap(
            tmp,
            color=final_label_key,
            ax=axs[1],
            show=False,
            title=title_post,
            legend_loc=None,
        )
        _annotate_cluster_short_ids(
            axs[1],
            post_umap,
            tmp.obs[final_label_key].astype(str).to_numpy(),
        )

        save_umap_multi(
            f"umap_pre_vs_post__{tag}__shortlegend",
            figdir=base,
            fig=fig,
            right=0.92,
        )

    except Exception as e:
        LOGGER.warning("Annotated UMAP (pre-vs-post/shortlegend) failed: %s", e)

    # ----------------------------
    # 6) PRE (batch; full legend)
    # ----------------------------
    try:
        tmp_pre = _tmp_with_umap(pre_umap)
        fig_pre_batch = sc.pl.umap(
            tmp_pre,
            color=batch_key,
            title=f"Pre scANVI annotated integration on {rid} (batch)",
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"umap_pre__{tag}__batch__fulllegend", figdir=base, fig=fig_pre_batch)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (pre/batch/fulllegend) failed: %s", e)

    # ----------------------------
    # 7) POST (batch; full legend)
    # ----------------------------
    try:
        tmp_post = _tmp_with_umap(post_umap)
        fig_post_batch = sc.pl.umap(
            tmp_post,
            color=batch_key,
            title=f"Post scANVI annotated integration on {rid} (batch)",
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"umap_post__{tag}__batch__fulllegend", figdir=base, fig=fig_post_batch)
    except Exception as e:
        LOGGER.warning("Annotated UMAP (post/batch/fulllegend) failed: %s", e)

    # ----------------------------
    # 8) PRE vs POST (batch; 2-panel; no legend)
    # ----------------------------
    try:
        tmp = adata.copy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        tmp.obsm["X_umap"] = pre_umap
        sc.pl.umap(
            tmp,
            color=batch_key,
            ax=axs[0],
            show=False,
            title=f"Pre scANVI annotated integration on {rid} (batch)",
            legend_loc=None,
        )

        tmp.obsm["X_umap"] = post_umap
        sc.pl.umap(
            tmp,
            color=batch_key,
            ax=axs[1],
            show=False,
            title=f"Post scANVI annotated integration on {rid} (batch)",
            legend_loc=None,
        )

        save_umap_multi(
            f"umap_pre_vs_post__{tag}__batch",
            figdir=base,
            fig=fig,
            right=0.92,
        )
    except Exception as e:
        LOGGER.warning("Annotated UMAP (pre-vs-post/batch) failed: %s", e)


# ----------------------------------------------------------------------
# Cluster tree
# ----------------------------------------------------------------------
import networkx as nx

@collect_plot_artifacts
def plot_cluster_tree(
        labels_per_resolution: Mapping[str, np.ndarray],
        resolutions: Sequence[float | str],
        figdir: Path | str,
        best_resolution: float | None = None,
        palette: list[str] | None = None,
        min_frac: float = 0.05,
        stem: str = "cluster_tree",
) -> None:
    """
    Cluster evolution tree using NetworkX with hierarchical forced layout.

    Nodes: (res_key, cluster_id)
    Edges: flows between clusters across adjacent resolutions.

    """
    figdir = _ensure_path(figdir)

    # ------------------------------------------------------------------
    # 0. Resolution normalization
    # ------------------------------------------------------------------
    res_sorted = _sorted_resolutions(resolutions)
    if len(res_sorted) < 2:
        LOGGER.warning("cluster_tree: need >=2 resolutions.")
        return

    res_keys = [_res_key(r) for r in res_sorted]

    # ------------------------------------------------------------------
    # 1. Build directed graph (edges = cluster flows)
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    cluster_sizes: Dict[tuple[str, int], int] = {}

    # Determine color function (cluster_id → color)
    if palette is not None:
        # Use provided palette directly
        palette = list(palette)
        color_lookup = lambda cid: palette[cid % len(palette)]
    else:
        # Fallback to tab20 if no palette provided
        cmap = plt.get_cmap("tab20")
        color_lookup = lambda cid: cmap(cid % cmap.N)

    for i in range(len(res_keys) - 1):
        k1, k2 = res_keys[i], res_keys[i + 1]

        if k1 not in labels_per_resolution or k2 not in labels_per_resolution:
            continue

        # Ensure integer cluster labels to make Series indices consistent
        labels1 = np.asarray(labels_per_resolution[k1]).astype(int)
        labels2 = np.asarray(labels_per_resolution[k2]).astype(int)

        df = pd.DataFrame({"r1": labels1, "r2": labels2})

        # counts over (r1, r2) pairs; r1/r2 are ints
        counts = df.value_counts().reset_index(name="n")

        # cluster sizes per resolution; indices are ints
        size_r1 = df["r1"].value_counts().sort_index()
        size_r2 = df["r2"].value_counts().sort_index()

        # Add all nodes from the first resolution level (only once)
        if i == 0:
            for cid in size_r1.index:
                cid_int = int(cid)
                node = (k1, cid_int)
                sz = int(size_r1.loc[cid_int])
                G.add_node(node, res_level=i, size=sz)
                cluster_sizes[node] = sz

        # Add edges + any new nodes in k2
        for _, row in counts.iterrows():
            c1 = int(row["r1"])
            c2 = int(row["r2"])
            n = int(row["n"])

            base_size = float(size_r1.loc[c1])
            frac = n / base_size if base_size > 0 else 0.0

            if frac < min_frac:
                continue

            node1 = (k1, c1)
            node2 = (k2, c2)

            # Ensure nodes exist with consistent sizes
            if node1 not in G:
                sz1 = int(size_r1.loc[c1])
                G.add_node(node1, res_level=i, size=sz1)
                cluster_sizes[node1] = sz1
            if node2 not in G:
                sz2 = int(size_r2.loc[c2])
                G.add_node(node2, res_level=i + 1, size=sz2)
                cluster_sizes[node2] = sz2

            # Add weighted/colored edge
            G.add_edge(
                node1,
                node2,
                weight=float(frac),
                flow=n,
                color=color_lookup(c1),
            )

    # ------------------------------------------------------------------
    # 2. Compute hierarchical positions
    # ------------------------------------------------------------------
    nodes_by_level: Dict[int, list] = {}
    for node in G.nodes:
        lvl = G.nodes[node]["res_level"]
        nodes_by_level.setdefault(lvl, []).append(node)

    # Compute x-coordinates centered per level
    x_pos: Dict[tuple[str, int], float] = {}
    for lvl, nodes in nodes_by_level.items():
        nodes_sorted = sorted(nodes, key=lambda x: x[1])
        n = len(nodes_sorted)
        xs = np.linspace(-0.5 * (n - 1), 0.5 * (n - 1), n)
        for node, x in zip(nodes_sorted, xs):
            x_pos[node] = float(x)

    # y-coordinate = level index (flipped vertically)
    y_pos = {node: -float(G.nodes[node]["res_level"]) for node in G.nodes}
    pos = {node: (x_pos[node], y_pos[node]) for node in G.nodes}

    # ------------------------------------------------------------------
    # 3. Node visuals
    # ------------------------------------------------------------------
    node_colors = [color_lookup(node[1]) for node in G.nodes]
    node_sizes = [30 + 6.0 * np.sqrt(G.nodes[node]["size"]) for node in G.nodes]

    # ------------------------------------------------------------------
    # 4. Edge visuals (same as before, but using already-computed attrs)
    # ------------------------------------------------------------------
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    for u, v, d in G.edges(data=True):
        w = float(d["weight"])  # fraction 0–1
        color = d.get("color", "gray")

        # Logistic width scaling
        width = 1.0 + 6.0 / (1.0 + np.exp(-8 * (w - 0.25)))
        width = float(min(width, 6.0))

        # Alpha scaling with clamping
        alpha = 1 / (1 + np.exp(-6 * (w - 0.15)))
        alpha = float(max(0.05, min(alpha, 1.0)))

        edge_colors.append(color)
        edge_widths.append(width)
        edge_alphas.append(alpha)

    edge_colors = np.array(edge_colors, dtype=object).tolist()
    edge_widths = np.array(edge_widths, dtype=float).tolist()
    edge_alphas = np.array(edge_alphas, dtype=float).tolist()

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(res_sorted)))

    lc = nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=edge_alphas,
        arrows=False,
        ax=ax,
    )
    if lc is not None:
        try:
            lc.set_zorder(1)
        except AttributeError:
            pass

    nc = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.4,
        ax=ax,
    )
    try:
        nc.set_zorder(3)
    except AttributeError:
        pass

    # ------------------------------------------------------------------
    # 6. Styling
    # ------------------------------------------------------------------
    ax.set_title("Cluster Evolution (NetworkX DAG)", fontsize=13)
    ax.set_xticks([])

    y_ticks = sorted(set(y_pos.values()))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{res_sorted[i]:.3f}" for i in range(len(res_sorted))])
    ax.set_ylabel("Resolution")

    # ------------------------------------------------------------------
    # 7. Highlight best resolution with horizontal dotted line + text box
    # ------------------------------------------------------------------
    if best_resolution is not None:
        r = float(best_resolution)

        # Find exact or closest resolution tick
        idx = np.argmin([abs(r - rr) for rr in res_sorted])
        y = -idx  # negative index = y-coordinate in the plot

        # Horizontal dotted line across entire width
        ax.axhline(
            y=y,
            color="red",
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            zorder=50,
        )

        # Annotation text box on the right side
        ax.text(
            ax.get_xlim()[1],  # far right
            y,
            f"Best res = {r:.2f}",
            color="red",
            fontsize=12,
            ha="right",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="red",
                boxstyle="round,pad=0.3",
                alpha=0.85,
            ),
            zorder=51,
        )

    ax.grid(False)
    plt.tight_layout()

    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})


@collect_plot_artifacts
def plot_compaction_flow(
    adata: "ad.AnnData",
    *,
    parent_round_id: str,
    child_round_id: str,
    figdir: "Path | str",
    min_frac: float = 0.02,
    stem: str = "compaction_flow",
    palette: list[str] | None = None,
    title: str | None = None,
) -> None:
    """
    Visualize compaction as a 2-level flow graph (parent clusters -> compacted clusters).

    Patch (Option A):
      - node labels use round cluster_display_map when available,
        so the flow chart matches UMAP/decoupler "pretty" cluster labels.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    figdir = _ensure_path(figdir)

    # Resolve round metadata + obs keys
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict):
        LOGGER.warning("compaction_flow: adata.uns['cluster_rounds'] missing/invalid.")
        return
    if parent_round_id not in rounds or child_round_id not in rounds:
        LOGGER.warning("compaction_flow: rounds %r or %r not found.", parent_round_id, child_round_id)
        return

    parent_round = rounds[parent_round_id]
    child_round = rounds[child_round_id]

    parent_obs_key = str(parent_round.get("labels_obs_key", ""))
    child_obs_key = str(child_round.get("labels_obs_key", ""))

    if parent_obs_key not in adata.obs or child_obs_key not in adata.obs:
        LOGGER.warning(
            "compaction_flow: required obs keys not found: parent=%r child=%r",
            parent_obs_key,
            child_obs_key,
        )
        return

    # ------------------------------------------------------------------
    # Display labels (pretty labels)
    # ------------------------------------------------------------------
    parent_dm = _round_display_map(parent_round)
    child_dm = _round_display_map(child_round)

    # robust color lookup by cluster id
    if palette is not None:
        palette = list(palette)

        def color_lookup(cid: str):
            idx = _safe_cluster_color_index(cid)
            return palette[idx % len(palette)]

    else:
        cmap = plt.get_cmap("tab20")

        def color_lookup(cid: str):
            idx = _safe_cluster_color_index(cid)
            return cmap(idx % cmap.N)

    parent = adata.obs[parent_obs_key].astype(str).to_numpy()
    child = adata.obs[child_obs_key].astype(str).to_numpy()

    df = pd.DataFrame({"parent": parent, "child": child})
    counts = df.value_counts().reset_index(name="n")

    parent_sizes = df["parent"].value_counts().to_dict()
    child_sizes = df["child"].value_counts().to_dict()

    # Build graph
    G = nx.DiGraph()

    # Add nodes
    for p, sz in parent_sizes.items():
        node = ("parent", str(p))
        G.add_node(node, level=0, size=int(sz), kind="parent")
    for c, sz in child_sizes.items():
        node = ("child", str(c))
        G.add_node(node, level=1, size=int(sz), kind="child")

    # Add edges
    for _, row in counts.iterrows():
        p = str(row["parent"])
        c = str(row["child"])
        n = int(row["n"])
        base = float(parent_sizes.get(p, 0))
        frac = (n / base) if base > 0 else 0.0
        if frac < float(min_frac):
            continue

        G.add_edge(
            ("parent", p),
            ("child", c),
            weight=float(frac),
            flow=int(n),
            color=color_lookup(p),
        )

    if G.number_of_edges() == 0:
        LOGGER.warning("compaction_flow: no edges above min_frac=%.3f", float(min_frac))
        return

    # Layout: two columns, sorted by size (descending)
    parents_sorted = sorted(parent_sizes.keys(), key=lambda k: (-int(parent_sizes[k]), str(k)))
    children_sorted = sorted(child_sizes.keys(), key=lambda k: (-int(child_sizes[k]), str(k)))

    x_parent, x_child = 0.0, 1.8

    # vertical positions: spread evenly
    y_parent = np.linspace(0, -max(1, len(parents_sorted) - 1), len(parents_sorted))
    y_child = np.linspace(0, -max(1, len(children_sorted) - 1), len(children_sorted))

    pos = {}
    for yy, p in zip(y_parent, parents_sorted):
        pos[("parent", str(p))] = (x_parent, float(yy))
    for yy, c in zip(y_child, children_sorted):
        pos[("child", str(c))] = (x_child, float(yy))

    # Node visuals
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        kind = G.nodes[node].get("kind", "")
        if kind == "parent":
            node_colors.append(color_lookup(node[1]))
        else:
            node_colors.append("#BBBBBB")  # compacted nodes neutral
        node_sizes.append(40 + 7.0 * np.sqrt(float(G.nodes[node].get("size", 1))))

    # Edge visuals
    edge_colors, edge_widths, edge_alphas = [], [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        edge_colors.append(d.get("color", "gray"))

        width = 0.8 + 7.0 / (1.0 + np.exp(-10 * (w - 0.2)))
        width = float(min(width, 7.0))
        edge_widths.append(width)

        alpha = 1.0 / (1.0 + np.exp(-8 * (w - 0.12)))
        alpha = float(max(0.05, min(alpha, 1.0)))
        edge_alphas.append(alpha)

    # ------------------------------------------------------------------
    # Labels: Option A (pretty labels from display maps)
    # ------------------------------------------------------------------
    parent_labels = {str(p): parent_dm.get(str(p), str(p)) for p in parents_sorted}
    child_labels = {str(c): child_dm.get(str(c), str(c)) for c in children_sorted}

    parent_labels = _make_unique_labels(parent_labels)
    child_labels = _make_unique_labels(child_labels)

    def _label(node):
        kind, cid = node
        if kind == "parent":
            return parent_labels.get(str(cid), str(cid))
        return child_labels.get(str(cid), str(cid))

    # Plot
    fig_h = max(5.0, 0.35 * max(len(parents_sorted), len(children_sorted)))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    lc = nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=edge_alphas,
        arrows=False,
        ax=ax,
    )
    if lc is not None:
        try:
            lc.set_zorder(1)
        except Exception:
            pass

    nc = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.4,
        ax=ax,
    )
    try:
        nc.set_zorder(3)
    except Exception:
        pass

    # annotate only nodes
    for node, (x, y) in pos.items():
        ax.text(
            x + 0.02,
            y,
            _label(node),
            fontsize=9,
            ha="left",
            va="center",
            color="black",
            zorder=10,
        )

    ttl = title or f"Compaction flow: {parent_round_id} → {child_round_id}"
    ax.set_title(ttl, fontsize=13)
    ax.set_xticks([x_parent, x_child])
    ax.set_xticklabels([parent_round_id, child_round_id])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    plt.tight_layout()

    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})



# ----------------------------------------------------------------------
# Stability curves (silhouette, stability, composite, tiny penalty)
# ----------------------------------------------------------------------
@collect_plot_artifacts
def plot_stability_curves(
        resolutions: Sequence[float | str],
        silhouette: Mapping[Any, Any],
        stability: Mapping[Any, Any],
        composite: Mapping[Any, Any],
        tiny_cluster_penalty: Mapping[Any, Any],
        best_resolution: float | str,
        plateaus: Sequence[Mapping[str, object]] | None,
        figdir: Path | str,
        stem: str = "cluster_selection_stability",
) -> None:
    """
    Plot structural + (optionally) biological metrics vs resolution.

    Structural components:
      - silhouette (centroid-based)
      - stability (smoothed ARI)
      - composite score (actual one used for selection)
      - tiny-cluster penalty
    """
    res_sorted = _sorted_resolutions(resolutions)

    sil = _extract_series(res_sorted, silhouette)
    stab = _extract_series(res_sorted, stability)
    comp = _extract_series(res_sorted, composite)
    tiny = _extract_series(res_sorted, tiny_cluster_penalty)

    fig, ax = plt.subplots(figsize=(8, 5))

    # plateau shading
    for xmin, xmax in _plateau_spans(plateaus or []):
        ax.axvspan(xmin, xmax, color="0.9", alpha=0.5)

    # structural curves
    ax.plot(res_sorted, sil, label="Centroid silhouette", color="tab:blue")
    ax.plot(res_sorted, stab, label="Stability (smoothed ARI)", color="tab:green")
    ax.plot(res_sorted, tiny, label="Tiny-cluster penalty", color="tab:orange")
    ax.plot(res_sorted, comp, label="Composite (used for selection)", color="tab:red")

    ax.axvline(float(best_resolution), color="k", linestyle="--")

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")
    ax.set_title("Cluster selection metrics vs resolution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    record_plot_artifact(stem, _ensure_path(figdir), fig)


@collect_plot_artifacts
def plot_biological_metrics(
        resolutions: Sequence[float | str],
        bio_homogeneity: Mapping[Any, Any],
        bio_fragmentation: Mapping[Any, Any],
        bio_ari: Mapping[Any, Any],
        selection_config: Mapping[str, Any],
        best_resolution: float | str,
        plateaus: Sequence[Mapping[str, object]] | None,
        figdir: Path | str,
        figure_formats: Sequence[str] = ("png", "pdf"),
        stem: str = "biological_metrics",
) -> None:
    """
    Biological-metrics-focused view:

    - normalized homogeneity
    - normalized fragmentation (inverted, higher=better)
    - normalized bio-ARI
    - biological composite (using selection weights)
    """
    res_sorted = _sorted_resolutions(resolutions)

    hom_raw = _extract_series(res_sorted, bio_homogeneity)
    frag_raw = _extract_series(res_sorted, bio_fragmentation)
    ari_raw = _extract_series(res_sorted, bio_ari)

    hom_norm = _normalize_array(hom_raw)
    frag_norm = _normalize_array(frag_raw)
    frag_good = 1.0 - frag_norm
    ari_norm = _normalize_array(ari_raw)

    w_hom = float(selection_config.get("w_hom", 0.0))
    w_frag = float(selection_config.get("w_frag", 0.0))
    w_bioari = float(selection_config.get("w_bioari", 0.0))

    bio_comp = w_hom * hom_norm + w_frag * frag_good + w_bioari * ari_norm

    fig, ax = plt.subplots(figsize=(8, 5))

    for xmin, xmax in _plateau_spans(plateaus or []):
        ax.axvspan(xmin, xmax, color="0.9", alpha=0.5)

    ax.plot(res_sorted, hom_norm, label="Homogeneity (norm.)", color="purple")
    ax.plot(
        res_sorted,
        frag_good,
        label="Fragmentation (low→high, norm.)",
        color="brown",
    )
    ax.plot(res_sorted, ari_norm, label="Bio-ARI (norm.)", color="magenta")
    ax.plot(
        res_sorted,
        bio_comp,
        label="Biological composite (norm.)",
        color="black",
        linestyle="--",
    )

    ax.axvline(float(best_resolution), color="k", linestyle="--")

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Normalized score")
    ax.set_title("Biological metrics vs resolution (bio-guided clustering)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    record_plot_artifact(stem, _ensure_path(figdir), fig)


@collect_plot_artifacts
def plot_composite_only(
        resolutions: Sequence[float | str],
        structural_comp: Mapping[Any, Any],
        biological_comp: Mapping[Any, Any] | None,
        total_comp: Mapping[Any, Any],
        best_resolution: float | str,
        plateaus: Sequence[Mapping[str, object]] | None,
        figdir: Path | str,
        stem: str = "composite_scores",
) -> None:
    """
    Plot only composite curves:
      - structural composite (norm.)
      - biological composite (norm.) [if exists]
      - total composite (norm.)

    Clean 3-line diagnostic.
    """
    res_sorted = _sorted_resolutions(resolutions)

    struct_raw = _extract_series(res_sorted, structural_comp)
    struct_norm = _normalize_array(struct_raw)

    total_raw = _extract_series(res_sorted, total_comp)
    total_norm = _normalize_array(total_raw)

    if biological_comp is not None:
        bio_raw = _extract_series(res_sorted, biological_comp)
        bio_norm = _normalize_array(bio_raw)
    else:
        bio_norm = None

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # plateau shading
    for xmin, xmax in _plateau_spans(plateaus or []):
        ax.axvspan(xmin, xmax, color="0.9", alpha=0.5)

    # main curves
    ax.plot(
        res_sorted,
        struct_norm,
        label="Structural composite (norm.)",
        color="tab:blue",
        linewidth=2,
    )

    if bio_norm is not None:
        ax.plot(
            res_sorted,
            bio_norm,
            label="Biological composite (norm.)",
            color="tab:green",
            linewidth=2,
        )

    ax.plot(
        res_sorted,
        total_norm,
        label="Total composite (norm.)",
        color="tab:red",
        linewidth=2,
    )

    ax.axvline(float(best_resolution), color="k", linestyle="--")

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Normalized composite score")
    ax.set_title("Composite score components")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    record_plot_artifact(stem, _ensure_path(figdir), fig)


# ----------------------------------------------------------------------
# Plateau highlights
# ----------------------------------------------------------------------
@collect_plot_artifacts
def plot_plateau_highlights(
        resolutions: Sequence[float | str],
        silhouette: Mapping[Any, Any],
        stability: Mapping[Any, Any],
        composite: Mapping[Any, Any],
        best_resolution: float | str,
        plateaus: Sequence[Mapping[str, object]] | None,
        figdir: Path | str,
        figure_formats: Sequence[str] = ("png", "pdf"),
        stem: str = "plateau_highlights",
) -> None:
    res_sorted = _sorted_resolutions(resolutions)
    sil = _extract_series(res_sorted, silhouette)
    stab = _extract_series(res_sorted, stability)
    comp = _extract_series(res_sorted, composite)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    def _shade(ax):
        for xmin, xmax in _plateau_spans(plateaus or []):
            ax.axvspan(xmin, xmax, color="0.9", alpha=0.5, zorder=0)
        ax.axvline(float(best_resolution), color="k", linestyle="--", linewidth=1)

    ax = axes[0]
    _shade(ax)
    ax.plot(res_sorted, sil, marker="o", color="tab:blue")
    ax.set_ylabel("Centroid silhouette")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    _shade(ax)
    ax.plot(res_sorted, stab, marker="o", color="tab:green")
    ax.set_ylabel("Smoothed stability (ARI)")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    _shade(ax)
    ax.plot(res_sorted, comp, marker="o", color="tab:red")
    ax.set_ylabel("Composite score")
    ax.set_xlabel("Resolution")
    ax.grid(True, alpha=0.2)

    fig.suptitle("Plateau-aware metrics for resolution selection", y=0.95)
    plt.tight_layout()
    record_plot_artifact(stem, _ensure_path(figdir), fig)


# -------------------------------------------------------------------------
# Cluster-level statistics
# -------------------------------------------------------------------------

@collect_plot_artifacts
def plot_cluster_sizes(
        adata,
        label_key: str,
        figdir: Path,
        tiny_threshold: int = 20,
        stem: str = "cluster_sizes",
):
    """
    Barplot of cluster sizes (absolute), with % label on each bar.
    Bars use the same colors as UMAP cluster coloring.
    """
    if label_key not in adata.obs:
        LOGGER.warning("plot_cluster_sizes: label_key '%s' missing.", label_key)
        return

    counts = adata.obs[label_key].value_counts().sort_index()
    clusters = counts.index.tolist()
    short_labels = [_extract_cnn_token(c) for c in clusters]
    sizes = counts.values.astype(int)
    total = sizes.sum()

    # colors from Scanpy palette
    color_map = _cluster_color_map(adata, label_key)
    if color_map:
        colors = [color_map.get(c) for c in clusters]
        if any(c is None for c in colors):
            cmap = plt.get_cmap("tab20")
            colors = [c if c is not None else cmap(i % cmap.N) for i, c in enumerate(colors)]
    else:
        LOGGER.warning("No cluster palette found for '%s'; using default.", label_key)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(clusters))]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(clusters)), 4))
    _clean_axes(ax)

    x = np.arange(len(clusters))
    bars = ax.bar(x, sizes, color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)

    # percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = 100 * height / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Cells")
    ax.set_title("Cluster sizes")
    _add_outside_legend(ax, clusters, colors, title="Cluster")

    fig.tight_layout()
    _finalize_categorical_x(fig, ax, rotate=45, ha="right", bottom=0.40)
    _reserve_bottom_for_xticklabels(fig, ax, rotation=45, fontsize=9, ha="right")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.34))
    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})
    close_plot(fig)


@collect_plot_artifacts
def plot_cluster_qc_summary(
    adata,
    label_key: str,
    figdir: Path,
    stem: str = "cluster_qc_summary",
):
    if label_key not in adata.obs:
        LOGGER.warning("plot_cluster_qc_summary: '%s' not in obs.", label_key)
        return

    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing = [m for m in metrics if m not in adata.obs]
    if missing:
        LOGGER.warning("Missing QC fields: %s", missing)
        return

    df = adata.obs[[label_key] + metrics].groupby(label_key).mean()
    clusters = df.index.astype(str).tolist()
    short_labels = [_extract_cnn_token(c) for c in clusters]
    color_map = _cluster_color_map(adata, label_key)
    if color_map:
        colors = [color_map.get(c) for c in clusters]
        if any(c is None for c in colors):
            cmap = plt.get_cmap("tab20")
            colors = [c if c is not None else cmap(i % cmap.N) for i, c in enumerate(colors)]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(clusters))]

    n = df.shape[0]
    fig_w = max(10, 0.35 * n)

    # 3 rows, shared x: avoids label clutter completely
    fig, axs = plt.subplots(3, 1, figsize=(fig_w, 7.5), sharex=True)
    fig.set_constrained_layout(False)

    x = np.arange(len(clusters))
    for ax, m in zip(axs, metrics):
        _clean_axes(ax)
        ax.bar(x, df[m].values, color=colors, edgecolor="black")
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("")  # only bottom axis will get label
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    axs[-1].set_xlabel("Cluster")
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(short_labels)

    # Hide x tick labels on upper panels
    for ax in axs[:-1]:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    # Rotate/reserve space ONLY once (bottom axis)
    _reserve_bottom_for_xticklabels(fig, axs[-1], rotation=45, fontsize=9, ha="right")

    # Reduce vertical spacing a bit
    fig.subplots_adjust(hspace=0.25)

    _add_outside_legend(axs[-1], clusters, colors, title="Cluster")
    fig.tight_layout()
    _finalize_categorical_x(fig, axs[-1], rotate=45, ha="right", bottom=0.40)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.34))
    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})
    close_plot(fig)


@collect_plot_artifacts
def plot_cluster_silhouette_by_cluster(
        adata,
        label_key: str,
        embedding_key: str,
        figdir: Path,
        stem: str = "cluster_silhouette_by_cluster",
):
    """
    Violin plot of silhouette values per cluster (true silhouette, not centroid-based).
    """
    if label_key not in adata.obs:
        LOGGER.warning("plot_cluster_silhouette_by_cluster: '%s' missing.", label_key)
        return
    if embedding_key not in adata.obsm:
        LOGGER.warning("Embedding '%s' missing.", embedding_key)
        return

    labels = adata.obs[label_key].to_numpy()
    X = adata.obsm[embedding_key]
    silvals = silhouette_samples(X, labels, metric="euclidean")

    df = pd.DataFrame({"cluster": labels, "silhouette": silvals})
    try:
        clusters = list(adata.obs[label_key].cat.categories)
    except Exception:
        clusters = sorted(df["cluster"].astype(str).unique().tolist())
    clusters = [str(c) for c in clusters]
    short_labels = [_extract_cnn_token(c) for c in clusters]
    color_map = _cluster_color_map(adata, label_key)
    if color_map:
        colors = [color_map.get(c) for c in clusters]
        if any(c is None for c in colors):
            cmap = plt.get_cmap("tab20")
            colors = [c if c is not None else cmap(i % cmap.N) for i, c in enumerate(colors)]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(clusters))]

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * df["cluster"].nunique()), 4.5))
    _clean_axes(ax)

    data = [df.loc[df["cluster"].astype(str) == c, "silhouette"].values for c in clusters]
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True)
    for i, box in enumerate(bp.get("boxes", [])):
        box.set(facecolor="none", edgecolor=colors[i], linewidth=1.2)
    for i, med in enumerate(bp.get("medians", [])):
        med.set(color=colors[i], linewidth=1.2)
    whiskers = bp.get("whiskers", [])
    caps = bp.get("caps", [])
    for i in range(len(colors)):
        for w in whiskers[2 * i: 2 * i + 2]:
            w.set(color=colors[i], linewidth=1.0)
        for c in caps[2 * i: 2 * i + 2]:
            c.set(color=colors[i], linewidth=1.0)
    for i, flier in enumerate(bp.get("fliers", [])):
        flier.set(markeredgecolor=colors[i % len(colors)], markerfacecolor="none")
    plt.suptitle("")  # remove pandas default
    ax.set_title("Silhouette distribution per cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _add_outside_legend(ax, clusters, colors, title="Cluster", face="none")

    fig.tight_layout()
    _finalize_categorical_x(fig, ax, rotate=45, ha="right", bottom=0.45)
    _reserve_bottom_for_xticklabels(fig, ax, rotation=45, fontsize=9, ha="right")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.34))
    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})
    close_plot(fig)


@collect_plot_artifacts
def plot_cluster_batch_composition(
    adata,
    label_key: str,
    batch_key: str,
    figdir: Path,
    stem: str = "cluster_batch_composition",
):
    """
    Stacked barplot showing fraction of each batch within each cluster.
    Legend is placed to the right (outside axes) to avoid overlaying bars.
    """
    if label_key not in adata.obs or batch_key not in adata.obs:
        LOGGER.warning("plot_cluster_batch_composition: required columns missing.")
        return

    df = (
        adata.obs[[label_key, batch_key]]
        .groupby([label_key, batch_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    frac = df.div(df.sum(axis=1), axis=0)
    clusters = frac.index.astype(str).tolist()
    short_labels = [_extract_cnn_token(c) for c in clusters]

    fig, ax = plt.subplots(figsize=(max(8, 0.40 * len(df)), 4.5))
    _clean_axes(ax)

    # Plot WITHOUT pandas legend (we'll add a clean one outside)
    frac.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
        edgecolor="black",
        linewidth=0.3,
        legend=False,
    )

    ax.grid(False)
    ax.set_ylabel("Fraction")
    ax.set_title("Batch composition per cluster")
    ax.set_xticklabels(short_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- Legend to the right, outside the plotting area ---
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        max_rows = 12
        ncol = max(1, math.ceil(len(labels) / max_rows))
        # make room on the right for legend
        fig.subplots_adjust(right=0.80)

        ax.legend(
            handles,
            labels,
            title=batch_key,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            ncol=ncol,
        )

    _reserve_bottom_for_xticklabels(fig, ax, rotation=45, fontsize=9, ha="right")

    # Keep your existing final x-axis layout helper
    _finalize_categorical_x(
        fig,
        ax,
        rotate=45,
        ha="right",
        bottom=max(fig.subplotpars.bottom, 0.42),
    )

    record_plot_artifact(stem, figdir, fig, savefig_kwargs={"bbox_inches": "tight"})
    close_plot(fig)

# -------------------------------------------------------------------------
# Decoupler net plots (msigdb / progeny / dorothea)
# -------------------------------------------------------------------------
def _decoupler_figdir(base: Path | None, net_name: str) -> Path:
    """
    Put decoupler plots under:
      cluster_and_annotate/decoupler/<net_name>/
    while staying compatible with record_plot_artifact() routing.
    """
    base = Path("cluster_and_annotate") if base is None else Path(base)
    return base / "decoupler" / str(net_name).lower().strip()


def _clean_feature_label(label: str, net_name: str) -> str:
    """Removes prefixes and replaces underscores for cleaner display."""
    label = str(label)
    net_name_lower = net_name.lower()

    # Remove MSigDB prefixes (e.g., HALLMARK_, REACTOME_)
    if "_" in label:
        parts = label.split("_", 1)
        # Check if the first part is a known MSigDB prefix
        if parts[0] in ["HALLMARK", "REACTOME", "KEGG", "BIOCARTA", "GOBP", "GOCC", "GOMF"]:
            label = parts[1]

    return label.replace("_", " ")


def _zscore_cols(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Z-score each column across rows (clusters)."""
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, np.nan)
    z = (df - mu) / (sd + eps)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _top_features_global(
    activity: pd.DataFrame,
    k: int,
    *,
    mode: str = "var",  # "var" or "mean_abs"
    signed: bool = True,
) -> list[str]:
    """
    Pick top-k features globally. activity is clusters x features.
    """
    if activity is None or activity.empty:
        return []
    A = activity.copy()
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if not signed:
        A = A.abs()

    if mode == "mean_abs":
        score = A.abs().mean(axis=0) if signed else A.mean(axis=0)
    else:
        score = A.var(axis=0)

    score = score.sort_values(ascending=False)
    return score.head(int(k)).index.astype(str).tolist()


def _wrap_labels(labels: Sequence[str], wrap_at: int = 38) -> list[str]:

    out: list[str] = []
    for s in labels:
        s = str(s)
        out.append(
            "\n".join(
                textwrap.wrap(
                    s,
                    width=int(wrap_at),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return out


def _msigdb_prefix(term: str) -> str:
    """
    Determine "GMT family" prefix for MSigDB-like pathway names.
    Examples:
      - "HALLMARK_TNFA_SIGNALING_VIA_NFKB" -> "HALLMARK"
      - "REACTOME_SOMETHING"              -> "REACTOME"
      - "PREFIX::TERM"                    -> "PREFIX"
    """
    term = str(term)
    if "::" in term:
        return term.split("::", 1)[0].strip() or "UNKNOWN"
    # Default MSigDB style: PREFIX_REST
    return (term.split("_", 1)[0].strip() or "UNKNOWN")


def _split_activity_for_msigdb(
    activity: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split MSigDB activity (clusters x pathways) into {prefix -> activity_sub}.
    For non-MSigDB nets, caller should not use this.
    """
    if activity is None or activity.empty:
        return {}
    cols = activity.columns.astype(str)
    prefixes = pd.Series(cols, index=cols).map(_msigdb_prefix)
    out: dict[str, pd.DataFrame] = {}
    for pfx, cols_idx in prefixes.groupby(prefixes).groups.items():
        cols_list = list(cols_idx)
        sub = activity.loc[:, cols_list].copy()
        out[str(pfx)] = sub
    return out


def _dynamic_left_margin_from_labels(labels: Sequence[str], *, base: float = 0.22) -> float:
    """
    Compute a left margin fraction [0,1] based on the longest label length.
    Tuned for horizontal barplots / dotplots with long y labels.
    """
    if labels is None:
        return float(np.clip(base, 0.28, 0.72))
    try:
        max_len = int(max(len(str(x)) for x in labels)) if len(labels) else 0
    except Exception:
        max_len = 0
    # ~120 chars -> ~0.70, ~20 chars -> ~0.34
    left = base + 0.0040 * max_len
    return float(np.clip(left, 0.28, 0.72))


def _dynamic_fig_width_for_barplot(labels: Sequence[str], *, min_w: float = 12.0, max_w: float = 26.0) -> float:
    """
    Compute figure width in inches to accommodate long pathway names without squishing.
    """
    try:
        max_len = int(max(len(str(x)) for x in labels)) if len(labels) else 0
    except Exception:
        max_len = 0
    # Increase width with label length; MSigDB often needs a lot.
    # 40 chars -> +2.4, 120 chars -> +9.6
    w = float(min_w + 0.08 * max_len)
    return float(np.clip(w, min_w, max_w))


@collect_plot_artifacts
def plot_decoupler_cluster_topn_barplots(
        activity: pd.DataFrame,
        *,
        net_name: str,
        figdir: Path | None,
        n: int = 10,
        use_abs: bool = False,
        split_signed: bool = False,
        n_pos: int | None = None,
        n_neg: int | None = None,
        stem_prefix: str = "cluster",
        title_prefix: Optional[str] = None,
) -> None:
    if activity is None or activity.empty:
        return

    outdir = _decoupler_figdir(figdir, net_name)
    A = activity.copy()
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for cl in A.index.astype(str):
        s = A.loc[cl].copy()

        if split_signed and not use_abs:
            n_pos_eff = int(n_pos) if n_pos is not None else int(n)
            n_neg_eff = int(n_neg) if n_neg is not None else int(n)
            pos = s[s > 0].sort_values(ascending=False).head(n_pos_eff)
            neg = s[s < 0].sort_values(ascending=True).head(n_neg_eff)
            if pos.empty and neg.empty:
                continue

            vals = pd.concat([pos, neg], axis=0)
            labels = [_clean_feature_label(l, net_name) for l in vals.index]
            colors = ["#3a7f3b"] * int(pos.shape[0]) + ["#b04a4a"] * int(neg.shape[0])
            vals_plot = vals

            left = _dynamic_left_margin_from_labels(labels, base=0.25)
            fig_w = _dynamic_fig_width_for_barplot(labels, min_w=12.0, max_w=28.0)
            fig_h = max(4.0, 0.45 * len(vals_plot) + 1.5)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            top_margin = 1.0 - (0.8 / fig_h)
            fig.subplots_adjust(left=left, right=0.96, top=top_margin, bottom=0.15)

            y = np.arange(len(vals_plot))
            ax.barh(y=y, width=vals_plot.values, color=colors, edgecolor="#1f2d3a", linewidth=0.8, zorder=3)
            ax.invert_yaxis()

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=12)
            ax.set_xlabel("Activity Score", fontsize=12, labelpad=10)

            max_abs = float(np.max(np.abs(vals_plot.values))) if len(vals_plot) else 0.0
            if max_abs > 0:
                ax.set_xlim(-1.05 * max_abs, 1.05 * max_abs)
            ax.axvline(0.0, color="#222222", linewidth=1.0)

            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            main_title = f"{title_prefix} • {cl}" if title_prefix else cl
            title_fontsize = 22
            if len(main_title) > 70:
                import textwrap
                main_title = textwrap.fill(main_title, width=70)
                title_fontsize = 18
            elif len(main_title) > 50:
                title_fontsize = 20
            ax.text(0.5, 1.08, main_title, transform=ax.transAxes, ha="center", weight="bold", fontsize=title_fontsize)
            ax.text(0.5, 1.02, str(net_name).upper(), transform=ax.transAxes, ha="center", color="#666666", fontsize=14)

            stem = f"{stem_prefix}_{cl}__top{int(n_pos_eff)}_up_down_bar"
            record_plot_artifact(stem, outdir, fig)
            close_plot(fig)
            continue

        s_rank = s.abs() if use_abs else s
        top = s_rank.sort_values(ascending=False).head(int(n))

        if top.empty:
            continue

        vals = s.loc[top.index] if use_abs else top
        vals_plot = vals.sort_values(ascending=True)
        clean_labels = [_clean_feature_label(l, net_name) for l in vals_plot.index]

        left = _dynamic_left_margin_from_labels(clean_labels, base=0.25)
        fig_w = _dynamic_fig_width_for_barplot(clean_labels, min_w=12.0, max_w=28.0)
        fig_h = max(4.0, 0.45 * len(vals_plot) + 1.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        top_margin = 1.0 - (0.8 / fig_h)
        fig.subplots_adjust(left=left, right=0.96, top=top_margin, bottom=0.15)

        if net_name.lower() == "progeny":
            bar_colors = ["#b04a4a" if v < 0 else "#3a7f3b" for v in vals_plot.values]
        else:
            bar_colors = "#3b84a8"

        y = np.arange(len(vals_plot))
        ax.barh(y=y, width=vals_plot.values, color=bar_colors, edgecolor="#1f2d3a", linewidth=0.8, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(clean_labels, fontsize=12)
        ax.set_xlabel("Activity Score", fontsize=12, labelpad=10)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        main_title = f"{title_prefix} • {cl}" if title_prefix else cl
        title_fontsize = 22
        if len(main_title) > 70:
            import textwrap
            main_title = textwrap.fill(main_title, width=70)
            title_fontsize = 18
        elif len(main_title) > 50:
            title_fontsize = 20
        ax.text(0.5, 1.08, main_title, transform=ax.transAxes, ha="center", weight="bold", fontsize=title_fontsize)
        ax.text(0.5, 1.02, str(net_name).upper(), transform=ax.transAxes, ha="center", color="#666666", fontsize=14)

        stem = f"{stem_prefix}_{cl}__top{int(n)}_bar"
        record_plot_artifact(stem, outdir, fig)
        close_plot(fig)

@collect_plot_artifacts
def plot_decoupler_activity_heatmap(
        activity: pd.DataFrame,
        *,
        net_name: str,
        figdir: Path | None,
        top_k: int = 30,
        rank_mode: str = "var",
        use_zscore: bool = True,
        wrap_labels: bool = True,   # kept for API compat; ignored (no wrapping)
        cmap: str = "viridis",
        stem: str = "heatmap_top",
        title_prefix: Optional[str] = None,
) -> None:
    if activity is None or activity.empty:
        return

    outdir = _decoupler_figdir(figdir, net_name)
    sub_raw, sub_z = _decoupler_balanced_clustered(activity, top_k=top_k, use_zscore=use_zscore)
    if sub_raw is None or sub_z is None:
        return
    sub = sub_z if use_zscore else sub_raw

    # --- NO WRAP: keep single-line labels ---
    sub.columns = [_clean_feature_label(c, net_name) for c in sub.columns]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_w = max(14.0, 6.0 + 0.45 * sub.shape[1])
    fig_h = max(10.0, 4.0 + 0.40 * sub.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(sub, ax=ax, cmap=cmap, cbar_kws={"shrink": 0.6})

    # Ticks: single-line labels, angled. No multiline alignment needed.
    ax.tick_params(axis="x", rotation=45, pad=8, labelsize=9)
    for t in ax.get_xticklabels():
        t.set_ha("right")
        t.set_va("top")

    ax.grid(False)

    # --- Dynamic margins based on label lengths (bottom for x labels) ---
    # Crude-but-effective heuristic: longer strings -> bigger bottom margin.
    max_x = max((len(str(x)) for x in sub.columns), default=10)
    # scale bottom within [0.22, 0.45]
    bottom = min(0.45, max(0.22, 0.22 + 0.0045 * max(0, max_x - 18)))
    # left margin for cluster names
    max_y = max((len(str(y)) for y in sub.index), default=10)
    left = min(0.35, max(0.18, 0.18 + 0.0030 * max(0, max_y - 18)))

    fig.subplots_adjust(bottom=bottom, left=left, right=0.92, top=0.92)

    record_plot_artifact(f"{stem}{int(top_k)}", outdir, fig)
    close_plot(fig)


def _decoupler_balanced_clustered(
    activity: pd.DataFrame,
    *,
    top_k: int,
    use_zscore: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if activity is None or activity.empty:
        return None, None

    sub_raw = activity.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scores = sub_raw.mean(axis=0)
    n_pos = int(max(1, int(top_k) // 2))
    n_neg = int(max(1, int(top_k) - n_pos))
    pos = scores[scores > 0].sort_values(ascending=False).head(n_pos)
    neg = scores[scores < 0].sort_values(ascending=True).head(n_neg)
    feats = list(pos.index) + list(neg.index)
    if not feats:
        return None, None

    sub_raw = sub_raw.loc[:, feats].copy()
    sub_z = _zscore_cols(sub_raw) if use_zscore else sub_raw.copy()
    sub_raw = sub_raw.copy()
    sub_z = sub_z.copy()
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        row_link = linkage(pdist(sub_z.values), method="average") if sub_z.shape[0] > 1 else None
        col_link = linkage(pdist(sub_z.values.T), method="average") if sub_z.shape[1] > 1 else None
        if row_link is not None:
            row_order = leaves_list(row_link)
            sub_raw = sub_raw.iloc[row_order, :]
            sub_z = sub_z.iloc[row_order, :]
        if col_link is not None:
            col_order = leaves_list(col_link)
            sub_raw = sub_raw.iloc[:, col_order]
            sub_z = sub_z.iloc[:, col_order]
    except Exception:
        pass

    return sub_raw, sub_z


@collect_plot_artifacts
def plot_decoupler_dotplot(
    activity: pd.DataFrame,
    *,
    net_name: str,
    figdir: Path | None,
    top_k: int = 30,
    rank_mode: str = "var",
    color_by: str = "z",
    size_by: str = "abs_raw",
    wrap_labels: bool = True,
    wrap_at: int = 25,
    cmap: str = "viridis",
    stem: str = "dotplot_top",
    title_prefix: Optional[str] = None,
) -> None:
    if activity is None or activity.empty:
        return

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    outdir = _decoupler_figdir(figdir, net_name)

    sub_raw, sub_z = _decoupler_balanced_clustered(activity, top_k=top_k, use_zscore=True)
    if sub_raw is None or sub_z is None:
        return

    color_mat = sub_raw if color_by == "raw" else sub_z
    cbar_label = "activity" if color_by == "raw" else "z-score"
    size_mat = sub_z.abs() if size_by == "abs_z" else sub_raw.abs()
    size_label = "|raw|" if size_by == "abs_raw" else "|z|"

    clusters = sub_raw.index.astype(str).tolist()
    features_disp = [_clean_feature_label(f, net_name) for f in sub_raw.columns]

    # --------------------------------------------------
    # FIGURE SETUP
    # --------------------------------------------------
    fig_w = max(22.0, 12.0 + 0.6 * len(clusters))
    fig_h = max(12.0, 5.0 + 0.5 * len(features_disp))
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    gs = GridSpec(1, 2, width_ratios=[1.0, 0.18], wspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    gs_right = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[0, 1],
        height_ratios=[0.65, 0.30, 0.05],
        hspace=0.15
    )
    ax_leg = fig.add_subplot(gs_right[0, 0])
    ax_cbar = fig.add_subplot(gs_right[1, 0])
    ax_sp = fig.add_subplot(gs_right[2, 0])
    ax_leg.axis("off")
    ax_sp.axis("off")

    # --------------------------------------------------
    # SCATTER
    # --------------------------------------------------
    rows = []
    for j, feat in enumerate(sub_raw.columns):
        for cl in clusters:
            rows.append({
                "cluster": cl,
                "feature": features_disp[j],
                "color": float(color_mat.loc[cl, feat]),
                "size": float(size_mat.loc[cl, feat]),
            })
    df = pd.DataFrame(rows)

    svals = df["size"].to_numpy()
    s_min, s_max = float(np.nanmin(svals)), float(np.nanmax(svals))

    def size_scale(v: float) -> float:
        return 30.0 + (v - s_min) / (s_max - s_min) * 250.0 if s_max > s_min else 80.0

    sca = ax.scatter(
        x=df["cluster"],
        y=df["feature"],
        s=[size_scale(v) for v in df["size"]],
        c=df["color"],
        cmap=cmap,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
    )

    ax.tick_params(axis="x", rotation=45, pad=8)
    ax.set_title(
        f"{(title_prefix + ' ') if title_prefix else ''}{net_name}",
        pad=18, size=20, weight="bold",
    )

    # Optional label wrapping
    if wrap_labels and wrap_at and wrap_at > 5:
        import textwrap
        from matplotlib.ticker import FixedLocator
        yticks = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.set_yticklabels(
            [textwrap.fill(str(t.get_text()), width=wrap_at) for t in ax.get_yticklabels()]
        )

    # --------------------------------------------------
    # SIZE LEGEND
    # --------------------------------------------------
    finite = svals[np.isfinite(svals)]
    if finite.size > 0:
        q = np.quantile(finite, [0.25, 0.50, 0.75])
        refs = np.unique(np.round(q, 2))
        refs = refs[refs > 0]

        handles = [
            Line2D(
                [0], [0],
                marker="o", linestyle="",
                markerfacecolor="gray",
                markeredgecolor="black",
                color="w",
                markersize=float(np.sqrt(size_scale(v))),
                alpha=0.7,
                label=f"{v:g}",
            )
            for v in refs
        ]
        ax_leg.legend(
            handles=handles,
            title=size_label,
            loc="upper left",
            frameon=False,
            labelspacing=1.2,
            title_fontsize=14,
            fontsize=12,
        )

    # --------------------------------------------------
    # COLORBAR
    # --------------------------------------------------
    cbar = fig.colorbar(sca, cax=ax_cbar)
    cbar.set_label(cbar_label, weight="bold", size=14)

    record_plot_artifact(f"{stem}{int(top_k)}_{cbar_label}_{size_by}", outdir, fig)
    close_plot(fig)


@collect_plot_artifacts
def plot_decoupler_all_styles(
    adata,
    *,
    net_key: str,
    net_name: Optional[str] = None,
    figdir: Path | None = None,
    heatmap_top_k: int = 30,
    bar_top_n: int = 10,
    bar_top_n_up: int | None = None,
    bar_top_n_down: int | None = None,
    bar_split_signed: bool = True,
    dotplot_top_k: int = 30,
) -> None:
    """
    Convenience wrapper:
    Reads adata.uns[net_key]["activity"] and makes:
      1) heatmap
      2) per-cluster topN barplots
      3) dotplot

    Behavior:
      - ALWAYS prefers pretty/display labels for plotting if available.
        (display mapping is recorded by run_decoupler_for_round)
      - MSigDB: prefers adata.uns["msigdb"]["activity_by_gmt"] if present,
        otherwise falls back to prefix splitting of the activity columns.
    """
    net_name = net_name or net_key
    block = adata.uns.get(net_key, {})
    activity = block.get("activity", None)

    try:
        LOGGER.info(
            "plot_decoupler_all_styles: net=%s activity_type=%s shape=%s",
            str(net_key),
            type(activity).__name__,
            getattr(activity, "shape", None),
        )
    except Exception:
        pass

    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        return

    # ------------------------------------------------------------------
    # Display labels (pretty labels) for plotting
    # ------------------------------------------------------------------
    def _get_display_map() -> dict[str, str]:
        # 1) Best: published into top-level resource payload by run_decoupler_for_round()
        try:
            cfg = block.get("config", {})
            if isinstance(cfg, dict):
                dm = cfg.get("cluster_display_map", None)
                if isinstance(dm, dict) and dm:
                    return {str(k): str(v) for k, v in dm.items()}
        except Exception:
            pass

        # 2) Next: round-owned decoupler display map (active round)
        try:
            rid = adata.uns.get("active_cluster_round", None)
            rid = str(rid) if rid is not None else None
            rounds = adata.uns.get("cluster_rounds", {})
            if rid and isinstance(rounds, dict) and rid in rounds:
                dec = rounds[rid].get("decoupler", {})
                if isinstance(dec, dict):
                    dm = dec.get("cluster_display_map", None)
                    if isinstance(dm, dict) and dm:
                        return {str(k): str(v) for k, v in dm.items()}
        except Exception:
            pass

        # 3) Fallback: if a pretty label column exists for the active round, derive mapping
        try:
            rid = adata.uns.get("active_cluster_round", None)
            rid = str(rid) if rid is not None else None
            if rid:
                labels_obs_key = None
                rounds = adata.uns.get("cluster_rounds", {})
                if isinstance(rounds, dict) and rid in rounds:
                    labels_obs_key = rounds[rid].get("labels_obs_key", None)
                pretty_key = f"{CLUSTER_LABEL_KEY}__{rid}"
                if labels_obs_key in adata.obs and pretty_key in adata.obs:
                    tmp = pd.DataFrame(
                        {
                            "cluster": adata.obs[str(labels_obs_key)].astype(str).values,
                            "pretty": adata.obs[str(pretty_key)].astype(str).values,
                        }
                    )
                    dm = (
                        tmp.groupby("cluster", observed=True)["pretty"]
                        .agg(lambda x: str(x.iloc[0]) if len(x) else "")
                        .to_dict()
                    )
                    dm = {str(k): str(v) for k, v in dm.items() if str(v)}
                    if dm:
                        return dm
        except Exception:
            pass

        return {}

    def _get_display_order() -> list[str]:
        """
        Returns the *display* row order (pretty labels), aligned to pseudobulk cluster order.
        Stored by run_decoupler_for_round() as:
          - block["config"]["cluster_display_labels"] (top-level publish)
          - rounds[rid]["decoupler"]["cluster_display_labels"] (round-owned)
        """
        # 1) Best: published into top-level resource payload by run_decoupler_for_round()
        try:
            cfg = block.get("config", {})
            if isinstance(cfg, dict):
                labels = cfg.get("cluster_display_labels", None)
                if isinstance(labels, (list, tuple)) and labels:
                    return [str(x) for x in labels]
        except Exception:
            pass

        # 2) Next: round-owned ordering (active round)
        try:
            rid = adata.uns.get("active_cluster_round", None)
            rid = str(rid) if rid is not None else None
            rounds = adata.uns.get("cluster_rounds", {})
            if rid and isinstance(rounds, dict) and rid in rounds:
                dec = rounds[rid].get("decoupler", {})
                if isinstance(dec, dict):
                    labels = dec.get("cluster_display_labels", None)
                    if isinstance(labels, (list, tuple)) and labels:
                        return [str(x) for x in labels]
        except Exception:
            pass

        return []

    display_map = _get_display_map()
    display_order = _get_display_order()

    def _apply_display_index(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if display_map:
            old = out.index.astype(str)
            new = old.map(lambda x: display_map.get(str(x), str(x)))

            # If mapping creates duplicates, make them unique but readable
            if pd.Index(new).has_duplicates:
                new = [f"{lbl} [{cid}]" for lbl, cid in zip(new, old)]

            out.index = pd.Index(new, name=out.index.name)

        # Enforce intended row order if we have it (order is in display-space)
        if display_order:
            want = [x for x in display_order if x in out.index]
            if want:
                out = out.reindex(want)

        return out

    # ------------------------------------------------------------------
    # Round ID for filenames (short) + titles (full)
    # ------------------------------------------------------------------
    rid_full = adata.uns.get("active_cluster_round", None)
    rid_full = str(rid_full) if rid_full else None
    rid_short = rid_full.split("_", 1)[0] if rid_full else None
    stem_prefix = f"{rid_short}_" if rid_short else ""

    # ------------------------------------------------------------------
    # MSigDB: split per GMT family (prefer round-precomputed activity_by_gmt)
    # ------------------------------------------------------------------
    if str(net_key).lower().strip() == "msigdb":
        splits = block.get("activity_by_gmt", None)
        if isinstance(splits, dict) and splits:
            splits = {str(k): v for k, v in splits.items() if isinstance(v, pd.DataFrame) and not v.empty}
        else:
            splits = _split_activity_for_msigdb(activity)

        if not splits:
            return

        # Stable-ish ordering (HALLMARK first if present, then alphabetical)
        ordered = sorted(
            splits.keys(),
            key=lambda x: (0 if str(x).upper() == "HALLMARK" else 1, str(x).upper()),
        )

        for pfx in ordered:
            sub = splits[pfx]
            if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
                continue

            sub_plot = _apply_display_index(sub)

            title_prefix = (
                f"{str(pfx).upper()} [{rid_full}]" if rid_full else str(pfx).upper()
            )

            plot_decoupler_activity_heatmap(
                sub_plot,
                net_name=net_name,
                figdir=figdir,
                top_k=heatmap_top_k,
                rank_mode="var",
                use_zscore=True,
                wrap_labels=True,
                stem=f"{stem_prefix}heatmap_top_{str(pfx).lower()}_",
                title_prefix=title_prefix,
            )

            plot_decoupler_cluster_topn_barplots(
                sub_plot,
                net_name=net_name,
                figdir=figdir,
                n=bar_top_n,
                use_abs=False,
                split_signed=bool(bar_split_signed) and str(net_name).lower() in ("dorothea", "msigdb"),
                n_pos=int(bar_top_n_up) if bar_top_n_up is not None else int(bar_top_n),
                n_neg=int(bar_top_n_down) if bar_top_n_down is not None else int(bar_top_n),
                stem_prefix=f"{stem_prefix}cluster_{str(pfx).lower()}",
                title_prefix=title_prefix,
            )

            plot_decoupler_dotplot(
                sub_plot,
                net_name=net_name,
                figdir=figdir,
                top_k=dotplot_top_k,
                rank_mode="var",
                color_by="z",
                size_by="abs_raw",
                wrap_labels=True,
                stem=f"{stem_prefix}dotplot_top_{str(pfx).lower()}_",
                title_prefix=title_prefix,
            )

        return

    # ------------------------------------------------------------------
    # Non-MSigDB nets: single set of plots
    # ------------------------------------------------------------------
    activity_plot = _apply_display_index(activity)

    title_prefix = f"{str(net_name)} [{rid_full}]" if rid_full else str(net_name)

    plot_decoupler_activity_heatmap(
        activity_plot,
        net_name=net_name,
        figdir=figdir,
        top_k=heatmap_top_k,
        rank_mode="var",
        use_zscore=True,
        wrap_labels=True,
        stem=f"{stem_prefix}heatmap_top_",
        title_prefix=title_prefix,
    )

    plot_decoupler_cluster_topn_barplots(
        activity_plot,
        net_name=net_name,
        figdir=figdir,
        n=bar_top_n,
        use_abs=False,
        split_signed=bool(bar_split_signed) and str(net_name).lower() in ("dorothea", "msigdb"),
        n_pos=int(bar_top_n_up) if bar_top_n_up is not None else int(bar_top_n),
        n_neg=int(bar_top_n_down) if bar_top_n_down is not None else int(bar_top_n),
        stem_prefix=f"{stem_prefix}cluster",
        title_prefix=title_prefix,
    )

    plot_decoupler_dotplot(
        activity_plot,
        net_name=net_name,
        figdir=figdir,
        top_k=dotplot_top_k,
        rank_mode="var",
        color_by="z",
        size_by="abs_raw",
        wrap_labels=True,
        stem=f"{stem_prefix}dotplot_top_",
        title_prefix=title_prefix,
    )


def _round_display_map(round_snapshot: dict) -> dict[str, str]:
    """
    Best-effort fetch of cluster pretty labels for a round.

    Expected primary location:
      round_snapshot["cluster_display_map"] : {raw_cluster_id -> "Cxx: Label"}

    Also checks a couple common nested locations used elsewhere.
    """
    if not isinstance(round_snapshot, dict):
        return {}

    # 1) canonical
    dm = round_snapshot.get("cluster_display_map", None)
    if isinstance(dm, dict) and dm:
        return {str(k): str(v) for k, v in dm.items()}

    # 2) sometimes stored under annotation payload
    ann = round_snapshot.get("annotation", None)
    if isinstance(ann, dict):
        dm = ann.get("cluster_display_map", None)
        if isinstance(dm, dict) and dm:
            return {str(k): str(v) for k, v in dm.items()}

    # 3) sometimes stored under decoupler payload
    dec = round_snapshot.get("decoupler", None)
    if isinstance(dec, dict):
        dm = dec.get("cluster_display_map", None)
        if isinstance(dm, dict) and dm:
            return {str(k): str(v) for k, v in dm.items()}

    return {}


def _safe_cluster_color_index(cid: str) -> int:
    """
    Convert cluster id to a stable non-negative int for palette indexing.
    Works for '0', '12', 'C03', 'UNKNOWN', etc.
    """
    s = str(cid)
    m = re.search(r"\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # stable-ish fallback
    return abs(hash(s)) % 10_000_000


def _make_unique_labels(labels: dict[str, str]) -> dict[str, str]:
    """
    If multiple raw ids map to the same display label, disambiguate by appending [raw].
    """
    # invert
    rev: dict[str, list[str]] = {}
    for raw, disp in labels.items():
        rev.setdefault(str(disp), []).append(str(raw))

    out = dict(labels)
    for disp, raws in rev.items():
        if len(raws) <= 1:
            continue
        for raw in raws:
            out[raw] = f"{disp} [{raw}]"
    return out
