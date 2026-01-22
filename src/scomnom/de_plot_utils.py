# src/scomnom/de_plot_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scanpy as sc


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _as_list(x: Optional[Iterable[str]]) -> list[str]:
    if x is None:
        return []
    return [str(v) for v in x]


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Required column {col!r} not found. Available: {list(df.columns)}")
    return df[col]


def _clip_padj(p: np.ndarray) -> np.ndarray:
    # Avoid inf/-inf in -log10; keep NaN as NaN
    out = p.astype(float, copy=True)
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], 1e-300, 1.0)
    return out


def _get_fig_from_scanpy_return(ret) -> Figure:
    """
    scanpy plotting can return:
      - Axes
      - list[Axes]
      - dict
      - None (when show=False, it may still create the current fig)
    We always return a Figure.
    """
    if isinstance(ret, Figure):
        return ret
    if isinstance(ret, Axes):
        return ret.figure
    if isinstance(ret, (list, tuple)) and ret:
        # list of axes
        if isinstance(ret[0], Axes):
            return ret[0].figure
    # fall back to current figure
    fig = plt.gcf()
    if not isinstance(fig, Figure):
        raise RuntimeError("Could not resolve a matplotlib Figure from scanpy return value.")
    return fig


def _select_top_genes(
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    padj_thresh: float = 0.05,
    top_n: int = 10,
    require_sig: bool = True,
) -> list[str]:
    """
    Rank genes by:
      1) padj ascending
      2) |logFC| descending
    """
    if df is None or df.empty:
        return []

    g = _safe_series(df, gene_col).astype(str)
    padj = pd.to_numeric(_safe_series(df, padj_col), errors="coerce")
    lfc = pd.to_numeric(_safe_series(df, lfc_col), errors="coerce")

    tmp = pd.DataFrame(
        {gene_col: g, padj_col: padj, lfc_col: lfc, "__abs_lfc": np.abs(lfc.to_numpy())}
    ).dropna(subset=[gene_col, padj_col, lfc_col])

    if require_sig:
        tmp = tmp[tmp[padj_col] < float(padj_thresh)]

    if tmp.empty:
        return []

    tmp = tmp.sort_values([padj_col, "__abs_lfc"], ascending=[True, False])
    genes = tmp[gene_col].head(int(top_n)).astype(str).tolist()

    # preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for x in genes:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -----------------------------------------------------------------------------
# Volcano plot
# -----------------------------------------------------------------------------
def volcano(
    df_de: pd.DataFrame,
    *,
    gene_col: str = "gene",
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (7.5, 6.0),
    alpha: float = 0.65,
    s: float = 10.0,
    show: bool = False,
) -> Figure:
    """
    Volcano plot from a DE table (pseudobulk or otherwise).

    Expects at least:
      - gene_col (default 'gene')
      - padj_col (default 'padj')
      - lfc_col (default 'log2FoldChange')

    Labels: chosen among significant genes by (padj asc, |lfc| desc), up to top_label_n.
    """
    if df_de is None or df_de.empty:
        # Still return an empty figure (helps orchestrator save placeholder)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No DE results", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    g = _safe_series(df_de, gene_col).astype(str)
    padj = pd.to_numeric(_safe_series(df_de, padj_col), errors="coerce")
    lfc = pd.to_numeric(_safe_series(df_de, lfc_col), errors="coerce")

    tmp = pd.DataFrame({gene_col: g, padj_col: padj, lfc_col: lfc}).dropna(
        subset=[gene_col, padj_col, lfc_col]
    )
    if tmp.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid rows (NaNs after parsing)", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    padj_np = _clip_padj(tmp[padj_col].to_numpy(dtype=float))
    y = -np.log10(padj_np)
    x = tmp[lfc_col].to_numpy(dtype=float)

    sig = (tmp[padj_col].to_numpy(dtype=float) < float(padj_thresh)) & (
        np.abs(x) > float(lfc_thresh)
    )

    fig, ax = plt.subplots(figsize=figsize)

    # plot non-sig then sig on top
    ax.scatter(
        x[~sig],
        y[~sig],
        c="#9aa0a6",  # soft gray
        s=s,
        alpha=alpha,
        linewidths=0.0,
        rasterized=True,
    )
    ax.scatter(
        x[sig],
        y[sig],
        c="#d93025",  # red
        s=s,
        alpha=min(1.0, alpha + 0.1),
        linewidths=0.0,
        rasterized=True,
    )

    # threshold lines
    ax.axhline(-np.log10(max(float(padj_thresh), 1e-300)), color="black", linestyle="--", lw=1)
    ax.axvline(float(lfc_thresh), color="black", linestyle="--", lw=1)
    ax.axvline(-float(lfc_thresh), color="black", linestyle="--", lw=1)

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 adjusted p-value")

    if title:
        ax.set_title(str(title))

    # labels
    if int(top_label_n) > 0:
        label_genes = _select_top_genes(
            tmp,
            gene_col=gene_col,
            padj_col=padj_col,
            lfc_col=lfc_col,
            padj_thresh=padj_thresh,
            top_n=int(top_label_n),
            require_sig=True,
        )
        if label_genes:
            # annotate those rows (pick first occurrence per gene)
            tmp_idx = tmp.reset_index(drop=True)
            for gg in label_genes:
                hits = np.where(tmp_idx[gene_col].to_numpy(dtype=str) == gg)[0]
                if hits.size == 0:
                    continue
                i = int(hits[0])
                ax.text(
                    float(tmp_idx.loc[i, lfc_col]),
                    float(-np.log10(max(float(tmp_idx.loc[i, padj_col]), 1e-300))),
                    gg,
                    fontsize=8,
                    ha="left" if float(tmp_idx.loc[i, lfc_col]) >= 0 else "right",
                    va="bottom",
                )

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Scanpy-based expression visualizations (return Figure; never save)
# -----------------------------------------------------------------------------
def dotplot_top_genes(
    adata,
    *,
    genes: Sequence[str],
    groupby: str,
    use_raw: bool = False,
    layer: Optional[str] = None,
    standard_scale: str = "var",
    dendrogram: bool = True,
    color_map: str = "viridis",
    figsize: Optional[tuple[float, float]] = None,
    show: bool = False,
) -> Figure:
    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    if not genes:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.text(0.5, 0.5, "No genes to plot", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    ret = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=groupby,
        use_raw=use_raw,
        layer=layer,
        standard_scale=standard_scale,
        dendrogram=dendrogram,
        color_map=color_map,
        figsize=figsize,
        show=False,  # always false; caller controls display
        return_fig=True,
    )

    # return_fig=True yields a DotPlot object with .make_figure()
    # but scanpy often returns it directly.
    try:
        fig = ret.figure if hasattr(ret, "figure") else None
        if isinstance(fig, Figure):
            if show:
                plt.show()
            return fig
    except Exception:
        pass

    # Fallback: attempt to make a figure, then use gcf
    try:
        if hasattr(ret, "make_figure"):
            ret.make_figure()
    except Exception:
        pass

    fig = _get_fig_from_scanpy_return(ret)
    if show:
        plt.show()
    return fig


def heatmap_top_genes(
    adata,
    *,
    genes: Sequence[str],
    groupby: str,
    use_raw: bool = False,
    layer: Optional[str] = None,
    standard_scale: str = "var",
    dendrogram: bool = True,
    cmap: str = "bwr",
    swap_axes: bool = True,
    show_gene_labels: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = False,
) -> Figure:
    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    if not genes:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.text(0.5, 0.5, "No genes to plot", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    ret = sc.pl.heatmap(
        adata,
        var_names=genes,
        groupby=groupby,
        use_raw=use_raw,
        layer=layer,
        swap_axes=swap_axes,
        standard_scale=standard_scale,
        cmap=cmap,
        dendrogram=dendrogram,
        show_gene_labels=show_gene_labels,
        figsize=figsize,
        show=False,
    )
    fig = _get_fig_from_scanpy_return(ret)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def violin_genes(
    adata,
    *,
    genes: Sequence[str],
    groupby: str,
    use_raw: bool = False,
    layer: Optional[str] = None,
    log: bool = True,
    multi_panel: bool = True,
    stripplot: bool = False,
    jitter: bool = False,
    rotation: int = 90,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = False,
) -> Figure:
    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    if not genes:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.text(0.5, 0.5, "No genes to plot", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    ret = sc.pl.violin(
        adata,
        keys=genes,
        groupby=groupby,
        use_raw=use_raw,
        layer=layer,
        log=log,
        multi_panel=multi_panel,
        stripplot=stripplot,
        jitter=jitter,
        rotation=rotation,
        figsize=figsize,
        show=False,
    )
    fig = _get_fig_from_scanpy_return(ret)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def umap_features_grid(
    adata,
    *,
    genes: Sequence[str],
    use_raw: bool = False,
    layer: Optional[str] = None,
    ncols: int = 3,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: Optional[float] = None,
    show: bool = False,
) -> Figure:
    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    if not genes:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.text(0.5, 0.5, "No genes to plot", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    ret = sc.pl.umap(
        adata,
        color=genes,
        use_raw=use_raw,
        layer=layer,
        ncols=int(max(1, ncols)),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        size=size,
        show=False,
    )
    fig = _get_fig_from_scanpy_return(ret)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
