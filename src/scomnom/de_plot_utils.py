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
def _unique_keep_order(xs: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x)
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _select_top_signed(
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    padj_thresh: float = 0.05,
    top_n: int = 10,
    direction: str = "up",  # "up" (lfc>0) or "down" (lfc<0)
) -> list[str]:
    if df is None or df.empty:
        return []
    g = _safe_series(df, gene_col).astype(str)
    padj = pd.to_numeric(_safe_series(df, padj_col), errors="coerce")
    lfc = pd.to_numeric(_safe_series(df, lfc_col), errors="coerce")

    tmp = pd.DataFrame({gene_col: g, padj_col: padj, lfc_col: lfc}).dropna(
        subset=[gene_col, padj_col, lfc_col]
    )
    tmp = tmp[tmp[padj_col] < float(padj_thresh)]
    if tmp.empty:
        return []

    if direction == "down":
        tmp = tmp[tmp[lfc_col] < 0]
        tmp["__rank"] = tmp[lfc_col].abs()
    else:
        tmp = tmp[tmp[lfc_col] > 0]
        tmp["__rank"] = tmp[lfc_col].abs()

    if tmp.empty:
        return []

    tmp = tmp.sort_values([padj_col, "__rank"], ascending=[True, False])
    return _unique_keep_order(tmp[gene_col].head(int(top_n)).astype(str).tolist())


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
    standard_scale: Optional[str] = "var",
    dendrogram: bool = False,
    color_map: str = "viridis",
    figsize: Optional[tuple[float, float]] = None,
    show: bool = False,
) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import scanpy as sc

    _normalize_scanpy_groupby_colors(adata, str(groupby))

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
        show=False,
        return_fig=True,
    )

    # ---- robustly obtain the matplotlib Figure
    fig = None
    if hasattr(ret, "figure") and isinstance(ret.figure, Figure):
        fig = ret.figure
    else:
        try:
            if hasattr(ret, "make_figure"):
                ret.make_figure()
            fig = plt.gcf()
        except Exception:
            fig = plt.gcf()

    # ---- cosmetics -------------------------------------------------
    # 1) avoid cutoff of rotated gene labels
    fig.subplots_adjust(
        left=0.25,   # space for long cluster names
        bottom=0.25  # space for rotated gene labels
    )

    # 2) remove gridlines everywhere
    for ax in fig.axes:
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        for gl in ax.get_ygridlines():
            gl.set_visible(False)

    # 3) cleaner look (optional but recommended)
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0.25, 0.25, 1, 1))

    if show:
        plt.show()
    return fig


def _normalize_scanpy_groupby_colors(adata, groupby: str) -> None:
    k = f"{groupby}_colors"
    if k not in adata.uns:
        return

    colors = adata.uns[k]

    # unwrap common serialization wrappers
    if isinstance(colors, dict):
        if "data" in colors:
            colors = colors["data"]
        elif "payload" in colors and isinstance(colors["payload"], dict) and "data" in colors["payload"]:
            colors = colors["payload"]["data"]

    try:
        colors = list(np.asarray(colors).astype(str))
    except Exception:
        del adata.uns[k]
        return

    adata.uns[k] = colors



from typing import Mapping, Any, Dict  # add at top

from typing import Mapping, Sequence, Optional
import numpy as np
import pandas as pd


def heatmap_top_genes(
    adata,
    *,
    genes: Sequence[str] | None = None,
    genes_by_cluster: Mapping[str, Sequence[str]] | None = None,
    groupby: str,
    use_raw: bool = False,
    layer: Optional[str] = None,
    cmap: str | None = None,
    show_cluster_colorbar: bool = True,
    scale_columns_by_size: bool = True,
    min_col_width: float = 0.5,
    max_col_width: float = 5.0,
    figsize: Optional[tuple[float, float]] = None,
    show_gene_labels: bool = True,
    z_clip: float | None = 2.5,
    show: bool = False,
):
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    if groupby not in adata.obs:
        raise KeyError(f"groupby={groupby!r} not found in adata.obs")

    # -----------------------------
    # 0) Resolve gene list
    # -----------------------------
    if genes_by_cluster is not None:
        vc0 = adata.obs[groupby].astype(str).value_counts()
        cluster_order0 = vc0.index.astype(str).tolist()
        flat, seen = [], set()
        for cl in cluster_order0:
            for g in (genes_by_cluster.get(str(cl), []) or []):
                g = str(g) if g is not None else ""
                if g and g not in seen:
                    flat.append(g)
                    seen.add(g)
        genes = flat

    genes = [str(g) for g in (genes or []) if g is not None and str(g) != ""]
    if not genes:
        raise ValueError("No genes provided.")

    # -----------------------------
    # 1) Aggregate and Z-score
    # -----------------------------
    vc = adata.obs[groupby].astype(str).value_counts()
    groups = vc.index.astype(str).tolist()

    sub = adata[:, genes]
    X = sub.layers[layer] if layer else (sub.raw.X if use_raw and sub.raw is not None else sub.X)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    df_x = pd.DataFrame(X, columns=genes)
    df_x[groupby] = adata.obs[groupby].astype(str).values
    mean = df_x.groupby(groupby, observed=True)[genes].mean().reindex(groups)

    M = mean.to_numpy(dtype=float)
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    if z_clip is not None:
        Z = np.clip(Z, -float(z_clip), float(z_clip))
    plot_mat = Z.T

    # -----------------------------
    # 2) Column widths (X-axis)
    # -----------------------------
    sizes = vc.reindex(groups).values.astype(float)
    if scale_columns_by_size:
        w = sizes / float(np.mean(sizes)) if np.mean(sizes) > 0 else np.ones_like(sizes)
        w = np.clip(w, float(min_col_width), float(max_col_width))
    else:
        w = np.ones(len(groups), dtype=float)

    x_edges = np.concatenate([[0.0], np.cumsum(w)])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_edges = np.arange(plot_mat.shape[0] + 1, dtype=float)

    # -----------------------------
    # 3) Cluster colors
    # -----------------------------
    colors = None
    if show_cluster_colorbar:
        try:
            _normalize_scanpy_groupby_colors(adata, str(groupby))
            colors = adata.uns.get(f"{groupby}_colors", None)
            if isinstance(colors, dict) and "data" in colors:
                colors = colors["data"]
            if colors is not None:
                colors = list(np.asarray(colors).astype(str))[: len(groups)]
        except Exception:
            colors = None

    # -----------------------------
    # 4) Colormap: Magenta-Black-Yellow
    # -----------------------------
    colors_list = ["#FF00FF", "#000000", "#FFFF00"]
    cmap_seurat = mcolors.LinearSegmentedColormap.from_list("seurat", colors_list, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-float(z_clip), vcenter=0.0, vmax=float(z_clip))

    # -----------------------------
    # 5) Figure & GridSpec
    # -----------------------------
    if figsize is None:
        W = max(8.0, 0.3 * float(x_edges[-1]) + 3.0)
        H = max(5.0, 0.2 * float(plot_mat.shape[0]) + 2.0)
        figsize = (W, H)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2 if colors is not None else 1,
        ncols=2,
        height_ratios=[0.02, 1.0] if colors is not None else [1.0],
        width_ratios=[1.0, 0.008],
        hspace=0.01, wspace=0.02
    )

    ax = fig.add_subplot(gs[-1, 0])
    cax = fig.add_subplot(gs[-1, 1])

    # -----------------------------
    # 6) Heatmap Draw (No Seams Fix)
    # -----------------------------
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        plot_mat,
        shading="flat",
        cmap=cmap_seurat,
        norm=norm,
        edgecolors="face",
        linewidth=0.1,
        antialiased=True,
        rasterized=True
    )
    ax.set_facecolor("#000000")

    # -----------------------------
    # 7) Colorbar
    # -----------------------------
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=8, length=0)

    # -----------------------------
    # 8) Cosmetics
    # -----------------------------
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_yticklabels(genes if show_gene_labels else [], fontsize=10)
    ax.set_xticks([])
    ax.tick_params(axis="both", which="both", length=0)
    sns.despine(ax=ax, left=True, bottom=True)

    # -----------------------------
    # 9) Top color bar + Angled Labels
    # -----------------------------
    if colors is not None:
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        for i in range(len(groups)):
            ax_top.add_patch(
                plt.Rectangle((float(x_edges[i]), 0.0), float(w[i]), 1.0, color=colors[i], lw=0)
            )
            # FIX: Schuine labels (45 graden) om overlap te voorkomen
            ax_top.text(
                x_centers[i], 1.2, groups[i],
                ha='left',           # 'left' i.c.m. rotatie zorgt voor mooie uitlijning
                va='bottom',
                rotation=45,         # Schuine hoek
                fontsize=11,
                fontweight='bold',
                transform=ax_top.get_xaxis_transform()
            )
        ax_top.set_xlim(0, float(x_edges[-1]))
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")

    if show:
        plt.show()

    return fig

    # -----------------------------
    # 7) Colorbar
    # -----------------------------
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=8, length=0)
    cax.set_ylabel("Z-score", fontsize=8, labelpad=2)

    # -----------------------------
    # 8) Axes cosmetics
    # -----------------------------
    ax.invert_yaxis()

    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_yticklabels(genes if show_gene_labels else [], fontsize=10)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(groups, rotation=90, fontsize=11)

    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()

    sns.despine(ax=ax, left=True, bottom=True)

    # -----------------------------
    # 9) Top cluster color bar (thin)
    # -----------------------------
    if colors is not None:
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        for i in range(len(groups)):
            ax_top.add_patch(
                plt.Rectangle(
                    (float(x_edges[i]), 0.0),
                    float(w[i]),
                    1.0,
                    color=colors[i],
                    lw=0,
                )
            )
        ax_top.set_xlim(0, float(x_edges[-1]))
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")
        ax_top.grid(False)

    if show:
        plt.show()

    return fig



def violin_grid_genes(
    adata,
    *,
    genes: Sequence[str],
    groupby: str,
    use_raw: bool = False,
    layer: str | None = None,
    ncols: int = 3,
    stripplot: bool = False,
    rotation: float = 90,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Figure:
    """
    Pretty multi-panel violin grid (API-callable).

    IMPORTANT:
      - Does NOT pass ncols into scanpy (scanpy.violin doesn't accept it reliably).
      - Instead, creates a matplotlib grid and plots one gene per axis via ax=...
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    _normalize_scanpy_groupby_colors(adata, str(groupby))

    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    genes = _unique_keep_order(genes)
    if not genes:
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.text(0.5, 0.5, "No genes to plot", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    ncols = int(max(1, ncols))
    n = len(genes)
    nrows = int(np.ceil(n / ncols))

    # heuristic sizing if not given
    if figsize is None:
        figsize = (max(8.0, 3.4 * min(ncols, n)), max(4.0, 2.6 * nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    # ---- plot one gene per axis
    for i, g in enumerate(genes):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        sc.pl.violin(
            adata,
            keys=[g],
            groupby=str(groupby),
            use_raw=bool(use_raw),
            layer=layer,
            show=False,
            stripplot=bool(stripplot),
            rotation=rotation,
            ax=ax,
        )

        # normalize titles
        try:
            ax.set_title(str(g))
        except Exception:
            pass

    # ---- turn off unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # ---- Hide cluster/category labels except on bottom row
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if not isinstance(ax, Axes) or not ax.get_visible():
                continue

            if r != nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelbottom=True)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(rotation)
                    lab.set_ha("right" if float(rotation) != 0 else "center")

    # ---- clean all axes (IMPORTANT: flatten the grid)
    for ax in np.ravel(axes):
        if not isinstance(ax, Axes) or not ax.get_visible():
            continue
        try:
            for gl in ax.get_ygridlines():
                gl.set_visible(False)
        except Exception:
            pass
        ax.grid(False)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig



def violin_genes(
    adata,
    *,
    genes,
    groupby: str,
    use_raw: bool = False,
    layer: str | None = None,
    show: bool = False,
    rotation: int | float | None = 90,
    figsize=None,
    **kwargs,
):
    """
    Violin plot wrapper that is robust to kwargs leakage (e.g. figsize=None).

    IMPORTANT:
      - Do NOT forward figsize=None into kwargs that may reach matplotlib artists.
      - If figsize is provided, pass it only as a real tuple (w, h).
    """
    import scanpy as sc
    import matplotlib.pyplot as plt

    _normalize_scanpy_groupby_colors(adata, str(groupby))

    # ---- normalize genes
    if genes is None:
        return plt.gcf()
    if isinstance(genes, (str, bytes)):
        genes = [str(genes)]
    genes = [str(g) for g in genes if g is not None]
    if len(genes) == 0:
        return plt.gcf()

    # ---- sanitize kwargs: never forward figsize unless it's a valid tuple
    # Users sometimes pass figsize=None via config; that must not reach artist.set(...)
    if "figsize" in kwargs and (kwargs["figsize"] is None):
        kwargs.pop("figsize", None)

    # Accept figsize either via explicit arg or kwargs (explicit wins)
    _figsize = figsize
    if _figsize is None and "figsize" in kwargs:
        _figsize = kwargs.pop("figsize")

    # Only keep figsize if it is a real (w,h) tuple/list of length 2
    if _figsize is not None:
        try:
            if not (isinstance(_figsize, (tuple, list)) and len(_figsize) == 2):
                _figsize = None
        except Exception:
            _figsize = None

    # ---- call scanpy
    # NOTE: sc.pl.violin accepts figsize, but it must not be forwarded into seaborn/mpl element kwargs.
    call_kwargs = dict(
        use_raw=bool(use_raw),
        layer=layer,
        show=bool(show),
    )

    if _figsize is not None:
        call_kwargs["figsize"] = _figsize

    # Any remaining kwargs are assumed to be valid scanpy.violin parameters
    call_kwargs.update(kwargs)

    ax = sc.pl.violin(
        adata,
        keys=genes,
        groupby=str(groupby),
        **call_kwargs,
    )

    # ---- rotate x tick labels (scanpy sometimes returns Axes or list of Axes)
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    if rotation is not None:
        for a in axes:
            if a is None:
                continue
            try:
                for lab in a.get_xticklabels():
                    lab.set_rotation(rotation)
                    lab.set_ha("right" if float(rotation) != 0 else "center")
            except Exception:
                pass

    for a in axes:
        if a is None:
            continue
        for gl in a.get_ygridlines():
            gl.set_visible(False)
        a.grid(False)
        a.yaxis.grid(False)
        a.xaxis.grid(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    return plt.gcf()



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


def plot_marker_genes_pseudobulk(
    adata,
    *,
    groupby: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    top_n_genes: int = 9,
    dotplot_top_n_genes: int | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    umap_ncols: int = 3,
) -> None:
    """
    Pseudobulk cluster-vs-rest plots.
    Saves under: marker_genes/pseudobulk/{volcano,dotplot,violin,umap,heatmap}/
    """
    from pathlib import Path
    from . import plot_utils

    _normalize_scanpy_groupby_colors(adata, str(groupby))
    dot_n = int(dotplot_top_n_genes) if dotplot_top_n_genes is not None else int(top_n_genes)

    figroot = Path("marker_genes") / "pseudobulk"
    d_volcano = figroot / "volcano"
    d_dot = figroot / "dotplot"
    d_violin = figroot / "violin"
    d_umap = figroot / "umap"
    d_heat = figroot / "heatmap"

    pb_block = adata.uns.get(store_key, {}).get("pseudobulk_cluster_vs_rest", {})
    pb_results = pb_block.get("results", {}) if isinstance(pb_block, dict) else {}
    if not isinstance(pb_results, dict) or not pb_results:
        return

    genes_by_cluster: dict[str, list[str]] = {}

    for cl, df_de in pb_results.items():
        fig = volcano(
            df_de,
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            title=f"Pseudobulk marker genes: {cl} vs rest",
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{cl}", figdir=d_volcano, fig=fig)

        topg = _select_top_genes(
            df_de,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_n_genes),
            require_sig=True,
        )
        topg_dot = _select_top_genes(
            df_v,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_n),
            require_sig=True,
        )
        genes_by_cluster[str(cl)] = topg

        if topg_dot:
            fig = dotplot_top_genes(
                adata,
                genes=topg_dot,
                groupby=str(groupby),
                use_raw=bool(use_raw),
                layer=layer,
                dendrogram=True,
                show=False,
            )
            plot_utils.save_multi(stem=f"dotplot__{cl}", figdir=d_dot, fig=fig)

        if topg:
            fig = violin_grid_genes(
                adata,
                genes=topg,
                groupby=str(groupby),
                use_raw=bool(use_raw),
                layer=layer,
                ncols=int(max(1, umap_ncols)),
                stripplot=False,
                show=False,
            )
            plot_utils.save_multi(stem=f"violin__{cl}", figdir=d_violin, fig=fig)

            fig = umap_features_grid(
                adata,
                genes=topg,
                use_raw=bool(use_raw),
                layer=layer,
                ncols=int(max(1, umap_ncols)),
                show=False,
            )
            plot_utils.save_multi(stem=f"umap__{cl}__top{int(top_n_genes)}", figdir=d_umap, fig=fig)

    # combined Seurat-style heatmap across clusters
    if any(v for v in genes_by_cluster.values()):
        fig = heatmap_top_genes(
            adata,
            genes_by_cluster=genes_by_cluster,
            groupby=str(groupby),
            use_raw=bool(use_raw),
            layer=layer,
            cmap="bwr",
            z_clip=3.0,
            show=False,
        )
        plot_utils.save_multi(stem="heatmap__marker_genes__all_clusters", figdir=d_heat, fig=fig)


def plot_marker_genes_ranksum(
    adata,
    *,
    groupby: str,
    markers_key: str,
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    top_n_genes: int = 9,
    dotplot_top_n_genes: int | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    umap_ncols: int = 3,
) -> None:
    """
    Cell-level rank_genes_groups plots.
    Saves under: marker_genes/ranksum/{volcano,dotplot,violin,umap,heatmap}/

    IMPORTANT BEHAVIOR:
      - If adata.uns[markers_key]["filtered_by_group"] exists (scomnom posthoc filtering),
        plotting uses that as the canonical marker table per cluster.
      - Otherwise falls back to scanpy.get.rank_genes_groups_df.
      - Never hard-returns just because Scanpy's structured 'names' array is missing.
    """
    from pathlib import Path
    from . import plot_utils
    from scanpy.get import rank_genes_groups_df

    _normalize_scanpy_groupby_colors(adata, str(groupby))
    dot_n = int(dotplot_top_n_genes) if dotplot_top_n_genes is not None else int(top_n_genes)

    figroot = Path("marker_genes") / "ranksum"
    d_volcano = figroot / "volcano"
    d_dot = figroot / "dotplot"
    d_violin = figroot / "violin"
    d_umap = figroot / "umap"
    d_heat = figroot / "heatmap"

    block = adata.uns.get(str(markers_key), None)
    if not isinstance(block, dict) or not block:
        return

    # ----------------------------
    # Resolve groups robustly
    # ----------------------------
    filtered_by_group = None
    if "filtered_by_group" in block and isinstance(block["filtered_by_group"], dict) and block["filtered_by_group"]:
        filtered_by_group = block["filtered_by_group"]
        groups = [str(k) for k in filtered_by_group.keys()]
    else:
        # fallback: derive from obs[groupby] (robust even if scanpy payload is odd)
        if groupby in adata.obs:
            groups = pd.Index(pd.unique(adata.obs[groupby].astype(str))).sort_values().tolist()
        else:
            # final fallback: try scanpy structured names array
            groups = None
            try:
                nm = block.get("names", None)
                if nm is not None and hasattr(nm, "dtype") and getattr(nm.dtype, "names", None):
                    groups = list(nm.dtype.names)
            except Exception:
                groups = None
            groups = groups or []

    if not groups:
        return

    def _normalize_df_for_volcano(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Accept either:
          - scanpy rank_genes_groups_df output (names/logfoldchanges/pvals_adj/...)
          - scomnom filtered_by_group tables (gene/logfoldchanges/pvals_adj/...)
          - already-normalized (gene/log2FoldChange/padj)

        Returns df with columns: gene, log2FoldChange, padj, plus whatever else.
        """
        if df_in is None or getattr(df_in, "empty", True):
            return pd.DataFrame(columns=["gene", "log2FoldChange", "padj", "pval", "score"])

        d = df_in.copy()

        # gene
        if "gene" not in d.columns:
            if "names" in d.columns:
                d = d.rename(columns={"names": "gene"})
            else:
                # try index
                d = d.reset_index().rename(columns={"index": "gene"})

        # lfc
        if "log2FoldChange" not in d.columns:
            if "logfoldchanges" in d.columns:
                d = d.rename(columns={"logfoldchanges": "log2FoldChange"})

        # padj/pval/score
        if "padj" not in d.columns:
            if "pvals_adj" in d.columns:
                d = d.rename(columns={"pvals_adj": "padj"})
        if "pval" not in d.columns:
            if "pvals" in d.columns:
                d = d.rename(columns={"pvals": "pval"})
        if "score" not in d.columns:
            if "scores" in d.columns:
                d = d.rename(columns={"scores": "score"})

        # keep at least required columns
        for col in ["gene", "log2FoldChange", "padj"]:
            if col not in d.columns:
                # create missing columns as NaN so volcano can still render a placeholder
                d[col] = np.nan

        d["gene"] = d["gene"].astype(str)
        return d

    genes_by_cluster: dict[str, list[str]] = {}

    for cl in groups:
        # ----------------------------
        # Fetch marker table per cluster
        # ----------------------------
        df = None
        if filtered_by_group is not None:
            df = filtered_by_group.get(str(cl), None)
        if df is None or getattr(df, "empty", True):
            try:
                df = rank_genes_groups_df(adata, group=str(cl), key=str(markers_key))
            except Exception:
                df = None

        df_v = _normalize_df_for_volcano(df)

        # If df_v has no valid numeric rows, still save a placeholder volcano
        fig = volcano(
            df_v,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            title=f"Ranksum marker genes: {cl} vs rest",
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{cl}", figdir=d_volcano, fig=fig)

        # top genes for expression plots
        topg = _select_top_genes(
            df_v,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_n_genes),
            require_sig=True,
        )
        genes_by_cluster[str(cl)] = topg

        topg_dot = _select_top_genes(
            df_v,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_n),
            require_sig=True,
        )

        if topg_dot:
            fig = dotplot_top_genes(
                adata,
                genes=topg_dot,
                groupby=str(groupby),
                use_raw=bool(use_raw),
                layer=layer,
                dendrogram=True,
                show=False,
            )
            plot_utils.save_multi(stem=f"dotplot__{cl}", figdir=d_dot, fig=fig)

        if topg:
            fig = violin_grid_genes(
                adata,
                genes=topg,
                groupby=str(groupby),
                use_raw=bool(use_raw),
                layer=layer,
                ncols=int(max(1, umap_ncols)),
                stripplot=False,
                show=False,
            )
            plot_utils.save_multi(stem=f"violin__{cl}", figdir=d_violin, fig=fig)

            fig = umap_features_grid(
                adata,
                genes=topg,
                use_raw=bool(use_raw),
                layer=layer,
                ncols=int(max(1, umap_ncols)),
                show=False,
            )
            plot_utils.save_multi(stem=f"umap__{cl}__top{int(top_n_genes)}", figdir=d_umap, fig=fig)

    # combined Seurat-style heatmap across clusters
    if any(v for v in genes_by_cluster.values()):
        fig = heatmap_top_genes(
            adata,
            genes_by_cluster=genes_by_cluster,
            groupby=str(groupby),
            use_raw=bool(use_raw),
            layer=layer,
            cmap="bwr",
            show_gene_labels=True,
            z_clip=3.0,
            show=False,
        )
        plot_utils.save_multi(stem="heatmap__marker_genes__all_clusters", figdir=d_heat, fig=fig)


def plot_condition_within_cluster(
    adata,
    *,
    cluster_key: str,
    condition_key: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    dotplot_top_n: int = 9,
    violin_top_n: int = 9,
    heatmap_top_n: int = 25,
    use_raw: bool = False,
    layer: str | None = None,
) -> None:
    """
    For each stored conditional DE result (per cluster, per contrast):
      - volcano
      - dotplot (top 9 up + top 9 down)
      - violin grid (top 9 up + top 9 down)
      - heatmap (top 25 up + top 25 down), within-cluster, grouped by condition

    Saves under:
      marker_genes/pseudobulk/condition_within_cluster__<condition_key>/<contrast_key>/{volcano,dotplot,violin,heatmap}/
    """
    from pathlib import Path
    from . import plot_utils

    block = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
    if not isinstance(block, dict) or not block:
        return

    base = Path("marker_genes") / "pseudobulk" / f"condition_within_cluster__{condition_key}"

    for k, payload in block.items():
        if not isinstance(payload, dict):
            continue
        df_de = payload.get("results", None)
        if df_de is None or getattr(df_de, "empty", True):
            continue

        # cluster id for subsetting
        group_value = payload.get("group_value", None)
        if group_value is None:
            # fallback heuristic: prefix before "__"
            group_value = str(k).split("__", 1)[0]

        # subset to this cluster only (expression plots should be within-cluster)
        if cluster_key not in adata.obs:
            continue
        mask = adata.obs[cluster_key].astype(str) == str(group_value)
        if not np.any(mask):
            continue
        sub = adata[mask].copy()

        # output folder per contrast key
        safe_k = str(k)
        out_volcano = base / safe_k / "volcano"
        out_dot = base / safe_k / "dotplot"
        out_violin = base / safe_k / "violin"
        out_heat = base / safe_k / "heatmap"

        # volcano
        fig = volcano(
            df_de,
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            title=str(k),
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{k}", figdir=out_volcano, fig=fig)

        # top up/down for dot/violin
        up9 = _select_top_signed(
            df_de,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dotplot_top_n),
            direction="up",
        )
        down9 = _select_top_signed(
            df_de,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dotplot_top_n),
            direction="down",
        )
        genes_9_9 = _unique_keep_order(list(up9) + list(down9))

        if genes_9_9 and condition_key in sub.obs:
            fig = dotplot_top_genes(
                sub,
                genes=genes_9_9,
                groupby=str(condition_key),
                use_raw=bool(use_raw),
                layer=layer,
                dendrogram=False,
                show=False,
            )
            plot_utils.save_multi(stem=f"dotplot__top{int(dotplot_top_n)}up_down__{k}", figdir=out_dot, fig=fig)

            fig = violin_grid_genes(
                sub,
                genes=genes_9_9,
                groupby=str(condition_key),
                use_raw=bool(use_raw),
                layer=layer,
                ncols=3,
                stripplot=False,
                show=False,
            )
            plot_utils.save_multi(stem=f"violin__top{int(violin_top_n)}up_down__{k}", figdir=out_violin, fig=fig)

        # heatmap: top 25 up + top 25 down
        up25 = _select_top_signed(
            df_de,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(heatmap_top_n),
            direction="up",
        )
        down25 = _select_top_signed(
            df_de,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(heatmap_top_n),
            direction="down",
        )
        genes_25_25 = _unique_keep_order(list(up25) + list(down25))

        if genes_25_25 and condition_key in sub.obs:
            fig = heatmap_top_genes(
                sub,
                genes=genes_25_25,
                groupby=str(condition_key),
                use_raw=bool(use_raw),
                layer=layer,
                cmap="bwr",
                show_gene_labels=True,
                z_clip=3.0,
                show=False,
            )
            plot_utils.save_multi(stem=f"heatmap__top{int(heatmap_top_n)}up_down__{k}", figdir=out_heat, fig=fig)


def plot_condition_within_cluster_all(
    adata,
    *,
    cluster_key: str,
    condition_key: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    dotplot_top_n: int = 9,
    violin_top_n: int = 9,
    heatmap_top_n: int = 25,
    use_raw: bool = False,
    layer: str | None = None,
) -> None:
    """
    Full conditional-DE plotting suite:
      1) global UMAP colored by condition_key (once)
      2) per-cluster/per-contrast plots via plot_condition_within_cluster()

    Saves UMAP under:
      marker_genes/pseudobulk/condition_within_cluster__<condition_key>/umap/
    """
    from pathlib import Path
    from . import plot_utils

    # 1) global condition UMAP (ONCE)
    if condition_key in adata.obs:
        plot_utils.umap_by(
            adata,
            keys=str(condition_key),
            figdir=Path("marker_genes") / "pseudobulk" / f"condition_within_cluster__{condition_key}" / "umap",
            stem=f"umap__{condition_key}",
        )

    # 2) per-cluster/per-contrast plots
    plot_condition_within_cluster(
        adata,
        cluster_key=str(cluster_key),
        condition_key=str(condition_key),
        store_key=str(store_key),
        alpha=float(alpha),
        lfc_thresh=float(lfc_thresh),
        top_label_n=int(top_label_n),
        dotplot_top_n=int(dotplot_top_n),
        violin_top_n=int(violin_top_n),
        heatmap_top_n=int(heatmap_top_n),
        use_raw=bool(use_raw),
        layer=layer,
    )
