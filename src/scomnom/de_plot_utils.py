# src/scomnom/de_plot_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import logging
import re
import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import io_utils

LOGGER = logging.getLogger(__name__)



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


def _normalize_levels_for_combo(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _safe_combo_token(value: str) -> str:
    s = str(value).strip()
    s = s.replace("/", "_").replace(":", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", s)
    return s


def _parse_plot_gene_filter_entry(raw: str) -> tuple[Optional[str], Optional[str]]:
    s = str(raw).strip()
    if not s:
        return None, None
    parts = [p.strip() for p in s.split(";") if p.strip()]
    cond = None
    expr = None
    for p in parts:
        low = p.lower()
        if low.startswith("condition_key="):
            cond = p.split("=", 1)[1].strip()
        elif low.startswith("condition="):
            cond = p.split("=", 1)[1].strip()
        elif low.startswith("cond="):
            cond = p.split("=", 1)[1].strip()
        elif low.startswith("expr="):
            expr = p.split("=", 1)[1].strip()
        elif expr is None:
            expr = p
    return cond, expr


def _filter_df_for_plot_genes(
    adata: ad.AnnData,
    df: pd.DataFrame,
    *,
    gene_col: str,
    plot_gene_filter: Optional[Sequence[str]],
    condition_key: Optional[str],
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    if not plot_gene_filter:
        return df
    if gene_col not in df.columns:
        return df

    parsed: list[tuple[Optional[str], str]] = []
    for raw in plot_gene_filter:
        cond, expr = _parse_plot_gene_filter_entry(raw)
        if not expr:
            continue
        parsed.append((cond, expr))
    if not parsed:
        return df

    exprs = [expr for cond, expr in parsed if cond is None or str(cond) == str(condition_key)]
    if not exprs:
        return df

    genes = df[gene_col].astype(str).tolist()
    meta = adata.var.copy()
    meta["gene"] = meta.index.astype(str)
    meta = meta.set_index("gene", drop=False)
    meta = meta.reindex(genes)

    df_idx = df.set_index(gene_col, drop=False)
    for c in df_idx.columns:
        if c not in meta.columns:
            meta[c] = df_idx[c]

    keep = pd.Series(True, index=meta.index)
    for expr in exprs:
        expr_norm = re.sub(r"\bnot_in\b", "not in", str(expr))
        try:
            matched = meta.query(expr_norm, engine="python")
            keep &= meta.index.isin(matched.index)
        except Exception as e:
            LOGGER.warning("plot_gene_filter failed for expr=%r: %s", str(expr), e)
    df_out = df_idx.loc[keep[keep].index].copy()
    return df_out.reset_index(drop=True)


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


def _select_top_signed_with_fallback(
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    padj_thresh: float = 0.05,
    top_n: int = 10,
    direction: str = "up",
) -> list[str]:
    genes = _select_top_signed(
        df,
        gene_col=gene_col,
        padj_col=padj_col,
        lfc_col=lfc_col,
        padj_thresh=padj_thresh,
        top_n=top_n,
        direction=direction,
    )
    if len(genes) >= int(top_n):
        return genes[: int(top_n)]
    if df is None or df.empty:
        return genes[: int(top_n)]
    g = _safe_series(df, gene_col).astype(str)
    padj = pd.to_numeric(_safe_series(df, padj_col), errors="coerce")
    lfc = pd.to_numeric(_safe_series(df, lfc_col), errors="coerce")
    tmp = pd.DataFrame({gene_col: g, padj_col: padj, lfc_col: lfc}).dropna(
        subset=[gene_col, padj_col, lfc_col]
    )
    if tmp.empty:
        return genes[: int(top_n)]
    if direction == "down":
        tmp = tmp[tmp[lfc_col] < 0]
    else:
        tmp = tmp[tmp[lfc_col] > 0]
    if tmp.empty:
        return genes[: int(top_n)]
    tmp["__rank"] = tmp[lfc_col].abs()
    tmp = tmp.sort_values([padj_col, "__rank"], ascending=[True, False])
    extra = _unique_keep_order(tmp[gene_col].astype(str).tolist())
    for gname in extra:
        if gname not in genes:
            genes.append(gname)
        if len(genes) >= int(top_n):
            break
    return genes[: int(top_n)]


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


def _select_top_genes_with_fallback(
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    padj_thresh: float = 0.05,
    top_n: int = 50,
) -> list[str]:
    genes = _select_top_genes(
        df,
        gene_col=gene_col,
        padj_col=padj_col,
        lfc_col=lfc_col,
        padj_thresh=padj_thresh,
        top_n=top_n,
        require_sig=True,
    )
    if len(genes) >= int(top_n):
        return genes[: int(top_n)]
    extra = _select_top_genes(
        df,
        gene_col=gene_col,
        padj_col=padj_col,
        lfc_col=lfc_col,
        padj_thresh=padj_thresh,
        top_n=top_n,
        require_sig=False,
    )
    for g in extra:
        if g not in genes:
            genes.append(g)
        if len(genes) >= int(top_n):
            break
    return genes[: int(top_n)]


def heatmap_top_genes_by_sample(
    adata,
    *,
    genes: Sequence[str],
    sample_key: str,
    condition_key: Optional[str] = None,
    annotation_keys: Optional[Sequence[str]] = None,
    legend_figdir: "Path | None" = None,
    legend_stem: str = "legend",
    use_raw: bool = False,
    layer: Optional[str] = None,
    z_clip: float | None = 3.0,
    cmap: str = "icefire",
    show: bool = False,
):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if not genes or sample_key not in adata.obs:
        return None

    genes = [str(g) for g in genes if g in adata.var_names]
    if not genes:
        return None

    sub = adata[:, genes]
    X = sub.layers[layer] if layer else (sub.raw.X if use_raw and sub.raw is not None else sub.X)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    df_x = pd.DataFrame(X, columns=genes)
    df_x[sample_key] = adata.obs[sample_key].astype(str).values
    mean = df_x.groupby(sample_key, observed=True)[genes].mean()

    M = mean.to_numpy(dtype=float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    if z_clip is not None:
        Z = np.clip(Z, -float(z_clip), float(z_clip))

    data = pd.DataFrame(Z, index=mean.index.astype(str), columns=genes).T

    n_samples = data.shape[1]
    n_genes = data.shape[0]
    fig_w = max(8.0, 0.35 * n_samples + 4.0)
    fig_h = max(8.0, 0.20 * n_genes + 4.0)

    col_colors = None
    legend_handles = []
    keys = [str(k).strip() for k in (annotation_keys or []) if k]
    if not keys and condition_key:
        keys = [str(condition_key)]
    keys = [k for k in keys if k in adata.obs]
    keys = _unique_keep_order(keys)

    if keys:
        cols = [sample_key] + keys
        obs = adata.obs[cols].copy()
        obs[sample_key] = obs[sample_key].astype(str)
        for k in keys:
            obs[k] = obs[k].astype(str)
        cond_by_sample: dict[str, dict[str, str]] = {}
        for s, sub in obs.groupby(sample_key, sort=False):
            per = {}
            for k in keys:
                vals = sub[k].dropna().astype(str).unique()
                if vals.size == 0:
                    continue
                per[k] = str(vals[0])
            if per:
                cond_by_sample[str(s)] = per
        from . import plot_utils
        import matplotlib.colors as mcolors

        def _coerce_palette(raw, n_expected: int) -> list[str] | None:
            if raw is None:
                return None
            colors = None
            if isinstance(raw, dict):
                items: list[tuple[int, str]] = []
                for k0, v0 in raw.items():
                    if str(k0).startswith("__"):
                        continue
                    try:
                        idx = int(k0)
                    except Exception:
                        continue
                    items.append((idx, str(v0)))
                if items:
                    items.sort(key=lambda t: t[0])
                    colors = [v for _, v in items]
            else:
                try:
                    colors = [str(c) for c in list(raw)]
                except Exception:
                    colors = None
            if not colors:
                return None
            out = []
            for c in colors:
                if mcolors.is_color_like(c):
                    out.append(c)
            if len(out) >= int(n_expected):
                return out
            return None

        color_rows = []
        palette_names = ["colorblind", "Set2", "Dark2", "Paired", "tab10", "tab20"]
        palette_offset = 0
        used_colors: set[str] = set()
        try:
            from scanpy.plotting.palettes import default_102
            palette_pool = [str(c) for c in list(default_102)]
        except Exception:
            palette_pool = [plt.matplotlib.colors.to_hex(c) for c in sns.color_palette("tab20", n_colors=20)]
        shared_palette = None
        composite_keys = set()
        if condition_key and ("." in str(condition_key) or "^" in str(condition_key)):
            composite_keys.add(str(condition_key))
        composite_keys.update([k for k in keys if ("." in k or "^" in k)])
        if composite_keys:
            for ck in composite_keys:
                colors_key = f"{ck}_colors"
                if colors_key in adata.uns:
                    try:
                        shared_palette = [str(c) for c in list(adata.uns.get(colors_key, []))]
                        if shared_palette:
                            break
                    except Exception:
                        shared_palette = None
        for idx, k in enumerate(keys):
            levels = [str(x) for x in pd.unique(pd.Series([v.get(k, "") for v in cond_by_sample.values()])).tolist() if str(x)]
            if not levels:
                continue
            levels_cat = None
            try:
                levels_cat = adata.obs[str(k)].astype("category").cat.categories.astype(str).tolist()
            except Exception:
                levels_cat = None
            if levels_cat:
                levels = [lv for lv in levels_cat if lv in levels]

            color_map = {}
            try:
                palette = None
                cats = []
                try:
                    cats = adata.obs[str(k)].astype("category").cat.categories.astype(str).tolist()
                except Exception:
                    cats = levels
                used_shared = False
                if shared_palette and cats:
                    need = len(cats)
                    if len(shared_palette) >= palette_offset + need:
                        palette = shared_palette[palette_offset: palette_offset + need]
                        palette_offset += need
                        used_shared = True
                if palette is None:
                    raw = adata.uns.get(f"{k}_colors", None)
                    palette = _coerce_palette(raw, len(cats))
                if palette is not None and cats:
                    if not used_shared:
                        # If palette overlaps with used colors, pick a distinct slice
                        overlap = any(str(c) in used_colors for c in palette[: len(cats)])
                        if overlap and palette_pool:
                            need = len(cats)
                            tries = 0
                            while tries < max(1, len(palette_pool)):
                                start = int(palette_offset) % max(1, len(palette_pool))
                                cand = []
                                for i in range(need):
                                    cand.append(palette_pool[(start + i) % len(palette_pool)])
                                if not any(c in used_colors for c in cand):
                                    palette = cand
                                    break
                                palette_offset += need
                                tries += 1
                        if palette_offset > 0:
                            shift = int(palette_offset) % max(1, len(palette))
                            if shift:
                                palette = list(palette[shift:]) + list(palette[:shift])
                    color_map = {str(cat): str(palette[i]) for i, cat in enumerate(cats)}
                    if not used_shared:
                        palette_offset += len(cats)
                        used_colors.update([str(palette[i]) for i in range(min(len(palette), len(cats)))])
            except Exception:
                color_map = {}

            if not color_map:
                pal_name = palette_names[idx % len(palette_names)]
                base = [plt.matplotlib.colors.to_hex(c) for c in sns.color_palette(pal_name, n_colors=max(3, len(levels) + palette_offset))]
                if palette_offset >= len(base):
                    palette_offset = 0
                color_list = base[palette_offset: palette_offset + len(levels)]
                if len(color_list) < len(levels):
                    color_list = (color_list * (1 + len(levels) // max(1, len(color_list))))[: len(levels)]
                color_map = dict(zip(levels, color_list))
                palette_offset += len(levels)

            row = [
                color_map.get(cond_by_sample.get(str(s), {}).get(k, ""), (0.85, 0.85, 0.85))
                for s in data.columns
            ]
            color_rows.append(pd.Series(row, index=data.columns, name=str(k)))
            legend_handles.extend(
                [
                    plt.matplotlib.patches.Patch(facecolor=color_map.get(lv, "#BBBBBB"), edgecolor="none", label=f"{k}={lv}")
                    for lv in levels
                ]
            )
        if color_rows:
            col_colors = color_rows

    col_ratio = 0.02
    if col_colors is not None:
        try:
            n_keys = len(col_colors) if isinstance(col_colors, list) else int(col_colors.shape[1])
            if n_genes > 0:
                col_ratio = max(0.003, 0.5 * float(n_keys) / float(n_genes))
            else:
                col_ratio = 0.02 * max(1, int(n_keys))
        except Exception:
            col_ratio = 0.02

    g = sns.clustermap(
        data,
        cmap=cmap,
        center=0.0,
        row_cluster=True,
        col_cluster=True,
        figsize=(fig_w, fig_h),
        yticklabels=True,
        xticklabels=True,
        cbar_kws={"label": "Z-score"},
        col_colors=col_colors,
        linewidths=0.4,
        linecolor="#b0b0b0",
        colors_ratio=(0.02, col_ratio),
    )
    g.ax_heatmap.tick_params(axis="y", labelsize=8)
    g.ax_heatmap.tick_params(axis="x", labelsize=7, rotation=45)
    g.ax_heatmap.grid(False)
    g.ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    if col_colors is not None and hasattr(g, "ax_col_colors") and g.ax_col_colors is not None:
        for coll in g.ax_col_colors.collections:
            try:
                coll.set_edgecolor("none")
                coll.set_linewidth(0.0)
            except Exception:
                pass
    if legend_handles:
        uniq = {}
        for h in legend_handles:
            lbl = h.get_label()
            if lbl not in uniq:
                uniq[lbl] = h
        handles = list(uniq.values())
        ncol = min(4, len(handles))
        title = str(condition_key) if (keys and len(keys) == 1) else "sample annotations"
        g.ax_col_dendrogram.legend(
            handles=handles,
            title=title,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=ncol,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
        if legend_figdir is not None:
            from . import plot_utils
            fig_leg_w = max(4.0, 1.6 * min(4, ncol))
            fig_leg_h = max(2.0, 0.25 * (len(handles) / max(1, ncol)) + 1.2)
            fig_leg, ax_leg = plt.subplots(figsize=(fig_leg_w, fig_leg_h))
            ax_leg.axis("off")
            ax_leg.legend(
                handles=handles,
                title=title,
                loc="center",
                ncol=ncol,
                frameon=False,
                fontsize=9,
                title_fontsize=10,
            )
            plot_utils.save_multi(stem=str(legend_stem), figdir=legend_figdir, fig=fig_leg)
            plt.close(fig_leg)
    if show:
        plt.show()
    return g.fig

def _filter_protein_coding(
    adata,
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    gene_type_col: str = "gene_type",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if gene_col not in df.columns:
        return df
    df2 = io_utils._add_gene_type_column(adata, df, gene_col=gene_col, gene_type_col=gene_type_col)
    if gene_type_col not in df2.columns:
        return df2
    pc = df2[df2[gene_type_col] == "protein_coding"].copy()
    if "gene_chrom" in pc.columns:
        pc = pc[pc["gene_chrom"].astype(str) != "MT"].copy()
    return pc


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
    label_genes: Optional[Sequence[str]] = None,
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
    plot_mask = (tmp[padj_col].to_numpy(dtype=float) > 0) & np.isfinite(
        tmp[padj_col].to_numpy(dtype=float)
    )
    tmp_plot = tmp.loc[plot_mask].copy()
    tmp_plot = tmp_plot[tmp_plot[padj_col] <= float(padj_thresh)].copy()
    if tmp_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No significant rows (padj > threshold)", ha="center", va="center")
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    padj_np = _clip_padj(tmp_plot[padj_col].to_numpy(dtype=float))
    y = -np.log10(padj_np)
    x = tmp_plot[lfc_col].to_numpy(dtype=float)

    x_abs = np.abs(x)
    if x_abs.size and np.isfinite(x_abs).any():
        x_cap = float(np.nanmax(x_abs))
    else:
        x_cap = 0.0
    if x_cap < float(lfc_thresh):
        x_cap = float(lfc_thresh) * 1.1
    x_pad = 0.05 * x_cap
    x_plot = x.copy()

    sig = (tmp_plot[padj_col].to_numpy(dtype=float) < float(padj_thresh)) & (
        np.abs(x) > float(lfc_thresh)
    )

    fig, ax = plt.subplots(figsize=figsize)

    # plot non-sig then sig on top
    ax.scatter(
        x_plot[~sig],
        y[~sig],
        c="#9aa0a6",  # soft gray
        s=s,
        alpha=alpha,
        linewidths=0.0,
        rasterized=True,
    )
    ax.scatter(
        x_plot[sig],
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
    ax.set_xlim(-(x_cap + x_pad), (x_cap + x_pad))

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 adjusted p-value")

    if title:
        ax.set_title(str(title))

    # labels
    if int(top_label_n) > 0:
        if label_genes is None:
            label_genes = _select_top_genes_with_fallback(
                tmp,
                gene_col=gene_col,
                padj_col=padj_col,
                lfc_col=lfc_col,
                padj_thresh=padj_thresh,
                top_n=int(top_label_n),
            )
        if label_genes:
            # annotate those rows (pick first occurrence per gene)
            tmp_idx = tmp_plot.reset_index(drop=True)
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
        bottom=0.30  # space for rotated gene labels
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

    fig.tight_layout(rect=(0.25, 0.30, 1, 1))

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
    # 2) Column widths
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
        # Increased base width for the right-hand labels
        W = max(9.0, 0.3 * float(x_edges[-1]) + 5.0)
        H = max(5.0, 0.2 * float(plot_mat.shape[0]) + 2.5)
        figsize = (W, H)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2 if colors is not None else 1,
        ncols=2,
        height_ratios=[0.02, 1.0] if colors is not None else [1.0],
        width_ratios=[1.0, 0.01],
        hspace=0.01, wspace=0.05
    )

    ax = fig.add_subplot(gs[-1, 0])
    cax = fig.add_subplot(gs[-1, 1])

    # -----------------------------
    # 6) Heatmap Draw (Kill Grid & Seams)
    # -----------------------------
    ax.grid(False)

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
    # 9) Top color bar + Labels (Increased Margin)
    # -----------------------------
    if colors is not None:
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_top.grid(False)
        for i in range(len(groups)):
            ax_top.add_patch(
                plt.Rectangle((float(x_edges[i]), 0.0), float(w[i]), 1.0, color=colors[i], lw=0)
            )
            ax_top.text(
                x_centers[i], 1.2, groups[i],
                ha='left', va='bottom',
                rotation=45,
                fontsize=11, fontweight='bold',
                transform=ax_top.get_xaxis_transform(),
                clip_on=False
            )
        ax_top.set_xlim(0, float(x_edges[-1]))
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")

    # Final margin adjustment to catch the right-most labels
    fig.subplots_adjust(right=0.85, top=0.85)

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
    rotation: float = 45,
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
        n_groups = int(adata.obs[str(groupby)].astype(str).nunique())
        per_plot_w = max(3.2, min(0.25 * n_groups, 8.0))
        figsize = (max(8.0, per_plot_w * min(ncols, n)), max(4.0, 3.2 * nrows))

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
        if layer in (None, "", "X"):
            ylabel = "Expression (X)" if not use_raw else "Expression (raw)"
        else:
            ylabel = f"Expression ({layer})"
        try:
            ax.set_ylabel(ylabel)
        except Exception:
            pass

    # ---- turn off unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # ---- Hide cluster/category labels except on last visible row per column
    for c in range(ncols):
        last_r = None
        for r in range(nrows):
            ax = axes[r][c]
            if not isinstance(ax, Axes):
                continue
            if not ax.get_visible():
                continue
            if not getattr(ax, "axison", True):
                continue
            last_r = r
        for r in range(nrows):
            ax = axes[r][c]
            if not isinstance(ax, Axes):
                continue
            if not ax.get_visible() or not getattr(ax, "axison", True):
                continue
            if last_r is None or r != last_r:
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
    rotation: int | float | None = 45,
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
    rasterize: bool = True,
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
    if bool(rasterize):
        for ax in fig.get_axes():
            try:
                for coll in ax.collections:
                    coll.set_rasterized(True)
            except Exception:
                pass
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
    dotplot_top_n_genes: int | None = 16,
    use_raw: bool = False,
    layer: str | None = None,
    umap_ncols: int = 3,
    plot_gene_filter: Optional[Sequence[str]] = None,
) -> None:
    """
    Pseudobulk cluster-vs-rest plots.
    Saves under: marker_genes/pseudobulk/{volcano,dotplot,violin,umap,heatmap}/
    """
    from pathlib import Path
    from . import plot_utils

    _normalize_scanpy_groupby_colors(adata, str(groupby))
    dot_n = int(dotplot_top_n_genes) if dotplot_top_n_genes is not None else int(top_n_genes)

    figroot = Path("markers") / "pseudobulk_markers"
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
        df_pc = _filter_protein_coding(adata, df_de, gene_col="gene")
        df_plot = _filter_df_for_plot_genes(
            adata,
            df_pc,
            gene_col="gene",
            plot_gene_filter=plot_gene_filter,
            condition_key=None,
        )
        label_genes = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_label_n),
        )
        fig = volcano(
            df_de,
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            label_genes=label_genes,
            title=f"Pseudobulk marker genes: {cl} vs rest",
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{cl}", figdir=d_volcano, fig=fig)

        topg = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_n_genes),
        )
        topg_dot = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_n),
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
    dotplot_top_n_genes: int | None = 16,
    use_raw: bool = False,
    layer: str | None = None,
    umap_ncols: int = 3,
    plot_gene_filter: Optional[Sequence[str]] = None,
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

    figroot = Path("markers") / "cell_level_markers"
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
        df_pc = _filter_protein_coding(adata, df_v, gene_col="gene")
        df_plot = _filter_df_for_plot_genes(
            adata,
            df_pc,
            gene_col="gene",
            plot_gene_filter=plot_gene_filter,
            condition_key=None,
        )
        label_genes = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_label_n),
        )

        # If df_v has no valid numeric rows, still save a placeholder volcano
        fig = volcano(
            df_v,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            label_genes=label_genes,
            title=f"Ranksum marker genes: {cl} vs rest",
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{cl}", figdir=d_volcano, fig=fig)

        # top genes for expression plots
        topg = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_n_genes),
        )
        genes_by_cluster[str(cl)] = topg

        topg_dot = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_n),
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
    dotplot_top_n: int = 16,
    violin_top_n: int = 12,
    heatmap_top_n: int = 25,
    use_raw: bool = False,
    layer: str | None = None,
    sample_key: str | None = None,
    plot_gene_filter: Optional[Sequence[str]] = None,
    annotation_keys: Optional[Sequence[str]] = None,
) -> None:
    """
    For each stored conditional DE result (per cluster, per contrast):
      - volcano
      - dotplot (top N/2 up + top N/2 down)
      - violin grid (top N/2 up + top N/2 down)
      - heatmap (top 25 up + top 25 down), within-cluster, grouped by condition

    Saves under:
      DE/pseudobulk_DE/<condition_key>/<contrast_key>/{volcano,dotplot,violin,heatmap}/
      (interaction: DE/pseudobulk_DE/<condition_key>__interaction/interaction/{volcano,dotplot,violin,heatmap}/)
    """
    from pathlib import Path
    from . import plot_utils

    if sample_key is None:
        try:
            sample_key = io_utils.infer_batch_key(adata, None)
        except Exception:
            sample_key = None

    block = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
    if not isinstance(block, dict) or not block:
        block = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group_multi", {})
    if not isinstance(block, dict) or not block:
        return

    base_default = Path("DE") / "pseudobulk_DE" / f"{condition_key}"

    for k, payload in block.items():
        if not isinstance(payload, dict):
            continue
        payload_ck = payload.get("condition_key", None)
        if payload_ck is not None and str(payload_ck) != str(condition_key):
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

        # output folder per contrast key (no per-cluster folders)
        test = payload.get("test", None)
        ref = payload.get("reference", None)
        is_interaction = bool(payload.get("interaction", False)) or ("^" in str(condition_key))
        if is_interaction:
            pair_dir = "interaction"
            base = Path("DE") / "pseudobulk_DE" / f"{condition_key}__interaction"
        else:
            base = base_default
            if test and ref:
                pair_dir = io_utils.sanitize_identifier(f"{test}_vs_{ref}")
            else:
                pair_dir = io_utils.sanitize_identifier(str(k))

        out_volcano = base / pair_dir / "volcano"
        out_dot = base / pair_dir / "dotplot"
        out_violin = base / pair_dir / "violin"
        out_heat = base / pair_dir / "heatmap"
        out_heat_sample = base / pair_dir / "heatmap_sample"

        df_pc = _filter_protein_coding(adata, df_de, gene_col="gene")
        df_plot = _filter_df_for_plot_genes(
            adata,
            df_pc,
            gene_col="gene",
            plot_gene_filter=plot_gene_filter,
            condition_key=str(condition_key),
        )
        label_genes = _select_top_genes_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(top_label_n),
        )

        # volcano
        fig = volcano(
            df_de,
            padj_thresh=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            label_genes=label_genes,
            title=str(k),
            show=False,
        )
        plot_utils.save_multi(stem=f"volcano__{group_value}__{pair_dir}", figdir=out_volcano, fig=fig)

        dot_half = max(1, int(dotplot_top_n) // 2)
        dot_remainder = int(dotplot_top_n) - dot_half
        vio_half = max(1, int(violin_top_n) // 2)
        vio_remainder = int(violin_top_n) - vio_half

        # top up/down for dotplot
        up_dot = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_half),
            direction="up",
        )
        down_dot = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(dot_remainder),
            direction="down",
        )
        genes_dot = _unique_keep_order(list(up_dot) + list(down_dot))

        # top up/down for violin
        up_vio = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(vio_half),
            direction="up",
        )
        down_vio = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(vio_remainder),
            direction="down",
        )
        genes_vio = _unique_keep_order(list(up_vio) + list(down_vio))

        plot_key = str(condition_key)
        if plot_key not in sub.obs and is_interaction:
            factor_a = payload.get("factor_a") if isinstance(payload, dict) else None
            factor_b = payload.get("factor_b") if isinstance(payload, dict) else None
            if factor_a is None or factor_b is None:
                if "^" in str(condition_key):
                    parts = str(condition_key).split("^", 1)
                    if len(parts) == 2:
                        factor_a, factor_b = parts[0], parts[1]
            if factor_a in sub.obs and factor_b in sub.obs:
                combo_key = f"{_safe_combo_token(factor_a)}.{_safe_combo_token(factor_b)}"
                if combo_key not in sub.obs:
                    a_vals = _normalize_levels_for_combo(sub.obs[str(factor_a)])
                    b_vals = _normalize_levels_for_combo(sub.obs[str(factor_b)])
                    sub.obs[combo_key] = a_vals + "." + b_vals
                plot_key = combo_key

        if genes_dot and plot_key in sub.obs:
            fig = dotplot_top_genes(
                sub,
                genes=genes_dot,
                groupby=str(plot_key),
                use_raw=bool(use_raw),
                layer=layer,
                dendrogram=False,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"dotplot__top{int(dotplot_top_n)}up_down__{group_value}__{pair_dir}",
                figdir=out_dot,
                fig=fig,
            )

        if genes_vio and plot_key in sub.obs:
            fig = violin_grid_genes(
                sub,
                genes=genes_vio,
                groupby=str(plot_key),
                use_raw=bool(use_raw),
                layer=layer,
                ncols=3,
                stripplot=False,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"violin__top{int(violin_top_n)}up_down__{group_value}__{pair_dir}",
                figdir=out_violin,
                fig=fig,
            )

        # heatmap: top 25 up + top 25 down
        up25 = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(heatmap_top_n),
            direction="up",
        )
        down25 = _select_top_signed_with_fallback(
            df_plot,
            gene_col="gene",
            padj_col="padj",
            lfc_col="log2FoldChange",
            padj_thresh=float(alpha),
            top_n=int(heatmap_top_n),
            direction="down",
        )
        genes_25_25 = _unique_keep_order(list(up25) + list(down25))

        if genes_25_25 and plot_key in sub.obs:
            fig = heatmap_top_genes(
                sub,
                genes=genes_25_25,
                groupby=str(plot_key),
                use_raw=bool(use_raw),
                layer=layer,
                cmap="bwr",
                show_gene_labels=True,
                z_clip=3.0,
                show=False,
            )
            plot_utils.save_multi(
                stem=f"heatmap__top{int(heatmap_top_n)}up_down__{group_value}__{pair_dir}",
                figdir=out_heat,
                fig=fig,
            )

        # sample-aggregated heatmap: top 50 DE genes per contrast
        if sample_key is not None and sample_key in sub.obs and plot_key in sub.obs:
            top50 = _select_top_genes_with_fallback(
                df_plot,
                gene_col="gene",
                padj_col="padj",
                lfc_col="log2FoldChange",
                padj_thresh=float(alpha),
                top_n=50,
            )
            fig = heatmap_top_genes_by_sample(
                sub,
                genes=top50,
                sample_key=str(sample_key),
                condition_key=str(plot_key),
                annotation_keys=annotation_keys,
                legend_figdir=out_heat_sample,
                legend_stem="legend",
                use_raw=bool(use_raw),
                layer=layer,
                z_clip=3.0,
                show=False,
            )
            if fig is not None:
                plot_utils.save_multi(
                    stem=f"heatmap_samples__top50__{group_value}__{pair_dir}",
                    figdir=out_heat_sample,
                    fig=fig,
                )


def plot_condition_within_cluster_all(
    adata,
    *,
    cluster_key: str,
    condition_key: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    dotplot_top_n: int = 16,
    violin_top_n: int = 12,
    heatmap_top_n: int = 25,
    use_raw: bool = False,
    layer: str | None = None,
    sample_key: str | None = None,
    plot_gene_filter: Optional[Sequence[str]] = None,
    annotation_keys: Optional[Sequence[str]] = None,
) -> None:
    """
    Full conditional-DE plotting suite:
      - per-cluster/per-contrast plots via plot_condition_within_cluster()
    """
    from pathlib import Path
    from . import plot_utils

    # per-cluster/per-contrast plots
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
        sample_key=sample_key,
        plot_gene_filter=plot_gene_filter,
        annotation_keys=annotation_keys,
    )


def plot_condition_umaps(
    adata,
    *,
    groupby: str,
    condition_keys: Sequence[str],
) -> None:
    """
    UMAPs colored by condition_key labels (global + per-cluster).

    Saves under:
      DE/UMAP/<condition_key>/umap__<condition_key>.*
      DE/UMAP/<condition_key>/umap__<condition_key>__<cluster>.*
    """
    from pathlib import Path
    import matplotlib.colors as mcolors
    import seaborn as sns
    from . import plot_utils

    outroot = Path("DE") / "UMAP"
    for ck in [str(k) for k in condition_keys]:
        plot_key = ck
        if plot_key not in adata.obs and "^" in ck:
            parts = ck.split("^", 1)
            if len(parts) == 2:
                factor_a, factor_b = parts[0], parts[1]
                if factor_a in adata.obs and factor_b in adata.obs:
                    plot_key = f"{_safe_combo_token(factor_a)}.{_safe_combo_token(factor_b)}"
                    if plot_key not in adata.obs:
                        a_vals = _normalize_levels_for_combo(adata.obs[str(factor_a)])
                        b_vals = _normalize_levels_for_combo(adata.obs[str(factor_b)])
                        adata.obs[plot_key] = a_vals + "." + b_vals
        if plot_key not in adata.obs:
            continue
        outdir = outroot / str(ck)
        try:
            cats = adata.obs[plot_key].astype("category").cat.categories
            n_cats = int(len(cats))
            palette = sns.color_palette("colorblind", n_colors=n_cats)
            colors = [mcolors.to_hex(c) for c in palette]
            adata.uns[f"{plot_key}_colors"] = list(colors)
        except Exception:
            pass

        plot_utils.umap_by(
            adata,
            keys=[plot_key],
            figdir=outdir,
            stem=f"umap__{ck}",
        )

        if groupby in adata.obs and "X_umap" in adata.obsm:
            clusters = pd.Index(pd.unique(adata.obs[str(groupby)].astype(str))).sort_values().tolist()
            for cl in clusters:
                m = adata.obs[str(groupby)].astype(str).to_numpy() == str(cl)
                if not m.any():
                    continue
                adata_umap = ad.AnnData(X=np.zeros((int(m.sum()), 0)))
                adata_umap.obs = adata.obs.loc[m, [plot_key]].copy()
                adata_umap.obsm["X_umap"] = adata.obsm["X_umap"][m]
                if f"{plot_key}_colors" in adata.uns:
                    adata_umap.uns[f"{plot_key}_colors"] = list(adata.uns[f"{plot_key}_colors"])
                plot_utils.umap_by(
                    adata_umap,
                    keys=[plot_key],
                    figdir=outdir,
                    stem=f"umap__{ck}__{cl}",
                )


def plot_contrast_conditional_markers(
    adata,
    *,
    groupby: str,
    contrast_key: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    top_n_genes: int = 12,
    dotplot_top_n_genes: int | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    sample_key: str | None = None,
    plot_gene_filter: Optional[Sequence[str]] = None,
    annotation_keys: Optional[Sequence[str]] = None,
) -> None:
    """
    Cell-level within-cluster contrast plots (from contrast_conditional results).
    Generates volcano/dotplot/violin per cluster+contrast.
    """
    from pathlib import Path
    from . import plot_utils, io_utils

    block = adata.uns.get(store_key, {}).get("contrast_conditional", {})
    results = block.get("results", {}) if isinstance(block, dict) else {}
    if not isinstance(results, dict) or not results:
        return

    figroot = Path("DE") / "cell_level_DE" / str(contrast_key)
    if sample_key is None:
        try:
            sample_key = io_utils.infer_batch_key(adata, None)
        except Exception:
            sample_key = None

    dot_n = int(dotplot_top_n_genes) if dotplot_top_n_genes is not None else int(top_n_genes)

    for cl, per_contrast in results.items():
        if not isinstance(per_contrast, dict):
            continue

        for pair_key, tables in per_contrast.items():
            if not isinstance(tables, dict):
                continue

            pair_dir = io_utils.sanitize_identifier(str(pair_key))
            pair_root = figroot / pair_dir
            d_volcano = pair_root / "volcano"
            d_dot = pair_root / "dotplot"
            d_violin = pair_root / "violin"
            d_umap = pair_root / "umap"
            d_heat_sample = pair_root / "heatmap_sample"

            wilcoxon_df = tables.get("wilcoxon", pd.DataFrame())
            combined_df = tables.get("combined", pd.DataFrame())

            # Build a normalized df for volcano/top-gene selection
            df_volc = None
            if isinstance(wilcoxon_df, pd.DataFrame) and not wilcoxon_df.empty:
                df_volc = pd.DataFrame(
                    {
                        "gene": wilcoxon_df.get("gene"),
                        "log2FoldChange": wilcoxon_df.get("cell_wilcoxon_logfc"),
                        "padj": wilcoxon_df.get("cell_wilcoxon_padj"),
                    }
                )
            elif isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
                df_volc = pd.DataFrame(
                    {
                        "gene": combined_df.get("gene"),
                        "log2FoldChange": combined_df.get("cell_wilcoxon_logfc"),
                        "padj": combined_df.get("cell_wilcoxon_padj"),
                    }
                )

            df_pc = None
            df_plot = None
            label_genes = None
            if df_volc is not None and not df_volc.empty:
                df_pc = _filter_protein_coding(adata, df_volc, gene_col="gene")
                df_plot = _filter_df_for_plot_genes(
                    adata,
                    df_pc,
                    gene_col="gene",
                    plot_gene_filter=plot_gene_filter,
                    condition_key=str(contrast_key),
                )
                label_genes = _select_top_genes_with_fallback(
                    df_plot,
                    gene_col="gene",
                    padj_col="padj",
                    lfc_col="log2FoldChange",
                    padj_thresh=float(alpha),
                    top_n=int(top_label_n),
                )
                fig = volcano(
                    df_volc,
                    padj_col="padj",
                    lfc_col="log2FoldChange",
                    padj_thresh=float(alpha),
                    lfc_thresh=float(lfc_thresh),
                    top_label_n=int(top_label_n),
                    label_genes=label_genes,
                    title=f"Within-cluster contrast: {cl} {pair_key}",
                    show=False,
                )
                plot_utils.save_multi(stem=f"volcano__{cl}__{pair_key}", figdir=d_volcano, fig=fig)

            dot_half = max(1, int(dot_n) // 2)
            dot_remainder = int(dot_n) - dot_half
            vio_half = max(1, int(top_n_genes) // 2)
            vio_remainder = int(top_n_genes) - vio_half
            if df_plot is None and df_pc is not None:
                df_plot = df_pc
            df_sel = df_plot if df_plot is not None else pd.DataFrame()

            up_dot = _select_top_signed_with_fallback(
                df_sel,
                gene_col="gene",
                padj_col="padj",
                lfc_col="log2FoldChange",
                padj_thresh=float(alpha),
                top_n=int(dot_half),
                direction="up",
            )
            down_dot = _select_top_signed_with_fallback(
                df_sel,
                gene_col="gene",
                padj_col="padj",
                lfc_col="log2FoldChange",
                padj_thresh=float(alpha),
                top_n=int(dot_remainder),
                direction="down",
            )
            topg_dot = _unique_keep_order(list(up_dot) + list(down_dot))

            up_vio = _select_top_signed_with_fallback(
                df_sel,
                gene_col="gene",
                padj_col="padj",
                lfc_col="log2FoldChange",
                padj_thresh=float(alpha),
                top_n=int(vio_half),
                direction="up",
            )
            down_vio = _select_top_signed_with_fallback(
                df_sel,
                gene_col="gene",
                padj_col="padj",
                lfc_col="log2FoldChange",
                padj_thresh=float(alpha),
                top_n=int(vio_remainder),
                direction="down",
            )
            topg = _unique_keep_order(list(up_vio) + list(down_vio))

            if topg_dot:
                m = adata.obs[str(groupby)].astype(str).to_numpy() == str(cl)
                adata_sub = adata[m].copy()
                fig = dotplot_top_genes(
                    adata_sub,
                    genes=topg_dot,
                    groupby=str(contrast_key),
                    use_raw=bool(use_raw),
                    layer=layer,
                    dendrogram=False,
                    show=False,
                )
                plot_utils.save_multi(stem=f"dotplot__{cl}__{pair_key}", figdir=d_dot, fig=fig)

            if topg:
                m = adata.obs[str(groupby)].astype(str).to_numpy() == str(cl)
                adata_sub = adata[m].copy()
                fig = violin_grid_genes(
                    adata_sub,
                    genes=topg,
                    groupby=str(contrast_key),
                    use_raw=bool(use_raw),
                    layer=layer,
                    ncols=3,
                    stripplot=False,
                    show=False,
                )
                plot_utils.save_multi(stem=f"violin__{cl}__{pair_key}", figdir=d_violin, fig=fig)

            # sample-aggregated heatmap (top 50 DE genes)
            if sample_key is not None:
                m = adata.obs[str(groupby)].astype(str).to_numpy() == str(cl)
                adata_sub = adata[m].copy()
                if sample_key in adata_sub.obs:
                    top50 = _select_top_genes_with_fallback(
                        df_sel,
                        gene_col="gene",
                        padj_col="padj",
                        lfc_col="log2FoldChange",
                        padj_thresh=float(alpha),
                        top_n=50,
                    )
                    fig = heatmap_top_genes_by_sample(
                        adata_sub,
                        genes=top50,
                        sample_key=str(sample_key),
                        condition_key=str(contrast_key),
                        annotation_keys=annotation_keys,
                        legend_figdir=d_heat_sample,
                        legend_stem="legend",
                        use_raw=bool(use_raw),
                        layer=layer,
                        z_clip=3.0,
                        show=False,
                    )
                    if fig is not None:
                        plot_utils.save_multi(
                            stem=f"heatmap_samples__top50__{cl}__{pair_key}",
                            figdir=d_heat_sample,
                            fig=fig,
                        )


def plot_contrast_conditional_markers_multi(
    adata,
    *,
    groupby: str,
    contrast_key: str,
    store_key: str = "scomnom_de",
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    top_label_n: int = 15,
    top_n_genes: int = 12,
    dotplot_top_n_genes: int | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    sample_key: str | None = None,
    plot_gene_filter: Optional[Sequence[str]] = None,
    annotation_keys: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot contrast-conditional markers for a specific contrast_key from
    adata.uns[store_key]["contrast_conditional_multi"].
    """
    block = adata.uns.get(store_key, {})
    multi = block.get("contrast_conditional_multi", None)
    if not isinstance(multi, dict):
        return

    payload = multi.get(str(contrast_key), None)
    if not isinstance(payload, dict):
        return

    orig = block.get("contrast_conditional", None)
    block["contrast_conditional"] = payload
    try:
        plot_contrast_conditional_markers(
            adata,
            groupby=str(groupby),
            contrast_key=str(contrast_key),
            store_key=str(store_key),
            alpha=float(alpha),
            lfc_thresh=float(lfc_thresh),
            top_label_n=int(top_label_n),
            top_n_genes=int(top_n_genes),
            dotplot_top_n_genes=dotplot_top_n_genes,
            use_raw=bool(use_raw),
            layer=layer,
            sample_key=sample_key,
            plot_gene_filter=plot_gene_filter,
            annotation_keys=annotation_keys,
        )
    finally:
        if orig is None:
            block.pop("contrast_conditional", None)
        else:
            block["contrast_conditional"] = orig


def plot_de_decoupler_payload(
    payload: dict,
    *,
    net_name: str,
    figdir: "Path",
    heatmap_top_k: int = 30,
    bar_top_n: int = 10,
    bar_top_n_up: int | None = None,
    bar_top_n_down: int | None = None,
    bar_split_signed: bool = True,
    dotplot_top_k: int = 30,
    title_prefix: str | None = None,
    pos_label: str | None = None,
    neg_label: str | None = None,
) -> None:
    """
    Plot decoupler activity payload produced from DE stats.
    """
    from . import plot_utils

    if not isinstance(payload, dict):
        return
    activity = payload.get("activity", None)
    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        return

    def _split_activity_signed(
        activity_df: pd.DataFrame,
        *,
        n_up: int,
        n_down: int,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if activity_df is None or activity_df.empty:
            return None, None
        A = activity_df.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scores = A.mean(axis=0)
        up = scores[scores > 0].sort_values(ascending=False).head(int(n_up))
        down = scores[scores < 0].sort_values(ascending=True).head(int(n_down))
        up_df = A.loc[:, up.index].copy() if not up.empty else None
        down_df = A.loc[:, down.index].copy() if not down.empty else None
        return up_df, down_df

    pos_text = str(pos_label) if pos_label else "up"
    neg_text = str(neg_label) if neg_label else "down"
    pos_stem = io_utils.sanitize_identifier(pos_text)
    neg_stem = io_utils.sanitize_identifier(neg_text)

    if str(net_name).lower().strip() == "msigdb":
        splits = payload.get("activity_by_gmt", None)
        if isinstance(splits, dict) and splits:
            splits = {str(k): v for k, v in splits.items() if isinstance(v, pd.DataFrame) and not v.empty}
        else:
            splits = plot_utils._split_activity_for_msigdb(activity)

        if not splits:
            return

        ordered = sorted(
            splits.keys(),
            key=lambda x: (0 if str(x).upper() == "HALLMARK" else 1, str(x).upper()),
        )

        for pfx in ordered:
            sub = splits[pfx]
            if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
                continue
            tprefix = f"{str(pfx).upper()}" if title_prefix is None else f"{str(pfx).upper()} [{title_prefix}]"
            plot_utils.plot_decoupler_cluster_topn_barplots(
                sub,
                net_name=str(net_name),
                figdir=figdir,
                n=int(bar_top_n),
                use_abs=False,
                split_signed=bool(bar_split_signed),
                n_pos=int(bar_top_n_up) if bar_top_n_up is not None else int(bar_top_n),
                n_neg=int(bar_top_n_down) if bar_top_n_down is not None else int(bar_top_n),
                stem_prefix=f"cluster_{str(pfx).lower()}",
                title_prefix=tprefix,
            )
            up_df, down_df = _split_activity_signed(
                sub,
                n_up=int(heatmap_top_k),
                n_down=int(heatmap_top_k),
            )
            if up_df is not None and not up_df.empty:
                plot_utils.plot_decoupler_activity_heatmap(
                    up_df,
                    net_name=str(net_name),
                    figdir=figdir,
                    top_k=int(heatmap_top_k),
                    rank_mode="var",
                    use_zscore=True,
                    wrap_labels=True,
                    stem=f"heatmap_top_{str(pfx).lower()}_{pos_stem}_",
                    title_prefix=f"{tprefix} ({pos_text})",
                )
                plot_utils.plot_decoupler_dotplot(
                    up_df,
                    net_name=str(net_name),
                    figdir=figdir,
                    top_k=int(dotplot_top_k),
                    rank_mode="var",
                    color_by="z",
                    size_by="abs_raw",
                    wrap_labels=True,
                    stem=f"dotplot_top_{str(pfx).lower()}_{pos_stem}_",
                    title_prefix=f"{tprefix} ({pos_text})",
                )
            if down_df is not None and not down_df.empty:
                plot_utils.plot_decoupler_activity_heatmap(
                    down_df,
                    net_name=str(net_name),
                    figdir=figdir,
                    top_k=int(heatmap_top_k),
                    rank_mode="var",
                    use_zscore=True,
                    wrap_labels=True,
                    stem=f"heatmap_top_{str(pfx).lower()}_{neg_stem}_",
                    title_prefix=f"{tprefix} ({neg_text})",
                )
                plot_utils.plot_decoupler_dotplot(
                    down_df,
                    net_name=str(net_name),
                    figdir=figdir,
                    top_k=int(dotplot_top_k),
                    rank_mode="var",
                    color_by="z",
                    size_by="abs_raw",
                    wrap_labels=True,
                    stem=f"dotplot_top_{str(pfx).lower()}_{neg_stem}_",
                    title_prefix=f"{tprefix} ({neg_text})",
                )
        return

    plot_utils.plot_decoupler_cluster_topn_barplots(
        activity,
        net_name=str(net_name),
        figdir=figdir,
        n=int(bar_top_n),
        use_abs=False,
        split_signed=bool(bar_split_signed) and str(net_name).lower() in ("dorothea", "msigdb"),
        n_pos=int(bar_top_n_up) if bar_top_n_up is not None else int(bar_top_n),
        n_neg=int(bar_top_n_down) if bar_top_n_down is not None else int(bar_top_n),
        stem_prefix="cluster",
        title_prefix=title_prefix,
    )
    up_df, down_df = _split_activity_signed(
        activity,
        n_up=int(heatmap_top_k),
        n_down=int(heatmap_top_k),
    )
    if up_df is not None and not up_df.empty:
        plot_utils.plot_decoupler_activity_heatmap(
            up_df,
            net_name=str(net_name),
            figdir=figdir,
            top_k=int(heatmap_top_k),
            rank_mode="var",
            use_zscore=True,
            wrap_labels=True,
            stem=f"heatmap_top_{pos_stem}_",
            title_prefix=f"{title_prefix} ({pos_text})" if title_prefix else pos_text,
        )
        plot_utils.plot_decoupler_dotplot(
            up_df,
            net_name=str(net_name),
            figdir=figdir,
            top_k=int(dotplot_top_k),
            rank_mode="var",
            color_by="z",
            size_by="abs_raw",
            wrap_labels=True,
            stem=f"dotplot_top_{pos_stem}_",
            title_prefix=f"{title_prefix} ({pos_text})" if title_prefix else pos_text,
        )
    if down_df is not None and not down_df.empty:
        plot_utils.plot_decoupler_activity_heatmap(
            down_df,
            net_name=str(net_name),
            figdir=figdir,
            top_k=int(heatmap_top_k),
            rank_mode="var",
            use_zscore=True,
            wrap_labels=True,
            stem=f"heatmap_top_{neg_stem}_",
            title_prefix=f"{title_prefix} ({neg_text})" if title_prefix else neg_text,
        )
        plot_utils.plot_decoupler_dotplot(
            down_df,
            net_name=str(net_name),
            figdir=figdir,
            top_k=int(dotplot_top_k),
            rank_mode="var",
            color_by="z",
            size_by="abs_raw",
            wrap_labels=True,
            stem=f"dotplot_top_{neg_stem}_",
            title_prefix=f"{title_prefix} ({neg_text})" if title_prefix else neg_text,
        )
