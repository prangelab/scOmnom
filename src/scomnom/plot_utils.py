from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Mapping, Iterable, List, Any

import logging
import math
from sklearn.metrics import silhouette_samples

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from matplotlib.collections import LineCollection

LOGGER = logging.getLogger(__name__)

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


# -------------------------------------------------------------------------
# Setup + saving
# -------------------------------------------------------------------------
def set_figure_formats(formats: Sequence[str]) -> None:
    global FIGURE_FORMATS
    FIGURE_FORMATS = list(formats)


def setup_scanpy_figs(figdir: Path, formats: Sequence[str] | None = None) -> None:
    """
    Configure Scanpy and global figure settings for scOmnom.
    """
    global ROOT_FIGDIR
    ROOT_FIGDIR = figdir.resolve()

    if formats is not None:
        set_figure_formats(formats)

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

    figdir.mkdir(parents=True, exist_ok=True)


def save_multi(stem: str, figdir: Path, fig=None) -> None:
    """
    Save the current matplotlib figure (or a provided figure) to multiple formats
    under ext / relative_path.

    Parameters
    ----------
    stem : str
        Base filename without extension.
    figdir : Path
        Directory where figures should be placed (subdirectories created automatically).
    fig : matplotlib.figure.Figure, optional
        If provided, activate and save this figure instead of the current active one.
    """
    import matplotlib.pyplot as plt
    global ROOT_FIGDIR

    if ROOT_FIGDIR is None:
        raise RuntimeError("ROOT_FIGDIR is not set. Call setup_scanpy_figs() first.")

    # If a figure is provided, activate it
    if fig is not None:
        plt.figure(fig.number)

    figdir = Path(figdir)

    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / figdir
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{stem}.{ext}"
        LOGGER.info("Saving figure: %s", outfile)
        plt.savefig(outfile, dpi=300)

    plt.close()


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _clean_axes(ax):
    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_alpha(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


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


def qc_scatter(adata, groupby: str, cfg):
    figdir = Path("QC_plots") / "qc_scatter"

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False,
    )
    save_multi("QC_scatter_mt", figdir)


def hvgs_and_pca_plots(adata, max_pcs_plot: int, cfg):
    figdir = Path("QC_plots") / "overview"

    sc.pl.highly_variable_genes(adata, show=False)
    save_multi("QC_highly_variable_genes", figdir)

    sc.pl.pca_variance_ratio(adata, n_pcs=max_pcs_plot, log=True, show=False)
    save_multi("QC_pca_variance_ratio", figdir)


def umap_by(adata, keys, figdir: Path | None = None, stem: str | None = None):
    """
    Plot UMAP colored by one or more keys.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with existing 'X_umap'.
    keys : str or list[str]
        Columns in adata.obs to color by.
    figdir : Path, optional
        Directory under ROOT_FIGDIR where figures are placed.
        Defaults to scanpy's current figdir.
    stem : str, optional
        Base filename (without extension). If None, uses 'QC_umap_<keys>'.
    """
    if isinstance(keys, str):
        keys = [keys]

    if stem is None:
        name = f"QC_umap_{'_'.join(keys)}"
    else:
        name = stem

    if figdir is None:
        figdir = ROOT_FIGDIR

    sc.pl.umap(adata, color=keys, use_raw=False, show=False)
    save_multi(name, figdir)


# -------------------------------------------------------------------------
# Cell-calling elbow/knee plot
# -------------------------------------------------------------------------
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

    save_multi(figpath_stem, figdir)


# -------------------------------------------------------------------------
# Final cell-counts plot
# -------------------------------------------------------------------------
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
    save_multi(stem="final_cell_counts", figdir=figdir_qc)

    plt.close(fig)


# -------------------------------------------------------------------------
# MT histogram
# -------------------------------------------------------------------------
def plot_mt_histogram(adata, cfg, suffix: str):
    figdir_qc = Path("QC_plots") / "qc_metrics"

    fig, ax = plt.subplots(figsize=(5, 4))
    _clean_axes(ax)

    ax.hist(adata.obs["pct_counts_mt"], bins=50, color="steelblue", alpha=0.85)

    ax.set_xlabel("Percent mitochondrial counts")
    ax.set_ylabel("Number of cells")
    ax.set_title("Distribution of mitochondrial content")

    fig.tight_layout()
    save_multi(f"{suffix}_QC_hist_pct_mt", figdir_qc)


# -------------------------------------------------------------------------
# QC plots: pre + post filter
# -------------------------------------------------------------------------
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

    save_multi(f"{stage}_QC_hist_total_counts", figdir_qc)
    plt.close()


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

    save_multi(f"{stage}_QC_hist_n_genes", figdir_qc)
    plt.close()


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
            rotation=90 if not horizontal else 0,
            show=False,
            stripplot=False,
            ax=ax,
        )

        ax.set_title(f"{metric} ({stage})")
        save_multi(f"{stem}_{stage}", figdir_qc)
        plt.close()


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
    save_multi(f"QC_complexity_{stage}", figdir)
    plt.close()

    # --------------------------------------------------------------
    # Scatter 2: total_counts vs pct_counts_mt
    # --------------------------------------------------------------
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
    )
    save_multi(f"QC_scatter_mt_{stage}", figdir)
    plt.close()

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
    save_multi("cellbender_removed_fraction_hist", figdir, fig)
    plt.close(fig)

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
        save_multi("cellbender_removed_fraction_per_sample", figdir, fig)
        plt.close(fig)

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
    save_multi("cellbender_gene_raw_vs_cb", figdir, fig)
    plt.close(fig)

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
    save_multi("cellbender_gene_removed_fraction", figdir, fig)
    plt.close(fig)

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
        save_multi("cellbender_removed_fraction_vs_library_mt", figdir, fig)
        plt.close(fig)

    LOGGER.info("Generated CellBender effect QC plots.")

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
    fig.tight_layout()
    save_multi("doublet_score_hist", figdir, fig)
    plt.close(fig)

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
        fig.tight_layout()
        save_multi("doublet_inferred_threshold_per_sample", figdir, fig)
        plt.close(fig)

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
    fig.tight_layout()
    save_multi("doublet_fraction_per_sample", figdir, fig)
    plt.close(fig)

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
    fig.tight_layout()
    save_multi("doublet_score_vs_total_counts", figdir, fig)
    plt.close(fig)

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
    fig.tight_layout()
    save_multi("doublet_score_violin_per_sample", figdir, fig)
    plt.close(fig)

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
    fig.tight_layout()
    save_multi("doublet_score_ecdf", figdir, fig)
    plt.close(fig)

    LOGGER.info("Generated SOLO doublet QC plots (per-sample inferred thresholds).")


def umap_plots(
        adata: ad.AnnData,
        *,
        batch_key: str,
        figdir: Path,
        cluster_key: str = "leiden",
) -> None:
    """
    Minimal UMAP plots.
    Generates:
      1) UMAP colored by batch
      2) UMAP colored by clusters (e.g. Leiden)

    Must be called AFTER UMAP computation.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)

    if "X_umap" not in adata.obsm:
        LOGGER.warning("Skipping UMAP plots: X_umap not found.")
        return

    # --------------------------------------------------
    # 1. UMAP colored by batch
    # --------------------------------------------------
    if batch_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=batch_key,
            show=False,
            return_fig=True,
        )
        save_multi("umap_batch", figdir, fig)
        plt.close(fig)
    else:
        LOGGER.warning("Batch key '%s' not found in adata.obs", batch_key)

    # --------------------------------------------------
    # 2. UMAP colored by clusters
    # --------------------------------------------------
    if cluster_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=cluster_key,
            legend_loc="on data",
            show=False,
            return_fig=True,
        )
        save_multi(f"umap_{cluster_key}", figdir, fig)
        plt.close(fig)
    else:
        LOGGER.warning("Cluster key '%s' not found in adata.obs", cluster_key)

    LOGGER.info("Generated UMAP plots (batch + %s)", cluster_key)


# -------------------------------------------------------------------------
# scIB-style results table
# -------------------------------------------------------------------------
def plot_scib_results_table(scaled: pd.DataFrame, figdir: Path) -> None:
    """
    Generate a scIB-style results table with circles for metrics and bars for aggregate scores.
    """
    df = scaled.copy()
    df = df.loc[~df.index.str.contains("Metric", case=False, na=False)]

    all_cols = df.columns.tolist()

    agg_metrics = ["Batch correction", "Bio conservation", "Total"]
    batch_metrics_expected = ["iLISI", "KBET", "Graph connectivity", "PCR comparison"]
    middle_metrics_expected = ["cLISI", "Silhouette batch"]

    agg_metrics = [c for c in agg_metrics if c in all_cols]
    batch_metrics = [c for c in batch_metrics_expected if c in all_cols]
    middle_metrics = [c for c in middle_metrics_expected if c in all_cols]

    exclude_list = agg_metrics + batch_metrics + middle_metrics
    bio_metrics = [c for c in all_cols if c not in exclude_list]

    normal_metrics = bio_metrics + middle_metrics + batch_metrics

    df = df[normal_metrics + agg_metrics]
    vals = df.values.astype(float)
    n_rows, n_cols = vals.shape

    cmap_metrics = mpl.colors.LinearSegmentedColormap.from_list(
        "PuGr", ["#5E3584", "white", "#99CC33", "#4A8F3F"], N=256
    )
    cmap_agg = mpl.cm.get_cmap("YlGnBu")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cell_w, cell_h = 1.2, 0.6
    agg_gap = 0.5

    fig_w = cell_w * (n_cols + 0.5) + agg_gap + 1.0
    fig_h = cell_h * (n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_h = cell_h * 0.7

    for i in range(n_rows):
        for j in range(n_cols):
            metric = df.columns[j]
            x_coord = j + agg_gap if metric in agg_metrics else j
            y_coord = n_rows - i - 0.5
            val = vals[i, j]
            current_cmap = cmap_agg if metric in agg_metrics else cmap_metrics

            if metric in agg_metrics:
                bar_color = current_cmap(norm(val))
                ax.barh(
                    y=y_coord,
                    width=val,
                    left=x_coord,
                    height=bar_h,
                    color=bar_color,
                    edgecolor="none",
                )

                ax.hlines(
                    y=y_coord + bar_h / 2,
                    xmin=x_coord,
                    xmax=x_coord + 1,
                    color="white",
                    linewidth=1,
                )
                ax.hlines(
                    y=y_coord - bar_h / 2,
                    xmin=x_coord,
                    xmax=x_coord + 1,
                    color="white",
                    linewidth=1,
                )
                ax.vlines(
                    x=x_coord,
                    ymin=y_coord - bar_h / 2,
                    ymax=y_coord + bar_h / 2,
                    color="white",
                    linewidth=1,
                )
                ax.vlines(
                    x=x_coord + 1,
                    ymin=y_coord - bar_h / 2,
                    ymax=y_coord + bar_h / 2,
                    color="white",
                    linewidth=1,
                )

                text_color = "white" if val > 0.4 else "black"
                ax.text(
                    x_coord + 0.05,
                    y_coord,
                    f"{val:.2f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    )
            else:
                ax.scatter(
                    x_coord + 0.5,
                    y_coord,
                    s=900 * cell_h,
                    c=[current_cmap(norm(val))],
                    edgecolor="0.8",
                    linewidth=0.8,
                    )
                text_color = "black" if (0.2 < val < 0.8) else "white"
                ax.text(
                    x_coord + 0.5,
                    y_coord,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    )

    ax.grid(False)

    for i in range(n_rows + 1):
        ax.axhline(i, color="0.85", linewidth=0.8, linestyle="dotted")

    bio_metrics_cur = bio_metrics
    middle_metrics_cur = middle_metrics
    batch_metrics_cur = batch_metrics

    bio_batch_divider_x = len(bio_metrics_cur) + len(middle_metrics_cur)
    ax.axvline(bio_batch_divider_x, color="0.3", linewidth=1.5, linestyle="-")

    agg_start = len(normal_metrics) + agg_gap
    ax.axvline(agg_start, color="0.3", linewidth=1.5)

    ax.set_xlim(0, n_cols + agg_gap)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_frame_on(False)
    ax.tick_params(axis="y", length=0)

    if bio_metrics_cur or middle_metrics_cur:
        bio_start = 0
        bio_width = len(bio_metrics_cur) + len(middle_metrics_cur)
        ax.text(
            bio_start + (bio_width / 2.0),
            -0.5,
            "Bio conservation",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            )

    if batch_metrics_cur:
        batch_start = len(bio_metrics_cur) + len(middle_metrics_cur)
        batch_width = len(batch_metrics_cur)
        ax.text(
            batch_start + (batch_width / 2.0),
            -0.5,
            "Batch correction",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            )

    if agg_metrics:
        agg_center = agg_start + (len(agg_metrics) / 2.0)
        ax.text(
            agg_center,
            -0.5,
            "Aggregate score",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for idx, label in enumerate(normal_metrics):
        ax.text(
            idx + 0.5,
            0.0,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
            )

    for idx, label in enumerate(agg_metrics):
        ax.text(
            agg_start + idx + 0.5,
            0.0,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
            )

    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(df.index.tolist(), fontsize=10)

    plt.tight_layout()
    save_multi("scIB_results_table", figdir)
    plt.close(fig)


# -------------------------------------------------------------------------
# CLUSTERING RESOLUTION / STABILITY PLOTS
# -------------------------------------------------------------------------
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
    save_multi("clustering_resolution_sweep", figdir)


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
    save_multi("clustering_stability_ari", figdir)


def plot_cluster_umaps(
        adata,
        label_key: str,
        batch_key: str,
        figdir: Path,
) -> None:
    """UMAPs colored by cluster and batch for the clustering stage."""
    sc.pl.umap(adata, color=[label_key], show=False)
    save_multi(f"cluster_umap_{label_key}", figdir)

    if batch_key in adata.obs:
        sc.pl.umap(adata, color=[batch_key], show=False)
        save_multi(f"cluster_umap_{batch_key}", figdir)

        sc.pl.umap(adata, color=[batch_key, label_key], legend_loc="on data", show=False)
        save_multi(f"cluster_umap_{batch_key}_and_{label_key}", figdir)


# ----------------------------------------------------------------------
# Cluster tree
# ----------------------------------------------------------------------
import networkx as nx

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

    save_multi(stem, figdir, fig)


# ----------------------------------------------------------------------
# Stability curves (silhouette, stability, composite, tiny penalty)
# ----------------------------------------------------------------------
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
    save_multi(stem, _ensure_path(figdir), fig)


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
    save_multi(stem, _ensure_path(figdir), fig)


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
    save_multi(stem, _ensure_path(figdir), fig)


# ----------------------------------------------------------------------
# Plateau highlights
# ----------------------------------------------------------------------
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
    save_multi(stem, _ensure_path(figdir), fig)


# -------------------------------------------------------------------------
# Cluster-level statistics
# -------------------------------------------------------------------------

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
    sizes = counts.values.astype(int)
    total = sizes.sum()

    # colors from Scanpy palette
    palette = adata.uns.get(f"{label_key}_colors", None)
    if palette is None:
        LOGGER.warning("No cluster palette found for '%s'; using default.", label_key)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(clusters))]
    else:
        colors = palette[:len(clusters)]

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 0.6), 4))
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
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Cluster sizes")

    fig.tight_layout()
    save_multi(stem, figdir)
    plt.close(fig)


def plot_cluster_qc_summary(
        adata,
        label_key: str,
        figdir: Path,
        stem: str = "cluster_qc_summary",
):
    """
    Mean QC metrics per cluster:
      - n_genes_by_counts
      - total_counts
      - pct_counts_mt
    """
    if label_key not in adata.obs:
        LOGGER.warning("plot_cluster_qc_summary: '%s' not in obs.", label_key)
        return

    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing = [m for m in metrics if m not in adata.obs]
    if missing:
        LOGGER.warning("Missing QC fields: %s", missing)
        return

    df = adata.obs[[label_key] + metrics].groupby(label_key).mean()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axs, metrics):
        _clean_axes(ax)
        df[m].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("Cluster")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    save_multi(stem, figdir)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _clean_axes(ax)

    df.boxplot(column="silhouette", by="cluster", ax=ax)
    plt.suptitle("")  # remove pandas default
    ax.set_title("Silhouette distribution per cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    save_multi(stem, figdir)
    plt.close(fig)



def plot_cluster_batch_composition(
        adata,
        label_key: str,
        batch_key: str,
        figdir: Path,
        stem: str = "cluster_batch_composition",
):
    """
    Stacked barplot showing fraction of each batch within each cluster.
    """
    if label_key not in adata.obs or batch_key not in adata.obs:
        LOGGER.warning("plot_cluster_batch_composition: required columns missing.")
        return

    df = (
        adata.obs[[label_key, batch_key]]
        .groupby([label_key, batch_key])
        .size()
        .unstack(fill_value=0)
    )
    frac = df.div(df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 4))
    _clean_axes(ax)

    frac.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_ylabel("Fraction")
    ax.set_title("Batch composition per cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    save_multi(stem, figdir)
    plt.close(fig)


def plot_ssgsea_cluster_topn_heatmap(
        adata: anndata.AnnData,
        cluster_key: str = "cluster_label",
        figdir: Path | None = None,
        n: int = 5,
        z_score: bool = False,
        cmap: str = "viridis",
):
    if figdir is None:
        raise ValueError("figdir must be provided.")
    if "ssgsea_cluster_means" not in adata.uns:
        LOGGER.warning("No ssGSEA cluster means found; skipping.")
        return

    df = adata.uns["ssgsea_cluster_means"].apply(pd.to_numeric, errors="coerce").fillna(0)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.cm as cm
    import numpy as np

    cmap_force = cm.get_cmap("viridis")  # TRUE viridis colormap

    prefixes = sorted({c.split("::")[0] for c in df.columns})

    for prefix in prefixes:

        cols = [c for c in df.columns if c.startswith(prefix + "::")]
        if not cols:
            continue

        sub = df[cols]

        if z_score:
            sub = (sub - sub.mean(0)) / sub.std(0).replace(0, np.nan)

        # ---- select top N ----
        top_terms = []
        for cl in sub.index:
            top_terms.extend(sub.loc[cl].nlargest(n).index.tolist())

        selected = sorted(set(top_terms))
        sub = sub[selected]

        # ---- shorten labels ----
        short_labels = [c.split("::", 1)[1].replace("_", " ") for c in selected]
        sub.columns = short_labels

        # ---- figure ----
        fig_w = 2.5 + 0.55 * len(selected)
        fig_h = 2.5 + 0.40 * len(sub)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # ---- ACTUAL HEATMAP ----
        sns.heatmap(
            sub,
            ax=ax,
            cmap=cmap_force,
            annot=False,
            cbar=True,
            linewidths=0,
            linecolor=None,
        )

        # ---- REMOVE ALL SPINES ----
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ---- REMOVE ALL GRIDLINES ----
        ax.grid(False)
        ax.set_axisbelow(False)

        # ---- ticks ----
        ax.tick_params(axis="x", rotation=60, labelsize=8, length=4, color="black")
        ax.tick_params(axis="y", rotation=0, labelsize=8, length=4, color="black")
        ax.minorticks_off()

        # ---- labels ----
        ax.set_xlabel("Pathway", fontsize=11)
        ax.set_ylabel(cluster_key, fontsize=11)

        title = f"Top {n} ssGSEA pathways per cluster ({prefix})"
        if z_score:
            title += " — Z-score"
        ax.set_title(title, fontsize=14, pad=12)

        # ---- MANUAL MARGINS ----
        fig.subplots_adjust(left=0.30, bottom=0.32, right=0.98, top=0.88)

        stem = f"ssgsea_top{n}_{prefix}"
        if z_score:
            stem += "_z"

        save_multi(stem, figdir, fig)
        plt.close(fig)

