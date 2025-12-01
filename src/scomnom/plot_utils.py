from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Mapping, Iterable, List, Any

import logging
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
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

    sc.settings.figdir = str(figdir)
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
    under ROOT_FIGDIR / ext / relative_path.

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

    figdir = figdir.resolve()
    rel = figdir.relative_to(ROOT_FIGDIR)

    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / rel
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
def qc_scatter(adata, groupby: str):
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False,
    )
    save_multi("QC_scatter_mt", Path(sc.settings.figdir))


def hvgs_and_pca_plots(adata, max_pcs_plot: int):
    sc.pl.highly_variable_genes(adata, show=False)
    save_multi("QC_highly_variable_genes", Path(sc.settings.figdir))

    sc.pl.pca_variance_ratio(adata, n_pcs=max_pcs_plot, log=True, show=False)
    save_multi("QC_pca_variance_ratio", Path(sc.settings.figdir))


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
        figdir = Path(sc.settings.figdir)

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
# Read comparisons
# -------------------------------------------------------------------------
def plot_read_comparison(
    ref_counts: dict,
    other_counts: dict,
    ref_label: str,
    other_label: str,
    figdir: Path,
    stem: str,
):
    samples = sorted(set(ref_counts) | set(other_counts))

    df = pd.DataFrame(
        {
            "sample": samples,
            ref_label: [ref_counts.get(s, 0) for s in samples],
            other_label: [other_counts.get(s, 0) for s in samples],
        }
    )

    x = np.arange(len(samples))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(6, len(samples) * 0.7), 5))
    _clean_axes(ax)

    ax.bar(
        x - width / 2,
        df[ref_label],
        width,
        label=ref_label,
        alpha=0.85,
        color="#999999",
    )

    other_bar = ax.bar(
        x + width / 2,
        df[other_label],
        width,
        label=other_label,
        alpha=0.85,
        color="steelblue",
    )

    # annotate % of ref_counts relative to other_counts on other_counts bars
    for i, rect in enumerate(other_bar):
        ref = df.loc[i, ref_label]
        other = df.loc[i, other_label]
        if other > 0:
            pct = 100 * other / ref
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("Reads")
    ax.set_title(f"{other_label} vs {ref_label}")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_multi(stem, figdir)


# -------------------------------------------------------------------------
# Final cell-counts plot
# -------------------------------------------------------------------------
def plot_final_cell_counts(adata, cfg) -> None:
    """Plot final per-sample cell counts with a mean line and summary box."""
    batch_key = cfg.batch_key
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

    figdir_qc = cfg.figdir / "QC_plots"
    save_multi(stem="final_cell_counts", figdir=figdir_qc)

    plt.close(fig)


# -------------------------------------------------------------------------
# MT histogram
# -------------------------------------------------------------------------
def plot_mt_histogram(adata, cfg, suffix: str):
    figdir_qc = cfg.figdir / "QC_plots"

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
def run_qc_plots_pre_filter(adata, cfg):
    if not cfg.make_figures:
        return

    figdir_qc = cfg.figdir / "QC_plots"

    if "doublet_score" in adata.obs:
        sc.pl.violin(
            adata,
            ["doublet_score"],
            groupby=cfg.batch_key,
            show=False,
        )
        save_multi("QC_doublet_score_violin_prefilter", figdir_qc)

    plot_mt_histogram(adata, cfg, "prefilter")


def run_qc_plots_postfilter(adata, cfg):
    from scanpy import settings as sc_settings

    if not cfg.make_figures:
        return

    old_figdir = sc_settings.figdir
    figdir_qc = cfg.figdir / "QC_plots"
    sc_settings.figdir = figdir_qc

    try:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            groupby=cfg.batch_key,
            show=False,
        )

        fig = plt.gcf()
        axs = fig.get_axes()

        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.25)

        first_width = axs[0].get_position().width
        for ax in axs:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, first_width, pos.height])

        save_multi("QC_violin_mt_counts_postfilter", figdir_qc)
        plt.close(fig)

        plot_mt_histogram(adata, cfg, "postfilter")

        sc.pl.scatter(
            adata,
            x="total_counts",
            y="n_genes_by_counts",
            color="pct_counts_mt",
            show=False,
        )
        save_multi("QC_complexity_prefilter", figdir_qc)

        sc.pl.pca_variance_ratio(
            adata,
            log=True,
            n_pcs=cfg.max_pcs_plot,
            show=False,
        )
        save_multi("QC_pca_variance_ratio", figdir_qc)

        sc.pl.umap(adata, color=[cfg.batch_key], show=False)
        save_multi("QC_umap_sample", figdir_qc)

        sc.pl.umap(adata, color=["leiden"], show=False)
        save_multi("QC_umap_leiden", figdir_qc)

        sc.pl.umap(
            adata,
            color=[cfg.batch_key, "leiden"],
            legend_loc="on data",
            show=False,
        )
        save_multi("QC_umap_per_sample_and_leiden", figdir_qc)

        sc.pl.scatter(
            adata,
            x="total_counts",
            y="pct_counts_mt",
            show=False,
        )
        save_multi("QC_scatter_mt", figdir_qc)

    finally:
        sc_settings.figdir = old_figdir


def barplot_before_after(df_counts: pd.DataFrame, figpath: Path, min_cells_per_sample: int):
    x = np.arange(len(df_counts))
    width = 0.40

    fig, ax = plt.subplots(figsize=(max(6, len(df_counts) * 0.65), 4))

    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_alpha(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    colors_before = ["#cccccc"] * len(df_counts)
    colors_after = [
        "red" if c < min_cells_per_sample else "steelblue" for c in df_counts["after"]
    ]

    bars_before = ax.bar(
        x - width / 2,
        df_counts["before"],
        width,
        label="Before filtering",
        alpha=0.8,
        color=colors_before,
    )
    bars_after = ax.bar(
        x + width / 2,
        df_counts["after"],
        width,
        label="After filtering",
        alpha=0.85,
        color=colors_after,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_counts["sample"], rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell counts before vs after filtering")
    ax.legend(frameon=False)

    for i, bar in enumerate(bars_after):
        height = bar.get_height()
        pct = df_counts.loc[i, "retained_pct"]
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=60,
            )

    fig.tight_layout()
    save_multi(figpath.stem, figpath.parent)


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
    if palette is None:
        cmap = plt.get_cmap("tab20")
        color_lookup = lambda cid: cmap(cid % cmap.N)
    else:
        palette = list(palette)
        color_lookup = lambda cid: palette[cid % len(palette)]

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
    figure_formats: Sequence[str] = ("png", "pdf"),
    stem: str = "cluster_selection_stability",
) -> None:
    res_sorted = _sorted_resolutions(resolutions)
    sil = _extract_series(res_sorted, silhouette)
    stab = _extract_series(res_sorted, stability)
    comp = _extract_series(res_sorted, composite)
    tiny = _extract_series(res_sorted, tiny_cluster_penalty)

    fig, ax = plt.subplots(figsize=(8, 5))

    for xmin, xmax in _plateau_spans(plateaus or []):
        ax.axvspan(xmin, xmax, color="0.9", alpha=0.5)

    ax.plot(res_sorted, sil, label="Centroid silhouette", color="tab:blue")
    ax.plot(res_sorted, stab, label="Stability (smoothed ARI)", color="tab:green")
    ax.plot(res_sorted, comp, label="Composite score", color="tab:red")
    ax.plot(res_sorted, tiny, label="Tiny-cluster penalty", color="tab:orange")

    ax.axvline(float(best_resolution), color="k", linestyle="--")

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")
    ax.set_title("Cluster selection metrics vs resolution")
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

