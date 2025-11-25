from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scanpy as sc
import matplotlib as mpl
import logging

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
ROOT_FIGDIR = None


# -------------------------------------------------------------------------
# Setup + saving
# -------------------------------------------------------------------------
def set_figure_formats(formats):
    global FIGURE_FORMATS
    FIGURE_FORMATS = list(formats)


def setup_scanpy_figs(figdir: Path, formats=None) -> None:
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


def save_multi(stem: str, figdir: Path, fig=None):
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

    # If a figure is provided, activate it
    if fig is not None:
        plt.figure(fig.number)

    figdir = figdir.resolve()
    rel = figdir.relative_to(ROOT_FIGDIR)

    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / rel
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{stem}.{ext}"
        LOGGER.info(f"Saving figure: {outfile}")
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
    return ax


# -------------------------------------------------------------------------
# SCANPY WRAPPERS
# -------------------------------------------------------------------------
def qc_scatter(adata, groupby: str):
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color="pct_counts_mt",
        show=False
    )
    save_multi("QC_scatter_mt", Path(sc.settings.figdir))


def hvgs_and_pca_plots(adata, max_pcs_plot: int):
    sc.pl.highly_variable_genes(adata, show=False)
    save_multi("QC_highly_variable_genes", Path(sc.settings.figdir))

    sc.pl.pca_variance_ratio(adata, n_pcs=max_pcs_plot, log=True, show=False)
    save_multi("QC_pca_variance_ratio", Path(sc.settings.figdir))


def umap_by(adata, keys):
    if isinstance(keys, str):
        keys = [keys]
    name = f"QC_umap_{'_'.join(keys)}"
    sc.pl.umap(adata, color=keys, use_raw=False, show=False)
    save_multi(name, Path(sc.settings.figdir))


# -------------------------------------------------------------------------
# Cell-calling elbow/knee plot
# -------------------------------------------------------------------------
def plot_elbow_knee(
    adata,
    figpath_stem: str,
    figdir: Path,
    title: str = "Barcode Rank UMI Knee Plot"
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

    df = pd.DataFrame({
        "sample": samples,
        ref_label: [ref_counts.get(s, 0) for s in samples],
        other_label: [other_counts.get(s, 0) for s in samples],
    })

    x = np.arange(len(samples))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(6, len(samples) * 0.7), 5))
    _clean_axes(ax)

    # bars
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
def plot_final_cell_counts(adata: ad.AnnData, cfg: LoadAndQCConfig) -> None:
    """Plot final per-sample cell counts with a mean line and summary box."""
    import matplotlib.pyplot as plt

    batch_key = cfg.batch_key
    if batch_key not in adata.obs:
        LOGGER.warning("batch_key '%s' not found in adata.obs; skipping plot.", batch_key)
        return

    counts = adata.obs[batch_key].value_counts().sort_index()
    mean_cells = counts.mean()
    total_cells = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")

    # Theme-consistent mean line — darker steelblue
    ax.axhline(mean_cells, linestyle="--", color="#1f4e79", linewidth=1.0)

    # Clean axes (removes grid, applies spine styling)
    _clean_axes(ax)

    ax.set_ylabel("Cell count")
    ax.set_title("Final cell counts per sample")

    # Summary stats box
    summary_text = (
        f"Total cells: {total_cells:,}\n"
        f"Mean per sample: {mean_cells:,.0f}"
    )
    ax.text(
        0.02, 0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
    )

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    # Correct save call
    figdir_qc = cfg.figdir / "QC_plots"
    save_multi(stem="final_cell_counts", figdir=figdir_qc)

    plt.close(fig)


# -------------------------------------------------------------------------
# MT histogram
# -------------------------------------------------------------------------
def plot_mt_histogram(adata, cfg, suffix):
    figdir_qc = cfg.figdir / "QC_plots"

    fig, ax = plt.subplots(figsize=(5, 4))
    _clean_axes(ax)

    ax.hist(
        adata.obs["pct_counts_mt"],
        bins=50,
        color="steelblue",
        alpha=0.85
    )

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
            show=False
        )
        save_multi("QC_doublet_score_violin_prefilter", figdir_qc)

    plot_mt_histogram(adata, cfg, "prefilter")


def run_qc_plots_postfilter(adata, cfg):
    from scanpy import settings as sc_settings
    old_figdir = sc_settings.figdir
    figdir_qc = cfg.figdir / "QC_plots"
    sc_settings.figdir = figdir_qc

    try:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            groupby=cfg.batch_key,
            show=False
        )

        fig = plt.gcf()
        axs = fig.get_axes()

        # Compact margins but keep readable
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.25)

        # Force all violin subplots to equal width
        first_width = axs[0].get_position().width
        for ax in axs:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, first_width, pos.height])

        save_multi("QC_violin_mt_counts_postfilter", figdir_qc)
        plt.close(fig)

        plot_mt_histogram(adata, cfg, "postfilter")

        # complexity
        sc.pl.scatter(
            adata,
            x="total_counts",
            y="n_genes_by_counts",
            color="pct_counts_mt",
            show=False
        )
        save_multi("QC_complexity_prefilter", figdir_qc)

        # PCA
        sc.pl.pca_variance_ratio(
            adata,
            log=True,
            n_pcs=cfg.max_pcs_plot,
            show=False
        )
        save_multi("QC_pca_variance_ratio", figdir_qc)

        # UMAPs
        sc.pl.umap(adata, color=[cfg.batch_key], show=False)
        save_multi("QC_umap_sample", figdir_qc)

        sc.pl.umap(adata, color=["leiden"], show=False)
        save_multi("QC_umap_leiden", figdir_qc)

        sc.pl.umap(
            adata,
            color=[cfg.batch_key, "leiden"],
            legend_loc="on data",
            show=False
        )
        save_multi("QC_umap_per_sample_and_leiden", figdir_qc)

        sc.pl.scatter(
            adata,
            x="total_counts",
            y="pct_counts_mt",
            show=False
        )
        save_multi("QC_scatter_mt", figdir_qc)

    finally:
        sc_settings.figdir = old_figdir


def barplot_before_after(df_counts: pd.DataFrame, figpath: Path, min_cells_per_sample: int):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(df_counts))
    width = 0.40

    fig, ax = plt.subplots(figsize=(max(6, len(df_counts) * 0.65), 4))

    # Clean axes
    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_alpha(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    colors_before = ["#cccccc"] * len(df_counts)
    colors_after = [
        "red" if c < min_cells_per_sample else "steelblue"
        for c in df_counts["after"]
    ]

    bars_before = ax.bar(
        x - width / 2,
        df_counts["before"],
        width,
        label="Before filtering",
        alpha=0.8,
        color=colors_before
    )
    bars_after = ax.bar(
        x + width / 2,
        df_counts["after"],
        width,
        label="After filtering",
        alpha=0.85,
        color=colors_after
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_counts["sample"], rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell counts before vs after filtering")
    ax.legend(frameon=False)

    # Add % retained on top of after bars
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
                rotation=60
            )

    fig.tight_layout()
    save_multi(figpath.stem, figpath.parent)


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
import numpy as np


def plot_scib_results_table(scaled: pd.DataFrame, figdir: Path) -> None:
    """Generate a scIB-style results table with circles for metrics and bars for aggregate scores.

    This final version includes:
    - Circles for individual metrics, bars for aggregate scores.
    - Dotted row separation lines at row boundaries.
    - Thick vertical dividers between metric groups and aggregate scores (except the right-most boundary).
    - Horizontal column labels at the top in a hierarchical grouping.
    - Complete removal of all internal grid lines and axes spines.
    """
    df = scaled.copy()
    df = df.loc[~df.index.str.contains("Metric", case=False, na=False)]

    # --- Data and Column Grouping (Robust against KeyErrors) ---
    all_cols = df.columns.tolist()

    # 1. Define Core Groupings
    agg_metrics = ["Batch correction", "Bio conservation", "Total"]
    batch_metrics_expected = ["iLISI", "KBET", "Graph connectivity", "PCR comparison"]
    middle_metrics_expected = ["cLISI", "Silhouette batch"]

    # 2. Filter Lists to Include Only Existing Columns
    agg_metrics = [c for c in agg_metrics if c in all_cols]
    batch_metrics = [c for c in batch_metrics_expected if c in all_cols]
    middle_metrics = [c for c in middle_metrics_expected if c in all_cols]

    # 3. Identify Bio Conservation Metrics (everything else that's not agg, batch, or middle)
    exclude_list = agg_metrics + batch_metrics + middle_metrics
    bio_metrics = [c for c in all_cols if c not in exclude_list]

    # 4. Construct Final Ordered List of Normal Metrics
    normal_metrics = bio_metrics + middle_metrics + batch_metrics

    df = df[normal_metrics + agg_metrics]
    vals = df.values.astype(float)
    n_rows, n_cols = vals.shape

    # --- Colormaps ---
    cmap_metrics = mpl.colors.LinearSegmentedColormap.from_list(
        "PuGr",
        ["#5E3584", "white", "#99CC33", "#4A8F3F"], N=256
    )
    cmap_agg = mpl.cm.get_cmap("YlGnBu")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # --- Figure Setup ---
    cell_w, cell_h = 1.2, 0.6
    agg_gap = 0.5

    fig_w = cell_w * (n_cols + 0.5) + agg_gap + 1.0
    fig_h = cell_h * (n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- Draw Markers (Circles and Bars) and Text ---
    bar_h = cell_h * 0.7

    for i in range(n_rows):
        for j in range(n_cols):
            metric = df.columns[j]
            x_coord = j + agg_gap if metric in agg_metrics else j
            y_coord = n_rows - i - 0.5
            val = vals[i, j]

            current_cmap = cmap_agg if metric in agg_metrics else cmap_metrics

            if metric in agg_metrics:
                # Draw BARS for aggregate scores
                bar_color = current_cmap(norm(val))

                ax.barh(
                    y=y_coord,
                    width=val,
                    left=x_coord,
                    height=bar_h,
                    color=bar_color,
                    edgecolor="none",
                )

                # Draw white outline/box to suppress grid lines and provide visual cell separation
                ax.hlines(y=y_coord + bar_h / 2, xmin=x_coord, xmax=x_coord + 1, color='white', linewidth=1)
                ax.hlines(y=y_coord - bar_h / 2, xmin=x_coord, xmax=x_coord + 1, color='white', linewidth=1)
                ax.vlines(x=x_coord, ymin=y_coord - bar_h / 2, ymax=y_coord + bar_h / 2, color='white', linewidth=1)
                ax.vlines(x=x_coord + 1, ymin=y_coord - bar_h / 2, ymax=y_coord + bar_h / 2, color='white', linewidth=1)

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
                # Draw CIRCLES for individual metrics
                ax.scatter(
                    x_coord + 0.5,
                    y_coord,
                    s=900 * cell_h,
                    c=[current_cmap(norm(val))],
                    edgecolor="0.8",
                    linewidth=0.8,
                )

                text_color = "black" if (val > 0.2 and val < 0.8) else "white"

                ax.text(
                    x_coord + 0.5,
                    y_coord,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    # --- Grid Lines (Final Positioning and Dividers) ---

    # 0. Disable all default grid lines
    ax.grid(False)

    # 1. Horizontal dotted lines at the ROW BOUNDARIES
    for i in range(n_rows + 1):
        ax.axhline(i, color="0.85", linewidth=0.8, linestyle="dotted")

    # 2. Vertical Divider between Bio Conservation and Batch Correction
    bio_batch_divider_x = len(bio_metrics) + len(middle_metrics)
    ax.axvline(bio_batch_divider_x, color="0.3", linewidth=1.5, linestyle="-")

    # 3. Vertical Dividers for Aggregate Scores
    agg_start = len(normal_metrics) + agg_gap
    ax.axvline(agg_start, color="0.3", linewidth=1.5)
    # The line below, which was the final right-most line, is removed.
    # ax.axvline(agg_start + len(agg_metrics), color="0.3", linewidth=1.5)

    # --- Axis and Labels Setup ---
    ax.set_xlim(0, n_cols + agg_gap)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_frame_on(False)

    # Hiding Y-Axis Ticks
    ax.tick_params(axis='y', length=0)

    # --- Column Headers (Hierarchical - Two Layers - TOP and HORIZONTAL) ---

    # 1. Top Layer Headers (Groups)
    if bio_metrics or middle_metrics:
        bio_start = 0
        bio_width = len(bio_metrics) + len(middle_metrics)
        ax.text(
            bio_start + (bio_width / 2.0),
            -0.5,
            "Bio conservation",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    if batch_metrics:
        batch_start = len(bio_metrics) + len(middle_metrics)
        batch_width = len(batch_metrics)
        ax.text(
            batch_start + (batch_width / 2.0),
            -0.5,
            "Batch correction",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
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
            fontweight="bold"
        )

    # 2. Bottom Layer Headers (Individual Metrics)
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

    # --- Y-axis (Method) Labels ---
    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(df.index.tolist(), fontsize=10)

    plt.tight_layout()
    save_multi("scIB_results_table", figdir)
    plt.close(fig)


def plot_clustering_resolution_sweep(
    resolutions: np.ndarray,
    silhouette_scores: List[float],
    n_clusters: List[int],
    penalized_scores: List[float],
    figdir: Path,
) -> None:
    """Plot silhouette, #clusters, and penalized score across resolutions."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Silhouette
    ax = axs[0]
    _clean_axes(ax)
    ax.plot(resolutions, silhouette_scores, marker="o")
    ax.set_title("Silhouette score")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")

    # Number of clusters
    ax = axs[1]
    _clean_axes(ax)
    ax.plot(resolutions, n_clusters, marker="o")
    ax.set_title("Number of clusters")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Clusters")

    # Penalized silhouette
    ax = axs[2]
    _clean_axes(ax)
    ax.plot(resolutions, penalized_scores, marker="o")
    ax.set_title("Penalized score\n(silhouette - α·N)")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Score")

    fig.tight_layout()
    save_multi("clustering_resolution_sweep", figdir)


def plot_clustering_ari_heatmap(
    ari_matrix: pd.DataFrame,
    figdir: Path,
) -> None:
    """Heatmap of ARI between clusterings at different resolutions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    _clean_axes(ax)
    sns.heatmap(
        ari_matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
    )
    ax.set_title("ARI between resolutions")
    plt.tight_layout()
    save_multi("clustering_ari_between_resolutions", figdir)


def plot_clustering_stability_ari(
    stability_aris: List[float],
    figdir: Path,
) -> None:
    """Line plot of ARI vs repetition for subsampling stability."""
    import matplotlib.pyplot as plt

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
    # Leiden / cluster
    sc.pl.umap(adata, color=[label_key], show=False)
    save_multi(f"cluster_umap_{label_key}", figdir)

    # Batch / sample
    if batch_key in adata.obs:
        sc.pl.umap(adata, color=[batch_key], show=False)
        save_multi(f"cluster_umap_{batch_key}", figdir)

        sc.pl.umap(adata, color=[batch_key, label_key], legend_loc="on data", show=False)
        save_multi(f"cluster_umap_{batch_key}_and_{label_key}", figdir)
