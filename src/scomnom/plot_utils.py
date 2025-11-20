from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Internal helper aesthetic
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
