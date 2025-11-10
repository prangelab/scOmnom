from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc


def setup_scanpy_figs(figdir: Path) -> None:
    sc.set_figure_params(dpi=300, facecolor="white")
    sc.settings.figdir = str(figdir)
    sc.settings.autoshow = False
    sc.settings.autosave = False
    sc.settings.file_format_figs = "png"
    figdir.mkdir(parents=True, exist_ok=True)


def qc_scatter(adata, groupby: str):
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",
                  save="_QC_scatter_mt.png")


def hvgs_and_pca_plots(adata, max_pcs_plot: int):
    sc.pl.highly_variable_genes(adata, save="_QC_highly_variable_genes.png")
    sc.pl.pca_variance_ratio(adata, n_pcs=max_pcs_plot, log=True, save="_QC_pca_variance_ratio.png")


def umap_by(adata, keys):
    if isinstance(keys, str):
        keys = [keys]
    sc.pl.umap(adata, color=keys, use_raw=False, save=f"_QC_umap_{'_'.join(keys)}.png")


def barplot_before_after(df_counts: pd.DataFrame, figpath: Path, min_cells_per_sample: int):
    x = np.arange(len(df_counts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(df_counts) * 0.6), 4))
    colors_before = ['lightgray'] * len(df_counts)
    colors_after = ['red' if c < min_cells_per_sample else 'steelblue' for c in df_counts['after']]
    bars_before = ax.bar(x - width / 2, df_counts['before'], width, label='Before filtering', alpha=0.7,
                         color=colors_before)
    bars_after = ax.bar(x + width / 2, df_counts['after'], width, label='After filtering', alpha=0.7,
                        color=colors_after)
    ax.set_xticks(x)
    ax.set_xticklabels(df_counts['sample'], rotation=45, ha='right')
    ax.set_ylabel('Number of cells')
    ax.set_title('Cell counts per sample (before vs after filtering)')
    ax.legend()
    plt.tight_layout()
    for i, bar in enumerate(bars_after):
        height = bar.get_height()
        pct = df_counts.loc[i, 'retained_pct']
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()


def plot_cellbender_comparison(raw_counts: dict, cb_counts: dict, figdir: Path) -> None:
    """
    Generate per-sample and aggregate plots comparing total read counts
    before vs after CellBender. Each barplot shows absolute read counts and
    percent retained above the filtered bar.

    Parameters
    ----------
    raw_counts : dict
        {sample: total_reads_before_cellbender}
    cb_counts : dict
        {sample: total_reads_after_cellbender}
    figdir : Path
        Base figure directory; will create QC_plots/cellbender/
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    figdir_cb = Path(figdir) / "QC_plots" / "cellbender"
    os.makedirs(figdir_cb, exist_ok=True)

    # build dataframe
    samples = sorted(set(raw_counts) | set(cb_counts))
    df = pd.DataFrame({
        "sample": samples,
        "raw_reads": [raw_counts.get(s, 0) for s in samples],
        "cb_reads": [cb_counts.get(s, 0) for s in samples],
    })
    df["pct_retained"] = np.where(
        df["raw_reads"] > 0,
        100 * df["cb_reads"] / df["raw_reads"],
        0,
    )
    df.to_csv(figdir_cb / "QC_reads_per_sample_cellbender.tsv", sep="\t", index=False)

    # helper for single barplot
    def _plot_single(row, outpath):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(1)
        width = 0.35
        bars_raw = ax.bar(x - width / 2, [row["raw_reads"]], width, color="lightgray", label="raw")
        bars_cb = ax.bar(x + width / 2, [row["cb_reads"]], width, color="steelblue", label="cellbender")
        pct = row["pct_retained"]
        if row["cb_reads"] > 0:
            ax.text(x + width / 2, row["cb_reads"], f"{pct:.1f}%", ha="center", va="bottom")
        ax.set_title(row["sample"])
        ax.set_ylabel("Total reads")
        ax.set_xticks([])
        ax.legend()
        plt.tight_layout()
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

    # per-sample plots
    for _, row in df.iterrows():
        _plot_single(row, figdir_cb / f"{row['sample']}_QC_reads_before_after_cellbender.png")

    # aggregate overview
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["raw_reads"], width, color="lightgray", label="raw")
    ax.bar(x + width / 2, df["cb_reads"], width, color="steelblue", label="cellbender")
    for i, pct in enumerate(df["pct_retained"]):
        if df.loc[i, "cb_reads"] > 0:
            ax.text(i + width / 2, df.loc[i, "cb_reads"], f"{pct:.1f}%", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Total reads")
    ax.legend()
    ax.set_title("CellBender: total reads per sample")
    plt.tight_layout()
    fig.savefig(figdir_cb / "QC_reads_before_after_cellbender_AGGREGATE.png", dpi=200)
    plt.close(fig)


def plot_final_cell_counts(adata, cfg) -> None:
    """
    Plot number of cells per sample after all filtering.
    Sorted descending, with total cell count displayed.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os

    figdir_qc = Path(cfg.figdir) / "QC_plots"
    os.makedirs(figdir_qc, exist_ok=True)

    counts = adata.obs[cfg.batch_key].value_counts().sort_values(ascending=False)
    df = pd.DataFrame({"sample": counts.index, "n_cells": counts.values})
    total_cells = int(df["n_cells"].sum())

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    bars = ax.bar(df["sample"], df["n_cells"], color="steelblue", alpha=0.8)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title("Final number of cells per sample (post-filtering)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate total
    ax.text(
        0.98, 0.95,
        f"Total: {total_cells:,} cells",
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7)
    )

    plt.tight_layout()
    outpath = figdir_qc / "QC_cells_final_per_sample.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
