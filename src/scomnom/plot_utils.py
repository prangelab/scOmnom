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


def barplot_full_pipeline(df_counts: pd.DataFrame, figpath: Path):
    x = np.arange(len(df_counts))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(df_counts) * 0.6), 4))
    colors = ['lightgray', 'orange', 'steelblue']
    bars_raw = ax.bar(x - width, df_counts['raw_10x'], width, label='Raw 10x', color=colors[0], alpha=0.7)
    bars_cb = ax.bar(x, df_counts['after_cellbender'], width, label='After CellBender', color=colors[1], alpha=0.7)
    bars_final = ax.bar(x + width, df_counts['final_filtered'], width, label='Final Filtered',
                        color=['red' if c < 50 else colors[2] for c in df_counts['final_filtered']], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_counts['sample'], rotation=45, ha='right')
    ax.set_ylabel('Number of cells')
    ax.set_title('Cell counts per sample across pipeline')
    ax.legend()
    plt.tight_layout()
    for i in range(len(df_counts)):
        h_cb = bars_cb[i].get_height(); pct_cb = df_counts.loc[i, 'pct_retained_cb']
    if h_cb > 0:
        ax.text(bars_cb[i].get_x() + bars_cb[i].get_width()/2, h_cb, f"{pct_cb:.1f}%", ha='center', fontsize=8)
    h_f = bars_final[i].get_height(); pct_f = df_counts.loc[i, 'pct_retained_final']
    if h_f > 0:
        ax.text(bars_final[i].get_x() + bars_final[i].get_width()/2, h_f, f"{pct_f:.1f}%", ha='center', fontsize=8)
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()