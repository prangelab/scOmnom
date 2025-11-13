from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib as mpl


# Global params
mpl.rcParams["xtick.alignment"] = "right"
mpl.rcParams["figure.subplot.bottom"] = 0.25
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.subplot.right"] = 0.9  # gives space for legends

def setup_scanpy_figs(figdir: Path) -> None:
    import matplotlib as mpl
    import scanpy as sc
    mpl.rcParams.update({
        "figure.constrained_layout.use": True,
        "figure.autolayout": True,
        "figure.subplot.right": 0.88,   # leave margin for legends
        "legend.loc": "best",
        "legend.frameon": False,
    })
    sc.set_figure_params(dpi=300, facecolor="white", figsize=(6, 5))
    sc.settings.figdir = str(figdir)
    sc.settings.autoshow = False
    sc.settings.autosave = False
    sc.settings.file_format_figs = "png"
    figdir.mkdir(parents=True, exist_ok=True)


def save_multi(fig_name: str, figdir, formats=("png", "pdf")):
    for ext in formats:
        plt.savefig(figdir / f"{fig_name}.{ext}", dpi=300)
    plt.close()


# ---- plotting wrappers ----
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
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{pct:.1f}%", ha='center', va='bottom', fontsize=8, rotation=60)
    figpath.parent.mkdir(parents=True, exist_ok=True)
    save_multi(figpath.stem, figpath.parent)


def plot_cellbender_comparison(raw_counts: dict, cb_counts: dict, figdir: Path) -> None:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    figdir_cb = Path(figdir) / "QC_plots" / "cellbender"
    os.makedirs(figdir_cb, exist_ok=True)

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

    def _plot_single(row, outpath_stem):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(1)
        width = 0.35
        ax.bar(x - width / 2, [row["raw_reads"]], width, color="lightgray", label="raw")
        ax.bar(x + width / 2, [row["cb_reads"]], width, color="steelblue", label="cellbender")
        pct = row["pct_retained"]
        if row["cb_reads"] > 0:
            ax.text(x + width / 2, row["cb_reads"], f"{pct:.1f}%", ha="center", va="bottom", rotation=60)
        ax.set_title(row["sample"])
        ax.set_ylabel("Total reads")
        ax.set_xticks([])
        ax.legend()
        plt.tight_layout()
        save_multi(outpath_stem, figdir_cb)

    for _, row in df.iterrows():
        stem = f"{row['sample']}_QC_reads_before_after_cellbender"
        _plot_single(row, stem)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["raw_reads"], width, color="lightgray", label="raw")
    ax.bar(x + width / 2, df["cb_reads"], width, color="steelblue", label="cellbender")
    for i, pct in enumerate(df["pct_retained"]):
        if df.loc[i, "cb_reads"] > 0:
            ax.text(i + width / 2, df.loc[i, "cb_reads"], f"{pct:.1f}%", ha="center", va="bottom", rotation=60)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Total reads")
    ax.legend()
    ax.set_title("CellBender: total reads per sample")
    plt.tight_layout()
    save_multi("QC_reads_before_after_cellbender_AGGREGATE", figdir_cb)



def plot_final_cell_counts(adata, cfg) -> None:
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
    ax.bar(df["sample"], df["n_cells"], color="steelblue", alpha=0.8)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title("Final number of cells per sample (post-filtering)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.text(
        0.98, 0.95,
        f"Total: {total_cells:,} cells",
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7)
    )

    plt.tight_layout()
    save_multi("QC_cells_final_per_sample", figdir_qc)


def plot_mt_histogram(adata, cfg, suffix):
    figdir_qc = cfg.figdir / "QC_plots"
    figdir_qc.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.hist(adata.obs["pct_counts_mt"], bins=50, color="steelblue", alpha=0.8)
    plt.xlabel("Percent mitochondrial counts")
    plt.ylabel("Number of cells")
    plt.title("Distribution of mitochondrial content")
    plt.tight_layout()
    save_multi(f"{suffix}_QC_hist_pct_mt", figdir_qc)


def run_qc_plots_pre_filter(adata: ad.AnnData, cfg: LoadAndQCConfig) -> None:
    if not cfg.make_figures:
        return

    figdir_qc = cfg.figdir / "QC_plots"
    figdir_qc.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = figdir_qc

    if "doublet_score" in adata.obs:
        sc.pl.violin(
            adata,
            ["doublet_score"],
            groupby=cfg.batch_key,
            show=False
        )
        save_multi("QC_doublet_score_violin_prefilter", figdir_qc)

    plot_mt_histogram(adata, cfg, "prefilter")


def run_qc_plots_postfilter(adata: ad.AnnData, cfg: LoadAndQCConfig):
    import os
    figdir_qc = cfg.figdir / "QC_plots"
    os.makedirs(figdir_qc, exist_ok=True)

    from scanpy import settings as sc_settings
    old_figdir = sc_settings.figdir
    sc_settings.figdir = figdir_qc

    try:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            groupby=cfg.batch_key,
            show=False
        )
        save_multi("QC_violin_mt_counts_postfilter", figdir_qc)

        plot_mt_histogram(adata, cfg, "postfilter")

        sc.pl.scatter(
            adata,
            x="total_counts",
            y="n_genes_by_counts",
            color="pct_counts_mt",
            show=False
        )
        save_multi("QC_complexity_prefilter", figdir_qc)

        sc.pl.pca_variance_ratio(
            adata,
            log=True,
            n_pcs=cfg.max_pcs_plot,
            show=False
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

