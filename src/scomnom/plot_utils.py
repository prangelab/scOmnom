from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib as mpl
import logging
LOGGER = logging.getLogger(__name__)

# Global params
mpl.rcParams["xtick.alignment"] = "right"
mpl.rcParams["figure.subplot.bottom"] = 0.25
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.subplot.right"] = 0.9  # gives space for legends

FIGURE_FORMATS = ["png", "pdf"]
ROOT_FIGDIR = None

def set_figure_formats(formats):
    global FIGURE_FORMATS
    FIGURE_FORMATS = list(formats)


def setup_scanpy_figs(figdir: Path, formats=None) -> None:
    global ROOT_FIGDIR
    ROOT_FIGDIR = figdir.resolve()

    import scanpy as sc

    if formats is not None:
        set_figure_formats(formats)

    sc.settings.figdir = str(figdir)
    sc.settings.autoshow = False
    sc.settings.autosave = False

    # Set a neutral figure style
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


def save_multi(stem: str, figdir: Path):
    import matplotlib.pyplot as plt
    global ROOT_FIGDIR

    figdir = figdir.resolve()
    root = ROOT_FIGDIR

    # compute rel path ALWAYS relative to the true figure root
    rel = figdir.relative_to(root)

    for ext in FIGURE_FORMATS:
        outdir = root / ext / rel
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{stem}.{ext}"

        LOGGER.info(f"Saving figure: {outfile}")
        plt.savefig(outfile, dpi=300)

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
    save_multi(figpath.stem, figpath.parent)


def plot_percent_retention(
    baseline_reads: dict,
    method_reads: dict,
    method_label: str,
    figdir: Path,
    stem: Optional[str] = None,
):
    """
    baseline_reads: dict sample -> reads (raw-unfiltered)
    method_reads:   dict sample -> reads (filtered, CB, etc.)
    method_label:   e.g. "cellranger_filtered", "cellbender"
    figdir:         Path to .../cell_qc
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    samples = sorted(set(baseline_reads) | set(method_reads))

    df = pd.DataFrame({
        "sample": samples,
        "baseline": [baseline_reads.get(s, 0) for s in samples],
        "method":   [method_reads.get(s, 0) for s in samples],
    })

    df["pct_retained"] = np.where(
        df["baseline"] > 0,
        100 * df["method"] / df["baseline"],
        0,
    )

    if stem is None:
        stem = f"pct_reads_retained_{method_label}"

    # ---- plot ----
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7), 5))

    bars = ax.bar(x, df["pct_retained"], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("% reads retained vs raw")
    ax.set_title(f"{method_label}: % reads retained")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=8, rotation=60
        )

    plt.tight_layout()
    save_multi(stem, figdir)

    # Save table
    df.to_csv(figdir / f"{stem}.tsv", sep="\t", index=False)


def plot_median_complexity(
    method_map: dict,
    method_label: str,
    figdir: Path,
    stem_prefix: Optional[str] = None,
):
    """
    method_map:   dict sample -> AnnData
    method_label: e.g. "raw_filtered", "cellranger_filtered", "cellbender"
    figdir:       Path to .../cell_qc
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    samples = sorted(method_map.keys())

    med_genes = []
    med_umis = []

    for s in samples:
        adata = method_map[s]
        med_genes.append(float(np.median(adata.obs["n_genes_by_counts"])))
        med_umis.append(float(np.median(adata.obs["total_counts"])))

    df = pd.DataFrame({
        "sample": samples,
        "median_genes": med_genes,
        "median_umis": med_umis,
    })

    if stem_prefix is None:
        stem_prefix = f"complexity_{method_label}"

    # ---- genes plot ----
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7), 5))
    x = np.arange(len(df))

    ax.bar(x, df["median_genes"], color="darkgreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Median genes per cell")
    ax.set_title(f"{method_label}: median genes per cell")
    plt.tight_layout()
    save_multi(f"{stem_prefix}_median_genes", figdir)

    # ---- UMIs plot ----
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7), 5))
    ax.bar(x, df["median_umis"], color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel("Median UMIs per cell")
    ax.set_title(f"{method_label}: median UMIs per cell")
    plt.tight_layout()
    save_multi(f"{stem_prefix}_median_umis", figdir)

    # table
    df.to_csv(figdir / f"{stem_prefix}_summary.tsv", sep="\t", index=False)


def plot_elbow_knee(
    adata: ad.AnnData,
    figpath_stem: str,
    figdir: Path,
    title: str = "Barcode Rank UMI Knee Plot"
):
    """
    Generate a Cell Ranger–style elbow/knee plot.

    - Sort barcodes by total UMIs (descending)
    - Plot rank vs UMIs
    - Compute and annotate knee point
    - log–log axes (matches Cell Ranger)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from kneed import KneeLocator

    # Compute raw counts from X or counts_raw
    if "counts_raw" in adata.layers:
        total = np.asarray(adata.layers["counts_raw"].sum(axis=1)).ravel()
    else:
        total = np.asarray(adata.X.sum(axis=1)).ravel()

    # Sort
    sorted_idx = np.argsort(total)[::-1]
    sorted_counts = total[sorted_idx]
    ranks = np.arange(1, len(sorted_counts) + 1)

    # Knee detection
    kl = KneeLocator(
        ranks,
        sorted_counts,
        curve="convex",
        direction="decreasing"
    )
    knee_rank = kl.elbow if kl.elbow is not None else None

    plt.figure(figsize=(6, 5))
    plt.plot(ranks, sorted_counts, lw=1)

    if knee_rank is not None:
        knee_value = sorted_counts[knee_rank - 1]
        plt.axvline(knee_rank, color="red", linestyle="--", label=f"Knee ~{knee_rank}")
        plt.axhline(knee_value, color="red", linestyle="--")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Barcode rank")
    plt.ylabel("Total UMI counts")
    plt.title(title)
    plt.tight_layout()

    save_multi(figpath_stem, figdir)


def plot_read_comparison(
    ref_counts: dict,
    other_counts: dict,
    ref_label: str,
    other_label: str,
    figdir: Path,
    stem: str,
):
    """
    Plot read counts per sample between two datasets.
    Samples are matched by name; missing samples are shown as zero.

    ref_counts: dict[sample -> reads]
    other_counts: dict[sample -> reads]
    figdir: directory where figures are stored
    stem: file stem for exporting
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Align sample names
    samples = sorted(set(ref_counts) | set(other_counts))
    df = pd.DataFrame({
        "sample": samples,
        ref_label: [ref_counts.get(s, 0) for s in samples],
        other_label: [other_counts.get(s, 0) for s in samples],
    })
    df["pct_retained"] = np.where(
        df[ref_label] > 0,
        100 * df[other_label] / df[ref_label],
        0
    )

    # ---- Plot ----
    plt.figure(figsize=(max(6, len(samples) * 0.8), 6))
    x = np.arange(len(samples))
    width = 0.35

    plt.bar(x - width / 2, df[ref_label], width,
            color="lightgray", label=ref_label)
    plt.bar(x + width / 2, df[other_label], width,
            color="steelblue", label=other_label)

    for i, pct in enumerate(df["pct_retained"]):
        if df.loc[i, other_label] > 0:
            plt.text(
                i + width / 2,
                df.loc[i, other_label],
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                rotation=60,
                fontsize=8,
            )

    plt.xticks(x, df["sample"], rotation=45, ha="right")
    plt.ylabel("Total reads")
    plt.title(f"Total reads per sample: {ref_label} vs {other_label}")
    plt.legend()
    plt.tight_layout()

    # Save using your multi-format saver
    save_multi(stem, figdir)

    # Also save the underlying table
    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / figdir.relative_to(ROOT_FIGDIR)
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / f"{stem}.tsv", sep="\t", index=False)


def plot_final_cell_counts(adata, cfg) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os

    figdir_qc = Path(cfg.figdir) / "QC_plots"

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

