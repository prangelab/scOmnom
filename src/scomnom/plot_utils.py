from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Mapping, Iterable, List, Any

import logging
from sklearn.metrics import silhouette_samples

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import re

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
RUN_FIG_SUBDIR: Path | None = None
RUN_KEY: str | None = None


# -------------------------------------------------------------------------
# Setup + saving
# -------------------------------------------------------------------------
def set_figure_formats(formats: Sequence[str]) -> None:
    global FIGURE_FORMATS
    FIGURE_FORMATS = list(formats)


def setup_scanpy_figs(figdir: Path, formats: Sequence[str] | None = None) -> None:
    """
    Configure Scanpy and global figure settings for scOmnom.

    figdir is the base root where per-format folders live, typically:
      <output_dir>/figures
    """
    global ROOT_FIGDIR, RUN_FIG_SUBDIR, RUN_KEY

    figdir = Path(figdir)
    ROOT_FIGDIR = figdir.resolve()

    if formats is not None:
        set_figure_formats(formats)

    # Reset run routing; it will be inferred lazily from first save_multi call.
    RUN_FIG_SUBDIR = None
    RUN_KEY = None

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

    ROOT_FIGDIR.mkdir(parents=True, exist_ok=True)


def save_multi(stem: str, figdir: Path, fig=None, *, savefig_kwargs: dict | None = None) -> None:
    """
    Save the current matplotlib figure (or a provided figure) to multiple formats.

    Output layout (Option B):
      ROOT_FIGDIR/<ext>/<RUN_FIG_SUBDIR>/<rel_figdir>/<stem>.<ext>

    Where RUN_FIG_SUBDIR is inferred from the first path component of `figdir`,
    e.g. integration -> integration_roundN.
    The filename stem is ALWAYS sanitized to be filesystem-safe.
    """
    import re
    import matplotlib.pyplot as plt
    global ROOT_FIGDIR, RUN_FIG_SUBDIR, RUN_KEY

    if ROOT_FIGDIR is None:
        raise RuntimeError("ROOT_FIGDIR is not set. Call setup_scanpy_figs() first.")

    # --------------------------------------------------
    # Sanitize filename stem
    # --------------------------------------------------
    def _safe_stem(s: str, max_len: int = 180) -> str:
        if s is None:
            return "figure"

        s = str(s)

        # Kill path separators first
        s = s.replace("/", "_").replace("\\", "_")

        # Replace other common offenders
        s = s.replace(":", "_")

        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # Replace anything not filesystem-safe
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

        # Collapse multiple underscores
        s = re.sub(r"_+", "_", s)

        # Trim junk from ends
        s = s.strip("._-")

        if not s:
            s = "figure"

        # Clamp length to avoid OS/path limits
        if len(s) > max_len:
            s = s[:max_len].rstrip("._-")

        return s

    stem = _safe_stem(stem)

    # --------------------------------------------------
    # Activate provided figure if any
    # --------------------------------------------------
    if fig is not None:
        plt.figure(fig.number)

    figdir = Path(figdir)

    # --------------------------------------------------
    # Save in all configured formats
    # --------------------------------------------------
    # Lazily infer run folder from the first save call
    if RUN_FIG_SUBDIR is None:
        RUN_KEY = _infer_run_key(figdir)
        RUN_FIG_SUBDIR = _next_round_subdir(
            root_figdir=ROOT_FIGDIR,
            formats=FIGURE_FORMATS,
            run_name=RUN_KEY,
        )
        LOGGER.info(
            "Figure run root: %s/<ext>/%s/",
            ROOT_FIGDIR,
            RUN_FIG_SUBDIR,
        )

        # Precreate per-format run dirs
        for ext in FIGURE_FORMATS:
            (ROOT_FIGDIR / ext / RUN_FIG_SUBDIR).mkdir(parents=True, exist_ok=True)

    # Always compute rel_figdir (avoid duplicate integration/integration)
    rel_figdir = figdir
    if RUN_KEY and rel_figdir.parts and rel_figdir.parts[0] == RUN_KEY:
        rel_figdir = Path(*rel_figdir.parts[1:])  # may become "."

    kwargs = dict(dpi=300)
    if savefig_kwargs:
        kwargs.update(savefig_kwargs)

    for ext in FIGURE_FORMATS:
        outdir = ROOT_FIGDIR / ext / RUN_FIG_SUBDIR / rel_figdir
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{stem}.{ext}"
        LOGGER.info("Saving figure: %s", outfile)
        plt.savefig(outfile, **kwargs)

    plt.close()


def save_umap_multi(
    stem: str,
    figdir: Path,
    fig: mpl.figure.Figure,
    *,
    pad_inches: float = 0.25,
    tight: bool = True,
    right: float | None = 0.78,
) -> None:
    """
    UMAP-only saver that prevents Scanpy legends / annotations from being clipped.
    Delegates actual saving to save_multi().
    """
    if right is not None:
        try:
            fig.subplots_adjust(right=right)
        except Exception:
            pass

    savefig_kwargs = {}
    if tight:
        savefig_kwargs["bbox_inches"] = "tight"
        savefig_kwargs["pad_inches"] = pad_inches

    save_multi(
        stem=stem,
        figdir=figdir,
        fig=fig,
        savefig_kwargs=savefig_kwargs,
    )


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _next_round_subdir(root_figdir: Path, formats: Sequence[str], run_name: str) -> Path:
    """
    Pick next <run_name>_roundN by scanning *only* that module's folders:
      <root_figdir>/<fmt>/<run_name>_roundN

    Other modules' folders (e.g. qc_round*) are ignored.
    """
    root_figdir = Path(root_figdir)
    rx = re.compile(rf"^{re.escape(run_name)}_round(\d+)$")

    existing: set[int] = set()

    for fmt in formats:
        fmt_dir = root_figdir / fmt
        if not fmt_dir.exists():
            continue

        for p in fmt_dir.iterdir():
            if not p.is_dir():
                continue
            m = rx.match(p.name)
            if m:
                existing.add(int(m.group(1)))

    n = 1
    while n in existing:
        n += 1

    return Path(f"{run_name}_round{n}")


def _infer_run_key(figdir: Path) -> str:
    """
    Infer module/run key from the first path component of figdir.
    Examples:
      Path("integration") -> "integration"
      Path("integration/umaps") -> "integration"
      Path("QC_plots/qc_metrics") -> "QC_plots"
    """
    figdir = Path(figdir)
    parts = figdir.parts
    if not parts:
        return "figures"
    return str(parts[0])


def _is_categorical_series(s: pd.Series) -> bool:
    return (
        pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    )


def _umap_figsize_for_key(adata: ad.AnnData, key: str) -> tuple[float, float]:
    """
    Choose a wide enough figure size so large categorical legends don't squash the UMAP.
    """
    base_w, base_h = 6.5, 5.5
    if key not in adata.obs:
        return base_w, base_h

    s = adata.obs[key]
    if _is_categorical_series(s):
        try:
            n = int(s.astype("category").cat.categories.size)
        except Exception:
            n = int(s.nunique(dropna=True))

        # widen with number of categories, but cap to keep things sane
        w = min(22.0, max(base_w, 7.5 + 0.35 * min(n, 50)))
        # a bit taller if legend becomes multi-row
        h = min(10.0, max(base_h, 5.5 + 0.06 * max(0, n - 20)))
        return float(w), float(h)

    return base_w, base_h


def _tune_umap_legend(fig: plt.Figure, n_cats: int) -> None:
    """
    Scanpy draws the legend on the main axis; move it to a right margin area and
    split into columns if it's tall.
    """
    if not fig.axes:
        return
    ax = fig.axes[0]
    leg = ax.get_legend()
    if leg is None:
        return

    # Choose columns to reduce legend height
    if n_cats <= 18:
        ncol = 1
    elif n_cats <= 40:
        ncol = 2
    else:
        ncol = 3

    try:
        leg.set_ncols(ncol)
    except Exception:
        try:
            leg._ncols = ncol  # older mpl
        except Exception:
            pass

    # Put legend in the right margin, vertically centered
    leg.set_bbox_to_anchor((1.02, 0.5))
    leg._loc = 6  # "center left" (works across mpl versions)
    try:
        leg.set_loc("center left")
    except Exception:
        pass


def _nice_gmt_name(gmt_path_or_name: str) -> str | None:
    """
    Try to convert an MSigDB GMT filename into a nice label.
    Returns None if we can't confidently infer.
    Examples:
      h.all.v2025.1.Hs.symbols.gmt -> HALLMARK
      c2.cp.reactome.v2025.1.Hs.symbols.gmt -> REACTOME
    """
    s = str(gmt_path_or_name)
    base = Path(s).name.lower()

    # hallmark collection
    if base.startswith("h.all."):
        return "HALLMARK"

    # common curated libs
    if "reactome" in base:
        return "REACTOME"
    if "kegg" in base:
        return "KEGG"
    if "wikipathways" in base:
        return "WIKIPATHWAYS"
    if "biocarta" in base:
        return "BIOCARTA"

    # If you later add more, do it here.
    return None


def _finalize_categorical_x(
    fig,
    ax,
    *,
    rotate: float = 45,
    ha: str = "right",
    bottom: float = 0.30,
    right: float = 0.98,
    left: float = 0.10,
    top: float = 0.92,
):
    """
    For plots with many categorical x tick labels:
    - rotate labels
    - reserve margins so labels don't get clipped
    """
    try:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha=ha)
    except Exception:
        pass

    try:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    except Exception:
        pass


def _clean_axes(ax):
    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_alpha(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def _reserve_bottom_for_xticklabels(
    fig,
    ax,
    *,
    rotation: float = 45,
    fontsize: float | None = None,
    ha: str = "right",
    extra_bottom: float = 0.03,
    max_bottom: float = 0.72,
):
    """
    Rotate x tick labels and reserve enough bottom margin so they don't clip.

    Heuristic: bottom margin increases with the longest label length.
    """
    labels = [t.get_text() for t in ax.get_xticklabels()]
    max_len = max((len(s) for s in labels), default=0)

    # Base bottom + length-dependent bump (works well for long cluster names)
    bottom = 0.26 + 0.0085 * max_len + extra_bottom
    bottom = float(min(max_bottom, bottom))

    try:
        for t in ax.get_xticklabels():
            t.set_rotation(rotation)
            t.set_ha(ha)
            if fontsize is not None:
                t.set_fontsize(fontsize)
    except Exception:
        pass

    # Critical: manual adjust AFTER any tight_layout / plotting
    try:
        fig.subplots_adjust(bottom=bottom)
    except Exception:
        pass


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
    Fixes squishing by allocating enough width for large legends.
    """
    import matplotlib.pyplot as plt

    if isinstance(keys, str):
        keys = [keys]

    if stem is None:
        name = f"QC_umap_{'_'.join(keys)}"
    else:
        name = stem

    if figdir is None:
        figdir = ROOT_FIGDIR

    # Plot each key separately so we can size per-legend and save cleanly.
    for key in keys:
        if key not in adata.obs:
            LOGGER.warning("umap_by: key '%s' not in adata.obs; skipping.", key)
            continue

        # Decide legend behavior
        s = adata.obs[key]
        is_cat = _is_categorical_series(s)
        if is_cat:
            try:
                n_cats = int(s.astype("category").cat.categories.size)
            except Exception:
                n_cats = int(s.nunique(dropna=True))
        else:
            n_cats = 0

        fig_w, fig_h = _umap_figsize_for_key(adata, key)

        # Create figure/axes explicitly (avoid Scanpy passing figsize to scatter)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass

        sc.pl.umap(
            adata,
            color=key,
            use_raw=False,
            show=False,
            ax=ax,  # <- key change
            legend_loc=("right margin" if is_cat else None),
            legend_fontsize=(10 if is_cat else None),
        )

        if is_cat and n_cats > 0:
            _tune_umap_legend(fig, n_cats)
            fig.subplots_adjust(left=0.06, right=0.72, top=0.92, bottom=0.08)
        else:
            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)

        save_multi(f"{name}_{key}", figdir, fig)
        plt.close(fig)


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

        # --- reserve space for long sample names ---
        n_cats = adata.obs[batch_key].astype("category").cat.categories.size

        # heuristic: more categories → more bottom margin
        bottom = 0.28 if n_cats <= 20 else 0.36 if n_cats <= 40 else 0.45

        fig = ax.figure

        if not horizontal:
            # categories are y ticklabels, give them room
            fig.subplots_adjust(left=0.35, right=0.98, top=0.92, bottom=0.10)

        _finalize_categorical_x(
            fig,
            ax,
            rotate=45 if horizontal else 0,  # when rotated layout, ticks are on y anyway
            ha="right",
            bottom=bottom,
        )

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


def plot_qc_filter_stack(
    adata,
    *,
    batch_key: str = "sample_id",
    figdir: Path,
    fname: str = "qc_filter_effects_stacked",
) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if "qc_filter_stats" not in adata.uns:
        raise KeyError("qc_filter_stats not found in adata.uns")

    df = adata.uns["qc_filter_stats"].copy()

    # --------------------------------------------------
    # Keep only per-sample cell filters
    # --------------------------------------------------
    df = df[
        (df["scope"] == "cell")
        & (df["batch"] != "ALL")
    ]

    if df.empty:
        raise ValueError("No per-sample QC filter stats available")

    # --------------------------------------------------
    # Determine filter order (as applied)
    # --------------------------------------------------
    filter_order = (
        df.groupby("filter")["n_before"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # --------------------------------------------------
    # Reconstruct fraction relative to ORIGINAL cell count
    # --------------------------------------------------
    rows = []

    for batch, g in df.groupby("batch"):
        g = g.set_index("filter").loc[filter_order]

        n0 = g["n_before"].iloc[0]  # original cells
        for filt, row in g.iterrows():
            rows.append(
                {
                    "batch": batch,
                    "filter": filt,
                    "frac_removed_total": row["n_removed"] / n0,
                }
            )

    plot_df = pd.DataFrame(rows)

    # Pivot: batch × filter
    plot_df = (
        plot_df
        .pivot(index="batch", columns="filter", values="frac_removed_total")
        .fillna(0.0)
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * plot_df.shape[0]), 4)
    )

    bottom = np.zeros(plot_df.shape[0])

    for filt in plot_df.columns:
        vals = plot_df[filt].values
        ax.bar(
            plot_df.index,
            vals,
            bottom=bottom,
            label=filt,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += vals

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of original cells removed")
    ax.set_title("QC filtering effects per sample (100% stacked)")
    ax.legend(
        title="Filter",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")

    fig.tight_layout()
    save_multi("QC Filter effects", figdir, fig)
    plt.close(fig)


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
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)

    if "X_umap" not in adata.obsm:
        LOGGER.warning("Skipping UMAP plots: X_umap not found.")
        return

    # 1) batch
    if batch_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=batch_key,
            show=False,
            return_fig=True,
            legend_loc="right margin",   # <- consistent, outside axes
        )
        save_umap_multi("umap_batch", figdir, fig, right=0.78)
    else:
        LOGGER.warning("Batch key '%s' not found in adata.obs", batch_key)

    # 2) clusters
    if cluster_key in adata.obs:
        fig = sc.pl.umap(
            adata,
            color=cluster_key,
            show=False,
            return_fig=True,
            legend_loc="right margin",   # <- NOT "on data" (avoids messy labels)
        )
        save_umap_multi(f"umap_{cluster_key}", figdir, fig, right=0.78)
    else:
        LOGGER.warning("Cluster key '%s' not found in adata.obs", cluster_key)

    LOGGER.info("Generated UMAP plots (batch + %s)", cluster_key)


# -------------------------------------------------------------------------
# scIB-style results table
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path


def plot_scib_results_table(scaled: pd.DataFrame) -> None:
    df = scaled.copy()

    # ------------------------------------------------------------------
    # 1. Clean and sort
    # ------------------------------------------------------------------
    df = df.loc[~df.index.str.contains("Metric", case=False, na=False)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Total", ascending=False)

    all_cols = df.columns.tolist()

    # ------------------------------------------------------------------
    # 2. Define categories
    # ------------------------------------------------------------------
    agg_metrics = [c for c in ["Batch correction", "Bio conservation", "Total"] if c in all_cols]
    batch_metrics = [
        c for c in ["iLISI", "KBET", "Graph connectivity", "PCR comparison", "Silhouette batch"]
        if c in all_cols
    ]
    bio_metrics = [c for c in all_cols if c not in agg_metrics + batch_metrics]

    ordered_cols = bio_metrics + batch_metrics + agg_metrics
    df = df[ordered_cols]

    # ------------------------------------------------------------------
    # 3. X-position mapping (tighter gaps, centered dividers)
    # ------------------------------------------------------------------
    cell_w, cell_h = 1.15, 0.85
    agg_gap = 0.6

    x_positions = {}
    x_curr = 0
    section_boundaries = []

    for i, col in enumerate(ordered_cols):
        if i > 0:
            prev_col = ordered_cols[i - 1]
            if (
                (prev_col in bio_metrics and col in batch_metrics)
                or (prev_col in batch_metrics and col in agg_metrics)
            ):
                # divider exactly centered in the gap
                section_boundaries.append(x_curr + agg_gap / 2)
                x_curr += agg_gap

        x_positions[col] = x_curr
        x_curr += 1

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(cell_w * (x_curr + 1), cell_h * (n_rows + 4)))

    # ------------------------------------------------------------------
    # 4. Draw data
    # ------------------------------------------------------------------
    for i, (method_name, row) in enumerate(df.iterrows()):
        y_coord = i + 0.5
        for col in df.columns:
            x_c = x_positions[col]
            val = row[col]
            if np.isnan(val):
                continue

            if col in agg_metrics:
                ax.barh(
                    y_coord,
                    val,
                    left=x_c,
                    height=cell_h * 0.55,
                    color="#a6bddb",
                    align="center",
                    zorder=2,
                )
                ax.text(
                    x_c + 0.05,
                    y_coord,
                    f"{val:.2f}",
                    va="center",
                    ha="left",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_c + 0.5,
                    y_coord,
                    s=1100,
                    c=[plt.cm.viridis(val)],
                    edgecolors="none",
                    zorder=2,
                )
                ax.text(
                    x_c + 0.5,
                    y_coord,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if val < 0.3 or val > 0.8 else "black",
                    zorder=3,
                )

    # ------------------------------------------------------------------
    # 5. Section dividers (shorter, centered)
    # ------------------------------------------------------------------
    for boundary in section_boundaries:
        ax.plot(
            [boundary, boundary],
            [0.2, n_rows - 0.2],
            color="gray",
            linestyle="-",
            linewidth=2.0,
            alpha=0.6,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # 6. Headers (lowered)
    # ------------------------------------------------------------------
    def draw_header(cols, title, y_top=-1.4, y_sub=-0.35):
        if not cols:
            return
        pos = [x_positions[c] for c in cols]
        center = (min(pos) + max(pos) + 1) / 2

        ax.text(
            center,
            y_top,
            title,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=15,
        )

        for c in cols:
            display_name = c.replace(" ", "\n")
            ax.text(
                x_positions[c] + 0.5,
                y_sub,
                display_name,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="medium",
            )

    draw_header(bio_metrics, "Biological Conservation")
    draw_header(batch_metrics, "Batch Correction")
    draw_header(agg_metrics, "Aggregate Scores")

    # ------------------------------------------------------------------
    # 7. Final styling
    # ------------------------------------------------------------------
    ax.set_ylim(-2.4, n_rows)
    ax.set_xlim(0, x_curr)
    ax.invert_yaxis()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # Row labels with baseline annotation
    new_labels = []
    for idx in df.index:
        idx_s = str(idx)
        if "unintegrated" in idx_s.lower():
            new_labels.append(f"{idx_s}\n(baseline)")
        elif "bbknn" in idx_s.lower():
            new_labels.append(f"{idx_s}\n(graph baseline)")
        else:
            new_labels.append(idx_s)

    ax.set_yticklabels(new_labels, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    save_multi("scIB_results_table", Path("integration"))
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

#------------------------------------------
# Plot UMAPs
#------------------------------------------
def plot_cluster_umaps(
    adata,
    label_key: str,
    batch_key: str,
    figdir: Path,
) -> None:
    fig = sc.pl.umap(adata, color=[label_key], show=False, return_fig=True, legend_loc="right margin")
    save_umap_multi(f"cluster_umap_{label_key}", figdir, fig, right=0.78)

    if batch_key in adata.obs:
        fig = sc.pl.umap(adata, color=[batch_key], show=False, return_fig=True, legend_loc="right margin")
        save_umap_multi(f"cluster_umap_{batch_key}", figdir, fig, right=0.78)

        fig = sc.pl.umap(
            adata,
            color=[batch_key, label_key],
            show=False,
            return_fig=True,
            legend_loc="right margin",
        )
        save_umap_multi(f"cluster_umap_{batch_key}_and_{label_key}", figdir, fig, right=0.78)



def plot_integration_umaps(
    adata,
    embedding_keys,
    batch_key: str,
    color: str,
) -> None:
    """
    Plot UMAPs for unintegrated + integrated embeddings.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import scanpy as sc
    import logging

    LOGGER = logging.getLogger(__name__)

    base = Path("integration")

    for emb in embedding_keys:
        if emb not in adata.obsm and emb != "BBKNN":
            LOGGER.warning("Embedding '%s' missing; skipping", emb)
            continue

        try:
            # ---------------------------
            # Single UMAP
            # ---------------------------
            tmp = adata.copy()

            if emb == "BBKNN":
                import bbknn
                bbknn.bbknn(tmp, batch_key=batch_key, use_rep="X_pca")
            else:
                sc.pp.neighbors(tmp, use_rep=emb)

            sc.tl.umap(tmp)

            fig = sc.pl.umap(
                tmp,
                color=color,
                title=emb,
                show=False,
                return_fig=True,
            )
            save_multi(f"umap_{emb}", figdir=base, fig=fig)
            plt.close(fig)

            # ---------------------------
            # Comparison vs Unintegrated
            # ---------------------------
            if emb != "Unintegrated" and "Unintegrated" in adata.obsm:

                # ===============================
                # Integrated embedding UMAP
                # ===============================
                tmp_int = adata.copy()

                if emb == "BBKNN":
                    import bbknn
                    bbknn.bbknn(tmp_int, batch_key=batch_key, use_rep="X_pca")
                else:
                    sc.pp.neighbors(tmp_int, use_rep=emb)

                sc.tl.umap(tmp_int)
                umap_int = tmp_int.obsm["X_umap"].copy()

                # ===============================
                # Unintegrated UMAP
                # ===============================
                tmp_raw = adata.copy()
                sc.pp.neighbors(tmp_raw, use_rep="Unintegrated")
                sc.tl.umap(tmp_raw)
                umap_raw = tmp_raw.obsm["X_umap"].copy()

                # ===============================
                # Plot side-by-side
                # ===============================
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                # LEFT: integrated embedding UMAP — hide labels to reduce clutter
                tmp_raw.obsm["X_umap"] = umap_int
                sc.pl.umap(
                    tmp_raw,
                    color=color,
                    ax=axs[0],
                    show=False,
                    title=emb,
                    legend_loc=None,  # <-- IMPORTANT: no "on data" labels on the left
                )

                # RIGHT: unintegrated UMAP — show labels on data
                tmp_raw.obsm["X_umap"] = umap_raw
                sc.pl.umap(
                    tmp_raw,
                    color=color,
                    ax=axs[1],
                    show=False,
                    title="Unintegrated",
                    legend_loc="on data",  # <-- labels only on the right
                )

                save_multi(
                    f"umap_{emb}_vs_Unintegrated",
                    figdir=base,
                    fig=fig,
                )
                plt.close(fig)

        except Exception as e:
            LOGGER.warning("Failed to plot UMAP for %s: %s", emb, e)


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

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(clusters)), 4))
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
    ax.set_xticklabels(clusters)
    ax.set_ylabel("Cells")
    ax.set_title("Cluster sizes")

    fig.tight_layout()
    _finalize_categorical_x(fig, ax, rotate=45, ha="right", bottom=0.40)
    _reserve_bottom_for_xticklabels(fig, ax, rotation=45, fontsize=9, ha="right")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.34))
    save_multi(stem, figdir)
    plt.close(fig)


def plot_cluster_qc_summary(
    adata,
    label_key: str,
    figdir: Path,
    stem: str = "cluster_qc_summary",
):
    if label_key not in adata.obs:
        LOGGER.warning("plot_cluster_qc_summary: '%s' not in obs.", label_key)
        return

    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing = [m for m in metrics if m not in adata.obs]
    if missing:
        LOGGER.warning("Missing QC fields: %s", missing)
        return

    df = adata.obs[[label_key] + metrics].groupby(label_key).mean()

    n = df.shape[0]
    fig_w = max(10, 0.35 * n)

    # 3 rows, shared x: avoids label clutter completely
    fig, axs = plt.subplots(3, 1, figsize=(fig_w, 7.5), sharex=True)
    fig.set_constrained_layout(False)

    for ax, m in zip(axs, metrics):
        _clean_axes(ax)
        df[m].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("")  # only bottom axis will get label

    axs[-1].set_xlabel("Cluster")

    # Hide x tick labels on upper panels
    for ax in axs[:-1]:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    # Rotate/reserve space ONLY once (bottom axis)
    _reserve_bottom_for_xticklabels(fig, axs[-1], rotation=45, fontsize=9, ha="right")

    # Reduce vertical spacing a bit
    fig.subplots_adjust(hspace=0.25)

    fig.tight_layout()
    save_multi(stem, figdir, fig)
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

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * df["cluster"].nunique()), 4.5))
    _clean_axes(ax)

    df.boxplot(column="silhouette", by="cluster", ax=ax)
    plt.suptitle("")  # remove pandas default
    ax.set_title("Silhouette distribution per cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    _finalize_categorical_x(fig, ax, rotate=45, ha="right", bottom=0.45)
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
    Legend is placed to the right (outside axes) to avoid overlaying bars.
    """
    import matplotlib.pyplot as plt

    if label_key not in adata.obs or batch_key not in adata.obs:
        LOGGER.warning("plot_cluster_batch_composition: required columns missing.")
        return

    df = (
        adata.obs[[label_key, batch_key]]
        .groupby([label_key, batch_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    frac = df.div(df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, 0.40 * len(df)), 4))
    _clean_axes(ax)

    # Reserve bottom for rotated x tick labels
    _reserve_bottom_for_xticklabels(fig, ax, rotation=45, fontsize=9, ha="right")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.36))

    # Plot WITHOUT pandas legend (we'll add a clean one outside)
    frac.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
        edgecolor="black",
        linewidth=0.3,
        legend=False,
    )

    ax.set_ylabel("Fraction")
    ax.set_title("Batch composition per cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- Legend to the right, outside the plotting area ---
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # make room on the right for legend
        fig.subplots_adjust(right=0.80)

        ax.legend(
            handles,
            labels,
            title=batch_key,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
        )

    # Keep your existing final x-axis layout helper
    _finalize_categorical_x(fig, ax, rotate=45, ha="right", bottom=0.42)

    save_multi(stem, figdir, fig)
    plt.close(fig)

# -------------------------------------------------------------------------
# Decoupler net plots (msigdb / progeny / dorothea)
# -------------------------------------------------------------------------
def _decoupler_figdir(base: Path | None, net_name: str) -> Path:
    """
    Put decoupler plots under:
      cluster_and_annotate/decoupler/<net_name>/
    while staying compatible with save_multi() routing.
    """
    base = Path("cluster_and_annotate") if base is None else Path(base)
    return base / "decoupler" / str(net_name).lower().strip()


def _zscore_cols(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Z-score each column across rows (clusters)."""
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, np.nan)
    z = (df - mu) / (sd + eps)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _top_features_global(
    activity: pd.DataFrame,
    k: int,
    *,
    mode: str = "var",  # "var" or "mean_abs"
    signed: bool = True,
) -> list[str]:
    """
    Pick top-k features globally. activity is clusters x features.
    """
    if activity is None or activity.empty:
        return []
    A = activity.copy()
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if not signed:
        A = A.abs()

    if mode == "mean_abs":
        score = A.abs().mean(axis=0) if signed else A.mean(axis=0)
    else:
        score = A.var(axis=0)

    score = score.sort_values(ascending=False)
    return score.head(int(k)).index.astype(str).tolist()


def _wrap_labels(labels: Sequence[str], wrap_at: int = 38) -> list[str]:
    import textwrap

    out: list[str] = []
    for s in labels:
        s = str(s)
        out.append(
            "\n".join(
                textwrap.wrap(
                    s,
                    width=int(wrap_at),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return out


def _msigdb_prefix(term: str) -> str:
    """
    Determine "GMT family" prefix for MSigDB-like pathway names.
    Examples:
      - "HALLMARK_TNFA_SIGNALING_VIA_NFKB" -> "HALLMARK"
      - "REACTOME_SOMETHING"              -> "REACTOME"
      - "PREFIX::TERM"                    -> "PREFIX"
    """
    term = str(term)
    if "::" in term:
        return term.split("::", 1)[0].strip() or "UNKNOWN"
    # Default MSigDB style: PREFIX_REST
    return (term.split("_", 1)[0].strip() or "UNKNOWN")


def _split_activity_for_msigdb(
    activity: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split MSigDB activity (clusters x pathways) into {prefix -> activity_sub}.
    For non-MSigDB nets, caller should not use this.
    """
    if activity is None or activity.empty:
        return {}
    cols = activity.columns.astype(str)
    prefixes = pd.Series(cols, index=cols).map(_msigdb_prefix)
    out: dict[str, pd.DataFrame] = {}
    for pfx, cols_idx in prefixes.groupby(prefixes).groups.items():
        cols_list = list(cols_idx)
        sub = activity.loc[:, cols_list].copy()
        out[str(pfx)] = sub
    return out


def _dynamic_left_margin_from_labels(labels: Sequence[str], *, base: float = 0.22) -> float:
    """
    Compute a left margin fraction [0,1] based on the longest label length.
    Tuned for horizontal barplots / dotplots with long y labels.
    """
    if labels is None:
        return float(np.clip(base, 0.28, 0.72))
    try:
        max_len = int(max(len(str(x)) for x in labels)) if len(labels) else 0
    except Exception:
        max_len = 0
    # ~120 chars -> ~0.70, ~20 chars -> ~0.34
    left = base + 0.0040 * max_len
    return float(np.clip(left, 0.28, 0.72))


def _dynamic_fig_width_for_barplot(labels: Sequence[str], *, min_w: float = 12.0, max_w: float = 26.0) -> float:
    """
    Compute figure width in inches to accommodate long pathway names without squishing.
    """
    try:
        max_len = int(max(len(str(x)) for x in labels)) if len(labels) else 0
    except Exception:
        max_len = 0
    # Increase width with label length; MSigDB often needs a lot.
    # 40 chars -> +2.4, 120 chars -> +9.6
    w = float(min_w + 0.08 * max_len)
    return float(np.clip(w, min_w, max_w))


def plot_decoupler_activity_heatmap(
    activity: pd.DataFrame,
    *,
    net_name: str,
    figdir: Path | None,
    top_k: int = 30,
    rank_mode: str = "var",  # "var" or "mean_abs"
    use_zscore: bool = True,
    wrap_labels: bool = True,
    wrap_at: int = 38,
    cmap: str = "viridis",
    stem: str = "heatmap_top",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Global heatmap: clusters (rows) x top-K features (cols).
    Uses z-scored activity across clusters by default for comparability.

    FIXES:
      - No overlay line grid
      - Dynamic margins
    """
    if activity is None or activity.empty:
        return

    import seaborn as sns

    outdir = _decoupler_figdir(figdir, net_name)

    A = activity.copy()
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feats = _top_features_global(A, k=top_k, mode=rank_mode, signed=True)
    if not feats:
        return
    sub = A.loc[:, feats].copy()

    if use_zscore:
        sub = _zscore_cols(sub)

    # Optional label wrapping (helps MSigDB)
    if wrap_labels:
        sub.columns = _wrap_labels(sub.columns, wrap_at=wrap_at)

    # Figure sizing
    fig_w = max(8.0, 2.5 + 0.35 * sub.shape[1])
    fig_h = max(4.5, 2.2 + 0.28 * sub.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Remove cell borders / overlay gridlines
    sns.heatmap(
        sub,
        ax=ax,
        cmap=cmap,
        cbar=True,
        linewidths=0.0,
        linecolor=None,
        square=False,
    )

    ax.set_xlabel("Feature" if net_name.lower() != "dorothea" else "TF")
    ax.set_ylabel("Cluster")

    ttl_prefix = f"{title_prefix}: " if title_prefix else ""
    ttl = f"{ttl_prefix}{net_name}: top {int(top_k)} activity ({'z-score' if use_zscore else 'raw'})"
    ax.set_title(ttl, fontsize=14, pad=12)

    # tick styling
    ax.tick_params(axis="x", rotation=60, labelsize=8, pad=6)
    ax.tick_params(axis="y", rotation=0, labelsize=9, pad=4)
    for t in ax.get_xticklabels():
        t.set_ha("right")
        t.set_rotation_mode("anchor")
        t.set_multialignment("right")

    # Dynamic bottom margin based on max wrapped label line length
    # (keeps long pathway names from colliding with the figure edge)
    try:
        max_lab_len = int(max(len(str(c)) for c in sub.columns))
    except Exception:
        max_lab_len = 0
    bottom = float(np.clip(0.22 + 0.0035 * max_lab_len, 0.22, 0.60))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=bottom)

    sfx = f"{stem}{int(top_k)}" + ("_z" if use_zscore else "_raw")
    save_multi(sfx, outdir, fig)
    plt.close(fig)


def plot_decoupler_cluster_topn_barplots(
    activity: pd.DataFrame,
    *,
    net_name: str,
    figdir: Path | None,
    n: int = 10,
    use_abs: bool = False,
    stem_prefix: str = "cluster",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Per-cluster Top-N barplots (modeled after your ssGSEA barplots).
    One figure per cluster.

    FIXES:
      - Dynamic left margin (already present) + improved
      - Dynamic width to avoid MSigDB squishing
    """
    if activity is None or activity.empty:
        return

    outdir = _decoupler_figdir(figdir, net_name)

    A = activity.copy()
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for cl in A.index.astype(str):
        s = A.loc[cl].copy()
        s_rank = s.abs() if use_abs else s

        top = s_rank.sort_values(ascending=False).head(int(n))
        if top.empty:
            continue

        vals = s.loc[top.index] if use_abs else top
        vals_plot = vals.sort_values(ascending=True)

        # Dynamic margins/width for long labels
        left = _dynamic_left_margin_from_labels(vals_plot.index, base=0.22)
        fig_w = _dynamic_fig_width_for_barplot(vals_plot.index, min_w=12.0, max_w=28.0)
        fig_h = max(2.8, 0.70 * len(vals_plot) + 1.8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=left, right=0.96, top=0.82, bottom=0.14)

        y = np.arange(len(vals_plot))
        ax.barh(
            y=y,
            width=vals_plot.values,
            color="#3b84a8",
            edgecolor="#1f2d3a",
            linewidth=0.6,
            zorder=3,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(vals_plot.index, fontsize=12)

        ax.set_xlabel("Activity", fontsize=13)
        ax.set_ylabel("Feature" if net_name.lower() != "dorothea" else "TF", fontsize=13)
        ax.invert_yaxis()

        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Title + subtitle (ssGSEA style)
        main_title = str(cl)
        if title_prefix:
            main_title = f"{title_prefix} • {main_title}"

        ax.text(
            0.5,
            1.10,
            main_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.04,
            str(net_name).upper(),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            color="#666666",
            clip_on=False,
        )

        stem = f"{stem_prefix}_{cl}__top{int(n)}_bar"
        if use_abs:
            stem += "_abs"

        save_multi(stem, outdir, fig)
        plt.close(fig)


def plot_decoupler_dotplot(
    activity: pd.DataFrame,
    *,
    net_name: str,
    figdir: Path | None,
    top_k: int = 30,
    rank_mode: str = "var",  # "var" or "mean_abs"
    color_by: str = "z",  # "z" or "raw"
    size_by: str = "abs_raw",  # "abs_raw" or "abs_z"
    wrap_labels: bool = True,
    wrap_at: int = 38,
    cmap: str = "viridis",
    stem: str = "dotplot_top",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Dotplot matrix:
      x = clusters, y = features (top-K global)
      color = z-score (default) or raw
      size  = abs(raw) (default) or abs(z)

    FIXES:
      - Add size legend
      - Dynamic left margin for long feature names
    """
    if activity is None or activity.empty:
        return

    outdir = _decoupler_figdir(figdir, net_name)

    A = activity.copy()
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feats = _top_features_global(A, k=top_k, mode=rank_mode, signed=True)
    if not feats:
        return

    sub_raw = A.loc[:, feats].copy()  # clusters x feats
    sub_z = _zscore_cols(sub_raw)

    # choose aesthetics
    if color_by == "raw":
        color_mat = sub_raw
        cbar_label = "activity"
    else:
        color_mat = sub_z
        cbar_label = "z-score"

    if size_by == "abs_z":
        size_mat = sub_z.abs()
        size_label = "|z|"
    else:
        size_mat = sub_raw.abs()
        size_label = "|raw|"

    clusters = sub_raw.index.astype(str).tolist()
    features = sub_raw.columns.astype(str).tolist()

    if wrap_labels:
        features_disp = _wrap_labels(features, wrap_at=wrap_at)
    else:
        features_disp = features

    # build long table
    rows: list[dict] = []
    for j, feat in enumerate(features):
        for cl in clusters:
            rows.append(
                {
                    "cluster": cl,
                    "feature": features_disp[j],
                    "color": float(color_mat.loc[cl, feat]),
                    "size": float(size_mat.loc[cl, feat]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # size scaling
    svals = df["size"].to_numpy(dtype=float)
    finite = np.isfinite(svals)
    if not finite.any():
        return
    s_min = float(np.nanmin(svals[finite]))
    s_max = float(np.nanmax(svals[finite]))

    def size_scale(v: float) -> float:
        if not np.isfinite(v):
            return 60.0
        if s_max <= s_min:
            return 120.0
        return float(60.0 + (v - s_min) / (s_max - s_min) * (340.0 - 60.0))

    sizes = np.array([size_scale(v) for v in svals], dtype=float)

    # figure sizing
    fig_w = max(9.0, 2.8 + 0.35 * len(clusters))
    fig_h = max(5.0, 2.2 + 0.28 * len(features))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # dynamic left margin for long feature names
    left = _dynamic_left_margin_from_labels(df["feature"].values, base=0.20)
    fig.subplots_adjust(left=left, right=0.84, top=0.88, bottom=0.18)

    sca = ax.scatter(
        x=df["cluster"].values,
        y=df["feature"].values,
        s=sizes,
        c=df["color"].values,
        cmap=cmap,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
        zorder=3,
    )

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Feature" if net_name.lower() != "dorothea" else "TF", fontsize=12)

    ttl_prefix = f"{title_prefix} • " if title_prefix else ""
    ax.set_title(
        f"{ttl_prefix}{net_name}: top {int(top_k)} dotplot ({cbar_label}, size={size_by})",
        fontsize=14,
        pad=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # colorbar
    cbar = fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)

    # --- size legend (quantile-based), placed near colorbar ---
    # pick 3 reference sizes at 25/50/75% of finite sizes
    refs = np.quantile(svals[finite], [0.25, 0.50, 0.75])
    refs = np.unique(np.round(refs, 3))
    refs = refs[refs > 0]
    if refs.size > 0:
        handles = [
            ax.scatter([], [], s=size_scale(float(v)), color="gray", alpha=0.6, edgecolors="none", label=str(v))
            for v in refs
        ]
        leg = ax.legend(
            handles=handles,
            title=size_label,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.65),
            frameon=False,
            borderaxespad=0.0,
        )
        if leg and leg.get_title():
            leg.get_title().set_fontsize(10)
        for txt in leg.get_texts():
            txt.set_fontsize(9)

    sfx = f"{stem}{int(top_k)}_{cbar_label.replace('-', '')}_{size_by}"
    save_multi(sfx, outdir, fig)
    plt.close(fig)


def plot_decoupler_all_styles(
    adata,
    *,
    net_key: str,
    net_name: Optional[str] = None,
    figdir: Path | None = None,
    heatmap_top_k: int = 30,
    bar_top_n: int = 10,
    dotplot_top_k: int = 30,
) -> None:
    """
    Convenience wrapper:
    Reads adata.uns[net_key]["activity"] and makes:
      1) heatmap
      2) per-cluster topN barplots
      3) dotplot

    FIXES:
      - MSigDB is split into one set of plots per GMT-family prefix
        (e.g. HALLMARK, REACTOME, ...), dynamically discovered.
    """
    net_name = net_name or net_key
    block = adata.uns.get(net_key, {})
    activity = block.get("activity", None)
    if activity is None or not isinstance(activity, pd.DataFrame) or activity.empty:
        return

    # MSigDB: split by GMT-family prefix and make all plot styles per prefix
    if str(net_key).lower().strip() == "msigdb":
        splits = _split_activity_for_msigdb(activity)
        if not splits:
            return

        # Stable-ish ordering (HALLMARK first if present, then alphabetical)
        ordered = sorted(splits.keys(), key=lambda x: (0 if x.upper() == "HALLMARK" else 1, x.upper()))

        for pfx in ordered:
            sub = splits[pfx]
            if sub is None or sub.empty:
                continue

            title_prefix = str(pfx).upper()

            plot_decoupler_activity_heatmap(
                sub,
                net_name=net_name,
                figdir=figdir,
                top_k=heatmap_top_k,
                rank_mode="var",
                use_zscore=True,
                wrap_labels=True,
                stem=f"heatmap_top_{pfx.lower()}_",
                title_prefix=title_prefix,
            )

            plot_decoupler_cluster_topn_barplots(
                sub,
                net_name=net_name,
                figdir=figdir,
                n=bar_top_n,
                use_abs=False,
                stem_prefix=f"cluster_{pfx.lower()}",
                title_prefix=title_prefix,
            )

            plot_decoupler_dotplot(
                sub,
                net_name=net_name,
                figdir=figdir,
                top_k=dotplot_top_k,
                rank_mode="var",
                color_by="z",
                size_by="abs_raw",
                wrap_labels=True,
                stem=f"dotplot_top_{pfx.lower()}_",
                title_prefix=title_prefix,
            )

        return

    # Non-MSigDB nets: unchanged behavior (single set of plots)
    plot_decoupler_activity_heatmap(
        activity,
        net_name=net_name,
        figdir=figdir,
        top_k=heatmap_top_k,
        rank_mode="var",
        use_zscore=True,
        wrap_labels=True,
    )

    plot_decoupler_cluster_topn_barplots(
        activity,
        net_name=net_name,
        figdir=figdir,
        n=bar_top_n,
        use_abs=False,
    )

    plot_decoupler_dotplot(
        activity,
        net_name=net_name,
        figdir=figdir,
        top_k=dotplot_top_k,
        rank_mode="var",
        color_by="z",
        size_by="abs_raw",
        wrap_labels=True,
    )
