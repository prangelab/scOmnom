# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import anndata as ad

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .logging_utils import init_logging

from .de_utils import (
    CountsSpec,
    MarkersSpec,
    compute_markers,
    de_cluster_vs_rest,
    de_condition_within_cluster,
    resolve_groupby_from_round,
)

LOGGER = logging.getLogger(__name__)


def run_markers_and_de(cfg) -> ad.AnnData:
    """
    Orchestrator wrapper for CLI / pipeline runs.

    IMPORTANT:
      - Notebook users should call functions in scomnom.de directly.
      - This orchestrator is the *only* place that should load/save AnnData.

    Expected cfg fields (minimal):
      - input_path: Path
      - output_dir: Path
      - logfile: Optional[Path]
      - figdir_name: str, figure_formats: list[str], make_figures: bool (optional)
      - batch_key/sample_key, groupby/round selection knobs (optional)
      - de settings: min_cells, alpha, condition_key, etc. (optional)
    """
    init_logging(getattr(cfg, "logfile", None))
    LOGGER.info("Starting markers_and_de orchestrator")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # Resolve groupby key (round-aware by default)
    groupby = resolve_groupby_from_round(
        adata,
        groupby=getattr(cfg, "groupby", None),
        label_source=getattr(cfg, "label_source", "pretty"),
        round_id=getattr(cfg, "round_id", None),
    )

    sample_key = getattr(cfg, "sample_key", None) or getattr(cfg, "batch_key", None) or adata.uns.get("batch_key")
    if sample_key is None:
        raise RuntimeError("markers_and_de: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")

    LOGGER.info("markers_and_de: groupby=%r, sample_key=%r", groupby, sample_key)

    # Markers
    markers_spec = MarkersSpec(
        key_added=str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon")),
        method=str(getattr(cfg, "markers_method", "wilcoxon")),
        n_genes=int(getattr(cfg, "markers_n_genes", 100)),
        rankby_abs=bool(getattr(cfg, "markers_rankby_abs", True)),
        use_raw=bool(getattr(cfg, "markers_use_raw", False)),
        downsample_threshold=int(getattr(cfg, "markers_downsample_threshold", 500_000)),
        downsample_max_per_group=int(getattr(cfg, "markers_downsample_max_per_group", 2_000)),
        random_state=int(getattr(cfg, "random_state", 42)),
    )
    compute_markers(adata, groupby=groupby, spec=markers_spec, inplace=True)

    # DE: cluster vs rest
    counts_spec = CountsSpec(
        priority_layers=tuple(getattr(cfg, "counts_layers", ("counts_cb", "counts_raw"))),
        allow_X=bool(getattr(cfg, "allow_X_counts", True)),
    )

    de_out = output_dir / "de_tables"
    de_out.mkdir(parents=True, exist_ok=True)

    manifest1 = de_cluster_vs_rest(
        adata,
        groupby=groupby,
        sample_key=str(sample_key),
        counts=counts_spec,
        min_cells_target=int(getattr(cfg, "min_cells_target", 20)),
        alpha=float(getattr(cfg, "alpha", 0.05)),
        out_dir=de_out / "cluster_vs_rest",
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        inplace=True,
    )

    # Optional condition DE
    condition_key = getattr(cfg, "condition_key", None)
    if condition_key:
        manifest2 = de_condition_within_cluster(
            adata,
            groupby=groupby,
            sample_key=str(sample_key),
            condition_key=str(condition_key),
            reference=getattr(cfg, "condition_reference", None),
            counts=counts_spec,
            min_cells=int(getattr(cfg, "min_cells_condition", 20)),
            alpha=float(getattr(cfg, "alpha", 0.05)),
            out_dir=de_out / f"condition_within_cluster__{condition_key}",
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            inplace=True,
        )
    else:
        manifest2 = None
        LOGGER.info("markers_and_de: condition_key not set; skipping condition DE.")

    # Minimal provenance
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "groupby": str(groupby),
            "sample_key": str(sample_key),
            "markers_key": markers_spec.key_added,
            "counts_layers": list(counts_spec.priority_layers),
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "manifests": {
                "cluster_vs_rest": manifest1.to_dict(orient="records"),
                "condition_within_cluster": manifest2.to_dict(orient="records") if manifest2 is not None else None,
            },
        }
    )

    from . import de_plot_utils

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if bool(getattr(cfg, "make_figures", True)):
        figroot = figdir / "markers_and_de"

        # --------------------------------------------------
        # Cluster vs Rest DE plots
        # --------------------------------------------------
        figdir_cvr = figroot / "cluster_vs_rest"

        # Overview / summary
        fig = de_plot_utils.plot_de_manifest_overview(
            manifest1,
            title=f"Cluster vs Rest (alpha={alpha:g})",
        )
        plot_utils.save_multi(
            stem="overview",
            figdir=figdir_cvr,
            fig=fig,
        )

        # Volcano grid
        fig = de_plot_utils.plot_volcano_grid_from_manifest(
            manifest1,
            alpha=alpha,
            max_panels=int(getattr(cfg, "volcano_max_panels", 16)),
            top_n_labels=int(getattr(cfg, "volcano_top_n_labels", 10)),
            title="Cluster vs Rest – volcano plots",
        )
        plot_utils.save_multi(
            stem="volcano_grid",
            figdir=figdir_cvr,
            fig=fig,
        )

        # Dotplot
        fig = de_plot_utils.plot_top_de_dotplot(
            adata,
            manifest1,
            groupby=str(groupby),
            alpha=alpha,
            top_n_per_group=int(getattr(cfg, "dotplot_top_n_per_group", 10)),
            use_raw=bool(getattr(cfg, "dotplot_use_raw", False)),
            layer=getattr(cfg, "dotplot_layer", None),
        )
        plot_utils.save_multi(
            stem="dotplot_top_de",
            figdir=figdir_cvr,
            fig=fig,
        )

        # Heatmap
        fig = de_plot_utils.plot_top_de_heatmap(
            adata,
            manifest1,
            groupby=str(groupby),
            alpha=alpha,
            top_n_per_group=int(getattr(cfg, "heatmap_top_n_per_group", 20)),
            use_raw=bool(getattr(cfg, "heatmap_use_raw", False)),
            layer=getattr(cfg, "heatmap_layer", None),
        )
        plot_utils.save_multi(
            stem="heatmap_top_de",
            figdir=figdir_cvr,
            fig=fig,
        )

        # --------------------------------------------------
        # Condition-within-cluster DE plots (if present)
        # --------------------------------------------------
        if manifest2 is not None:
            figdir_cond = figroot / f"condition_within_cluster__{condition_key}"

            fig = de_plot_utils.plot_de_manifest_overview(
                manifest2,
                title=f"Condition within cluster: {condition_key} (alpha={alpha:g})",
            )
            plot_utils.save_multi(
                stem="overview",
                figdir=figdir_cond,
                fig=fig,
            )

            fig = de_plot_utils.plot_volcano_grid_from_manifest(
                manifest2,
                alpha=alpha,
                max_panels=int(getattr(cfg, "volcano_max_panels", 16)),
                top_n_labels=int(getattr(cfg, "volcano_top_n_labels", 10)),
                title=f"{condition_key} – volcano plots",
            )
            plot_utils.save_multi(
                stem="volcano_grid",
                figdir=figdir_cond,
                fig=fig,
            )

    # Save
    out_zarr = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".zarr")
    LOGGER.info("Saving dataset → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if bool(getattr(cfg, "save_h5ad", False)):
        out_h5ad = output_dir / (str(getattr(cfg, "output_name", "adata.markers_and_de")) + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")

    LOGGER.info("Finished markers_and_de orchestrator")
    return adata
