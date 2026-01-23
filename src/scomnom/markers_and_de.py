# src/scomnom/markers_and_de.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .logging_utils import init_logging

from .de_utils import (
    PseudobulkSpec,
    CellLevelMarkerSpec,
    PseudobulkDEOptions,
    compute_markers_celllevel,
    de_cluster_vs_rest_pseudobulk,
    de_condition_within_group_pseudobulk,
    resolve_group_key, de_condition_within_group_pseudobulk_multi,
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
    LOGGER.info("Starting markers-and-de...")

    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    figdir = output_dir / str(getattr(cfg, "figdir_name", "figures"))
    plot_utils.setup_scanpy_figs(figdir, getattr(cfg, "figure_formats", ["png", "pdf"]))

    adata = io_utils.load_dataset(getattr(cfg, "input_path"))

    # Resolve groupby key (round-aware by default)
    groupby = resolve_group_key(
        adata,
        groupby=getattr(cfg, "groupby", None),
        round_id=getattr(cfg, "round_id", None),
        prefer_pretty=(str(getattr(cfg, "label_source", "pretty")).lower() == "pretty"),
    )

    sample_key = getattr(cfg, "sample_key", None) or getattr(cfg, "batch_key", None) or adata.uns.get("batch_key")
    if sample_key is None:
        raise RuntimeError("markers_and_de: sample_key/batch_key missing (and adata.uns['batch_key'] not set).")

    LOGGER.info("markers_and_de: groupby=%r, sample_key=%r", groupby, sample_key)

    # Markers
    markers_spec = CellLevelMarkerSpec(
        method=str(getattr(cfg, "markers_method", "wilcoxon")),
        n_genes=int(getattr(cfg, "markers_n_genes", 100)),
        use_raw=bool(getattr(cfg, "markers_use_raw", False)),
        layer=getattr(cfg, "markers_layer", None),
        rankby_abs=bool(getattr(cfg, "markers_rankby_abs", True)),
        max_cells_per_group=int(getattr(cfg, "markers_downsample_max_per_group", 2000)),
        random_state=int(getattr(cfg, "random_state", 42)),
    )

    markers_key = str(getattr(cfg, "markers_key", "cluster_markers_wilcoxon"))
    compute_markers_celllevel(
        adata,
        groupby=groupby,
        round_id=getattr(cfg, "round_id", None),
        spec=markers_spec,
        key_added=markers_key,
        store=True,
    )

    # DE: cluster vs rest
    LOGGER.info(
        "Pseudobulk Markers: cluster-vs-rest "
        "(sample_key=%r, counts_layer=%r, min_cells_per_sample_group=%d, min_samples_per_level=%d)",
        pb_spec.sample_key,
        pb_spec.counts_layer,
        pb_opts.min_cells_per_sample_group,
        pb_opts.min_samples_per_level,
    )

    # Choose first available layer from cfg.counts_layers; else fall back to X if allowed.
    layer_candidates = list(getattr(cfg, "counts_layers", ["counts_cb", "counts_raw"]))
    counts_layer = None
    for layer in layer_candidates:
        if layer in adata.layers:
            counts_layer = layer
            break

    if counts_layer is None and not bool(getattr(cfg, "allow_X_counts", True)):
        raise RuntimeError(
            f"markers_and_de: none of counts_layers found in adata.layers: {layer_candidates}, and allow_X_counts=False"
        )

    pb_spec = PseudobulkSpec(
        sample_key=str(sample_key),
        counts_layer=counts_layer,  # can be None -> uses adata.X
    )

    pb_opts = PseudobulkDEOptions(
        min_cells_per_sample_group=int(getattr(cfg, "min_cells_target", 20)),
        min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
        alpha=float(getattr(cfg, "alpha", 0.05)),
        shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
    )

    marker_genes_all_clusters = de_cluster_vs_rest_pseudobulk(
        adata,
        groupby=groupby,
        round_id=getattr(cfg, "round_id", None),
        spec=pb_spec,
        opts=pb_opts,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        store=True,
        n_cpus=int(getattr(cfg, "n_jobs", 1)),
    )

    # Optional condition DE
    condition_key = getattr(cfg, "condition_key", None)
    if condition_key:

        # per-cluster condition DE
        groups = pd.Index(pd.unique(adata.obs[groupby].astype(str))).sort_values()
        conditional_de_genes_all_clusters = {}

        cond_opts = PseudobulkDEOptions(
            min_cells_per_sample_group=int(getattr(cfg, "min_cells_condition", 20)),
            min_samples_per_level=int(getattr(cfg, "min_samples_per_level", 2)),
            alpha=float(getattr(cfg, "alpha", 0.05)),
            shrink_lfc=bool(getattr(cfg, "shrink_lfc", True)),
        )

        condition_contrasts = list(getattr(cfg, "condition_contrasts", [])) or None

        for g in groups:
            _ = de_condition_within_group_pseudobulk_multi(
                adata,
                group_value=str(g),
                groupby=groupby,
                round_id=getattr(cfg, "round_id", None),
                condition_key=str(condition_key),
                spec=pb_spec,
                opts=cond_opts,
                contrasts=condition_contrasts,
                store_key=str(getattr(cfg, "store_key", "scomnom_de")),
                store=True,
                n_cpus=int(getattr(cfg, "n_jobs", 1)),
            )
    else:
        conditional_de_genes_all_clusters = None

    # Minimal provenance
    adata.uns.setdefault("markers_and_de", {})
    adata.uns["markers_and_de"].update(
        {
            "version": __version__,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "groupby": str(groupby),
            "sample_key": str(sample_key),
            "markers_key": str(markers_key),
            "counts_layers_candidates": list(layer_candidates),
            "counts_layer_used": counts_layer,
            "alpha": float(getattr(cfg, "alpha", 0.05)),
            "tables": {
                "cluster_vs_rest_clusters": sorted(list(marker_genes_all_clusters.keys()))
                if hasattr(marker_genes_all_clusters, "to_dict")
                else None,
                "condition_within_cluster_clusters": sorted(
                list(conditional_de_genes_all_clusters.keys())) if conditional_de_genes_all_clusters else None
                if (conditional_de_genes_all_clusters is not None and hasattr(conditional_de_genes_all_clusters, "to_dict"))
                else None,
            },
        }
    )

    # ------------------------------------------------------------------
    # Write tables
    # ------------------------------------------------------------------
    # Markers: cell-level rank_genes_groups -> CSVs
    io_utils.export_rank_genes_groups_tables(
        adata,
        key_added=markers_key,
        output_dir=output_dir,
        groupby=str(groupby),
        prefix="celllevel_markers",
    )

    # Markers: cell-level rank_genes_groups -> XLSX
    io_utils.export_rank_genes_groups_excel(
        adata,
        key_added=markers_key,
        output_dir=output_dir,
        groupby=str(groupby),
        filename="celllevel_markers.xlsx",
        max_genes=int(getattr(cfg, "markers_n_genes", 100)),
    )

    # Pseudobulk cluster-vs-rest: CSV folder
    io_utils.export_pseudobulk_de_tables(
        adata,
        output_dir=output_dir,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        groupby=str(groupby),
        condition_key=None,
    )

    # Pseudobulk cluster-vs-rest: single XLSX
    io_utils.export_pseudobulk_cluster_vs_rest_excel(
        adata,
        output_dir=output_dir,
        store_key=str(getattr(cfg, "store_key", "scomnom_de")),
    )

    if condition_key:
        # Condition within cluster: CSV folder
        io_utils.export_pseudobulk_de_tables(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            groupby=str(groupby),
            condition_key=str(condition_key),
        )

        # Condition within cluster: single XLSX
        io_utils.export_pseudobulk_condition_within_cluster_excel(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            condition_key=str(condition_key),
        )

    # ------------------------------------------------------------------
    # Contrast-conditional mode (pairwise A vs B within each cluster)
    # ------------------------------------------------------------------
    if bool(getattr(cfg, "contrast_conditional_de", False)):
        from .de_utils import ContrastConditionalSpec, contrast_conditional_markers

        contrast_key = getattr(cfg, "contrast_key", None) or str(sample_key)
        cc_spec = ContrastConditionalSpec(
            contrast_key=str(contrast_key),
            methods=tuple(getattr(cfg, "contrast_methods", ("wilcoxon", "logreg"))),
            min_cells_per_level_in_cluster=int(getattr(cfg, "contrast_min_cells_per_level", 50)),
            max_cells_per_level_in_cluster=int(getattr(cfg, "contrast_max_cells_per_level", 2000)),
            min_total_counts=int(getattr(cfg, "contrast_min_total_counts", 10)),
            pseudocount=float(getattr(cfg, "contrast_pseudocount", 1.0)),
            cl_alpha=float(getattr(cfg, "contrast_cl_alpha", 0.05)),
            cl_min_abs_logfc=float(getattr(cfg, "contrast_cl_min_abs_logfc", 0.25)),
            lr_min_abs_coef=float(getattr(cfg, "contrast_lr_min_abs_coef", 0.25)),
            pb_min_abs_log2fc=float(getattr(cfg, "contrast_pb_min_abs_log2fc", 0.5)),
            random_state=int(getattr(cfg, "random_state", 42)),
        )

        _ = contrast_conditional_markers(
            adata,
            groupby=str(groupby),
            round_id=getattr(cfg, "round_id", None),
            spec=cc_spec,
            pb_spec=pb_spec,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
            store=True,
        )

        # write tables (CSV + XLSX + summary)
        io_utils.export_contrast_conditional_markers_tables(
            adata,
            output_dir=output_dir,
            store_key=str(getattr(cfg, "store_key", "scomnom_de")),
        )

    # ------------------------------------------------------------------
    # Plotting (CLI behavior: create figs + save via save_multi)
    # ------------------------------------------------------------------
    from . import de_plot_utils

    if bool(getattr(cfg, "make_figures", True)):
        store_key = str(getattr(cfg, "store_key", "scomnom_de"))
        alpha = float(getattr(cfg, "alpha", 0.05))
        lfc_thresh = float(getattr(cfg, "plot_lfc_thresh", 1.0))
        top_label_n = int(getattr(cfg, "plot_volcano_top_label_n", 15))

        # how many genes to include in expression plots
        top_n_per_cluster = int(getattr(cfg, "plot_top_n_per_cluster", 10))
        max_genes_total = int(getattr(cfg, "plot_max_genes_total", 80))

        use_raw = bool(getattr(cfg, "plot_use_raw", False))
        layer = getattr(cfg, "plot_layer", None)

        # Cap UMAP grid to 3x3 (9 genes) regardless of ncols knob
        ncols = 3

        # IMPORTANT: save_multi expects figdir to be RELATIVE and start with the run key.
        # Do NOT prefix with output_dir or "figures" here.
        figroot = Path("markers_and_de")

        # ---------------------------
        # helpers
        # ---------------------------
        def _select_top_genes(df, *, padj_thresh: float, top_n: int) -> list[str]:
            if df is None or getattr(df, "empty", True):
                return []
            if "gene" not in df.columns or "padj" not in df.columns or "log2FoldChange" not in df.columns:
                return []
            tmp = df[["gene", "padj", "log2FoldChange"]].copy()
            tmp["gene"] = tmp["gene"].astype(str)
            tmp["padj"] = pd.to_numeric(tmp["padj"], errors="coerce")
            tmp["log2FoldChange"] = pd.to_numeric(tmp["log2FoldChange"], errors="coerce")
            tmp = tmp.dropna(subset=["gene", "padj", "log2FoldChange"])
            tmp = tmp[tmp["padj"] < float(padj_thresh)]
            if tmp.empty:
                return []
            tmp["__abs_lfc"] = tmp["log2FoldChange"].abs()
            tmp = tmp.sort_values(["padj", "__abs_lfc"], ascending=[True, False])
            genes = tmp["gene"].head(int(top_n)).tolist()
            # unique preserve order
            out = []
            seen = set()
            for g in genes:
                if g and g not in seen:
                    out.append(g)
                    seen.add(g)
            return out

        def _append_unique(acc: list[str], genes: list[str], limit: int) -> list[str]:
            seen = set(acc)
            for g in genes:
                if g not in seen:
                    acc.append(g)
                    seen.add(g)
                if len(acc) >= limit:
                    break
            return acc

        # ---------------------------
        # 1) Cluster-vs-rest volcanoes + gene list
        # ---------------------------
        pb = adata.uns.get(store_key, {}).get("pseudobulk_cluster_vs_rest", {})
        results_by_cluster = pb.get("results", {}) if isinstance(pb, dict) else {}
        if not isinstance(results_by_cluster, dict):
            results_by_cluster = {}

        genes_for_expression_plots: list[str] = []

        if results_by_cluster:
            figdir_pb = figroot / "cluster_vs_rest"

            for cl, df_de in results_by_cluster.items():
                title = f"Cluster vs rest: {cl}"
                fig = de_plot_utils.volcano(
                    df_de,
                    padj_thresh=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    title=title,
                    show=False,
                )
                plot_utils.save_multi(
                    stem=f"volcano__cluster_vs_rest__{cl}",
                    figdir=figdir_pb,
                    fig=fig,
                )

                top_genes = _select_top_genes(df_de, padj_thresh=alpha, top_n=top_n_per_cluster)
                genes_for_expression_plots = _append_unique(
                    genes_for_expression_plots,
                    top_genes,
                    limit=max_genes_total,
                )
        else:
            LOGGER.warning("markers_and_de: no pseudobulk_cluster_vs_rest results found under adata.uns[%r].",
                           store_key)

        # If we collected genes, make scanpy expression plots once
        if genes_for_expression_plots:
            figdir_expr = figroot / "cluster_vs_rest" / "expression"

            fig = de_plot_utils.dotplot_top_genes(
                adata,
                genes=genes_for_expression_plots,
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem="dotplot__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            fig = de_plot_utils.heatmap_top_genes(
                adata,
                genes=genes_for_expression_plots,
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem="heatmap__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            # HARD CAP: 3x3 grid max (9 genes)
            umap_genes = genes_for_expression_plots[:9]
            fig = de_plot_utils.umap_features_grid(
                adata,
                genes=umap_genes,
                use_raw=use_raw,
                layer=layer,
                ncols=ncols,
                show=False,
            )
            plot_utils.save_multi(
                stem="umap_features__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

            fig = de_plot_utils.violin_genes(
                adata,
                genes=genes_for_expression_plots[: min(len(genes_for_expression_plots), 24)],
                groupby=str(groupby),
                use_raw=use_raw,
                layer=layer,
                show=False,
            )
            plot_utils.save_multi(
                stem="violin__cluster_vs_rest__topgenes",
                figdir=figdir_expr,
                fig=fig,
            )

        # ---------------------------
        # 2) Condition-within-cluster plots (if present)
        # ---------------------------
        cond_block = adata.uns.get(store_key, {}).get("pseudobulk_condition_within_group", {})
        if isinstance(cond_block, dict) and cond_block and condition_key:
            figdir_cond = figroot / f"condition_within_cluster__{condition_key}"

            # One volcano per stored contrast key
            for k, payload in cond_block.items():
                if not isinstance(payload, dict):
                    continue
                df_de = payload.get("results", None)
                if df_de is None:
                    continue

                title = str(k)
                fig = de_plot_utils.volcano(
                    df_de,
                    padj_thresh=alpha,
                    lfc_thresh=lfc_thresh,
                    top_label_n=top_label_n,
                    title=title,
                    show=False,
                )
                plot_utils.save_multi(
                    stem=f"volcano__condition_within_cluster__{k}",
                    figdir=figdir_cond,
                    fig=fig,
                )
        elif condition_key:
            LOGGER.info(
                "markers_and_de: no pseudobulk_condition_within_group results found for condition_key=%r.",
                condition_key,
            )

        # ------------------------------------------------------------------
        # Contrast-conditional plots (pairwise A vs B within each cluster)
        # ------------------------------------------------------------------
        if bool(getattr(cfg, "contrast_conditional_de", False)):
            store_key = str(getattr(cfg, "store_key", "scomnom_de"))
            figroot_cc = figroot / "contrast_conditional"

            # plotting knobs (safe defaults)
            alpha_cc = float(getattr(cfg, "contrast_cl_alpha", 0.05))
            lfc_thresh_cc = float(getattr(cfg, "plot_lfc_thresh", 1.0))  # reuse your existing knob
            top_label_n_cc = int(getattr(cfg, "plot_volcano_top_label_n", 15))

            top_n_genes = int(getattr(cfg, "contrast_plot_top_n_genes", 12))
            max_cells_per_level_plot = int(getattr(cfg, "contrast_plot_max_cells_per_level", 600))
            random_state = int(getattr(cfg, "random_state", 42))

            # optional global guards (avoid 5000 plots on giant runs)
            max_pairs_to_plot = int(getattr(cfg, "contrast_plot_max_pairs", 2000))
            max_clusters_to_plot = int(getattr(cfg, "contrast_plot_max_clusters", 2000))

            # cap UMAP grid here too
            ncols = 3

            cc = adata.uns.get(store_key, {}).get("contrast_conditional", {})
            cc_results = cc.get("results", {}) if isinstance(cc, dict) else {}
            contrast_key = cc.get("contrast_key", None) if isinstance(cc, dict) else None

            if not isinstance(cc_results, dict) or not cc_results:
                LOGGER.warning(
                    "markers_and_de: contrast_conditional enabled but no results found under adata.uns[%r].",
                    store_key,
                )
            else:
                if not contrast_key:
                    # fallback: use sample_key
                    contrast_key = str(sample_key)

                if contrast_key not in adata.obs:
                    raise KeyError(f"contrast_conditional plotting: contrast_key={contrast_key!r} not in adata.obs")

                # -------- helpers --------
                def _pick_genes_from_combined(df_comb: pd.DataFrame, n: int) -> list[str]:
                    """
                    Prefer Tier1, then Tier2, then Tier3.
                    Within tier: consensus_score desc, then cl_padj asc, then |pb_log2fc| desc.
                    """
                    if df_comb is None or getattr(df_comb, "empty", True):
                        return []
                    if "gene" not in df_comb.columns:
                        return []

                    tmp = df_comb.copy()
                    tmp["gene"] = tmp["gene"].astype(str)

                    # normalize columns if present
                    if "consensus_score" in tmp.columns:
                        tmp["consensus_score"] = pd.to_numeric(tmp["consensus_score"], errors="coerce")
                    else:
                        tmp["consensus_score"] = np.nan

                    if "cl_padj" in tmp.columns:
                        tmp["cl_padj"] = pd.to_numeric(tmp["cl_padj"], errors="coerce")
                    else:
                        tmp["cl_padj"] = np.nan

                    if "pb_log2fc" in tmp.columns:
                        tmp["pb_log2fc"] = pd.to_numeric(tmp["pb_log2fc"], errors="coerce")
                        tmp["__abs_pb"] = tmp["pb_log2fc"].abs()
                    else:
                        tmp["__abs_pb"] = np.nan

                    tiers = ["Tier1", "Tier2", "Tier3"]
                    out: list[str] = []
                    seen: set[str] = set()

                    for t in tiers:
                        if "consensus_tier" in tmp.columns:
                            sub = tmp[tmp["consensus_tier"].astype(str) == t].copy()
                        else:
                            sub = tmp.copy() if t == "Tier3" else tmp.iloc[0:0].copy()

                        if sub.empty:
                            continue

                        sub = sub.sort_values(
                            ["consensus_score", "cl_padj", "__abs_pb"],
                            ascending=[False, True, False],
                            na_position="last",
                        )

                        for g in sub["gene"].tolist():
                            if not g or g in seen:
                                continue
                            out.append(g)
                            seen.add(g)
                            if len(out) >= int(n):
                                return out

                    return out[: int(n)]

                def _downsample_cluster_pair_indices(
                        *,
                        cluster_key: str,
                        cluster_value: str,
                        contrast_key: str,
                        A: str,
                        B: str,
                        max_per_level: int,
                        random_state: int,
                ) -> np.ndarray:
                    """
                    Stratified downsample within (cluster_value) for contrast levels A and B.
                    Returns global indices to subset adata for plotting.
                    """
                    rng = np.random.default_rng(int(random_state))

                    cl_mask = (adata.obs[cluster_key].astype(str).to_numpy() == str(cluster_value))
                    if cl_mask.sum() == 0:
                        return np.array([], dtype=int)

                    idx_cl = np.where(cl_mask)[0]
                    lv = adata.obs.iloc[idx_cl][contrast_key].astype(str).to_numpy()

                    idxA = idx_cl[np.where(lv == str(A))[0]]
                    idxB = idx_cl[np.where(lv == str(B))[0]]

                    if idxA.size == 0 or idxB.size == 0:
                        return np.array([], dtype=int)

                    if max_per_level > 0 and idxA.size > max_per_level:
                        idxA = rng.choice(idxA, size=int(max_per_level), replace=False)
                    if max_per_level > 0 and idxB.size > max_per_level:
                        idxB = rng.choice(idxB, size=int(max_per_level), replace=False)

                    idx = np.concatenate([idxA, idxB])
                    rng.shuffle(idx)
                    return idx

                # iterate (with guards)
                n_pairs_done = 0
                n_clusters_done = 0

                for cl, pairs in cc_results.items():
                    if n_clusters_done >= max_clusters_to_plot:
                        break
                    if not isinstance(pairs, dict) or not pairs:
                        continue

                    n_clusters_done += 1

                    for pair_key, payload in pairs.items():
                        if n_pairs_done >= max_pairs_to_plot:
                            break
                        if not isinstance(payload, dict):
                            continue

                        # parse A/B from pair_key "A_vs_B"
                        if isinstance(pair_key, str) and "_vs_" in pair_key:
                            A, B = pair_key.split("_vs_", 1)
                        else:
                            continue

                        df_w = payload.get("wilcoxon", None)
                        df_c = payload.get("combined", None)

                        # 1) volcano from Wilcoxon (cl_padj + cl_logfc)
                        fig = de_plot_utils.volcano(
                            df_w if isinstance(df_w, pd.DataFrame) else pd.DataFrame(),
                            gene_col="gene",
                            padj_col="cl_padj",
                            lfc_col="cl_logfc",
                            padj_thresh=float(alpha_cc),
                            lfc_thresh=float(lfc_thresh_cc),
                            top_label_n=int(top_label_n_cc),
                            title=f"{cl} :: {A} vs {B} ({contrast_key})",
                            show=False,
                        )
                        plot_utils.save_multi(
                            stem=f"volcano__contrast_conditional__{cl}__{A}_vs_{B}",
                            figdir=figroot_cc / "volcano",
                            fig=fig,
                        )

                        # 2) expression plots for consensus genes (downsample within cluster+pair)
                        genes = _pick_genes_from_combined(df_c, n=top_n_genes) if isinstance(df_c, pd.DataFrame) else []
                        if genes:
                            idx_plot = _downsample_cluster_pair_indices(
                                cluster_key=str(groupby),
                                cluster_value=str(cl),
                                contrast_key=str(contrast_key),
                                A=str(A),
                                B=str(B),
                                max_per_level=int(max_cells_per_level_plot),
                                random_state=int(random_state),
                            )
                            if idx_plot.size > 0:
                                adata_plot = adata[idx_plot].copy()

                                # keep only A/B for clean groupby ordering
                                adata_plot = adata_plot[
                                    adata_plot.obs[contrast_key].astype(str).isin([str(A), str(B)])
                                ].copy()

                                figdir_expr = figroot_cc / "expression" / f"{cl}__{A}_vs_{B}"

                                fig = de_plot_utils.dotplot_top_genes(
                                    adata_plot,
                                    genes=genes,
                                    groupby=str(contrast_key),
                                    use_raw=use_raw,
                                    layer=layer,
                                    show=False,
                                )
                                plot_utils.save_multi(
                                    stem=f"dotplot__{cl}__{A}_vs_{B}",
                                    figdir=figdir_expr,
                                    fig=fig,
                                )

                                fig = de_plot_utils.heatmap_top_genes(
                                    adata_plot,
                                    genes=genes,
                                    groupby=str(contrast_key),
                                    use_raw=use_raw,
                                    layer=layer,
                                    show=False,
                                )
                                plot_utils.save_multi(
                                    stem=f"heatmap__{cl}__{A}_vs_{B}",
                                    figdir=figdir_expr,
                                    fig=fig,
                                )

                                # UMAP feature grid only if UMAP exists; cap to 3x3 visually
                                if "X_umap" in adata_plot.obsm:
                                    fig = de_plot_utils.umap_features_grid(
                                        adata_plot,
                                        genes=genes[:9],
                                        use_raw=use_raw,
                                        layer=layer,
                                        ncols=int(ncols),
                                        show=False,
                                    )
                                    plot_utils.save_multi(
                                        stem=f"umap_features__{cl}__{A}_vs_{B}",
                                        figdir=figdir_expr,
                                        fig=fig,
                                    )

                                fig = de_plot_utils.violin_genes(
                                    adata_plot,
                                    genes=genes,
                                    groupby=str(contrast_key),
                                    use_raw=use_raw,
                                    layer=layer,
                                    show=False,
                                )
                                plot_utils.save_multi(
                                    stem=f"violin__{cl}__{A}_vs_{B}",
                                    figdir=figdir_expr,
                                    fig=fig,
                                )

                        n_pairs_done += 1

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
