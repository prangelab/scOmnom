# src/<your_package>/cluster_and_annotate.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from .config import ClusterAnnotateConfig
from .logging_utils import init_logging
from . import io_utils, plot_utils, reporting
from .plot_utils import _extract_series


from .clustering_utils import (
    CLUSTER_LABEL_KEY,
    _ensure_embedding,
    run_BISC,
    set_active_round,
    _next_round_index,
    _make_round_id,
    _res_key,
    _precompute_celltypist,
    _run_celltypist_annotation,

)
from .annotation_utils import (
    run_decoupler_for_round,
)
from .compaction_utils import create_compacted_round_from_parent_round

LOGGER = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Small orchestrator-only helpers (kept here)
# -------------------------------------------------------------------------
def _plot_round_clustering_diagnostics(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    *,
    embedding_key: str,
    batch_key: str | None,
) -> None:
    """ROUNDS-ONLY plotting for the active round."""
    if not cfg.make_figures:
        return

    active_round_id = adata.uns.get("active_cluster_round", None)
    active_round_id = str(active_round_id) if active_round_id else None
    rounds = adata.uns.get("cluster_rounds", {})
    if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
        LOGGER.warning("Plots: no active cluster round found; skipping clustering diagnostics.")
        return

    rinfo = rounds[active_round_id]
    sweep = rinfo.get("sweep", {}) if isinstance(rinfo.get("sweep", {}), dict) else {}
    diag = rinfo.get("diagnostics", {}) if isinstance(rinfo.get("diagnostics", {}), dict) else {}

    figdir_cluster = Path("cluster_and_annotate") / active_round_id / "clustering"

    tested_res = diag.get("tested_resolutions", sweep.get("resolutions", [])) or []
    res_sorted = sorted(float(r) for r in tested_res) if tested_res else []
    if not res_sorted:
        LOGGER.warning("Plots: no tested resolutions found in round '%s'.", active_round_id)
        return

    best_res = rinfo.get("best_resolution", None)

    sil_dict = diag.get("silhouette_centroid", {}) or {}
    n_dict = diag.get("cluster_counts", {}) or {}
    stab_dict = diag.get("resolution_stability", {}) or {}
    comp_dict = diag.get("composite_scores", {}) or {}
    tiny_dict = diag.get("tiny_cluster_penalty", {}) or {}

    sil_arr = _extract_series(res_sorted, sil_dict)
    n_arr = _extract_series(res_sorted, n_dict)

    pen_dict = diag.get("penalized_scores", None)
    if isinstance(pen_dict, dict) and pen_dict:
        pen_arr = _extract_series(res_sorted, pen_dict)
    else:
        alpha = float(getattr(cfg, "penalty_alpha", 0.0))
        pen_arr = np.array([float(s) - alpha * float(n) for s, n in zip(sil_arr, n_arr)], dtype=float)

    plot_utils.plot_clustering_resolution_sweep(
        resolutions=np.array(res_sorted, dtype=float),
        silhouette_scores=[float(x) for x in sil_arr],
        n_clusters=[int(round(x)) if np.isfinite(x) else 0 for x in n_arr],
        penalized_scores=[float(x) for x in pen_arr],
        figdir=figdir_cluster,
    )

    # Cluster tree (rebuild from sweep-added obs keys)
    labels_per_resolution: dict[str, np.ndarray] = {}
    for r in res_sorted:
        obs_key = f"{cfg.label_key}_{float(r):.2f}"
        if obs_key in adata.obs:
            labels_per_resolution[_res_key(r)] = adata.obs[obs_key].to_numpy()

    if len(labels_per_resolution) >= 2:
        plot_utils.plot_cluster_tree(
            labels_per_resolution=labels_per_resolution,
            resolutions=res_sorted,
            figdir=figdir_cluster,
            best_resolution=float(best_res) if best_res is not None else None,
        )

    plot_utils.plot_cluster_umaps(
        adata=adata,
        label_key=cfg.label_key,
        batch_key=batch_key,
        figdir=figdir_cluster,
    )

    stability = rinfo.get("stability", {}) if isinstance(rinfo.get("stability", {}), dict) else {}
    stability_aris = stability.get("subsampling_ari", []) or []
    plot_utils.plot_clustering_stability_ari(
        stability_aris=[float(x) for x in stability_aris],
        figdir=figdir_cluster,
    )

    if best_res is not None and res_sorted and stab_dict and comp_dict and tiny_dict:
        plateaus = sweep.get("plateaus", None)
        if isinstance(plateaus, str):
            try:
                plateaus = json.loads(plateaus)
            except Exception:
                plateaus = None

        plot_utils.plot_stability_curves(
            resolutions=res_sorted,
            silhouette=sil_dict,
            stability=stab_dict,
            composite=comp_dict,
            tiny_cluster_penalty=tiny_dict,
            best_resolution=float(best_res),
            plateaus=plateaus if isinstance(plateaus, list) else None,
            figdir=figdir_cluster,
        )

    # Bio metrics plot (if present)
    if (
        getattr(cfg, "bio_guided_clustering", False)
        and sweep.get("bio_homogeneity") is not None
        and sweep.get("bio_fragmentation") is not None
        and sweep.get("bio_ari") is not None
        and best_res is not None
        and res_sorted
    ):
        bh = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_homogeneity"])}
        bf = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_fragmentation"])}
        ba = {_res_key(r): float(v) for r, v in zip(res_sorted, sweep["bio_ari"])}

        plateaus = sweep.get("plateaus", None)
        if isinstance(plateaus, str):
            try:
                plateaus = json.loads(plateaus)
            except Exception:
                plateaus = None

        plot_utils.plot_biological_metrics(
            resolutions=res_sorted,
            bio_homogeneity=bh,
            bio_fragmentation=bf,
            bio_ari=ba,
            selection_config=sweep.get("selection_config", {}),
            best_resolution=float(best_res),
            plateaus=plateaus if isinstance(plateaus, list) else None,
            figdir=figdir_cluster,
            figure_formats=cfg.figure_formats,
        )


def _export_round_annotations_csv(adata: ad.AnnData, cfg: ClusterAnnotateConfig) -> None:
    if cfg.annotation_csv is None:
        return

    active_round_id = adata.uns.get("active_cluster_round", None)
    active_round_id = str(active_round_id) if active_round_id else None
    rounds = adata.uns.get("cluster_rounds", {})

    if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
        LOGGER.warning("annotation_csv requested but no active cluster round found; skipping export.")
        return

    rinfo = rounds[active_round_id]
    ann = rinfo.get("annotation", {}) if isinstance(rinfo.get("annotation", {}), dict) else {}
    cluster_key = rinfo.get("cluster_key", None)

    cols: list[str] = []
    if cluster_key and cluster_key in adata.obs:
        cols.append(str(cluster_key))

    ct_cluster_key = ann.get("celltypist_cluster_key", f"{cfg.celltypist_cluster_label_key}__{active_round_id}")
    pretty_key = ann.get("pretty_cluster_key", f"{CLUSTER_LABEL_KEY}__{active_round_id}")

    if ct_cluster_key in adata.obs:
        cols.append(ct_cluster_key)
    if pretty_key in adata.obs:
        cols.append(pretty_key)

    if not cols:
        LOGGER.warning("annotation_csv requested, but no annotation columns found for round '%s'.", active_round_id)
        return

    LOGGER.info("Exporting cluster annotations for round '%s' with columns: %s", active_round_id, cols)
    io_utils.export_cluster_annotations(adata, columns=cols, out_path=cfg.annotation_csv)


def _json_encode_round_plateaus_in_place(adata: ad.AnnData) -> None:
    """Make plateaus HDF5/Zarr-safe by JSON encoding under active round sweep."""
    active_round_id = adata.uns.get("active_cluster_round", None)
    active_round_id = str(active_round_id) if active_round_id else None
    rounds = adata.uns.get("cluster_rounds", {})

    if not active_round_id or not isinstance(rounds, dict) or active_round_id not in rounds:
        return

    rinfo = rounds[active_round_id]
    sweep = rinfo.get("sweep", None)
    if not isinstance(sweep, dict):
        return

    plateaus = sweep.get("plateaus", None)
    if isinstance(plateaus, list):
        sweep["plateaus"] = json.dumps(plateaus)
        rinfo["sweep"] = sweep
        rounds[active_round_id] = rinfo
        adata.uns["cluster_rounds"] = rounds


# -------------------------------------------------------------------------
# Public orchestrator
# -------------------------------------------------------------------------
def run_clustering(cfg: ClusterAnnotateConfig) -> ad.AnnData:
    """
    Full clustering + annotation pipeline (round-native):

    - Load integrated AnnData
    - Infer batch key + resolve embedding
    - Precompute CellTypist (once, optional)
    - neighbors/UMAP
    - BISC (round creation)
    - plots (round-aware)
    - CellTypist annotation (round-aware)
    - Decoupler (round-aware)
    - Optional compaction (new round)
    - Save outputs
    """
    init_logging(cfg.logfile)
    LOGGER.info("Starting cluster_and_annotate")

    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)
    adata = io_utils.load_dataset(cfg.input_path)

    batch_key = io_utils.infer_batch_key(adata, cfg.batch_key)
    cfg.batch_key = batch_key

    embedding_key = _ensure_embedding(adata, cfg.embedding_key)
    LOGGER.info("Using embedding_key='%s', batch_key='%s'", embedding_key, batch_key)

    # CellTypist precompute (must happen BEFORE BISC for bioARI)
    celltypist_labels, celltypist_proba = _precompute_celltypist(adata, cfg)

    # neighbors + UMAP
    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata)

    # --- BISC round ---
    run_BISC(
        adata,
        cfg,
        embedding_key=embedding_key,
        celltypist_labels=celltypist_labels,
        celltypist_proba=celltypist_proba,
        round_suffix="BISC",
    )

    if cfg.make_figures:
        _plot_round_clustering_diagnostics(adata, cfg, embedding_key=embedding_key, batch_key=batch_key)

    # ------------------------------------------------------------------
    # CellTypist annotation + pretty cluster labels (ROUND-AWARE)
    # ------------------------------------------------------------------
    active_round_id = adata.uns.get("active_cluster_round", None)
    active_round_id = str(active_round_id) if active_round_id else None

    cluster_key_for_annotation = cfg.label_key
    try:
        rounds = adata.uns.get("cluster_rounds", {})
        if active_round_id and isinstance(rounds, dict) and active_round_id in rounds:
            rk = rounds[active_round_id].get("cluster_key", None)
            if rk and rk in adata.obs:
                cluster_key_for_annotation = str(rk)
    except Exception:
        pass

    ann_keys = _run_celltypist_annotation(
        adata,
        cfg,
        cluster_key=cluster_key_for_annotation,
        round_id=active_round_id,
        precomputed_labels=celltypist_labels,
        precomputed_proba=celltypist_proba,
    )

    if cfg.make_figures and ann_keys is not None:
        figdir_cluster = Path("cluster_and_annotate")
        round_part = ann_keys.get("round_id", active_round_id) or "r0"
        figdir_ct = figdir_cluster / str(round_part) / "clustering"

        plot_utils.umap_by(adata, keys=ann_keys["celltypist_cell_key"], figdir=figdir_ct, stem="umap_celltypist_celllevel")
        plot_utils.umap_by(adata, keys=ann_keys["celltypist_cluster_key"], figdir=figdir_ct, stem="umap_celltypist_clusterlevel")
        plot_utils.umap_by(adata, keys=ann_keys["pretty_cluster_key"], figdir=figdir_ct, stem="umap_pretty_cluster_label")

        id_key = ann_keys["pretty_cluster_key"]
        plot_utils.plot_cluster_sizes(adata, id_key, figdir_ct)
        plot_utils.plot_cluster_qc_summary(adata, id_key, figdir_ct)
        plot_utils.plot_cluster_silhouette_by_cluster(adata, id_key, embedding_key, figdir_ct)
        if batch_key is not None:
            plot_utils.plot_cluster_batch_composition(adata, id_key, batch_key, figdir_ct)

    # legacy-ish pointers
    ca_uns = adata.uns.get("cluster_and_annotate", {})
    if ann_keys is not None:
        ca_uns["celltypist_label_key"] = ann_keys["celltypist_cell_key"]
        ca_uns["celltypist_cluster_label_key"] = ann_keys["celltypist_cluster_key"]
        ca_uns["cluster_label_key"] = ann_keys["pretty_cluster_key"]
        adata.uns["cluster_and_annotate"] = ca_uns

    # Optional CSV export (round-aware)
    _export_round_annotations_csv(adata, cfg)

    # ------------------------------------------------------------------
    # Decoupler (round-aware)
    # ------------------------------------------------------------------
    if getattr(cfg, "run_decoupler", False):
        run_decoupler_for_round(adata, cfg, round_id=None)  # uses active round
    else:
        LOGGER.info("Decoupler: disabled (run_decoupler=False).")

    if cfg.make_figures and getattr(cfg, "run_decoupler", False):
        active_round_id = adata.uns.get("active_cluster_round", None)
        active_round_id = str(active_round_id) if active_round_id else None
        rounds = adata.uns.get("cluster_rounds", {})
        if not active_round_id or active_round_id not in rounds:
            LOGGER.warning("Decoupler plots: no active cluster round found; skipping.")
        else:
            figdir_round = Path("cluster_and_annotate") / active_round_id

            if "msigdb" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="msigdb",
                    net_name=f"MSigDB",
                    figdir=figdir_round,
                    heatmap_top_k=30,
                    bar_top_n=10,
                    dotplot_top_k=30,
                )
            if "progeny" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="progeny",
                    net_name=f"PROGENy",
                    figdir=figdir_round,
                    heatmap_top_k=14,
                    bar_top_n=8,
                    dotplot_top_k=14,
                )
            if "dorothea" in adata.uns:
                plot_utils.plot_decoupler_all_styles(
                    adata,
                    net_key="dorothea",
                    net_name=f"DoRothEA",
                    figdir=figdir_round,
                    heatmap_top_k=40,
                    bar_top_n=10,
                    dotplot_top_k=35,
                )

    # ------------------------------------------------------------------
    # Compaction (new round)
    # ------------------------------------------------------------------
    if getattr(cfg, "enable_compacting", False):
        if not getattr(cfg, "run_decoupler", False):
            LOGGER.warning(
                "Compaction requested (enable_compacting=True) but run_decoupler=False. "
                "Compaction requires decoupler activities; skipping."
            )
        else:
            parent_round_id = adata.uns.get("active_cluster_round", None)
            parent_round_id = str(parent_round_id) if parent_round_id else None
            rounds = adata.uns.get("cluster_rounds", {})
            if not parent_round_id or not isinstance(rounds, dict) or parent_round_id not in rounds:
                LOGGER.warning("Compaction requested but no valid active parent round found; skipping.")
            else:
                # make a new round id
                new_round_id = _make_round_id(_next_round_index(adata), "compacted")

                # create compacted round (writes obs + registers round)
                create_compacted_round_from_parent_round(
                    adata,
                    cfg,
                    parent_round_id=parent_round_id,
                    new_round_id=new_round_id,
                    celltypist_obs_key=str(getattr(cfg, "celltypist_label_key", "")),
                    notes=f"Compacted from {parent_round_id}",
                    min_cells=int(getattr(cfg, "compact_min_cells", 0) or 0),
                    zscore_scope=str(
                        getattr(cfg, "compact_zscore_scope", "within_celltypist_label")
                        or "within_celltypist_label"
                    ),
                    grouping=str(
                        getattr(cfg, "compact_grouping", "connected_components")
                        or "connected_components"
                    ),
                    skip_unknown_celltypist_groups=bool(
                        getattr(cfg, "compact_skip_unknown_celltypist_groups", False)
                    ),
                    thr_progeny=float(getattr(cfg, "thr_progeny", 0.98) or 0.98),
                    thr_dorothea=float(getattr(cfg, "thr_dorothea", 0.98) or 0.98),
                    thr_msigdb_default=float(getattr(cfg, "thr_msigdb_default", 0.98) or 0.98),
                    thr_msigdb_by_gmt=getattr(cfg, "thr_msigdb_by_gmt", None),
                    msigdb_required=bool(getattr(cfg, "msigdb_required", True)),
                )

                # activate compacted round
                set_active_round(adata, new_round_id, publish_decoupler=False)

                # rebuild CT cluster labels + pretty labels for compacted round
                try:
                    rk = adata.uns["cluster_rounds"][new_round_id]["cluster_key"]
                    _run_celltypist_annotation(
                        adata,
                        cfg,
                        cluster_key=str(rk),
                        round_id=new_round_id,
                        precomputed_labels=celltypist_labels,
                        precomputed_proba=celltypist_proba,
                    )
                except Exception as e:
                    LOGGER.warning("Compaction: failed to rebuild CellTypist labels for '%s': %s", new_round_id, e)

                # decoupler for compacted round
                try:
                    run_decoupler_for_round(adata, cfg, round_id=new_round_id)
                except Exception as e:
                    LOGGER.warning("Compaction: decoupler failed for compacted round '%s': %s", new_round_id, e)

                # quick sanity log: did we publish resources?
                try:
                    LOGGER.info(
                        "Compaction[%s] post-decoupler: active_round=%r uns_has=%s",
                        new_round_id,
                        adata.uns.get("active_cluster_round", None),
                        {k: (k in adata.uns) for k in ("msigdb", "progeny", "dorothea")},
                    )
                except Exception:
                    pass

                # compacted round plots
                if cfg.make_figures:
                    # --- clustering/umap plots ---
                    try:
                        figdir_cluster = Path("cluster_and_annotate") / new_round_id / "clustering"
                        plot_utils.plot_cluster_umaps(
                            adata=adata,
                            label_key=str(adata.uns["cluster_rounds"][new_round_id]["cluster_key"]),
                            batch_key=batch_key,
                            figdir=figdir_cluster,
                        )
                        pretty_key = f"{CLUSTER_LABEL_KEY}__{new_round_id}"
                        if pretty_key in adata.obs:
                            plot_utils.umap_by(
                                adata,
                                keys=pretty_key,
                                figdir=figdir_cluster,
                                stem="umap_pretty_cluster_label",
                            )
                        plot_utils.plot_compaction_flow(
                            adata,
                            parent_round_id=parent_round_id,
                            child_round_id=new_round_id,
                            figdir=Path("cluster_and_annotate") / new_round_id / "clustering",
                            min_frac=0.02,
                        )
                    except Exception as e:
                        LOGGER.warning("Compaction: failed to plot compacted-round UMAPs: %s", e)

                    # --- decoupler plots for the compacted round ---
                    if getattr(cfg, "run_decoupler", False):
                        try:
                            figdir_round = Path("cluster_and_annotate") / new_round_id

                            if "msigdb" in adata.uns:
                                plot_utils.plot_decoupler_all_styles(
                                    adata,
                                    net_key="msigdb",
                                    net_name=f"MSigDB",
                                    figdir=figdir_round,
                                    heatmap_top_k=30,
                                    bar_top_n=10,
                                    dotplot_top_k=30,
                                )

                            if "progeny" in adata.uns:
                                plot_utils.plot_decoupler_all_styles(
                                    adata,
                                    net_key="progeny",
                                    net_name=f"PROGENy",
                                    figdir=figdir_round,
                                    heatmap_top_k=14,
                                    bar_top_n=8,
                                    dotplot_top_k=14,
                                )

                            if "dorothea" in adata.uns:
                                plot_utils.plot_decoupler_all_styles(
                                    adata,
                                    net_key="dorothea",
                                    net_name=f"DoRothEA",
                                    figdir=figdir_round,
                                    heatmap_top_k=40,
                                    bar_top_n=10,
                                    dotplot_top_k=35,
                                )

                        except Exception as e:
                            LOGGER.warning("Compaction: failed to plot decoupler for compacted round '%s': %s",
                                           new_round_id, e)

    else:
        LOGGER.info("Compaction: disabled (enable_compacting=False).")

    # Make plateaus HDF5/Zarr-safe
    _json_encode_round_plateaus_in_place(adata)

    if cfg.make_figures:
        try:
            reporting.generate_cluster_and_annotate_report(
                fig_root=Path(cfg.figdir) / "cluster_and_annotate",
                cfg=cfg,
                version=__version__,
                adata=adata,
            )
            LOGGER.info("Wrote cluster-and-annotate report.")
        except Exception as e:
            LOGGER.warning("Failed to generate cluster-and-annotate report: %s", e)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_zarr = cfg.resolved_output_dir / (cfg.output_name + ".zarr")
    LOGGER.info("Saving clustered/annotated dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if getattr(cfg, "save_h5ad", False):
        out_h5ad = cfg.resolved_output_dir / (cfg.output_name + ".h5ad")
        LOGGER.warning("Writing additional H5AD output (loads full matrix into RAM): %s", out_h5ad)
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
        LOGGER.info("Saved clustered/annotated H5AD → %s", out_h5ad)

    LOGGER.info("Finished cluster_and_annotate")
    return adata
