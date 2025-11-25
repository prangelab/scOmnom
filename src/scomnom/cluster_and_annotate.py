from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .config import ClusterAnnotateConfig
from .load_and_filter import setup_logging
from . import io_utils, plot_utils
from .io_utils import get_celltypist_model


LOGGER = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _ensure_embedding(adata: ad.AnnData, embedding_key: str) -> str:
    """
    Ensure the chosen embedding exists; if not, try to recover from integration metadata.
    Returns the actual embedding key to use.
    """
    if embedding_key in adata.obsm:
        return embedding_key

    # Try integration metadata if present
    if "integration" in adata.uns:
        best = adata.uns["integration"].get("best_embedding")
        if best and best in adata.obsm:
            LOGGER.warning(
                "Embedding key '%s' not found. Falling back to integration best_embedding='%s'.",
                embedding_key,
                best,
            )
            return best

    raise KeyError(
        f"Embedding key '{embedding_key}' not found in adata.obsm and no usable fallback found."
    )


def _compute_resolutions(cfg: ClusterAnnotateConfig) -> np.ndarray:
    return np.linspace(cfg.res_min, cfg.res_max, cfg.n_resolutions, endpoint=True)


def _resolution_sweep(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
) -> Tuple[float, Dict[str, object]]:
    """
    Sweep over a range of Leiden resolutions and compute:
    - Silhouette score in the chosen embedding
    - Number of clusters
    - Penalized score = silhouette - alpha * n_clusters
    - ARI matrix between all resolutions (Clustree-style)
    """
    resolutions = _compute_resolutions(cfg)
    clusterings: Dict[float, np.ndarray] = {}
    silhouette_scores: List[float] = []
    n_clusters_list: List[int] = []
    penalized_scores: List[float] = []

    # Sweep resolutions
    for res in resolutions:
        key = f"{cfg.label_key}_{res:.2f}"
        LOGGER.info("Running Leiden clustering at resolution %.2f -> key '%s'", res, key)
        sc.tl.leiden(
            adata,
            resolution=float(res),
            key_added=key,
            random_state=cfg.random_state,
            flavor="igraph",
        )
        labels = adata.obs[key].to_numpy()
        clusterings[res] = labels

        n_clusters = int(np.unique(labels).size)
        n_clusters_list.append(n_clusters)

        if n_clusters <= 1:
            sil = -1.0
        else:
            sil = float(silhouette_score(adata.obsm[embedding_key], labels))

        silhouette_scores.append(sil)
        penalized_scores.append(sil - cfg.penalty_alpha * n_clusters)

        LOGGER.info(
            "Resolution %.2f: %d clusters, silhouette=%.3f, penalized=%.3f",
            res,
            n_clusters,
            sil,
            penalized_scores[-1],
        )

    # ARI across all resolutions
    col_names = [f"{r:.2f}" for r in resolutions]
    ari_matrix = pd.DataFrame(index=col_names, columns=col_names, dtype=float)

    for i, r1 in enumerate(resolutions):
        for j, r2 in enumerate(resolutions):
            ari = adjusted_rand_score(clusterings[r1], clusterings[r2])
            ari_matrix.iat[i, j] = float(ari)

    penalized_scores_arr = np.asarray(penalized_scores, dtype=float)
    best_idx = int(np.argmax(penalized_scores_arr))
    best_res = float(resolutions[best_idx])

    LOGGER.info(
        "Selected optimal resolution %.2f (penalized silhouette=%.3f)",
        best_res,
        penalized_scores_arr[best_idx],
    )

    sweep = {
        "resolutions": resolutions,
        "silhouette_scores": silhouette_scores,
        "n_clusters": n_clusters_list,
        "penalized_scores": penalized_scores,
        "ari_matrix": ari_matrix,
    }
    return best_res, sweep


def _subsampling_stability(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    embedding_key: str,
    best_res: float,
) -> List[float]:
    """
    Subsampling stability analysis:
    - Cluster full data at best_res (reference)
    - Repeat subsampling of cells and recompute clustering
    - Compute ARI vs reference clustering on overlapping cells
    """
    ref_key = f"{cfg.label_key}_stab_ref"
    LOGGER.info("Computing reference clustering for stability at resolution %.2f", best_res)
    sc.tl.leiden(
        adata,
        resolution=float(best_res),
        key_added=ref_key,
        random_state=cfg.random_state,
        flavor="igraph",
    )
    ref_labels = adata.obs[ref_key].copy()

    rng = np.random.default_rng(cfg.random_state)
    stability_aris: List[float] = []

    for i in range(cfg.stability_repeats):
        rng_i = np.random.default_rng(cfg.random_state + i)
        n_sub = int(round(cfg.subsample_frac * adata.n_obs))
        cells = rng_i.choice(adata.obs_names.to_numpy(), size=n_sub, replace=False)
        sub = adata[cells].copy()

        LOGGER.info("Stability repeat %d/%d: %d cells", i + 1, cfg.stability_repeats, n_sub)

        sc.pp.neighbors(sub, use_rep=embedding_key)
        sc.tl.leiden(
            sub,
            resolution=float(best_res),
            key_added=f"{cfg.label_key}_sub",
            random_state=cfg.random_state + i,
            flavor="igraph",
        )

        overlap = adata.obs_names.intersection(sub.obs_names)
        ari = adjusted_rand_score(
            ref_labels.loc[overlap],
            sub.obs.loc[overlap, f"{cfg.label_key}_sub"],
        )
        stability_aris.append(float(ari))

    mean_ari = float(np.mean(stability_aris)) if stability_aris else float("nan")
    LOGGER.info(
        "Subsampling stability: mean ARI over %d repeats = %.3f",
        cfg.stability_repeats,
        mean_ari,
    )
    return stability_aris


def _apply_final_clustering(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
    best_res: float,
) -> None:
    """
    Apply the chosen resolution as the final clustering into cfg.label_key
    and set a consistent color palette.
    """
    LOGGER.info(
        "Applying final Leiden clustering at resolution %.2f -> key '%s'",
        best_res,
        cfg.label_key,
    )
    sc.tl.leiden(
        adata,
        resolution=float(best_res),
        key_added=cfg.label_key,
        random_state=cfg.random_state,
        flavor="igraph",
    )

    # Assign palette
    try:
        from scanpy.plotting.palettes import default_102

        cats = adata.obs[cfg.label_key].cat.categories
        adata.uns[f"{cfg.label_key}_colors"] = default_102[: len(cats)]
    except Exception as e:
        LOGGER.warning("Could not set Leiden color palette: %s", e)


def _run_celltypist_annotation(
    adata: ad.AnnData,
    cfg: ClusterAnnotateConfig,
) -> str | None:
    """
    Run CellTypist on the dataset if cfg.celltypist_model is provided.
    Returns the column name containing the primary annotation, or None if skipped.
    """
    if cfg.celltypist_model is None:
        LOGGER.info("No CellTypist model provided; skipping annotation.")
        return None

    LOGGER.info("Normalizing expression for CellTypist annotation...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Resolve model: local path → OK, cache → OK, remote → download
    LOGGER.info("Resolving CellTypist model: %s", cfg.celltypist_model)
    model_path = get_celltypist_model(cfg.celltypist_model)

    from celltypist.models import Model
    import celltypist

    LOGGER.info("Loading CellTypist model from %s", model_path)
    model = Model.load(str(model_path))

    LOGGER.info("Running CellTypist annotation (majority_voting=%s)...",
                cfg.celltypist_majority_voting)

    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=cfg.celltypist_majority_voting,
    )

    # Prefer majority voting labels if present
    if "majority_voting" in predictions.predicted_labels:
        labels = predictions.predicted_labels["majority_voting"]
    else:
        labels = predictions.predicted_labels

    adata.obs[cfg.celltypist_label_key] = labels
    adata.obs[cfg.final_label_key] = adata.obs[cfg.celltypist_label_key]

    LOGGER.info(
        "Added CellTypist labels to adata.obs['%s'] and final labels to adata.obs['%s']",
        cfg.celltypist_label_key,
        cfg.final_label_key,
    )
    return cfg.celltypist_label_key


# -------------------------------------------------------------------------
# Public orchestrator
# -------------------------------------------------------------------------
def run_clustering(cfg: ClusterAnnotateConfig) -> ad.AnnData:
    """
    Full clustering + annotation pipeline:

    - Load integrated AnnData
    - Infer batch key (if needed)
    - Build neighbors/UMAP from chosen embedding
    - Resolution sweep (silhouette, penalized score, ARI matrix)
    - Subsampling stability analysis
    - Apply final clustering at optimal resolution
    - Optional CellTypist annotation
    - Generate plots via plot_utils
    - Save final h5ad and optional annotation CSV
    """
    setup_logging(cfg.logfile)
    LOGGER.info("Starting cluster_and_annotate")

    # ------------------------------------------------------------------
    # Load data and keys
    # ------------------------------------------------------------------
    in_path: Path = cfg.input_path
    adata = sc.read_h5ad(str(in_path))

    batch_key = io_utils.infer_batch_key(adata, cfg.batch_key)
    cfg.batch_key = batch_key

    embedding_key = _ensure_embedding(adata, cfg.embedding_key)
    LOGGER.info("Using embedding_key='%s', batch_key='%s'", embedding_key, batch_key)

    # ------------------------------------------------------------------
    # Neighbors + UMAP (using chosen embedding)
    # ------------------------------------------------------------------
    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata)

    # ------------------------------------------------------------------
    # Setup figures
    # ------------------------------------------------------------------
    if cfg.make_figures:
        plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)
    figdir_cluster = cfg.figdir / "cluster_and_annotate"

    # ------------------------------------------------------------------
    # Resolution sweep
    # ------------------------------------------------------------------
    best_res, sweep = _resolution_sweep(adata, cfg, embedding_key)

    if cfg.make_figures:
        plot_utils.plot_clustering_resolution_sweep(
            resolutions=sweep["resolutions"],
            silhouette_scores=sweep["silhouette_scores"],
            n_clusters=sweep["n_clusters"],
            penalized_scores=sweep["penalized_scores"],
            figdir=figdir_cluster,
        )
        plot_utils.plot_clustering_ari_heatmap(
            ari_matrix=sweep["ari_matrix"],
            figdir=figdir_cluster,
        )

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------
    stability_aris = _subsampling_stability(adata, cfg, embedding_key, best_res)
    if cfg.make_figures:
        plot_utils.plot_clustering_stability_ari(
            stability_aris=stability_aris,
            figdir=figdir_cluster,
        )

    # ------------------------------------------------------------------
    # Apply final clustering
    # ------------------------------------------------------------------
    _apply_final_clustering(adata, cfg, best_res)

    # UMAPs by cluster / batch
    if cfg.make_figures:
        plot_utils.plot_cluster_umaps(
            adata=adata,
            label_key=cfg.label_key,
            batch_key=batch_key,
            figdir=figdir_cluster,
        )

    # ------------------------------------------------------------------
    # CellTypist annotation (optional)
    # ------------------------------------------------------------------
    annotation_col = _run_celltypist_annotation(adata, cfg)
    if cfg.make_figures and annotation_col is not None:
        plot_utils.umap_by(adata, keys=annotation_col)

    # ------------------------------------------------------------------
    # Store metadata
    # ------------------------------------------------------------------
    adata.uns.setdefault("cluster_and_annotate", {})
    adata.uns["cluster_and_annotate"].update(
        {
            "embedding_key": embedding_key,
            "batch_key": batch_key,
            "label_key": cfg.label_key,
            "best_resolution": float(best_res),
            "resolutions": [float(r) for r in sweep["resolutions"]],
            "silhouette_scores": [float(x) for x in sweep["silhouette_scores"]],
            "n_clusters": [int(x) for x in sweep["n_clusters"]],
            "penalized_scores": [float(x) for x in sweep["penalized_scores"]],
            "stability_ari": [float(x) for x in stability_aris],
            "celltypist_model": cfg.celltypist_model,
            "celltypist_label_key": cfg.celltypist_label_key if annotation_col else None,
            "final_label_key": cfg.final_label_key if annotation_col else None,
        }
    )

    # ------------------------------------------------------------------
    # Export annotation table (optional)
    # ------------------------------------------------------------------
    if cfg.annotation_csv is not None and annotation_col is not None:
        io_utils.export_cluster_annotations(
            adata,
            columns=[cfg.label_key, annotation_col],
            out_path=cfg.annotation_csv,
        )

    # ------------------------------------------------------------------
    # Save clustered + annotated h5ad
    # ------------------------------------------------------------------
    if cfg.output_path is not None:
        out_path = cfg.output_path
    else:
        out_path = in_path.with_name(f"{in_path.stem}.clustered.annotated.h5ad")

    io_utils.save_adata(adata, out_path)
    LOGGER.info("Finished cluster_and_annotate. Wrote %s", out_path)

    return adata
