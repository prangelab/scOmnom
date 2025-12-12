# src/scomnom/process_and_integrate.py

from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import torch

from .config import ProcessAndIntegrateConfig
from . import io_utils, plot_utils
from .logging_utils import init_logging

torch.set_float32_matmul_precision("high")
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_label_key(adata: ad.AnnData, label_key: str) -> None:
    if label_key not in adata.obs:
        raise KeyError(
            f"Label key '{label_key}' missing from adata.obs. "
            f"Available: {list(adata.obs.columns)[:20]}..."
        )


def _select_device():
    if torch.cuda.is_available():
        return "gpu", "auto"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", "auto"


def _is_oom_error(e: Exception) -> bool:
    txt = str(e).lower()
    return (
        "out of memory" in txt
        or "cuda error" in txt
        or "cublas_status_alloc_failed" in txt
        or ("mps" in txt and "oom" in txt)
    )


def _auto_scvi_epochs(n_cells: int) -> int:
    if n_cells < 50_000:
        return 80
    if n_cells < 200_000:
        return 60
    return 40


# ---------------------------------------------------------------------
# Generic SCVI trainer (used everywhere)
# ---------------------------------------------------------------------
def _train_scvi(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    layer: Optional[str],
    purpose: str,
):
    """
    Train an SCVI model with auto batch-size + auto epochs.

    purpose: "solo" | "integration" (logging only)
    """
    from scvi.model import SCVI

    accelerator, devices = _select_device()
    epochs = _auto_scvi_epochs(adata.n_obs)

    batch_ladder = [1024, 512, 256, 128, 64, 32]
    last_err = None

    LOGGER.info(
        "Training SCVI for %s (n_cells=%d, epochs=%d)",
        purpose,
        adata.n_obs,
        epochs,
    )

    for bsz in batch_ladder:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*setup_anndata is overwriting.*")
                SCVI.setup_anndata(
                    adata,
                    layer=layer,
                    batch_key=batch_key,
                )

            model = SCVI(adata)
            model.train(
                max_epochs=epochs,
                accelerator=accelerator,
                devices=devices,
                batch_size=bsz,
                enable_progress_bar=False,
            )

            LOGGER.info("SCVI trained successfully (batch_size=%d)", bsz)
            return model

        except RuntimeError as e:
            if _is_oom_error(e):
                LOGGER.warning("OOM at batch_size=%d, retrying smaller...", bsz)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_err = e
                continue
            raise

    raise RuntimeError("SCVI training failed") from last_err


# ---------------------------------------------------------------------
# SOLO (always global)
# ---------------------------------------------------------------------
def run_solo_with_scvi(
    adata: ad.AnnData,
    batch_key: Optional[str],
    doublet_score_threshold: float,
) -> ad.AnnData:
    from scvi.external import SOLO

    LOGGER.info("Running SOLO doublet detection (global)")

    layer = "counts_raw" if "counts_raw" in adata.layers else None
    scvi_model = _train_scvi(
        adata,
        batch_key=batch_key,
        layer=layer,
        purpose="solo",
    )

    accelerator, devices = _select_device()

    solo = SOLO.from_scvi_model(scvi_model)
    solo.train(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
    )

    probs = solo.predict(soft=True)
    scores = probs["doublet"].to_numpy()

    adata.obs["doublet_score"] = scores
    adata.obs["predicted_doublet"] = (
        scores > doublet_score_threshold
    ).astype("category")

    LOGGER.info(
        "Doublets detected: %d / %d (%.2f%%)",
        int((scores > doublet_score_threshold).sum()),
        adata.n_obs,
        float((scores > doublet_score_threshold).mean() * 100),
    )

    return adata


# ---------------------------------------------------------------------
# Cleanup after SOLO
# ---------------------------------------------------------------------
def cleanup_after_solo(
    adata: ad.AnnData,
    batch_key: str,
    min_cells_per_sample: int,
) -> ad.AnnData:
    if "predicted_doublet" in adata.obs:
        n0 = adata.n_obs
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()
        LOGGER.info("Removed doublets: kept %d / %d", adata.n_obs, n0)

    if min_cells_per_sample > 0:
        vc = adata.obs[batch_key].value_counts()
        small = vc[vc < min_cells_per_sample].index
        if len(small):
            n0 = adata.n_obs
            adata = adata[~adata.obs[batch_key].isin(small)].copy()
            LOGGER.info(
                "Dropped %d small samples (<%d cells); kept %d / %d",
                len(small),
                min_cells_per_sample,
                adata.n_obs,
                n0,
            )
    return adata


# ---------------------------------------------------------------------
# Normalization + PCA
# ---------------------------------------------------------------------
def normalize_and_hvg(adata: ad.AnnData, n_top_genes: int, batch_key: str) -> ad.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
    )
    return adata


def pca_neighbors_umap(adata: ad.AnnData, max_pcs_plot: int) -> ad.AnnData:
    from kneed import KneeLocator

    sc.tl.pca(adata)
    vr = adata.uns["pca"]["variance_ratio"]
    kl = KneeLocator(range(1, len(vr) + 1), vr, curve="convex", direction="decreasing")
    n_pcs = int(kl.elbow or max_pcs_plot)

    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    adata.uns["n_pcs_elbow"] = n_pcs
    return adata


def cluster_unintegrated(adata: ad.AnnData) -> ad.AnnData:
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden")
    return adata


# ---------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------
def _run_integrations(
    adata: ad.AnnData,
    methods: Sequence[str],
    batch_key: str,
    label_key: str,
) -> tuple[ad.AnnData, List[str]]:
    """
    Run requested integration methods and write embeddings into adata.obsm.

    Returns
    -------
    (adata, created_keys)
    """
    # Ensure PCA exists and store as "Unintegrated"
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    adata.obsm["Unintegrated"] = np.asarray(adata.obsm["X_pca"])
    created: List[str] = ["Unintegrated"]

    method_set = {m.lower() for m in (methods or [])}

    scvi_model = None  # local to this integration run

    # scVI
    if "scvi" in method_set:
        LOGGER.info("Training SCVI for integration")
        scvi_model = _train_scvi(
            adata,
            batch_key=batch_key,
            layer="counts" if "counts" in adata.layers else None,
            purpose="integration",
        )
        adata.obsm["scVI"] = np.asarray(
            scvi_model.get_latent_representation(adata)
        )
        created.append("scVI")

    # scANVI
    if "scanvi" in method_set:
        try:
            # Train SCVI only if it does not already exist
            if scvi_model is None:
                LOGGER.info(
                    "scANVI requested without prior scVI; training SCVI backbone first"
                )
                scvi_model = _train_scvi(
                    adata,
                    batch_key=batch_key,
                    layer="counts" if "counts" in adata.layers else None,
                    purpose="integration",
                )

            emb = _run_scanvi_from_scvi(scvi_model, adata, label_key)
            adata.obsm["scANVI"] = np.asarray(emb)
            created.append("scANVI")

        except Exception as e:
            LOGGER.warning("scANVI failed: %s", e)

    # BBKNN (UMAP is 2D, but that’s fine for benchmarking if you want; many prefer PCA/latent)
    if "bbknn" in method_set:
        try:
            emb = _run_bbknn_embedding(adata, batch_key=batch_key)
            adata.obsm["BBKNN"] = np.asarray(emb)
            created.append("BBKNN")
        except Exception as e:
            LOGGER.warning("BBKNN failed: %s", e)

    # Safety: verify keys exist
    missing = [k for k in created if k not in adata.obsm]
    if missing:
        raise RuntimeError(f"Integration embeddings missing from adata.obsm: {missing}")

    return adata, created



def _select_best_embedding(
    adata: ad.AnnData,
    embedding_keys: Sequence[str],
    batch_key: str,
    label_key: str,
    n_jobs: int,
    figdir: Path,
) -> str:
    """
    Run scIB benchmarking and select the best integration embedding.

    Returns
    -------
    str
        Name of the best embedding key.
    """
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    _ensure_label_key(adata, label_key)

    LOGGER.info("Running scIB benchmarking on embeddings: %s", list(embedding_keys))

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=list(embedding_keys),
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=n_jobs,
    )

    bm.benchmark()

    # ------------------------------------------------------------------
    # Retrieve results
    # ------------------------------------------------------------------
    raw = bm.get_results(min_max_scale=False)
    scaled = bm.get_results(min_max_scale=True)

    # Save tables
    raw_path = figdir.parent / "integration_metrics_raw.tsv"
    scaled_path = figdir.parent / "integration_metrics_scaled.tsv"
    raw.to_csv(raw_path, sep="\t")
    scaled.to_csv(scaled_path, sep="\t")

    plot_utils.plot_scib_results_table(scaled, figdir)

    # ------------------------------------------------------------------
    # Clean + score
    # ------------------------------------------------------------------
    scaled_str = scaled.astype(str)
    numeric = scaled_str.apply(pd.to_numeric, errors="coerce")

    # Drop metadata rows if present
    numeric = numeric.drop(index=["Metric Type"], errors="ignore")
    numeric = numeric.dropna(axis=1, how="all")

    LOGGER.info("scIB results table (head):\n%s", numeric.head())

    valid_embeddings: List[str] = []
    for emb in numeric.index:
        row = numeric.loc[emb]
        if row.notna().any() and np.isfinite(row).any():
            valid_embeddings.append(emb)
        else:
            LOGGER.warning(
                "Dropping embedding '%s' due to all-NaN or non-finite metrics.",
                emb,
            )

    if not valid_embeddings:
        LOGGER.error(
            "All embeddings had invalid scores; falling back to 'Unintegrated'."
        )
        return "Unintegrated"

    scores = numeric.loc[valid_embeddings].mean(axis=1)
    best = scores.idxmax()

    if isinstance(best, tuple):
        best = best[0]

    LOGGER.info("Selected best embedding: '%s'", best)
    return str(best)


def _plot_umaps_for_embeddings(
    adata: ad.AnnData,
    embedding_keys: Sequence[str],
    *,
    color: str,
    figdir: Path,
) -> None:
    """
    For each embedding in adata.obsm, compute neighbors/UMAP in a temp copy and save plots.
    Also saves side-by-side UMAP comparison vs Unintegrated.
    """
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)

    for method in embedding_keys:
        if method not in adata.obsm:
            LOGGER.warning("Skipping UMAP plot: embedding '%s' not found in adata.obsm", method)
            continue

        try:
            LOGGER.info("Plotting UMAPs for method: %s", method)

            # Single UMAP for this embedding
            tmp = adata.copy()
            sc.pp.neighbors(tmp, use_rep=method)
            sc.tl.umap(tmp)

            fig = sc.pl.umap(
                tmp,
                color=color,
                show=False,
                return_fig=True,
            )
            plot_utils.save_multi(f"{method}_umap", figdir, fig=fig)
            plt.close(fig)

            # Two-panel comparison vs Unintegrated (if available and not itself)
            if method != "Unintegrated" and "Unintegrated" in adata.obsm:
                tmp2 = adata.copy()

                sc.pp.neighbors(tmp2, use_rep=method)
                sc.tl.umap(tmp2)
                umap_integrated = tmp2.obsm["X_umap"].copy()

                sc.pp.neighbors(tmp2, use_rep="Unintegrated")
                sc.tl.umap(tmp2)
                umap_unintegrated = tmp2.obsm["X_umap"].copy()

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                tmp2.obsm["X_umap"] = umap_integrated
                sc.pl.umap(
                    tmp2,
                    color=color,
                    ax=axs[0],
                    show=False,
                    title=f"{method}",
                )

                tmp2.obsm["X_umap"] = umap_unintegrated
                sc.pl.umap(
                    tmp2,
                    color=color,
                    ax=axs[1],
                    show=False,
                    title="Unintegrated",
                )

                plot_utils.save_multi(f"{method}_vs_Unintegrated_umap", figdir, fig=fig)
                plt.close(fig)

        except Exception as e:
            LOGGER.warning("UMAP plotting for %s failed: %s", method, e)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_process_and_integrate(cfg: ProcessAndIntegrateConfig) -> ad.AnnData:
    init_logging(cfg.logfile)
    LOGGER.info("Starting process-and-integrate")

    figroot = cfg.output_dir / cfg.figdir_name
    plot_utils.setup_scanpy_figs(figroot, cfg.figure_formats)
    figdir_integration = figroot / "integration"
    figdir_integration.mkdir(parents=True, exist_ok=True)

    adata_full = io_utils.load_dataset(cfg.input_path)

    batch_key = cfg.batch_key or adata_full.uns.get("batch_key")
    if batch_key is None:
        raise RuntimeError("batch_key missing")

    LOGGER.info("Using batch_key='%s'", batch_key)

    adata_full = run_solo_with_scvi(
        adata_full,
        batch_key=batch_key,
        doublet_score_threshold=cfg.doublet_score_threshold,
    )

    adata = cleanup_after_solo(
        adata_full,
        batch_key=batch_key,
        min_cells_per_sample=cfg.min_cells_per_sample,
    )

    out_zarr = cfg.output_dir / (cfg.output_name + ".CHECKPOINT.zarr")
    LOGGER.info("Saving CHECKPOINT dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    adata = normalize_and_hvg(adata, cfg.n_top_genes, batch_key)
    adata = pca_neighbors_umap(adata, cfg.max_pcs_plot)
    adata = cluster_unintegrated(adata)

    methods = cfg.methods or ["bbknn", "scvi", "scanvi"]
    if "scanvi" in {methods}:
        _ensure_label_key(adata, cfg.label_key)
    adata, emb_keys = _run_integrations(
        adata,
        methods=methods,
        batch_key=batch_key,
        label_key=label_key,
    )

    best = _select_best_embedding(
        adata,
        embedding_keys=emb_keys,
        batch_key=batch_key,
        label_key=cfg.label_key,
        n_jobs=cfg.benchmark_n_jobs,
        figdir=figdir_integration,
    )

    _plot_umaps_for_embeddings(
        adata,
        embedding_keys=emb_keys,
        color=batch_key,
        figdir=figdir_integration,
    )

    adata.obsm["X_integrated"] = adata.obsm[best]
    sc.pp.neighbors(adata, use_rep="X_integrated")
    sc.tl.umap(adata)

    out_zarr = cfg.output_dir / (cfg.output_name + ".integrated.zarr")
    LOGGER.info("Saving integrated dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if getattr(cfg, "save_h5ad", False):
        out_h5ad = cfg.output_dir / (cfg.output_name + ".integrated.h5ad")
        LOGGER.warning(
            "Writing additional H5AD output (loads full matrix into RAM): %s",
            out_h5ad,
        )
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
        LOGGER.info("Saved integrated H5AD → %s", out_h5ad)

    LOGGER.info("Finished process-and-integrate")
    return adata

