from __future__ import annotations
import logging
import sys
from os import mkdir
from pathlib import Path
from typing import List, Optional, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd

from .config import IntegrationConfig
from .load_and_filter import setup_logging
from . import io_utils,plot_utils
import torch
torch.set_float32_matmul_precision('high')

torch.set_float32_matmul_precision('high')

LOGGER = logging.getLogger(__name__)

DEFAULT_METHODS: tuple[str, ...] = ("Scanorama", "Harmony", "scVI", "BBKNN")
SCANVI_NAME = "scANVI"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _subset_to_hvgs(adata: ad.AnnData) -> ad.AnnData:
    if "highly_variable" not in adata.var:
        raise ValueError("Missing adata.var['highly_variable'].")
    mask = adata.var["highly_variable"].to_numpy()
    if mask.sum() == 0:
        raise ValueError("No HVGs found.")
    return adata[:, mask].copy()


def _ensure_label_key(adata: ad.AnnData, label_key: str) -> None:
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' missing.")


def _run_scanorama_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    import scanorama

    LOGGER.info("Running Scanorama")
    batch_cats = adata.obs[batch_key].unique().tolist()
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adata_list)

    dim = adata_list[0].obsm["X_scanorama"].shape[1]
    out = np.zeros((adata.n_obs, dim), dtype=float)
    for b, sub in zip(batch_cats, adata_list):
        mask = (adata.obs[batch_key] == b).to_numpy()
        out[mask] = sub.obsm["X_scanorama"]
    return out


def _run_harmony_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    import harmonypy as hm

    LOGGER.info("Running Harmony")
    if "X_pca" not in adata.obsm:
        raise ValueError("Missing X_pca for Harmony.")
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key)
    return ho.Z_corr.T

def _run_bbknn_embedding(adata: AnnData, batch_key: str) -> np.ndarray:
    """
    Run BBKNN in an isolated copy so original neighbors graph is untouched.
    Returns a UMAP embedding (n_cells × 2) as a NumPy array.
    """
    LOGGER.info("Running BBKNN")

    import bbknn
    import scanpy as sc

    # Work on a copy to avoid modifying real neighbors
    tmp = adata.copy()

    # Build BBKNN neighbors
    bbknn.bbknn(tmp, batch_key=batch_key)

    # Compute UMAP based on the BBKNN neighbor graph
    sc.tl.umap(tmp)

    # Extract embedding
    emb = tmp.obsm["X_umap"].copy()

    return emb

# scVI and scANVI ------------------------------------------------------------

def _fit_scvi(adata: ad.AnnData, batch_key: str, n_latent: int = 30, max_epochs: int = 400):
    import scvi

    if "counts" not in adata.layers:
        raise ValueError("Missing adata.layers['counts'] for scVI.")

    LOGGER.info("Running scVI")
    scvi.settings.seed = 0
    scvi.settings.dl_num_workers = 0

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True, enable_progress_bar=False)
    return model, model.get_latent_representation()


def _run_scanvi_from_scvi(scvi_model, adata: ad.AnnData, label_key: str, max_epochs: int = 400) -> np.ndarray:
    from scvi.model import SCANVI

    LOGGER.info("Running scANVI")
    lvae = SCANVI.from_scvi_model(scvi_model, adata=adata, labels_key=label_key, unlabeled_category="Unknown")
    lvae.train(max_epochs=max_epochs, n_samples_per_label=100, early_stopping=True, enable_progress_bar=False)
    return lvae.get_latent_representation()


# Method orchestrator --------------------------------------------------------

def _run_all_embeddings(
    adata: ad.AnnData,
    methods: Sequence[str],
    batch_key: str,
    label_key: str,
) -> List[str]:

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)

    # Always include unintegrated PCA
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    created = ["Unintegrated"]

    method_set = {m.lower() for m in methods}
    scvi_model = None

    # ------------------------
    # SCVI/SCANVI shared model
    # ------------------------
    if {"scvi", "scanvi"} & method_set:
        scvi_model, emb = _fit_scvi(adata, batch_key=batch_key)
        adata.obsm["scVI"] = emb
        created.append("scVI")

    # ------------------------
    # Other methods
    # ------------------------
    for m in methods:
        key = m.lower()
        try:
            if key == "scanorama":
                adata.obsm["Scanorama"] = _run_scanorama_embedding(adata, batch_key)
                created.append("Scanorama")

            elif key == "harmony":
                adata.obsm["Harmony"] = _run_harmony_embedding(adata, batch_key)
                created.append("Harmony")

            elif key == "bbknn":
                LOGGER.info("Running BBKNN")
                emb = _run_bbknn_embedding(adata, batch_key=batch_key)
                adata.obsm["BBKNN"] = emb
                created.append("BBKNN")

            elif key == "scvi":
                # Already done above
                continue

            elif key == "scanvi":
                adata.obsm[SCANVI_NAME] = _run_scanvi_from_scvi(scvi_model, adata, label_key)
                created.append(SCANVI_NAME)

        except Exception as e:
            LOGGER.warning("Method '%s' failed: %s", m, e)

    return created


def _select_best_embedding(
    adata: ad.AnnData,
    embedding_keys: Sequence[str],
    batch_key: str,
    label_key: str,
    n_jobs: int,
    figdir: Path,
) -> str:
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    _ensure_label_key(adata, label_key)

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

    #mkdir(figdir.parent / "svg")
    #bm.plot_results_table(show=False, save_dir=figdir.parent / "svg")

    raw = bm.get_results(min_max_scale=False)
    scaled = bm.get_results(min_max_scale=True)

    plot_utils.plot_scib_results_table(scaled, figdir)

    raw.to_csv(figdir.parent.parent / "integration_metrics_raw.tsv", sep="\t")
    scaled.to_csv(figdir.parent.parent / "integration_metrics_scaled.tsv", sep="\t")

    # Convert everything to string first to avoid mixed dtypes
    scaled_str = scaled.astype(str)

    # Coerce everything to numeric (strings → numeric or NaN)
    numeric = scaled_str.apply(pd.to_numeric, errors="coerce")

    # Drop the Metric Type row
    numeric = numeric.drop(index=["Metric Type"], errors="ignore")

    # Drop columns that are entirely NaN (scIB sometimes leaves these)
    numeric = numeric.dropna(axis=1, how="all")

    LOGGER.info(f"scIB results table:\n{numeric.head()}")

    valid_embeddings = []
    for emb in numeric.index:
        row = numeric.loc[emb]
        if row.notna().any() and np.isfinite(row).any():
            valid_embeddings.append(emb)
        else:
            LOGGER.warning(f"Dropping embedding '{emb}' due to all-NaN or non-finite metrics")

    if not valid_embeddings:
        LOGGER.error("All embeddings had invalid (NaN) scores. Falling back to 'Unintegrated'.")
        return "Unintegrated"

    # Compute mean score only across valid embeddings
    scores = numeric.loc[valid_embeddings].mean(axis=1)

    best = scores.idxmax()
    if isinstance(best, tuple):
        best = best[0]

    LOGGER.info(f"Selected best embedding: '{best}'")
    return str(best)



def run_integration(cfg: IntegrationConfig) -> ad.AnnData:
    setup_logging(cfg.logfile)

    in_path: Path = cfg.input_path
    out_path: Path = cfg.output_path or in_path.with_name(f"{in_path.stem}.integrated.h5ad")

    # Figure directory root
    figroot = out_path.parent / "figures"
    plot_utils.setup_scanpy_figs(figroot)

    # Base directory for integration-specific plots:
    figdir = figroot / "integration"

    # Handle output file path (directory vs file)
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / f"{in_path.stem}.integrated.h5ad"
    elif not out_path.suffix:
        out_path = out_path / f"{in_path.stem}.integrated.h5ad"

    # Load data, infer batch key, compute HVGs
    full = sc.read_h5ad(str(in_path))
    batch_key = io_utils.infer_batch_key(full, cfg.batch_key)
    cfg.batch_key = batch_key
    hvg = _subset_to_hvgs(full)

    if "X_pca" not in hvg.obsm:
        sc.tl.pca(hvg)

    methods = cfg.methods or list(DEFAULT_METHODS)

    # Run all embeddings
    emb_keys = _run_all_embeddings(hvg, methods, batch_key, cfg.label_key)

    # Benchmark and choose best embedding
    best = _select_best_embedding(
        hvg, emb_keys, batch_key, cfg.label_key, cfg.benchmark_n_jobs, figdir
    )

    # Copy embeddings to full-data object
    for k in emb_keys:
        full.obsm[k] = hvg.obsm[k]

    # Compute integrated neighbors + UMAP
    full.obsm["X_integrated"] = full.obsm[best]
    sc.pp.neighbors(full, use_rep="X_integrated")
    sc.tl.umap(full)

    # Store metadata
    full.uns.setdefault("integration", {})
    full.uns["integration"].update({
        "methods": emb_keys,
        "best_embedding": best,
        "batch_key": batch_key,
        "label_key": cfg.label_key,
        "input_path": str(in_path),
        "benchmark_metrics_raw_path": str(figdir / "integration_metrics_raw.tsv"),
        "benchmark_metrics_scaled_path": str(figdir / "integration_metrics_scaled.tsv"),
    })

    # Plot per-method UMAPs
    import matplotlib.pyplot as plt

    for method in emb_keys:
        try:
            LOGGER.info(f"Plotting UMAPs for method: {method}")

            # ---- Single-panel integrated UMAP ----
            tmp = full.copy()
            sc.pp.neighbors(tmp, use_rep=method)
            sc.tl.umap(tmp)

            fig = sc.pl.umap(
                tmp,
                color=cfg.batch_key,
                show=False,
                return_fig=True,
            )
            plot_utils.save_multi(f"{method}_umap", figdir)
            plt.close(fig)

            # ---- Two-panel: integrated (left) vs unintegrated (right) ----
            # Skip the unintegrated embedding
            if method == "Unintegrated":
                continue
            tmp2 = full.copy()

            sc.pp.neighbors(tmp2, use_rep=method)
            sc.tl.umap(tmp2)
            umap_integrated = tmp2.obsm["X_umap"].copy()

            sc.pp.neighbors(tmp2, use_rep="Unintegrated")
            sc.tl.umap(tmp2)
            umap_unintegrated = tmp2.obsm["X_umap"].copy()

            # Build two-panel figure
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            tmp2.obsm["X_umap"] = umap_integrated
            sc.pl.umap(
                tmp2,
                color=cfg.batch_key,
                ax=axs[0],
                show=False,
                title=f"{method}",
            )

            tmp2.obsm["X_umap"] = umap_unintegrated
            sc.pl.umap(
                tmp2,
                color=cfg.batch_key,
                ax=axs[1],
                show=False,
                title="Unintegrated",
            )

            plot_utils.save_multi(f"{method}_vs_Unintegrated_umap", figdir)
            plt.close(fig)

        except Exception as e:
            LOGGER.warning("UMAP plotting for %s failed: %s", method, e)

    # ------------------------------------------------------------------
    # Save integrated data
    # ------------------------------------------------------------------
    io_utils.save_adata(full, out_path)
    return full

