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

from .config import IntegrateConfig
from . import io_utils, plot_utils
from .logging_utils import init_logging

torch.set_float32_matmul_precision("high")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_scvi_layer(adata: ad.AnnData) -> Optional[str]:
    if "counts_cb" in adata.layers:
        return "counts_cb"
    if "counts_raw" in adata.layers:
        return "counts_raw"
    return None

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

def _auto_scanvi_epochs(n_cells: int) -> int:
    if n_cells < 50_000:
        return 100
    if n_cells < 200_000:
        return 150
    return 200


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
# scANVI (reuses integration SCVI model)
# ---------------------------------------------------------------------
def _run_scanvi_from_scvi(
    scvi_model,
    adata: ad.AnnData,
    label_key: str,
):
    """
    Run scANVI using an existing, already-trained SCVI model.

    Parameters
    ----------
    scvi_model
        Trained SCVI model from the integration stage
    adata
        AnnData object used for integration
    label_key
        Key in adata.obs with cluster / cell-type labels

    Returns
    -------
    np.ndarray
        Latent representation (n_cells × n_latent)
    """
    from scvi.model import SCANVI

    _ensure_label_key(adata, label_key)

    accelerator, devices = _select_device()
    max_epochs = _auto_scanvi_epochs(adata.n_obs)

    LOGGER.info(
        "Running scANVI (reuse SCVI, n_cells=%d, max_epochs=%d)",
        adata.n_obs,
        max_epochs,
    )

    # Build SCANVI on top of trained SCVI
    lvae = SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )

    # Train scANVI
    lvae.train(
        max_epochs=max_epochs,
        n_samples_per_label=100,
        early_stopping=True,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=devices,
    )

    return np.asarray(lvae.get_latent_representation())


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

        layer = _get_scvi_layer(adata)

        scvi_model = _train_scvi(
            adata,
            batch_key=batch_key,
            layer=layer,
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adata, created

def _run_bbknn_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    import bbknn

    tmp = adata.copy()

    if "X_pca" not in tmp.obsm:
        sc.tl.pca(tmp)

    bbknn.bbknn(tmp, batch_key=batch_key)

    # IMPORTANT: embedding = PCA, graph = BBKNN
    return np.asarray(tmp.obsm["X_pca"])



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
    raw_path = Path(cfg.output_dir) / "integration_metrics_raw.tsv"
    scaled_path = Path(cfg.output_dir) / "integration_metrics_scaled.tsv"
    raw.to_csv(raw_path, sep="\t")
    scaled.to_csv(scaled_path, sep="\t")

    plot_utils.plot_scib_results_table(scaled)

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


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_integrate(cfg: ProcessAndIntegrateConfig) -> ad.AnnData:
    init_logging(cfg.logfile)
    LOGGER.info("Starting integration module")

    # Configure Scanpy/Matplotlib figure behavior + formats
    plot_utils.setup_scanpy_figs(cfg.figdir_name, cfg.figure_formats)

    adata = io_utils.load_dataset(cfg.input_path)

    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key is None:
        raise RuntimeError("batch_key missing")

    LOGGER.info("Using batch_key='%s'", batch_key)

    methods = cfg.methods or ["scVI", "scANVI", "BBKNN"]

    adata, emb_keys = _run_integrations(
        adata,
        methods=methods,
        batch_key=batch_key,
        label_key=cfg.label_key,
    )

    best = _select_best_embedding(
        adata,
        embedding_keys=emb_keys,
        batch_key=batch_key,
        label_key=cfg.label_key,
        n_jobs=cfg.benchmark_n_jobs,
        figdir=Path("integration"),
    )

    plot_utils.plot_integration_umaps(
        adata,
        embedding_keys=emb_keys,
        color=batch_key,
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

    LOGGER.info("Finished integration module")
    return adata

