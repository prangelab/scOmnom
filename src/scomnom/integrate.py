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

from scomnom import __version__
from .config import IntegrateConfig
from . import io_utils, plot_utils, reporting
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
                enable_progress_bar=True,
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


def _run_scpoli(
    adata: ad.AnnData,
    batch_key: str,
    label_key: Optional[str],
) -> np.ndarray:
    """
    Run scPoli integration.

    scPoli models sample relationships via a learned batch embedding.
    Optionally supervised via label_key.
    """
    from scvi.model import SCPOLI

    LOGGER.info("Training scPoli for integration")

    SCPOLI.setup_anndata(
        adata,
        batch_key=batch_key,
        cell_type_key=label_key,
    )

    model = SCPOLI(adata)
    model.train(
        max_epochs=50,
        enable_progress_bar=False,
    )

    return np.asarray(model.get_latent_representation())


def _run_scanorama(
    adata: ad.AnnData,
    batch_key: str,
    *,
    use_rep: str = "X_pca",
) -> np.ndarray:
    import scanorama

    if use_rep != "X_pca":
        raise ValueError("Scanorama integration expects PCA (X_pca)")

    LOGGER.info("Running Scanorama integration")

    # Split AnnData by batch
    adatas = [
        adata[adata.obs[batch_key] == b].copy()
        for b in adata.obs[batch_key].astype("category").cat.categories
    ]

    # Run Scanorama
    scanorama.correct_scanpy(
        adatas,
        return_dimred=True,
        dimred=adata.obsm["X_pca"].shape[1],
        verbose=False,
    )

    # Reassemble corrected data
    # Use index-based mapping instead of .map() for speed and safety
    Z = np.zeros(
        (adata.n_obs, adata.obsm["X_pca"].shape[1]),
        dtype=adata.obsm["X_pca"].dtype,
    )

    for sub in adatas:
        if "X_scanorama" in sub.obsm:
            # Get integer positions of these cells in the original adata
            idx = adata.obs_names.get_indexer(sub.obs_names)
            Z[idx] = sub.obsm["X_scanorama"]
        else:
            # If a batch failed, use its original PCA so the matrix isn't zeros
            idx = adata.obs_names.get_indexer(sub.obs_names)
            Z[idx] = sub.obsm["X_pca"]
            LOGGER.warning("Scanorama failed for batch %s; using original PCA", sub.obs[batch_key][0])

    return Z
# ---------------------------------------------------------------------
# Run Harmony
# ---------------------------------------------------------------------
def _run_harmony(
    adata: ad.AnnData,
    batch_key: str,
    *,
    use_rep: str = "X_pca",
) -> np.ndarray:
    import harmonypy as hm

    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep} not found in adata.obsm")

    LOGGER.info("Running Harmony integration")

    Z = np.asarray(adata.obsm[use_rep])          # (n_cells, n_pcs)
    meta = adata.obs[[batch_key]].copy()

    ho = hm.run_harmony(
        Z,
        meta,
        vars_use=[batch_key],
        verbose=False,
    )

    # IMPORTANT: transpose
    Z_corr = np.asarray(ho.Z_corr).T              # (n_cells, n_pcs)

    if Z_corr.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Harmony output shape mismatch: {Z_corr.shape}"
        )

    return Z_corr


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

    # Harmony
    if "harmony" in method_set:
        try:
            emb = _run_harmony(
                adata,
                batch_key=batch_key,
                use_rep="X_pca",
            )
            adata.obsm["Harmony"] = emb
            created.append("Harmony")
        except Exception as e:
            LOGGER.warning("Harmony failed: %s", e)

    # Scanorama
    if "scanorama" in method_set:
        try:
            emb = _run_scanorama(
                adata,
                batch_key=batch_key,
                use_rep="X_pca",
            )
            adata.obsm["Scanorama"] = emb
            created.append("Scanorama")
        except Exception as e:
            LOGGER.warning("Scanorama failed: %s", e)

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

    # scPoli
    if "scpoli" in method_set:
        try:
            emb = _run_scpoli(
                adata,
                batch_key=batch_key,
                label_key=label_key,
            )
            adata.obsm["scPoli"] = emb
            created.append("scPoli")
        except Exception as e:
            LOGGER.warning("scPoli failed: %s", e)

    # BBKNN
    if "bbknn" in method_set:
        try:
            LOGGER.info("Running BBKNN (graph-only baseline)")
            _run_bbknn(adata, batch_key=batch_key)
            # Do not put in 'created' as it does not actually create an embedding
        except Exception as e:
            LOGGER.warning("BBKNN failed: %s", e)

    # Safety: verify keys exist
    missing = [k for k in created if k not in adata.obsm]
    if missing:
        raise RuntimeError(f"Integration embeddings missing from adata.obsm: {missing}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adata, created

def _run_bbknn(adata: ad.AnnData, batch_key: str) -> None:
    import bbknn

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)

    bbknn.bbknn(
        adata,
        batch_key=batch_key,
        use_rep="X_pca",
    )

    adata.uns["neighbors_BBKNN"] = adata.uns["neighbors"].copy()
    adata.obsp["connectivities_BBKNN"] = adata.obsp["connectivities"].copy()
    adata.obsp["distances_BBKNN"] = adata.obsp["distances"].copy()

def _select_best_embedding(
    adata: ad.AnnData,
    embedding_keys: Sequence[str],
    batch_key: str,
    label_key: str,
    n_jobs: int,
    output_dir: Path,
) -> str:
    """
    Run scIB benchmarking and select the best integration embedding.

    Selection strategy (baseline-aware):
      1. batch > Unintegrated AND bio > Unintegrated
      2. bio > Unintegrated
      3. batch > Unintegrated
      4. fallback: highest Total

    Supervised methods are allowed to win, but are explicitly annotated.
    """
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    _ensure_label_key(adata, label_key)

    # --------------------------------------------------
    # Method classes (for logging / reporting only)
    # --------------------------------------------------
    METHOD_CLASS = {
        "Unintegrated": "baseline",
        "BBKNN": "baseline",
        "Harmony": "unsupervised",
        "Scanorama": "unsupervised",
        "scVI": "unsupervised",
        "scANVI": "supervised",
        "scPoli": "supervised",
    }

    benchmark_embeddings = [
        e for e in embedding_keys
        if not e.upper().startswith("BBKNN")
    ]

    LOGGER.info(
        "Running scIB benchmarking on embeddings: %s",
        benchmark_embeddings,
    )

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=benchmark_embeddings,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=n_jobs,
    )

    bm.benchmark()

    # --------------------------------------------------
    # Retrieve + save results
    # --------------------------------------------------
    raw = bm.get_results(min_max_scale=False)
    scaled = bm.get_results(min_max_scale=True)

    raw.to_csv(output_dir / "integration_metrics_raw.tsv", sep="\t")
    scaled.to_csv(output_dir / "integration_metrics_scaled.tsv", sep="\t")

    plot_utils.plot_scib_results_table(scaled)

    # --------------------------------------------------
    # Clean numeric table
    # --------------------------------------------------
    numeric = (
        scaled.astype(str)
        .apply(pd.to_numeric, errors="coerce")
        .drop(index=["Metric Type"], errors="ignore")
        .dropna(axis=1, how="all")
    )

    if "Unintegrated" not in numeric.index:
        LOGGER.error("Unintegrated baseline missing from scIB table.")
        return numeric["Total"].idxmax()

    baseline = numeric.loc["Unintegrated"]

    # --------------------------------------------------
    # Helper masks
    # --------------------------------------------------
    bio_ok = numeric["Bio conservation"] > baseline["Bio conservation"]
    batch_ok = numeric["Batch correction"] > baseline["Batch correction"]

    # Never compare baseline to itself
    candidates = numeric.index != "Unintegrated"

    # --------------------------------------------------
    # Tiered selection
    # --------------------------------------------------
    tier1 = numeric.loc[candidates & bio_ok & batch_ok]
    tier2 = numeric.loc[candidates & bio_ok]
    tier3 = numeric.loc[candidates & batch_ok]

    if not tier1.empty:
        best = tier1["Total"].idxmax()
        reason = "bio > baseline AND batch > baseline"
    elif not tier2.empty:
        best = tier2["Total"].idxmax()
        reason = "bio > baseline"
    elif not tier3.empty:
        best = tier3["Total"].idxmax()
        reason = "batch > baseline"
    else:
        best = numeric.loc[candidates, "Total"].idxmax()
        reason = "fallback (highest Total)"

    # --------------------------------------------------
    # Logging + annotation
    # --------------------------------------------------
    method_class = METHOD_CLASS.get(best, "unknown")

    LOGGER.info(
        "Selected best embedding: '%s' (%s) — reason: %s",
        best,
        method_class,
        reason,
    )

    if method_class == "supervised":
        unsup = [
            m for m in numeric.index
            if METHOD_CLASS.get(m) == "unsupervised"
        ]
        if unsup:
            best_unsup = numeric.loc[unsup, "Total"].idxmax()
            LOGGER.info(
                "Best unsupervised alternative: '%s' (Total=%.3f)",
                best_unsup,
                numeric.loc[best_unsup, "Total"],
            )

    return str(best)



def _load_scib_table_from_disk(output_dir: Path) -> pd.DataFrame:
    path = Path("integration_metrics_scaled.tsv")
    if not path.exists():
        raise RuntimeError(
            f"scIB metrics file not found on disk: {path}"
        )
    return pd.read_csv(path, sep="\t", index_col=0)
# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_integrate(cfg: ProcessAndIntegrateConfig) -> ad.AnnData:
    init_logging(cfg.logfile)
    LOGGER.info("Starting integration module")

    # Configure Scanpy/Matplotlib figure behavior + formats
    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    adata = io_utils.load_dataset(cfg.input_path)

    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key is None:
        raise RuntimeError("batch_key missing")
    LOGGER.info("Using batch_key='%s'", batch_key)

    methods = cfg.methods or ["scVI", "scANVI", "Harmony", "Scanorama", "BBKNN"]

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
        output_dir=cfg.output_dir,
    )

    plot_utils.plot_integration_umaps(
        adata,
        embedding_keys=emb_keys,
        color=batch_key,
    )

    if best.lower().startswith("bbknn"):
        LOGGER.warning(
            "BBKNN selected as best integration; "
            "using BBKNN neighbor graph (graph-based integration)."
        )

        import bbknn
        bbknn.bbknn(
            adata,
            batch_key=batch_key,
            use_rep="X_pca",
        )

        # For API consistency only; BBKNN integration lives in the graph
        adata.obsm["X_integrated"] = adata.obsm["X_pca"]

    else:
        adata.obsm["X_integrated"] = adata.obsm[best]
        sc.pp.neighbors(adata, use_rep="X_integrated")

    sc.tl.umap(adata)

    reporting.generate_integration_report(
        fig_root=cfg.figdir,
        version=__version__,
        adata=adata,
        batch_key=batch_key,
        label_key=cfg.label_key,
        methods=methods,
        benchmark_n_jobs=cfg.benchmark_n_jobs,
        selected_embedding=best,
    )

    out_zarr = cfg.output_dir / (cfg.output_name + ".zarr")
    LOGGER.info("Saving integrated dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if getattr(cfg, "save_h5ad", False):
        out_h5ad = cfg.output_dir / (cfg.output_name + ".h5ad")
        LOGGER.warning(
            "Writing additional H5AD output (loads full matrix into RAM): %s",
            out_h5ad,
        )
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
        LOGGER.info("Saved integrated H5AD → %s", out_h5ad)

    LOGGER.info("Finished integration module")
    return adata

