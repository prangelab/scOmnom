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

from .config import ProcessAndIntegrateConfig  # or IntegrationConfig / whatever you call it
from . import io_utils, plot_utils

torch.set_float32_matmul_precision("high")

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Logging helper (same pattern as load_and_filter)
# ---------------------------------------------------------------------
def setup_logging(logfile: Optional[Path]) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid double formatting / Rich interception)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(logfile), mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

# Helper function
def _ensure_label_key(adata: ad.AnnData, label_key: str) -> None:
    if label_key not in adata.obs:
        raise KeyError(
            f"Label key '{label_key}' missing from adata.obs. "
            f"Available: {list(adata.obs.columns)[:20]}..."
        )


# ---------------------------------------------------------------------
# SOLO + SCVI (copied from old load_and_filter, slightly wrapped)
# ---------------------------------------------------------------------
def run_solo_with_scvi(
    adata: ad.AnnData,
    batch_key: Optional[str],
    doublet_score_threshold: float,
    restrict_to_batch: bool = False,
) -> tuple[ad.AnnData, "scvi.model.SCVI"]:

    """
    Run SCVI + SOLO on the given AnnData.

    Returns:
      - adata with 'doublet_score' and 'predicted_doublet' in .obs
      - trained SCVI model (for later reuse in integration)
    """
    import scvi
    from scvi.model import SCVI
    from scvi.external import SOLO

    LOGGER.info("Running SOLO doublet detection (SCVI → SOLO) with auto batch-size scaling...")

    # Choose layer for counts
    layer = "counts_raw" if "counts_raw" in adata.layers else None
    if layer is None:
        LOGGER.warning("No counts_raw layer found; using adata.X for SCVI + SOLO.")
    else:
        LOGGER.info("Using layer '%s' for SCVI + SOLO.", layer)

    # Device selection
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        LOGGER.info("SCVI/SOLO accelerator: GPU")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        LOGGER.info("SCVI/SOLO accelerator: Apple MPS (M1/M2/M3)")
    else:
        accelerator = "cpu"
        devices = "auto"
        LOGGER.info("SCVI/SOLO accelerator: CPU")

    n_cells = adata.n_obs

    # Heuristic epochs for SCVI
    if n_cells < 50_000:
        scvi_epochs = 80
    elif n_cells < 200_000:
        scvi_epochs = 60
    else:
        scvi_epochs = 40

    solo_epochs = 10
    batch_ladder = [1024, 512, 256, 128, 64, 32]

    def _is_oom_error(e: Exception) -> bool:
        txt = str(e).lower()
        return (
            "out of memory" in txt
            or "cuda error" in txt
            or "cublas_status_alloc_failed" in txt
            or ("mps" in txt and "oom" in txt)
        )

    chosen_batch = None
    last_exception: Optional[Exception] = None

    # Try SCVI with decreasing batch sizes
    LOGGER.info("Training SCVI backbone for SOLO...")
    for b in batch_ladder:
        LOGGER.info("Trying SCVI batch_size=%d ...", b)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*setup_anndata is overwriting.*")
                SCVI.setup_anndata(
                    adata,
                    layer=layer,
                    batch_key=batch_key,
                )

            vae = SCVI(adata)

            vae.train(
                max_epochs=scvi_epochs,
                accelerator=accelerator,
                devices=devices,
                batch_size=b,
            )

            LOGGER.info("SCVI successfully trained with batch_size=%d", b)
            chosen_batch = b
            scvi_model = vae
            break

        except RuntimeError as e:
            if _is_oom_error(e):
                LOGGER.warning(
                    "OOM while training SCVI at batch_size=%d; retrying with smaller batch...",
                    b,
                )
                last_exception = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            raise

    if chosen_batch is None:
        LOGGER.error("All batch sizes failed for SCVI; last error: %s", last_exception)
        raise last_exception or RuntimeError("SCVI training failed for unknown reasons")

    # Train SOLO with the same batch size
    LOGGER.info("Training SOLO with batch_size=%d (epochs=%d)", chosen_batch, solo_epochs)
    if restrict_to_batch:
        if batch_key is None:
            raise ValueError(
                "solo_restrict_to_batch=True requires batch_key to be set."
            )

        LOGGER.info(
            "Running SOLO per batch using batch_key='%s'",
            batch_key,
        )

        solo = SOLO.from_scvi_model(
            scvi_model,
            restrict_to_batch=batch_key,
        )
    else:
        LOGGER.info("Running SOLO globally across all batches.")
        solo = SOLO.from_scvi_model(scvi_model)

    try:
        solo.train(
            max_epochs=solo_epochs,
            accelerator=accelerator,
            devices=devices,
            batch_size=chosen_batch,
        )
    except RuntimeError as e:
        if _is_oom_error(e):
            LOGGER.error(
                "Unexpected OOM while training SOLO with batch_size=%d "
                "after SCVI had succeeded.",
                chosen_batch,
            )
        raise

    LOGGER.info("Predicting doublet probabilities via SOLO...")
    probs = solo.predict(soft=True)

    # scvi-tools >= 1.1 returns a DataFrame; older versions return ndarray
    if isinstance(probs, pd.DataFrame):
        if "doublet" not in probs.columns:
            raise KeyError(
                f"Expected 'doublet' column in SOLO predictions, got: {list(probs.columns)}"
            )
        doublet_scores = probs["doublet"].to_numpy()
    else:
        # legacy ndarray behavior
        doublet_scores = probs[:, 1]

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = doublet_scores > doublet_score_threshold

    LOGGER.info(
        "Doublets detected: %d / %d (%.2f%%)",
        int(adata.obs["predicted_doublet"].sum()),
        adata.n_obs,
        float(adata.obs["predicted_doublet"].mean() * 100.0),
    )

    # Cast boolean to categorical for convenience
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype("category")

    return adata, scvi_model


# ---------------------------------------------------------------------
# Cleanup after SOLO: remove doublets / high-mt / tiny samples
# ---------------------------------------------------------------------
def cleanup_after_solo(
    adata: ad.AnnData,
    batch_key: str,
    max_pct_mt: float,
    min_cells_per_sample: int,
) -> ad.AnnData:
    """
    Remove predicted doublets, high-mt cells and samples with too few cells.
    """

    # Remove doublets
    if "predicted_doublet" in adata.obs:
        n_before = adata.n_obs
        mask = ~adata.obs["predicted_doublet"].astype(bool).to_numpy()
        adata = adata[mask].copy()
        LOGGER.info(
            "Removed predicted doublets: kept %d / %d cells.",
            adata.n_obs,
            n_before,
        )

    # Remove high-mt cells
    if "pct_counts_mt" in adata.obs:
        n_before = adata.n_obs
        mask = adata.obs["pct_counts_mt"] < max_pct_mt
        adata = adata[mask].copy()
        LOGGER.info(
            "MT filter: kept %d / %d cells (pct_counts_mt < %.1f).",
            adata.n_obs,
            n_before,
            max_pct_mt,
        )
    else:
        LOGGER.warning("pct_counts_mt not found in adata.obs; skipping mitochondrial filter.")

    # Drop tiny samples
    if min_cells_per_sample > 0 and batch_key in adata.obs:
        vc = adata.obs[batch_key].value_counts()
        small = vc[vc < min_cells_per_sample].index.tolist()
        if small:
            n_before = adata.n_obs
            mask = ~adata.obs[batch_key].isin(small)
            adata = adata[mask].copy()
            LOGGER.info(
                "Dropped %d samples with < %d cells; kept %d / %d cells.",
                len(small),
                min_cells_per_sample,
                adata.n_obs,
                n_before,
            )
    else:
        LOGGER.warning(
            "Either min_cells_per_sample == 0 or batch_key '%s' not in adata.obs; "
            "skipping per-sample cell-count filtering.",
            batch_key,
        )

    return adata


# ---------------------------------------------------------------------
# Normalisation, HVGs, PCA/UMAP, clustering (unintegrated)
# ---------------------------------------------------------------------
def normalize_and_hvg(adata: ad.AnnData, n_top_genes: int, batch_key: str) -> ad.AnnData:
    """
    Library-size normalisation, log1p and HVG selection.
    Stores raw counts in .layers['counts'].
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
    )
    return adata


def pca_neighbors_umap(
    adata: ad.AnnData,
    max_pcs_plot: int,
) -> ad.AnnData:
    """
    Run PCA, select number of PCs via elbow, construct neighbors and UMAP.
    """
    from kneed import KneeLocator

    sc.tl.pca(adata)
    pvar = adata.uns["pca"]["variance_ratio"]
    kl = KneeLocator(
        range(1, len(pvar) + 1),
        pvar,
        curve="convex",
        direction="decreasing",
    )
    n_pcs_elbow = kl.elbow or max_pcs_plot
    n_pcs_elbow = int(n_pcs_elbow)

    LOGGER.info("Using n_pcs=%d for neighbors/UMAP (elbow or max_pcs_plot).", n_pcs_elbow)
    sc.pp.neighbors(adata, n_pcs=n_pcs_elbow)
    sc.tl.umap(adata)
    adata.uns["n_pcs_elbow"] = n_pcs_elbow
    return adata


def cluster_unintegrated(
    adata: ad.AnnData,
    resolution: float = 1.0,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Single Leiden clustering on the unintegrated UMAP/neighbors graph.
    """
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        random_state=random_state,
        key_added="leiden",
    )
    LOGGER.info(
        "Leiden clustering complete (resolution=%.2f). Found %d clusters.",
        resolution,
        adata.obs["leiden"].nunique(),
    )
    return adata


# ---------------------------------------------------------------------
# Integration methods: BBKNN, scVI, scANVI (reusing SCVI model)
# ---------------------------------------------------------------------
def _run_bbknn_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    """
    Run BBKNN in an isolated copy so original neighbors graph is untouched.
    Returns a UMAP embedding (n_cells × 2) as a NumPy array.
    """
    LOGGER.info("Running BBKNN")

    import bbknn

    tmp = adata.copy()
    bbknn.bbknn(tmp, batch_key=batch_key)
    sc.tl.umap(tmp)
    emb = tmp.obsm["X_umap"].copy()
    return emb


def _run_scanvi_from_scvi(
    scvi_model,
    adata: ad.AnnData,
    label_key: str,
    max_epochs: int = 400,
) -> np.ndarray:
    from scvi.model import SCANVI
    import torch

    LOGGER.info("Running scANVI (with multi-GPU support)")

    # ------------------------
    # Device detection (same as SOLO + scVI)
    # ------------------------
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        LOGGER.info("scANVI accelerator: GPU")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        LOGGER.info("scANVI accelerator: Apple MPS")
    else:
        accelerator = "cpu"
        devices = "auto"
        LOGGER.info("scANVI accelerator: CPU")

    # ------------------------
    # Build SCANVI from the trained scVI model
    # ------------------------
    lvae = SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )

    # ------------------------
    # Train SCANVI with multi-GPU acceleration
    # ------------------------
    lvae.train(
        max_epochs=max_epochs,
        n_samples_per_label=100,
        early_stopping=True,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=devices,
    )

    return lvae.get_latent_representation()



def _run_integrations_with_existing_scvi(
    adata: ad.AnnData,
    scvi_model,
    methods: Sequence[str],
    batch_key: str,
    label_key: str,
) -> List[str]:
    """
    Run requested integration methods, reusing an existing SCVI model.
    Returns list of embedding keys created in .obsm.
    """

    # Ensure PCA exists and store as "Unintegrated"
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    created = ["Unintegrated"]

    method_set = {m.lower() for m in methods}

    # scVI (latent representation)
    if "scvi" in method_set:
        LOGGER.info("Computing scVI latent representation from pre-trained SCVI model.")
        emb = scvi_model.get_latent_representation(adata)
        adata.obsm["scVI"] = emb
        created.append("scVI")

    # scANVI
    if "scanvi" in method_set:
        try:
            emb = _run_scanvi_from_scvi(scvi_model, adata, label_key)
            adata.obsm["scANVI"] = emb
            created.append("scANVI")
        except Exception as e:
            LOGGER.warning("scANVI failed: %s", e)

    # BBKNN
    if "bbknn" in method_set:
        try:
            emb = _run_bbknn_embedding(adata, batch_key=batch_key)
            adata.obsm["BBKNN"] = emb
            created.append("BBKNN")
        except Exception as e:
            LOGGER.warning("BBKNN failed: %s", e)

    return created


# ---------------------------------------------------------------------
# scIB benchmarking (mostly copied from old integrate)
# ---------------------------------------------------------------------
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

    raw = bm.get_results(min_max_scale=False)
    scaled = bm.get_results(min_max_scale=True)

    plot_utils.plot_scib_results_table(scaled, figdir)

    # Save results
    raw_path = figdir.parent / "integration_metrics_raw.tsv"
    scaled_path = figdir.parent / "integration_metrics_scaled.tsv"
    raw.to_csv(raw_path, sep="\t")
    scaled.to_csv(scaled_path, sep="\t")

    # Convert to numeric for summary
    scaled_str = scaled.astype(str)
    numeric = scaled_str.apply(pd.to_numeric, errors="coerce")

    # Drop Metadata row if present
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

def _save_solo_checkpoint_clean(
    *,
    outdir: Path,
    adata_cleaned: ad.AnnData,
) -> None:
    ckpt_dir = outdir / "checkpoints" / "solo_scvi"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving SOLO/SCVI checkpoint → %s", ckpt_dir)

    io_utils.save_dataset(
        adata_cleaned,
        ckpt_dir / "adata_cleaned.zarr",
        fmt="zarr",
    )

    LOGGER.info("SOLO/SCVI checkpoint clean saved successfully.")

def _save_solo_checkpoint_no_clean(
    *,
    outdir: Path,
    scvi_model,
    adata_full: ad.AnnData,
) -> None:
    ckpt_dir = outdir / "checkpoints" / "solo_scvi"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving SOLO/SCVI checkpoint → %s", ckpt_dir)

    # Save SCVI model
    scvi_model.save(
        ckpt_dir / "scvi_model",
        overwrite=True,
    )

    # Save AnnData objects
    io_utils.save_dataset(
        adata_full,
        ckpt_dir / "adata_full.zarr",
        fmt="zarr",
    )

    LOGGER.info("SOLO/SCVI checkpoint no clean saved successfully.")

def _save_solo_checkpoint_finetuned(
    *,
    outdir: Path,
    scvi_model,
) -> None:
    ckpt_dir = outdir / "checkpoints" / "solo_scvi"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving SOLO/SCVI checkpoint → %s", ckpt_dir)

    # Save SCVI model
    scvi_model.save(
        ckpt_dir / "scvi_model_finetuned",
        overwrite=True,
    )

    LOGGER.info("SOLO/SCVI checkpoint finetuned saved successfully.")
# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_process_and_integrate(cfg: ProcessAndIntegrateConfig) -> ad.AnnData:
    """
    Full module:
      1. Load filtered dataset
      2. Run SCVI + SOLO (doublet detection)
      3. Cleanup (doublets, mt, tiny samples)
      4. Normalise + HVGs
      5. PCA / neighbors / UMAP (unintegrated) + Leiden clustering
      6. Run integration methods (BBKNN, scVI, scANVI) using pre-trained SCVI
      7. Benchmark with scIB and choose best embedding
      8. Compute final integrated neighbors + UMAP
      9. Save dataset (Zarr by default, optional H5AD)
    """

    # ---------------------------------------------------------
    # Logging & figures
    # ---------------------------------------------------------
    setup_logging(cfg.logfile)
    LOGGER.info("Starting process-and-integrate")

    # Figure root + Scanpy figdir
    figroot = cfg.output_dir / cfg.figdir_name
    plot_utils.setup_scanpy_figs(figroot, cfg.figure_formats)

    figdir_integration = figroot / "integration"
    figdir_integration.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load input
    # ---------------------------------------------------------
    in_path = Path(cfg.input_path)
    adata_full = io_utils.load_dataset(in_path)

    # Batch key
    batch_key = cfg.batch_key or adata_full.uns.get("batch_key", None)
    if batch_key is None:
        raise RuntimeError(
            "batch_key not provided in config and not found in adata.uns['batch_key']."
        )
    cfg.batch_key = batch_key  # keep in config for downstream calls
    LOGGER.info("Using batch_key='%s'", batch_key)

    # ---------------------------------------------------------
    # 1) SCVI + SOLO doublet detection
    # ---------------------------------------------------------
    LOGGER.info("Running SOLO + SCVI on full filtered dataset...")
    adata_full, scvi_model = run_solo_with_scvi(
        adata_full,
        batch_key=batch_key,
        doublet_score_threshold=cfg.doublet_score_threshold,
        restrict_to_batch=cfg.solo_restrict_to_batch,
    )

    # ---------------------------------------------------------
    # CHECKPOINT (critical safety net)
    # ---------------------------------------------------------
    _save_solo_checkpoint_no_clean(
        outdir=cfg.output_dir,
        scvi_model=scvi_model,
        adata_full=adata_full,
    )

    # ---------------------------------------------------------
    # 2) Cleanup after SOLO
    # ---------------------------------------------------------
    LOGGER.info("Running cleanup after SOLO (doublets, mt, tiny samples)...")
    adata = cleanup_after_solo(
        adata_full,
        batch_key=batch_key,
        max_pct_mt=cfg.max_pct_mt,
        min_cells_per_sample=cfg.min_cells_per_sample,
    )

    # ---------------------------------------------------------
    # CHECKPOINT (critical safety net)
    # ---------------------------------------------------------
    _save_solo_checkpoint_clean(
        outdir=cfg.output_dir,
        adata_cleaned=adata,
    )

    LOGGER.info(
        "Post-SOLO cleanup dataset: %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )

    # ---------------------------------------------------------
    # OPTIONAL: SCVI fine-tuning on the cleaned dataset
    # ---------------------------------------------------------
    if cfg.scvi_refine_after_solo:
        LOGGER.info(
            "Fine-tuning SCVI on cleaned dataset for %d epochs...",
            cfg.scvi_refine_epochs,
        )

        # Device detection (same as SCANVI/SOLO)
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = "auto"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = "auto"

        # Attach new AnnDataManager for the filtered dataset
        from scvi.model import SCVI
        SCVI.setup_anndata(
            adata,
            layer="counts" if "counts" in adata.layers else None,
            batch_key=batch_key,
        )
        scvi_model._adata_manager = SCVI._get_adata_manager(adata)  # swap managers safely

        # Fine-tune SCVI in place
        scvi_model.train(
            max_epochs=cfg.scvi_refine_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=False,
            early_stopping=True,
        )

        LOGGER.info("SCVI fine-tune completed.")
    else:
        LOGGER.info("Skipping SCVI fine-tuning (cfg.scvi_refine_after_solo=False).")

    # ---------------------------------------------------------
    # CHECKPOINT (critical safety net)
    # ---------------------------------------------------------
    _save_solo_checkpoint_finetuned(
        outdir=cfg.output_dir,
        scvi_model=scvi_model,
    )
    # ---------------------------------------------------------
    # 3) Normalisation + HVG + PCA/UMAP + Leiden (unintegrated)
    # ---------------------------------------------------------
    LOGGER.info("Normalising and selecting HVGs...")
    adata = normalize_and_hvg(
        adata,
        n_top_genes=cfg.n_top_genes,
        batch_key=batch_key,
    )

    LOGGER.info("Running PCA / neighbors / UMAP (unintegrated)...")
    adata = pca_neighbors_umap(
        adata,
        max_pcs_plot=cfg.max_pcs_plot,
    )

    LOGGER.info("Running Leiden clustering on unintegrated space...")
    adata = cluster_unintegrated(
        adata,
        resolution=1.0,
        random_state=42,
    )

    # Ensure label_key for scANVI / scIB
    label_key = cfg.label_key
    if label_key not in adata.obs:
        LOGGER.warning(
            "label_key='%s' not found in adata.obs. "
            "scANVI and some scIB metrics may fail.",
            label_key,
        )
    else:
        LOGGER.info("Using label_key='%s' for scANVI + scIB.", label_key)

    # ---------------------------------------------------------
    # 4) Integration methods using pre-trained SCVI
    # ---------------------------------------------------------
    methods = cfg.methods or ["bbknn", "scvi", "scanvi"]
    LOGGER.info("Running integration methods: %s", methods)

    emb_keys = _run_integrations_with_existing_scvi(
        adata,
        scvi_model=scvi_model,
        methods=methods,
        batch_key=batch_key,
        label_key=label_key,
    )

    LOGGER.info("Created integration embeddings: %s", emb_keys)

    # ---------------------------------------------------------
    # 5) Benchmark and choose best embedding
    # ---------------------------------------------------------
    best = _select_best_embedding(
        adata,
        embedding_keys=emb_keys,
        batch_key=batch_key,
        label_key=label_key,
        n_jobs=cfg.benchmark_n_jobs,
        figdir=figdir_integration,
    )

    # ---------------------------------------------------------
    # 6) Final integrated neighbors + UMAP
    # ---------------------------------------------------------
    LOGGER.info("Computing final integrated neighbors + UMAP using '%s' embedding.", best)

    adata.obsm["X_integrated"] = adata.obsm[best]
    sc.pp.neighbors(adata, use_rep="X_integrated")
    sc.tl.umap(adata)

    # Store metadata
    adata.uns.setdefault("integration", {})
    adata.uns["integration"].update(
        {
            "methods": emb_keys,
            "best_embedding": best,
            "batch_key": batch_key,
            "label_key": label_key,
            "input_path": str(in_path),
            "metrics_raw_tsv": str(figdir_integration.parent / "integration_metrics_raw.tsv"),
            "metrics_scaled_tsv": str(figdir_integration.parent / "integration_metrics_scaled.tsv"),
        }
    )

    # ---------------------------------------------------------
    # 7) Plot per-method UMAPs (same pattern as old integrate)
    # ---------------------------------------------------------
    import matplotlib.pyplot as plt

    for method in emb_keys:
        try:
            LOGGER.info("Plotting UMAPs for method: %s", method)

            # Single-panel integrated UMAP
            tmp = adata.copy()
            sc.pp.neighbors(tmp, use_rep=method)
            sc.tl.umap(tmp)

            fig = sc.pl.umap(
                tmp,
                color=batch_key,
                show=False,
                return_fig=True,
            )
            plot_utils.save_multi(f"{method}_umap", figdir_integration)
            plt.close(fig)

            # Two-panel: method vs Unintegrated
            if method == "Unintegrated":
                continue

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
                color=batch_key,
                ax=axs[0],
                show=False,
                title=f"{method}",
            )

            tmp2.obsm["X_umap"] = umap_unintegrated
            sc.pl.umap(
                tmp2,
                color=batch_key,
                ax=axs[1],
                show=False,
                title="Unintegrated",
            )

            plot_utils.save_multi(f"{method}_vs_Unintegrated_umap", figdir_integration)
            plt.close(fig)
        except Exception as e:
            LOGGER.warning("UMAP plotting for %s failed: %s", method, e)

    # ---------------------------------------------------------
    # 8) Save integrated dataset
    # ---------------------------------------------------------
    out_stem = cfg.output_dir / (cfg.output_name + ".integrated")
    out_zarr = out_stem.with_suffix(".zarr")

    LOGGER.info("Saving integrated dataset as Zarr → %s", out_zarr)
    io_utils.save_dataset(adata, out_zarr, fmt="zarr")

    if getattr(cfg, "save_h5ad", False):
        out_h5ad = out_stem.with_suffix(".h5ad")
        LOGGER.warning(
            "User requested H5AD output in addition to Zarr. "
            "This may be large for big datasets."
        )
        io_utils.save_dataset(adata, out_h5ad, fmt="h5ad")
        LOGGER.info("Saved integrated H5AD → %s", out_h5ad)

    LOGGER.info("Finished process-and-integrate")
    return adata
