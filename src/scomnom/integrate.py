# src/scomnom/integrate.py

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
    """
    Run Scanorama integration and return a global embedding (n_cells × dim).
    """
    import numpy as np
    import scanorama
    from scipy import sparse

    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key!r} not found in adata.obs")

    LOGGER.info("Running Scanorama integration")

    # -----------------------------
    # Use HVGs if available
    # -----------------------------
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        if hvg_mask.sum() >= 200:  # sanity floor
            adata_run = adata[:, hvg_mask].copy()
        else:
            LOGGER.warning(
                "highly_variable present but too few HVGs (%d); using all genes for Scanorama",
                int(hvg_mask.sum()),
            )
            adata_run = adata.copy()
    else:
        adata_run = adata.copy()

    # Ensure float32 (Scanorama may densify internally; this helps memory)
    if sparse.issparse(adata_run.X):
        adata_run.X = adata_run.X.astype(np.float32)
    else:
        adata_run.X = np.asarray(adata_run.X, dtype=np.float32)

    # -----------------------------
    # Split by batch
    # -----------------------------
    batches = adata_run.obs[batch_key].astype("category").cat.categories.tolist()
    adatas = [adata_run[adata_run.obs[batch_key] == b].copy() for b in batches]

    # -----------------------------
    # Run Scanorama
    # -----------------------------
    # Scanorama produces an embedding in each batch object: obsm["X_scanorama"]
    MAX_SCANORAMA_DIM = 20
    dim = MAX_SCANORAMA_DIM  # Scanorama "dimred" space (not PCA space)

    adatas_cor = scanorama.correct_scanpy(
        adatas,
        return_dimred=True,
        dimred=dim,
        verbose=False,
    )

    # -----------------------------
    # Reassemble global embedding (n_cells × dim)
    # -----------------------------
    Z = np.zeros((adata.n_obs, dim), dtype=np.float32)

    for b, sub in zip(batches, adatas_cor):
        if "X_scanorama" not in sub.obsm:
            # If this happens, something truly failed upstream
            LOGGER.warning(
                "Scanorama did not produce X_scanorama for batch %s; using PCA slice as fallback",
                str(b),
            )
            # fallback: if PCA exists, take first dim components; otherwise zeros remain
            if use_rep in adata.obsm and adata.obsm[use_rep].shape[1] >= dim:
                idx = adata.obs_names.get_indexer(sub.obs_names)
                Z[idx] = np.asarray(adata.obsm[use_rep][idx, :dim], dtype=np.float32)
            continue

        idx = adata.obs_names.get_indexer(sub.obs_names)
        emb = np.asarray(sub.obsm["X_scanorama"], dtype=np.float32)

        if emb.shape[1] != dim:
            raise RuntimeError(
                f"Scanorama embedding dim mismatch for batch {b}: got {emb.shape[1]} expected {dim}"
            )

        Z[idx] = emb

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
    Run requested integration methods and write embeddings / graphs into adata.

    Policy:
      - PCA is ALWAYS recomputed on HVGs (mask_var='highly_variable') to guarantee substrate correctness.
      - Harmony/Scanorama/BBKNN operate on HVG PCA.
      - scVI/scANVI/scPoli are trained on HVGs only; embeddings are written back to full adata.
      - BBKNN is graph-only (no embedding in .obsm). We still include "BBKNN" in `created`
        so downstream plotting can generate a BBKNN UMAP using the BBKNN neighbor graph.
    """
    # ----------------------------
    # Preconditions: HVGs
    # ----------------------------
    if "highly_variable" not in adata.var:
        raise RuntimeError(
            "Expected HVGs in adata.var['highly_variable'] (from load-and-filter). "
            "Refusing to integrate on all genes implicitly."
        )

    hvg_mask = adata.var["highly_variable"].to_numpy()
    if hvg_mask.sum() == 0:
        raise RuntimeError("adata.var['highly_variable'] is present but selects 0 genes.")

    # ----------------------------
    # PCA on HVGs (ALWAYS recompute)
    # ----------------------------
    LOGGER.info("Recomputing PCA on HVGs (mask_var='highly_variable')")

    # Wipe existing PCA/related state to avoid confusion
    adata.obsm.pop("X_pca", None)
    adata.uns.pop("pca", None)
    for k in ("PCs", "variance_ratio", "variance"):
        # scanpy stores under adata.uns["pca"], but keep this defensive
        try:
            adata.varm.pop(k, None)
        except Exception:
            pass

    sc.tl.pca(
        adata,
        mask_var="highly_variable",
        svd_solver="arpack",
    )

    if "X_pca" not in adata.obsm:
        raise RuntimeError("PCA failed: adata.obsm['X_pca'] missing after recompute.")

    # Baseline embedding for benchmarking
    adata.obsm["Unintegrated"] = np.asarray(adata.obsm["X_pca"])
    created: List[str] = ["Unintegrated"]

    method_set = {m.lower() for m in (methods or [])}

    # Helper: ensure a full-size embedding is aligned to full adata obs_names
    def _write_embedding_from_hvg(full_key: str, Z_hvg: np.ndarray, obs_names_hvg) -> None:
        if Z_hvg.shape[0] != len(obs_names_hvg):
            raise RuntimeError(f"{full_key}: latent rows != HVG adata cells")

        if not np.array_equal(np.asarray(obs_names_hvg), np.asarray(adata.obs_names)):
            order = pd.Index(obs_names_hvg).get_indexer(adata.obs_names)
            if (order < 0).any():
                raise RuntimeError(f"{full_key}: HVG adata missing cells from full adata")
            Z_full = Z_hvg[order, :]
        else:
            Z_full = Z_hvg

        adata.obsm[full_key] = np.asarray(Z_full)

    # Create HVG-only AnnData for scVI-family training
    adata_hvg = adata[:, hvg_mask].copy()

    # ----------------------------
    # Harmony (HVG PCA)
    # ----------------------------
    if "harmony" in method_set:
        try:
            emb = _run_harmony(
                adata,
                batch_key=batch_key,
                use_rep="X_pca",
            )
            adata.obsm["Harmony"] = np.asarray(emb)
            created.append("Harmony")
        except Exception as e:
            LOGGER.warning("Harmony failed: %s", e)

    # ----------------------------
    # Scanorama (HVG PCA)
    # ----------------------------
    if "scanorama" in method_set:
        try:
            emb = _run_scanorama(
                adata,
                batch_key=batch_key,
                use_rep="X_pca",
            )
            adata.obsm["Scanorama"] = np.asarray(emb)
            created.append("Scanorama")
        except Exception as e:
            LOGGER.warning("Scanorama failed: %s", e)

    # ----------------------------
    # scVI/scANVI (HVGs)
    # ----------------------------
    scvi_model = None
    if "scvi" in method_set or "scanvi" in method_set:
        try:
            LOGGER.info("Training scVI backbone on HVGs for integration")

            layer = _get_scvi_layer(adata_hvg)  # counts_cb > counts_raw > None
            scvi_model = _train_scvi(
                adata_hvg,
                batch_key=batch_key,
                layer=layer,
                purpose="integration",
            )

            if "scvi" in method_set:
                Z = np.asarray(scvi_model.get_latent_representation(adata_hvg))
                _write_embedding_from_hvg("scVI", Z, adata_hvg.obs_names)
                created.append("scVI")

            if "scanvi" in method_set:
                Zs = _run_scanvi_from_scvi(scvi_model, adata_hvg, label_key)
                _write_embedding_from_hvg("scANVI", np.asarray(Zs), adata_hvg.obs_names)
                created.append("scANVI")

        except Exception as e:
            LOGGER.warning("scVI/scANVI failed: %s", e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------------------
    # scPoli (HVGs)
    # ----------------------------
    if "scpoli" in method_set:
        try:
            Z = _run_scpoli(
                adata_hvg,
                batch_key=batch_key,
                label_key=label_key,
            )
            _write_embedding_from_hvg("scPoli", np.asarray(Z), adata_hvg.obs_names)
            created.append("scPoli")
        except Exception as e:
            LOGGER.warning("scPoli failed: %s", e)

    # ----------------------------
    # BBKNN (graph-only; HVG PCA)
    # ----------------------------
    if "bbknn" in method_set:
        try:
            LOGGER.info("Running BBKNN (graph-only baseline) on HVG PCA")
            _run_bbknn(adata, batch_key=batch_key)
            created.append("BBKNN")  # graph-only; no adata.obsm["BBKNN"]
        except Exception as e:
            LOGGER.warning("BBKNN failed: %s", e)

    # ----------------------------
    # Safety: verify embedding keys that are supposed to exist
    # ----------------------------
    expected_obsm = [k for k in created if k not in ("BBKNN",)]
    missing = [k for k in expected_obsm if k not in adata.obsm]
    if missing:
        raise RuntimeError(f"Integration embeddings missing from adata.obsm: {missing}")

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
    *,
    benchmark_threshold: int = 100000,
    benchmark_n_cells: int = 100000,
    benchmark_random_state: int = 42,
) -> str:
    """
    Run scIB benchmarking and select the best integration embedding.

      - If adata.n_obs > benchmark_threshold: stratified subsample by batch_key
        to benchmark_n_cells (default 100k), then run scIB on that subset.
      - Else: run scIB on full data.

    Selection strategy (baseline-aware):
      1) batch > Unintegrated AND bio > Unintegrated
      2) bio > Unintegrated
      3) batch > Unintegrated
      4) fallback: highest Total

    Returns:
      best embedding key (string)
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

    # --------------------------------------------------
    # Subsample (stratified by batch) if needed
    # --------------------------------------------------
    n_total = int(adata.n_obs)
    do_subsample = (
        benchmark_threshold is not None
        and benchmark_n_cells is not None
        and n_total > int(benchmark_threshold)
        and int(benchmark_n_cells) < n_total
    )

    if do_subsample:
        if batch_key not in adata.obs:
            raise KeyError(
                f"batch_key '{batch_key}' not found in adata.obs; cannot stratify subsample."
            )

        target = int(benchmark_n_cells)
        rng = np.random.default_rng(int(benchmark_random_state))

        # batch sizes
        batch_series = adata.obs[batch_key].astype("category")
        batches = batch_series.cat.categories.tolist()
        batch_counts = batch_series.value_counts().reindex(batches).to_numpy()
        batch_props = batch_counts / batch_counts.sum()

        # initial allocation
        alloc = np.floor(batch_props * target).astype(int)

        # ensure we don't allocate more than available per batch
        alloc = np.minimum(alloc, batch_counts)

        # ensure at least 1 from each batch that exists (if target allows)
        if target >= len(batches):
            alloc = np.maximum(alloc, (batch_counts > 0).astype(int))
            alloc = np.minimum(alloc, batch_counts)

        # fix rounding to hit exact target (as close as possible)
        current = int(alloc.sum())
        leftover = target - current

        if leftover > 0:
            # distribute leftover to batches with remaining capacity
            capacity = batch_counts - alloc
            eligible = np.where(capacity > 0)[0]
            if eligible.size > 0:
                # weight by remaining capacity
                weights = capacity[eligible].astype(float)
                weights = weights / weights.sum() if weights.sum() > 0 else None
                picks = rng.choice(eligible, size=leftover, replace=True, p=weights)
                for i in picks:
                    if alloc[i] < batch_counts[i]:
                        alloc[i] += 1

        elif leftover < 0:
            # remove extras from batches with alloc > 1 (or >0 if needed)
            remove = -leftover
            removable = np.where(alloc > 0)[0]
            if removable.size > 0:
                weights = alloc[removable].astype(float)
                weights = weights / weights.sum() if weights.sum() > 0 else None
                picks = rng.choice(removable, size=remove, replace=True, p=weights)
                for i in picks:
                    if alloc[i] > 0:
                        alloc[i] -= 1

        # final indices
        picked_obs = []
        obs_names = adata.obs_names.to_numpy()

        for b, k in zip(batches, alloc):
            if k <= 0:
                continue
            idx = np.where(batch_series.to_numpy() == b)[0]
            if idx.size == 0:
                continue
            if k >= idx.size:
                chosen = idx
            else:
                chosen = rng.choice(idx, size=int(k), replace=False)
            picked_obs.append(chosen)

        if not picked_obs:
            raise RuntimeError("Stratified subsample produced zero cells (unexpected).")

        picked = np.concatenate(picked_obs)
        rng.shuffle(picked)

        # Safety: clip to target (can happen due to min-1 logic)
        if picked.size > target:
            picked = picked[:target]

        LOGGER.info(
            "scIB benchmarking: stratified subsample enabled (%d → %d cells; threshold=%d)",
            n_total,
            int(picked.size),
            int(benchmark_threshold),
        )

        # Copy is important because Benchmarker writes neighbors/graphs into adata
        adata_bm = adata[picked].copy()

    else:
        LOGGER.info(
            "scIB benchmarking: using full dataset (%d cells; threshold=%s)",
            n_total,
            str(benchmark_threshold),
        )
        adata_bm = adata.copy()

    # --------------------------------------------------
    # Run benchmarking
    # --------------------------------------------------
    LOGGER.info(
        "Running scIB benchmarking on embeddings: %s",
        benchmark_embeddings,
    )

    bm = Benchmarker(
        adata_bm,
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

    raw.to_csv(Path(output_dir) / "integration_metrics_raw.tsv", sep="\t")
    scaled.to_csv(Path(output_dir) / "integration_metrics_scaled.tsv", sep="\t")

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
        return str(numeric["Total"].idxmax())

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
    method_class = METHOD_CLASS.get(str(best), "unknown")

    LOGGER.info(
        "Selected best embedding: '%s' (%s) — reason: %s",
        str(best),
        method_class,
        reason,
    )

    if method_class == "supervised":
        unsup = [m for m in numeric.index if METHOD_CLASS.get(m) == "unsupervised"]
        if unsup:
            best_unsup = numeric.loc[unsup, "Total"].idxmax()
            LOGGER.info(
                "Best unsupervised alternative: '%s' (Total=%.3f)",
                str(best_unsup),
                float(numeric.loc[best_unsup, "Total"]),
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

    plot_keys = list(emb_keys)
    if "bbknn" in {m.lower() for m in methods}:
        plot_keys.append("BBKNN")

    plot_keys = list(dict.fromkeys(plot_keys))

    plot_utils.plot_integration_umaps(
        adata,
        embedding_keys=plot_keys,
        batch_key=batch_key,
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

    adata.uns.setdefault("integration", {})
    adata.uns["integration"].update({
        "best_embedding": best,
        "available_embeddings": list(emb_keys),
        "selection_timestamp": datetime.utcnow().isoformat(),
    })

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

