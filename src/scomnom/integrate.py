from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import anndata as ad
import numpy as np
import scanpy as sc

from .load_and_qc import setup_logging
from . import io_utils

LOGGER = logging.getLogger(__name__)


DEFAULT_METHODS: tuple[str, ...] = ("Scanorama", "LIGER", "Harmony", "scVI")
SCANVI_NAME = "scANVI"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _subset_to_hvgs(adata: ad.AnnData) -> ad.AnnData:
    if "highly_variable" not in adata.var:
        raise ValueError(
            "adata.var['highly_variable'] is missing. "
            "Run load_and_qc first to compute HVGs."
        )
    mask = adata.var["highly_variable"].to_numpy()
    if mask.sum() == 0:
        raise ValueError("No highly_variable genes found in adata.var.")
    return adata[:, mask].copy()


def _infer_batch_key(adata: ad.AnnData, batch_key: Optional[str]) -> str:
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
        return batch_key

    # Heuristics consistent with load_and_qc defaults
    for candidate in ("sample", "sample_id", "batch"):
        if candidate in adata.obs:
            LOGGER.info("Using inferred batch_key='%s'", candidate)
            return candidate

    raise KeyError(
        "Could not infer batch key. "
        "Please provide batch_key explicitly (e.g. 'sample')."
    )


def _ensure_label_key(adata: ad.AnnData, label_key: str) -> None:
    if label_key not in adata.obs:
        raise KeyError(
            f"label_key '{label_key}' not found in adata.obs. "
            "Run clustering (leiden) before integration benchmarking."
        )


def _run_scanorama_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    try:
        import scanorama
    except ImportError as e:
        raise RuntimeError(
            "Scanorama is not installed but 'Scanorama' was requested.\n"
            "Install with `pip install scanorama`."
        ) from e

    LOGGER.info("Running Scanorama integration")
    batch_cats = adata.obs[batch_key].unique().tolist()
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]

    # In-place integration
    scanorama.integrate_scanpy(adata_list)

    n_cells = adata.n_obs
    dim = adata_list[0].obsm["X_scanorama"].shape[1]
    emb = np.zeros((n_cells, dim), dtype=adata_list[0].obsm["X_scanorama"].dtype)

    for i, b in enumerate(batch_cats):
        mask = (adata.obs[batch_key] == b).to_numpy()
        emb[mask] = adata_list[i].obsm["X_scanorama"]

    return emb


def _run_liger_embedding(adata: ad.AnnData, batch_key: str, k: int = 30) -> np.ndarray:
    try:
        import pyliger
    except ImportError as e:
        raise RuntimeError(
            "pyliger is not installed but 'LIGER' was requested.\n"
            "Install with `pip install pyliger`."
        ) from e

    if "counts" not in adata.layers:
        raise ValueError(
            "adata.layers['counts'] is missing. "
            "load_and_qc should have created this layer."
        )

    LOGGER.info("Running LIGER integration (k=%d)", k)

    bdata = adata.copy()
    # LIGER expects (normalized) counts; here we give it counts layer
    X_counts = bdata.layers["counts"]
    # pyliger.create_liger with make_sparse=False expects dense arrays
    bdata.X = X_counts.toarray() if hasattr(X_counts, "toarray") else X_counts

    batch_cats = bdata.obs[batch_key].unique().tolist()
    adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]

    for i, ad_batch in enumerate(adata_list):
        ad_batch.uns["sample_name"] = batch_cats[i]
        ad_batch.uns["var_gene_idx"] = np.arange(bdata.n_vars)

    liger_data = pyliger.create_liger(
        adata_list, remove_missing=False, make_sparse=False
    )
    liger_data.var_genes = bdata.var_names

    pyliger.normalize(liger_data)
    pyliger.scale_not_center(liger_data)
    pyliger.optimize_ALS(liger_data, k=k)
    pyliger.quantile_norm(liger_data)

    n_cells = adata.n_obs
    dim = liger_data.adata_list[0].obsm["H_norm"].shape[1]
    emb = np.zeros((n_cells, dim), dtype=liger_data.adata_list[0].obsm["H_norm"].dtype)

    for i, b in enumerate(batch_cats):
        mask = (adata.obs[batch_key] == b).to_numpy()
        emb[mask] = liger_data.adata_list[i].obsm["H_norm"]

    return emb


def _run_harmony_embedding(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    try:
        import harmonypy as hm
    except ImportError as e:
        raise RuntimeError(
            "harmonypy is not installed but 'Harmony' was requested.\n"
            "Install with `pip install harmonypy`."
        ) from e

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "adata.obsm['X_pca'] is missing. "
            "load_and_qc should have computed PCA."
        )

    LOGGER.info("Running Harmony integration")
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key)
    return ho.Z_corr.T


def _fit_scvi(
    adata: ad.AnnData,
    batch_key: str,
    n_latent: int = 30,
    max_epochs: int = 400,
    use_gpu: bool = False,
    seed: int = 0,
):
    try:
        import scvi
    except ImportError as e:
        raise RuntimeError(
            "scvi-tools is not installed but 'scVI'/'scANVI' was requested.\n"
            "Install with `pip install scvi-tools`."
        ) from e

    if "counts" not in adata.layers:
        raise ValueError(
            "adata.layers['counts'] is missing. "
            "load_and_qc should have created this layer."
        )

    LOGGER.info(
        "Fitting scVI (n_latent=%d, max_epochs=%d, use_gpu=%s)",
        n_latent,
        max_epochs,
        use_gpu,
    )

    scvi.settings.seed = seed
    scvi.settings.dl_num_workers = 0

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(
        adata, gene_likelihood="nb", n_layers=2, n_latent=n_latent
    )

    accelerator = "gpu" if use_gpu else "cpu"
    # Let scvi-tools decide devices when running on GPU; ignore on CPU
    train_kwargs = dict(
        max_epochs=max_epochs,
        accelerator=accelerator,
        enable_progress_bar=False,
        early_stopping=True,
    )
    if use_gpu:
        train_kwargs["devices"] = "auto"

    model.train(**train_kwargs)
    emb = model.get_latent_representation()
    return model, emb


def _run_scanvi_from_scvi(
    scvi_model,
    adata: ad.AnnData,
    label_key: str,
    unlabeled_category: str = "Unknown",
    max_epochs: int = 400,
    use_gpu: bool = False,
) -> np.ndarray:
    try:
        from scvi.model import SCANVI
    except ImportError as e:
        raise RuntimeError(
            "scvi-tools is not installed but 'scANVI' was requested.\n"
            "Install with `pip install scvi-tools`."
        ) from e

    if label_key not in adata.obs:
        raise KeyError(
            f"label_key '{label_key}' not found in adata.obs, "
            "required for scANVI."
        )

    LOGGER.info(
        "Fitting scANVI (label_key=%s, max_epochs=%d, use_gpu=%s)",
        label_key,
        max_epochs,
        use_gpu,
    )

    lvae = SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category=unlabeled_category,
    )

    accelerator = "gpu" if use_gpu else "cpu"
    train_kwargs = dict(
        max_epochs=max_epochs,
        n_samples_per_label=100,
        accelerator=accelerator,
        enable_progress_bar=False,
        early_stopping=True,
    )
    if use_gpu:
        train_kwargs["devices"] = "auto"

    lvae.train(**train_kwargs)
    return lvae.get_latent_representation()


def _run_all_embeddings(
    adata: ad.AnnData,
    methods: Sequence[str],
    batch_key: str,
    label_key: str,
    use_gpu: bool,
) -> List[str]:
    """
    Run selected integration methods on `adata` (HVG subset) and
    store embeddings in `adata.obsm`.

    Returns the list of obsm keys that were successfully created.
    """
    # Baseline
    if "X_pca" not in adata.obsm:
        raise ValueError(
            "adata.obsm['X_pca'] is missing. "
            "load_and_qc should have computed PCA before integration."
        )
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    created: List[str] = ["Unintegrated"]

    method_set = {m.lower() for m in methods}
    LOGGER.info("Requested integration methods: %s", ", ".join(methods))

    # Methods that depend on scVI
    needs_scvi = {"scvi", "scanvi"} & method_set
    scvi_model = None
    if needs_scvi:
        scvi_model, scvi_emb = _fit_scvi(adata, batch_key=batch_key, use_gpu=use_gpu)
        adata.obsm["scVI"] = scvi_emb
        created.append("scVI")

    for m in methods:
        key = m.lower()
        try:
            if key == "scanorama":
                emb = _run_scanorama_embedding(adata, batch_key=batch_key)
                adata.obsm["Scanorama"] = emb
                created.append("Scanorama")

            elif key == "liger":
                emb = _run_liger_embedding(adata, batch_key=batch_key)
                adata.obsm["LIGER"] = emb
                created.append("LIGER")

            elif key == "harmony":
                emb = _run_harmony_embedding(adata, batch_key=batch_key)
                adata.obsm["Harmony"] = emb
                created.append("Harmony")

            elif key == "scvi":
                # Already run above
                continue

            elif key == "scanvi":
                if scvi_model is None:
                    raise RuntimeError(
                        "scANVI requested but scVI model was not fitted."
                    )
                emb = _run_scanvi_from_scvi(
                    scvi_model, adata, label_key=label_key, use_gpu=use_gpu
                )
                adata.obsm[SCANVI_NAME] = emb
                created.append(SCANVI_NAME)

            else:
                LOGGER.warning("Unknown integration method '%s'; skipping", m)

        except ImportError as e:
            LOGGER.warning(
                "Skipping method '%s' because a required package is missing: %s",
                m,
                e,
            )
        except Exception as e:
            LOGGER.exception("Integration method '%s' failed: %s", m, e)

    LOGGER.info("Successfully computed embeddings: %s", ", ".join(created))
    return created


def _select_best_embedding(
    adata: ad.AnnData,
    embedding_keys: Sequence[str],
    batch_key: str,
    label_key: str,
    n_jobs: int = 1,
    figdir: Optional[Path] = None,
) -> str:
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    import pandas as pd

    LOGGER.info("Running scib-metrics benchmark")
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

    if figdir is not None:
        figdir.mkdir(parents=True, exist_ok=True)
        bm.plot_results_table(min_max_scale=False, save_dir=str(figdir))

    # Raw + scaled results to disk for inspection
    results_raw = bm.get_results(min_max_scale=False)
    results_scaled = bm.get_results(min_max_scale=True)

    if figdir is not None:
        raw_path = figdir / "integration_metrics_raw.tsv"
        scaled_path = figdir / "integration_metrics_scaled.tsv"
        results_raw.to_csv(raw_path, sep="\t")
        results_scaled.to_csv(scaled_path, sep="\t")
        LOGGER.info("Wrote raw metrics to %s", raw_path)
        LOGGER.info("Wrote scaled metrics to %s", scaled_path)

    # Select best based on mean of min-max scaled metrics
    df = results_scaled
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric metrics found in scib-metrics results.")

    scores = df[numeric_cols].mean(axis=1)
    # In scib-metrics, index is the embedding name ("Embedding" index)
    best_embedding = scores.idxmax()

    if isinstance(best_embedding, tuple):
        # Defensive: in case of MultiIndex
        best_embedding = best_embedding[0]

    LOGGER.info(
        "Best integration embedding according to scib-metrics: %s", best_embedding
    )
    return str(best_embedding)


def run_integration(cfg: IntegrationConfig) -> ad.AnnData:
    """
    Orchestrate integration + benchmarking on a preprocessed AnnData.

    Reads cfg.input_path, runs requested integration methods on HVGs,
    benchmarks them using scib-metrics, selects the best embedding,
    recomputes neighbors/UMAP on the full adata using that embedding,
    writes the integrated file to disk, and returns the full AnnData.
    """
    setup_logging(cfg.logfile)

    in_path = cfg.input_path
    out_path = (
        cfg.output_path
        if cfg.output_path is not None
        else in_path.with_name(f"{in_path.stem}.integrated.h5ad")
    )

    LOGGER.info("Reading preprocessed AnnData from %s", in_path)
    full_adata = sc.read_h5ad(str(in_path))

    # Resolve batch key
    batch_key = _infer_batch_key(full_adata, batch_key=cfg.batch_key)

    # HVG subset for integration + benchmarking
    hvg_adata = _subset_to_hvgs(full_adata)

    # Ensure PCA exists
    if "X_pca" not in hvg_adata.obsm:
        LOGGER.info("X_pca missing on HVG subset; computing PCA")
        sc.tl.pca(hvg_adata)

    # Determine methods
    if cfg.methods is None:
        methods = list(DEFAULT_METHODS)
        if cfg.include_scanvi:
            methods.append("scANVI")
    else:
        methods = cfg.methods

    LOGGER.info(
        "Starting integration on HVGs (n_obs=%d, n_vars=%d) using methods: %s",
        hvg_adata.n_obs,
        hvg_adata.n_vars,
        ", ".join(methods),
    )

    # Run all requested embeddings
    embedding_keys = _run_all_embeddings(
        hvg_adata,
        methods=methods,
        batch_key=batch_key,
        label_key=cfg.label_key,
        use_gpu=cfg.use_gpu,
    )

    # Benchmark
    figdir = in_path.parent / "figures" / "integration"
    best_embedding = _select_best_embedding(
        hvg_adata,
        embedding_keys=embedding_keys,
        batch_key=batch_key,
        label_key=cfg.label_key,
        n_jobs=cfg.benchmark_n_jobs,
        figdir=figdir,
    )

    # Transfer embeddings back to full data
    for key in embedding_keys:
        full_adata.obsm[key] = hvg_adata.obsm[key]

    # Alias final embedding
    full_adata.obsm["X_integrated"] = full_adata.obsm[best_embedding]

    LOGGER.info("Computing neighbors/UMAP using best embedding '%s'", best_embedding)
    sc.pp.neighbors(full_adata, use_rep="X_integrated")
    sc.tl.umap(full_adata)

    # Metadata
    full_adata.uns.setdefault("integration", {})
    full_adata.uns["integration"]["methods"] = embedding_keys
    full_adata.uns["integration"]["best_embedding"] = best_embedding
    full_adata.uns["integration"]["batch_key"] = batch_key
    full_adata.uns["integration"]["label_key"] = cfg.label_key
    full_adata.uns["integration"]["input_path"] = str(in_path)
    full_adata.uns["integration"]["benchmark_metrics_raw_path"] = str(
        figdir / "integration_metrics_raw.tsv"
    )
    full_adata.uns["integration"]["benchmark_metrics_scaled_path"] = str(
        figdir / "integration_metrics_scaled.tsv"
    )

    # Save
    io_utils.save_adata(full_adata, out_path)

    LOGGER.info(
        "Finished integration. Best embedding='%s'. Wrote integrated AnnData to %s",
        best_embedding,
        out_path,
    )
    return full_adata

