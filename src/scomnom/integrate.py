# src/scomnom/integrate.py

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scomnom import __version__
from . import io_utils, plot_utils, reporting
from .config import IntegrateConfig
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
    """Train an SCVI model with auto batch-size + auto epochs."""
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
) -> np.ndarray:
    """Run scANVI using an existing, already-trained SCVI model."""
    from scvi.model import SCANVI

    _ensure_label_key(adata, label_key)

    accelerator, devices = _select_device()
    max_epochs = _auto_scanvi_epochs(adata.n_obs)

    LOGGER.info(
        "Running scANVI (reuse SCVI, n_cells=%d, max_epochs=%d, labels_key=%r)",
        adata.n_obs,
        max_epochs,
        label_key,
    )

    lvae = SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )

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
# Optional: scVI-latent preflight labels for scANVI
# ---------------------------------------------------------------------
def _compute_scanvi_prelabels(
    adata_hvg: ad.AnnData,
    *,
    scvi_latent: np.ndarray,
    batch_key: str,
    out_key: str = "scanvi_prelabels",
    resolutions: Optional[Sequence[float]] = None,
    max_prelabel_clusters: int = 25,
    min_stability_ok: float = 0.60,
    parsimony_eps: float = 0.03,
    w_stab: float = 0.50,
    w_sil: float = 0.35,
    w_tiny: float = 0.15,
    batch_trap_threshold: float = 0.90,
    batch_trap_min_cells: int = 200,
    tiny_cluster_min_cells: int = 30,
) -> str:
    """
    Create coarse, structural labels for scANVI by sweeping Leiden resolutions
    on scVI latent space, then selecting a parsimonious resolution using
    BISC-inspired structural scoring:

      score = w_stab * stability_norm + w_sil * centroid_sil_norm + w_tiny * tiny_penalty_norm

    where:
      - stability = smoothed ARI between adjacent resolutions (higher is better)
      - centroid silhouette = _centroid_silhouette (higher is better)
      - tiny penalty = discourages many tiny clusters (higher is better)

    Additional selection constraints:
      - prefer resolutions with n_clusters <= max_prelabel_clusters (configurable; default 25)
      - prefer interior resolutions (exclude endpoints when possible)
      - require stability >= min_stability_ok when feasible
      - pick the *lowest* resolution within (1 - parsimony_eps) of best

    Writes:
      - adata_hvg.obsm["X_scvi_latent"]
      - adata_hvg.obs[out_key]

    Returns:
      out_key
    """
    from sklearn.metrics import adjusted_rand_score

    # Import the centroid silhouette used by BISC
    # (kept as local import to avoid circular imports in some layouts)
    try:
        from .clustering_utils import _centroid_silhouette  # type: ignore
    except Exception:
        # fallback if integrate.py is executed in a different import context
        from scomnom.clustering_utils import _centroid_silhouette  # type: ignore

    if batch_key not in adata_hvg.obs:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")

    Z = np.asarray(scvi_latent)
    if Z.ndim != 2 or Z.shape[0] != adata_hvg.n_obs:
        raise RuntimeError(
            f"scVI latent shape mismatch: got {Z.shape}, expected ({adata_hvg.n_obs}, k)"
        )

    adata_hvg.obsm["X_scvi_latent"] = Z

    # default sparse grid (coarse sweep)
    if not resolutions:
        resolutions = [0.2, 0.6, 1.0, 1.4, 1.8]

    res_list = [float(r) for r in resolutions]
    res_list = sorted(res_list)

    # Ensure neighbors once on the latent
    sc.pp.neighbors(adata_hvg, use_rep="X_scvi_latent")

    # ------------------------------------------------------------------
    # Helpers (self-contained mini-BISC)
    # ------------------------------------------------------------------
    def _normalize_scores(d: dict[float, float]) -> dict[float, float]:
        if not d:
            return {}
        vals = np.array(list(d.values()), dtype=float)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            return {k: 0.0 for k in d}
        return {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}

    def _compute_tiny_cluster_penalty(cluster_sizes: np.ndarray, tiny_threshold: int) -> float:
        """
        Higher is better. Penalizes:
          - many tiny clusters
          - many cells falling into tiny clusters
        """
        cluster_sizes = np.asarray(cluster_sizes, dtype=int)
        total_clusters = int(cluster_sizes.size)
        total_cells = int(cluster_sizes.sum())

        if total_clusters == 0 or total_cells == 0:
            return 1.0

        tiny_mask = cluster_sizes < int(tiny_threshold)
        n_tiny = int(np.sum(tiny_mask))
        cells_in_tiny = int(np.sum(cluster_sizes[tiny_mask]))

        frac_tiny_clusters = n_tiny / total_clusters
        penalty_cluster_fraction = 1.0 - frac_tiny_clusters

        frac_cells_in_tiny = cells_in_tiny / total_cells
        penalty_cell_fraction = 1.0 - frac_cells_in_tiny

        return float(0.5 * (penalty_cluster_fraction + penalty_cell_fraction))

    def _compute_smoothed_stability(
        res_sorted: list[float],
        ari_adjacent: dict[tuple[float, float], float],
    ) -> dict[float, float]:
        stab: dict[float, float] = {}
        for i, r in enumerate(res_sorted):
            terms: list[float] = []
            if i > 0:
                r_prev = res_sorted[i - 1]
                if (r_prev, r) in ari_adjacent:
                    terms.append(float(ari_adjacent[(r_prev, r)]))
            if i < len(res_sorted) - 1:
                r_next = res_sorted[i + 1]
                if (r, r_next) in ari_adjacent:
                    terms.append(float(ari_adjacent[(r, r_next)]))
            stab[r] = float(np.mean(terms)) if terms else 0.0
        return stab

    def _pick_parsimonious(cands: list[float], scores: dict[float, float], eps: float) -> float | None:
        if not cands:
            return None
        best = max(cands, key=lambda r: scores.get(r, -np.inf))
        best_val = float(scores.get(best, -np.inf))
        if not np.isfinite(best_val):
            return min(cands)
        near = [r for r in cands if float(scores.get(r, -np.inf)) >= (1.0 - float(eps)) * best_val]
        return min(near) if near else best

    # ------------------------------------------------------------------
    # Sweep: Leiden + metrics per resolution
    # ------------------------------------------------------------------
    labels_by_res: dict[float, np.ndarray] = {}
    n_clusters_by_res: dict[float, int] = {}
    sizes_by_res: dict[float, np.ndarray] = {}
    sil_by_res: dict[float, float] = {}
    tiny_by_res: dict[float, float] = {}

    for r in res_list:
        key = f"__scanvi_pre_leiden_{r:.3f}"
        sc.tl.leiden(adata_hvg, resolution=float(r), key_added=key)

        labels = adata_hvg.obs[key].astype(str).to_numpy()
        labels_by_res[r] = labels

        vc = pd.Series(labels).value_counts()
        n_clusters = int(vc.size)
        n_clusters_by_res[r] = n_clusters

        sizes = vc.to_numpy(dtype=int)
        sizes_by_res[r] = sizes

        sil = float(_centroid_silhouette(Z, labels))
        sil_by_res[r] = sil

        tiny = float(_compute_tiny_cluster_penalty(sizes, tiny_threshold=int(tiny_cluster_min_cells)))
        tiny_by_res[r] = tiny

        LOGGER.info(
            "scanVI prelabels sweep: res=%.3f -> n_clusters=%d, centroid_silhouette=%.4f, tiny_penalty=%.4f",
            float(r),
            int(n_clusters),
            float(sil) if np.isfinite(sil) else float("nan"),
            float(tiny) if np.isfinite(tiny) else float("nan"),
        )

    # ------------------------------------------------------------------
    # Stability: ARI adjacent + smoothing
    # ------------------------------------------------------------------
    ari_adjacent: dict[tuple[float, float], float] = {}
    for r1, r2 in zip(res_list[:-1], res_list[1:]):
        ari = float(adjusted_rand_score(labels_by_res[r1], labels_by_res[r2]))
        ari_adjacent[(r1, r2)] = ari

    stability_by_res = _compute_smoothed_stability(res_list, ari_adjacent)

    # ------------------------------------------------------------------
    # Composite score (normalized)
    # ------------------------------------------------------------------
    sil_norm = _normalize_scores(sil_by_res)
    tiny_norm = _normalize_scores(tiny_by_res)
    stab_norm = _normalize_scores(stability_by_res)

    composite: dict[float, float] = {}
    for r in res_list:
        composite[r] = float(
            float(w_stab) * stab_norm.get(r, 0.0)
            + float(w_sil) * sil_norm.get(r, 0.0)
            + float(w_tiny) * tiny_norm.get(r, 0.0)
        )

        LOGGER.info(
            "scanVI prelabels score: res=%.3f -> stability=%.4f (norm=%.3f), sil=%.4f (norm=%.3f), tiny=%.4f (norm=%.3f) | composite=%.4f",
            float(r),
            float(stability_by_res.get(r, 0.0)),
            float(stab_norm.get(r, 0.0)),
            float(sil_by_res.get(r, float("nan"))),
            float(sil_norm.get(r, 0.0)),
            float(tiny_by_res.get(r, float("nan"))),
            float(tiny_norm.get(r, 0.0)),
            float(composite[r]),
        )

    # ------------------------------------------------------------------
    # Candidate set selection (BISC-ish)
    # ------------------------------------------------------------------
    # Prefer interior resolutions when possible
    interior = res_list[1:-1] if len(res_list) > 2 else res_list

    # Feasible: stability >= min_stability_ok
    feasible = [r for r in interior if float(stability_by_res.get(r, 0.0)) >= float(min_stability_ok)]

    # Cap cluster count if requested
    if max_prelabel_clusters is not None and int(max_prelabel_clusters) > 0:
        feasible = [r for r in feasible if int(n_clusters_by_res.get(r, 0)) <= int(max_prelabel_clusters)]

    # If feasible set is empty, relax in stages:
    search_set = feasible
    relaxed_reason = None

    if not search_set:
        relaxed_reason = "no resolution met stability+cap constraints; relaxing constraints"
        search_set = interior.copy()

        if max_prelabel_clusters is not None and int(max_prelabel_clusters) > 0:
            capped = [r for r in search_set if int(n_clusters_by_res.get(r, 0)) <= int(max_prelabel_clusters)]
            if capped:
                search_set = capped
                relaxed_reason = "relaxed stability constraint (kept cap)"

    if not search_set:
        # absolute fallback: everything
        search_set = res_list.copy()
        relaxed_reason = "relaxed to all resolutions (cap may be impossible)"

    if relaxed_reason:
        LOGGER.warning("scanVI prelabels: %s. Candidates=%s", relaxed_reason, search_set)

    # Parsimonious pick among near-best
    best_res = _pick_parsimonious(search_set, composite, eps=float(parsimony_eps))
    if best_res is None:
        raise RuntimeError("Failed to select a preflight resolution for scANVI labels")

    chosen_key = f"__scanvi_pre_leiden_{best_res:.3f}"
    raw = adata_hvg.obs[chosen_key].astype(str)

    # ------------------------------------------------------------------
    # Guardrail 1: tiny clusters -> Unknown
    # ------------------------------------------------------------------
    counts = raw.value_counts()
    tiny_clusters = set(counts[counts < int(tiny_cluster_min_cells)].index.astype(str))

    # ------------------------------------------------------------------
    # Guardrail 2: batch trap -> Unknown
    # ------------------------------------------------------------------
    batch = adata_hvg.obs[batch_key].astype(str)
    trap_clusters = set()

    for cid, n_c in counts.items():
        cid = str(cid)
        if int(n_c) < int(batch_trap_min_cells):
            continue
        m = raw == cid
        frac = float(batch[m].value_counts(normalize=True).max())
        if frac >= float(batch_trap_threshold):
            trap_clusters.add(cid)

    unknown_clusters = sorted(tiny_clusters | trap_clusters)

    out = raw.copy()
    if unknown_clusters:
        out = out.where(~out.isin(unknown_clusters), other="Unknown")

    out = out.astype("category")
    adata_hvg.obs[out_key] = out

    # Log summary
    n_unknown = int((out.astype(str) == "Unknown").sum())
    LOGGER.info(
        "scanVI prelabels selected: res=%.3f -> n_clusters=%d; stability=%.4f; centroid_silhouette=%.4f; tiny_penalty=%.4f; composite=%.4f; Unknown=%d (tiny<%d: %d clusters; batch_trap>=%.2f & n>=%d: %d clusters; cap<=%d)",
        float(best_res),
        int(n_clusters_by_res.get(best_res, 0)),
        float(stability_by_res.get(best_res, 0.0)),
        float(sil_by_res.get(best_res, float("nan"))),
        float(tiny_by_res.get(best_res, float("nan"))),
        float(composite.get(best_res, float("nan"))),
        int(n_unknown),
        int(tiny_cluster_min_cells),
        int(len(tiny_clusters)),
        float(batch_trap_threshold),
        int(batch_trap_min_cells),
        int(len(trap_clusters)),
        int(max_prelabel_clusters),
    )

    return out_key


# ---------------------------------------------------------------------
# Scanorama
# ---------------------------------------------------------------------

def _run_scanorama(
    adata: ad.AnnData,
    batch_key: str,
    *,
    use_rep: str = "X_pca",
) -> np.ndarray:
    import scanorama
    from scipy import sparse

    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key!r} not found in adata.obs")

    LOGGER.info("Running Scanorama integration")

    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        if hvg_mask.sum() >= 200:
            adata_run = adata[:, hvg_mask].copy()
        else:
            LOGGER.warning(
                "highly_variable present but too few HVGs (%d); using all genes for Scanorama",
                int(hvg_mask.sum()),
            )
            adata_run = adata.copy()
    else:
        adata_run = adata.copy()

    if sparse.issparse(adata_run.X):
        adata_run.X = adata_run.X.astype(np.float32)
    else:
        adata_run.X = np.asarray(adata_run.X, dtype=np.float32)

    batches = adata_run.obs[batch_key].astype("category").cat.categories.tolist()
    adatas = [adata_run[adata_run.obs[batch_key] == b].copy() for b in batches]

    MAX_SCANORAMA_DIM = 20
    dim = MAX_SCANORAMA_DIM

    adatas_cor = scanorama.correct_scanpy(
        adatas,
        return_dimred=True,
        dimred=dim,
        verbose=False,
    )

    Z = np.zeros((adata.n_obs, dim), dtype=np.float32)

    for b, sub in zip(batches, adatas_cor):
        if "X_scanorama" not in sub.obsm:
            LOGGER.warning(
                "Scanorama did not produce X_scanorama for batch %s; using PCA slice as fallback",
                str(b),
            )
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
# Harmony
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

    Z = np.asarray(adata.obsm[use_rep])
    meta = adata.obs[[batch_key]].copy()

    ho = hm.run_harmony(
        Z,
        meta,
        vars_use=[batch_key],
        verbose=False,
    )

    Z_corr = np.asarray(ho.Z_corr)

    if Z_corr.shape == (Z.shape[1], Z.shape[0]):
        LOGGER.warning(
            "Harmony returned Z_corr as (n_pcs, n_cells)=%s; transposing to (n_cells, n_pcs).",
            Z_corr.shape,
        )
        Z_corr = Z_corr.T
    elif Z_corr.shape != Z.shape:
        raise RuntimeError(
            f"Harmony output shape mismatch: got {Z_corr.shape}, expected {Z.shape} "
            f"(or transposed {(Z.shape[1], Z.shape[0])})."
        )

    if Z_corr.shape[0] != adata.n_obs:
        raise RuntimeError(f"Harmony output shape mismatch: {Z_corr.shape}")

    return Z_corr


# ---------------------------------------------------------------------
# Integration runners
# ---------------------------------------------------------------------

def _run_integrations(
    adata: ad.AnnData,
    cfg: IntegrateConfig,
    *,
    methods: Sequence[str],
    batch_key: str,
) -> tuple[ad.AnnData, List[str]]:
    """Run requested integration methods and write embeddings / graphs into adata."""

    if "highly_variable" not in adata.var:
        raise RuntimeError(
            "Expected HVGs in adata.var['highly_variable'] (from load-and-filter). "
            "Refusing to integrate on all genes implicitly."
        )

    hvg_mask = adata.var["highly_variable"].to_numpy()
    if hvg_mask.sum() == 0:
        raise RuntimeError("adata.var['highly_variable'] is present but selects 0 genes.")

    LOGGER.info("Recomputing PCA on HVGs (mask_var='highly_variable')")

    adata.obsm.pop("X_pca", None)
    adata.uns.pop("pca", None)
    for k in ("PCs", "variance_ratio", "variance"):
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

    adata.obsm["Unintegrated"] = np.asarray(adata.obsm["X_pca"])
    created: List[str] = ["Unintegrated"]

    method_set = {m.lower() for m in (methods or [])}

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

    adata_hvg = adata[:, hvg_mask].copy()

    # ----------------------------
    # Harmony
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
    # Scanorama
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
    # scVI/scANVI
    # ----------------------------
    scvi_model = None
    if "scvi" in method_set or "scanvi" in method_set:
        try:
            LOGGER.info("Training scVI backbone on HVGs for integration")

            layer = _get_scvi_layer(adata_hvg)
            scvi_model = _train_scvi(
                adata_hvg,
                batch_key=batch_key,
                layer=layer,
                purpose="integration",
            )

            # Always compute latent once (needed for preflight labels)
            Z_scvi = np.asarray(scvi_model.get_latent_representation(adata_hvg))

            if "scvi" in method_set:
                _write_embedding_from_hvg("scVI", Z_scvi, adata_hvg.obs_names)
                created.append("scVI")

            if "scanvi" in method_set:
                # ----------------------------
                # Enhanced mode (flag-gated): preflight labels on scVI latent
                # ----------------------------
                use_preflight = str(
                    getattr(cfg, "scanvi_label_source", "leiden")
                ).lower() == "bisc_light"

                if use_preflight:
                    # --- BISC-light structural preflight for scANVI ---
                    labels_key_for_scanvi = _compute_scanvi_prelabels(
                        adata_hvg,
                        scvi_latent=Z_scvi,
                        batch_key=batch_key,
                        out_key=str(getattr(cfg, "scanvi_prelabels_key", "scanvi_prelabels")),
                        resolutions=getattr(cfg, "scanvi_preflight_resolutions", None),

                        # ---- new, explicit structural controls ----
                        max_prelabel_clusters=int(
                            getattr(cfg, "scanvi_max_prelabel_clusters", 25)
                        ),

                        # ---- selection / parsimony ----
                        min_stability_ok=float(
                            getattr(cfg, "scanvi_preflight_min_stability", 0.60)
                        ),
                        parsimony_eps=float(
                            getattr(cfg, "scanvi_preflight_parsimony_eps", 0.03)
                        ),

                        # ---- score weights (structural-only) ----
                        w_stab=float(getattr(cfg, "scanvi_w_stability", 0.50)),
                        w_sil=float(getattr(cfg, "scanvi_w_silhouette", 0.35)),
                        w_tiny=float(getattr(cfg, "scanvi_w_tiny", 0.15)),

                        # ---- guardrails ----
                        batch_trap_threshold=float(
                            getattr(cfg, "scanvi_batch_trap_threshold", 0.90)
                        ),
                        batch_trap_min_cells=int(
                            getattr(cfg, "scanvi_batch_trap_min_cells", 200)
                        ),
                        tiny_cluster_min_cells=int(
                            getattr(cfg, "scanvi_tiny_cluster_min_cells", 30)
                        ),
                    )
                else:
                    # Default: plain Leiden labels already present
                    labels_key_for_scanvi = str(getattr(cfg, "scanvi_labels_key", "leiden"))

                Zs = _run_scanvi_from_scvi(scvi_model, adata_hvg, labels_key_for_scanvi)
                _write_embedding_from_hvg("scANVI", np.asarray(Zs), adata_hvg.obs_names)
                created.append("scANVI")

        except Exception as e:
            LOGGER.warning("scVI/scANVI failed: %s", e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------------------
    # BBKNN
    # ----------------------------
    if "bbknn" in method_set:
        try:
            LOGGER.info("Running BBKNN (graph-only baseline) on HVG PCA")
            _run_bbknn(adata, batch_key=batch_key)
            created.append("BBKNN")
        except Exception as e:
            LOGGER.warning("BBKNN failed: %s", e)

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


# ---------------------------------------------------------------------
# scIB selection
# ---------------------------------------------------------------------

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
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    _ensure_label_key(adata, label_key)

    METHOD_CLASS = {
        "Unintegrated": "baseline",
        "BBKNN": "baseline",
        "Harmony": "unsupervised",
        "Scanorama": "unsupervised",
        "scVI": "unsupervised",
        "scANVI": "supervised",
        "scPoli": "supervised",
    }

    created = [str(e) for e in (embedding_keys or [])]
    created_set = set(created)

    ALLOWED_EXISTING = [
        "Unintegrated",
        "Harmony",
        "Scanorama",
        "scVI",
        "scANVI",
        "scPoli",
    ]

    def _is_valid_embedding_key(k: str) -> bool:
        return bool(k) and (not k.upper().startswith("BBKNN"))

    def _is_valid_embedding_matrix(k: str) -> bool:
        if k not in adata.obsm:
            return False
        try:
            Z = np.asarray(adata.obsm[k])
        except Exception:
            return False
        if Z.ndim != 2:
            return False
        if Z.shape[0] != int(adata.n_obs):
            return False
        if Z.shape[1] < 2:
            return False
        if not np.isfinite(Z).all():
            return False
        return True

    benchmark_embeddings: list[str] = []
    for e in created:
        if _is_valid_embedding_key(e) and _is_valid_embedding_matrix(e):
            benchmark_embeddings.append(e)

    pre_existing_added: list[str] = []
    for k in ALLOWED_EXISTING:
        if k in created_set:
            continue
        if _is_valid_embedding_key(k) and _is_valid_embedding_matrix(k):
            benchmark_embeddings.append(k)
            pre_existing_added.append(k)

    benchmark_embeddings = list(dict.fromkeys(benchmark_embeddings))

    if pre_existing_added:
        LOGGER.info(
            "scIB benchmarking: including pre-existing embeddings from adata.obsm (not recomputed this run): %s",
            pre_existing_added,
        )

    if not benchmark_embeddings:
        raise RuntimeError(
            "No valid embeddings available for scIB benchmarking. "
            "Expected at least one embedding in adata.obsm among created keys or known existing keys."
        )

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

        batch_series = adata.obs[batch_key].astype("category")
        batches = batch_series.cat.categories.tolist()
        batch_counts = batch_series.value_counts().reindex(batches).to_numpy()
        batch_props = batch_counts / batch_counts.sum()

        alloc = np.floor(batch_props * target).astype(int)
        alloc = np.minimum(alloc, batch_counts)

        if target >= len(batches):
            alloc = np.maximum(alloc, (batch_counts > 0).astype(int))
            alloc = np.minimum(alloc, batch_counts)

        current = int(alloc.sum())
        leftover = target - current

        if leftover > 0:
            capacity = batch_counts - alloc
            eligible = np.where(capacity > 0)[0]
            if eligible.size > 0:
                weights = capacity[eligible].astype(float)
                weights = weights / weights.sum() if weights.sum() > 0 else None
                picks = rng.choice(eligible, size=leftover, replace=True, p=weights)
                for i in picks:
                    if alloc[i] < batch_counts[i]:
                        alloc[i] += 1

        elif leftover < 0:
            remove = -leftover
            removable = np.where(alloc > 0)[0]
            if removable.size > 0:
                weights = alloc[removable].astype(float)
                weights = weights / weights.sum() if weights.sum() > 0 else None
                picks = rng.choice(removable, size=remove, replace=True, p=weights)
                for i in picks:
                    if alloc[i] > 0:
                        alloc[i] -= 1

        picked_obs = []
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

        if picked.size > target:
            picked = picked[:target]

        LOGGER.info(
            "scIB benchmarking: stratified subsample enabled (%d → %d cells; threshold=%d)",
            n_total,
            int(picked.size),
            int(benchmark_threshold),
        )

        adata_bm = adata[picked].copy()

    else:
        LOGGER.info(
            "scIB benchmarking: using full dataset (%d cells; threshold=%s)",
            n_total,
            str(benchmark_threshold),
        )
        adata_bm = adata.copy()

    LOGGER.info("Running scIB benchmarking on embeddings: %s", benchmark_embeddings)

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

    raw = bm.get_results(min_max_scale=False)
    scaled = bm.get_results(min_max_scale=True)

    raw.to_csv(Path(output_dir) / "integration_metrics_raw.tsv", sep="\t")
    scaled.to_csv(Path(output_dir) / "integration_metrics_scaled.tsv", sep="\t")

    plot_utils.plot_scib_results_table(scaled)

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

    bio_ok = numeric["Bio conservation"] > baseline["Bio conservation"]
    batch_ok = numeric["Batch correction"] > baseline["Batch correction"]

    candidates = numeric.index != "Unintegrated"

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


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def run_integrate(cfg: IntegrateConfig) -> ad.AnnData:
    init_logging(cfg.logfile)
    LOGGER.info("Starting integration module")

    plot_utils.setup_scanpy_figs(cfg.figdir, cfg.figure_formats)

    adata = io_utils.load_dataset(cfg.input_path)

    batch_key = cfg.batch_key or adata.uns.get("batch_key")
    if batch_key is None:
        raise RuntimeError("batch_key missing")
    LOGGER.info("Using batch_key='%s'", batch_key)

    methods = cfg.methods or ["scVI", "scANVI", "Harmony", "Scanorama", "BBKNN"]

    adata, emb_keys = _run_integrations(
        adata,
        cfg,
        methods=methods,
        batch_key=batch_key,
    )

    best = _select_best_embedding(
        adata,
        embedding_keys=emb_keys,
        batch_key=batch_key,
        label_key=cfg.label_key,
        n_jobs=cfg.benchmark_n_jobs,
        output_dir=cfg.output_dir,
        benchmark_threshold=cfg.benchmark_threshold,
        benchmark_n_cells=cfg.benchmark_n_cells,
        benchmark_random_state=cfg.benchmark_random_state,
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
            "BBKNN selected as best integration; using BBKNN neighbor graph (graph-based integration)."
        )

        import bbknn

        bbknn.bbknn(
            adata,
            batch_key=batch_key,
            use_rep="X_pca",
        )

        adata.obsm["X_integrated"] = adata.obsm["X_pca"]

    else:
        adata.obsm["X_integrated"] = adata.obsm[best]
        sc.pp.neighbors(adata, use_rep="X_integrated")

    adata.uns.setdefault("integration", {})
    adata.uns["integration"].update(
        {
            "best_embedding": best,
            "available_embeddings": list(emb_keys),
            "selection_timestamp": datetime.utcnow().isoformat(),
            "scanvi_label_source": str(getattr(cfg, "scanvi_label_source", "leiden")),
        }
    )

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
