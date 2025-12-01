#!/usr/bin/env python3
"""
Demo script: Synthetic dataset for illustrating the auto-cluster algorithm.

This version produces:
- Multi-level clustering structure
- Misaligned biological labels
- Cluster size imbalance
- Biological fragmentation
- Noise cells
- Visible plateaus & composite-score differences

Output: all standard cluster-and-annotate figures.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from scomnom.cluster_and_annotate import run_clustering
from scomnom.config import ClusterAnnotateConfig


# -----------------------------------------------------------------------------
# Synthetic data generator
# -----------------------------------------------------------------------------

def make_synthetic_explainer_dataset(
    random_state: int = 42,
    n_clusters: int = 5,
    n_dim: int = 15,
) -> ad.AnnData:
    """
    Generate a structured synthetic dataset with built-in biological mismatches.

    The design intentionally introduces:
    - Multi-scale Gaussian clusters
    - Unequal cluster sizes
    - Subcluster structure (fine structure)
    - Biological label contamination (heterogeneity)
    - Noise cells (garbage cluster)
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # 1. Define uneven cluster sizes
    # ------------------------------------------------------------
    cluster_sizes = np.array([300, 600, 1200, 450, 250])  # Very unequal
    assert len(cluster_sizes) == n_clusters

    # ------------------------------------------------------------
    # 2. Generate coarse cluster centroids far apart
    # ------------------------------------------------------------
    coarse_centroids = rng.normal(0, 15, size=(n_clusters, n_dim))

    X_list = []
    labels_struct = []

    for cid, size in enumerate(cluster_sizes):
        # Main Gaussian cloud
        base = coarse_centroids[cid]

        # Add subtle fine structure inside clusters
        X_main = base + rng.normal(0, 3.0, size=(size, n_dim))

        # Add multimodal substructure (resolve at high resolution)
        bump = rng.normal(0, 0.7, size=(size, n_dim))
        X_sub = X_main + bump

        X_list.append(X_sub)
        labels_struct.append(np.full(size, cid))

    # Stack all structured cells
    X = np.vstack(X_list)
    labels_struct = np.concatenate(labels_struct)

    # ------------------------------------------------------------
    # 3. Add noise cells (5%)
    # ------------------------------------------------------------
    n_noise = int(0.05 * X.shape[0])
    X_noise = rng.normal(0, 25, size=(n_noise, n_dim))  # big variance
    labels_noise = np.full(n_noise, -1)

    X = np.vstack([X, X_noise])
    labels_struct = np.concatenate([labels_struct, labels_noise])

    n_total = X.shape[0]

    # ------------------------------------------------------------
    # 4. Biological labels with structured mismatches
    # ------------------------------------------------------------
    bio_labels = labels_struct.copy()

    # Biological label space collapsed (coarser biology)
    # e.g. 5 structural clusters → 3 biological groups
    mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, -1: 2}
    bio_labels = np.array([mapping[x] for x in bio_labels])

    # Inject controlled contamination (heterogeneity)
    for cid in range(n_clusters):
        idx = np.where(labels_struct == cid)[0]
        contam_n = max(5, int(0.12 * len(idx)))  # 12%
        contam_idx = rng.choice(idx, size=contam_n, replace=False)
        allowed = np.setdiff1d(np.unique(bio_labels), [mapping[cid]])
        bio_labels[contam_idx] = rng.choice(allowed, size=contam_n, replace=True)

    # Noise cells get random biology
    noise_idx = np.where(labels_struct == -1)[0]
    if len(noise_idx) > 0:
        bio_labels[noise_idx] = rng.choice(np.unique(bio_labels), size=len(noise_idx))

    # ------------------------------------------------------------
    # 5. Build AnnData
    # ------------------------------------------------------------
    adata = ad.AnnData(X)
    adata.obs["bio"] = pd.Categorical(bio_labels)
    adata.obs["struct_true"] = pd.Categorical(labels_struct)

    # Set up neighbors / UMAP space for the structural part
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X")
    sc.tl.umap(adata)

    return adata


# -----------------------------------------------------------------------------
# Main demo procedure
# -----------------------------------------------------------------------------

def main():
    print("Generating synthetic explainer dataset...")
    adata = make_synthetic_explainer_dataset()

    outdir = Path("synthetic_demo_results")
    outdir.mkdir(exist_ok=True)

    cfg = ClusterAnnotateConfig(
        output_path=outdir / "synthetic.clustered.annotated.h5ad",
        embedding_key="X_umap",
        label_key="leiden",

        bio_guided_clustering=True,
        w_hom=0.15,
        w_frag=0.10,
        w_bioari=0.15,

        res_min=0.2,
        res_max=1.8,
        n_resolutions=12,
        penalty_alpha=0.02,

        stability_repeats=4,
        subsample_frac=0.8,
        random_state=42,

        tiny_cluster_size=25,
        min_cluster_size=20,
        min_plateau_len=3,
        max_cluster_jump_frac=0.4,
        stability_threshold=0.85,

        make_figures=True,
        figure_formats=["png"],
        figdir_name="figs_synth_demo",

    )

    run_clustering(cfg)

    print("\nSynthetic demo complete.")
    print(f"→ Results written to {outdir}")


if __name__ == "__main__":
    main()
