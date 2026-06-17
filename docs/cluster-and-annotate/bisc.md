# BISC: Biology-Informed Structural Clustering

BISC automates Leiden resolution selection by combining structural, stability, and biological signals.

The goal is to replace manual resolution tuning with a reproducible, data-driven procedure.

---

#### Inputs

* `--embedding-key` (default `X_integrated`) is used for neighbors, UMAP, and clustering.
* `--batch-key` is used for batch diagnostics and plotting (auto-detected if not provided).
* CellTypist labels are optional but enable bio-guided scoring (`--bio-guided`).

#### Outputs

* A new clustering round in `adata.uns["cluster_rounds"]` with the chosen resolution.
* The selected round id is recorded in `adata.uns["active_cluster_round"]`.
* Cluster labels are stored in `adata.obs` under the round’s `cluster_key`.

#### When to tune

* Adjust `res_min`, `res_max`, and `n_resolutions` if the dataset is extremely small or highly heterogeneous.
* Disable bio-guided clustering (`--no-bio-guided`) if no suitable CellTypist model exists.

---

#### Resolution sweep

Leiden clustering is evaluated over a resolution range:

* `res_min`: **0.1**
* `res_max`: **2.5**
* `n_resolutions`: **25**

For each resolution, BISC records:

* number of clusters
* cluster size distribution
* centroid-based silhouette separation
* penalties for excessive cluster fragmentation

A small penalty proportional to cluster count is applied:

* `penalty_alpha`: **0.02**

---

#### Structural stability (ARI)

BISC measures clustering stability using the **Adjusted Rand Index (ARI)** between adjacent resolutions.

* high ARI → clustering structure is stable
* low ARI → clusters split or merge aggressively

Stability is smoothed across neighboring resolutions and used to identify robust regions.

**Key defaults:**

* `stability_threshold`: **0.85**
* `min_plateau_len`: **3**
* `max_cluster_jump_frac`: **0.4**
* `min_cluster_size`: **20**
* `tiny_cluster_size`: **20**

---

#### Biological metrics (CellTypist-guided)

If CellTypist predictions are available, BISC incorporates biological consistency metrics.

This is enabled by default (`bio_guided_clustering = True`).

Only high-confidence cells are considered, using a probability-based mask:

* entropy limit: **≤ 0.5**
* entropy quantile: **0.7**
* minimum margin (top1 − top2): **0.10**

Safety gates:

* **≥ 500 cells**
* **≥ 5% of the dataset**

The biological metrics used are:

#### 1. Biological homogeneity

Mean fraction of the dominant CellTypist label within clusters.

* weight: `w_hom = 0.15`

#### 2. Biological fragmentation

Penalizes clusters that contain multiple large biological subgroups.

* weight: `w_frag = 0.10`

#### 3. Biological ARI

ARI between cluster labels and CellTypist labels (on confident cells).

* weight: `w_bioari = 0.15`

These metrics encourage clusterings that align with known biology while avoiding over-fragmentation.

---

#### Composite score and resolution selection

All metrics are normalized and combined into a single composite score:

* structural stability (`w_stab = 0.50`)
* silhouette separation (`w_sil = 0.35`)
* tiny cluster penalty (`w_tiny = 0.15`)
* optional biological metrics (weights above)

The final resolution is selected from stable plateaus, with a bias toward simpler solutions when scores are comparable.

---
