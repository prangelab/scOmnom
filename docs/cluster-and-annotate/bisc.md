# BISC: Biology-Informed Structural Clustering

BISC automates Leiden resolution selection by combining structural separation, adjacent-resolution stability, and biological signals.

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
* `n_resolutions`: **25** (minimum **3**)

For each resolution, BISC records:

* number of clusters
* cluster size distribution
* centroid-based silhouette separation
* tiny-cluster burden
* agreement between adjacent resolutions

BISC does not subtract a global cluster-count penalty. Complexity is controlled by the tiny-cluster term, plateau constraints, biological fragmentation when available, and the parsimonious selection rule described below.

---

#### Adjacent-resolution stability (ARI)

BISC measures adjacent-resolution stability using the **Adjusted Rand Index (ARI)** between neighboring partitions in the full-data resolution sweep.

* high ARI → clustering structure changes little between neighboring resolutions
* low ARI → clusters split or merge aggressively

Adjacent-resolution stability is smoothed across neighboring resolutions and used to identify robust regions.

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

These metrics contribute to candidate ranking within the selected structurally stable region. When the number of confident biological labels is available, BISC also limits feasible candidates to at most 2.5 times that label count. Together, the weighted components and complexity guardrail favor biologically coherent partitions while controlling over-fragmentation.

---

#### Composite score and resolution selection

All metrics are normalized and combined into a single composite score:

* adjacent-resolution stability (`w_stab = 0.50`)
* silhouette separation (`w_sil = 0.35`)
* tiny cluster penalty (`w_tiny = 0.15`)
* optional biological metrics (weights above)

The final resolution is selected from stable structural plateaus that contain feasible candidates. Within the chosen plateau, BISC ranks candidates using the complete structural and biological composite, then selects the lowest resolution whose score is within 3% of the best score. This favors a simpler partition when the measured evidence is comparable. When no detected plateau contains a feasible candidate, BISC applies the stability-knee fallback across the feasible interior resolution set.

Four fixed safeguards define the validated selector behavior:

| Safeguard | Value | Role |
|---|---:|---|
| Minimum feasible adjacent-resolution stability | 0.60 | Restricts the candidate search set to partitions with sufficient local agreement when available. |
| Parsimony tolerance | 3% | Selects the lowest resolution whose composite score is within 3% of the best candidate in the chosen plateau. |
| Biological cluster-count limit | 2.5x | Limits feasible cluster counts to 2.5 times the number of confident biological reference labels when biological guidance is active. |
| Absolute minimum cluster size | 5 cells | Excludes partitions containing smaller clusters from structural plateau detection. |

These safeguards are fixed properties of the validated BISC selector rather than command-line tuning parameters. Their values are stored with every new BISC sweep. Use the resolution lens (`res_min`, `res_max`, and `n_resolutions`) as the primary control over the biological granularity under evaluation.

#### Post-selection subsampling reproducibility

After selecting a resolution, BISC rebuilds the chosen partition on repeated cell subsamples and compares each result with the full-data partition by ARI. The default evaluation uses five repeats retaining 80% of cells (`stability_repeats = 5`, `subsample_frac = 0.8`). This post-selection subsampling reproducibility is a diagnostic of the chosen result; it does not contribute to resolution selection. The existing option and archive names are retained for compatibility.

Historical scOmnom archives may contain a `penalized_scores` diagnostic from an earlier selector design. The field remains loadable for reproducibility, but current BISC runs neither generate nor use it.

---
