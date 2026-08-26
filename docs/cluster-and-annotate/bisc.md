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

Raw adjacent ARI edges define stable plateaus. Contiguous strong edges at or above the stability threshold form a plateau core. A shorter core may recruit the strongest neighboring support edge until it reaches the minimum span; an already qualified core is not expanded. Rescued candidates that touch or overlap are merged because no transition remains between them. Candidates separated by an unused edge remain distinct. Plateaus are therefore disjoint, while support edges cannot cause unrestricted growth across a resolution sweep.

The two-sided smoothed stability curve remains part of the structural score and diagnostic output, but it no longer defines plateau membership or boundaries.

**Key defaults:**

* `stability_threshold`: **0.85**
* `min_plateau_len`: **3**
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

Selection proceeds in four ordered stages:

1. BISC identifies raw-edge structural plateaus.
2. It chooses one exact structural probe per plateau using plateau-local stability, silhouette separation, and the tiny-cluster term. The 3% parsimony tolerance is not used here.
3. Every candidate resolution is rebuilt on the same repeated cell subsamples. For each full-data plateau, BISC measures fixed-resolution probe reproducibility, the fraction of subsamples in which all internal edges retain support, and the persistence of its local boundary valleys. It does not redetect or select plateaus recursively within each subsample. The plateau persistence score is the weakest of the mean probe, internal-edge, and boundary persistence values. The plateau with the highest persistence score is selected; exact ties prefer the less complex probe.
4. Within the selected plateau, BISC ranks every feasible resolution using the complete structural and biological composite. It applies the 3% parsimony tolerance once, at this final stage, and chooses the near-best resolution with the fewest clusters.

Biological guidance therefore acts across the complete selected plateau rather than as a post-hoc override among structural probes. When no detected plateau contains a feasible candidate, BISC applies the stability-knee fallback across the feasible interior resolution set.

Five fixed safeguards define the validated selector behavior:

| Safeguard | Value | Role |
|---|---:|---|
| Minimum feasible adjacent-resolution stability | 0.60 | Restricts the candidate search set to partitions with sufficient local agreement when available. |
| Plateau support fraction | 0.50 | Defines the support-edge level halfway between the 0.60 feasibility floor and the configured strong-edge threshold; support edges may complete a short core but cannot broaden an already qualified core. |
| Parsimony tolerance | 3% | Selects the least cluster-heavy resolution whose full composite is within 3% of the best candidate in the chosen plateau. |
| Biological cluster-count limit | 2.5x | Limits the final within-plateau candidates to 2.5 times the number of confident biological reference labels when biological guidance is active; it does not remove structural plateaus from cross-plateau comparison. |
| Absolute minimum cluster size | 5 cells | Excludes pathological partitions from selection without changing raw-edge plateau geometry. |

These safeguards are fixed properties of the current BISC selector rather than command-line tuning parameters. Their values and the selector version are stored with every new BISC sweep. Use the resolution lens (`res_min`, `res_max`, and `n_resolutions`) as the primary control over the biological granularity under evaluation.

#### Subsampling reproducibility

BISC rebuilds the complete resolution sweep on repeated cell subsamples. One neighbour graph is reused for all resolutions within each repeat. The default evaluation uses five repeats retaining 80% of cells (`stability_repeats = 5`, `subsample_frac = 0.8`). The paired sweeps provide three distinct forms of evidence: agreement with each fixed-resolution full-data partition, retention of support across each full-data plateau, and persistence of the local valleys that delimit that plateau. These measurements test the full-data candidates without recursively rerunning the selector. The selected final resolution reuses its already computed fixed-resolution reproducibility values.

The native outputs distinguish these quantities. `plateau_persistence` compares partition, internal-edge, and boundary persistence across structural scales. `plateau_boundary_persistence` reports how often each full-data edge retains its support or separator state. `clustering_stability_ari` reports fixed-resolution reproducibility for the final partition. The selection curve continues to show the raw adjacent ARI edges and structural probe positions. All edge-level values, plateau persistence summaries, the selected and runner-up plateaus, persistence and complexity gaps, selector version, and final decision are retained in the clustering round metadata. Confidence is reported as `clear` for one eligible structural plateau, `multiscale` for more than one, `unstable` when the selected plateau's weakest persistence component is below the feasibility floor, and `weak` when BISC must use the no-plateau fallback.

Historical scOmnom archives may contain a `penalized_scores` diagnostic from an earlier selector design. The field remains loadable for reproducibility, but current BISC runs neither generate nor use it.

---
