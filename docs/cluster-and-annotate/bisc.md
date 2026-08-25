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

Raw adjacent ARI edges define stable plateaus. A run of strong edges at or above the stability threshold forms a plateau core. When a core is shorter than the minimum plateau span, connected support edges may extend it only until that minimum is reached. Once a core is long enough, weaker neighboring edges do not broaden it. This prevents an unstable transition leaving a plateau from erasing the stable region before it.

The two-sided smoothed stability curve remains part of the structural score and diagnostic output, but it no longer defines plateau membership or boundaries.

**Key defaults:**

* `stability_threshold`: **0.85**
* `min_plateau_len`: **3**
* `min_cluster_size`: **20**
* `tiny_cluster_size`: **20**

`max_cluster_jump_frac` remains accepted in configuration files for compatibility with earlier releases but no longer deletes individual plateau members. Raw partition agreement now carries the structural-boundary decision.

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
3. Each probe is rebuilt over repeated cell subsamples. The probe with the highest mean fixed-resolution ARI selects the plateau; exact ties prefer the less complex probe.
4. Within the selected plateau, BISC ranks every feasible resolution using the complete structural and biological composite. It applies the 3% parsimony tolerance once, at this final stage, and chooses the near-best resolution with the fewest clusters.

Biological guidance therefore acts across the complete selected plateau rather than as a post-hoc override among structural probes. When no detected plateau contains a feasible candidate, BISC applies the stability-knee fallback across the feasible interior resolution set.

Four fixed safeguards define the validated selector behavior:

| Safeguard | Value | Role |
|---|---:|---|
| Minimum feasible adjacent-resolution stability | 0.60 | Restricts the candidate search set to partitions with sufficient local agreement when available. |
| Plateau support fraction | 0.50 | Defines the support-edge level halfway between the 0.60 feasibility floor and the configured strong-edge threshold; support edges only complete short cores. |
| Parsimony tolerance | 3% | Selects the least cluster-heavy resolution whose full composite is within 3% of the best candidate in the chosen plateau. |
| Biological cluster-count limit | 2.5x | Limits the final within-plateau candidates to 2.5 times the number of confident biological reference labels when biological guidance is active; it does not remove structural plateaus from cross-plateau comparison. |
| Absolute minimum cluster size | 5 cells | Excludes pathological partitions from selection without changing raw-edge plateau geometry. |

These safeguards are fixed properties of the validated BISC selector rather than command-line tuning parameters. Their values are stored with every new BISC sweep. Use the resolution lens (`res_min`, `res_max`, and `n_resolutions`) as the primary control over the biological granularity under evaluation.

#### Subsampling reproducibility

BISC rebuilds each structural plateau probe on repeated cell subsamples and compares it with the same fixed-resolution full-data partition by ARI. These values select among plateaus. The default evaluation uses five repeats retaining 80% of cells (`stability_repeats = 5`, `subsample_frac = 0.8`). After the final within-plateau call, BISC also records subsampling reproducibility for the selected partition; when the final resolution is itself a probe, the already computed values are reused.

The native outputs distinguish the two roles. `plateau_probe_reproducibility` compares structural scales, while `clustering_stability_ari` reports the final partition diagnostic. The selection curve shows the raw adjacent ARI edges that define plateau geometry and places probe markers on the structural score rather than the full biological composite. Plateau boundaries, probes, reproducibility summaries, the selected and runner-up plateaus, reproducibility and complexity gaps, selector version, and final decision are retained in the clustering round metadata. Confidence is reported conservatively as `clear` for one eligible structural plateau, `multiscale` for more than one, and `weak` when BISC must use the no-plateau fallback.

Historical scOmnom archives may contain a `penalized_scores` diagnostic from an earlier selector design. The field remains loadable for reproducibility, but current BISC runs neither generate nor use it.

---
