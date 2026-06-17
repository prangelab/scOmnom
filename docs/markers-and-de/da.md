# DA (differential abundance / composition)

The DA submodule tests whether **cluster proportions change across conditions**. It works on per-sample cell-type counts and provides multiple backends, each with different assumptions:

**Condition key syntax (DA)**

DA supports the same `A` and `A:B` composite syntax as DE, plus `A@B` to run `A` within each `B` level. Interaction syntax (`A^B`) is **DE-only**. See the DE section above for examples.

You can pass one or more keys (same behavior as DE), and results are written into per-key subfolders under DA tables/figures.

**Reference selection**

* Default reference is `most_stable`, chosen as the cluster with the lowest median absolute deviation of proportions among clusters with mean proportion ≥ `min_mean_prop`.
* If none meet the threshold, the most abundant cluster is used.

**Methods**

* `sccoda`: Bayesian compositional model (pertpy scCODA). Uses NUTS sampling and inclusion probabilities with FDR control. Works directly from cell-level data but models sample-level composition.
* `glm`: Per-cluster binomial GLM on counts with a log-total offset (statsmodels). Supports covariates and returns log2-scale effects with Wald tests and BH FDR.
* `clr`: Centered log-ratio transform of proportions followed by Mann–Whitney tests for each pair of condition levels. Returns log2 fold-change of mean proportions and per-pair FDR.
* `graph`: Graph-based DA on neighborhoods in the integrated embedding, inspired by MiloR. Neighborhoods are sampled from stratified seeds, tested with **NB-GLM** on neighborhood counts (with sample-size offsets), then corrected by **spatial weighted BH FDR**. Outputs neighborhood metadata and diagnostics.

**When to use which DA method**

| Method | Best for | Key assumptions | Notes |
| --- | --- | --- | --- |
| `sccoda` | Small-to-moderate sample sizes with clear compositional shifts | Compositional Bayesian model; needs a reference | Most conservative; provides inclusion probabilities |
| `glm` | Larger sample sizes, covariate adjustment | Binomial GLM with log-total offset | Fast, interpretable log2 effects |
| `clr` | Simple two-group comparisons, quick screening | CLR transform + nonparametric test | Pairwise only; less model structure |
| `graph` | Substructure or neighborhood-level shifts | Embedding neighborhoods + NB-GLM + spatial FDR | Milo-style local DA; conservative under many tests |

**Automatic method behavior**

* GLM is automatically skipped for 2-level conditions (CLR is used instead).
* Conditions with too few replicates per level are skipped with an explicit warning.
* GraphDA applies support filters before testing (`min_nonzero_samples_per_level`) and effect shrinkage (`effect_shrink_k`) to reduce unstable calls.

**GraphDA significance and QC**

GraphDA reports both raw and adjusted evidence:

* `pval`: neighborhood-level NB-GLM p-value
* `fdr_bh`: standard BH correction
* `fdr_spatial`: weighted BH correction using neighborhood spatial weights
* `fdr`: current primary significance column (set to `fdr_spatial`)

Why this matters:

* Many neighborhoods overlap and are not independent tests.
* Spatial FDR is the primary call metric for Milo-style neighborhood DA.
* It is possible to see directional effects with no FDR-significant neighborhoods if multiplicity is high and raw p-values are moderate.

**Outputs**

Per condition key (and per `A@B=<level>` expansion), DA writes:

* Tables: `tables/DA_tables_<run>/<condition_tag>/`
* Figures: `figures/DA/<condition_tag>/`

Key DA tables include:

* `composition_global_sccoda.tsv`
* `composition_global_glm.tsv` (when eligible)
* `composition_global_clr.tsv`
* `composition_global_graph.tsv`
* `composition_consensus.tsv`
* `composition_graph_neighborhoods.tsv`
* `graphda_diagnostics.tsv`
* `composition_settings.txt`

Key DA figures include:

* composition summaries (`composition_stacked_bar_100`, `composition_stacked_comparison`, `composition_flow`)
* global effects (`composition_effects_global_sccoda`, `composition_effects_global_clr`)
* GraphDA summaries (`graphda_effects_by_cluster`, `graphda_top_neighborhoods`, `graphda_top_by_cluster`)
* GraphDA QC (`graphda_qc_pval_vs_fdr`, `graphda_qc_cluster_power`)

---
