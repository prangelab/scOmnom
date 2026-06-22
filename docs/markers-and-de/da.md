# DA (differential abundance / composition)

The DA submodule tests whether cluster or neighborhood abundance changes across conditions. It starts from the active clustering round, counts cells per sample and population, and runs one or more composition backends on those sample-level counts.

By default, scOmnom runs all DA methods:

```bash
scomnom markers-and-de da \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --condition-keys treatment
```

Use `--method` to run a subset. The option is repeatable and also accepts comma-separated values:

```bash
scomnom markers-and-de da ... --condition-keys treatment --method graph
scomnom markers-and-de da ... --condition-keys treatment --method sccoda --method clr
scomnom markers-and-de da ... --condition-keys treatment --method graph,clr
```

## Method Choice

| Method | Runs by default | Use for | Main evidence |
| --- | --- | --- | --- |
| `sccoda` | yes | Global compositional shifts across annotated populations | Bayesian inclusion probabilities and FDR-controlled effects |
| `glm` | yes, but skipped for 2-level conditions | Multi-level conditions and covariate-adjusted global effects | Per-cluster binomial GLM coefficients and BH FDR |
| `clr` | yes | Simple pairwise screening across condition levels | CLR-transformed proportions, Mann-Whitney tests, pairwise FDR |
| `graph` | yes | Local abundance shifts in the integrated embedding | Neighborhood NB-GLM, spatial weighted BH FDR, GraphDA diagnostics |

For most routine runs, leave the default method set on. Use a subset when you want a faster exploratory pass (`--method clr,graph`), a GraphDA-only neighborhood analysis (`--method graph`), or a compact global-composition run without neighborhood testing (`--method sccoda,clr,glm`).

## Conditions

DA supports the same condition-key syntax as within-cluster DE, except interaction syntax (`A^B`) is DE-only.

| Syntax | Meaning | Example |
| --- | --- | --- |
| `A` | Test one `adata.obs` column. | `--condition-keys treatment` |
| `A:B` | Build a composite condition from multiple columns. | `--condition-keys treatment:genotype` creates levels such as `treated.KO` and `vehicle.WT`. |
| `A@B` | Run `A` separately within each level of `B`. | `--condition-keys treatment@genotype` runs treatment DA inside each genotype level. |

You can repeat `--condition-keys` to run multiple condition definitions in one command. Each condition key, including each expanded `A@B=<level>` run, gets its own tables and figure folder.

```bash
scomnom markers-and-de da \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --condition-keys treatment \
  --condition-keys treatment:genotype \
  --condition-keys treatment@genotype
```

## Shared Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. If omitted, scOmnom uses the standard results-location logic. |
| `--output-name` | inferred from input, module, and round | Saved AnnData name when dataset saving is enabled. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. The normal saved object remains the scOmnom dataset format. |
| `--n-jobs` | `1` | Parallel workers across condition-key tasks; BLAS threads are capped internally for DA. |
| `--round-id` | active clustering round | Selects which clustering round supplies the population labels. |
| `--replicate-key` | `adata.uns["batch_key"]`, then `sample_id` | Sample/replicate column used for per-sample counts. Must exist in `adata.obs`. |
| `--condition-keys` | required | One or more condition definitions. |
| `--covariates` | none | Repeatable sample-level covariates for scCODA, GLM, and GraphDA design formulas. |
| `--method` | `sccoda`, `glm`, `clr`, `graph` | Repeat or comma-separate to select methods. |
| `--alpha` | `0.05` | FDR/significance threshold used in consensus summaries and plots. |
| `--min-cells-per-sample-cluster` | `20` | Stored DA setting for minimum cell support per sample-cluster combination. |

DA skips a condition task when any condition level has fewer than 2 samples. This guard is applied before method-specific testing.

## Reference Selection

scCODA requires a reference population. scOmnom chooses one automatically unless you provide it.

| Option | Default | Notes |
| --- | --- | --- |
| `--reference` | `most_stable` | Use the automatically selected stable reference population, or pass a specific cluster label. |
| `--min-mean-prop` | `0.01` | Minimum mean proportion a cluster must reach to be considered for `most_stable`. |

`most_stable` chooses the cluster with the lowest median absolute deviation of per-sample proportions among clusters with mean proportion at least `--min-mean-prop`. If none pass that threshold, the most abundant cluster is used.

## scCODA

`sccoda` is the Bayesian global composition backend. It uses pertpy scCODA on cell-level input while modeling sample-level composition internally.

| Setting | Default | Notes |
| --- | --- | --- |
| Method selector | included in default `--method` | Disable by selecting methods that omit `sccoda`. |
| Reference | `--reference most_stable` | Required by scCODA; auto-selected unless overridden. |
| FDR | `--alpha 0.05` | Passed to scCODA inclusion-probability FDR control. |
| Covariates | `--covariates` none | Added to the scCODA formula after the condition key. |
| NUTS samples | `10000` | Internal default; not exposed as a CLI option in the current DA command. |
| Warmup samples | `max(1000, samples / 10)` | Internal default; not exposed as a CLI option in the current DA command. |
| Random seed | `42` | Internal scCODA RNG key. |

The output table is `composition_global_sccoda.tsv`.

## GLM

`glm` fits a per-cluster binomial GLM on sample-level counts with a log-total offset. It is useful when there are more than two condition levels or when covariates are important.

| Setting | Default | Notes |
| --- | --- | --- |
| Method selector | included in default `--method` | Disable by selecting methods that omit `glm`. |
| Condition levels | skipped for 2-level conditions | For 2-level conditions, use CLR for the simple pairwise comparison. |
| Minimum samples per level | `2` | GLM is skipped when any condition level has fewer than 2 samples. |
| Covariates | `--covariates` none | Included in the GLM design matrix. |
| Multiple testing | BH FDR | Applied across GLM rows. |

The output table is `composition_global_glm.tsv` when the method is eligible and returns results.

## CLR

`clr` runs a centered log-ratio transform of per-sample cluster proportions, then tests every pair of condition levels with Mann-Whitney tests.

| Setting | Default | Notes |
| --- | --- | --- |
| Method selector | included in default `--method` | Disable by selecting methods that omit `clr`. |
| Pseudocount | `1e-6` | Internal value used before log transform and log2 fold-change calculation. |
| Contrasts | all pairwise condition-level combinations | Generated automatically from the condition levels. |
| Multiple testing | BH FDR per pairwise block | Reported in `fdr`. |

The output table is `composition_global_clr.tsv`.

## GraphDA

`graph` tests local abundance shifts in the integrated embedding. It samples neighborhoods from stratified seed cells, counts each neighborhood per sample, tests those counts with a negative-binomial GLM and sample-size offset, then applies spatial weighted BH FDR.

| Option | Default | Notes |
| --- | --- | --- |
| `--graph-n-seeds` | `2000` | Number of seed neighborhoods to sample, capped by cell count. |
| `--graph-k-ref` | `30` | Reference neighbor rank used for neighborhood radius and initial neighborhood size. |
| `--graph-max-k` | `200` | Maximum neighbors available to the nearest-neighbor search; internally at least `k_ref + 1`. |
| `--graph-min-size` | `20` | Minimum neighborhood size and minimum cluster size for stratified seed coverage. |
| `--graph-random-state` | `42` | Seed for neighborhood sampling. |
| `--graph-min-nonzero-samples-per-level` | `3` | Minimum nonzero sample support per condition level for a neighborhood to be tested. |
| `--graph-effect-shrink-k` | `10.0` | Shrinks neighborhood effects toward zero when total nonzero sample support is low. |
| `--graph-n-permutations` | `0` | Deprecated and ignored; GraphDA now uses NB-GLM plus spatial weighted BH FDR. |

GraphDA uses `adata.obsm["X_integrated"]` when present. If it is missing, it falls back to `adata.uns["integration"]["best_embedding"]` when that embedding exists.

### GraphDA Significance And QC

GraphDA reports both raw and adjusted evidence:

| Column | Meaning |
| --- | --- |
| `pval` | Neighborhood-level NB-GLM p-value. |
| `fdr_bh` | Standard BH correction. |
| `fdr_spatial` | Weighted BH correction using neighborhood spatial weights. |
| `fdr` | Primary significance column, currently set to `fdr_spatial`. |
| `effect_raw` | Unshrunk neighborhood log2-scale effect when available. |
| `effect_shrunk` | Support-shrunk effect. |
| `effect` | Primary effect column, currently set to the shrunk effect. |

Many GraphDA neighborhoods overlap, so `fdr_spatial` is the primary call metric. It is normal to see directional raw effects without FDR-significant neighborhoods when the test burden is high or raw p-values are moderate.

The main output tables are `composition_global_graph.tsv`, `composition_graph_neighborhoods.tsv`, and `graphda_diagnostics.tsv`.

## Consensus

After all requested methods finish, scOmnom writes `composition_consensus.tsv`. This table summarizes method agreement per cluster and contrast, including the number of methods run, number significant, mean effect direction, sign agreement, and a `high_confidence_da` flag.

The current high-confidence flag is GraphDA-centered: it requires significant GraphDA evidence plus directional agreement from CLR or scCODA when those methods are present. Use it as a compact triage column, then inspect the method-specific tables for the actual evidence.

## Plotting And Output Controls

| Option | Default | Notes |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create DA figures after computation. |
| `--regenerate-figures` | off | Rebuild figures from stored DA results without recomputation. Requires figures to be enabled. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |

Per condition key, including expanded `A@B=<level>` keys, DA writes:

* tables: `tables/DA_tables_<round>_roundN/<condition_tag>/`;
* figures: `figures/DA/<condition_tag>/`;
* settings: `composition_settings.txt`.

Key DA tables include:

* `composition_global_sccoda.tsv`;
* `composition_global_glm.tsv` when eligible;
* `composition_global_clr.tsv`;
* `composition_global_graph.tsv`;
* `composition_consensus.tsv`;
* `composition_graph_neighborhoods.tsv`;
* `graphda_diagnostics.tsv`;
* `composition_settings.txt`.

Key DA figures include:

* composition summaries: `composition_stacked_bar_100`, `composition_stacked_comparison`, `composition_flow`;
* global effects: `composition_effects_global_sccoda`, `composition_effects_global_clr`, GLM/CLR volcanoes;
* GraphDA summaries: `graphda_effects_by_cluster`, `graphda_top_neighborhoods`, `graphda_top_by_cluster`;
* GraphDA QC: `graphda_qc_pval_vs_fdr`, `graphda_qc_cluster_power`.

---
