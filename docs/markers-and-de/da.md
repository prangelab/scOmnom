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
scomnom markers-and-de da ... --condition-keys treatment --method milo
scomnom markers-and-de da ... --condition-keys treatment --method sccoda --method clr
scomnom markers-and-de da ... --condition-keys treatment --method milo,clr
```

## Method Choice

| Method | Runs by default | Use for | Main evidence |
| --- | --- | --- | --- |
| `sccoda` | yes | Global compositional shifts across annotated populations | Bayesian inclusion probabilities and FDR-controlled effects |
| `glm` | yes | Global per-cluster composition effects with optional covariate adjustment | Per-cluster binomial GLM coefficients and BH FDR |
| `clr` | yes | Simple pairwise screening across condition levels | CLR-transformed proportions, Mann-Whitney tests, pairwise FDR |
| `milo` | yes | Local abundance shifts in the integrated embedding | Refined neighbourhood count models and spatial FDR |

For most routine runs, leave the default method set on. Use a subset when you want a faster exploratory pass (`--method clr,milo`), a Milo-only neighbourhood analysis (`--method milo`), or a compact global-composition run without neighbourhood testing (`--method sccoda,clr,glm`). The legacy selector `--method graph` is accepted as a deprecated alias and is normalized to `milo` before execution.

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
| `--covariates` | none | Repeatable sample-level covariates for scCODA, GLM, and Milo design formulas. |
| `--method` | `sccoda`, `glm`, `clr`, `milo` | Repeat or comma-separate to select methods. |
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

`glm` fits a per-cluster binomial GLM on sample-level composition. For each cluster, the response is modeled as successes versus failures (`cluster_count` and `sample_total - cluster_count`), so coefficients describe changes in cluster abundance relative to the rest of the sample. It works for two-level and multi-level condition designs, and is especially useful when covariates are important.

| Setting | Default | Notes |
| --- | --- | --- |
| Method selector | included in default `--method` | Disable by selecting methods that omit `glm`. |
| Condition levels | at least `2` | GLM is skipped only when fewer than 2 condition levels remain after dropping missing values. |
| Minimum samples per level | `2` | GLM is skipped when any condition level has fewer than 2 samples. |
| Covariates | `--covariates` none | Included in the GLM design matrix. |
| Multiple testing | BH FDR | Applied across GLM rows. |

The output table is `composition_global_glm.tsv` when the method is eligible and returns results. It includes `fit_warning` and `n_fit_warnings` columns so model-fit warnings are visible in downstream review.

## CLR

`clr` runs a centered log-ratio transform of per-sample cluster proportions, then tests every pair of condition levels with Mann-Whitney tests.

| Setting | Default | Notes |
| --- | --- | --- |
| Method selector | included in default `--method` | Disable by selecting methods that omit `clr`. |
| Pseudocount | `1e-6` | Internal value used before log transform and log2 fold-change calculation. |
| Contrasts | all pairwise condition-level combinations | Generated automatically from the condition levels. |
| Multiple testing | BH FDR per pairwise block | Reported in `fdr`. |

The output table is `composition_global_clr.tsv`.

## Milo

`milo` tests local abundance shifts in the integrated embedding through pertpy's maintained implementation of Milo. It builds a Scanpy k-nearest-neighbour graph, samples graph vertices, refines them to representative index cells, collapses duplicate representatives, counts each retained neighbourhood per sample, and fits a negative-binomial count model. Milo's distance-weighted spatial FDR is the primary multiple-testing correction.

| Option | Default | Notes |
| --- | --- | --- |
| `--milo-scale` | `custom` | Named neighbourhood-scale preset. The default custom values equal balanced M05, while preserving direct numeric overrides; `local`, `balanced`, and `broad` replace all three scale parameters. |
| `--milo-n-seeds` | `1000` | Target number of initially sampled graph vertices, capped by cell count. Representative refinement and duplicate collapse may reduce the final count. |
| `--milo-k-ref` | `75` | Number of neighbours used to construct the Scanpy graph. It is bounded to `n_cells - 1` on small data. |
| `--milo-min-size` | `50` | Minimum membership of a refined neighbourhood. Smaller neighbourhoods are recorded but not tested. |
| `--milo-random-state` | `42` | Seed for Milo's initial graph-vertex sampling. |
| `--milo-min-nonzero-samples-per-level` | `3` | Minimum nonzero sample support in both levels of each pairwise contrast. The filter and correction family are contrast-specific. |
| `--milo-solver` | `pydeseq2` | Negative-binomial solver used through pertpy. `edger` is the closest to original Milo but requires R, rpy2, edgeR, limma, and statmod. |
| `--milo-group-regions` / `--no-milo-group-regions` | enabled | Group significant overlapping neighbourhoods into direction-concordant DA regions. The raw neighbourhood results are always retained. |
| `--milo-group-min-overlap` | `1` | Minimum number of shared cells required to connect two significant neighbourhoods. |
| `--milo-group-max-lfc-delta` | none | Optional maximum absolute difference between neighbourhood log2 fold changes allowed along a region edge. |
| `--milo-extreme-log2fc` | `3.0` | Absolute neighbourhood log2 fold change flagged for review. This does not clip, shrink, or remove the estimate. |
| `--milo-broad-coverage-fraction` | `0.5` | Unique-cell coverage fraction at which a contrast is flagged as broad and requires explicit compositional review. |
| `--graph-max-k` | `200` | Deprecated and ignored; retained temporarily for command compatibility. |
| `--graph-effect-shrink-k` | `10.0` | Deprecated and ignored; Milo log-fold changes are reported directly. |
| `--graph-n-permutations` | `0` | Deprecated and ignored; Milo uses count-model inference and spatial FDR. |

The former active `--graph-*` names remain deprecated aliases for their `--milo-*` counterparts. Milo scale presets are:

| `--milo-scale` | Initial vertices | Graph neighbours | Minimum size | Use when |
| --- | --- | --- | --- | --- |
| `custom` | `1000` | `75` | `50` | Routine M05 defaults, with direct control through the three numeric options. |
| `local` | `2000` | `30` | `20` | Fine local neighbourhoods and higher spatial detail. |
| `balanced` | `1000` | `75` | `50` | Middle ground for routine DA runs. |
| `broad` | `300` | `150` | `100` | Broader neighbourhoods and fewer tests. |

Milo uses `adata.obsm["X_integrated"]` when present. If it is missing, it falls back to `adata.uns["integration"]["best_embedding"]` when that embedding exists.

### Milo Significance And QC

Milo reports both raw and adjusted evidence:

| Column | Meaning |
| --- | --- |
| `pval` | Neighbourhood-level count-model p-value. |
| `fdr_bh` | Ordinary count-model FDR for that pairwise contrast. |
| `fdr_spatial` | Milo spatial FDR using inverse kth-neighbour distance weights. |
| `fdr` | Primary significance column, currently set to `fdr_spatial`. |
| `effect_raw` | Milo log2 fold change in neighbourhood abundance for test versus reference. |
| `effect` | Primary effect column, identical to `effect_raw`. |
| `effect_requires_review` | Flags extreme effects or effects that only meet the minimum nonzero-sample support. |
| `effect_review_reason` | Machine-readable reason for review; `ok` when no flag is raised. |
| `region_id` | Grouped DA region containing this significant neighbourhood, or missing for non-significant/ungrouped rows. |

Many Milo neighbourhoods overlap, so `fdr_spatial` is the primary call metric. Every output row records the solver, pairwise levels, nonzero-sample support, and number of neighbourhoods in its correction family. Solver failures stop the Milo run with the failed contrast identified; scOmnom does not silently substitute GLM or CLR results.

### Default Region Grouping

By default, scOmnom treats significant neighbourhoods as local measurements that must be consolidated before biological interpretation. Within each contrast, significant neighbourhoods are connected when they share at least `--milo-group-min-overlap` cells and have the same effect direction. Connected neighbourhoods form a DA region. Setting `--milo-group-max-lfc-delta` additionally prevents an edge when the two raw neighbourhood effects differ by more than the requested value.

Region grouping does not change Milo p-values, spatial FDR values, or raw log2 fold changes. It changes the reporting unit. Each region reports its neighbourhood count, unique-cell coverage, median and interquartile neighbourhood effect, minimum spatial FDR, dominant parent cluster, and parent-cluster purity. Disable grouping with `--no-milo-group-regions` only when raw neighbourhood-level output is explicitly required.

`composition_milo_region_sample_counts.tsv` includes every sample in the contrast, including zero counts, with both region cell counts and within-sample fractions. This table should be inspected before interpreting a large regional effect. `milo_coverage.tsv` reports the number of significant neighbourhoods, grouped regions, unique cells covered, and fraction of the contrast's cells covered. Coverage at or above `--milo-broad-coverage-fraction` is flagged for explicit review. Raw significant-neighbourhood counts must not be interpreted as independent discoveries.

`milo_diagnostics.tsv` summarizes Milo behaviour per cluster. In addition to tested/significant neighbourhood counts, it reports neighbourhood support summaries and a `milo_recommendation` field. Recommendations that do not start with `ok_` point to likely tuning actions, such as using broader neighbourhoods, fewer seeds, or lower nonzero-sample support requirements.

The neighbourhood table additionally records the initial sampling proportion, requested and actual graph size, refined and retained neighbourhood counts, representative index cells, neighbourhood sizes, kth-neighbour distances, annotation fractions, support, and test eligibility. The main output tables are `composition_global_milo.tsv`, `composition_milo_neighborhoods.tsv`, `composition_milo_regions.tsv`, `composition_milo_region_sample_counts.tsv`, `milo_coverage.tsv`, and `milo_diagnostics.tsv`.

## Consensus

After all requested methods finish, scOmnom writes `composition_consensus.tsv`. Milo regions are first mapped to their dominant parent cluster; raw neighbourhood identifiers are never compared directly with CLR or scCODA cluster labels. The table then summarizes method agreement per cluster and contrast, including the number of methods run, number significant, mean effect direction, sign agreement, and a `high_confidence_da` flag.

The high-confidence flag is Milo-centred: it requires significant grouped Milo evidence plus a significant, directionally concordant CLR or scCODA result. `da_evidence_tier` distinguishes `cross_scale_supported`, `cross_scale_discordant`, `local_milo_only`, `global_only`, and `no_supported_da`. A local-only Milo region can be biologically valid, particularly for a state within a broader cluster, but it requires region-level and sample-level inspection rather than being promoted automatically to a global composition claim.

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
* `composition_global_milo.tsv`;
* `composition_consensus.tsv`;
* `composition_milo_neighborhoods.tsv`;
* `composition_milo_regions.tsv`;
* `composition_milo_region_sample_counts.tsv`;
* `milo_coverage.tsv`;
* `milo_diagnostics.tsv`;
* `composition_settings.txt`.

Key DA figures include:

* composition summaries: `composition_stacked_bar_100`, `composition_stacked_comparison`, `composition_flow`;
* global effects: `composition_effects_global_sccoda`, `composition_effects_global_clr`, GLM/CLR volcanoes;
* Milo summaries: `milo_da_regions`, `milo_effects_by_cluster`, `milo_top_neighborhoods`, `milo_top_by_cluster`;
* Milo QC: `milo_qc_pval_vs_fdr`, `milo_qc_cluster_power`.

---
