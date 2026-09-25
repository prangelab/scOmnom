# Enrichment

The enrichment submodule has three decoupler modes and a custom module-scoring command:

| Entry point | Input | Main use |
| --- | --- | --- |
| `scomnom enrichment cluster` | AnnData with a clustering round | Descriptive MSigDB, PROGENy, and DoRothEA profiles for clusters or cluster-condition groups. |
| `scomnom enrichment de` | Exported DE result tables | Run the same pathway/TF activity backends from DE statistics, without loading AnnData. |
| `scomnom enrichment sample` | AnnData with a clustering round and replicate metadata | Infer activities per replicate-population count pseudobulk and fit covariate-adjusted activity contrasts. |
| `scomnom enrichment module-score` | AnnData plus user gene modules | Score custom gene programs per cell, then summarize by cluster or cluster-condition. |

For decoupler-based enrichment, the default resource set is MSigDB HALLMARK + REACTOME, PROGENy, and DoRothEA. MSigDB can also use custom `.gmt` files.

## Cluster Enrichment

`enrichment cluster` recomputes round-native pseudobulk expression for the selected clustering round, then runs decoupler resources on that expression matrix.

Cluster and cluster-condition aggregates are descriptive profiles. They do not provide biological-replicate effect estimates or confidence intervals; use `enrichment sample` for those quantities.

```bash
scomnom enrichment cluster \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types
```

Add `--condition-key` when you want enrichment profiles for `cluster x condition` groups instead of one profile per cluster:

```bash
scomnom enrichment cluster \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types \
  --condition-key treatment
```

### Cluster Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. If omitted, scOmnom uses the standard results-location logic. |
| `--output-name` | inferred from input, module, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--n-jobs` | `1` | Reserved for consistency with other downstream commands. |
| `--round-id` | active clustering round | Selects which clustering round supplies the population labels. |
| `--condition-key` | none | Optional condition key for cluster-by-condition pseudobulk. |
| `--gene-filter` | none | Repeatable pandas-query expressions against `adata.var`; applied before enrichment. |

Round-native enrichment stores results back into the selected round. It does not create a new clustering round.

### Condition Groups

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One enrichment profile per cluster. |
| `A` | Single `adata.obs` key | One enrichment profile per `cluster x A level`. |
| `A:B` | Composite key | One enrichment profile per `cluster x all combinations of A and B`. |

Examples:

```bash
scomnom enrichment cluster ... --round-id r5_broad_cell_types
scomnom enrichment cluster ... --round-id r5_broad_cell_types --condition-key treatment
scomnom enrichment cluster ... --round-id r5_broad_cell_types --condition-key treatment:genotype
```

### Pseudobulk Source

| Option | Default | Notes |
| --- | --- | --- |
| `--decoupler-pseudobulk-agg` | `mean` | Aggregation used for round-native pseudobulk expression. |
| `--decoupler-use-raw` / `--no-decoupler-use-raw` | `--decoupler-use-raw` | Prefer raw-like count sources when available. |
| Preferred count layers | `counts_cb`, then `counts_raw` | Internal order when raw-like layers are available. |

The pseudobulk input is genes by cluster, or genes by cluster-condition group when `--condition-key` is set.

### Shared Decoupler Settings

| Option | Default | Notes |
| --- | --- | --- |
| `--decoupler-method` | `consensus` | Fallback method for resources that do not have a resource-specific method set. |
| `--decoupler-consensus-methods` | `ulm`, `mlm`, `wsum` | At least two distinct constituents combined by decoupler's signed per-method z-score consensus. `wsum` resolves to WAGGR weighted-sum scoring. |
| `--decoupler-min-n-targets` | `5` | Fallback minimum target count for resources without a resource-specific value. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | split signed bars | Plot positive and negative activities separately. |
| `--decoupler-bar-top-n-up` | none | Optional cap on positive barplot entries. |
| `--decoupler-bar-top-n-down` | none | Optional cap on negative barplot entries. |

### MSigDB

MSigDB runs by default with HALLMARK and REACTOME gene sets.

| Option | Default | Notes |
| --- | --- | --- |
| `--msigdb-gene-sets` | `HALLMARK,REACTOME` | Comma-separated MSigDB keywords or paths to `.gmt` files. |
| `--msigdb-method` | `consensus` | Decoupler method for MSigDB activity. |
| `--msigdb-min-n-targets` | `5` | Minimum overlap between the expression universe and a gene set. |

Example with a custom GMT:

```bash
scomnom enrichment cluster ... \
  --msigdb-gene-sets HALLMARK,REACTOME,/path/to/custom_programs.gmt
```

### PROGENy

PROGENy runs by default. Disable it with `--no-run-progeny`.

| Option | Default | Notes |
| --- | --- | --- |
| `--run-progeny` / `--no-run-progeny` | `--run-progeny` | Toggle PROGENy. |
| `--progeny-method` | `consensus` | Decoupler method. |
| `--progeny-min-n-targets` | `5` | Minimum target overlap. |
| `--progeny-top-n` | `100` | Top weighted genes per pathway from the PROGENy resource. |
| `--progeny-organism` | `human` | Organism passed to the decoupler resource loader. |

For non-human organisms, scOmnom first tries the requested decoupler PROGENy resource. If unavailable, it falls back to the human resource and translates targets through HCOP orthology before scoring.

### DoRothEA

DoRothEA runs by default. Disable it with `--no-run-dorothea`.

| Option | Default | Notes |
| --- | --- | --- |
| `--run-dorothea` / `--no-run-dorothea` | `--run-dorothea` | Toggle DoRothEA. |
| `--dorothea-method` | `consensus` | Decoupler method. |
| `--dorothea-min-n-targets` | `5` | Minimum TF-target overlap. |
| `--dorothea-confidence` | `A,B,C` | DoRothEA confidence levels to keep. |
| `--dorothea-organism` | `human` | Organism passed to the decoupler resource loader. |

For non-human organisms, scOmnom first tries the requested decoupler DoRothEA resource. If unavailable, it falls back to the human resource and translates TFs/targets through HCOP orthology before scoring.

### Cluster Gene Filtering

`--gene-filter` filters genes before decoupler activity inference. Filters are evaluated as pandas-query expressions against `adata.var`, and repeated filters are combined with logical AND.

```bash
scomnom enrichment cluster ... \
  --gene-filter "not gene.str.startswith('MT-')" \
  --gene-filter "not gene.str.startswith('RPL')" \
  --gene-filter "not gene.str.startswith('RPS')"
```

If required metadata such as `gene_type` or `gene_chrom` is missing, scOmnom tries to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of silently continuing unfiltered.

### Cluster Plotting And Outputs

| Option | Default | Notes |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create enrichment plots and reports. |
| `--regenerate-figures` | off | Rebuild figures from stored round payloads without recomputation. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |

Cluster enrichment writes:

* figures: `figures/<fmt>/enrichment_<round>_roundN/`;
* report: `figures/<fmt>/enrichment_<round>_roundN/enrichment_report.html`;
* saved AnnData: `adata.enrichment_<round>.zarr.tar.zst` by default;
* round payloads under `adata.uns["cluster_rounds"][round_id]["decoupler"]`.

## DE-Table Enrichment

`enrichment de` reads exported DE tables and computes pathway/TF activity from the DE statistic column. It does not load or modify AnnData.

```bash
scomnom enrichment de \
  --input-dir results/tables/de_r5_broad_cell_types_round1 \
  --de-decoupler-source pseudobulk
```

### DE Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-dir`, `-i` | required | DE table directory, for example `results/tables/de_<round>_roundN`. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | `enrichment_de_<input_dir_name>` | Used for output folder naming. |
| `--n-jobs` | `1` | Parallelism setting for the command. |
| `--gene-filter` | none | Repeatable pandas-query expressions against columns available in exported DE tables. |
| `--de-decoupler-source` | `auto` | Which DE tables to use: `auto`, `all`, `pseudobulk`, `cell`, or `none`. |
| `--de-decoupler-stat-col` | `stat` | Statistic column used as the signed ranking/activity input. |

Source behavior:

| Source | Meaning |
| --- | --- |
| `auto` | Use available DE sources with the normal preference logic. |
| `all` | Use both pseudobulk and cell-level DE tables when present. |
| `pseudobulk` | Restrict to pseudobulk DE tables. |
| `cell` | Restrict to cell-level DE tables. |
| `none` | Skip DE-derived decoupler. |

### DE Resource Settings

The DE-table mode uses the same decoupler resources and resource-specific defaults as cluster enrichment:

| Resource | Default | Disable/change |
| --- | --- | --- |
| MSigDB | `HALLMARK,REACTOME`; method `consensus`; min targets `5` | Change with `--msigdb-gene-sets`, `--msigdb-method`, `--msigdb-min-n-targets`. |
| PROGENy | enabled; method `consensus`; min targets `5`; top genes `100`; organism `human` | Disable with `--no-run-progeny`; change with `--progeny-*` options. |
| DoRothEA | enabled; method `consensus`; min targets `5`; confidence `A,B,C`; organism `human` | Disable with `--no-run-dorothea`; change with `--dorothea-*` options. |

The same non-human PROGENy/DoRothEA HCOP fallback applies in DE-table enrichment mode.

MSigDB DE enrichment can also run GSEA and joint decoupler/GSEA summaries through the DE-enrichment engine. The command currently exposes the decoupler resource knobs directly; the GSEA/joint defaults follow the enrichment configuration.

### DE Plotting And Outputs

| Option | Default | Notes |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create enrichment plots and reports. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | split signed bars | Plot positive and negative activities separately. |
| `--decoupler-bar-top-n-up` | none | Optional cap on positive barplot entries. |
| `--decoupler-bar-top-n-down` | none | Optional cap on negative barplot entries. |

DE-table enrichment writes:

* figures: `figures/<fmt>/enrichment_de_<inputdir>_roundN/`;
* tables: `tables/enrichment_de_<inputdir>_roundN/`;
* report: `figures/<fmt>/enrichment_de_<inputdir>_roundN/enrichment_de_report.html`.

## Sample Enrichment

`scomnom enrichment sample` produces one activity observation per eligible replicate-population library and provides scoring, inference, tables, figures, and AnnData output. The prespecified Kang validation and controls are complete. No new public Python function is exposed, and the deprecated `markers-and-de enrichment` route retains only its existing commands.

Independent libraries:

```bash
scomnom enrichment sample \
  --input-path results/adata.clustered.annotated.zarr.tar.zst \
  --round-id r1_scANVI_compacted \
  --replicate-key donor_id \
  --condition-key sex \
  --contrast female:male \
  --covariates age,BMI \
  --target-groups C03 \
  --dorothea-method ulm
```

Paired libraries:

```bash
scomnom enrichment sample \
  --input-path results/kang.clustered.annotated.zarr.tar.zst \
  --replicate-key sample_id \
  --condition-key condition \
  --contrast stimulated:control \
  --covariates donor_id \
  --subject-key donor_id
```

`replicate_key` identifies one sample library. `subject_key` identifies the donor contributing libraries in both compared conditions and must also appear in `--covariates`. Numeric subject IDs are categorical fixed effects. Complete pairs are retained after library QC and complete-case filtering; every excluded library is recorded. Multiple libraries for the same subject-condition combination are rejected. A donor identifier should not be included as a covariate when every donor contributes only one library.

### Sample Inputs And Models

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData loaded through scOmnom I/O. |
| `--replicate-key` | required for analysis | Sample-library identifier in `obs`; omit for figure regeneration. |
| `--output-dir`, `-o` | nearest `results/` ancestor, otherwise `results/` beside input | Output root. |
| `--output-name` | `adata.enrichment_sample_<round>` | Dataset stem. |
| `--save-h5ad` / `--no-save-h5ad` | off | Additional H5AD output. |
| `--round-id` | active round | Round supplying stable population IDs and display labels. |
| `--condition-key` | none | Condition column; omit for scoring without inference. |
| `--contrast` | none | `TEST:REFERENCE`; repeatable or comma-separated. |
| `--reference` | none | Explicit denominator for exactly two observed levels when no contrast is supplied. |
| `--covariates` | none | Numeric or categorical covariates; repeatable or comma-separated. |
| `--subject-key` | none | Explicit pairing identifier, also included in `--covariates`. |
| `--target-groups` | all round populations | Stable IDs, `Cnn` codes, or complete display labels; repeatable or comma-separated. |
| `--counts-layer` | `auto` | `auto`, `counts_cb`, `counts_raw`, or `X`. |
| `--min-cells-per-replicate-group` | `20` | Minimum cells per library-population pseudobulk. |
| `--min-replicates-per-level` | `3` | Minimum included libraries in each contrast level. |
| `--min-replicates-total` | `6` | Minimum total included libraries for independent designs. |
| `--min-complete-subjects` | `3` | Minimum complete subjects for paired designs. |
| `--gene-filter` | none | Repeatable `adata.var` query expressions; commas inside a query are preserved. |

Count selection prefers `counts_cb`, then `counts_raw`, then validated `X`; it never selects `adata.raw`. The selected matrix must contain finite, nonnegative, integer-like counts. An unavailable or invalid explicitly selected assay is fatal. Counts are summed by replicate and population and normalized as `log1p(counts / library_size * 1,000,000)`. Library sizes are calculated before gene filtering. By default, only genes with zero total counts across eligible round pseudobulks are removed. Population selection does not alter this shared gene universe.

The model is `activity ~ covariates + condition`, with OLS, HC3 standard errors, Student t inference, and 95% confidence intervals. Raw effects represent test minus reference. Standardized effects divide the coefficient and interval by the activity outcome's sample SD on included model rows (`ddof=1`); the binary condition indicator is not standardized. HC3 permits unequal residual variances but does not guarantee reliable inference with very small samples.

BH correction includes all successfully tested activities separately within each population-resource-contrast family. Plot selectors do not restrict this family. Missing covariates, incomplete pairs, insufficient support, constant activities, and unidentifiable designs are audited; covariates are never silently removed to obtain a fit. Globally absent contrast levels and ambiguous replicate metadata are fatal.

### Sample Resources

| Option | Default | Meaning |
| --- | --- | --- |
| `--decoupler-method` | `consensus` | Default scoring method for all enabled resources. |
| `--decoupler-consensus-methods` | `ulm,mlm,wsum` | Repeatable or comma-separated; at least two successful constituents are required. Failures are recorded. |
| `--decoupler-min-n-targets` | `5` | Default minimum target overlap. |
| `--run-msigdb` / `--no-run-msigdb` | on | Enable MSigDB. |
| `--msigdb-gene-sets` | `HALLMARK,REACTOME` | Repeatable/comma-separated collections or GMT paths; every requested collection must resolve and load. |
| `--msigdb-method`, `--msigdb-min-n-targets` | inherit general defaults | MSigDB overrides. |
| `--run-progeny` / `--no-run-progeny` | on | Enable PROGENy. |
| `--progeny-method`, `--progeny-min-n-targets` | inherit general defaults | PROGENy overrides. |
| `--progeny-top-n` | `100` | Targets per pathway. |
| `--progeny-organism` | `human` | Resource organism. |
| `--run-dorothea` / `--no-run-dorothea` | on | Enable DoRothEA. |
| `--dorothea-method`, `--dorothea-min-n-targets` | inherit general defaults | DoRothEA overrides. |
| `--dorothea-confidence` | `A,B,C` | Repeatable/comma-separated confidence levels. |
| `--dorothea-organism` | `human` | Resource organism. |
| `--plot-activity` | leading activities | Exact activity names for forest and sample plots; repeatable/comma-separated. Does not restrict inference or the overview. |
| `--figure-formats`, `-F` | `png,pdf` | Figure formats; repeatable/comma-separated. |
| `--make-figures` / `--no-make-figures` | on | Generate figures alongside analysis tables. |
| `--regenerate-figures` | off | Render stored tables without recomputing scores or models, or rewriting the dataset. |
| `--analysis-id` | sole stored analysis | Regeneration selector; required when the selected round contains multiple sample analyses. |

Each population is scored separately with the existing decoupler backend. Scores are not assumed comparable across populations. Selecting `--dorothea-method ulm` runs ULM alone for that resource; a failure does not trigger a substitute method.

### Sample Outputs

Tables live under `tables/enrichment_sample_<round>_roundN/`: `pseudobulk_qc.tsv`, `activity_scores.tsv`, `activity_contrasts.tsv`, `model_audit.tsv`, `model_exclusions.tsv`, and `resource_provenance.tsv`. `settings.json` records the command tokens, resolved settings, provenance, units, and run status. The QC table distinguishes eligibility from selection for scoring. `n_excluded` includes libraries outside a requested contrast as well as QC and model exclusions; reasons appear in `model_exclusions.tsv`.

In `resource_provenance.tsv`, `n_genes` is the dataset-wide filtered gene count and `n_scoring_genes` counts genes with nonzero expression in that population's eligible libraries. `target_overlap` uses the latter gene set, matching decoupler's handling of empty features; activities below the minimum are marked `insufficient_target_overlap`.

The archived output is `adata.enrichment_sample_<round>.zarr.tar.zst`, with optional H5AD. Tables and audit payloads are stored under `adata.uns["cluster_rounds"][round_id]["sample_enrichment"][analysis_id]`. Existing cluster enrichment and DE payloads remain separate. Older objects without `sample_enrichment` remain valid. Output naming that would replace the input dataset is rejected.

### Sample Figures And Regeneration

Figures are saved under `figures/<format>/enrichment_sample_<round>_roundN/`. Each population has a cell-count/library-size QC plot, including excluded libraries. Each successfully tested population-resource-contrast family has an overview of all standardized effects versus `-log10(FDR)`, a forest plot with 95% intervals and numeric FDR labels, and individual-library plots. Defaults show up to ten forest estimates and three sample activities, ranked by FDR, then absolute standardized effect, then name. `--plot-activity` replaces these selections without changing the tested family. Unknown activity names are rejected.

Sample plots show unadjusted scores and a separately labelled adjusted effect and interval. Crosses mark scored libraries excluded from the model. Connecting lines require explicit pairing and two model-included libraries from the same subject. Scoring-only runs show QC and up to three activities per population-resource, chosen alphabetically unless explicitly selected; they do not show inferential estimates.

Regeneration reads saved configuration and result tables, so count assays, resource downloads, and model fits are unnecessary:

```bash
scomnom enrichment sample \
  --input-path results/adata.enrichment_sample_r1.zarr.tar.zst \
  --regenerate-figures \
  --round-id r1 \
  --analysis-id enrichment_sample_r1_round1 \
  --plot-activity STAT2,IRF9 \
  --figure-formats png,pdf
```

Only input/output location, round/analysis selection, and rendering options are accepted during regeneration; analysis overrides are rejected. Regenerated figures receive a new `<analysis_id>_regeneration_roundN` folder, with a separate manifest under `figures/regeneration/`. The source dataset and original tables remain unchanged.

## Module Score

`enrichment module-score` scores custom gene modules per cell, then summarizes those scores by cluster or by cluster-condition group.

```bash
scomnom enrichment module-score \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types \
  --module-file gene_programs.tsv \
  --module-set-name immune_programs
```

### Module Inputs

| Option | Default | Notes |
| --- | --- | --- |
| `--module-file` | required | Repeatable. Supports `.gmt`, `.tsv`, `.csv`, `.txt`, and `.list`. |
| `--module-set-name` | first module file stem | Stable name used in output names and stored score keys. |

Supported module file formats:

| Format | Expected structure |
| --- | --- |
| `.gmt` | Standard GMT: module name, description, then genes. |
| `.tsv` / `.csv` | Prefer columns named `module` and `gene`; also accepts `set`/`signature` and `genes`/`symbol`. If no known names exist, the first two columns are interpreted as module and gene. |
| `.txt` / `.list` | One gene per line; the file stem becomes the module name. |

### Module Grouping

Module scoring uses the same grouped-state syntax as `enrichment cluster`.

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One summary profile per cluster. |
| `A` | Single `adata.obs` key | One summary profile per `cluster x A level`. |
| `A:B` | Composite key | One summary profile per `cluster x all combinations of A and B`. |

### Module Score Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--module-score-method` | `scanpy` | Backend: `scanpy` or `aucell`. |
| `--module-score-use-raw` / `--no-module-score-use-raw` | `--no-module-score-use-raw` | Use `adata.raw`; cannot be combined with `--module-score-layer`. |
| `--module-score-layer` | none | Use a named `adata.layers` matrix; cannot be combined with `--module-score-use-raw`. |
| `--module-score-ctrl-size` | `50` | Scanpy control-gene pool size. Used only by `scanpy`. |
| `--module-score-n-bins` | `25` | Scanpy expression bin count. Used only by `scanpy`. |
| `--module-score-random-state` | `0` | Scanpy random seed. |
| `--module-score-max-umaps` | `12` | Maximum module score columns to plot on UMAP. |

`scanpy` uses `scanpy.tl.score_genes`. `aucell` uses `decoupler.mt.aucell` with `tmin=1`.

### Module Score Outputs

Module-score writes:

* figures: `figures/<fmt>/module_score_<set>_<round>_roundN/`;
* tables: `tables/module_score_<set>_<round>_roundN/`;
* report: `figures/<fmt>/module_score_<set>_<round>_roundN/module_score_report.html`;
* saved AnnData: `adata.module_score_<set>_<round>.zarr.tar.zst` by default;
* per-cell score columns in `adata.obs` named `module_score__<round>__<set>__<module>`;
* round-level payloads under `adata.uns["cluster_rounds"][round_id]["module_scores"]`.

Key tables:

* `module_meta.tsv`;
* `module_score_summary_mean.tsv`;
* `module_score_summary_median.tsv`;
* `module_score_summary_mean_z.tsv`;
* `module_score_group_sizes.tsv`;
* `__settings.txt`.

---
