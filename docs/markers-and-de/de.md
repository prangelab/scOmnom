# DE (within-cluster contrasts)

Within-cluster DE compares **condition levels inside each cluster**, e.g. `treated vs vehicle` within each cell type. You can provide a single `--condition-key` or multiple `--condition-keys`, plus optional explicit contrasts (`--contrasts`).

**Condition key syntax**

| Syntax | Meaning | Resulting behavior                                                           |
| --- | --- |------------------------------------------------------------------------------|
| `A` | Single obs key | Compare levels of `A` within each cluster                                    |
| `A:B` | Composite key | Create a combined key across **all combinations** of `A` and `B` levels      |
| `A@B` | Within‑B | Run **A within each level of B** (usually what you want)                     |
| `A^B` | Interaction | Run interaction contrasts between A and B (requires a pseudobulk DESeq2 run) |

**Examples**

Single treatment contrast within each cluster:

```bash
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-key treatment \
  --contrasts treated_vs_vehicle
```

This compares `treated` against `vehicle` inside each cluster. In `A_vs_B`, `A` is the test level and `B` is the reference level.

Explicit treatment contrasts within each genotype:

```bash
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-keys treatment:genotype \
  --contrasts treated.KO_vs_vehicle.KO \
  --contrasts treated.WT_vs_vehicle.WT
```

This first creates composite condition labels from `treatment` and `genotype`, then runs exactly the requested contrasts:

| Contrast | Meaning |
| --- | --- |
| `treated.KO_vs_vehicle.KO` | Treated vs vehicle among `KO` samples, within each cluster |
| `treated.WT_vs_vehicle.WT` | Treated vs vehicle among `WT` samples, within each cluster |

Shorthand for treatment within genotype:

```bash
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-keys treatment@genotype
```

This runs treatment contrasts within each genotype level. The shorthand uses the resolved treatment reference level; explicit `--contrasts` are ignored for `A@B`, so use `treatment:genotype` when you want to spell out the exact composite contrasts.

Treatment-by-genotype interaction:

```bash
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-keys treatment^genotype \
  --run pseudobulk
```

This tests whether the treatment effect differs by genotype inside each cluster. Interaction contrasts require the pseudobulk DESeq2 path.

**Core controls**

| Option | Default | Meaning |
| --- | --- | --- |
| `--run` | `both` | Run `cell`, `pseudobulk`, or `both` DE engines. |
| `--group-key` | active round labels | Explicit `adata.obs` grouping column. If omitted, scOmnom resolves the active clustering round. |
| `--label-source` | `pretty` | Label display source when resolving round labels. |
| `--round-id` | active round | Cluster round used for grouping. |
| `--replicate-key` | `adata.uns["batch_key"]` | Sample/replicate column for pseudobulk. Required for pseudobulk if no batch key is stored. |
| `--condition-key` | none | Single condition key. Either this or `--condition-keys` is required. |
| `--condition-keys` | none | Repeatable/comma-separated condition specs, including `A`, `A:B`, `A@B`, and `A^B`. |
| `--contrasts` | all available pairs or reference-vs-other pairs | Explicit contrast(s), such as `treated_vs_vehicle`. Ignored for `A@B`. |
| `--min-pct` | `0.25` | Minimum expression prevalence used by both engines. |
| `--min-diff-pct` | `0.25` | Minimum prevalence difference used by cell-level contrasts and stored in pseudobulk settings. |
| `--n-jobs` | `1` | CPU count passed through to DE orchestration and plotting. |
| `--max-workers` | `8` | Worker cap for DE task scheduling. |

**Cell-level within-cluster DE**

* Runs contrast-conditional markers per cluster using `wilcoxon` and `logreg`.
* Uses `min_pct` and `min_diff_pct` to focus on robust signals.
* Does not require the pseudobulk sample guard.
* Typically much faster than pseudobulk, which makes it practical for iterative test runs.

| Setting | Default | Meaning |
| --- | --- | --- |
| `--run cell` | off unless selected | Run only the cell-level engine. |
| `--run both` | default | Run cell-level contrasts plus pseudobulk when the pseudobulk sample guard passes. |
| contrast methods | `wilcoxon`, `logreg` | Effective cell-level methods for within-cluster DE. These are config defaults, not separate user-facing CLI flags for this subcommand. |
| minimum cells per condition level | `50` | Config-level guard used by cell-level contrast testing. |
| maximum cells per condition level | `2000` | Config-level downsampling guard for cell-level contrast testing. |
| minimum total counts | `10` | Config-level low-count guard for cell-level contrast testing. |
| pseudocount | `1.0` | Config-level pseudocount used in cell-level effect calculations. |
| `--min-pct` | `0.25` | Minimum expression prevalence. |
| `--min-diff-pct` | `0.25` | Minimum absolute prevalence difference. |

**Pseudobulk within-cluster DE**

* Aggregates counts by sample and cluster, then runs DESeq2 per cluster and contrast.
* Requires at least **6 unique samples** (guarded).
* Supports sample-level covariates and minimum cells per sample-cluster.
* Usually substantially slower than cell-level mode, but more rigorous for replicate-aware inference.

| Option | Default | Meaning |
| --- | --- | --- |
| `--run pseudobulk` | off unless selected | Run only the pseudobulk engine. |
| `--pb-counts-layer` | `counts_cb`, `counts_raw` | Candidate count layers, in priority order. Repeat or comma-separate to customize. |
| `--pb-allow-x-counts` / `--no-pb-allow-x-counts` | enabled | Allow fallback to `adata.X` if no requested count layer is found. |
| `--pb-min-cells-per-replicate-group` | `20` | Minimum cells required for each sample-by-cluster library. |
| `--pb-alpha` | `0.05` | FDR/significance threshold. |
| `--pb-store-key` | `scomnom_de` | `adata.uns` key for stored DE payloads. |
| `--pb-max-genes` | none | Optional cap on exported pseudobulk DE genes. |
| `--pb-min-counts-per-lib` | `0` | Minimum counts required for a pseudobulk library at the CLI layer. |
| `--pb-min-lib-pct` | `0.0` | Minimum fraction of pseudobulk libraries where a gene must pass count filtering. |
| `--pb-covariates` | none | Sample-level covariates for the DESeq2 design. Repeat or comma-separate. |
| `--prune-uns-de` / `--no-prune-uns-de` | enabled | Prune bulky DE payloads before saving the output AnnData. |

**Target populations**

* `--target-groups` restricts DE to selected cluster/population labels from `--group-key` (repeatable and comma-separated).
* Applies to both cell-level and pseudobulk phases, so unrelated populations are skipped in computation and outputs.

| Option | Default | Meaning |
| --- | --- | --- |
| `--target-groups` | none | Restrict computation to selected cluster/population labels. Can be repeated or comma-separated. |
| `--group-key` | active round labels | Label namespace used to match `--target-groups`. |

**Gene filtering**

* `--gene-filter` filters genes before DE is calculated.
* Filters are applied to both the cell-level and pseudobulk DE inputs.
* Filters are combined with logical AND and evaluated against `adata.var`.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* `--plot-gene-filter` remains plot-only and does not change the DE statistics.

| Option | Default | Meaning |
| --- | --- | --- |
| `--gene-filter` | none | Repeatable pandas-query filter applied before DE computation. |
| `--plot-gene-filter` | none | Repeatable pandas-query filter applied only to plot gene selection. |
| `--plot-sample-annotation-keys` | condition keys | Extra `adata.obs` keys used for sample heatmap annotations. |

**DE → pathway/TF activity (decoupler)**

* Optional activity inference from DE statistics for each contrast.
* Sources can be `cell`, `pseudobulk`, `all`, or `auto` (prefer pseudobulk if present).
* Supports MSigDB, PROGENy, and DoRothEA with configurable methods and target filters.
* MSigDB GSEA is enabled by default for `markers-and-de de` and writes additional `msigdb_gsea` and `msigdb_joint` outputs alongside the decoupler summaries.

| Option | Default | Meaning |
| --- | --- | --- |
| `--de-decoupler-source` | `auto` | Source for DE-derived activity inference: `auto`, `all`, `pseudobulk`, `cell`, or `none`. |
| `--de-decoupler-stat-col` | `stat` | DE statistic column used as the activity input. |
| `--decoupler-method` | `consensus` | Default decoupler method. |
| `--decoupler-consensus-methods` | `ulm`, `mlm`, `wsum` | Methods combined when using `consensus`; user can repeat this option. |
| `--decoupler-min-n-targets` | `5` | Minimum targets per source. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | enabled | Split decoupler barplots into positive and negative activity directions. |
| `--decoupler-bar-top-n-up` | none | Number of positive activities to show in split barplots. |
| `--decoupler-bar-top-n-down` | none | Number of negative activities to show in split barplots. |
| `--msigdb-gene-sets` | `HALLMARK`, `REACTOME` | Comma-separated MSigDB keywords or `.gmt` paths. |
| `--msigdb-method` | `consensus` | Method override for MSigDB. |
| `--msigdb-min-n-targets` | `5` | Minimum targets per MSigDB pathway. |
| `--run-progeny` / `--no-run-progeny` | enabled | Enable PROGENy activity inference. |
| `--progeny-method` | `consensus` | Method override for PROGENy. |
| `--progeny-min-n-targets` | `5` | Minimum targets per PROGENy pathway. |
| `--progeny-top-n` | `100` | Number of top PROGENy target genes per pathway. |
| `--progeny-organism` | `human` | Organism used for PROGENy resources. |
| `--run-dorothea` / `--no-run-dorothea` | enabled | Enable DoRothEA TF activity inference. |
| `--dorothea-method` | `consensus` | Method override for DoRothEA. |
| `--dorothea-min-n-targets` | `5` | Minimum targets per TF regulon. |
| `--dorothea-confidence` | `A,B,C` | DoRothEA confidence levels to include. |
| `--dorothea-organism` | `human` | Organism used for DoRothEA resources. |

**GSEA controls for `markers-and-de de`**

* `--run-gsea/--no-run-gsea` enables or disables MSigDB preranked GSEA on the DE statistic matrix.
* `--gsea-min-size`, `--gsea-max-size`, and `--gsea-eps` control pathway-size filtering and numerical tolerance for the Python `gseapy` backend.
* `--gsea-rank-col` lets you override the DE column used for GSEA ranking. If omitted, scOmnom uses the same statistic requested through `--de-decoupler-stat-col`.
* `--joint-enrichment-alpha` controls the adjusted-significance threshold used when building the `msigdb_joint` concordance summary.
* `--joint-enrichment-top-n` controls how many concordant up/down pathways are shown in the compact GSEA and joint plots.
* `--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant` toggles whether joint summaries require matching decoupler and GSEA direction.
* `--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig` toggles whether joint summaries require GSEA adjusted significance.
* `--joint-enrichment-leading-edge-top-n` controls how many leading-edge genes are previewed in compact GSEA summaries.

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-gsea` / `--no-run-gsea` | enabled | Enable MSigDB preranked GSEA from DE statistics. |
| `--gsea-min-size` | `10` | Minimum gene-set size. |
| `--gsea-max-size` | `500` | Maximum gene-set size. |
| `--gsea-eps` | `1e-10` | Numerical tolerance for the GSEA backend. |
| `--gsea-rank-col` | `--de-decoupler-stat-col` | Preferred DE statistic for GSEA ranking. |
| `--joint-enrichment-alpha` | `0.05` | Adjusted-significance threshold for joint summaries. |
| `--joint-enrichment-top-n` | `20` | Number of up/down pathways shown in compact GSEA and joint plots. |
| `--joint-enrichment-require-concordant` / `--no-joint-enrichment-require-concordant` | enabled | Require matching decoupler and GSEA directions in joint summaries. |
| `--joint-enrichment-require-gsea-sig` / `--no-joint-enrichment-require-gsea-sig` | enabled | Require GSEA adjusted significance in joint summaries. |
| `--joint-enrichment-leading-edge-top-n` | `8` | Number of leading-edge genes previewed in compact summaries. |

**Outputs**

* Tables: `tables/de_<round>_roundN/cell_based/` and `tables/de_<round>_roundN/pseudobulk_based/`
* Figures: `figures/<fmt>/de_<round>_roundN/`
* Default AnnData output name: `adata.de_<round>.zarr.tar.zst`

| Option | Default | Meaning |
| --- | --- | --- |
| `--output-dir`, `-o` | inferred results directory | Output root. |
| `--output-name` | `adata.de_<round>` | Base name for the output AnnData. |
| `--save-h5ad` / `--no-save-h5ad` | disabled | Also write an H5AD copy. |
| `--make-figures` / `--no-make-figures` | enabled | Create DE plots and reports. |
| `--regenerate-figures` | disabled | Rebuild figures from stored results without recomputing DE. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Figure formats to write. |
| `--plot-lfc-thresh` | `1.0` | Log-fold-change threshold highlighted in volcano-style plots. |
| `--plot-volcano-top-label-n` | `15` | Number of top genes labeled on volcano plots. |
| `--plot-top-n-per-cluster` | `12` | Top genes per cluster for standard expression plots. |
| `--plot-dotplot-top-n-per-cluster` | `16` | Top genes per cluster for dotplots. |
| `--plot-max-genes-total` | `80` | Maximum total genes shown in combined displays. |
| `--plot-use-raw` / `--no-plot-use-raw` | disabled | Use `adata.raw` for expression plots. |
| `--plot-layer` | none | Specific layer for expression plots. |
| `--plot-umap-ncols` | `3` | Number of columns for UMAP panels. |

**HPC stability/throughput recipe (large pseudobulk runs)**

For large DE jobs on HPC, use a conservative worker cap and disable nested BLAS/OpenMP threading in the job script:

Base template: `slurm/scomnom_template.job`

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Then run DE with:

```bash
scomnom markers-and-de de ... --run pseudobulk --max-workers 8
```

Notes:
* `--max-workers` defaults to `8` for DE.
* Keep `--n-jobs $SLURM_CPUS_PER_TASK`; `--max-workers` controls DE task concurrency.

---
