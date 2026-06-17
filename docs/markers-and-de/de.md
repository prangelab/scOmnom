# DE (within-cluster contrasts)

Within-cluster DE compares **condition levels inside each cluster**, e.g. `treated vs control` within each cell type. You can provide a single `--condition-key` or multiple `--condition-keys`, plus optional explicit contrasts (`--contrasts`).

**Condition key syntax**

| Syntax | Meaning | Resulting behavior                                                           |
| --- | --- |------------------------------------------------------------------------------|
| `A` | Single obs key | Compare levels of `A` within each cluster                                    |
| `A:B` | Composite key | Create a combined key across **all combinations** of `A` and `B` levels      |
| `A@B` | Within‑B | Run **A within each level of B** (usually what you want)                     |
| `A^B` | Interaction | Run interaction contrasts between A and B (requires a pseudobulk DESeq2 run) |

**Examples**

* `--condition-keys treatment`  
  Compare treatment levels within each cluster.
* `--condition-keys MASLD:sex`  
  Full combinations of MASLD × sex.
* `--condition-keys treatment@sex`  
  Treatment contrasts within each sex.
* `--condition-keys treatment^sex`  
  Treatment‑by‑sex interaction contrasts.

**Cell-level within-cluster DE**

* Runs contrast-conditional markers per cluster using `wilcoxon` and `logreg` (configurable).
* Uses `min_pct` and `min_diff_pct` to focus on robust signals.
* Does not require the pseudobulk sample guard.
* Typically much faster than pseudobulk, which makes it practical for iterative test runs.

**Pseudobulk within-cluster DE**

* Aggregates counts by sample and cluster, then runs DESeq2 per cluster and contrast.
* Requires at least **6 unique samples** (guarded).
* Supports sample-level covariates and minimum cells per sample-cluster.
* Usually substantially slower than cell-level mode, but more rigorous for replicate-aware inference.

**Target populations**

* `--target-groups` restricts DE to selected cluster/population labels from `--group-key` (repeatable and comma-separated).
* Applies to both cell-level and pseudobulk phases, so unrelated populations are skipped in computation and outputs.

**Gene filtering**

* `--gene-filter` filters genes before DE is calculated.
* Filters are applied to both the cell-level and pseudobulk DE inputs.
* Filters are combined with logical AND and evaluated against `adata.var`.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* `--plot-gene-filter` remains plot-only and does not change the DE statistics.

**DE → pathway/TF activity (decoupler)**

* Optional activity inference from DE statistics for each contrast.
* Sources can be `cell`, `pseudobulk`, `all`, or `auto` (prefer pseudobulk if present).
* Supports MSigDB, PROGENy, and DoRothEA with configurable methods and target filters.
* MSigDB GSEA is enabled by default for `markers-and-de de` and writes additional `msigdb_gsea` and `msigdb_joint` outputs alongside the decoupler summaries.

**GSEA controls for `markers-and-de de`**

* `--run-gsea/--no-run-gsea` enables or disables MSigDB preranked GSEA on the DE statistic matrix.
* `--gsea-min-size`, `--gsea-max-size`, and `--gsea-eps` control pathway-size filtering and numerical tolerance for the Python `gseapy` backend.
* `--gsea-rank-col` lets you override the DE column used for GSEA ranking. If omitted, scOmnom uses the same statistic requested through `--de-decoupler-stat-col`.
* `--joint-enrichment-alpha` controls the adjusted-significance threshold used when building the `msigdb_joint` concordance summary.
* `--joint-enrichment-top-n` controls how many concordant up/down pathways are shown in the compact GSEA and joint plots.
* `--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant` toggles whether joint summaries require matching decoupler and GSEA direction.
* `--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig` toggles whether joint summaries require GSEA adjusted significance.
* `--joint-enrichment-leading-edge-top-n` controls how many leading-edge genes are previewed in compact GSEA summaries.

**Outputs**

* Tables: `tables/de_<round>_roundN/cell_based/` and `tables/de_<round>_roundN/pseudobulk_based/`

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
* If `--run both` is unstable on very large jobs, split into separate `--run pseudobulk` and `--run cell` runs with separate output folders.
* Figures: `figures/<fmt>/de_<round>_roundN/`
* Default AnnData output name: `adata.de_<round>.zarr.tar.zst`

---
