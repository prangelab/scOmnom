# Markers (cluster-vs-rest)

Markers are computed per cluster against all other cells. The module supports **cell-level markers**, **pseudobulk markers**, or **both** (`--run cell|pseudobulk|both`).

By default, `markers` runs both engines:

```bash
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.zarr \
  --run both
```

If no `--group-key` is provided, scOmnom resolves the active clustering round and uses that round's stable cluster labels. If `--round-id` is provided, that round is used instead. Pseudobulk markers also need a replicate/sample key; scOmnom uses `--replicate-key`, then falls back to `adata.uns["batch_key"]` when available.

## Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `--run` | `both` | Marker engine to run: `cell`, `pseudobulk`, or `both`. |
| `--group-key` | active round labels | Explicit `adata.obs` column for groups. If omitted, the active clustering round is used. |
| `--label-source` | `pretty` | Label display source when resolving a round. |
| `--round-id` | active round | Cluster round used for marker grouping. |
| `--replicate-key` | `adata.uns["batch_key"]` | Sample/replicate column for pseudobulk. Required for pseudobulk if no batch key is stored. |
| `--min-pct` | `0.25` | Minimum expression prevalence in the target group or rest group. |
| `--min-diff-pct` | `0.25` | Minimum absolute prevalence difference between target and rest for cell-level filtering. |
| `--n-jobs` | `1` | Parallel jobs passed into pseudobulk DE. |
| `--max-workers` | `8` | Maximum parallel worker tasks for DE phases. |

**Cell-level markers**

* Uses Scanpy `rank_genes_groups` with `wilcoxon`, `t-test`, or `logreg`.
* Optional per-cluster downsampling for very large datasets.
* Filters by `min_pct` and `min_diff_pct` to remove low-coverage or weakly differential genes.
* Keeps positive markers by default; negative fold-change genes are removed from the stored marker payload.

| Option | Default | Meaning |
| --- | --- | --- |
| `--cell-method` | `wilcoxon` | Scanpy method: `wilcoxon`, `t-test`, or `logreg`. |
| `--cell-n-genes` | `300` | Maximum genes retained per cluster. |
| `--cell-rankby-abs` / `--no-cell-rankby-abs` | enabled | Scanpy ranking flag. Because markers are positive-only by default, scOmnom forces absolute ranking off during the stored positive-marker pass. |
| `--cell-use-raw` / `--no-cell-use-raw` | disabled | Use `adata.raw` for Scanpy marker testing. |
| `--cell-downsample-max-per-group` | `2000` | Maximum cells per cluster used for cell-level marker testing when downsampling is triggered. Set to `0` to disable this guard. |
| `--random-state` | `42` | Random seed for stratified downsampling. |

**Pseudobulk markers**

* Aggregates counts by sample and cluster, then runs DESeq2 (via PyDESeq2).
* Requires at least **6 unique samples** (guard to avoid unstable estimates).
* Uses count layers in priority order: `counts_cb`, `counts_raw`, then `adata.X` (if allowed).
* Supports sample-level covariates (`--pb-covariates`) and filters on minimum cells per sample-cluster.

| Option | Default | Meaning |
| --- | --- | --- |
| `--pb-counts-layer` | `counts_cb`, `counts_raw` | Candidate count layers, in priority order. Repeat or comma-separate to customize. |
| `--pb-allow-x-counts` / `--no-pb-allow-x-counts` | enabled | Allow fallback to `adata.X` if no requested count layer is found. |
| `--pb-min-cells-per-replicate-group` | `20` | Minimum cells required for each sample-by-cluster pseudobulk library. |
| `--pb-alpha` | `0.05` | FDR/significance threshold used by the pseudobulk DE output. |
| `--pb-min-counts-per-lib` | `0` | Minimum counts required for a pseudobulk library in the CLI layer. |
| `--pb-min-lib-pct` | `0.0` | Minimum fraction of libraries in which a gene must pass count filtering. |
| `--pb-max-genes` | none | Optional cap on exported pseudobulk marker genes. |
| `--pb-covariates` | none | Sample-level covariates for the DESeq2 design. Repeat or comma-separate. |
| `--pb-store-key` | `scomnom_de` | `adata.uns` key for stored pseudobulk marker payloads. |

## Plotting and output controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | enabled | Create marker plots and HTML report. |
| `--regenerate-figures` | disabled | Rebuild figures from stored marker results without recomputing markers. |
| `--figure-formats`, `-F` | `png`, `pdf` | Figure formats to write. |
| `--plot-lfc-thresh` | `1.0` | Log-fold-change threshold highlighted in volcano-style plots. |
| `--plot-volcano-top-label-n` | `15` | Number of top genes labeled on volcano plots. |
| `--plot-top-n-per-cluster` | `12` | Top genes per cluster for standard marker plots. |
| `--plot-dotplot-top-n-per-cluster` | `16` | Top genes per cluster for dotplots. |
| `--plot-max-genes-total` | `80` | Maximum total genes shown in combined marker displays. |
| `--plot-use-raw` / `--no-plot-use-raw` | disabled | Use `adata.raw` for expression plots. |
| `--plot-layer` | none | Specific layer for expression plots. |
| `--plot-umap-ncols` | `3` | Number of columns for marker UMAP panels. |
| `--gene-filter` | none | Repeatable pandas-query filters applied before marker computation. |
| `--plot-gene-filter` | none | Repeatable filters applied to plot gene selection only. |

**Outputs**

* Tables: `tables/markers_<round>_roundN/cell_based/` and `tables/markers_<round>_roundN/pseudobulk_based/`
* Reports and figures: `figures/<fmt>/markers_<round>_roundN/`
* Updated AnnData: marker payloads are stored in `adata.uns` and written to the output dataset.

---
