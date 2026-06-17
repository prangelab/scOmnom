# Markers (cluster-vs-rest)

Markers are computed per cluster against all other cells. The module supports **cell-level markers**, **pseudobulk markers**, or **both** (`--run cell|pseudobulk|both`).

**Cell-level markers**

* Uses Scanpy `rank_genes_groups` with `wilcoxon`, `t-test`, or `logreg`.
* Optional per-cluster downsampling for very large datasets.
* Filters by `min_pct` and `min_diff_pct` to remove low-coverage or weakly differential genes.

**Pseudobulk markers**

* Aggregates counts by sample and cluster, then runs DESeq2 (via PyDESeq2).
* Requires at least **6 unique samples** (guard to avoid unstable estimates).
* Uses count layers in priority order: `counts_cb`, `counts_raw`, then `adata.X` (if allowed).
* Supports sample-level covariates (`--pb-covariates`) and filters on minimum cells per sample-cluster.

**Outputs**

* Tables: `tables/marker_tables_<run>/cell_based/` and `tables/marker_tables_<run>/pseudobulk_based/`
* Reports: `figures/markers/markers_report.html` (plus PDFs/PNGs under `figures/markers/`)

---
