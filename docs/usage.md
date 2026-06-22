# Usage

`scOmnom` exposes a Typer-based command-line interface.

```bash
scomnom --help
```

This page is a compact route map rather than a full tutorial. Validation-backed tutorials can be added once example runs are finalized.

Each subcommand provides detailed help:

```bash
scomnom <command> --help
```

The standard workflow consists of:

1. `load-and-filter`
2. `integrate`
3. `cluster-and-annotate`
4. `markers-and-de markers` (recommended before naming/subsetting)
5. `integrate --annotated-run` (optional refinement)
6. optional focused subset refinement
7. `adata-ops rename` (optional but common before DE/DA)
8. `markers-and-de de`
9. `markers-and-de da`
10. `markers-and-de enrichment cluster` (optional)
11. `markers-and-de ccc ...` for LIANA, NicheNet, or MEBOCOST cell-cell communication analyses (optional)

Optional branches:

* External AnnData import: use `adata-ops import` when starting from an already filtered AnnData object produced outside scOmnom.
* Subset refinement: use `adata-ops subset`, rerun the subset through `integrate`, `cluster-and-annotate`, and optional `integrate --annotated-run`, then merge labels back with `adata-ops annotation-merge`.
* Annotated integration refinement: use `integrate --annotated-run` after initial clustering and annotation when label-aware projection is useful.
* Downstream interpretation: run enrichment and CCC analyses after marker discovery, DE/DA, and final label cleanup.

Detailed import and subset mechanics live in [AnnData Operations](adata-ops.md), especially [Import external AnnData](adata-ops/import.md), [Subset](adata-ops/subset.md), and [Annotation Merge](adata-ops/annotation-merge.md). The [Pipeline Overview](pipeline-overview.md) shows where the optional subset and annotated-refinement loops fit in the full workflow.

Minimal end-to-end example (stringing outputs across modules):

```bash
# 1) Load, QC, and filter
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/

# 2) Integrate batches
scomnom integrate \
  --input-path results/adata.filtered.zarr \
  --output-dir results/ \
  --celltypist-model Immune_All_Low.pkl

# 3) Cluster and annotate
scomnom cluster-and-annotate \
  --input-path results/adata.integrated.zarr \
  --output-dir results/

# 4) Optional: refined integration after annotation
scomnom integrate \
  --input-path results/adata.clustered.annotated.zarr \
  --annotated-run \
  --output-dir results/ \
  --output-name adata.clustered.annotated.projected

# If you skip step 4, use results/adata.clustered.annotated.zarr below.

# 5a) Markers (cluster-vs-rest)
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.projected.zarr

# 5b) DE (within-cluster contrasts)
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --condition-key condition

# 5c) DA (composition)
scomnom markers-and-de da \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr \
  --condition-key condition

# 5d) Optional CCC analysis
scomnom markers-and-de ccc liana \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr \
  --condition-key condition
```

---
