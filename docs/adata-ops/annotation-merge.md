# Annotation merge (subset back-merge)

Use `adata-ops annotation-merge` to project refined labels from one or more subset datasets back into the parent dataset.

Typical use case:

* subset a parent object to a lineage of interest
* re-cluster and re-annotate the subset
* merge those refined subset labels back into the original parent round as a new `subset_annotation` layer

By default this creates a new subset-annotation round (for example `r4_subset_annotation`) and keeps earlier rounds intact.

Minimal example:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --output-dir results \
  --annotation-merge-round-name subset_annotation
```

Multiple child subsets can be merged in one run by repeating `--child-path`:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_myeloid.zarr.tar.zst \
  --output-dir results \
  --annotation-merge-round-name subset_annotation
```

Key controls:

* `--round-id`: parent round to merge into (default: active parent round)
* `--child-round-id`: round to read from each child (default: active child round)
* `--child-source-field`: explicit child `obs` column to merge (default: child round pretty labels)
* `--annotation-merge-round-name`: suffix for newly created merge round
* `--update-existing-round` with `--target-round-id`: update an existing subset-annotation round in place instead of creating a new numbered round

In-place update example:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --output-dir results \
  --target-round-id r4_subset_annotation \
  --update-existing-round
```

---
