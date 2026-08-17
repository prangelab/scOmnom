# Manual renaming of cluster labels

If you want to override the automatic cluster labels with curated names, use `adata-ops rename`.

This creates a new clustering round with your manual names. By default it preserves the existing cluster ids
and only replaces the display labels. With `--collapse-same-labels`, clusters renamed to the same target label
are merged and renumbered by size into a new coarser annotation layer. The original labels remain intact and
you can switch between rounds by setting `active_cluster_round`.

The mapping file must be a two-column, tab-delimited text file with no header. The first column is the
cluster code (`Cnn`), the second column is the new label.

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --output-dir results \
  --output-name adata.clustered.annotated.projected.markers.renamed \
  --rename-idents-file rename.tsv \
  --rename-round-name refined_idents
```

Example mapping file (`rename.tsv`):

```tsv
C03	SOX10+ PMP22+ Schwann cells
C07	TREM1hi PLIN2hi iLAMs
C12	PECAM1+ endothelial cells
```

You can control the new manual label round name with:

```bash
--rename-round-name refined_idents
```

This will create a new round id like `r5_refined_idents`. Default suffix is `manual_rename`

To build a coarser layer such as broad cell types, merge identical renamed labels into one new cluster state:

Example broad-cell-type mapping:

```tsv
C00	T cells
C03	T cells
C05	Myeloid cells
C07	Myeloid cells
C12	Stromal cells
```

With `--collapse-same-labels`, the two T-cell rows become one new T-cell cluster state, the two myeloid rows become one new myeloid cluster state, and the resulting clusters are renumbered by size.

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --output-dir results \
  --rename-idents-file broad_cell_types.tsv \
  --rename-round-name broad_cell_types \
  --collapse-same-labels \
  --no-set-active
```

Use `--set-active/--no-set-active` to control whether the new rename round becomes the default active round.

If you want to revise an existing manual rename round in place rather than creating a new numbered round,
use `--update-existing-round` together with an explicit `--target-round-id`:

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --output-dir results \
  --output-name adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation \
  --rename-idents-file broad_cell_types.tsv \
  --target-round-id r5_broad_cell_types \
  --update-existing-round \
  --collapse-same-labels \
  --no-set-active
```

Three UMAPs are emitted for the renamed labels:

* one with the full right-side legend
* one with no legend
* one with a right-side legend plus `Cnn` overlaid on clusters

---
