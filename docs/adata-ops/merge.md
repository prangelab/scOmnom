# Full multi-adata merge (`adata-ops merge`)

Use `adata-ops merge` when you want to concatenate two or more full datasets into one new AnnData object
(not just overlay subset labels back into a parent).

Minimal example (merge complete inputs):

```bash
scomnom adata-ops merge \
  -i results/dataset_A.zarr.tar.zst \
  -i results/dataset_B.zarr.tar.zst \
  -o results \
  --dataset-short-label reset \
  --dataset-short-label anchor
```

Subset merge example (only explicit cluster selections are kept):

```bash
scomnom adata-ops merge \
  -i results/dataset_A.zarr.tar.zst \
  -i results/dataset_B.zarr.tar.zst \
  -o results \
  --subset-merge merge.tsv \
  --round-id r5_archetypes
```

`merge.tsv` format (two columns, tab-delimited, one selection per row):

```text
dataset_A	C00
dataset_A	Macrophages
dataset_B	C07
```

Key behavior:

* Repeat `-i/--input-path` for every dataset to merge.
* Optionally provide `--dataset-short-label` once per input (same order as `-i`) to control compact source labels in merged cluster naming. Fallback is `dataset1`, `dataset2`, ...
* Input basenames must be unique (used as dataset IDs in subset TSV matching and provenance).
* If `--subset-merge` is provided, only explicitly listed selections are merged.
* Cluster tokens in `merge.tsv` can be cluster IDs (`Cnn`) or exact pretty labels.
* `--round-id` controls which annotation layer is used for token resolution in scOmnom objects.
* For non-scOmnom objects, use `--cluster-key` (or fallback to `leiden`) for subset resolution.
* Feature join is configurable with `--join {outer,inner}` (default: `outer`).
* By default, stale inherited embeddings are discarded and PCA/neighbors/UMAP are recomputed; disable with `--no-recompute-embedding`.

Outputs:

* merged AnnData output (`adata.merged.zarr.tar.zst` by default)
* merge diagnostics and UMAPs under `figures/png/merge/...` and `figures/pdf/merge/...`
* provenance metadata stored in `adata.uns["merge"]` (inputs, selections, resolved labels, per-dataset counts)
* source metadata columns in `obs`: `merge_source_dataset`, `merge_source_dataset_short`, `merge_source_cluster_id`, `merge_source_cluster_label`, `merge_source_cluster_composite`, plus merged labels `merge_cluster_id`, `merge_cluster_label`

---
