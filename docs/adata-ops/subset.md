# Subset

Use `adata-ops subset` to split a parent AnnData object into one or more named subsets based on cluster membership.

This is the usual entry point for focused re-analysis:

1. Run the parent object through `integrate`, `cluster-and-annotate`, and optionally `integrate --annotated-run`.
2. Use `adata-ops subset` on the projected parent object to extract one or more lineages or populations.
3. Run each subset through the same `integrate -> cluster-and-annotate -> integrate --annotated-run` pattern.
4. Use [`adata-ops annotation-merge`](annotation-merge.md) to write the refined subset labels back into the parent object.
5. Continue with parent-level [`adata-ops rename`](rename.md), DE, and DA.

The subset mapping is cluster-round aware. The preferred mapping form is `Cnn -> subset_name`, and the public API also accepts `subset_name -> [Cnn, ...]` for in-memory workflows.

## CLI

```bash
scomnom adata-ops subset \
  --input-path results/adata.clustered.annotated.projected.zarr \
  --subset-mapping-tsv subset_map.tsv \
  --output-dir results/
```

Useful options:

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--input-path`, `-i` | yes | none | Parent dataset to split (`.zarr`, `.zarr.tar.zst`, or `.h5ad`). |
| `--subset-mapping-tsv`, `-s` | yes | none | Two-column tab-delimited mapping file, no header: `Cnn<TAB>subset_name`. |
| `--output-dir`, `-o` | no | input parent directory | Output root. Subset datasets are written under `subsets/`; the summary table is written under `tables/`. |
| `--output-format` | no | inferred from input, usually `zarr` | Output format for subset datasets: `zarr` or `h5ad`. |
| `--round-id` | no | active cluster round | Cluster round used to resolve labels and parse `Cnn` codes. Use this when the parent carries multiple rounds and you do not want the active one. |

## Mapping TSV

The TSV has exactly two columns and no header. The first column is a strict `Cnn` cluster code, and the second column is the subset output name. Multiple rows may point to the same subset name.

```tsv
C00	lymphoid
C03	lymphoid
C07	myeloid
C11	stromal
```

This would create three subset datasets: one containing clusters `C00` and `C03`, one containing `C07`, and one containing `C11`.

Each cluster code can only map to one subset. If a requested `Cnn` code is not present in the resolved cluster labels, the command stops instead of silently creating an incomplete subset.

## Outputs

For an input named `adata.clustered.annotated.projected.zarr`, outputs look like:

```text
results/
├── subsets/
│   ├── adata.clustered.annotated.projected__subset_lymphoid.zarr
│   ├── adata.clustered.annotated.projected__subset_myeloid.zarr
│   └── adata.clustered.annotated.projected__subset_stromal.zarr
└── tables/
    └── adata.clustered.annotated.projected__subset_summary.tsv
```

Each subset keeps the matching cells, refreshes round metadata for the smaller object, and stores the subset name in `adata.uns["Compartment"]`.

See also the public API entry [`subset_adata_by_cluster_mapping`](../api-reference.md#subset_adata_by_cluster_mapping).
