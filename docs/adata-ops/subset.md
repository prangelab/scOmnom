# Subset

Use `adata-ops subset` to split a parent AnnData object into one or more named subsets based on cluster membership.

This is the usual entry point for focused re-analysis:

1. Run the parent object through `integrate`, `cluster-and-annotate`, and optionally `integrate --annotated-run`.
2. Use `adata-ops subset` on the projected parent object to extract one or more lineages or populations.
3. Run each subset through the same `integrate -> cluster-and-annotate -> integrate --annotated-run` pattern.
4. Use [`adata-ops annotation-merge`](annotation-merge.md) to write the refined subset labels back into the parent object.
5. Continue with parent-level [`adata-ops rename`](rename.md), DE, and DA.

The subset mapping is cluster-round aware. The preferred mapping form is `Cnn -> subset_name`, and the public API also accepts `subset_name -> [Cnn, ...]` for in-memory workflows.

See also the public API entry [`subset_adata_by_cluster_mapping`](../api-reference.md#subset_adata_by_cluster_mapping).
