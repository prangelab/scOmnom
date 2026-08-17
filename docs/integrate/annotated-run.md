# Secondary integration: annotated refinement (`--annotated-run`)

After clustering and annotation, `integrate` can be run a **second time** in a supervised refinement mode.

This mode is **optional** and intended for producing cleaner label-aware embeddings and UMAPs after final biological labels are known. It is deliberately secondary because it uses labels derived from the first integration and clustering pass.

Do not treat the annotated scANVI embedding as a fresh unbiased basis for reclustering from scratch. The supervision labels come from the current annotation state, so reclustering the same cells on the supervised embedding would be circular: the labels help shape the embedding, and the embedding would then be used to rediscover labels related to the same labels. Use it for projection, visualization, label-aware refinement, and downstream interpretation, while keeping the original unsupervised clustering round available as the primary clustering evidence.

#### Key properties

* **Not the default mode**
* **Requires manual input and output naming**
* **Runs scANVI only**
* **Does not overwrite previous results**
* **Should not replace the first-pass clustering evidence**

#### What it does

1. Takes an *annotated* dataset (from `cluster-and-annotate`) as input
2. Extracts final cluster labels from the selected cluster round
3. Reconstructs a CellTypist confidence mask
4. Uses only high-confidence cells as supervision
5. Trains scANVI to *project* all cells into a refined latent space
6. Benchmarks embeddings using the chosen truth labels
7. Creates a derived *projection round* to preserve clustering state

Low-confidence cells are labeled as `Unknown` for supervision, preventing overfitting to noisy annotations.

#### Required usage pattern

You **must** explicitly set:

* the input path (annotated dataset)
* a new output name (there is no default)

Example:

```bash
scomnom integrate \
  --input-path results/cluster_and_annotate/adata.clustered.annotated.zarr \
  --annotated-run \
  --output-dir results/integrate_annotated/ \
  --output-name adata.clustered.annotated.projected \
  --benchmark-n-jobs 16
```

Notes:

* The input must point to an **annotated** object
* The output name is **not auto-derived**
* Results are additive and non-destructive

---
