# Usage

`scOmnom` exposes a Typer-based command-line interface.

```bash
scomnom --help
```

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
6. `adata-ops rename` (optional but common before DE/DA)
7. `markers-and-de de`
8. `markers-and-de da`
9. `markers-and-de enrichment cluster` (optional)

Optional subset branch between steps 5 and 6:
* `adata-ops subset` on the projected parent object
* run subset objects through the same `integrate -> cluster-and-annotate -> integrate --annotated-run` pattern
* `adata-ops annotation-merge` back into the parent object
* then continue with parent-level `adata-ops rename`, `de`, and `da`

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
  --output-dir results/

# 3) Cluster and annotate
scomnom cluster-and-annotate \
  --input-path results/adata.integrated.zarr \
  --output-dir results/ \
  --celltypist-model Immune_All_Low.pkl

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
```

---
