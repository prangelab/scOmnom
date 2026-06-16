# scOmnom

`scOmnom` is a CLI-first, reproducible single-cell RNA-seq workflow for large datasets. It is built on the scVerse ecosystem and is designed for robust execution on local macOS workstations and HPC systems.

The manual covers the complete workflow:

1. `load-and-filter` for droplet selection, QC, HVG selection, and doublet detection
2. `integrate` for batch correction and benchmarking
3. `cluster-and-annotate` for BISC-guided clustering, CellTypist labels, and decoupler activities
4. optional annotated integration for scANVI refinement
5. `markers-and-de` for markers, DE, DA, enrichment, and cell-cell communication analyses

## Quick Start

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

# 4) Marker discovery
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.zarr

# 5) Within-cluster differential expression
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-key condition

# 6) Differential abundance
scomnom markers-and-de da \
  --input-path results/adata.clustered.annotated.markers.de.zarr \
  --condition-key condition
```

Use `scomnom --help` and `scomnom <command> --help` for command-specific options.

## Core References

- [Manual](manual.md): full workflow reference, examples, expected outputs, AnnData conventions, and HPC notes.
- [API Reference](api-reference.md): public Python API exposed through `scomnom`, `scomnom.adata_ops`, `scomnom.markers_and_de`, and `scomnom.plotting`.
- [Changelog](changelog.md): release notes and user-facing behavior changes.
- [Contributing](contributing.md): project contribution guidelines.
