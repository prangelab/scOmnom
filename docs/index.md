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

## QC Defaults

`load-and-filter` applies conservative default QC:

| Filter | Default |
| --- | --- |
| `--min-cells` | `3` |
| `--min-genes` | `500` |
| `--min-counts` | `none` |
| `--min-counts-mad` | `5.0` |
| `--min-counts-quantile` | `none` |
| `--min-counts-auto-activate-quantile` | `0.01` |
| `--min-counts-auto-activate-below` | `1000` |
| `--max-pct-mt` | `5.0` |
| `--max-genes-mad` | `5.0` |
| `--max-genes-quantile` | `0.999` |
| `--max-counts-mad` | `5.0` |
| `--max-counts-quantile` | `0.999` |
| `--expected-doublet-rate` | `0.1` |

See [Filtering Defaults And Rationale](load-and-filter/filtering.md) for the full table of defaults and tuning guidance.

## SLURM Scripts

Example SLURM job scripts are included under [`slurm/`](https://github.com/prangelab/scOmnom/tree/main/slurm). They are configured as starting points for SURF's Snellius cluster and should be adapted for local module names, CUDA/driver versions, wall time, CPU allocation, memory policy, and GPU resources.

See [SLURM and HPC](hpc-slurm.md) for the manual section.

## Core References

- [Manual](manual.md): full workflow reference, examples, expected outputs, AnnData conventions, and HPC notes.
- [Data conventions](adata-structure.md): scOmnom AnnData structure, count layers, clustering rounds, annotations, and notebook IO guidance.
- [Design goals](design-goals.md): placeholder for the project design principles.
- [API Reference](api-reference.md): public Python API exposed through `scomnom`, `scomnom.adata_ops`, `scomnom.markers_and_de`, and `scomnom.plotting`.
- [Changelog](changelog.md): release notes and user-facing behavior changes.
- [Contributing](contributing.md): project contribution guidelines.
