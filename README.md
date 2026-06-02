# scOmnom

`scOmnom` is a CLI-first, reproducible single-cell RNA-seq workflow for large datasets. It is built on the scVerse ecosystem and is designed for robust execution on local macOS workstations and HPC systems.

The pipeline combines established single-cell packages with scOmnom-specific workflow machinery, including:

* OOM-aware dataset IO and sidecar serialization for heavy AnnData payloads
* BISC, a biology-informed structural clustering workflow for Leiden resolution selection
* round-aware clustering, annotation, enrichment, renaming, subsetting, and merge operations
* CellBender-aware count layer conventions
* robust marker discovery, within-cluster DE, cell-cell communication, enrichment, and differential abundance
* GraphDA, a graph-local differential abundance implementation using neighborhood counts and spatial weighted FDR

For the full workflow manual, see [docs/manual.md](docs/manual.md). For the public Python API, see [API_REFERENCE.md](API_REFERENCE.md).

## Installation

Create the platform-specific environment:

```bash
# Linux / HPC
micromamba create -f environment_linux.yml
micromamba activate scOmnom_env

# macOS
conda env create -f environment_macos.yml
conda activate scOmnom_env
```

Install the package in editable mode:

```bash
pip install -e .
```

This registers the `scomnom` command-line interface.

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

## Standard Workflow

The recommended full workflow is:

1. `load-and-filter`
2. `integrate`
3. `cluster-and-annotate`
4. `markers-and-de markers`
5. optional `integrate --annotated-run`
6. optional `adata-ops rename`, `adata-ops subset`, or `adata-ops annotation-merge`
7. `markers-and-de de`
8. `markers-and-de da`
9. optional `markers-and-de enrichment cluster`
10. optional `markers-and-de ccc ...`

Detailed command examples, expected outputs, AnnData conventions, and HPC notes are in [docs/manual.md](docs/manual.md).

## Data Conventions

All new or modified code should load and save datasets through:

* `scomnom.load_dataset`
* `scomnom.save_dataset`

For count-based methods, scOmnom prefers assays in this order:

1. `adata.layers["counts_cb"]`
2. `adata.layers["counts_raw"]`
3. `adata.X` as a last fallback

Clustering state is stored as rounds in `adata.uns["cluster_rounds"]`, with the active round in `adata.uns["active_cluster_round"]`.

## Documentation

* [Full manual](docs/manual.md)
* [API reference](API_REFERENCE.md)
* [Design goals](DESIGNGOALS.md)
* [Contributing](contributing.md)
* [Changelog](changelog.md)

## License

MIT License.
