# scOmnom

`scOmnom` is a CLI-first, reproducible single-cell RNA-seq workflow for large datasets. It is built on the scVerse ecosystem and is designed for robust execution on local Linux or macOS systems and HPC systems.

The pipeline combines established single-cell packages with scOmnom-specific workflow machinery, including:

* memory pressure-aware dataset IO and sidecar serialization for heavy AnnData payloads
* BISC, a biology-informed structural clustering workflow for Leiden resolution selection
* round-aware clustering, annotation, enrichment, renaming, subsetting, and merge operations
* CellBender-aware count layer conventions
* robust marker discovery, within-cluster DE, cell-cell communication, enrichment, and differential abundance
* GraphDA, a graph-local differential abundance implementation using neighborhood counts and spatial weighted FDR

For the workflow manual, see [prangelab.org/scOmnom](https://prangelab.org/scOmnom/). For the public Python API, see [API_REFERENCE.md](API_REFERENCE.md).

## Installation

Clone the repository:

```bash
git clone https://github.com/prangelab/scOmnom.git
cd scOmnom
```

Create the platform-specific environment:

```bash
# Linux / HPC
conda env create -f environment_linux.yml
conda activate scOmnom_env

# macOS
conda env create -f environment_macos.yml
conda activate scOmnom_env
```

Install the package:

```bash
pip install .
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
  --output-dir results/ \
  --celltypist-model Immune_All_Low.pkl

# 3) Cluster and annotate
scomnom cluster-and-annotate \
  --input-path results/adata.integrated.zarr \
  --output-dir results/

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

Default `load-and-filter` thresholds:

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
| `--doublet-score-mode` | `auto` |
| `--solo-sparse-nnz-limit` | `1500000000` |
| `--solo-max-cells-per-block` | `none` |

See the [filtering defaults and rationale](https://prangelab.org/scOmnom/load-and-filter/filtering/) section for details.

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

Detailed command examples, expected outputs, AnnData conventions, and HPC notes are in the [manual](https://prangelab.org/scOmnom/).

## Data Conventions

When accessing a scOmnom AnnData object in a Python session, always load and save through:

* `scomnom.load_dataset`
* `scomnom.save_dataset`

For count-based methods, scOmnom prefers assays in this order:

1. `adata.layers["counts_cb"]`
2. `adata.layers["counts_raw"]`
3. `adata.X` as a last fallback

Clustering state is stored as rounds in `adata.uns["cluster_rounds"]`, with the active round in `adata.uns["active_cluster_round"]`.

## Documentation

* [Manual](https://prangelab.org/scOmnom/)
* [API reference](API_REFERENCE.md)
* [Contributing](contributing.md)
* [Changelog](changelog.md)

## License

MIT License.
