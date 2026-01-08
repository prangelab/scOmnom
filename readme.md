# scOmnom

Modular scRNA-seq preprocessing and analysis pipeline built on the scVerse ecosystem.

`scOmnom` is a **CLI-first**, reproducible workflow for large-scale single-cell RNA-seq data, designed for robustness, scalability, and efficient execution on HPC systems.

At its current stage, `scOmnom` provides two stable pipeline stages:

1. **Load & filter** — droplet selection, QC, HVG selection, and doublet detection
2. **Integrate** — multi-method batch correction with optional benchmarking

Clustering and annotation functionality is under active development and not yet considered stable.

---

## Quickstart

Minimal end-to-end example using the recommended input mode (raw counts + CellBender):

```bash
# Activate environment
conda activate scOmnom_env

# Load, QC, and filter
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/load_and_filter/

# Integrate batches
scomnom integrate \
  --input-path results/load_and_filter/adata.merged.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

---

## Installation

`scOmnom` is installed into a dedicated conda environment defined in `environment.yml`.

### 1) Create the conda environment

Before creating the environment, **Linux users with NVIDIA GPUs should inspect `environment.yml` and uncomment the `pytorch-cuda` line**, adjusting the CUDA version if needed to match their driver.

From the repository root:

```bash
conda env create -f environment.yml
conda activate scOmnom_env
```

This installs all required dependencies from `conda-forge`, `bioconda`, and related channels, including the Scanpy and PyTorch ecosystems.

### 2) Install `scOmnom`

Install the package in editable mode:

```bash
pip install -e .
```

This registers the `scomnom` command-line interface and ensures local code changes take effect immediately.

---

## Usage

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

---

## Pipeline overview

```
┌────────────────────────────────────────────────────────────┐
│                        Input data                           │
│                                                            │
│  raw_feature_bc_matrix / filtered_feature_bc_matrix /      │
│  CellBender outputs                                        │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                  load-and-filter                           │
│                                                            │
│  • Identify droplets containing real cells                 │
│    (only if raw-only input)                                │
│  • QC filtering                                            │
│  • HVG selection                                           │
│  • Doublet detection                                       │
│                                                            │
│  → merged AnnData (Zarr) + QC figures                      │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                        integrate                           │
│                                                            │
│  • Batch correction (scVI, BBKNN, Harmony, …)              │
│  • scIB benchmarking                                       │
│                                                            │
│  → integrated AnnData + figures                            │
└────────────────────────────────────────────────────────────┘
```

---

## Input modes (identifying droplets containing real cells)

All input directory arguments point to a **single parent directory** containing **multiple per-sample entries**. Samples are discovered using glob patterns, which can be overridden if needed.

Default patterns used by `load-and-filter`:

- **Raw Cell Ranger matrices**: `*.raw_feature_bc_matrix`
- **Filtered Cell Ranger matrices**: `*.filtered_feature_bc_matrix`
- **CellBender outputs**: `*.cellbender_filtered.output`
  - expected accompanying files:
    - `.cellbender_out_filtered.h5`
    - `.cellbender_out_cell_barcodes.csv`

> **Important:** `scOmnom` only performs *de novo* identification of droplets containing real cells when **only raw counts** are provided. In all other modes, droplets are taken directly from **Cell Ranger** or **CellBender**.

### Preferred mode: raw counts + CellBender

- `--raw-sample-dir` points to a directory containing multiple per-sample folders matching `*.raw_feature_bc_matrix/`
- `--cellbender-dir` points to a directory containing multiple per-sample CellBender outputs matching `*.cellbender_filtered.output*`

Droplets containing real cells are taken from **CellBender**, while raw counts are used for QC comparisons and diagnostics.

### Alternative modes

#### CellBender-corrected counts only

- `--cellbender-dir`

Droplets containing real cells are taken directly from **CellBender**.

#### Cell Ranger filtered matrices

- `--filtered-sample-dir` points to a directory containing multiple per-sample folders matching `*.filtered_feature_bc_matrix/`

Droplets containing real cells are taken from **Cell Ranger**.

#### Raw matrices only (droplet identification by scOmnom)

- `--raw-sample-dir`

`scOmnom` identifies droplets containing real cells using a custom method combining **knee-point detection** and **Gaussian mixture modeling (GMM)**. The inferred droplets are used for downstream QC.

---

## Load and filter (QC + preprocessing)

This stage performs:

- identification of droplets containing real cells **only in raw-only mode**
- quality control filtering
- HVG selection
- doublet detection
- dataset merging across samples

Example (preferred mode):

```bash
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/load_and_filter/
```

Outputs:

- merged AnnData object (Zarr by default)
- QC figures in `figures/`
- `load-and-filter.log`

---

## Integrate (batch correction + benchmarking)

Runs one or more integration methods and optionally benchmarks them.

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.merged.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

Supported methods include: `scVI`, `scANVI`, `scPoli`, `Harmony`, `Scanorama`, `BBKNN`.
Default methods: All but scPoli. (scPoli is used for a second bio-informed integration pass after clustering)
Logs are written to `integrate.log`.

---

## Clustering and annotation (status)

The `cluster-and-annotate` module is **under active development**.

- Its interface and outputs are not yet considered stable
- It is intentionally not documented as part of the standard workflow
- APIs and defaults may change without notice

---

## Running on HPC / SLURM clusters

Example SLURM job scripts are provided:

```text
slurm/
├── scomnom_loadandfilter.job
└── scomnom_integrate.job
```

These scripts are configured for **SURF’s Snellius** compute cluster. Users on other systems must adapt module names, CUDA/driver versions, and conda initialization paths.

For large datasets, running on a **GPU partition is strongly recommended**, particularly for scVI/scANVI-based integration.

### Performance reference (Snellius)

Benchmarks obtained using **1× NVIDIA A100 GPU** and **18 CPU cores**.

On Snellius, this CPU allocation implies a **minimum memory slice of 120 GB**, even though the example job scripts do not explicitly request that amount.

- **~40k cells**
  - `load-and-filter`: < 20 minutes
  - `integrate`: < 30 minutes

- **~400k cells**
  - `load-and-filter`: < 1.5 hours
  - `integrate`: < 7 hours

In practice, the pipeline is likely to run with **substantially less memory (≈ ≤ 60 GB)**, depending on e.g., selected integration methods.

When submitting jobs, ensure `sbatch` is called with appropriate wall time, CPU core reservations, and GPU resources.

The provided SLURM scripts are intended as starting points, not drop-in solutions.

---

## Versioning and changelog

### v0.1.0

- Initial public release
- Stable CLI for:
  - `load-and-filter`
  - `integrate`
- HPC-tested on datasets up to ~400k cells
- Clustering and annotation under active development

---

## License

MIT License.

