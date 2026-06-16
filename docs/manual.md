# scOmnom Manual

This manual contains the full scOmnom workflow reference. For a short project overview and quick start, see [README.md](../README.md).

Modular scRNA-seq preprocessing and analysis pipeline built on the scVerse ecosystem.

`scOmnom` is a **CLI-first**, reproducible workflow for large-scale single-cell RNA-seq data, designed for robustness, scalability, and efficient execution on HPC systems.

At its current stage, `scOmnom` provides a multi-stage pipeline:

1. **Load & filter** — droplet selection, QC, HVG selection, and doublet detection
2. **Integrate** — multi-method batch correction with optional benchmarking
3. **Cluster & annotate** — BISC-guided clustering, CellTypist labels, and decoupler activities
4. **Annotated integration (optional)** — scANVI refinement for clean UMAPs
5. **Markers and DE** — marker discovery, within-cluster DE, and differential abundance

---

## Installation

`scOmnom` is installed into a dedicated environment defined in a platform-specific YAML file.

### 1) Create the environment

Use the YAML that matches your platform:

- Linux/HPC: `environment_linux.yml`
- macOS (Apple Silicon / MPS): `environment_macos.yml`

From the repository root:

```bash
conda env create -f environment_linux.yml
conda activate scOmnom_env
```

On the HPC, the equivalent `micromamba` flow is:

```bash
micromamba create -f environment_linux.yml
micromamba activate scOmnom_env
```

On macOS, create the environment with:

```bash
conda env create -f environment_macos.yml
conda activate scOmnom_env
```

This installs all required dependencies from `conda-forge`, `bioconda`, and related channels, including the Scanpy and PyTorch ecosystems.

### 2) Install `scOmnom`

Install the package in editable mode:

```bash
pip install -e .
```

This registers the `scomnom` command-line interface and ensures local code changes take effect immediately.

### Optional: run with a memory guard on macOS

For large local runs on macOS, you can wrap `scomnom` with [`scripts/run_with_mem_limit.py`](/Users/k.h.prange/Library/CloudStorage/OneDrive-AmsterdamUMC/Documenten/Tech/scOmnom/scripts/run_with_mem_limit.py) to kill the run if either the combined process-tree RSS or macOS system pressure indicators grow too far.

```bash
python scripts/run_with_mem_limit.py \
  --rss-limit-gb 28 \
  --compressed-limit-gb 12 \
  --compressed-delta-limit-gb 4 \
  --swap-used-limit-gb 8 \
  --swap-used-delta-limit-gb 3 \
  --pressure-consecutive-breaches 2 \
  --poll-seconds 5 \
  -- scomnom load-and-filter -c data/cellbender -r data/raw -o results -m metadata.tsv
```

The RSS threshold is summed across the parent process plus worker subprocesses. On macOS, the wrapper also samples system compressed memory and swap usage, records a baseline at startup, and prints both current values and their deltas. Swap is treated as the primary kill signal; compressed-memory thresholds are only allowed to trigger when swap is also in use.

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
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                  cluster-and-annotate                      │
│                                                            │
│  • BISC resolution selection                               │
│  • CellTypist annotation                                   │
│  • Decoupler activities + optional compaction              │
│                                                            │
│  → annotated AnnData + figures                             │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│          integrate --annotated-run (optional)              │
│                                                            │
│  • scANVI refinement using final labels                    │
│  • clean UMAPs / latent space                              │
│                                                            │
│  → projected AnnData + figures                             │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                    markers-and-de                          │
│                                                            │
│  • markers (cluster-vs-rest)                               │
│  • de (within-cluster contrasts)                           │
│  • da (composition)                                        │
│                                                            │
│  → tables + reports + updated AnnData                      │
└────────────────────────────────────────────────────────────┘
```

---

## scOmnom AnnData conventions

`scOmnom` stores results in standard AnnData fields with a few consistent conventions so downstream modules can find prior outputs.

### Counts layers

* `counts_raw`: raw (uncorrected) counts
* `counts_cb`: CellBender-corrected counts
* `adata.X`: used only when no preferred counts layer is available and a command explicitly allows fallback

### Integration outputs

* Integrated embeddings are stored in `adata.obsm` (for example `X_integrated` or method-specific embeddings).
* The selected best embedding is recorded in `adata.uns["integration"]["best_embedding"]` when available.

### Clustering rounds

* Each clustering run creates a new round in `adata.uns["cluster_rounds"]`.
* The currently active round id is stored in `adata.uns["active_cluster_round"]`.
* Each round stores its `cluster_key` and, when available, a `pretty_cluster_key` that points to human-readable labels.

### Annotations and activities

* CellTypist outputs and pretty labels are stored in `adata.obs` and referenced from `adata.uns["cluster_and_annotate"]`.
* Decoupler activities are stored in `adata.uns` under keys such as `msigdb`, `progeny`, and `dorothea`.

### Where to look

* `adata.uns["integration"]`: integration metadata, including the selected best embedding (if available)
* `adata.uns["cluster_rounds"]`: all clustering rounds and their settings
* `adata.uns["active_cluster_round"]`: the currently active clustering round id
* `adata.uns["cluster_and_annotate"]`: pointers to CellTypist and pretty label keys
* `adata.uns["markers_and_de"]`: provenance for markers, DE, and DA runs
* `adata.uns["scomnom_de"]`: detailed DE tables and summaries (pseudobulk and cell-level contrasts)

---

## Using scOmnom AnnData in notebooks/scripts

For manual analysis in notebooks or scripts, always use `load_dataset()` and `save_dataset()` from `src/scomnom/io_utils.py`. This ensures consistent handling of Zarr/H5AD, metadata, and safety checks.

For large Zarr saves with heavy `adata.uns` payloads, `save_dataset()` now stores heavy payloads via sidecar serialization under `__scomnom_payloads__/v1` inside the same store, which reduces save-time memory spikes while keeping `load_dataset()` round-trip behavior.

See also: [`API_REFERENCE.md`](API_REFERENCE.md) for the current Python API namespaces (`scomnom.plotting`, `scomnom.adata_ops`).

```python
from pathlib import Path
from scomnom.io_utils import load_dataset, save_dataset

adata = load_dataset("results/integrate/adata.integrated.zarr")

# ... custom analysis ...

out_path = Path("results/custom/adata.custom.zarr")
save_dataset(adata, out_path, fmt="zarr")
```

Notes:

* Avoid calling `anndata.read_*` or `adata.write_*` directly in new code.
* Use `fmt="h5ad"` only when needed (it loads the full matrix into RAM).

H5AD example (only if you need a single-file artifact and have enough RAM):

```python
out_path = Path("results/custom/adata.custom.h5ad")
save_dataset(adata, out_path, fmt="h5ad")
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

### Quick entry point

Example (preferred mode, raw + CellBender):

```bash
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/load_and_filter/
```

Default lower-count QC behavior:

* `min_genes` remains a fixed threshold.
* `total_counts` is filtered per sample using a lower auto cutoff derived as the stricter of:
  * `--min-counts-mad 5.0`
  * `--min-counts-quantile 0.05`
* The automatic lower-count filter only activates for samples whose:
  * `--min-counts-auto-activate-quantile 0.01`
  * falls below `--min-counts-auto-activate-below 1000`
* `--min-counts` remains available as an additional fixed floor on top of the automatic cutoff.
* The intended default policy is:
  * fixed `min_counts` stays off
  * automatic lower-count filtering stays on
  * defaults clean up obvious low-count noise in ordinary datasets
  * stronger fixed intervention is available when a dataset clearly needs it
* Use `--min-counts-mad none`, `--min-counts-quantile none`,
  `--min-counts-auto-activate-quantile none`, and/or
  `--min-counts-auto-activate-below none` to disable components.

Examples:

```bash
# keep defaults
scomnom load-and-filter ...

# add an extra fixed floor
scomnom load-and-filter ... --min-counts 1000

# disable the quantile component
scomnom load-and-filter ... --min-counts-quantile none

# disable all automatic lower-count filtering
scomnom load-and-filter ... \
  --min-counts-mad none \
  --min-counts-quantile none

# keep auto cutoff settings but disable the activation gate
scomnom load-and-filter ... \
  --min-counts-auto-activate-quantile none \
  --min-counts-auto-activate-below none
```

### What it does

* Discovers samples from the provided parent directories (glob patterns can be overridden).
* Resolves droplets from CellBender or Cell Ranger, or infers droplets in raw-only mode.
* Runs QC filters, including fixed `min_genes`, per-sample lower-count filtering on `total_counts`, upper-tail trimming, HVG selection, and doublet detection.
* Merges per-sample AnnData into a single dataset with consistent metadata.

### Outputs

* merged AnnData object (`.zarr`, optionally `.h5ad`)
* QC figures under `figures/`
* `load-and-filter.log`

---

## Integrate (batch correction + benchmarking)

The `integrate` module performs batch correction across samples while preserving biological signal. It supports multiple integration algorithms, optional benchmarking with `scIB`, and an advanced supervision strategy for `scANVI`.

By default, integration is **non-destructive**: all embeddings, metrics, and selections are stored in the output `AnnData` object.

---

### Quick entry point

A typical integration run starts from the output of `load-and-filter`:

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

### What it does

* recompute PCA on HVGs
* run multiple integration methods
* benchmark them using `scIB`
* select a best embedding
* compute UMAPs
* write figures and metrics to disk
* save an integrated `AnnData` object (Zarr by default)

Logs are written to `integrate.log`.

---

### Supported integration methods

Supported methods include:

* **scVI / scANVI** — variational autoencoders (recommended for large or complex datasets)
* **Harmony** — fast linear batch correction
* **Scanorama** — panoramic batch integration
* **BBKNN** — graph-based batch correction (baseline)

**Default methods:** all of the above.

Methods can be restricted explicitly:

```bash
scomnom integrate \
  --input-path adata.filtered.zarr \
  --batch-key sample_id \
  --methods scVI scANVI Harmony \
  --output-dir results/integrate/
```

---

### Intelligent scANVI supervision (BISC)

When `scANVI` is enabled, `scOmnom` can generate supervision labels automatically instead of relying on a single user-chosen clustering.

The supervision step uses **BISC** on the **scVI latent space** with a structural-only sweep:

1. **Latent-space resolution sweep**
   A coarse, limited bandwith Leiden sweep is performed on the scVI latent space.

2. **Structural selection**
   Candidate resolutions are scored using stability, centroid-based silhouette, and tiny-cluster penalties.

3. **Parsimony**
   The lowest resolution near the top score is selected.

This produces labels used **only for scANVI supervision**, which will be replaced by downstream clustering of the final integrated embedding.

---

### scIB benchmarking and truth labels

Integration methods are benchmarked using `scIB`.

By default, benchmarking uses **CellTypist confident cell-level labels**. CellTypist is a supervised cell type classifier that assigns a label and a confidence score to each cell. Cells that do not pass the confidence mask are excluded from benchmarking.

CellTypist is used downstream as well: the same predictions and confidence mask are reused during `cluster-and-annotate` for biologically informed clustering diagnostics and cluster-level annotation. For consistent results, choose an appropriate CellTypist model for your dataset up front.

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

To select a specific model during integration:

```bash
scomnom integrate \
  --celltypist-model Immune_All_Low.pkl \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

You can override the truth labels via:

```bash
--scib-truth-label-key <key>
```

Valid options include:

* `celltypist` (default)
* `leiden`
* `final` / final cluster labels from `cluster-and-annotate`

This choice affects **benchmarking only** and does not influence integration itself.

---

### Secondary integration: annotated refinement (`--annotated-run`)

After clustering and annotation, `integrate` can be run a **second time** in a supervised refinement mode.

This mode is **optional** and intended for producing cleaner embeddings and UMAPs after final biological labels are known.

#### Key properties

* **Not the default mode**
* **Requires manual input and output naming**
* **Runs scANVI only**
* **Does not overwrite previous results**

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

### Outputs

Each integration run produces:

* integrated `AnnData` object (`.zarr`, optionally `.h5ad`)
* scIB metric tables (raw + scaled)
* UMAPs and diagnostics saved under:

```
figures/
├── png/
│   └── integration_roundN/
└── pdf/
    └── integration_roundN/
```

Each run gets its own `integration_round*` subdirectory to keep results reproducible and auditable.

---

### Relationship to `cluster-and-annotate`

The recommended high-level workflow is:

```
load-and-filter
   ↓
integrate
   ↓
cluster-and-annotate
   ↓
markers-and-de markers (recommended)
   ↓
integrate --annotated-run (optional)
   ↓
adata-ops rename (optional, often used)
   ↓
markers-and-de de
   ↓
markers-and-de da
   ↓
markers-and-de enrichment cluster (optional)
```

Optional subset loop after markers and before final rename/DE/DA:

```
parent projected object
   ↓
markers-and-de markers
   ↓
adata-ops subset
   ↓
[subset object(s)] integrate -> cluster-and-annotate -> integrate --annotated-run
   ↓
adata-ops annotation-merge (back into parent)
   ↓
adata-ops rename
```

The secondary integration step is **purely optional** and should be used only when:

* clustering and annotation are finalized
* improved visualization or refined latent structure is desired

---


## Cluster and annotate (BISC + compaction)

The cluster-and-annotate module performs automated clustering, biologically informed resolution selection, cell type annotation, and optional cluster compaction on an integrated dataset.

It is designed to be run after integrate and operates on the selected integrated embedding (typically X_integrated).

At a high level, this module provides:

* automated Leiden resolution selection using BISC
* biologically informed clustering diagnostics
* CellTypist-based cluster annotation (reusing existing CellTypist outputs if present)
* decoupler-based pathway and transcription factor activity analysis
* optional cluster compaction to improve interpretability

---

### Quick entry point

Minimal example showing only the essential inputs:

```bash
scomnom cluster-and-annotate \
  --input-path results/integrate/adata.integrated.zarr \
  --output-dir results/cluster_and_annotate/ \
  --celltypist-model Immune_All_Low.pkl
```

**Notes:**

* The default embedding (`X_integrated`) is used automatically.
* CellTypist is enabled by default and strongly recommended.
* All other parameters have sensible defaults and can be customized if needed.

---

### CellTypist models

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

For model descriptions, training data, and usage guidance, see the official CellTypist documentation:

[https://www.celltypist.org](https://www.celltypist.org)

---

### Reproducibility and non-destructive design

All clustering, annotation, and compaction steps in scOmnom are **non-destructive**:

* intermediate results are preserved
* no clustering step overwrites a previous result
* all decisions (parameters, metrics, diagnostics) are stored in the dataset

This design ensures that clustering results are fully reproducible, auditable, and comparable across runs.

---

### Annotation merge (subset back-merge)

Use `adata-ops annotation-merge` to project refined labels from one or more subset datasets back into the parent dataset.

Typical use case:

* subset a parent object to a lineage of interest
* re-cluster and re-annotate the subset
* merge those refined subset labels back into the original parent round as a new `subset_annotation` layer

By default this creates a new subset-annotation round (for example `r4_subset_annotation`) and keeps earlier rounds intact.

Minimal example:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --output-dir results \
  --annotation-merge-round-name subset_annotation
```

Multiple child subsets can be merged in one run by repeating `--child-path`:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_myeloid.zarr.tar.zst \
  --output-dir results \
  --annotation-merge-round-name subset_annotation
```

Key controls:

* `--round-id`: parent round to merge into (default: active parent round)
* `--child-round-id`: round to read from each child (default: active child round)
* `--child-source-field`: explicit child `obs` column to merge (default: child round pretty labels)
* `--annotation-merge-round-name`: suffix for newly created merge round
* `--update-existing-round` with `--target-round-id`: update an existing subset-annotation round in place instead of creating a new numbered round

In-place update example:

```bash
scomnom adata-ops annotation-merge \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --child-path results/adata.subset_reclustered_tcells.zarr.tar.zst \
  --output-dir results \
  --target-round-id r4_subset_annotation \
  --update-existing-round
```

---

### Full multi-adata merge (`adata-ops merge`)

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

### Manual renaming of cluster labels

If you want to override the automatic cluster labels with curated names, use `adata-ops rename`.

This creates a new clustering round with your manual names. By default it preserves the existing cluster ids
and only replaces the display labels. With `--collapse-same-labels`, clusters renamed to the same target label
are merged and renumbered by size into a new coarser annotation layer. The original labels remain intact and
you can switch between rounds by setting `active_cluster_round`.

The mapping file must be a two-column, tab-delimited text file with no header. The first column is the
cluster code (`Cnn`), the second column is the new label.

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --output-dir results \
  --output-name adata.clustered.annotated.projected.markers.renamed \
  --rename-idents-file rename.tsv \
  --rename-round-name refined_idents
```

Example mapping file (`rename.tsv`):

```
C03	SOX10+ PMP22+ Schwann cells
C07	TREM1hi PLIN2hi iLAMs
```

You can control the new manual label round name with:

```bash
--rename-round-name refined_idents
```

This will create a new round id like `r5_refined_idents`. Default suffix is `manual_rename`

To build a coarser layer such as archetypes, merge identical renamed labels into one new cluster state:

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --output-dir results \
  --rename-idents-file archetypes_rename.tsv \
  --rename-round-name archetypes \
  --collapse-same-labels \
  --no-set-active
```

Use `--set-active/--no-set-active` to control whether the new rename round becomes the default active round.

If you want to revise an existing manual rename round in place rather than creating a new numbered round,
use `--update-existing-round` together with an explicit `--target-round-id`:

```bash
scomnom adata-ops rename \
  --input-path results/adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation.zarr.tar.zst \
  --output-dir results \
  --output-name adata.clustered.annotated.projected.markers.renamed__annotation_merge_r4_subset_annotation \
  --rename-idents-file archetypes_rename.tsv \
  --target-round-id r5_archetypes \
  --update-existing-round \
  --collapse-same-labels \
  --no-set-active
```

Three UMAPs are emitted for the renamed labels:

* one with the full right-side legend
* one with no legend
* one with a right-side legend plus `Cnn` overlaid on clusters

---

### BISC: Biology-Informed Structural Clustering

BISC automates Leiden resolution selection by combining structural, stability, and biological signals.

The goal is to replace manual resolution tuning with a reproducible, data-driven procedure.

---

#### Inputs

* `--embedding-key` (default `X_integrated`) is used for neighbors, UMAP, and clustering.
* `--batch-key` is used for batch diagnostics and plotting (auto-detected if not provided).
* CellTypist labels are optional but enable bio-guided scoring (`--bio-guided`).

#### Outputs

* A new clustering round in `adata.uns["cluster_rounds"]` with the chosen resolution.
* The selected round id is recorded in `adata.uns["active_cluster_round"]`.
* Cluster labels are stored in `adata.obs` under the round’s `cluster_key`.

#### When to tune

* Adjust `res_min`, `res_max`, and `n_resolutions` if the dataset is extremely small or highly heterogeneous.
* Disable bio-guided clustering (`--no-bio-guided`) if no suitable CellTypist model exists.

---

#### Resolution sweep

Leiden clustering is evaluated over a resolution range:

* `res_min`: **0.1**
* `res_max`: **2.5**
* `n_resolutions`: **25**

For each resolution, BISC records:

* number of clusters
* cluster size distribution
* centroid-based silhouette separation
* penalties for excessive cluster fragmentation

A small penalty proportional to cluster count is applied:

* `penalty_alpha`: **0.02**

---

#### Structural stability (ARI)

BISC measures clustering stability using the **Adjusted Rand Index (ARI)** between adjacent resolutions.

* high ARI → clustering structure is stable
* low ARI → clusters split or merge aggressively

Stability is smoothed across neighboring resolutions and used to identify robust regions.

**Key defaults:**

* `stability_threshold`: **0.85**
* `min_plateau_len`: **3**
* `max_cluster_jump_frac`: **0.4**
* `min_cluster_size`: **20**
* `tiny_cluster_size`: **20**

---

#### Biological metrics (CellTypist-guided)

If CellTypist predictions are available, BISC incorporates biological consistency metrics.

This is enabled by default (`bio_guided_clustering = True`).

Only high-confidence cells are considered, using a probability-based mask:

* entropy limit: **≤ 0.5**
* entropy quantile: **0.7**
* minimum margin (top1 − top2): **0.10**

Safety gates:

* **≥ 500 cells**
* **≥ 5% of the dataset**

The biological metrics used are:

#### 1. Biological homogeneity

Mean fraction of the dominant CellTypist label within clusters.

* weight: `w_hom = 0.15`

#### 2. Biological fragmentation

Penalizes clusters that contain multiple large biological subgroups.

* weight: `w_frag = 0.10`

#### 3. Biological ARI

ARI between cluster labels and CellTypist labels (on confident cells).

* weight: `w_bioari = 0.15`

These metrics encourage clusterings that align with known biology while avoiding over-fragmentation.

---

#### Composite score and resolution selection

All metrics are normalized and combined into a single composite score:

* structural stability (`w_stab = 0.50`)
* silhouette separation (`w_sil = 0.35`)
* tiny cluster penalty (`w_tiny = 0.15`)
* optional biological metrics (weights above)

The final resolution is selected from stable plateaus, with a bias toward simpler solutions when scores are comparable.

---

### Cluster annotation

After clustering:

* CellTypist labels are aggregated at the cluster level
* clusters are ordered by descending size
* stable, human-readable labels are generated, e.g.:

```
C00: Hepatocyte
C01: Cholangiocyte
C02: Fibroblast
```

Clusters with insufficient high-confidence cells are labeled **Unknown** to avoid overinterpretation.

---

### Compaction: merging redundant clusters

Large datasets can yield clusters that are transcriptionally distinct but *biologically redundant*. The optional **compaction** step merges such clusters using **multi-view biological agreement** computed from decoupler activities.

Compaction is **CellTypist-grouped**: cluster pairs are only compared *within the same* CellTypist cluster label group (with an option to skip `Unknown` groups).

#### What signals are used

Compaction requires decoupler outputs from:

- **PROGENy** activity (required)
- **DoRothEA** activity (required)
- **MSigDB activity split by GMT** (required by default; can be made optional)

#### Similarity metric

For each pair of clusters within the same CellTypist group:

- activities are **z-scored** (see scope below)
- similarity is computed as **cosine similarity** on the z-scored vectors

For MSigDB, similarity is computed using a **Top-K union** strategy (default **K = 25**):
- take the union of the top-|z| features from both clusters (by absolute z-score)
- compute cosine similarity on that union vector

#### Z-score scope (with guardrail)

Compaction supports two z-scoring modes:

- `global` (recommended): z-score each activity column across **all clusters**
- `within_celltypist_label`: z-score each activity column **within each CellTypist group**

Guardrail: `within_celltypist_label` falls back to `global` if a CellTypist group contains fewer than **4** clusters (to avoid unstable z-scores).

Default in the config:
- `compact_zscore_scope = "global"`

#### Adaptive thresholds (per CellTypist group)

Rather than using fixed similarity thresholds only, compaction computes **adaptive per-group thresholds** based on within-group similarity distributions:

- for each view (PROGENy / DoRothEA / each MSigDB GMT block), compute the **0.90 quantile** of pairwise similarities within the CellTypist group
- then apply a **floor** (minimum strictness)
- then apply a **cap** (maximum strictness) from user-provided thresholds

This yields effective thresholds that can relax when a CellTypist group is intrinsically heterogeneous, but never below conservative floors.

Floors:
- PROGENy floor: **0.70**
- DoRothEA floor: **0.60**
- MSigDB floors:
  - `HALLMARK`: **0.60**
  - `REACTOME`: **0.45**
  - default for other GMTs: **0.50**

Caps (from config defaults; can be overridden):
- `thr_progeny = 0.98`
- `thr_dorothea = 0.98`
- `thr_msigdb_default = 0.98`
- optional per-GMT caps via `thr_msigdb_by_gmt`

#### MSigDB majority rule

MSigDB is evaluated per GMT block, then combined using a majority rule:

- if there is **1** GMT: require **1/1**
- if there are **2** GMTs: require **1/2**
- if there are **≥ 3** GMTs: require `ceil(0.67 × n_gmts)` passing GMTs

Default MSigDB majority fraction: **0.67**.

If `msigdb_required = False`, MSigDB is not required for passing (but similarities are still computed if available).

#### Pass condition and grouping

An edge (merge suggestion) between two clusters is created only if all required views pass:

- PROGENy similarity ≥ effective PROGENy threshold **AND**
- DoRothEA similarity ≥ effective DoRothEA threshold **AND**
- MSigDB majority rule passes (if required)

Edges are then converted into merge groups using one of:

- `connected_components` (default): merge connected components of the pass graph
- `clique`: greedy clique cover on the pass graph (stricter grouping)

Default in the config:
- `compact_grouping = "connected_components"`

#### Size and label filters

- `compact_min_cells` (default `0`): clusters smaller than this are excluded from compaction decisions
- `compact_skip_unknown_celltypist_groups` (default `False`): optionally exclude CellTypist groups labeled `Unknown`/`UNKNOWN`

#### Compaction outputs

Compaction produces:

* a new clustering stored as a new, non-destructive result
* a full **pairwise audit table** of edges with per-view similarities, adaptive thresholds (and which scope was effectively used), and MSigDB majority statistics
* a **decision log** listing multi-cluster merge groups and their rationale
* a mapping from original cluster ids to compacted ids (e.g. `C00`, `C01`, …), sorted by total component size

---

### Decoupler configuration

Decoupler is enabled by default (`run_decoupler = True`) and is used for:

* cluster-level pseudobulk activity inference
* biological diagnostics
* compaction decisions

**Defaults:**

* aggregation: **mean**
* method: **consensus** (`ulm`, `mlm`, `wsum`)
* databases **MSigDB, DoRothEA, PROGENy**: (`HALLMARK`, `REACTOME`)
* minimum targets per source: **5**

All decoupler settings — including databases, methods, thresholds, and organisms — are fully configurable.

For details on decoupler methods and resources, see:

[https://decoupler-py.readthedocs.io](https://decoupler-py.readthedocs.io)

---

### Outputs

* clustered/annotated AnnData (`.zarr`, optionally `.h5ad`) in the output directory
* figures under `figures/cluster_and_annotate/` (organized by round)
* `cluster-and-annotate.log` under `logs/`
* optional per-cluster annotation CSV (only if `--annotation-csv` is provided)

### Reporting

All modules generate a self-contained HTML report including:

* Run information
* All generated plots

The report is written to:

```
figures/<MODULE>/<MODULE>_report.html
```

---

## Markers and DE (markers-and-de)

The `markers-and-de` module covers four related analyses:

* **Markers**: cluster-vs-rest marker discovery
* **DE**: within-cluster differential expression across condition levels
* **DA**: differential abundance (compositional shifts of cell types)
* **Enrichment**: pathway and TF activity scoring either from a clustering round or from exported DE result tables

All four subcommands are CLI-first and store results in tables plus a self-contained report.

### Quick entry points

```bash
# Marker discovery (cluster-vs-rest)
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.projected.zarr

# Within-cluster DE (condition contrasts)
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --condition-key condition

# Differential abundance (composition)
scomnom markers-and-de da \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr \
  --condition-key condition

# Round-native enrichment
scomnom markers-and-de enrichment cluster \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --round-id r4_subset_annotation \
  --condition-key sex \
  --gene-filter "not gene.str.startswith('MT-')"

# DE-table enrichment
scomnom markers-and-de enrichment de \
  --input-dir results/tables/de_r5_archetypes_round1 \
  --gene-filter "not gene.str.startswith('MT-')"

# Module scoring from user-defined signatures
scomnom markers-and-de enrichment module-score \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --round-id r5_archetypes \
  --module-file signatures.gmt \
  --module-set-name curated_programs
```

---

### Markers (cluster-vs-rest)

Markers are computed per cluster against all other cells. The module supports **cell-level markers**, **pseudobulk markers**, or **both** (`--run cell|pseudobulk|both`).

**Cell-level markers**

* Uses Scanpy `rank_genes_groups` with `wilcoxon`, `t-test`, or `logreg`.
* Optional per-cluster downsampling for very large datasets.
* Filters by `min_pct` and `min_diff_pct` to remove low-coverage or weakly differential genes.

**Pseudobulk markers**

* Aggregates counts by sample and cluster, then runs DESeq2 (via PyDESeq2).
* Requires at least **6 unique samples** (guard to avoid unstable estimates).
* Uses count layers in priority order: `counts_cb`, `counts_raw`, then `adata.X` (if allowed).
* Supports sample-level covariates (`--pb-covariates`) and filters on minimum cells per sample-cluster.

**Outputs**

* Tables: `tables/marker_tables_<run>/cell_based/` and `tables/marker_tables_<run>/pseudobulk_based/`
* Reports: `figures/markers/markers_report.html` (plus PDFs/PNGs under `figures/markers/`)

---

### LIANA CCC

`scomnom markers-and-de ccc liana` runs LIANA-based ligand-receptor analysis on an annotated AnnData object and writes tables, figures, and an `adata.uns["markers_and_de"]["ccc"]["liana"]` payload.

If `--dataset-key` is omitted, scOmnom runs the normal within-object LIANA workflow. Cross-tissue / cross-dataset filtering is only activated when `--dataset-key` is provided together with `--source-level` and `--target-level`.

**Assay selection**

For count-based CCC input, scOmnom prefers assays in this order:

* `counts_cb`
* `counts_raw`
* `adata.X` as a final fallback

Use `--use-raw` only when you explicitly want `adata.raw`.

Use `--input-mode lognorm` when you want LIANA to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` instead of running directly on raw counts. This creates `adata.layers["lognorm_counts_cb"]` or `adata.layers["lognorm_counts_raw"]` on demand and then uses that layer for LIANA.

**Condition key syntax**

LIANA CCC supports the same round-aware condition syntax used elsewhere in the pipeline:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | No condition split | One LIANA run on the full object |
| `A` | Single obs key | One LIANA run per level of `A` |
| `A:B` | Composite key | One LIANA run per joint level of `A` and `B` |
| `A@B` | A within B | One LIANA run per level of `B`, with `A` treated as the condition label inside each subset |

**Cross-tissue mode**

For merged multi-tissue objects, LIANA CCC can be restricted to cross-dataset signaling candidates by specifying:

* `--dataset-key`: obs column defining tissue / dataset origin
* `--source-level`: allowed sender tissue / dataset level(s)
* `--target-level`: allowed receiver tissue / dataset level(s)
* `--signal-scope`: `all` or `secreted`

In this mode, scOmnom still runs LIANA on the merged AnnData object, but filters the returned interactions to the requested source and target tissue levels. With `--signal-scope secreted`, scOmnom further restricts the retained interactions to CellChatDB routes annotated as `Secreted Signaling`.

**CellChatDB-backed pathway families**

The LIANA family / route plots use a vendored CellChatDB annotation table at [src/scomnom/resources/cellchatdb_interaction_annotations.tsv](/Users/k.h.prange/Library/CloudStorage/OneDrive-AmsterdamUMC/Documenten/Tech/scOmnom/src/scomnom/resources/cellchatdb_interaction_annotations.tsv:1). This table was exported one time from the official `jinworks/CellChat` package source at version `2.2.0.9001`.

scOmnom first tries to map each LIANA `ligand_complex` / `receptor_complex` pair onto CellChatDB and uses CellChat `pathway_name` values such as `BMP`, `TGFb`, `NOTCH`, or `CXCL` as the route family. If no exact CellChatDB match is found, scOmnom falls back to its internal heuristic family classifier.

The vendored export preserves CellChat's own row-level `version` field, which currently contains both:

* `CellChatDB v1`
* `CellChatDB v2`

**Example**

```bash
scomnom markers-and-de ccc liana \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr.tar.zst \
  --condition-key masld_status@timepoint
```

```bash
scomnom markers-and-de ccc liana \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --dataset-key tissue \
  --source-level liver \
  --target-level fat \
  --signal-scope secreted
```

**Outputs**

* Figures: `figures/<fmt>/ccc_liana_<round>_roundN/`
* Tables: `tables/ccc_liana_<round>_roundN/`
* Route-family summary: `route_family_summary.tsv`

**Focused donor-level rescoring**

`scomnom markers-and-de ccc liana-paired` rescales a focused LIANA candidate table at donor or sample level instead of rerunning pooled LIANA discovery. It is meant for downstream paired analyses once pooled LIANA has already identified candidate ligand-receptor edges.

**Current scope**

* reads a candidate LR table such as `liana_rank_aggregate.tsv`
* supports the same `A` and `A@B` condition-subset syntax as pooled LIANA
* supports cross-tissue sender/receiver restriction via `--dataset-key`, `--source-level`, and `--target-level`
* supports the same LIANA expression input modes as pooled LIANA:
  * `--input-mode counts`
  * `--input-mode lognorm`
* recomputes donor/sample-level ligand-receptor scores from cluster-level ligand and receptor complex expression using:
  * `sqrt(ligand_expr * receptor_expr)`
* aggregates those donor-level edge scores to route-family summaries and within-condition group-effect tables
* paired rescoring defaults to a donor-sparser threshold set than pooled discovery:
  * `--min-sender-cells 5`
  * `--min-receiver-cells 5`
  * `--min-scored-donors-per-group 3`
* always writes missingness summary tables so donor-level sparsity is visible before interpreting empty effect tables
* optional focused figures for:
  * paired route-family effect dotplots
  * paired ligand-receptor edge effect strip plots
* writes focused donor-level tables and stores them under `adata.uns["markers_and_de"]["ccc"]["liana_paired_rescore"]`

**Example**

```bash
scomnom markers-and-de ccc liana-paired \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --candidate-events results/tables/ccc_liana_r5_round1/condition__sex_at_MASLD/MASLD=yes/female/liana_rank_aggregate.tsv \
  --dataset-key tissue \
  --source-level vfat \
  --target-level liv \
  --pairing-key sample_id \
  --condition-key sex@MASLD \
  --condition-value yes \
  --compare-level female \
  --compare-level male \
  --input-mode lognorm
```

**Outputs**

* Figures: `figures/<fmt>/ccc_liana_<round>_roundN/paired_rescore/<condition>/`
* Tables: `tables/ccc_liana_<round>_roundN/paired_rescore/<condition>/`
* Key tables:
  * `liana_paired_lr_edge_scores.tsv`
  * `liana_paired_route_scores.tsv`
  * `liana_paired_lr_edge_missingness.tsv`
  * `liana_paired_route_missingness.tsv`
  * `liana_paired_lr_edge_effects.tsv`
  * `liana_paired_route_effects.tsv`
  * `liana_paired_settings.tsv`

---

### NicheNet CCC

`scomnom markers-and-de ccc nichenet` runs a sender-focused NicheNet analysis for one receiver cluster or, by default, all receiver clusters in batch mode. It is intended to complement LIANA rather than replace it: LIANA gives candidate sender-receiver structure, whereas NicheNet prioritizes ligands that best explain a receiver transcriptional program.

**Current scope**

The first `scOmnom` NicheNet implementation is intentionally narrow:

* receiver selection via `--receiver-cluster`
  * `--receiver-cluster all` is the default and batches over every cluster
  * any specific raw cluster id or pretty label runs only that receiver
* receiver gene set from either:
  * `--gene-list-file`, or
  * receiver-cluster DE between exactly two `--compare-level` values
* optional cross-tissue sender/receiver restriction via:
  * `--dataset-key`
  * `--source-level`
  * `--target-level`
* optional expression input mode for sender/receiver expressed-gene filtering:
  * `--input-mode counts`
  * `--input-mode lognorm`

If `--dataset-key` is omitted, scOmnom runs the normal within-object sender-focused NicheNet workflow.

Use `--input-mode lognorm` when you want NicheNet to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` before computing sender and receiver expressed-gene sets. This follows the same normalization convention used by LIANA and MEBOCOST.

**Condition key syntax**

NicheNet currently supports:

* `A`: compare two levels of `A` within the full object
* `A@B`: compare two levels of `A` separately within each level of `B`

**Runtime requirements**

This backend shells out to `Rscript` and expects the R package `nichenetr` to be installed. In the current `v1` implementation, scOmnom uses the official human NicheNet model files downloaded from the NicheNet Zenodo-hosted URLs at runtime, so internet access is required the first time the analysis is run.

The environment YAML files include the required R runtime and helper packages, but `nichenetr` itself is currently not pinned there as a Conda package because the available Conda builds are stale and not consistently cross-platform.

scOmnom now expects `nichenetr` in a local non-synced cache library. On macOS the default is `~/Library/Caches/scOmnom/r-libs/nichenet/`; on Linux it is `~/.cache/scOmnom/r-libs/nichenet/`. If it is missing, `ccc nichenet` errors with an explicit hint and suggests `--install-missing-r-deps`. If you want scOmnom to bootstrap that library automatically, run:

```bash
scomnom markers-and-de ccc nichenet \
  ... \
  --install-missing-r-deps
```

To install it manually into the same cache library, run:

```bash
tmpdir=$(mktemp -d) && \
R_LIBS_USER="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
R_LIBS="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
/Users/k.h.prange/miniconda3/envs/scOmnom_env/bin/Rscript -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE, showWarnings=FALSE); .libPaths(c(Sys.getenv("R_LIBS_USER"), .Library, .Library.site)); pkgs <- c("DiceKriging", "emoa", "fdrtool", "mlrMBO"); missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing_pkgs)) install.packages(missing_pkgs, repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))' && \
git clone --depth 1 https://github.com/saeyslab/nichenetr.git "$tmpdir/nichenetr" && \
mkdir -p "$HOME/Library/Caches/scOmnom/r-libs/nichenet" && \
R_LIBS_USER="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
R_LIBS="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
/Users/k.h.prange/miniconda3/envs/scOmnom_env/bin/R CMD INSTALL --no-test-load -l "$HOME/Library/Caches/scOmnom/r-libs/nichenet" "$tmpdir/nichenetr"
```

**Example**

```bash
scomnom markers-and-de ccc nichenet \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --dataset-key tissue \
  --source-level liver \
  --target-level fat \
  --condition-key masld_status@timepoint \
  --condition-value 5_years_post \
  --compare-level better \
  --compare-level worse
```

**Outputs**

* Figures: `figures/<fmt>/ccc_nichenet_<round>_roundN/`
* Tables: `tables/ccc_nichenet_<round>_roundN/`
* Key tables:
  * `nichenet_ligand_activity.tsv`
  * `nichenet_ligand_target_links.tsv`
  * `nichenet_ligand_receptor_links.tsv`
  * `nichenet_receiver_de.tsv`

---

### MEBOCOST CCC

`scomnom markers-and-de ccc mebocost` runs MEBOCOST metabolite-mediated cell-cell communication inference on an annotated AnnData object and writes tables, figures, and an `adata.uns["markers_and_de"]["ccc"]["mebocost"]` payload.

**Current scope**

The first `scOmnom` MEBOCOST implementation is intentionally focused on the core communication table workflow:

* per-run MEBOCOST communication inference from a single AnnData subset
* optional `A` and `A@B` condition expansion, aligned with the LIANA syntax
* optional cross-tissue sender/receiver restriction via:
  * `--dataset-key`
  * `--source-level`
  * `--target-level`
* optional MEBOCOST expression input mode via:
  * `--input-mode counts`
  * `--input-mode lognorm`
* summary outputs and baseline plots for:
  * top metabolite-sensor events
  * metabolite super-class and sensor-annotation summary tables/barplots
  * source-target event-count heatmaps
  * source-target mean-score heatmaps
  * per-source and per-target cluster breakdown barplots
  * per-source and per-target metabolite super-class / sensor-annotation summaries
  * condition-split compare plots for source-target heatmaps, top events, and annotation classes when `--compare-level` defines a multi-run bucket
* saved communication tables are enriched with HMDB metabolite classes and MEBOCOST sensor annotations when available

If `--dataset-key` is omitted, scOmnom runs the normal within-object MEBOCOST workflow.

Use `--input-mode lognorm` when you want MEBOCOST to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` instead of running directly on raw counts. This creates `adata.layers["lognorm_counts_cb"]` or `adata.layers["lognorm_counts_raw"]` on demand and then uses that layer for MEBOCOST.

**Runtime requirements**

This backend uses the Python package `MEBOCOST` from the Chen Lab GitHub repository. The upstream Python package does not bundle the metabolite database/config files that the constructor expects, so scOmnom also maintains a local MEBOCOST resource cache and generated `mebocost.conf` under the user cache directory.

If either the Python package or the upstream resource bundle is missing, `ccc mebocost` errors with an explicit install hint and suggests `--install-missing-python-deps`.

To let scOmnom bootstrap it automatically, run:

```bash
scomnom markers-and-de ccc mebocost \
  ... \
  --install-missing-python-deps
```

**Example**

```bash
scomnom markers-and-de ccc mebocost \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --dataset-key tissue \
  --source-level liver \
  --target-level fat \
  --condition-key masld_status@timepoint \
  --condition-value 5_years_post \
  --compare-level better \
  --compare-level worse
```

**Outputs**

* Figures: `figures/<fmt>/ccc_mebocost_<round>_roundN/`
* Tables: `tables/ccc_mebocost_<round>_roundN/`
* Key tables:
  * `mebocost_commu_res.tsv`
  * `mebocost_sig_res.tsv`
  * `mebocost_source_target_summary.tsv`

**Focused donor-level rescoring**

`scomnom markers-and-de ccc mebocost-paired` rescales a focused candidate event table at donor or sample level instead of rerunning full pooled discovery. It is meant for downstream paired analyses once pooled MEBOCOST has already identified candidate metabolite-sensor routes.

**Current scope**

* reads a candidate event file such as `mebocost_sig_res.tsv`
* supports the same `A` and `A@B` condition-subset syntax as pooled MEBOCOST
* supports cross-tissue sender/receiver restriction via `--dataset-key`, `--source-level`, and `--target-level`
* computes donor/sample-level event scores for:
  * `--score-method mebocost-metabolite-sensor`
  * `--score-method associated-gene-proxy`
* paired rescoring defaults to:
  * `--min-sender-cells 5`
  * `--min-receiver-cells 5`
  * `--min-scored-donors-per-group 3`
* always writes missingness summary tables so all-missing donor splits are easy to diagnose
* optional focused figures for:
  * paired route summaries
  * sample-by-route heatmaps
  * top condition effects from the paired group-effect table
* writes focused donor-level tables and stores them under `adata.uns["markers_and_de"]["ccc"]["mebocost_paired_rescore"]`

**Example**

```bash
scomnom markers-and-de ccc mebocost-paired \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --candidate-events results/tables/ccc_mebocost_r5_round1/condition__masld_status_at_timepoint/timepoint=5_years_post/worse/mebocost_sig_res.tsv \
  --dataset-key tissue \
  --source-level vfat \
  --target-level liv \
  --pairing-key sample_id \
  --condition-key sex@MASLD \
  --condition-value yes \
  --input-mode lognorm
```

**Outputs**

* Figures: `figures/<fmt>/ccc_mebocost_<round>_roundN/paired_rescore/<condition>/`
* Tables: `tables/ccc_mebocost_<round>_roundN/paired_rescore/<condition>/`
* Key tables:
  * `mebocost_paired_event_scores.tsv`
  * `mebocost_paired_route_scores.tsv`
  * `mebocost_paired_event_missingness.tsv`
  * `mebocost_paired_route_missingness.tsv`
  * `mebocost_paired_group_effects.tsv`
  * `mebocost_paired_settings.tsv`

---

### Enrichment

The enrichment submodule has three entry points:

* `scomnom markers-and-de enrichment cluster`: run decoupler directly on an existing clustering round
* `scomnom markers-and-de enrichment de`: run decoupler from exported DE result tables without loading AnnData
* `scomnom markers-and-de enrichment module-score`: score user-defined gene programs per cell and summarize them by cluster or cluster-condition

The first two modes support MSigDB, custom `.gmt` files, PROGENy, and DoRothEA. The third mode uses user-defined module definitions with per-cell module scoring.

#### Enrichment cluster (round-native pathway / TF activity)

This mode uses round-native pseudobulk expression built from the selected round labels, then computes pathway / TF activity using MSigDB, PROGENy, and DoRothEA.

**Round selection**

* `--round-id` selects which round to score.
* If omitted, the active round is used.
* Enrichment is stored back into that round; it does not create a new round.

**Supported resources**

* `MSigDB` from built-in keywords or custom `.gmt` files via `--msigdb-gene-sets`
* `PROGENy`
* `DoRothEA`

These use the same decoupler method settings as the clustering and DE workflows.

**Condition key syntax (enrichment cluster)**

Enrichment supports either cluster-only pseudobulk or cluster-by-condition pseudobulk:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One enrichment profile per cluster |
| `A` | Single obs key | One enrichment profile per `cluster × A level` |
| `A:B` | Composite key | One enrichment profile per `cluster × all combinations of A and B` |

Examples:

* `--round-id r5_archetypes`  
  Enrichment per archetype cluster.
* `--round-id r5_archetypes --condition-key sex`  
  Enrichment per `cluster × sex`.
* `--round-id r5_archetypes --condition-key sex:masld_status`  
  Enrichment per `cluster × sex × masld_status` combination.

**Gene filtering**

* `--gene-filter` filters genes before enrichment is computed, using pandas-query expressions against `adata.var`.
* Filters are combined with logical AND.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* This changes the enrichment input universe, unlike `--plot-gene-filter`, which is plot-only in DE.

Example:

* `--gene-filter "not gene.str.startswith('MT-')"`
* `--gene-filter "not gene.str.startswith('RPL')"`
* `--gene-filter "not gene.str.startswith('RPS')"`

**Outputs**

* Figures: `figures/<fmt>/enrichment_<round>_roundN/`
* Report: `figures/<fmt>/enrichment_<round>_roundN/enrichment_report.html`
* Default AnnData output name: `adata.enrichment_<round>.zarr.tar.zst`
* Stored round payload includes the selected condition key, applied gene filters, and retained gene counts for provenance

#### Enrichment de (from exported DE tables)

This mode reads exported DE CSV tables from a DE results folder and runs the same MSigDB / PROGENy / DoRothEA decoupler backends on the DE statistics. For MSigDB it can also run a Python preranked GSEA pass and build a joint concordance summary between decoupler activity direction and GSEA direction. It reuses the signed up/down barplot layout from the existing DE decoupler workflow.

**Inputs**

* `--input-dir` should point at a DE tables folder such as `results/tables/de_r5_archetypes_round1`
* No AnnData input is required

**Supported DE sources**

* `--de-decoupler-source auto` uses any available pseudobulk and cell-level DE tables
* `--de-decoupler-source pseudobulk` limits enrichment to exported pseudobulk DE tables
* `--de-decoupler-source cell` limits enrichment to exported cell-level DE tables

**Gene filtering**

* `--gene-filter` is applied before enrichment on the DE-derived gene statistics
* Filters are evaluated against the gene metadata present in the exported DE tables, including `gene` and exported columns such as `gene_type`

**Outputs**

* Figures: `figures/<fmt>/enrichment_de_<inputdir>_roundN/`
* Tables: `tables/enrichment_de_<inputdir>_roundN/`
* Report: `figures/<fmt>/enrichment_de_<inputdir>_roundN/enrichment_de_report.html`
* MSigDB outputs can include decoupler activity plots, compact GSEA summaries, and concordance-only joint summaries when `run_gsea` is enabled

#### Enrichment module-score (user-defined signatures)

This mode scores custom gene modules per cell, then summarizes those scores by cluster or by `cluster × condition`.

**Backend**

* `--module-score-method scanpy` uses `scanpy.tl.score_genes`
* `--module-score-method aucell` uses `decoupler.mt.aucell` for rank-based AUCell scoring

**Module inputs**

* `--module-file` is repeatable
* supported formats:
  * `.gmt`: one or more modules per file
  * `.tsv` / `.csv`: either `module,gene` style two-column tables or a single-column gene list
  * `.txt`: one gene per line, interpreted as a single module named after the file stem
* `--module-set-name` gives a stable name to the scored module collection and is used in output naming

**Condition key syntax**

Module scoring uses the same grouped-state syntax as `enrichment cluster`:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One summary profile per cluster |
| `A` | Single obs key | One summary profile per `cluster × A level` |
| `A:B` | Composite key | One summary profile per `cluster × all combinations of A and B` |

**Outputs**

* Figures: `figures/<fmt>/module_score_<set>_<round>_roundN/`
* Tables: `tables/module_score_<set>_<round>_roundN/`
* Report: `figures/<fmt>/module_score_<set>_<round>_roundN/module_score_report.html`
* Default AnnData output name: `adata.module_score_<set>_<round>.zarr.tar.zst`
* The selected round stores summary tables and score metadata under `adata.uns["cluster_rounds"][round_id]["module_scores"]`
* Per-cell score columns are written to round-specific `adata.obs` columns as `module_score__<round>__<set>__<module>`

---

### DE (within-cluster contrasts)

Within-cluster DE compares **condition levels inside each cluster**, e.g. `treated vs control` within each cell type. You can provide a single `--condition-key` or multiple `--condition-keys`, plus optional explicit contrasts (`--contrasts`).

**Condition key syntax**

| Syntax | Meaning | Resulting behavior                                                           |
| --- | --- |------------------------------------------------------------------------------|
| `A` | Single obs key | Compare levels of `A` within each cluster                                    |
| `A:B` | Composite key | Create a combined key across **all combinations** of `A` and `B` levels      |
| `A@B` | Within‑B | Run **A within each level of B** (usually what you want)                     |
| `A^B` | Interaction | Run interaction contrasts between A and B (requires a pseudobulk DESeq2 run) |

**Examples**

* `--condition-keys treatment`  
  Compare treatment levels within each cluster.
* `--condition-keys MASLD:sex`  
  Full combinations of MASLD × sex.
* `--condition-keys treatment@sex`  
  Treatment contrasts within each sex.
* `--condition-keys treatment^sex`  
  Treatment‑by‑sex interaction contrasts.

**Cell-level within-cluster DE**

* Runs contrast-conditional markers per cluster using `wilcoxon` and `logreg` (configurable).
* Uses `min_pct` and `min_diff_pct` to focus on robust signals.
* Does not require the pseudobulk sample guard.
* Typically much faster than pseudobulk, which makes it practical for iterative test runs.

**Pseudobulk within-cluster DE**

* Aggregates counts by sample and cluster, then runs DESeq2 per cluster and contrast.
* Requires at least **6 unique samples** (guarded).
* Supports sample-level covariates and minimum cells per sample-cluster.
* Usually substantially slower than cell-level mode, but more rigorous for replicate-aware inference.

**Target populations**

* `--target-groups` restricts DE to selected cluster/population labels from `--group-key` (repeatable and comma-separated).
* Applies to both cell-level and pseudobulk phases, so unrelated populations are skipped in computation and outputs.

**Gene filtering**

* `--gene-filter` filters genes before DE is calculated.
* Filters are applied to both the cell-level and pseudobulk DE inputs.
* Filters are combined with logical AND and evaluated against `adata.var`.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* `--plot-gene-filter` remains plot-only and does not change the DE statistics.

**DE → pathway/TF activity (decoupler)**

* Optional activity inference from DE statistics for each contrast.
* Sources can be `cell`, `pseudobulk`, `all`, or `auto` (prefer pseudobulk if present).
* Supports MSigDB, PROGENy, and DoRothEA with configurable methods and target filters.
* MSigDB GSEA is enabled by default for `markers-and-de de` and writes additional `msigdb_gsea` and `msigdb_joint` outputs alongside the decoupler summaries.

**GSEA controls for `markers-and-de de`**

* `--run-gsea/--no-run-gsea` enables or disables MSigDB preranked GSEA on the DE statistic matrix.
* `--gsea-min-size`, `--gsea-max-size`, and `--gsea-eps` control pathway-size filtering and numerical tolerance for the Python `gseapy` backend.
* `--gsea-rank-col` lets you override the DE column used for GSEA ranking. If omitted, scOmnom uses the same statistic requested through `--de-decoupler-stat-col`.
* `--joint-enrichment-alpha` controls the adjusted-significance threshold used when building the `msigdb_joint` concordance summary.
* `--joint-enrichment-top-n` controls how many concordant up/down pathways are shown in the compact GSEA and joint plots.
* `--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant` toggles whether joint summaries require matching decoupler and GSEA direction.
* `--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig` toggles whether joint summaries require GSEA adjusted significance.
* `--joint-enrichment-leading-edge-top-n` controls how many leading-edge genes are previewed in compact GSEA summaries.

**Outputs**

* Tables: `tables/de_<round>_roundN/cell_based/` and `tables/de_<round>_roundN/pseudobulk_based/`

**HPC stability/throughput recipe (large pseudobulk runs)**

For large DE jobs on HPC, use a conservative worker cap and disable nested BLAS/OpenMP threading in the job script:

Base template: `slurm/scomnom_template.job`

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Then run DE with:

```bash
scomnom markers-and-de de ... --run pseudobulk --max-workers 8
```

Notes:
* `--max-workers` defaults to `8` for DE.
* Keep `--n-jobs $SLURM_CPUS_PER_TASK`; `--max-workers` controls DE task concurrency.
* If `--run both` is unstable on very large jobs, split into separate `--run pseudobulk` and `--run cell` runs with separate output folders.
* Figures: `figures/<fmt>/de_<round>_roundN/`
* Default AnnData output name: `adata.de_<round>.zarr.tar.zst`

---

### Cell-level vs pseudobulk: why both exist

* **Cell-level** tests are sensitive and can detect subtle expression shifts, but they can overweight large samples, inflate p-values on big datasets, and ignore replicate structure. By default, scOmnom caps marker output at 300 genes per cluster for cell-level runs.
* **Pseudobulk** respects sample-level independence, improves control of confounders via covariates, applies the `min_pct` filter, and yields more conservative inference, but requires enough replicates and loses single-cell resolution.

Running both can be informative: cell-level for discovery, pseudobulk for robustness.
For very large jobs, if `--run both` is unstable (for example due to memory pressure or native-thread failures), run two separate commands instead (`--run pseudobulk` and `--run cell`) into separate output folders and combine the interpretation afterward.

---

### DA (differential abundance / composition)

The DA submodule tests whether **cluster proportions change across conditions**. It works on per-sample cell-type counts and provides multiple backends, each with different assumptions:

**Condition key syntax (DA)**

DA supports the same `A` and `A:B` composite syntax as DE, plus `A@B` to run `A` within each `B` level. Interaction syntax (`A^B`) is **DE-only**. See the DE section above for examples.

You can pass one or more keys (same behavior as DE), and results are written into per-key subfolders under DA tables/figures.

**Reference selection**

* Default reference is `most_stable`, chosen as the cluster with the lowest median absolute deviation of proportions among clusters with mean proportion ≥ `min_mean_prop`.
* If none meet the threshold, the most abundant cluster is used.

**Methods**

* `sccoda`: Bayesian compositional model (pertpy scCODA). Uses NUTS sampling and inclusion probabilities with FDR control. Works directly from cell-level data but models sample-level composition.
* `glm`: Per-cluster binomial GLM on counts with a log-total offset (statsmodels). Supports covariates and returns log2-scale effects with Wald tests and BH FDR.
* `clr`: Centered log-ratio transform of proportions followed by Mann–Whitney tests for each pair of condition levels. Returns log2 fold-change of mean proportions and per-pair FDR.
* `graph`: Graph-based DA on neighborhoods in the integrated embedding, inspired by MiloR. Neighborhoods are sampled from stratified seeds, tested with **NB-GLM** on neighborhood counts (with sample-size offsets), then corrected by **spatial weighted BH FDR**. Outputs neighborhood metadata and diagnostics.

**When to use which DA method**

| Method | Best for | Key assumptions | Notes |
| --- | --- | --- | --- |
| `sccoda` | Small-to-moderate sample sizes with clear compositional shifts | Compositional Bayesian model; needs a reference | Most conservative; provides inclusion probabilities |
| `glm` | Larger sample sizes, covariate adjustment | Binomial GLM with log-total offset | Fast, interpretable log2 effects |
| `clr` | Simple two-group comparisons, quick screening | CLR transform + nonparametric test | Pairwise only; less model structure |
| `graph` | Substructure or neighborhood-level shifts | Embedding neighborhoods + NB-GLM + spatial FDR | Milo-style local DA; conservative under many tests |

**Automatic method behavior**

* GLM is automatically skipped for 2-level conditions (CLR is used instead).
* Conditions with too few replicates per level are skipped with an explicit warning.
* GraphDA applies support filters before testing (`min_nonzero_samples_per_level`) and effect shrinkage (`effect_shrink_k`) to reduce unstable calls.

**GraphDA significance and QC**

GraphDA reports both raw and adjusted evidence:

* `pval`: neighborhood-level NB-GLM p-value
* `fdr_bh`: standard BH correction
* `fdr_spatial`: weighted BH correction using neighborhood spatial weights
* `fdr`: current primary significance column (set to `fdr_spatial`)

Why this matters:

* Many neighborhoods overlap and are not independent tests.
* Spatial FDR is the primary call metric for Milo-style neighborhood DA.
* It is possible to see directional effects with no FDR-significant neighborhoods if multiplicity is high and raw p-values are moderate.

**Outputs**

Per condition key (and per `A@B=<level>` expansion), DA writes:

* Tables: `tables/DA_tables_<run>/<condition_tag>/`
* Figures: `figures/DA/<condition_tag>/`

Key DA tables include:

* `composition_global_sccoda.tsv`
* `composition_global_glm.tsv` (when eligible)
* `composition_global_clr.tsv`
* `composition_global_graph.tsv`
* `composition_consensus.tsv`
* `composition_graph_neighborhoods.tsv`
* `graphda_diagnostics.tsv`
* `composition_settings.txt`

Key DA figures include:

* composition summaries (`composition_stacked_bar_100`, `composition_stacked_comparison`, `composition_flow`)
* global effects (`composition_effects_global_sccoda`, `composition_effects_global_clr`)
* GraphDA summaries (`graphda_effects_by_cluster`, `graphda_top_neighborhoods`, `graphda_top_by_cluster`)
* GraphDA QC (`graphda_qc_pval_vs_fdr`, `graphda_qc_cluster_power`)

---

### Running on HPC / SLURM clusters

Example SLURM job scripts are provided:

```text
slurm/
├── scomnom_template.job
├── scomnom_1_load_and_filter.job
├── scomnom_2_integrate.job
├── scomnom_3_cluster_and_annotate.job
├── scomnom_4_markers.job
├── scomnom_5_integrate_annotated_run.job
├── scomnom_5a_subset.job
├── scomnom_5b_annotation_merge.job
├── scomnom_6_rename.job
├── scomnom_6a_rename_archetypes.job
├── scomnom_7_de.job
├── scomnom_8_da.job
└── scomnom_9_enrichment_cluster.job
```

These scripts are configured for **SURF’s Snellius** compute cluster. Users on other systems must adapt module names, CUDA/driver versions, and conda initialization paths.

For large datasets, running on a **GPU partition is strongly recommended**, particularly for:

* `load-and-filter` doublet detection (scVI-SOLO)
* `integrate` (scVI/scANVI)

#### Performance reference (Snellius)

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


## License

MIT License.
