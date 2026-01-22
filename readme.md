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

The `integrate` module performs batch correction across samples while preserving biological signal. It supports multiple integration algorithms, optional benchmarking with `scIB`, and an advanced supervision strategy for `scANVI`.

By default, integration is **non-destructive**: all embeddings, metrics, and selections are stored in the output `AnnData` object.

---

### Standard integration run

A typical integration run starts from the output of `load-and-filter`:

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

This will:

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

**Default methods:** all of the above except `scPoli`.

Methods can be restricted explicitly:

```bash
scomnom integrate \
  --input-path adata.filtered.zarr \
  --batch-key sample_id \
  --methods scVI scANVI Harmony \
  --output-dir results/integrate/
```

---

### Intelligent scANVI supervision (BISC-light)

When `scANVI` is enabled, `scOmnom` can automatically generate supervision labels instead of relying on a single, user-chosen clustering.

This **BISC-light** procedure is inspired by the full BISC framework used in `cluster-and-annotate` (see that section for detailed documentation).

At a high level:

1. **Latent-space resolution sweep**
   A coarse Leiden sweep is performed on the scVI latent space.

2. **Structural scoring**
   Candidate resolutions are scored using:

   * clustering stability (smoothed ARI between adjacent resolutions)
   * centroid-based silhouette separation
   * penalties for excessive tiny clusters

3. **Parsimony selection**
   The *lowest* resolution within a small epsilon of the best score is selected.

4. **Guardrails**

   * clusters dominated by a single batch are masked (`Unknown`)
   * very small clusters are masked

The resulting labels are used **only for scANVI supervision** and are written to `adata.obs` (default: `scanvi_prelabels`). They do *not* overwrite downstream clustering or annotation results.

This provides robust, automated supervision without requiring curated biological labels.

---

### scIB benchmarking and truth labels

Integration methods are benchmarked using `scIB`.

By default, benchmarking uses:

* **`leiden`** labels (computed during `load-and-filter`, resolution = 1.0)

This can be overridden via:

```bash
--scib-truth-label-key <key>
```

Valid options include:

* `leiden` (default, recommended for general use)
* `final` / final cluster labels from `cluster-and-annotate`
* any explicit `adata.obs` column name

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

### Output files

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
integrate --annotated-run (optional)
```

The secondary integration step is **purely optional** and should be used only when:

* clustering and annotation are finalized
* improved visualization or refined latent structure is desired

---

# Cluster and annotate (BISC + compaction)

The cluster-and-annotate module performs automated clustering, biologically informed resolution selection, cell type annotation, and optional cluster compaction on an integrated dataset.

It is designed to be run after integrate and operates on the selected integrated embedding (typically X_integrated).

At a high level, this module provides:

* automated Leiden resolution selection using BISC
* biologically informed clustering diagnostics
* CellTypist-based cluster annotation
* decoupler-based pathway and transcription factor activity analysis
* optional cluster compaction to improve interpretability
* a self-contained HTML report

---

## Minimal usage

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

## CellTypist models

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

For model descriptions, training data, and usage guidance, see the official CellTypist documentation:

[https://www.celltypist.org](https://www.celltypist.org)

---

## Reproducibility and non-destructive design

All clustering, annotation, and compaction steps in scOmnom are **non-destructive**:

* intermediate results are preserved
* no clustering step overwrites a previous result
* all decisions (parameters, metrics, diagnostics) are stored in the dataset

This design ensures that clustering results are fully reproducible, auditable, and comparable across runs.

---

## BISC: Biology-Informed Structural Clustering

BISC automates Leiden resolution selection by combining structural, stability, and biological signals.

The goal is to replace manual resolution tuning with a reproducible, data-driven procedure.

---

### Resolution sweep

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

### Structural stability (ARI)

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

### Biological metrics (CellTypist-guided)

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

### Composite score and resolution selection

All metrics are normalized and combined into a single composite score:

* structural stability (`w_stab = 0.50`)
* silhouette separation (`w_sil = 0.35`)
* tiny cluster penalty (`w_tiny = 0.15`)
* optional biological metrics (weights above)

The final resolution is selected from stable plateaus, with a bias toward simpler solutions when scores are comparable.

---

## Cluster annotation

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

## Compaction: merging redundant clusters

Large datasets can yield clusters that are transcriptionally distinct but *biologically redundant*. The optional **compaction** step merges such clusters using **multi-view biological agreement** computed from decoupler activities.

Compaction is **CellTypist-grouped**: cluster pairs are only compared *within the same* CellTypist cluster label group (with an option to skip `Unknown` groups).

### What signals are used

Compaction requires decoupler outputs from:

- **PROGENy** activity (required)
- **DoRothEA** activity (required)
- **MSigDB activity split by GMT** (required by default; can be made optional)

### Similarity metric

For each pair of clusters within the same CellTypist group:

- activities are **z-scored** (see scope below)
- similarity is computed as **cosine similarity** on the z-scored vectors

For MSigDB, similarity is computed using a **Top-K union** strategy (default **K = 25**):
- take the union of the top-|z| features from both clusters (by absolute z-score)
- compute cosine similarity on that union vector

### Z-score scope (with guardrail)

Compaction supports two z-scoring modes:

- `global` (recommended): z-score each activity column across **all clusters**
- `within_celltypist_label`: z-score each activity column **within each CellTypist group**

Guardrail: `within_celltypist_label` falls back to `global` if a CellTypist group contains fewer than **4** clusters (to avoid unstable z-scores).

Default in the config:
- `compact_zscore_scope = "global"`

### Adaptive thresholds (per CellTypist group)

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

### MSigDB majority rule

MSigDB is evaluated per GMT block, then combined using a majority rule:

- if there is **1** GMT: require **1/1**
- if there are **2** GMTs: require **1/2**
- if there are **≥ 3** GMTs: require `ceil(0.67 × n_gmts)` passing GMTs

Default MSigDB majority fraction: **0.67**.

If `msigdb_required = False`, MSigDB is not required for passing (but similarities are still computed if available).

### Pass condition and grouping

An edge (merge suggestion) between two clusters is created only if all required views pass:

- PROGENy similarity ≥ effective PROGENy threshold **AND**
- DoRothEA similarity ≥ effective DoRothEA threshold **AND**
- MSigDB majority rule passes (if required)

Edges are then converted into merge groups using one of:

- `connected_components` (default): merge connected components of the pass graph
- `clique`: greedy clique cover on the pass graph (stricter grouping)

Default in the config:
- `compact_grouping = "connected_components"`

### Size and label filters

- `compact_min_cells` (default `0`): clusters smaller than this are excluded from compaction decisions
- `compact_skip_unknown_celltypist_groups` (default `False`): optionally exclude CellTypist groups labeled `Unknown`/`UNKNOWN`

### Outputs

Compaction produces:

- a new clustering stored as a new, non-destructive result
- a full **pairwise audit table** of edges with:
  - per-view similarities
  - adaptive thresholds (and which scope was effectively used)
  - MSigDB majority statistics and failing GMTs
- a **decision log** listing multi-cluster merge groups and their rationale
- a mapping from original cluster ids to compacted ids (e.g. `C00`, `C01`, …), sorted by total component size

---

## Decoupler configuration

Decoupler is enabled by default (`run_decoupler = True`) and is used for:

* cluster-level pseudobulk activity inference
* biological diagnostics
* compaction decisions

**Defaults:**

* aggregation: **mean**
* method: **consensus** (`ulm`, `mlm`, `wsum`)
* databases **MSigDB, DoRohEA, PROGENy"**: (`HALLMARK`, `REACTOME`)
* minimum targets per source: **5**

All decoupler settings — including databases, methods, thresholds, and organisms — are fully configurable.

For details on decoupler methods and resources, see:

[https://decoupler-py.readthedocs.io](https://decoupler-py.readthedocs.io)

---

## Reporting

All modules generate a self-contained HTML report including:

* Run information
* All genreated plots

The report is written to:

```
figures/<MODULE>/<MODULE>_report.html
```

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


## License

MIT License.

