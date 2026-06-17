# Integrate (batch correction + benchmarking)

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

