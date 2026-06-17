# Cluster and annotate (BISC + compaction)

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
