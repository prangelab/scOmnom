# Compaction: merging redundant clusters

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

* a new clustering stored as a new, non-destructive AnnData clustering round
* a full **pairwise audit table** of edges with per-view similarities, adaptive thresholds (and which scope was effectively used), and MSigDB majority statistics
* a **decision log** listing multi-cluster merge groups and their rationale
* a mapping from original cluster ids to compacted ids (e.g. `C00`, `C01`, …), sorted by total component size

---
