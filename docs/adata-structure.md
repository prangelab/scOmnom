# scOmnom AnnData Structure

`scOmnom` stores results in standard AnnData fields with a few consistent conventions so downstream modules can find prior outputs.

### Counts layers

* `counts_raw`: raw (uncorrected) counts
* `counts_cb`: CellBender-corrected counts
* `adata.X`: used only when no preferred counts layer is available and a command explicitly allows fallback

External retained-cell AnnData objects imported with `adata-ops import` are standardized by copying the selected count source into `counts_raw`. They do not receive `counts_cb` unless that layer already exists in the imported object.

### Integration outputs

* Integrated embeddings are stored in `adata.obsm` (for example `X_integrated` or method-specific embeddings).
* The selected best embedding is recorded in `adata.uns["integration"]["best_embedding"]` when available.

### Clustering rounds

scOmnom stores clustering and annotation state as **rounds**. A round is a named clustering state with its own raw cluster labels, display labels, annotation pointers, decoupler payloads, and provenance. Rounds allow the same AnnData object to carry multiple related cluster states without overwriting earlier results.

These internal clustering rounds are distinct from filesystem output folders such as `integration_round1` or `de_r3_manual_rename_round1`. See [Output Organization](output-organization.md) for the distinction.

At the object level:

* `adata.uns["cluster_rounds"]`: dictionary of `round_id -> round metadata`
* `adata.uns["cluster_round_order"]`: creation/order list for registered rounds
* `adata.uns["active_cluster_round"]`: the round used by default when a command does not receive `--round-id`

Schematic:

```text
AnnData
├── obs
│   ├── leiden                              active raw cluster alias
│   ├── cluster_label                       active pretty-label alias
│   ├── celltypist_cluster_label            active cluster-level CellTypist alias
│   ├── leiden__r0_X_integrated_BISC        round-scoped raw labels
│   ├── cluster_label__r0_X_integrated_BISC round-scoped pretty labels
│   └── celltypist_cluster_label__r0_...    round-scoped cluster-level CellTypist labels
├── obsm
│   ├── X_integrated
│   ├── X_umap
│   └── celltypist_proba
└── uns
    ├── active_cluster_round: r0_X_integrated_BISC
    ├── cluster_round_order: [r0_X_integrated_BISC, r1_manual_rename, ...]
    └── cluster_rounds
        └── r0_X_integrated_BISC
            ├── round_id
            ├── parent_round_id
            ├── kind / round_type / notes / created_utc
            ├── cluster_key
            ├── labels_obs_key
            ├── best_resolution / sweep / cfg
            ├── cluster_sizes / cluster_order / cluster_display_map
            ├── annotation
            │   ├── celltypist_cell_key
            │   ├── celltypist_cluster_key
            │   └── pretty_cluster_key
            ├── decoupler
            ├── qc / stability / diagnostics
            └── compacting
```

The important distinction is between **round-scoped fields** and **active aliases**:

* Round-scoped columns are stable history. Examples include `leiden__r0_X_integrated_BISC`, `cluster_label__r0_X_integrated_BISC`, and `celltypist_cluster_label__r0_X_integrated_BISC`.
* Active aliases are convenience fields for older code and simple plotting. Examples include `adata.obs["leiden"]`, `adata.obs["cluster_label"]`, and `adata.obs["celltypist_cluster_label"]`.
* When the active round changes, scOmnom mirrors that round's labels into the active aliases where possible. This keeps older code that expects a single `leiden` or `cluster_label` column working, but the round metadata remains the authoritative source.

Each round records the keys needed to recover its state:

| Round field | Meaning |
| --- | --- |
| `round_id` | Stable round identifier, usually `rN_<description>` |
| `parent_round_id` | Parent round when this is derived from another round, such as rename, compaction, or annotation merge |
| `cluster_key` | Compatibility/raw cluster key that should point to this round when active |
| `labels_obs_key` | The `adata.obs` column containing this round's raw cluster labels |
| `annotation.pretty_cluster_key` | The `adata.obs` column containing display labels such as `C00: T cells` |
| `annotation.celltypist_cluster_key` | The round-scoped cluster-level CellTypist label column |
| `cluster_order` | Size-ordered cluster ids used for stable `C00`, `C01`, ... display codes |
| `cluster_display_map` | Mapping from raw cluster id to display label |
| `decoupler` | Round-specific decoupler payload pointers/results |
| `cfg`, `sweep`, `best_resolution` | Provenance for the clustering/selection step |

Downstream modules resolve labels in this order: explicit `--round-id` when provided, otherwise `adata.uns["active_cluster_round"]`. This is why a derived manual rename or subset-annotation round can become the default for markers, DE, DA, enrichment, and CCC without deleting the original clustering round.

Imported external AnnData objects can also receive an imported clustering round. In that case, the round records the original source label column and mirrors the imported labels into the active aliases for compatibility with downstream scOmnom modules.

### Annotations and activities

* Cell-level CellTypist outputs are stored in `adata.obs` and `adata.obsm`; cluster-level CellTypist labels and pretty labels are round-scoped in `adata.obs` and referenced from the active round.
* Decoupler activities are attached to the relevant round in `adata.uns["cluster_rounds"][round_id]["decoupler"]`.
* `adata.uns["cluster_and_annotate"]` keeps compatibility pointers to the currently active CellTypist and pretty-label keys.

### Where to look

* `adata.uns["integration"]`: integration metadata, including the selected best embedding (if available)
* `adata.uns["cluster_rounds"]`: all clustering rounds and their settings
* `adata.uns["active_cluster_round"]`: the currently active clustering round id
* `adata.uns["cluster_and_annotate"]`: compatibility pointers to the active CellTypist and pretty-label keys
* `adata.uns["markers_and_de"]`: provenance for markers, DE, and DA runs
* `adata.uns["scomnom_de"]`: detailed DE tables and summaries (pseudobulk and cell-level contrasts)

---

## Using scOmnom AnnData in notebooks/scripts

When accessing a scOmnom AnnData object in a Python session, always load and save through `scomnom.load_dataset()` and `scomnom.save_dataset()`. This ensures consistent handling of Zarr/H5AD, metadata, and safety checks.

For large Zarr saves with heavy `adata.uns` payloads, `save_dataset()` now stores heavy payloads via sidecar serialization under `__scomnom_payloads__/v1` inside the same store, which reduces save-time memory spikes while keeping `load_dataset()` round-trip behavior.

See also: the [API reference](api-reference.md) for the current Python API namespaces (`scomnom.plotting`, `scomnom.adata_ops`).

```python
from pathlib import Path
from scomnom import load_dataset, save_dataset

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
