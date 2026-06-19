# scOmnom AnnData Structure

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
