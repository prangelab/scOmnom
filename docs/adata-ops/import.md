# Import external AnnData (`adata-ops import`)

Use `adata-ops import` when you already have a filtered AnnData object from another workflow and want to bring it into scOmnom conventions. This is an import path for retained-cell AnnData objects, not raw 10x matrices. For raw matrices, use [Load And Filter](../load-and-filter.md).

The command leaves the expression matrix and existing annotations in place, then adds the scOmnom fields that downstream modules expect:

* copies the chosen count source into `adata.layers["counts_raw"]`
* records a detected or requested batch/sample key in `adata.uns["batch_key"]`
* creates an imported clustering round from an existing `obs` label column when one is available
* registers a detected or requested integration embedding as `adata.uns["integration"]["best_embedding"]`
* stores import provenance in `adata.uns["import_provenance"]`

Minimal example:

```bash
scomnom adata-ops import \
  --input-path external_adata.h5ad \
  --output-dir results
```

Explicit example:

```bash
scomnom adata-ops import \
  --input-path external_adata.h5ad \
  --output-dir results \
  --output-name external.imported \
  --source-count-layer counts \
  --cluster-key cell_type \
  --batch-key sample_id \
  --embedding-key X_pca_harmony \
  --round-name external_labels
```

## Inputs

Supported input formats are the same dataset formats handled by `scomnom.load_dataset()`:

* `.h5ad`
* `.zarr`
* `.zarr.tar.zst`

The input should already contain the cells you intend to analyze. The importer does not run CellBender matching, ambient correction, mitochondrial/ribosomal filters, doublet detection, or MAD-based QC. Those steps belong to `load-and-filter`.

## Options and defaults

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-path`, `-i` | required | External AnnData dataset to import |
| `--output-dir`, `-o` | nearest sensible `results/` location | Output directory |
| `--output-name` | `<input_stem>.imported` | Output basename |
| `--output-format` | `h5ad` for `.h5ad` inputs, otherwise compressed Zarr | Output format: `h5ad` or `zarr` |
| `--source-count-layer` | auto-detect | Layer containing retained-cell counts; use `X` to copy `adata.X` |
| `--cluster-key` | auto-detect | `obs` column used to create the imported clustering round |
| `--batch-key` | auto-detect | `obs` column stored as `adata.uns["batch_key"]` |
| `--embedding-key` | auto-detect | `obsm` key registered as the imported best embedding |
| `--round-name` | `imported` | Suffix used for the created imported clustering round |

Auto-detected count-layer candidates are checked in this order:

```text
raw, counts, counts_raw, counts_filtered, counts_cb
```

If none of those layers exist, `adata.X` is used only when it looks count-like. Pass `--source-count-layer X` explicitly if the count matrix is intentionally stored in `adata.X`.

Auto-detected cluster keys are checked in this order:

```text
full_clustering, initial_clustering, cluster_label, leiden, louvain,
seurat_clusters, clusters, cluster, cell_type, celltype
```

Auto-detected batch keys are checked in this order:

```text
sample_id, sample, patient_id, batch, donor_id, donor, orig.ident
```

Auto-detected embedding keys are checked in this order:

```text
X_integrated, X_pca_harmony, X_scANVI, X_scVI, X_scanorama, X_bbknn
```

## Output conventions

The imported dataset receives `adata.layers["counts_raw"]` as the canonical raw-count layer for downstream count-based modules. It does not create `counts_cb`, because no CellBender-corrected matrix has been produced by this import step.

When `--cluster-key` is detected or supplied, scOmnom creates a new clustering round with `kind="import"` and makes it active. The imported labels are also propagated into the standard active aliases such as `adata.obs["cluster_label"]` so downstream modules can use the object like a native scOmnom clustering result. The round metadata remains the authoritative record; see [scOmnom AnnData Structure](../adata-structure.md).

When no cluster key is available, the dataset is still imported with count-layer and provenance conventions, but no meaningful clustering round is created. Run clustering or provide an explicit `--cluster-key` before using round-aware downstream modules.

The importer records a one-row summary internally and stores the following provenance fields in `adata.uns["import_provenance"]`:

* source path
* source count layer
* stored count layer
* source cluster key
* source batch key
* source embedding key
* created round id

## Related commands

Use `adata-ops metadata-import` when you only want to add or replace `obs` metadata columns on an existing scOmnom object. Use `adata-ops import` when the whole AnnData object itself is coming from an external workflow.

---
