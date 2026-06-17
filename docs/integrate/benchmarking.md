# scIB benchmarking and truth labels

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
