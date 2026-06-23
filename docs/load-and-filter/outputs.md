# Outputs

`load-and-filter` writes the initial merged scOmnom dataset and QC artifacts.

Key outputs:

- merged AnnData object, typically `adata.filtered.zarr` or an archived Zarr output depending on the selected output format
- optional `.h5ad` output when requested
- QC figures under `figures/`
- `doublets_per_sample.tsv`, with per-sample SOLO thresholds and observed doublet rates
- `load-and-filter.log`

Relevant AnnData metadata:

- `adata.uns["doublet_calling"]`: expected rate, inferred per-sample thresholds, and observed doublet fractions
- `adata.uns["solo_scoring"]`: requested/effective SOLO scoring mode, sparse operation estimate, fallback reason, and block summaries

The resulting AnnData object is the normal input for [`integrate`](../integrate.md).

For the AnnData structure used across the project, see [scOmnom AnnData Structure](../adata-structure.md).
