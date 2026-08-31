# Outputs

`load-and-filter` writes the initial merged scOmnom dataset and QC artifacts.

Key outputs:

- merged AnnData object, typically `adata.filtered.zarr` or an archived Zarr output depending on the selected output format
- optional `.h5ad` output when requested
- QC figures under `figures/`
- `doublets_per_sample.tsv`, with per-sample SOLO thresholds and observed doublet rates when doublet detection runs
- `doublet_detection_status.tsv` when `--skip-doublet-detection` is used
- `load-and-filter.log`

Relevant AnnData metadata:

- `adata.uns["doublet_calling"]`: configured expected rate, derived per-sample score thresholds, and observed doublet fractions; or an explicit `performed=False` skip record
- `adata.uns["solo_scoring"]`: requested/effective SOLO scoring mode, sparse operation estimate, fallback reason, and block summaries; or an explicit `performed=False` skip record

The resulting AnnData object is the normal input for [`integrate`](../integrate.md).

For the AnnData structure used across the project, see [scOmnom AnnData Structure](../adata-structure.md).
