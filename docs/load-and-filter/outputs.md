# Outputs

`load-and-filter` writes the initial merged scOmnom dataset and QC artifacts.

Key outputs:

- merged AnnData object, typically `adata.filtered.zarr` or an archived Zarr output depending on the selected output format
- optional `.h5ad` output when requested
- QC figures under `figures/`
- `load-and-filter.log`

The resulting AnnData object is the normal input for [`integrate`](../integrate.md).

For the AnnData structure used across the project, see [scOmnom AnnData Structure](../adata-structure.md).
