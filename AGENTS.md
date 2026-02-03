# scOmnom project-wide agent instructions

These rules are persistent for all threads and apply across the whole project.

## Data IO (mandatory)
- All dataset loading must go through `load_dataset()` in `src/scomnom/io_utils.py`.
- All dataset saving must go through `save_dataset()` in `src/scomnom/io_utils.py`.
- Do not call `adata.write_zarr`, `adata.write_h5ad`, `ad.read_zarr`, or `ad.read_h5ad` directly in new or modified code.

## Plot Saving (mandatory)
- All plot saving must go through `save_multi()` in `src/scomnom/plot_utils.py`.
- Do not call `plt.savefig` or `fig.savefig` directly in new or modified code.
- When saving figures, ensure `setup_scanpy_figs(...)` has been called so `save_multi()` can route outputs correctly.
