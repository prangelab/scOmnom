# scOmnom API Reference

## Module `scomnom`

### `load_dataset`

Signature:
```python
load_dataset(path: Path) -> ad.AnnData
```

What it does:
- Loads a dataset from `.zarr`, `.zarr.tar.zst`, or `.h5ad` into an AnnData object.

Parameters:
- `path`: Dataset input path.

Returns:
- `anndata.AnnData`.

### `save_dataset`

Signature:
```python
save_dataset(adata: ad.AnnData, out_path: Path, fmt: str = "zarr", archive: bool = True) -> None
```

What it does:
- Saves an AnnData object to zarr or h5ad using the project IO path.

Parameters:
- `adata`: AnnData object to save.
- `out_path`: Output dataset path.
- `fmt`: Output format (`"zarr"` or `"h5ad"`).
- `archive`: Archive behavior for zarr outputs.

Returns:
- `None`.

## Module `scomnom.adata_ops`

### `rename_idents`

Signature:
```python
rename_idents(adata: ad.AnnData, mapping: Dict[str, str], parent_round_id: str | None = None, round_name: str = "manual_rename", collapse_same_labels: bool = False, set_active: bool = True, notes: str | None = "Manual rename of pretty labels.") -> str
```

What it does:
- Creates a manual rename round using strict `Cnn` keys and updates labels accordingly. Optionally collapses clusters that are renamed to the same target label and rebuilds the round with fresh size-ordered `Cnn` numbering.

Parameters:
- `adata`: AnnData with cluster-round metadata.
- `mapping`: Map from strict `Cnn` IDs (for example `"C01"`) to new label text.
- `parent_round_id`: Source round ID; if omitted, active round is used.
- `round_name`: Name assigned to the new round.
- `collapse_same_labels`: If `True`, merge clusters that share the same renamed label into one new cluster state.
- `set_active`: Whether to set the new round as active.
- `notes`: Optional notes stored with round metadata.

Returns:
- `str` new round ID.

### `subset_adata_by_cluster_mapping`

Signature:
```python
subset_adata_by_cluster_mapping(adata: ad.AnnData, subset_mapping: Mapping[str, str] | Mapping[str, list[str]], round_id: str | None = None) -> tuple[dict[str, ad.AnnData], pd.DataFrame]
```

What it does:
- Splits AnnData into named subsets based on cluster membership.

Parameters:
- `adata`: Source AnnData.
- `subset_mapping`: Preferred form is `Cnn -> subset_name` (TSV-like). Alternate form `subset_name -> [Cnn, ...]` is also accepted.
- `round_id`: Optional cluster round used for resolving labels.

Returns:
- `tuple[dict[str, AnnData], DataFrame]`: subset objects and summary table.

## Module `scomnom.plotting`

### `plot_de_heatmap_top_genes_by_sample`

Signature:
```python
plot_de_heatmap_top_genes_by_sample(adata, genes: Sequence[str], sample_key: str, condition_key: Optional[str] = None, annotation_keys: Optional[Sequence[str]] = None, use_raw: bool = False, layer: Optional[str] = None, z_clip: float | None = 3.0, cmap: str = 'icefire', display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Sample-level heatmap of selected DE genes with optional annotation bars.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `sample_key`: Sample key in `adata.obs` for sample-level heatmap.
- `condition_key`: Condition column/key used for grouping contrasts.
- `annotation_keys`: Optional annotation keys to display as sample side colors.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `z_clip`: Clip z-scores to this absolute value.
- `cmap`: Colormap name.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_volcano`

Signature:
```python
plot_de_volcano(df_de: pd.DataFrame, gene_col: str = 'gene', padj_col: str = 'padj', lfc_col: str = 'log2FoldChange', padj_thresh: float = 0.05, lfc_thresh: float = 1.0, top_label_n: int = 15, label_genes: Optional[Sequence[str]] = None, title: Optional[str] = None, figsize: tuple[float, float] = (7.5, 6.0), alpha: float = 0.65, s: float = 10.0, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Volcano plot from a DE table (pseudobulk or otherwise).

Parameters:
- `df_de`: Differential expression result table.
- `gene_col`: Column name for gene identifiers.
- `padj_col`: Column name for adjusted p-values.
- `lfc_col`: Column name for log fold-change.
- `padj_thresh`: Adjusted p-value threshold.
- `lfc_thresh`: Absolute log fold-change threshold.
- `top_label_n`: Number of top genes to annotate.
- `label_genes`: Optional explicit gene list to annotate.
- `title`: Optional title override.
- `figsize`: Figure size in inches.
- `alpha`: Plot alpha for points.
- `s`: Point size.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_dotplot_top_genes`

Signature:
```python
plot_de_dotplot_top_genes(adata, genes: Sequence[str], groupby: str, use_raw: bool = False, layer: Optional[str] = None, standard_scale: Optional[str] = 'var', dendrogram: bool = False, swap_axes: bool = False, group_order: Optional[Sequence[str]] = None, color_map: str = 'viridis', figsize: Optional[tuple[float, float]] = None, dot_min: float = 0.0, dot_max: float = 180.0, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Dotplot of selected DE genes across groups/clusters.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `groupby`: Grouping key in `adata.obs`.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `standard_scale`: Scanpy dotplot standardization mode.
- `dendrogram`: Whether to draw dendrogram in dotplot.
- `swap_axes`: Whether to transpose dotplot orientation.
- `group_order`: Optional order of groups shown on the categorical axis.
- `color_map`: Dotplot colormap name.
- `figsize`: Figure size in inches.
- `dot_min`: Minimum displayed dot size after normalization.
- `dot_max`: Maximum displayed dot size after normalization.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_heatmap_top_genes`

Signature:
```python
plot_de_heatmap_top_genes(adata, genes: Sequence[str] | None = None, genes_by_cluster: Mapping[str, Sequence[str]] | None = None, groupby: str, use_raw: bool = False, layer: Optional[str] = None, cmap: str | None = None, show_cluster_colorbar: bool = True, scale_columns_by_size: bool = True, min_col_width: float = 0.5, max_col_width: float = 5.0, figsize: Optional[tuple[float, float]] = None, show_gene_labels: bool = True, z_clip: float | None = 2.5, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Grouped heatmap of selected DE genes or per-cluster gene sets.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `genes_by_cluster`: Per-cluster gene list mapping for combined heatmap.
- `groupby`: Grouping key in `adata.obs`.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `cmap`: Colormap name.
- `show_cluster_colorbar`: Whether to show top cluster color bar in heatmap.
- `scale_columns_by_size`: Scale heatmap column widths by group size.
- `min_col_width`: Minimum scaled heatmap column width.
- `max_col_width`: Maximum scaled heatmap column width.
- `figsize`: Figure size in inches.
- `show_gene_labels`: Whether to render gene labels on heatmap y-axis.
- `z_clip`: Clip z-scores to this absolute value.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_violin_grid_genes`

Signature:
```python
plot_de_violin_grid_genes(adata, genes: Sequence[str], groupby: str, display_groupby: str | None = None, use_raw: bool = False, layer: str | None = None, ncols: int = 3, stripplot: bool = False, dot_size: float | None = None, rotation: float = 45, figsize: tuple[float, float] | None = None, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Multi-panel violin grid of selected DE genes across groups.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `groupby`: Grouping key in `adata.obs`.
- `display_groupby`: Optional display grouping key for violin grid.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `ncols`: Number of subplot columns.
- `stripplot`: Whether to overlay stripplot points.
- `dot_size`: Strip-dot marker size; if `0`, strip dots are hidden.
- `rotation`: X-label rotation angle.
- `figsize`: Figure size in inches.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_violin_genes`

Signature:
```python
plot_de_violin_genes(adata, genes, groupby: str, use_raw: bool = False, layer: str | None = None, rotation: int | float | None = 45, figsize = None, dot_size: float | None = None, display: bool = True, file: str | Path | None = None, return_fig: bool = False, **kwargs)
```

What it plots:
- Violin plot(s) for selected DE genes across groups.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `groupby`: Grouping key in `adata.obs`.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `rotation`: X-label rotation angle.
- `figsize`: Figure size in inches.
- `dot_size`: Strip-dot marker size; if `0`, strip dots are hidden.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.
- `**kwargs`: Additional keyword arguments forwarded to the underlying plotting function.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_umap_features_grid`

Signature:
```python
plot_de_umap_features_grid(adata, genes: Sequence[str], use_raw: bool = False, layer: Optional[str] = None, ncols: int = 3, palette: Optional[Sequence[str] | Mapping[str, str]] = None, cmap: str = 'viridis', vmin: Optional[float] = None, vmax: Optional[float] = None, size: Optional[float] = None, show_umap_corner_axes: bool = False, rasterize: bool = True, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Grid of UMAP feature plots for selected genes.

Parameters:
- `adata`: Input AnnData object.
- `genes`: Genes/features to plot.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `ncols`: Number of subplot columns.
- `palette`: Optional palette for categorical coloring when applicable.
- `cmap`: Colormap name.
- `vmin`: Lower bound for colormap scaling.
- `vmax`: Upper bound for colormap scaling.
- `size`: Point size for UMAP scatter.
- `show_umap_corner_axes`: Draw small UMAP1/UMAP2 orientation arrows in the lower-left corner.
- `rasterize`: Rasterize collections for lighter vector exports.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.

### `plot_de_umap_single`

Signature:
```python
plot_de_umap_single(adata, color: str | None = None, use_raw: bool = False, layer: Optional[str] = None, palette: Optional[Sequence[str] | Mapping[str, str]] = None, cmap: str = 'viridis', vmin: Optional[float] = None, vmax: Optional[float] = None, size: Optional[float] = None, legend_loc: str | None = 'right margin', title: str | None = None, show_umap_corner_axes: bool = False, rasterize: bool = True, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Single UMAP panel colored by one key or gene (default is an ident-like obs key if available).

Parameters:
- `adata`: Input AnnData object.
- `color`: Obs key or gene to color by; if omitted, an ident-like key is auto-selected.
- `use_raw`: Use `adata.raw` expression values if available.
- `layer`: AnnData layer name to use for expression.
- `palette`: Optional palette for categorical coloring.
- `cmap`: Colormap name for continuous coloring.
- `vmin`: Lower bound for colormap scaling.
- `vmax`: Upper bound for colormap scaling.
- `size`: Point size for UMAP scatter.
- `legend_loc`: Legend location passed to Scanpy.
- `title`: Optional title override.
- `show_umap_corner_axes`: Draw small UMAP1/UMAP2 orientation arrows in the lower-left corner.
- `rasterize`: Rasterize collections for lighter vector exports.
- `display`: If true, display figure(s) in notebook/interactive sessions.
- `file`: Optional output file path including extension.
- `return_fig`: If true, return figure object(s); otherwise return `None`.

Returns:
- `None` when `return_fig=False` (default).
- A Matplotlib `Figure` when `return_fig=True` and a single panel is generated.
- `list[Figure]` when `return_fig=True` and multiple panels are generated.
