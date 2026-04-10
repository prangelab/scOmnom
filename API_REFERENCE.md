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

## Namespaces `scomnom.adata_ops` and `scomnom.markers_and_de`

`scomnom.adata_ops` remains the compatibility namespace for AnnData-oriented helpers.
The enrichment helpers are also exposed under `scomnom.markers_and_de` for notebook workflows.

### `rename_idents`

Signature:
```python
rename_idents(adata: ad.AnnData, mapping: Dict[str, str], parent_round_id: str | None = None, new_round_id: str | None = None, round_name: str = "manual_rename", collapse_same_labels: bool = False, update_existing_round: bool = False, set_active: bool = True, notes: str | None = "Manual rename of pretty labels.") -> str
```

What it does:
- Creates a manual rename round using strict `Cnn` keys and updates labels accordingly. Optionally collapses clusters that are renamed to the same target label and rebuilds the round with fresh size-ordered `Cnn` numbering.

Parameters:
- `adata`: AnnData with cluster-round metadata.
- `mapping`: Map from strict `Cnn` IDs (for example `"C01"`) to new label text.
- `parent_round_id`: Source round ID; if omitted, active round is used.
- `new_round_id`: Explicit target round ID. Primarily useful together with `update_existing_round=True`.
- `round_name`: Name assigned to the new round.
- `collapse_same_labels`: If `True`, merge clusters that share the same renamed label into one new cluster state.
- `update_existing_round`: If `True`, update the explicitly targeted existing `manual_rename` round instead of creating a new one.
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

### `enrichment_cluster`

Signature:
```python
enrichment_cluster(adata: ad.AnnData, *, round_id: str | None = None, condition_key: str | None = None, gene_filter: Sequence[str] = (), decoupler_pseudobulk_agg: str = "mean", decoupler_use_raw: bool = True, decoupler_method: str = "consensus", decoupler_consensus_methods: Sequence[str] | None = ("ulm", "mlm", "wsum"), decoupler_min_n_targets: int = 5, msigdb_gene_sets: Sequence[str] | None = ("HALLMARK", "REACTOME"), msigdb_method: str = "consensus", msigdb_min_n_targets: int = 5, run_progeny: bool = True, progeny_method: str = "consensus", progeny_min_n_targets: int = 5, progeny_top_n: int = 100, progeny_organism: str = "human", run_dorothea: bool = True, dorothea_method: str = "consensus", dorothea_min_n_targets: int = 5, dorothea_confidence: Sequence[str] | None = ("A", "B", "C"), dorothea_organism: str = "human") -> dict
```

What it does:
- Runs round-native enrichment on an in-memory AnnData object and stores the decoupler payload on the selected round.

Parameters:
- `adata`: AnnData with cluster-round metadata and expression data.
- `round_id`: Optional target round. If omitted, the active round is used.
- `condition_key`: Optional condition key for cluster-by-condition enrichment. Composite `A:B` syntax is supported.
- `gene_filter`: Optional gene filter expressions applied before enrichment.
- `decoupler_pseudobulk_agg`: Aggregation for round-native pseudobulk (`"mean"` or `"median"`).
- `decoupler_use_raw`: Whether to prefer raw-like counts for pseudobulk.
- `decoupler_method`: Decoupler scoring method.
- `decoupler_consensus_methods`: Methods combined when `decoupler_method="consensus"`.
- `decoupler_min_n_targets`: Minimum targets per pathway/TF set.
- `msigdb_gene_sets`: MSigDB keywords and/or `.gmt` files.
- `msigdb_method`: Decoupler method for MSigDB.
- `msigdb_min_n_targets`: Minimum MSigDB targets retained.
- `run_progeny`: Whether to compute PROGENy.
- `progeny_method`: Decoupler method for PROGENy.
- `progeny_min_n_targets`: Minimum PROGENy targets retained.
- `progeny_top_n`: PROGENy top-N footprint size.
- `progeny_organism`: PROGENy organism.
- `run_dorothea`: Whether to compute DoRothEA.
- `dorothea_method`: Decoupler method for DoRothEA.
- `dorothea_min_n_targets`: Minimum DoRothEA targets retained.
- `dorothea_confidence`: DoRothEA confidence levels.
- `dorothea_organism`: DoRothEA organism.

Returns:
- `dict`: The selected round's stored decoupler payload.

### `enrichment_de_from_tables`

Signature:
```python
enrichment_de_from_tables(input_dir: str | Path, *, gene_filter: Sequence[str] = (), de_decoupler_source: str = "auto", de_decoupler_stat_col: str = "stat", decoupler_method: str = "consensus", decoupler_consensus_methods: Sequence[str] | None = ("ulm", "mlm", "wsum"), decoupler_min_n_targets: int = 5, msigdb_gene_sets: Sequence[str] | None = ("HALLMARK", "REACTOME"), msigdb_method: str = "consensus", msigdb_min_n_targets: int = 5, run_progeny: bool = True, progeny_method: str = "consensus", progeny_min_n_targets: int = 5, progeny_top_n: int = 100, progeny_organism: str = "human", run_dorothea: bool = True, dorothea_method: str = "consensus", dorothea_min_n_targets: int = 5, dorothea_confidence: Sequence[str] | None = ("A", "B", "C"), dorothea_organism: str = "human") -> dict[str, dict[str, dict[str, dict[str, object]]]]
```

What it does:
- Runs enrichment directly from exported DE CSV tables on disk, without loading AnnData.

Parameters:
- `input_dir`: Folder containing exported DE result tables.
- `gene_filter`: Optional gene filter expressions applied before enrichment.
- `de_decoupler_source`: Which DE table source to use (`"auto"`, `"all"`, `"pseudobulk"`, `"cell"`, `"none"`).
- `de_decoupler_stat_col`: Preferred statistic column used to build the enrichment matrix.
- `decoupler_method`: Decoupler scoring method.
- `decoupler_consensus_methods`: Methods combined when `decoupler_method="consensus"`.
- `decoupler_min_n_targets`: Minimum targets per pathway/TF set.
- `msigdb_gene_sets`: MSigDB keywords and/or `.gmt` files.
- `msigdb_method`: Decoupler method for MSigDB.
- `msigdb_min_n_targets`: Minimum MSigDB targets retained.
- `run_progeny`: Whether to compute PROGENy.
- `progeny_method`: Decoupler method for PROGENy.
- `progeny_min_n_targets`: Minimum PROGENy targets retained.
- `progeny_top_n`: PROGENy top-N footprint size.
- `progeny_organism`: PROGENy organism.
- `run_dorothea`: Whether to compute DoRothEA.
- `dorothea_method`: Decoupler method for DoRothEA.
- `dorothea_min_n_targets`: Minimum DoRothEA targets retained.
- `dorothea_confidence`: DoRothEA confidence levels.
- `dorothea_organism`: DoRothEA organism.

Returns:
- `dict[str, dict[str, dict[str, dict[str, object]]]]`: Nested payload bundle keyed as `condition_key -> contrast -> source -> payload`.

### `module_score`

Signature:
```python
module_score(adata: ad.AnnData, *, module_files: Sequence[str | Path], module_set_name: str | None = None, round_id: str | None = None, condition_key: str | None = None, module_score_method: str = "scanpy", module_score_use_raw: bool = False, module_score_layer: str | None = None, module_score_ctrl_size: int = 50, module_score_n_bins: int = 25, module_score_random_state: int = 0, module_score_max_umaps: int = 12) -> dict
```

What it does:
- Runs user-defined module scoring on an in-memory AnnData object and stores the payload on the selected round.

Parameters:
- `adata`: AnnData with cluster-round metadata and expression data.
- `module_files`: Module definition files (`.gmt`, `.tsv`, `.csv`, or single-module `.txt`).
- `module_set_name`: Optional stable name for the module collection. Defaults to the first file stem.
- `round_id`: Optional target round. If omitted, the active round is used.
- `condition_key`: Optional condition key for cluster-by-condition summaries. Composite `A:B` syntax is supported.
- `module_score_method`: Module-scoring backend. `"scanpy"` uses `scanpy.tl.score_genes`; `"aucell"` uses `decoupler.mt.aucell`.
- `module_score_use_raw`: Whether to score using `adata.raw`.
- `module_score_layer`: Optional AnnData layer to score from.
- `module_score_ctrl_size`: Control gene set size for `scanpy.tl.score_genes`.
- `module_score_n_bins`: Number of expression bins for control gene sampling.
- `module_score_random_state`: Random seed for control gene sampling.
- `module_score_max_umaps`: Stored for CLI parity; does not affect in-memory computation.

Returns:
- `dict`: The selected round's stored module-score payload.

## Module `scomnom.plotting`

### `plot_decoupler_payload`

Signature:
```python
plot_decoupler_payload(payload: dict, net_name: str, heatmap_top_k: int = 30, bar_top_n: int = 10, bar_top_n_up: int | None = None, bar_top_n_down: int | None = None, bar_split_signed: bool = True, dotplot_top_k: int = 30, title_prefix: str | None = None, round_id: str | None = None, cluster_display_map: Mapping[str, str] | None = None, cluster_display_labels: Sequence[str] | None = None, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Cluster-style enrichment payloads returned by `scomnom.markers_and_de.enrichment_cluster(...)`.

Parameters:
- `payload`: Either a single network payload (for example `decoupler_payload["msigdb"]`) or the full round decoupler bundle returned by `enrichment_cluster(...)`.
- `net_name`: Network name to plot (`"msigdb"`, `"progeny"`, or `"dorothea"`).
- `heatmap_top_k`: Number of features in the heatmap.
- `bar_top_n`: Number of features in the per-cluster barplots.
- `bar_top_n_up`: Optional positive-feature cap for signed barplots.
- `bar_top_n_down`: Optional negative-feature cap for signed barplots.
- `bar_split_signed`: Whether to split signed barplots into positive and negative blocks.
- `dotplot_top_k`: Number of features in the dotplot.
- `title_prefix`: Optional title prefix added to plots.
- `round_id`: Optional round label added to titles and stems when the payload itself does not carry one.
- `cluster_display_map`: Optional explicit cluster display mapping override.
- `cluster_display_labels`: Optional explicit display order override.
- `display`: Whether to display figures inline.
- `file`: Optional explicit output file path.
- `return_fig`: Whether to return figure handles.

Returns:
- `None`, one `matplotlib.figure.Figure`, or a list of figures depending on `return_fig`.

### `plot_de_decoupler_payload`

Signature:
```python
plot_de_decoupler_payload(payload: dict, net_name: str, heatmap_top_k: int = 30, bar_top_n: int = 10, bar_top_n_up: int | None = None, bar_top_n_down: int | None = None, bar_split_signed: bool = True, dotplot_top_k: int = 30, title_prefix: str | None = None, pos_label: str | None = None, neg_label: str | None = None, display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- DE-derived enrichment payloads returned by `scomnom.markers_and_de.enrichment_de_from_tables(...)`.

Parameters:
- `payload`: Either a single network payload (for example `source_payload["nets"]["msigdb"]`) or a full source payload containing a `nets` block.
- `net_name`: Network name to plot (`"msigdb"`, `"progeny"`, or `"dorothea"`).
- `heatmap_top_k`: Number of features in the heatmap.
- `bar_top_n`: Number of features in the barplots.
- `bar_top_n_up`: Optional positive-feature cap for signed barplots.
- `bar_top_n_down`: Optional negative-feature cap for signed barplots.
- `bar_split_signed`: Whether to split signed barplots into positive and negative blocks.
- `dotplot_top_k`: Number of features in the dotplot.
- `title_prefix`: Optional title prefix added to plots.
- `pos_label`: Optional label for the positive direction.
- `neg_label`: Optional label for the negative direction.
- `display`: Whether to display figures inline.
- `file`: Optional explicit output file path.
- `return_fig`: Whether to return figure handles.

Returns:
- `None`, one `matplotlib.figure.Figure`, or a list of figures depending on `return_fig`.

### `plot_module_score_summary_heatmap`

Signature:
```python
plot_module_score_summary_heatmap(summary: pd.DataFrame, stem: str = "module_score_summary_mean_z", title: str | None = None, cmap: str = "vlag", display: bool = True, file: str | Path | None = None, return_fig: bool = False)
```

What it plots:
- Heatmap of summarized module-score values, typically the `summary_mean_z` block returned by `scomnom.markers_and_de.module_score(...)`.

Parameters:
- `summary`: Module-score summary matrix with groups on rows and modules on columns.
- `stem`: Output stem when saving through the API wrapper.
- `title`: Optional plot title.
- `cmap`: Heatmap colormap.
- `display`: Whether to display figures inline.
- `file`: Optional explicit output file path.
- `return_fig`: Whether to return figure handles.

Returns:
- `None` or a `matplotlib.figure.Figure` depending on `return_fig`.

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
