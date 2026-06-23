# Filtering Defaults And Rationale

`load-and-filter` applies conservative QC filters before samples are merged. The defaults are designed to remove obvious low-quality cells while avoiding strong dataset-specific assumptions.

## Summary Of Default Filters

| Filter | Default | Purpose |
| --- | --- | --- |
| `--min-cells` | `3` | Remove genes detected in too few cells. |
| `--min-genes` | `500` | Remove cells with too few detected genes. |
| `--min-counts` | `none` | Optional fixed lower UMI-count floor; off by default. |
| `--min-counts-mad` | `5.0` | Candidate lower-tail `total_counts` cutoff: median minus `k * MAD`. |
| `--min-counts-quantile` | `none` | Candidate lower-tail `total_counts` quantile cutoff; off by default. |
| `--min-counts-auto-activate-quantile` | `0.01` | Activation quantile for automatic lower-count filtering. |
| `--min-counts-auto-activate-below` | `1000` | Only activate automatic lower-count filtering when the activation quantile is below this UMI count floor. |
| `--max-pct-mt` | `5.0` | Remove cells with high mitochondrial percentage. |
| `--max-genes-mad` | `5.0` | Candidate upper-tail `n_genes_by_counts` cutoff: median plus `k * MAD`. |
| `--max-genes-quantile` | `0.999` | Candidate upper-tail `n_genes_by_counts` quantile cutoff. |
| `--max-counts-mad` | `5.0` | Candidate upper-tail `total_counts` cutoff: median plus `k * MAD`. |
| `--max-counts-quantile` | `0.999` | Candidate upper-tail `total_counts` quantile cutoff. |
| `--expected-doublet-rate` | `0.1` | Per-sample doublet fraction used to threshold SOLO scores. |

## Lower-Tail Filtering

Lower-tail filters remove cells with unusually low `total_counts`, which usually represent low-RNA droplets or damaged/low-quality cells that passed the initial droplet selection.

For `total_counts`, scOmnom can compute candidate lower cutoffs from:

- a MAD rule: median minus `k * MAD`, where MAD is the median absolute deviation
- a lower quantile rule
- an optional fixed UMI-count floor

By default:

- fixed `--min-counts` is off
- lower quantile filtering is off (`--min-counts-quantile none`)
- the candidate MAD cutoff uses `--min-counts-mad 5.0`
- the automatic cutoff only activates for samples whose `--min-counts-auto-activate-quantile 0.01` falls below `--min-counts-auto-activate-below 1000`

The effective lower cutoff is the stricter active cutoff among the fixed floor and the automatic candidate cutoff. The activation gate prevents ordinary samples from being filtered just because a formal MAD cutoff can be computed.

The rationale is to avoid imposing a universal hard UMI floor across datasets. A fixed floor can be appropriate when a dataset clearly needs it, but scOmnom defaults to a per-sample rule so ordinary samples are not over-filtered.

Examples:

```bash
# keep defaults
scomnom load-and-filter ...

# add an extra fixed floor
scomnom load-and-filter ... --min-counts 1000

# add an explicit lower quantile cutoff
scomnom load-and-filter ... --min-counts-quantile 0.05

# disable all automatic lower-count filtering
scomnom load-and-filter ... \
  --min-counts-mad none \
  --min-counts-quantile none

# keep auto cutoff settings but disable the activation gate
scomnom load-and-filter ... \
  --min-counts-auto-activate-quantile none \
  --min-counts-auto-activate-below none
```

## Upper-Tail Filtering

Upper-tail filters remove outlier cells with unusually high detected genes or total counts, which can reflect doublets, multiplets, or technical artifacts.

For each metric, scOmnom can compute candidate upper cutoffs from:

- a MAD rule: median plus `k * MAD`, where MAD is the median absolute deviation
- an upper quantile rule

The stricter available cutoff is used for each metric. For upper tails, "stricter" means the lower of the active candidate cutoffs.

Defaults:

- `--max-genes-mad 5.0`
- `--max-genes-quantile 0.999`
- `--max-counts-mad 5.0`
- `--max-counts-quantile 0.999`

## Mitochondrial Filtering

Cells with `pct_counts_mt` above `--max-pct-mt` are removed. The default is `5.0`.

This is intentionally simple and conservative. Datasets with expected high mitochondrial signal can override the threshold.

## Gene And Sample Guards

`--min-cells 3` removes genes detected in too few cells.

`--min-genes 500` removes cells with too few detected genes.

`--min-cells-per-sample 20` guards against retaining samples with too few cells after QC.

## Doublet Detection

Doublet detection is part of the load-and-filter stage after basic QC. scOmnom uses **SOLO**, via scVI, to assign a continuous `doublet_score` to each cell.

The scoring and calling are separated:

1. scOmnom trains one scVI model on the merged post-QC object.
2. SOLO scores are generated either globally or in planned blocks, depending on `--doublet-score-mode`.
3. scOmnom stores `adata.obs["doublet_score"]`.
4. scOmnom thresholds those scores **per sample** using `--expected-doublet-rate`.
5. Cells above the per-sample score threshold are marked in `adata.obs["predicted_doublet"]` and removed.

The default expected doublet rate is `--expected-doublet-rate 0.1`, meaning the highest-scoring 10% of cells in each sample are called as doublets. The inferred per-sample thresholds and observed rates are written to `doublets_per_sample.tsv` and stored in `adata.uns["doublet_calling"]`.

By default, `--doublet-score-mode auto` estimates the sparse operation size required for global SOLO scoring. If the estimate is at or below `--solo-sparse-nnz-limit`, scoring runs globally. If the estimate is larger, scOmnom switches to blocked SOLO scoring to reduce memory pressure. Blocks are planned from the preferred count matrix (`counts_cb`, then `counts_raw`, then `adata.X`) and respect sample boundaries where possible.

Tuning options:

| Option | Default | What it changes |
| --- | --- | --- |
| `--expected-doublet-rate` | `0.1` | Fraction of cells called as doublets per sample after SOLO scoring. Lower values are less aggressive; higher values remove more cells. |
| `--doublet-score-mode` | `auto` | SOLO scoring mode: `global`, `blocked`, or `auto`. In `auto`, large estimated sparse operations switch to blocked scoring. |
| `--solo-sparse-nnz-limit` | `1500000000` | Sparse operation estimate used by `auto` mode and by the block planner. Lower values make blocked scoring activate earlier and produce smaller blocks. |
| `--solo-max-cells-per-block` | `none` | Optional cap on cells per blocked SOLO scoring chunk. Use when memory pressure remains high despite the sparse operation limit. |
| `--apply-doublet-score` | off | Reuse an existing pre-doublet AnnData object with `doublet_score` and only reapply doublet thresholding. |
| `--apply-doublet-score-path` | `<out>/adata.merged.zarr` | Input object for `--apply-doublet-score`. |

Example forcing blocked scoring:

```bash
scomnom load-and-filter ... \
  --doublet-score-mode blocked \
  --solo-max-cells-per-block 50000
```

SOLO scoring metadata is stored in `adata.uns["solo_scoring"]`, including the requested mode, effective mode, count layer, global sparse estimate, fallback reason, and block summaries.

Recommended compute: run this module on a **GPU node** when possible because SOLO trains scVI/SOLO models. The rest of the filtering is mostly CPU-bound, but doublet scoring is the part that benefits strongly from GPU acceleration.
