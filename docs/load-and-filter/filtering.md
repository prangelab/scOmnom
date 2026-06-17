# Filtering Defaults And Rationale

`load-and-filter` applies conservative QC filters before samples are merged. The defaults are designed to remove obvious low-quality cells while avoiding strong dataset-specific assumptions.

## Summary Of Default Filters

| Filter | Default | Purpose |
| --- | --- | --- |
| `--min-cells` | `3` | Remove genes detected in too few cells. |
| `--min-genes` | `500` | Remove cells with too few detected genes. |
| `--min-counts` | `none` | Optional fixed lower UMI-count floor; off by default. |
| `--min-counts-mad` | `5.0` | Automatic per-sample lower `total_counts` cutoff. |
| `--min-counts-quantile` | `none` | Optional lower quantile cutoff; off by default. |
| `--min-counts-auto-activate-quantile` | `0.01` | Activation quantile for automatic lower-count filtering. |
| `--min-counts-auto-activate-below` | `1000` | Only activate automatic lower-count filtering when the activation quantile is below this UMI count floor. |
| `--max-pct-mt` | `5.0` | Remove cells with high mitochondrial percentage. |
| `--max-genes-mad` | `5.0` | Upper-tail filter for `n_genes_by_counts`. |
| `--max-genes-quantile` | `0.999` | Upper quantile cap for `n_genes_by_counts`. |
| `--max-counts-mad` | `5.0` | Upper-tail filter for `total_counts`. |
| `--max-counts-quantile` | `0.999` | Upper quantile cap for `total_counts`. |
| `--expected-doublet-rate` | `0.1` | Expected doublet rate used by doublet detection. |

## Lower-Count Filtering

The lower-count filter targets obvious low-count noise on `total_counts`.

By default:

- fixed `--min-counts` is off
- lower quantile filtering is off
- automatic lower-count filtering uses `--min-counts-mad 5.0`
- automatic lower-count filtering only activates for samples whose `--min-counts-auto-activate-quantile 0.01` falls below `--min-counts-auto-activate-below 1000`

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

For each metric, scOmnom computes candidate cutoffs from:

- a MAD rule: median plus `k * MAD`
- an upper quantile rule

The stricter available cutoff is used.

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

Doublet detection is part of the load-and-filter stage after basic QC. The default expected doublet rate is `--expected-doublet-rate 0.1`.

Doublet behavior is controlled separately from the basic filtering thresholds so users can tune QC stringency and doublet handling independently.
