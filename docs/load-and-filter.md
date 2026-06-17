# Load and filter (QC + preprocessing)

This stage performs:

- identification of droplets containing real cells **only in raw-only mode**
- quality control filtering
- HVG selection
- doublet detection
- dataset merging across samples

### Quick entry point

Example (preferred mode, raw + CellBender):

```bash
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/load_and_filter/
```

Default lower-count QC behavior:

* `min_genes` remains a fixed threshold.
* `total_counts` is filtered per sample by default using:
  * `--min-counts-mad 5.0`
* The automatic lower-count filter only activates for samples whose:
  * `--min-counts-auto-activate-quantile 0.01`
  * falls below `--min-counts-auto-activate-below 1000`
* `--min-counts` remains available as an additional fixed floor.
* `--min-counts-quantile` remains available as an optional stricter lower-bound component, but stays off by default.
* The intended default policy is:
  * fixed `min_counts` stays off
  * fixed lower quantile filtering stays off
  * automatic lower-count filtering stays on
  * defaults clean up obvious low-count noise in ordinary datasets
  * stronger fixed intervention is available when a dataset clearly needs it
* Use `--min-counts-mad none`, `--min-counts-quantile none`,
  `--min-counts-auto-activate-quantile none`, and/or
  `--min-counts-auto-activate-below none` to disable components.

Examples:

```bash
# keep defaults
scomnom load-and-filter ...

# add an extra fixed floor
scomnom load-and-filter ... --min-counts 1000

# add an explicit lower quantile cutoff
scomnom load-and-filter ... --min-counts-quantile 0.05

# disable the quantile component
scomnom load-and-filter ... --min-counts-quantile none

# disable all automatic lower-count filtering
scomnom load-and-filter ... \
  --min-counts-mad none \
  --min-counts-quantile none

# keep auto cutoff settings but disable the activation gate
scomnom load-and-filter ... \
  --min-counts-auto-activate-quantile none \
  --min-counts-auto-activate-below none
```

### What it does

* Discovers samples from the provided parent directories (glob patterns can be overridden).
* Resolves droplets from CellBender or Cell Ranger, or infers droplets in raw-only mode.
* Runs QC filters, including fixed `min_genes`, per-sample lower-count filtering on `total_counts`, upper-tail trimming, HVG selection, and doublet detection.
* Merges per-sample AnnData into a single dataset with consistent metadata.

### Outputs

* merged AnnData object (`.zarr`, optionally `.h5ad`)
* QC figures under `figures/`
* `load-and-filter.log`

---
