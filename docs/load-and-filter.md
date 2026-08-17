# Load And Filter

The `load-and-filter` module builds the initial merged AnnData object from raw, filtered, and/or CellBender-corrected 10x inputs.

Its scope is:

- sample discovery from parent input directories
- droplet/cell selection according to the available input mode
- per-sample QC filtering
- highly variable gene selection
- doublet detection
- merge into one scOmnom AnnData object

Recommended compute: use a **GPU node** when possible. Most of this module is CPU-friendly, but SOLO doublet detection trains scVI/SOLO models and is much faster on GPU. For large datasets, `--doublet-score-mode auto` can switch SOLO scoring from global prediction to blocked prediction to reduce memory pressure.

## Quick Entry Point

Preferred mode, using raw counts plus CellBender outputs:

```bash
scomnom load-and-filter \
  --raw-sample-dir path/to/all_raw_samples/ \
  --cellbender-dir path/to/all_cellbender_outputs/ \
  --metadata-tsv path/to/metadata.tsv \
  --batch-key sample_id \
  --out results/load_and_filter/
```

## Subsections

- [Input modes](load-and-filter/input-modes.md)
- [Filtering defaults and rationale](load-and-filter/filtering.md)
- [Outputs](load-and-filter/outputs.md)
