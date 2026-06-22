# Input modes (identifying droplets containing real cells)

All input directory arguments point to a **single parent directory** containing **multiple per-sample entries**. Samples are discovered using glob patterns, which can be overridden if needed.

Default patterns used by `load-and-filter`:

- **Raw Cell Ranger matrices**: `*.raw_feature_bc_matrix`
- **Filtered Cell Ranger matrices**: `*.filtered_feature_bc_matrix`
- **CellBender outputs**: `*.cellbender_filtered.output`
  - expected accompanying files:
    - `.cellbender_out_filtered.h5`
    - `.cellbender_out_cell_barcodes.csv`

> **Important:** `scOmnom` only performs *de novo* identification of droplets containing real cells when **only raw counts** are provided. In all other modes, droplets are taken directly from **Cell Ranger** or **CellBender**.

### Preferred mode: raw counts + CellBender

- `--raw-sample-dir` points to a directory containing multiple per-sample folders matching `*.raw_feature_bc_matrix/`
- `--cellbender-dir` points to a directory containing multiple per-sample CellBender outputs matching `*.cellbender_filtered.output*`

Droplets containing real cells are taken from **CellBender**, while raw counts are used for QC comparisons and diagnostics.

### Alternative modes

#### CellBender-corrected counts only

- `--cellbender-dir`

Droplets containing real cells are taken directly from **CellBender**.

#### Cell Ranger filtered matrices

- `--filtered-sample-dir` points to a directory containing multiple per-sample folders matching `*.filtered_feature_bc_matrix/`

Droplets containing real cells are taken from **Cell Ranger**.

#### Raw matrices only (droplet identification by scOmnom)

- `--raw-sample-dir`

`scOmnom` identifies droplets containing real cells using a custom method combining **knee-point detection** and **Gaussian mixture modeling (GMM)**. The inferred droplets are used for downstream QC.

Technical sketch:

1. Compute total UMI counts for every barcode in each raw sample.
2. Sort barcodes by total UMI count to build a barcode-rank curve.
3. Fit a two-component GMM on `log10(total_counts + 1)` to separate low-count background barcodes from higher-count cell-like barcodes.
4. Compute the intersection between the background and cell-like Gaussian components.
5. Compute the knee of the barcode-rank curve.
6. Use a tightened geometric mean of the GMM and knee thresholds as the initial barcode cutoff.

This first-pass raw-only barcode caller currently does **not** expose separate CLI knobs for the GMM/knee cutoff itself. Users tune raw-only runs mainly through the downstream QC filters, which are applied after this initial barcode selection.

Common raw-only tuning options:

| Option | What it changes |
| --- | --- |
| `--min-genes` | Removes low-complexity cells after barcode calling. |
| `--min-counts` | Adds an explicit lower UMI-count floor. Off by default. |
| `--min-counts-mad` | Controls the automatic per-sample lower-count cutoff. |
| `--min-counts-quantile` | Adds an explicit lower quantile cutoff. Off by default. |
| `--min-counts-auto-activate-quantile` / `--min-counts-auto-activate-below` | Controls when the automatic lower-count cutoff activates. |
| `--max-pct-mt` | Controls mitochondrial filtering. |
| `--max-genes-mad` / `--max-genes-quantile` | Controls upper-tail filtering for detected genes. |
| `--max-counts-mad` / `--max-counts-quantile` | Controls upper-tail filtering for UMI counts. |
| `--expected-doublet-rate` | Tunes downstream SOLO doublet calling. |

See [Filtering Defaults And Rationale](filtering.md) for the default values and examples.

---
