# Cell-level vs pseudobulk: why both exist

* **Cell-level** tests are sensitive and can detect subtle expression shifts, but they can overweight large samples, inflate p-values on big datasets, and ignore replicate structure. By default, scOmnom caps marker output at 300 genes per cluster for cell-level runs.
* **Pseudobulk** respects sample-level independence, improves control of confounders via covariates, applies the `min_pct` filter, and yields more conservative inference, but requires enough replicates and loses single-cell resolution.

Running both can be informative: cell-level for discovery, pseudobulk for robustness.
For very large jobs, if `--run both` is unstable (for example due to memory pressure or native-thread failures), run two separate commands instead (`--run pseudobulk` and `--run cell`) into separate output folders and combine the interpretation afterward.

---
