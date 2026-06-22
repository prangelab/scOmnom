# Cell-level vs pseudobulk

## Pseudobulk first

For replicate-aware differential expression, prefer pseudobulk whenever there are enough biological samples. It aggregates cells within each sample and population, then tests sample-level libraries. That keeps the sample, donor, or replicate as the unit of inference instead of treating thousands of cells from the same individual as independent biological replicates.

This matters most in large single-cell datasets: more cells give better estimates of the expression profile within a sample, but they do not create more independent samples. Pseudobulk keeps that distinction explicit, gives the model a natural place for sample-level covariates, and makes effect sizes and FDR better aligned with the usual biological question: whether a population changes reproducibly across samples.

In scOmnom, pseudobulk DE uses the active clustering round to create sample-by-population count summaries, applies the configured gene filters, and runs the replicate-level tests from there. Pseudobulk is guarded when there are fewer than 6 unique samples, because below that point the model has too little replicate structure to estimate stable group-level evidence.

## Cell-level as a secondary engine

Cell-level tests are the older and more permissive companion. They are still useful for:

* discovery and quick triage;
* low-sample-count studies where pseudobulk cannot run;
* exploratory checks before investing in full pseudobulk jobs;
* sensitive screens to compare against pseudobulk-supported hits.

Interpret them as exploratory or supporting evidence. Cell-level tests use cells as statistical observations, so they can overweight large samples and make large datasets look more certain than the biological replicate structure really supports. By default, scOmnom caps marker output at 300 genes per cluster for cell-level runs.

## Running both

`--run both` is useful when you want pseudobulk as the primary evidence layer and cell-level results as a sensitive companion. For very large jobs, or when memory pressure/native-thread failures make a combined run unstable, run two separate commands instead:

```bash
scomnom markers-and-de de ... --run pseudobulk
scomnom markers-and-de de ... --run cell
```

Write those runs to separate output folders and combine the interpretation afterward.

---
