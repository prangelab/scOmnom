# scIB benchmarking and truth labels

Integration methods are benchmarked using `scIB`.

By default, benchmarking uses **CellTypist confident cell-level labels**. CellTypist is a supervised cell type classifier that assigns a label and independent logistic scores to each cell. scOmnom row-normalizes these scores for entropy, retains the raw top1-top2 margin, and excludes cells that do not pass both confidence gates from benchmarking.

CellTypist is used downstream as well: the same predictions and confidence mask are reused during `cluster-and-annotate` for biologically informed clustering diagnostics and cluster-level annotation. For consistent results, choose an appropriate CellTypist model for your dataset up front.

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

To select a specific model during integration:

```bash
scomnom integrate \
  --celltypist-model Immune_All_Low.pkl \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

You can override the truth labels via:

```bash
--scib-truth-label-key <key>
```

Valid options include:

* `celltypist` (default)
* `leiden`
* `final` / final cluster labels from `cluster-and-annotate`

This choice affects **benchmarking only** and does not influence integration itself.

## Embedding Selection

scOmnom applies a Pareto-aware hierarchy to the scaled scIB aggregate columns.
Every candidate must first improve scIB's scaled `Total` over `Unintegrated` by
more than a numerical tolerance. Eligible candidates are then considered in
this order:

1. embeddings that improve both `Bio conservation` and `Batch correction`;
2. embeddings that improve `Bio conservation` while accepting a batch trade-off;
3. embeddings that improve `Batch correction` while accepting a biological trade-off.

Within the first non-empty tier, scOmnom selects the highest `Total`, followed
by biological conservation, batch correction, and embedding name as
deterministic tie-breakers. `Unintegrated` is retained when no candidate
improves the aggregate score. This keeps the intended preference for balanced
improvement while preventing an integration with a net-worse scIB aggregate
from replacing the baseline.

The full decision table records each embedding's tier, component deltas,
aggregate delta, eligibility, selected status, policy, and tolerance. The
selector uses the scaled table produced by the current benchmark; raw and
scaled scIB aggregates are not interchangeable selection inputs.

## Edge Cases

scIB batch-correction metrics require at least two non-empty batch levels. If the selected `--batch-key` has only one level, scOmnom skips scIB batch benchmarking, selects `Unintegrated` when available (otherwise the first valid embedding), and writes an audit table under `integration_metrics/integration_single_batch_selection*.tsv`.

If benchmarking runs but the only scored representation is `Unintegrated`, scOmnom selects `Unintegrated` and writes `integration_no_candidate_selection*.tsv` instead of failing the run. This can happen when every requested integration method fails or no candidate embedding is available beyond the PCA baseline.

---
