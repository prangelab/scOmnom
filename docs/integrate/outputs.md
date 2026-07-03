# Outputs

Each integration run produces:

* integrated `AnnData` object (`.zarr`, optionally `.h5ad`)
* scIB metric tables (raw + scaled)
* single-batch or no-candidate selection audit tables when scIB cannot compare candidate embeddings
* UMAPs and diagnostics saved under:

```
figures/
├── png/
│   └── integration_roundN/
└── pdf/
    └── integration_roundN/
```

Each run gets its own `integration_round*` subdirectory to keep results reproducible and auditable. This is an output run counter on disk, not an AnnData clustering round; see [Output Organization](../output-organization.md).

Integration metric and audit tables are written under `integration_metrics/`. Standard multi-batch benchmarking writes `integration_metrics_raw*.tsv` and `integration_metrics_scaled*.tsv`. Single-batch runs write `integration_single_batch_selection*.tsv`; runs with no candidate beyond `Unintegrated` write `integration_no_candidate_selection*.tsv`.

---
