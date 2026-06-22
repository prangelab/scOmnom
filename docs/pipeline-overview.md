# Pipeline overview

```
┌────────────────────────────────────────────────────────────┐
│                        Input data                           │
│                                                            │
│  raw_feature_bc_matrix / filtered_feature_bc_matrix /      │
│  CellBender outputs                                        │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                  load-and-filter                           │
│                                                            │
│  • Identify droplets containing real cells                 │
│    (only if raw-only input)                                │
│  • QC filtering                                            │
│  • HVG selection                                           │
│  • Doublet detection                                       │
│                                                            │
│  → merged AnnData (Zarr) + QC figures                      │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                        integrate                           │
│                                                            │
│  • Batch correction (scVI, BBKNN, Harmony, …)              │
│  • CellTypist-backed truth labels                          │
│  • scIB benchmarking                                       │
│                                                            │
│  → integrated AnnData + figures                            │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                  cluster-and-annotate                      │
│                                                            │
│  • BISC resolution selection                               │
│  • Cluster labels from CellTypist predictions              │
│  • Decoupler activities + optional compaction              │
│                                                            │
│  → annotated AnnData + figures                             │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│          integrate --annotated-run (optional)              │
│                                                            │
│  • scANVI refinement using final labels                    │
│  • clean UMAPs / latent space                              │
│                                                            │
│  → projected AnnData + figures                             │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                    markers-and-de                          │
│                                                            │
│  • markers (cluster-vs-rest)                               │
│  • de (within-cluster contrasts)                           │
│  • da (composition)                                        │
│                                                            │
│  → tables + reports + updated AnnData                      │
└────────────────────────────────────────────────────────────┘
```

scOmnom keeps two kinds of provenance as the pipeline runs:

* Output folders use run counters such as `integration_round1` or `de_r3_manual_rename_round1` so figures, reports, and tables from separate runs are not overwritten.
* AnnData clustering rounds live inside `adata.uns["cluster_rounds"]` and define which biological label state downstream modules should use.

See [Output Organization](output-organization.md) for the distinction between filesystem output rounds and internal clustering rounds.

Reports are part of the same pipeline output layer: modules write self-contained HTML reports alongside figures and tables. See [Reporting](reporting.md) for the general report convention.

## Optional loops

The straight-through path above is the default route. Two loops are common once first-pass clusters exist:

* **Annotated refinement:** rerun `integrate --annotated-run` after `cluster-and-annotate` when final labels should guide a cleaner scANVI embedding. This is optional and should be treated as label-aware projection, not as fresh unbiased clustering evidence.
* **Subset refinement:** after first-pass marker review, split selected populations with `adata-ops subset`, rerun each subset through `integrate`, `cluster-and-annotate`, and optional `integrate --annotated-run`, then merge refined labels back into the parent with `adata-ops annotation-merge`.

These loops create new output folders and, when labels change, new AnnData clustering rounds rather than overwriting the earlier state.

---
