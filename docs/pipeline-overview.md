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
│  • CellTypist annotation                                   │
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

---
