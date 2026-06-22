# Outputs

Each integration run produces:

* integrated `AnnData` object (`.zarr`, optionally `.h5ad`)
* scIB metric tables (raw + scaled)
* UMAPs and diagnostics saved under:

```
figures/
├── png/
│   └── integration_roundN/
└── pdf/
    └── integration_roundN/
```

Each run gets its own `integration_round*` subdirectory to keep results reproducible and auditable. This is an output run counter on disk, not an AnnData clustering round; see [Output Organization](../output-organization.md).

---
