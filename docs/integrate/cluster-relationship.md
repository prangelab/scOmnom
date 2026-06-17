# Relationship to `cluster-and-annotate`

The recommended high-level workflow is:

```
load-and-filter
   ↓
integrate
   ↓
cluster-and-annotate
   ↓
markers-and-de markers (recommended)
   ↓
integrate --annotated-run (optional)
   ↓
adata-ops rename (optional, often used)
   ↓
markers-and-de de
   ↓
markers-and-de da
   ↓
markers-and-de enrichment cluster (optional)
```

Optional subset loop after markers and before final rename/DE/DA:

```
parent projected object
   ↓
markers-and-de markers
   ↓
adata-ops subset
   ↓
[subset object(s)] integrate -> cluster-and-annotate -> integrate --annotated-run
   ↓
adata-ops annotation-merge (back into parent)
   ↓
adata-ops rename
```

The secondary integration step is **purely optional** and should be used only when:

* clustering and annotation are finalized
* improved visualization or refined latent structure is desired

---
