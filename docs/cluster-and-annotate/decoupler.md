# Decoupler configuration

Decoupler is enabled by default (`run_decoupler = True`) and is used for:

* cluster-level pseudobulk activity inference
* biological diagnostics
* compaction decisions

**Defaults:**

* aggregation: **mean**
* method: **consensus** (`ulm`, `mlm`, `wsum`)
* databases **MSigDB, DoRothEA, PROGENy**: (`HALLMARK`, `REACTOME`)
* minimum targets per source: **5**

All decoupler settings — including databases, methods, thresholds, and organisms — are fully configurable.

For details on decoupler methods and resources, see:

[https://decoupler-py.readthedocs.io](https://decoupler-py.readthedocs.io)

---
