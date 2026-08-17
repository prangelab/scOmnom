# CellTypist models

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

Choose the model during `integrate` when possible so benchmarking, BISC bio-guidance, and cluster-level labels all share the same stored CellTypist predictions. `cluster-and-annotate` can still recompute predictions with `--force-celltypist-recompute` if the model needs to change after integration.

For model descriptions, training data, and usage guidance, see the official CellTypist documentation:

[https://www.celltypist.org](https://www.celltypist.org)

---
