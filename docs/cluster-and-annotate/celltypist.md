# CellTypist models

To list available CellTypist models:

```bash
scomnom cluster-and-annotate --list-models
```

Choose the model during `integrate` when possible so benchmarking, BISC bio-guidance, and cluster-level labels all share the same stored CellTypist predictions. `cluster-and-annotate` can still recompute predictions with `--force-celltypist-recompute` if the model needs to change after integration.

CellTypist exposes one logistic score per label; these scores are not a multiclass probability distribution and do not generally sum to one. scOmnom therefore row-normalizes a copy of the score matrix before calculating Shannon entropy, while retaining the raw top1-top2 score difference for the margin gate. The entropy cutoff is adaptive: it uses the larger of the configured baseline and the observed entropy quantile.

Cluster-level CellTypist labels are assigned only when enough confident cells are available both in absolute number and as a fraction of the cluster. The winning label must also exceed `--pretty-label-min-purity` (default `0.50`) among confident cells. Ties and insufficiently pure clusters remain `Unknown`. Per-cluster counts, coverage, winner and runner-up fractions, and the assignment reason are stored in the clustering round's annotation audit.

For model descriptions, training data, and usage guidance, see the official CellTypist documentation:

[https://www.celltypist.org](https://www.celltypist.org)

---
