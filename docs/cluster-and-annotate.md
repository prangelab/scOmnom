# Cluster And Annotate

The `cluster-and-annotate` module turns an integrated embedding into a round-aware biological annotation state. It covers BISC-guided Leiden resolution selection, CellTypist-backed annotation, optional enrichment activities, and compaction of redundant clusters.

This module is where scOmnom's cluster-round structure becomes central: each run creates or updates round metadata in `adata.uns["cluster_rounds"]`, while active labels and pretty labels are referenced from the selected round.

## Quick Entry Point

```bash
scomnom cluster-and-annotate \
  --input-path results/integrate/adata.integrated.zarr \
  --output-dir results/cluster_and_annotate/ \
  --celltypist-model Immune_All_Low.pkl
```

## Subsections

- [CellTypist models](cluster-and-annotate/celltypist.md)
- [Reproducibility and non-destructive design](cluster-and-annotate/reproducibility.md)
- [BISC](cluster-and-annotate/bisc.md)
- [Cluster annotation](cluster-and-annotate/cluster-annotation.md)
- [Compaction](cluster-and-annotate/compaction.md)
- [Decoupler configuration](cluster-and-annotate/decoupler.md)
- [Outputs](cluster-and-annotate/outputs.md)
- [Reporting](cluster-and-annotate/reporting.md)

For subsetting, annotation merge, manual renaming, and multi-dataset merge workflows, see [AnnData Operations](adata-ops.md).
