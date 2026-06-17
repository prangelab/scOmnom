# Integrate

The `integrate` module performs batch correction across samples while preserving biological signal. It supports multiple integration algorithms, optional benchmarking with `scIB`, and an advanced supervision strategy for `scANVI`.

By default, integration is non-destructive: embeddings, metrics, selected representations, UMAPs, and provenance are stored in the output `AnnData` object rather than replacing upstream count layers.

## Quick Entry Point

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/
```

## Subsections

- [Supported integration methods](integrate/methods.md)
- [Intelligent scANVI supervision](integrate/scanvi-supervision.md)
- [scIB benchmarking and truth labels](integrate/benchmarking.md)
- [Annotated refinement](integrate/annotated-run.md)
- [Outputs](integrate/outputs.md)
- [Relationship to cluster-and-annotate](integrate/cluster-relationship.md)
