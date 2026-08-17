# Integrate

The `integrate` module performs batch correction across samples while preserving biological signal. It supports multiple integration algorithms, optional benchmarking with `scIB`, CellTypist-backed truth labels, and an advanced supervision strategy for `scANVI`.

By default, integration is non-destructive: embeddings, metrics, selected representations, UMAPs, and provenance are stored in the output `AnnData` object rather than replacing upstream count layers.

Recommended compute: use a **GPU node** when running the default method set. scVI/scANVI training is the expensive part and benefits strongly from GPU acceleration. CPU-only runs are possible for smaller datasets or when restricting methods to non-neural options such as Harmony, Scanorama, or BBKNN.

## Quick Entry Point

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --output-dir results/integrate/ \
  --celltypist-model Immune_All_Low.pkl
```

## Subsections

- [Supported integration methods](integrate/methods.md)
- [Intelligent scANVI supervision](integrate/scanvi-supervision.md)
- [scIB benchmarking and truth labels](integrate/benchmarking.md)
- [Annotated refinement](integrate/annotated-run.md)
- [Outputs](integrate/outputs.md)
