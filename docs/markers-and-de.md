# Markers And DE

The `markers-and-de` command group covers downstream biological interpretation after clustering and annotation. It includes marker discovery, within-cluster differential expression, differential abundance, enrichment, module scoring, and cell-cell communication workflows.

The subcommands are intentionally grouped together because they share round selection, label handling, plotting/reporting conventions, and output provenance.

## Quick Entry Points

```bash
# Marker discovery
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.zarr

# Within-cluster DE
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.markers.zarr \
  --condition-key condition

# Differential abundance
scomnom markers-and-de da \
  --input-path results/adata.clustered.annotated.markers.de.zarr \
  --condition-key condition

# Cell-cell communication
scomnom markers-and-de ccc liana \
  --input-path results/adata.clustered.annotated.markers.de.zarr \
  --condition-key condition
```

## Subsections

- [Markers](markers-and-de/markers.md)
- [Within-cluster DE](markers-and-de/de.md)
- [Cell-level vs pseudobulk](markers-and-de/cell-vs-pseudobulk.md)
- [Differential abundance](markers-and-de/da.md)
- [Enrichment](markers-and-de/enrichment.md)
- [Cell-cell communication](markers-and-de/ccc.md)
- [LIANA CCC](markers-and-de/liana.md)
- [NicheNet CCC](markers-and-de/nichenet.md)
- [MEBOCOST CCC](markers-and-de/mebocost.md)
