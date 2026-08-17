# AnnData Operations

The `adata-ops` command group contains operations that import, reshape, relabel, or combine AnnData objects without rerunning the full upstream workflow.

Use these commands when you need to bring an external retained-cell AnnData object into scOmnom conventions, split a parent object into focused subsets, merge refined subset labels back into the parent, rename cluster labels, or combine multiple datasets into one AnnData object.

## Subsections

- [Import external AnnData](adata-ops/import.md)
- [Subset](adata-ops/subset.md)
- [Annotation merge](adata-ops/annotation-merge.md)
- [Manual renaming](adata-ops/rename.md)
- [Full multi-adata merge](adata-ops/merge.md)
