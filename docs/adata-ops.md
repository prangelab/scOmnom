# AnnData Operations

The `adata-ops` command group contains operations that reshape, relabel, or combine existing scOmnom AnnData objects without rerunning the full upstream workflow.

Use these commands after clustering and annotation when you need to split a parent object into focused subsets, merge refined subset labels back into the parent, rename cluster labels, or combine multiple datasets into one AnnData object.

## Subsections

- [Subset](adata-ops/subset.md)
- [Annotation merge](adata-ops/annotation-merge.md)
- [Manual renaming](adata-ops/rename.md)
- [Full multi-adata merge](adata-ops/merge.md)
