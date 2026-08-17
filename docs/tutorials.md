# Tutorials

These tutorials are companion walkthroughs for running scOmnom on compact public PBMC datasets. They are meant to sit beside the reference manual: use the tutorials for a concrete end-to-end route, and use the manual pages for option-level detail.

![scOmnom tutorial workflow](tutorials/panels/graphical_abstract_draft.png)

## Available Tutorials

| Tutorial | Focus | Continue to |
| --- | --- | --- |
| [PBMC10k Data Processing](tutorials/data-processing-pbmc10k.md) | CellBender-aware input handling, load/filter, integration, clustering, subset refinement, merge-back, custom annotation layers, and markers. | Start here for the core processing workflow. |
| [Kang IFN-beta PBMC DE](tutorials/kang-ifnb-de.md) | Replicate-aware DE, enrichment, differential abundance, and condition-split LIANA CCC. | Use after the processing tutorial when you want condition-aware downstream analysis. |

The PBMC10k tutorial includes small helper scripts under [`tutorials/code/`](tutorials/code/select_tnk_refinement_subset.py). These scripts are intentionally lightweight wrappers around scOmnom AnnData IO and annotation conventions.

The panel images are first-pass online guide composites. They are included to make the workflow inspectable while the text and final tutorial assets continue to mature.

---
