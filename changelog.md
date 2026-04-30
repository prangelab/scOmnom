# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0: [dec 2025]
Implemented a working version of the load-and-filter and integrate modules.

## 0.1.1 [jan 2026]
Minor bug fixes

## 0.1.2 [13-01-2026]
Added 'versioning' to figure folders. Pipeline now doesnt overwrite previous output but keeps sequential '.roundN' folders containing figures per module. Also updated slurm example scripts to enable this.

## 0.1.3 [19-01-2026]
Added first stable version of the cluster-and-annotate module. It is now usable, but still under active development so subject to changes.
### Bug fixes
#### Gene annotation padding bug (mt / QC flags):
Fixed an issue during sample merging where boolean adata.var columns (e.g. mt, ribo, hb) could be upcast to float due to missing values introduced when padding to the union gene set, causing Zarr write failures in some environments.
#### Nullable string categories write failure:
Resolved a crash when writing padded Zarr files caused by pandas nullable string dtypes (e.g. StringArray / ArrowStringArray) in adata.var categorical columns (notably gene_ids), by coercing to storage-safe string representations compatible with older anndata versions.

## 0.1.4 [20-01-2026]
Improved layout of reports.

## 0.1.5 [20-01-2026]
Optimised resolution sweep by removing full ARI table computation. Only adjacent ARIs are needed.
Fixed bug preventing color values to be read from adata.

## 0.1.6 [23-1-2026]
Fixed bug causing violin labels to be drawn horizontal.

## 0.1.7 [27-1-2026]
Fixed bug were cluster-and-annotate was using raw counts instead of cellbender counts.

## 0.1.8 [29-01-2026]
Added first working version of marker calling in the DE module. Module is still under construction, do not use yet.

## 0.1.9 [29-01-2026]
Bug-fix integrate --annotated-run was trying to pick a new best embedding instead of just plotting the annotated scANVI UMAP.

## 0.1.10 [30-01-2026]
Changed reporting behaviour. It now saves a report per round and for each format.

## 0.1.11 [05-02-2026]
Fixed some cluster-and-annotate plots. DE module is now feature complete, but still in testing phase.

## 0.2.0 [06-03-2026]
DE and DA modules are now finished and operational. Changed file output format to compressed and archived *.zarr.tar.zst format. This prevents inode exhaustion. Archives are auto detected at load. Implemented subsetting in the new adata-ops module. Subsetted modules can be reclustered by feeding them to cluster-and-annotate, which has been updated to always recalculate PCs and HVGs to mathc the subsetted data. Added manual cluster renaming support, later moved to `adata-ops rename` in 0.3.1. Added plot-only modes to markers-and-de allowing to refresh plots (eg with the newly renamed idents) without redoing all the computations.

## 0.2.1 [10-03-2026]
Added API epxosure for plotting functions. Also added example notebooks for API usage. Changed plotting engine so that plots create plotArtifactgs, which th ecaller can choose to plot or save to disk. this separates output generation from the plottin gfuciotns and enbale API versus CLI usage of the same functions.

## 0.3.0 [30-03-2026]
Extended adata-ops with annotation-merge, allowing refined subset annotations to be written back into a parent object as a new subset_annotation round or merged into an existing one. Standardized shallow round creation across rename, compaction, projection, and annotation-merge. Added merge-annotation UMAP and cluster QC plots. Split environment files by platform/architecture, fixed an out-of-memory issue in save_dataset, and fixed cluster ordering in violin plots.

## 0.3.1 [31-03-2026]
Moved manual renaming from cluster-and-annotate into adata-ops as `adata-ops rename`, so subset refinement now lives fully under adata-ops. Fixed rename logging and summaries so they report the correct parent round id.

## 0.3.2 [02-04-2026]
Added an option to rename to collapse same name clusters into broader categories.

## 0.3.3 [08-04-2026]
Made markers-and-de outputs round-aware in both result folder names and default AnnData output names, so concurrent runs on different rounds no longer collide. Also removed redundant nested DE/DA/markers directory layers inside those round-namespaced output folders.

## 0.4.0 [13-04-2026]
Added `markers-and-de enrichment` as a dedicated command group with:
- `enrichment cluster` for round-native enrichment on AnnData rounds (including condition-aware grouping syntax such as `sex:MASLD`)
- `enrichment de` for enrichment directly from exported DE tables without loading AnnData
- `enrichment module-score` for custom module scoring with Scanpy backend plus an AUCell backend option

Added gene prefiltering support via `--gene-filter` across enrichment and DE execution paths (separate from plot-only `--plot-gene-filter`), with strict failure behavior for invalid filters and improved gene-annotation handling for filters based on `gene_type`, `gene_chrom`, and `gene_id`.

Extended public API exposure for enrichment and module-score workflows and added notebook-facing plotting support for decoupler outputs.

Improved save robustness and memory behavior around Zarr serialization by:
- introducing sidecar storage for complex/large `uns` payloads with automatic rehydration on load
- hardening object-dtype handling for sidecar payloads and Zarr metadata coercion
- adding save-stage memory checkpoints in `save_dataset` to make HPC OOM diagnostics explicit
- applying DE uns-pruning before enrichment save to reduce carried memory footprint

## 0.4.1 [13-04-2026]
Documentation and release housekeeping update:
- added README section for `adata-ops annotation-merge` (subset back-merge workflow, key flags, and examples)
- corrected changelog ordering for the `0.3.2` / `0.3.3` entries

## 0.5.0 [28-04-2026]
Stability and robustness release focused on DE execution and dataset serialization:
- fixed within-cluster DE gene-filter indexing mismatch that could crash conditional runs with `IndexError` when filtered gene sets were used
- added `--target-groups` support for within-cluster DE so users can restrict analysis to selected populations
- capped and unified DE worker control via `--max-workers` for both pseudobulk and cell-level phases to reduce native-thread instability on large HPC runs
- improved DE report run-folder detection to support round-aware directory names like `de_<round_id>_roundN` (case-insensitive and backward-compatible)

Serialization and save-path hardening:
- hardened sidecar writing for nested object dtypes (including structured arrays with object fields) to avoid Zarr object-dtype resolution failures
- made sidecar payload failures non-fatal during save so core dataset writes still complete, with warning-level diagnostics for skipped payloads
- reduced `save_dataset` log noise by moving memory checkpoint and per-payload sidecar write lines to debug-level output

Documentation and CLI ergonomics:
- documented a practical fallback strategy for large DE workloads: split `--run both` into separate `--run pseudobulk` and `--run cell` jobs when needed
- added CLI support to force CellTypist recomputation with explicit reuse/refresh control

## 0.6.0 [30-04-2026]
HPC workflow standardization release:
- DE execution stability fix: hardened within-cluster pseudobulk scheduling and thread behavior for HPC runs, preventing recurrent segfault/oversubscription patterns seen in large multi-contrast jobs; documented recommended worker/thread settings in the SLURM examples
- added a generic SLURM template (`slurm/scomnom_template.job`) with configurable command slot, scratch staging, figure-tree carry-over, strict shell safety (`set -euo pipefail`), and BLAS/OpenMP oversubscription guards (`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/IGRAPH` set to 1)
- added numbered module SLURM examples covering the main pipeline and optional subset branch:
  - `scomnom_1_load_and_filter.job`
  - `scomnom_2_integrate.job`
  - `scomnom_3_cluster_and_annotate.job`
  - `scomnom_4_integrate_annotated_run.job`
  - `scomnom_4a_subset.job`
  - `scomnom_4b_annotation_merge.job`
  - `scomnom_5_rename.job`
  - `scomnom_5a_rename_archetypes.job`
  - `scomnom_6_de.job`
  - `scomnom_7_da.job`
  - `scomnom_8_enrichment_cluster.job`
- removed legacy unnumbered SLURM examples to avoid duplication and drift
- updated README workflow documentation to reflect the optional subset/merge branch and the expanded numbered SLURM examples list
