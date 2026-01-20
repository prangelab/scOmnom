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