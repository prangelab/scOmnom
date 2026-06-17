# NicheNet CCC

`scomnom markers-and-de ccc nichenet` runs a sender-focused NicheNet analysis for one receiver cluster or, by default, all receiver clusters in batch mode. It is intended to complement LIANA rather than replace it: LIANA gives candidate sender-receiver structure, whereas NicheNet prioritizes ligands that best explain a receiver transcriptional program.

**Current scope**

The first `scOmnom` NicheNet implementation is intentionally narrow:

* receiver selection via `--receiver-cluster`
  * `--receiver-cluster all` is the default and batches over every cluster
  * any specific raw cluster id or pretty label runs only that receiver
* receiver gene set from either:
  * `--gene-list-file`, or
  * receiver-cluster DE between exactly two `--compare-level` values
* optional cross-tissue sender/receiver restriction via:
  * `--dataset-key`
  * `--source-level`
  * `--target-level`
* optional expression input mode for sender/receiver expressed-gene filtering:
  * `--input-mode counts`
  * `--input-mode lognorm`

If `--dataset-key` is omitted, scOmnom runs the normal within-object sender-focused NicheNet workflow.

Use `--input-mode lognorm` when you want NicheNet to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` before computing sender and receiver expressed-gene sets. This follows the same normalization convention used by LIANA and MEBOCOST.

**Condition key syntax**

NicheNet currently supports:

* `A`: compare two levels of `A` within the full object
* `A@B`: compare two levels of `A` separately within each level of `B`

**Runtime requirements**

This backend shells out to `Rscript` and expects the R package `nichenetr` to be installed. In the current `v1` implementation, scOmnom uses the official human NicheNet model files downloaded from the NicheNet Zenodo-hosted URLs at runtime, so internet access is required the first time the analysis is run.

The environment YAML files include the required R runtime and helper packages, but `nichenetr` itself is currently not pinned there as a Conda package because the available Conda builds are stale and not consistently cross-platform.

scOmnom now expects `nichenetr` in a local non-synced cache library. On macOS the default is `~/Library/Caches/scOmnom/r-libs/nichenet/`; on Linux it is `~/.cache/scOmnom/r-libs/nichenet/`. If it is missing, `ccc nichenet` errors with an explicit hint and suggests `--install-missing-r-deps`. If you want scOmnom to bootstrap that library automatically, run:

```bash
scomnom markers-and-de ccc nichenet \
  ... \
  --install-missing-r-deps
```

To install it manually into the same cache library, run:

```bash
tmpdir=$(mktemp -d) && \
R_LIBS_USER="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
R_LIBS="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
Rscript -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE, showWarnings=FALSE); .libPaths(c(Sys.getenv("R_LIBS_USER"), .Library, .Library.site)); pkgs <- c("DiceKriging", "emoa", "fdrtool", "mlrMBO"); missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing_pkgs)) install.packages(missing_pkgs, repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))' && \
git clone --depth 1 https://github.com/saeyslab/nichenetr.git "$tmpdir/nichenetr" && \
mkdir -p "$HOME/Library/Caches/scOmnom/r-libs/nichenet" && \
R_LIBS_USER="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
R_LIBS="$HOME/Library/Caches/scOmnom/r-libs/nichenet" \
R CMD INSTALL --no-test-load -l "$HOME/Library/Caches/scOmnom/r-libs/nichenet" "$tmpdir/nichenetr"
```

**Example**

```bash
scomnom markers-and-de ccc nichenet \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --dataset-key tissue \
  --source-level liver \
  --target-level fat \
  --condition-key masld_status@timepoint \
  --condition-value 5_years_post \
  --compare-level better \
  --compare-level worse
```

**Outputs**

* Figures: `figures/<fmt>/ccc_nichenet_<round>_roundN/`
* Tables: `tables/ccc_nichenet_<round>_roundN/`
* Key tables:
  * `nichenet_ligand_activity.tsv`
  * `nichenet_ligand_target_links.tsv`
  * `nichenet_ligand_receptor_links.tsv`
  * `nichenet_receiver_de.tsv`

---
