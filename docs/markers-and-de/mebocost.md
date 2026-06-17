# MEBOCOST CCC

`scomnom markers-and-de ccc mebocost` runs MEBOCOST metabolite-mediated cell-cell communication inference on an annotated AnnData object and writes tables, figures, and an `adata.uns["markers_and_de"]["ccc"]["mebocost"]` payload.

**Current scope**

The first `scOmnom` MEBOCOST implementation is intentionally focused on the core communication table workflow:

* per-run MEBOCOST communication inference from a single AnnData subset
* optional `A` and `A@B` condition expansion, aligned with the LIANA syntax
* optional cross-tissue sender/receiver restriction via:
  * `--dataset-key`
  * `--source-level`
  * `--target-level`
* optional MEBOCOST expression input mode via:
  * `--input-mode counts`
  * `--input-mode lognorm`
* summary outputs and baseline plots for:
  * top metabolite-sensor events
  * metabolite super-class and sensor-annotation summary tables/barplots
  * source-target event-count heatmaps
  * source-target mean-score heatmaps
  * per-source and per-target cluster breakdown barplots
  * per-source and per-target metabolite super-class / sensor-annotation summaries
  * condition-split compare plots for source-target heatmaps, top events, and annotation classes when `--compare-level` defines a multi-run bucket
* saved communication tables are enriched with HMDB metabolite classes and MEBOCOST sensor annotations when available

If `--dataset-key` is omitted, scOmnom runs the normal within-object MEBOCOST workflow.

Use `--input-mode lognorm` when you want MEBOCOST to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` instead of running directly on raw counts. This creates `adata.layers["lognorm_counts_cb"]` or `adata.layers["lognorm_counts_raw"]` on demand and then uses that layer for MEBOCOST.

**Runtime requirements**

This backend uses the Python package `MEBOCOST` from the Chen Lab GitHub repository. The upstream Python package does not bundle the metabolite database/config files that the constructor expects, so scOmnom also maintains a local MEBOCOST resource cache and generated `mebocost.conf` under the user cache directory.

If either the Python package or the upstream resource bundle is missing, `ccc mebocost` errors with an explicit install hint and suggests `--install-missing-python-deps`.

To let scOmnom bootstrap it automatically, run:

```bash
scomnom markers-and-de ccc mebocost \
  ... \
  --install-missing-python-deps
```

**Example**

```bash
scomnom markers-and-de ccc mebocost \
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

* Figures: `figures/<fmt>/ccc_mebocost_<round>_roundN/`
* Tables: `tables/ccc_mebocost_<round>_roundN/`
* Key tables:
  * `mebocost_commu_res.tsv`
  * `mebocost_sig_res.tsv`
  * `mebocost_source_target_summary.tsv`

**Focused donor-level rescoring**

`scomnom markers-and-de ccc mebocost-paired` rescales a focused candidate event table at donor or sample level instead of rerunning full pooled discovery. It is meant for downstream paired analyses once pooled MEBOCOST has already identified candidate metabolite-sensor routes.

**Current scope**

* reads a candidate event file such as `mebocost_sig_res.tsv`
* supports the same `A` and `A@B` condition-subset syntax as pooled MEBOCOST
* supports cross-tissue sender/receiver restriction via `--dataset-key`, `--source-level`, and `--target-level`
* computes donor/sample-level event scores for:
  * `--score-method mebocost-metabolite-sensor`
  * `--score-method associated-gene-proxy`
* paired rescoring defaults to:
  * `--min-sender-cells 5`
  * `--min-receiver-cells 5`
  * `--min-scored-donors-per-group 3`
* always writes missingness summary tables so all-missing donor splits are easy to diagnose
* optional focused figures for:
  * paired route summaries
  * sample-by-route heatmaps
  * top condition effects from the paired group-effect table
* writes focused donor-level tables and stores them under `adata.uns["markers_and_de"]["ccc"]["mebocost_paired_rescore"]`

**Example**

```bash
scomnom markers-and-de ccc mebocost-paired \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --candidate-events results/tables/ccc_mebocost_r5_round1/condition__masld_status_at_timepoint/timepoint=5_years_post/worse/mebocost_sig_res.tsv \
  --dataset-key tissue \
  --source-level vfat \
  --target-level liv \
  --pairing-key sample_id \
  --condition-key sex@MASLD \
  --condition-value yes \
  --input-mode lognorm
```

**Outputs**

* Figures: `figures/<fmt>/ccc_mebocost_<round>_roundN/paired_rescore/<condition>/`
* Tables: `tables/ccc_mebocost_<round>_roundN/paired_rescore/<condition>/`
* Key tables:
  * `mebocost_paired_event_scores.tsv`
  * `mebocost_paired_route_scores.tsv`
  * `mebocost_paired_event_missingness.tsv`
  * `mebocost_paired_route_missingness.tsv`
  * `mebocost_paired_group_effects.tsv`
  * `mebocost_paired_settings.tsv`

---
