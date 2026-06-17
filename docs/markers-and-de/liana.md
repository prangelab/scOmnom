# LIANA CCC

`scomnom markers-and-de ccc liana` runs LIANA-based ligand-receptor analysis on an annotated AnnData object and writes tables, figures, and an `adata.uns["markers_and_de"]["ccc"]["liana"]` payload.

If `--dataset-key` is omitted, scOmnom runs the normal within-object LIANA workflow. Cross-tissue / cross-dataset filtering is only activated when `--dataset-key` is provided together with `--source-level` and `--target-level`.

**Assay selection**

For count-based CCC input, scOmnom prefers assays in this order:

* `counts_cb`
* `counts_raw`
* `adata.X` as a final fallback

Use `--use-raw` only when you explicitly want `adata.raw`.

Use `--input-mode lognorm` when you want LIANA to build and reuse a cached log-normalized layer from `counts_cb` or `counts_raw` instead of running directly on raw counts. This creates `adata.layers["lognorm_counts_cb"]` or `adata.layers["lognorm_counts_raw"]` on demand and then uses that layer for LIANA.

**Condition key syntax**

LIANA CCC supports the same round-aware condition syntax used elsewhere in the pipeline:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | No condition split | One LIANA run on the full object |
| `A` | Single obs key | One LIANA run per level of `A` |
| `A:B` | Composite key | One LIANA run per joint level of `A` and `B` |
| `A@B` | A within B | One LIANA run per level of `B`, with `A` treated as the condition label inside each subset |

**Cross-tissue mode**

For merged multi-tissue objects, LIANA CCC can be restricted to cross-dataset signaling candidates by specifying:

* `--dataset-key`: obs column defining tissue / dataset origin
* `--source-level`: allowed sender tissue / dataset level(s)
* `--target-level`: allowed receiver tissue / dataset level(s)
* `--signal-scope`: `all` or `secreted`

In this mode, scOmnom still runs LIANA on the merged AnnData object, but filters the returned interactions to the requested source and target tissue levels. With `--signal-scope secreted`, scOmnom further restricts the retained interactions to CellChatDB routes annotated as `Secreted Signaling`.

**CellChatDB-backed pathway families**

The LIANA family / route plots use a vendored CellChatDB annotation table at [`src/scomnom/resources/cellchatdb_interaction_annotations.tsv`](https://github.com/prangelab/scOmnom/blob/main/src/scomnom/resources/cellchatdb_interaction_annotations.tsv). This table was exported one time from the official `jinworks/CellChat` package source at version `2.2.0.9001`.

scOmnom first tries to map each LIANA `ligand_complex` / `receptor_complex` pair onto CellChatDB and uses CellChat `pathway_name` values such as `BMP`, `TGFb`, `NOTCH`, or `CXCL` as the route family. If no exact CellChatDB match is found, scOmnom falls back to its internal heuristic family classifier.

The vendored export preserves CellChat's own row-level `version` field, which currently contains both:

* `CellChatDB v1`
* `CellChatDB v2`

**Example**

```bash
scomnom markers-and-de ccc liana \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr.tar.zst \
  --condition-key masld_status@timepoint
```

```bash
scomnom markers-and-de ccc liana \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --dataset-key tissue \
  --source-level liver \
  --target-level fat \
  --signal-scope secreted
```

**Outputs**

* Figures: `figures/<fmt>/ccc_liana_<round>_roundN/`
* Tables: `tables/ccc_liana_<round>_roundN/`
* Route-family summary: `route_family_summary.tsv`

**Focused donor-level rescoring**

`scomnom markers-and-de ccc liana-paired` rescales a focused LIANA candidate table at donor or sample level instead of rerunning pooled LIANA discovery. It is meant for downstream paired analyses once pooled LIANA has already identified candidate ligand-receptor edges.

**Current scope**

* reads a candidate LR table such as `liana_rank_aggregate.tsv`
* supports the same `A` and `A@B` condition-subset syntax as pooled LIANA
* supports cross-tissue sender/receiver restriction via `--dataset-key`, `--source-level`, and `--target-level`
* supports the same LIANA expression input modes as pooled LIANA:
  * `--input-mode counts`
  * `--input-mode lognorm`
* recomputes donor/sample-level ligand-receptor scores from cluster-level ligand and receptor complex expression using:
  * `sqrt(ligand_expr * receptor_expr)`
* aggregates those donor-level edge scores to route-family summaries and within-condition group-effect tables
* paired rescoring defaults to a donor-sparser threshold set than pooled discovery:
  * `--min-sender-cells 5`
  * `--min-receiver-cells 5`
  * `--min-scored-donors-per-group 3`
* always writes missingness summary tables so donor-level sparsity is visible before interpreting empty effect tables
* optional focused figures for:
  * paired route-family effect dotplots
  * paired ligand-receptor edge effect strip plots
* writes focused donor-level tables and stores them under `adata.uns["markers_and_de"]["ccc"]["liana_paired_rescore"]`

**Example**

```bash
scomnom markers-and-de ccc liana-paired \
  --input-path results/adata.merged_liver_fat.clustered.annotated.zarr.tar.zst \
  --candidate-events results/tables/ccc_liana_r5_round1/condition__sex_at_MASLD/MASLD=yes/female/liana_rank_aggregate.tsv \
  --dataset-key tissue \
  --source-level vfat \
  --target-level liv \
  --pairing-key sample_id \
  --condition-key sex@MASLD \
  --condition-value yes \
  --compare-level female \
  --compare-level male \
  --input-mode lognorm
```

**Outputs**

* Figures: `figures/<fmt>/ccc_liana_<round>_roundN/paired_rescore/<condition>/`
* Tables: `tables/ccc_liana_<round>_roundN/paired_rescore/<condition>/`
* Key tables:
  * `liana_paired_lr_edge_scores.tsv`
  * `liana_paired_route_scores.tsv`
  * `liana_paired_lr_edge_missingness.tsv`
  * `liana_paired_route_missingness.tsv`
  * `liana_paired_lr_edge_effects.tsv`
  * `liana_paired_route_effects.tsv`
  * `liana_paired_settings.tsv`

---
