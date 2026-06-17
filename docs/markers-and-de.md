# Markers and DE (markers-and-de)

The `markers-and-de` module covers four related analyses:

* **Markers**: cluster-vs-rest marker discovery
* **DE**: within-cluster differential expression across condition levels
* **DA**: differential abundance (compositional shifts of cell types)
* **Enrichment**: pathway and TF activity scoring either from a clustering round or from exported DE result tables

All four subcommands are CLI-first and store results in tables plus a self-contained report.

### Quick entry points

```bash
# Marker discovery (cluster-vs-rest)
scomnom markers-and-de markers \
  --input-path results/adata.clustered.annotated.projected.zarr

# Within-cluster DE (condition contrasts)
scomnom markers-and-de de \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --condition-key condition

# Differential abundance (composition)
scomnom markers-and-de da \
  --input-path results/adata.clustered.annotated.projected.markers.de.zarr \
  --condition-key condition

# Round-native enrichment
scomnom markers-and-de enrichment cluster \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --round-id r4_subset_annotation \
  --condition-key sex \
  --gene-filter "not gene.str.startswith('MT-')"

# DE-table enrichment
scomnom markers-and-de enrichment de \
  --input-dir results/tables/de_r5_archetypes_round1 \
  --gene-filter "not gene.str.startswith('MT-')"

# Module scoring from user-defined signatures
scomnom markers-and-de enrichment module-score \
  --input-path results/adata.clustered.annotated.projected.markers.zarr \
  --round-id r5_archetypes \
  --module-file signatures.gmt \
  --module-set-name curated_programs
```

---

### Markers (cluster-vs-rest)

Markers are computed per cluster against all other cells. The module supports **cell-level markers**, **pseudobulk markers**, or **both** (`--run cell|pseudobulk|both`).

**Cell-level markers**

* Uses Scanpy `rank_genes_groups` with `wilcoxon`, `t-test`, or `logreg`.
* Optional per-cluster downsampling for very large datasets.
* Filters by `min_pct` and `min_diff_pct` to remove low-coverage or weakly differential genes.

**Pseudobulk markers**

* Aggregates counts by sample and cluster, then runs DESeq2 (via PyDESeq2).
* Requires at least **6 unique samples** (guard to avoid unstable estimates).
* Uses count layers in priority order: `counts_cb`, `counts_raw`, then `adata.X` (if allowed).
* Supports sample-level covariates (`--pb-covariates`) and filters on minimum cells per sample-cluster.

**Outputs**

* Tables: `tables/marker_tables_<run>/cell_based/` and `tables/marker_tables_<run>/pseudobulk_based/`
* Reports: `figures/markers/markers_report.html` (plus PDFs/PNGs under `figures/markers/`)

---

### LIANA CCC

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

### NicheNet CCC

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

### MEBOCOST CCC

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

### Enrichment

The enrichment submodule has three entry points:

* `scomnom markers-and-de enrichment cluster`: run decoupler directly on an existing clustering round
* `scomnom markers-and-de enrichment de`: run decoupler from exported DE result tables without loading AnnData
* `scomnom markers-and-de enrichment module-score`: score user-defined gene programs per cell and summarize them by cluster or cluster-condition

The first two modes support MSigDB, custom `.gmt` files, PROGENy, and DoRothEA. The third mode uses user-defined module definitions with per-cell module scoring.

#### Enrichment cluster (round-native pathway / TF activity)

This mode uses round-native pseudobulk expression built from the selected round labels, then computes pathway / TF activity using MSigDB, PROGENy, and DoRothEA.

**Round selection**

* `--round-id` selects which round to score.
* If omitted, the active round is used.
* Enrichment is stored back into that round; it does not create a new round.

**Supported resources**

* `MSigDB` from built-in keywords or custom `.gmt` files via `--msigdb-gene-sets`
* `PROGENy`
* `DoRothEA`

These use the same decoupler method settings as the clustering and DE workflows.

**Condition key syntax (enrichment cluster)**

Enrichment supports either cluster-only pseudobulk or cluster-by-condition pseudobulk:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One enrichment profile per cluster |
| `A` | Single obs key | One enrichment profile per `cluster × A level` |
| `A:B` | Composite key | One enrichment profile per `cluster × all combinations of A and B` |

Examples:

* `--round-id r5_archetypes`  
  Enrichment per archetype cluster.
* `--round-id r5_archetypes --condition-key sex`  
  Enrichment per `cluster × sex`.
* `--round-id r5_archetypes --condition-key sex:masld_status`  
  Enrichment per `cluster × sex × masld_status` combination.

**Gene filtering**

* `--gene-filter` filters genes before enrichment is computed, using pandas-query expressions against `adata.var`.
* Filters are combined with logical AND.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* This changes the enrichment input universe, unlike `--plot-gene-filter`, which is plot-only in DE.

Example:

* `--gene-filter "not gene.str.startswith('MT-')"`
* `--gene-filter "not gene.str.startswith('RPL')"`
* `--gene-filter "not gene.str.startswith('RPS')"`

**Outputs**

* Figures: `figures/<fmt>/enrichment_<round>_roundN/`
* Report: `figures/<fmt>/enrichment_<round>_roundN/enrichment_report.html`
* Default AnnData output name: `adata.enrichment_<round>.zarr.tar.zst`
* Stored round payload includes the selected condition key, applied gene filters, and retained gene counts for provenance

#### Enrichment de (from exported DE tables)

This mode reads exported DE CSV tables from a DE results folder and runs the same MSigDB / PROGENy / DoRothEA decoupler backends on the DE statistics. For MSigDB it can also run a Python preranked GSEA pass and build a joint concordance summary between decoupler activity direction and GSEA direction. It reuses the signed up/down barplot layout from the existing DE decoupler workflow.

**Inputs**

* `--input-dir` should point at a DE tables folder such as `results/tables/de_r5_archetypes_round1`
* No AnnData input is required

**Supported DE sources**

* `--de-decoupler-source auto` uses any available pseudobulk and cell-level DE tables
* `--de-decoupler-source pseudobulk` limits enrichment to exported pseudobulk DE tables
* `--de-decoupler-source cell` limits enrichment to exported cell-level DE tables

**Gene filtering**

* `--gene-filter` is applied before enrichment on the DE-derived gene statistics
* Filters are evaluated against the gene metadata present in the exported DE tables, including `gene` and exported columns such as `gene_type`

**Outputs**

* Figures: `figures/<fmt>/enrichment_de_<inputdir>_roundN/`
* Tables: `tables/enrichment_de_<inputdir>_roundN/`
* Report: `figures/<fmt>/enrichment_de_<inputdir>_roundN/enrichment_de_report.html`
* MSigDB outputs can include decoupler activity plots, compact GSEA summaries, and concordance-only joint summaries when `run_gsea` is enabled

#### Enrichment module-score (user-defined signatures)

This mode scores custom gene modules per cell, then summarizes those scores by cluster or by `cluster × condition`.

**Backend**

* `--module-score-method scanpy` uses `scanpy.tl.score_genes`
* `--module-score-method aucell` uses `decoupler.mt.aucell` for rank-based AUCell scoring

**Module inputs**

* `--module-file` is repeatable
* supported formats:
  * `.gmt`: one or more modules per file
  * `.tsv` / `.csv`: either `module,gene` style two-column tables or a single-column gene list
  * `.txt`: one gene per line, interpreted as a single module named after the file stem
* `--module-set-name` gives a stable name to the scored module collection and is used in output naming

**Condition key syntax**

Module scoring uses the same grouped-state syntax as `enrichment cluster`:

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One summary profile per cluster |
| `A` | Single obs key | One summary profile per `cluster × A level` |
| `A:B` | Composite key | One summary profile per `cluster × all combinations of A and B` |

**Outputs**

* Figures: `figures/<fmt>/module_score_<set>_<round>_roundN/`
* Tables: `tables/module_score_<set>_<round>_roundN/`
* Report: `figures/<fmt>/module_score_<set>_<round>_roundN/module_score_report.html`
* Default AnnData output name: `adata.module_score_<set>_<round>.zarr.tar.zst`
* The selected round stores summary tables and score metadata under `adata.uns["cluster_rounds"][round_id]["module_scores"]`
* Per-cell score columns are written to round-specific `adata.obs` columns as `module_score__<round>__<set>__<module>`

---

### DE (within-cluster contrasts)

Within-cluster DE compares **condition levels inside each cluster**, e.g. `treated vs control` within each cell type. You can provide a single `--condition-key` or multiple `--condition-keys`, plus optional explicit contrasts (`--contrasts`).

**Condition key syntax**

| Syntax | Meaning | Resulting behavior                                                           |
| --- | --- |------------------------------------------------------------------------------|
| `A` | Single obs key | Compare levels of `A` within each cluster                                    |
| `A:B` | Composite key | Create a combined key across **all combinations** of `A` and `B` levels      |
| `A@B` | Within‑B | Run **A within each level of B** (usually what you want)                     |
| `A^B` | Interaction | Run interaction contrasts between A and B (requires a pseudobulk DESeq2 run) |

**Examples**

* `--condition-keys treatment`  
  Compare treatment levels within each cluster.
* `--condition-keys MASLD:sex`  
  Full combinations of MASLD × sex.
* `--condition-keys treatment@sex`  
  Treatment contrasts within each sex.
* `--condition-keys treatment^sex`  
  Treatment‑by‑sex interaction contrasts.

**Cell-level within-cluster DE**

* Runs contrast-conditional markers per cluster using `wilcoxon` and `logreg` (configurable).
* Uses `min_pct` and `min_diff_pct` to focus on robust signals.
* Does not require the pseudobulk sample guard.
* Typically much faster than pseudobulk, which makes it practical for iterative test runs.

**Pseudobulk within-cluster DE**

* Aggregates counts by sample and cluster, then runs DESeq2 per cluster and contrast.
* Requires at least **6 unique samples** (guarded).
* Supports sample-level covariates and minimum cells per sample-cluster.
* Usually substantially slower than cell-level mode, but more rigorous for replicate-aware inference.

**Target populations**

* `--target-groups` restricts DE to selected cluster/population labels from `--group-key` (repeatable and comma-separated).
* Applies to both cell-level and pseudobulk phases, so unrelated populations are skipped in computation and outputs.

**Gene filtering**

* `--gene-filter` filters genes before DE is calculated.
* Filters are applied to both the cell-level and pseudobulk DE inputs.
* Filters are combined with logical AND and evaluated against `adata.var`.
* If required gene metadata such as `gene_type` or `gene_chrom` is missing, scOmnom will try to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of continuing unfiltered.
* `--plot-gene-filter` remains plot-only and does not change the DE statistics.

**DE → pathway/TF activity (decoupler)**

* Optional activity inference from DE statistics for each contrast.
* Sources can be `cell`, `pseudobulk`, `all`, or `auto` (prefer pseudobulk if present).
* Supports MSigDB, PROGENy, and DoRothEA with configurable methods and target filters.
* MSigDB GSEA is enabled by default for `markers-and-de de` and writes additional `msigdb_gsea` and `msigdb_joint` outputs alongside the decoupler summaries.

**GSEA controls for `markers-and-de de`**

* `--run-gsea/--no-run-gsea` enables or disables MSigDB preranked GSEA on the DE statistic matrix.
* `--gsea-min-size`, `--gsea-max-size`, and `--gsea-eps` control pathway-size filtering and numerical tolerance for the Python `gseapy` backend.
* `--gsea-rank-col` lets you override the DE column used for GSEA ranking. If omitted, scOmnom uses the same statistic requested through `--de-decoupler-stat-col`.
* `--joint-enrichment-alpha` controls the adjusted-significance threshold used when building the `msigdb_joint` concordance summary.
* `--joint-enrichment-top-n` controls how many concordant up/down pathways are shown in the compact GSEA and joint plots.
* `--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant` toggles whether joint summaries require matching decoupler and GSEA direction.
* `--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig` toggles whether joint summaries require GSEA adjusted significance.
* `--joint-enrichment-leading-edge-top-n` controls how many leading-edge genes are previewed in compact GSEA summaries.

**Outputs**

* Tables: `tables/de_<round>_roundN/cell_based/` and `tables/de_<round>_roundN/pseudobulk_based/`

**HPC stability/throughput recipe (large pseudobulk runs)**

For large DE jobs on HPC, use a conservative worker cap and disable nested BLAS/OpenMP threading in the job script:

Base template: `slurm/scomnom_template.job`

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Then run DE with:

```bash
scomnom markers-and-de de ... --run pseudobulk --max-workers 8
```

Notes:
* `--max-workers` defaults to `8` for DE.
* Keep `--n-jobs $SLURM_CPUS_PER_TASK`; `--max-workers` controls DE task concurrency.
* If `--run both` is unstable on very large jobs, split into separate `--run pseudobulk` and `--run cell` runs with separate output folders.
* Figures: `figures/<fmt>/de_<round>_roundN/`
* Default AnnData output name: `adata.de_<round>.zarr.tar.zst`

---

### Cell-level vs pseudobulk: why both exist

* **Cell-level** tests are sensitive and can detect subtle expression shifts, but they can overweight large samples, inflate p-values on big datasets, and ignore replicate structure. By default, scOmnom caps marker output at 300 genes per cluster for cell-level runs.
* **Pseudobulk** respects sample-level independence, improves control of confounders via covariates, applies the `min_pct` filter, and yields more conservative inference, but requires enough replicates and loses single-cell resolution.

Running both can be informative: cell-level for discovery, pseudobulk for robustness.
For very large jobs, if `--run both` is unstable (for example due to memory pressure or native-thread failures), run two separate commands instead (`--run pseudobulk` and `--run cell`) into separate output folders and combine the interpretation afterward.

---

### DA (differential abundance / composition)

The DA submodule tests whether **cluster proportions change across conditions**. It works on per-sample cell-type counts and provides multiple backends, each with different assumptions:

**Condition key syntax (DA)**

DA supports the same `A` and `A:B` composite syntax as DE, plus `A@B` to run `A` within each `B` level. Interaction syntax (`A^B`) is **DE-only**. See the DE section above for examples.

You can pass one or more keys (same behavior as DE), and results are written into per-key subfolders under DA tables/figures.

**Reference selection**

* Default reference is `most_stable`, chosen as the cluster with the lowest median absolute deviation of proportions among clusters with mean proportion ≥ `min_mean_prop`.
* If none meet the threshold, the most abundant cluster is used.

**Methods**

* `sccoda`: Bayesian compositional model (pertpy scCODA). Uses NUTS sampling and inclusion probabilities with FDR control. Works directly from cell-level data but models sample-level composition.
* `glm`: Per-cluster binomial GLM on counts with a log-total offset (statsmodels). Supports covariates and returns log2-scale effects with Wald tests and BH FDR.
* `clr`: Centered log-ratio transform of proportions followed by Mann–Whitney tests for each pair of condition levels. Returns log2 fold-change of mean proportions and per-pair FDR.
* `graph`: Graph-based DA on neighborhoods in the integrated embedding, inspired by MiloR. Neighborhoods are sampled from stratified seeds, tested with **NB-GLM** on neighborhood counts (with sample-size offsets), then corrected by **spatial weighted BH FDR**. Outputs neighborhood metadata and diagnostics.

**When to use which DA method**

| Method | Best for | Key assumptions | Notes |
| --- | --- | --- | --- |
| `sccoda` | Small-to-moderate sample sizes with clear compositional shifts | Compositional Bayesian model; needs a reference | Most conservative; provides inclusion probabilities |
| `glm` | Larger sample sizes, covariate adjustment | Binomial GLM with log-total offset | Fast, interpretable log2 effects |
| `clr` | Simple two-group comparisons, quick screening | CLR transform + nonparametric test | Pairwise only; less model structure |
| `graph` | Substructure or neighborhood-level shifts | Embedding neighborhoods + NB-GLM + spatial FDR | Milo-style local DA; conservative under many tests |

**Automatic method behavior**

* GLM is automatically skipped for 2-level conditions (CLR is used instead).
* Conditions with too few replicates per level are skipped with an explicit warning.
* GraphDA applies support filters before testing (`min_nonzero_samples_per_level`) and effect shrinkage (`effect_shrink_k`) to reduce unstable calls.

**GraphDA significance and QC**

GraphDA reports both raw and adjusted evidence:

* `pval`: neighborhood-level NB-GLM p-value
* `fdr_bh`: standard BH correction
* `fdr_spatial`: weighted BH correction using neighborhood spatial weights
* `fdr`: current primary significance column (set to `fdr_spatial`)

Why this matters:

* Many neighborhoods overlap and are not independent tests.
* Spatial FDR is the primary call metric for Milo-style neighborhood DA.
* It is possible to see directional effects with no FDR-significant neighborhoods if multiplicity is high and raw p-values are moderate.

**Outputs**

Per condition key (and per `A@B=<level>` expansion), DA writes:

* Tables: `tables/DA_tables_<run>/<condition_tag>/`
* Figures: `figures/DA/<condition_tag>/`

Key DA tables include:

* `composition_global_sccoda.tsv`
* `composition_global_glm.tsv` (when eligible)
* `composition_global_clr.tsv`
* `composition_global_graph.tsv`
* `composition_consensus.tsv`
* `composition_graph_neighborhoods.tsv`
* `graphda_diagnostics.tsv`
* `composition_settings.txt`

Key DA figures include:

* composition summaries (`composition_stacked_bar_100`, `composition_stacked_comparison`, `composition_flow`)
* global effects (`composition_effects_global_sccoda`, `composition_effects_global_clr`)
* GraphDA summaries (`graphda_effects_by_cluster`, `graphda_top_neighborhoods`, `graphda_top_by_cluster`)
* GraphDA QC (`graphda_qc_pval_vs_fdr`, `graphda_qc_cluster_power`)

---
