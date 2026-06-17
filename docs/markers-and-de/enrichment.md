# Enrichment

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
