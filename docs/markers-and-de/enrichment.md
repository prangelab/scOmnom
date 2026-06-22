# Enrichment

The enrichment submodule has three entry points:

| Entry point | Input | Main use |
| --- | --- | --- |
| `scomnom markers-and-de enrichment cluster` | AnnData with a clustering round | Run MSigDB, PROGENy, and DoRothEA on round-native pseudobulk expression. |
| `scomnom markers-and-de enrichment de` | Exported DE result tables | Run the same pathway/TF activity backends from DE statistics, without loading AnnData. |
| `scomnom markers-and-de enrichment module-score` | AnnData plus user gene modules | Score custom gene programs per cell, then summarize by cluster or cluster-condition. |

For decoupler-based enrichment, the default resource set is MSigDB HALLMARK + REACTOME, PROGENy, and DoRothEA. MSigDB can also use custom `.gmt` files.

## Cluster Enrichment

`enrichment cluster` recomputes round-native pseudobulk expression for the selected clustering round, then runs decoupler resources on that expression matrix.

```bash
scomnom markers-and-de enrichment cluster \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types
```

Add `--condition-key` when you want enrichment profiles for `cluster x condition` groups instead of one profile per cluster:

```bash
scomnom markers-and-de enrichment cluster \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types \
  --condition-key treatment
```

### Cluster Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. If omitted, scOmnom uses the standard results-location logic. |
| `--output-name` | inferred from input, module, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--n-jobs` | `1` | Reserved for consistency with other markers-and-de commands. |
| `--round-id` | active clustering round | Selects which clustering round supplies the population labels. |
| `--condition-key` | none | Optional condition key for cluster-by-condition pseudobulk. |
| `--gene-filter` | none | Repeatable pandas-query expressions against `adata.var`; applied before enrichment. |

Round-native enrichment stores results back into the selected round. It does not create a new clustering round.

### Condition Groups

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One enrichment profile per cluster. |
| `A` | Single `adata.obs` key | One enrichment profile per `cluster x A level`. |
| `A:B` | Composite key | One enrichment profile per `cluster x all combinations of A and B`. |

Examples:

```bash
scomnom markers-and-de enrichment cluster ... --round-id r5_broad_cell_types
scomnom markers-and-de enrichment cluster ... --round-id r5_broad_cell_types --condition-key treatment
scomnom markers-and-de enrichment cluster ... --round-id r5_broad_cell_types --condition-key treatment:genotype
```

### Pseudobulk Source

| Option | Default | Notes |
| --- | --- | --- |
| `--decoupler-pseudobulk-agg` | `mean` | Aggregation used for round-native pseudobulk expression. |
| `--decoupler-use-raw` / `--no-decoupler-use-raw` | `--decoupler-use-raw` | Prefer raw-like count sources when available. |
| Preferred count layers | `counts_cb`, then `counts_raw` | Internal order when raw-like layers are available. |

The pseudobulk input is genes by cluster, or genes by cluster-condition group when `--condition-key` is set.

### Shared Decoupler Settings

| Option | Default | Notes |
| --- | --- | --- |
| `--decoupler-method` | `consensus` | Fallback method for resources that do not have a resource-specific method set. |
| `--decoupler-consensus-methods` | `ulm`, `mlm`, `wsum` | Constituent methods used by consensus. Repeat the option to override. |
| `--decoupler-min-n-targets` | `5` | Fallback minimum target count for resources without a resource-specific value. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | split signed bars | Plot positive and negative activities separately. |
| `--decoupler-bar-top-n-up` | none | Optional cap on positive barplot entries. |
| `--decoupler-bar-top-n-down` | none | Optional cap on negative barplot entries. |

### MSigDB

MSigDB runs by default with HALLMARK and REACTOME gene sets.

| Option | Default | Notes |
| --- | --- | --- |
| `--msigdb-gene-sets` | `HALLMARK,REACTOME` | Comma-separated MSigDB keywords or paths to `.gmt` files. |
| `--msigdb-method` | `consensus` | Decoupler method for MSigDB activity. |
| `--msigdb-min-n-targets` | `5` | Minimum overlap between the expression universe and a gene set. |

Example with a custom GMT:

```bash
scomnom markers-and-de enrichment cluster ... \
  --msigdb-gene-sets HALLMARK,REACTOME,/path/to/custom_programs.gmt
```

### PROGENy

PROGENy runs by default. Disable it with `--no-run-progeny`.

| Option | Default | Notes |
| --- | --- | --- |
| `--run-progeny` / `--no-run-progeny` | `--run-progeny` | Toggle PROGENy. |
| `--progeny-method` | `consensus` | Decoupler method. |
| `--progeny-min-n-targets` | `5` | Minimum target overlap. |
| `--progeny-top-n` | `100` | Top weighted genes per pathway from the PROGENy resource. |
| `--progeny-organism` | `human` | Organism passed to the decoupler resource loader. |

### DoRothEA

DoRothEA runs by default. Disable it with `--no-run-dorothea`.

| Option | Default | Notes |
| --- | --- | --- |
| `--run-dorothea` / `--no-run-dorothea` | `--run-dorothea` | Toggle DoRothEA. |
| `--dorothea-method` | `consensus` | Decoupler method. |
| `--dorothea-min-n-targets` | `5` | Minimum TF-target overlap. |
| `--dorothea-confidence` | `A,B,C` | DoRothEA confidence levels to keep. |
| `--dorothea-organism` | `human` | Organism passed to the decoupler resource loader. |

### Cluster Gene Filtering

`--gene-filter` filters genes before decoupler activity inference. Filters are evaluated as pandas-query expressions against `adata.var`, and repeated filters are combined with logical AND.

```bash
scomnom markers-and-de enrichment cluster ... \
  --gene-filter "not gene.str.startswith('MT-')" \
  --gene-filter "not gene.str.startswith('RPL')" \
  --gene-filter "not gene.str.startswith('RPS')"
```

If required metadata such as `gene_type` or `gene_chrom` is missing, scOmnom tries to annotate `adata.var` before filtering. If annotation lookup or filter evaluation fails, the run aborts instead of silently continuing unfiltered.

### Cluster Plotting And Outputs

| Option | Default | Notes |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create enrichment plots and reports. |
| `--regenerate-figures` | off | Rebuild figures from stored round payloads without recomputation. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |

Cluster enrichment writes:

* figures: `figures/<fmt>/enrichment_<round>_roundN/`;
* report: `figures/<fmt>/enrichment_<round>_roundN/enrichment_report.html`;
* saved AnnData: `adata.enrichment_<round>.zarr.tar.zst` by default;
* round payloads under `adata.uns["cluster_rounds"][round_id]["decoupler"]`.

## DE-Table Enrichment

`enrichment de` reads exported DE tables and computes pathway/TF activity from the DE statistic column. It does not load or modify AnnData.

```bash
scomnom markers-and-de enrichment de \
  --input-dir results/tables/de_r5_broad_cell_types_round1 \
  --de-decoupler-source pseudobulk
```

### DE Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-dir`, `-i` | required | DE table directory, for example `results/tables/de_<round>_roundN`. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | `enrichment_de_<input_dir_name>` | Used for output folder naming. |
| `--n-jobs` | `1` | Parallelism setting for the command. |
| `--gene-filter` | none | Repeatable pandas-query expressions against columns available in exported DE tables. |
| `--de-decoupler-source` | `auto` | Which DE tables to use: `auto`, `all`, `pseudobulk`, `cell`, or `none`. |
| `--de-decoupler-stat-col` | `stat` | Statistic column used as the signed ranking/activity input. |

Source behavior:

| Source | Meaning |
| --- | --- |
| `auto` | Use available DE sources with the normal preference logic. |
| `all` | Use both pseudobulk and cell-level DE tables when present. |
| `pseudobulk` | Restrict to pseudobulk DE tables. |
| `cell` | Restrict to cell-level DE tables. |
| `none` | Skip DE-derived decoupler. |

### DE Resource Settings

The DE-table mode uses the same decoupler resources and resource-specific defaults as cluster enrichment:

| Resource | Default | Disable/change |
| --- | --- | --- |
| MSigDB | `HALLMARK,REACTOME`; method `consensus`; min targets `5` | Change with `--msigdb-gene-sets`, `--msigdb-method`, `--msigdb-min-n-targets`. |
| PROGENy | enabled; method `consensus`; min targets `5`; top genes `100`; organism `human` | Disable with `--no-run-progeny`; change with `--progeny-*` options. |
| DoRothEA | enabled; method `consensus`; min targets `5`; confidence `A,B,C`; organism `human` | Disable with `--no-run-dorothea`; change with `--dorothea-*` options. |

MSigDB DE enrichment can also run GSEA and joint decoupler/GSEA summaries through the DE-enrichment engine. The command currently exposes the decoupler resource knobs directly; the GSEA/joint defaults follow the enrichment configuration.

### DE Plotting And Outputs

| Option | Default | Notes |
| --- | --- | --- |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create enrichment plots and reports. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | split signed bars | Plot positive and negative activities separately. |
| `--decoupler-bar-top-n-up` | none | Optional cap on positive barplot entries. |
| `--decoupler-bar-top-n-down` | none | Optional cap on negative barplot entries. |

DE-table enrichment writes:

* figures: `figures/<fmt>/enrichment_de_<inputdir>_roundN/`;
* tables: `tables/enrichment_de_<inputdir>_roundN/`;
* report: `figures/<fmt>/enrichment_de_<inputdir>_roundN/enrichment_de_report.html`.

## Module Score

`enrichment module-score` scores custom gene modules per cell, then summarizes those scores by cluster or by cluster-condition group.

```bash
scomnom markers-and-de enrichment module-score \
  --input-path adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types \
  --module-file gene_programs.tsv \
  --module-set-name immune_programs
```

### Module Inputs

| Option | Default | Notes |
| --- | --- | --- |
| `--module-file` | required | Repeatable. Supports `.gmt`, `.tsv`, `.csv`, `.txt`, and `.list`. |
| `--module-set-name` | first module file stem | Stable name used in output names and stored score keys. |

Supported module file formats:

| Format | Expected structure |
| --- | --- |
| `.gmt` | Standard GMT: module name, description, then genes. |
| `.tsv` / `.csv` | Prefer columns named `module` and `gene`; also accepts `set`/`signature` and `genes`/`symbol`. If no known names exist, the first two columns are interpreted as module and gene. |
| `.txt` / `.list` | One gene per line; the file stem becomes the module name. |

### Module Grouping

Module scoring uses the same grouped-state syntax as `enrichment cluster`.

| Syntax | Meaning | Resulting behavior |
| --- | --- | --- |
| omitted | Round only | One summary profile per cluster. |
| `A` | Single `adata.obs` key | One summary profile per `cluster x A level`. |
| `A:B` | Composite key | One summary profile per `cluster x all combinations of A and B`. |

### Module Score Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--module-score-method` | `scanpy` | Backend: `scanpy` or `aucell`. |
| `--module-score-use-raw` / `--no-module-score-use-raw` | `--no-module-score-use-raw` | Use `adata.raw`; cannot be combined with `--module-score-layer`. |
| `--module-score-layer` | none | Use a named `adata.layers` matrix; cannot be combined with `--module-score-use-raw`. |
| `--module-score-ctrl-size` | `50` | Scanpy control-gene pool size. Used only by `scanpy`. |
| `--module-score-n-bins` | `25` | Scanpy expression bin count. Used only by `scanpy`. |
| `--module-score-random-state` | `0` | Scanpy random seed. |
| `--module-score-max-umaps` | `12` | Maximum module score columns to plot on UMAP. |

`scanpy` uses `scanpy.tl.score_genes`. `aucell` uses `decoupler.mt.aucell` with `tmin=1`.

### Module Score Outputs

Module-score writes:

* figures: `figures/<fmt>/module_score_<set>_<round>_roundN/`;
* tables: `tables/module_score_<set>_<round>_roundN/`;
* report: `figures/<fmt>/module_score_<set>_<round>_roundN/module_score_report.html`;
* saved AnnData: `adata.module_score_<set>_<round>.zarr.tar.zst` by default;
* per-cell score columns in `adata.obs` named `module_score__<round>__<set>__<module>`;
* round-level payloads under `adata.uns["cluster_rounds"][round_id]["module_scores"]`.

Key tables:

* `module_meta.tsv`;
* `module_score_summary_mean.tsv`;
* `module_score_summary_median.tsv`;
* `module_score_summary_mean_z.tsv`;
* `module_score_group_sizes.tsv`;
* `__settings.txt`.

---
