# LIANA CCC

`scomnom ccc liana` runs LIANA ligand-receptor analysis on cluster labels from a selected scOmnom clustering round. It writes method tables, summary tables, figures, and an `adata.uns["markers_and_de"]["ccc"]["liana"]` payload.

```bash
scomnom ccc liana \
  --input-path results/adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types
```

Use condition keys when you want separate LIANA runs per condition level:

```bash
scomnom ccc liana \
  --input-path results/adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types \
  --condition-key treatment
```

## Pooled LIANA Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | inferred from input, `ccc_liana`, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--n-jobs` | `1` | Passed to LIANA methods. |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create summary plots. |
| `--round-id` | active clustering round | Selects the sender/receiver labels. |
| `--group-key` | resolved from round | Override the group column directly. |
| `--label-source` | `pretty` | Use pretty labels where available. |

## Conditions

| Syntax | Behavior |
| --- | --- |
| omitted | One LIANA run on the full object. |
| `A` | One LIANA run per level of `A`. |
| `A:B` | One LIANA run per joint level of `A` and `B`. |
| `A@B` | Run `A` within each level of `B`. |

| Option | Default | Notes |
| --- | --- | --- |
| `--condition-key` | none | Repeatable/comma-separated. Supports `A`, `A:B`, and `A@B`. |
| `--condition-value` | none | Restrict selected condition levels. For `A@B`, filters the `B` context levels. |
| `--compare-level` | none | Restrict levels of the comparison variable. For `A@B`, filters the `A` levels. |

## Cross-tissue Filtering

Cross-tissue mode is activated only when `--dataset-key` is supplied. In that mode, LIANA is still run on the object/subset, but returned interactions are filtered to the requested sender and receiver dataset levels.

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset-key` | none | `adata.obs` column defining tissue/dataset origin. |
| `--source-level` | none | Allowed sender dataset levels. Required with `--dataset-key`. |
| `--target-level` | none | Allowed receiver dataset levels. Required with `--dataset-key`. |
| `--signal-scope` | `all` | `all` or `secreted`. `secreted` keeps CellChatDB routes annotated as secreted signaling. |

```bash
scomnom ccc liana \
  --input-path results/adata.merged_dataset_A_dataset_B.zarr.tar.zst \
  --dataset-key dataset \
  --source-level dataset_A \
  --target-level dataset_B \
  --signal-scope secreted \
  --input-mode lognorm
```

## LIANA Methods

By default, scOmnom runs LIANA `rank_aggregate`. The aggregate uses a sparse-safer constituent set when no extra LIANA methods are requested: `cellphonedb`, `natmi`, `sca`, and `logfc`.

| Option | Default | Notes |
| --- | --- | --- |
| `--liana-method` | `rank_aggregate` | Repeatable. Add individual LIANA methods or use only a specific method. |
| `--resource` | `consensus` | LIANA ligand-receptor resource. |
| `--expr-prop` | `0.1` | Minimum expression proportion passed to LIANA. |
| `--n-perms` | `1000` | Permutations for permutation-based methods. Use `0` to disable. |
| `--seed` | `42` | Random seed. |
| `--return-all-lrs` / `--no-return-all-lrs` | `--no-return-all-lrs` | Whether LIANA returns all ligand-receptor pairs. |
| `--top-n` | `250` | Rows retained for top summaries. |
| `--plot-top-n` | `60` | Rows used in plots. |

Examples:

```bash
scomnom ccc liana ... --liana-method rank_aggregate
scomnom ccc liana ... --liana-method cellphonedb --liana-method natmi
```

## Expression Input

| Option | Default | Notes |
| --- | --- | --- |
| `--input-mode` | `lognorm` | `lognorm` builds/reuses a library-normalized log1p layer; `counts` is an explicit expert opt-in. |
| `--lognorm-target-sum` | `10000` | Target sum for `--input-mode lognorm`. |
| `--use-raw` / `--no-use-raw` | `--no-use-raw` | Use `adata.raw` explicitly; only valid with `--input-mode counts`. |
| `--layer` | none | Explicit layer override. Cannot be combined with `--use-raw`; scOmnom records but does not transform a user-supplied layer. |

By default, scOmnom selects `adata.layers["counts_cb"]` and then `adata.layers["counts_raw"]` as the source count assay, builds `lognorm_counts_cb` or `lognorm_counts_raw` with library normalization to 10,000 counts per cell followed by `log1p`, and reuses that sparse layer only when its recorded source and target sum match. Explicit `--input-mode counts` uses the preferred count assay without transformation and emits a warning because library depth can affect interaction rankings. Explicitly requested missing layers or `adata.raw` fail before inference.

## CellChatDB Route Families

Route-family plots use the vendored CellChatDB annotation table at `src/scomnom/resources/cellchatdb_interaction_annotations.tsv`. scOmnom first maps each LIANA ligand-receptor pair to CellChatDB route families such as `BMP`, `TGFb`, `NOTCH`, or `CXCL`; unmatched pairs fall back to a descriptive heuristic classifier. Top-interaction and route-summary tables record `route_family_source` as `CellChatDB` or `heuristic`. Heuristic families are exploratory labels, not database-supported pathway assignments. The strict cross-tissue `secreted` filter retains only interactions explicitly annotated as secreted signaling by CellChatDB.

## Pooled Outputs

Pooled LIANA writes:

* figures: `figures/<fmt>/ccc_liana_<round>_roundN/`;
* tables: `tables/ccc_liana_<round>_roundN/`;
* saved AnnData: `adata.ccc_liana_<round>.zarr.tar.zst` by default;
* payloads under `adata.uns["markers_and_de"]["ccc"]["liana"]["runs"]`.

Key tables:

* `liana_<method>.tsv`;
* `liana_<primary_method>_top.tsv`;
* `source_target_summary.tsv`;
* `route_family_summary.tsv`;
* `liana_settings.tsv`.

Key figures include source-target heatmaps, send/receive summaries, circos plots, top interaction plots, route-family plots, target-cluster plots, and condition-split comparison plots when multiple comparable condition runs are present.

## Focused Sample-level Rescoring

`scomnom ccc liana-paired` rescores a focused LIANA candidate table at the sample level. Use it after pooled LIANA has identified candidate ligand-receptor edges and you want replicate-level effect summaries. The command name is retained for compatibility; the statistical design is selected explicitly with `--design independent` or `--design paired`.

```bash
scomnom ccc liana-paired \
  --input-path results/adata.merged_dataset_A_dataset_B.zarr.tar.zst \
  --candidate-events results/tables/ccc_liana_r5_round1/liana_rank_aggregate.tsv \
  --dataset-key dataset \
  --source-level dataset_A \
  --target-level dataset_B \
  --sample-key sample_id \
  --subject-key donor_id \
  --design paired \
  --condition-key treatment \
  --compare-level treated \
  --compare-level vehicle \
  --min-complete-pairs 3 \
  --input-mode lognorm
```

### Rescoring Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--candidate-events` | required | Candidate LIANA table, usually `liana_rank_aggregate.tsv` or a filtered derivative. |
| `--sample-key` | `sample_id` | Independent scoring unit. Each value must map to exactly one condition/context combination and, when supplied, one subject. `--pairing-key` is retained as a compatibility alias. |
| `--subject-key` | none | Subject identifier used to match repeated measurements. Required for `--design paired`. |
| `--design` | `independent` | `independent` uses Mann-Whitney inference; `paired` uses complete-pair Wilcoxon inference. |
| `--condition-key` | none | Repeatable/comma-separated. Supports `A` and `A@B`. |
| `--condition-value` | none | Restrict context levels for `A@B`. |
| `--compare-level` | none | Optional levels of the primary condition variable to keep. |
| `--input-mode` | `lognorm` | `lognorm` by default; `counts` is an explicit expert opt-in. |
| `--lognorm-target-sum` | `10000` | Target sum for log-normalized layer creation. |
| `--source-filter` | none | Filter candidate sender labels. |
| `--target-filter` | none | Filter candidate receiver labels. |
| `--ligand-filter` | none | Filter candidate ligand complexes. |
| `--receptor-filter` | none | Filter candidate receptor complexes. |
| `--route-family-filter` | none | Filter candidate route families. |
| `--max-edges` | `200` | Maximum candidate edges retained for scoring. |
| `--min-sender-cells` | `5` | Minimum sender cells per sample. |
| `--min-receiver-cells` | `5` | Minimum receiver cells per sample. |
| `--min-scored-samples-per-group` | `3` | Minimum scored samples per group for independent effects. `--min-scored-donors-per-group` is retained as a compatibility alias. |
| `--min-complete-pairs` | `3` | Minimum subjects with scores in both groups for paired effects. |

The workflow scores each candidate edge as `sqrt(ligand_expr * receptor_expr)` per sample, then summarizes edge scores into route-family scores and group-effect tables. Independent designs report a two-sided Mann-Whitney p-value and Cliff's delta. Paired designs match samples by `--subject-key`, retain complete subject pairs, and report a two-sided Wilcoxon signed-rank p-value and paired rank-biserial effect size. Benjamini-Hochberg correction is applied across the emitted hypothesis family.

Each sample must map to one level of every selected condition or context column. A donor identifier that spans multiple conditions is therefore not a valid `--sample-key`; provide separate sample and subject columns instead. Paired mode also rejects duplicate sample observations for the same subject-condition combination. Ligand and receptor complexes are scored only when every named subunit is present; incomplete complexes remain unscored with an explicit missingness reason.

Rescoring defaults to the same log-normalized expression contract as pooled discovery. Candidate route families retain their recorded source; a route family supplied without provenance is marked `provided_unverified`. Candidate selection precedes these focused tests, so their p-values describe the supplied candidate set rather than a new genome-wide discovery screen.

Candidate discovery does not guarantee replicate-level support. An edge is testable only when both its sender and receiver populations meet their cell-count thresholds in enough independent samples or complete subject pairs. Review the missingness tables and the reported sample or pair counts before interpreting an empty or sparse effect table; do not lower support thresholds solely to recover significance.

### Rescoring Outputs

Focused LIANA rescoring writes:

* figures: `figures/<fmt>/ccc_liana_paired_<round>_roundN/`;
* tables: `tables/ccc_liana_paired_<round>_roundN/`;
* payloads under `adata.uns["markers_and_de"]["ccc"]["liana_paired_rescore"]`.

Key tables:

* `liana_paired_lr_edge_scores.tsv`;
* `liana_paired_route_scores.tsv`;
* `liana_paired_lr_edge_missingness.tsv`;
* `liana_paired_route_missingness.tsv`;
* `liana_paired_lr_edge_effects.tsv`;
* `liana_paired_route_effects.tsv`;
* `liana_paired_settings.tsv`.

---
