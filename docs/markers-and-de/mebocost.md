# MEBOCOST CCC

`scomnom markers-and-de ccc mebocost` runs MEBOCOST metabolite-mediated cell-cell communication inference on cluster labels from a selected scOmnom clustering round. It writes communication tables, annotation summaries, figures, and an `adata.uns["markers_and_de"]["ccc"]["mebocost"]` payload.

```bash
scomnom markers-and-de ccc mebocost \
  --input-path results/adata.clustered.annotated.zarr.tar.zst \
  --round-id r5_broad_cell_types
```

## Pooled MEBOCOST Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | inferred from input, `ccc_mebocost`, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--n-jobs` | `1` | General command parallelism setting. |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create summary plots. |
| `--round-id` | active clustering round | Selects sender/receiver labels. |
| `--group-key` | resolved from round | Override the group column directly. |
| `--label-source` | `pretty` | Use pretty labels where available. |

## Conditions

| Syntax | Behavior |
| --- | --- |
| omitted | One MEBOCOST run on the full object. |
| `A` | One MEBOCOST run per level of `A`. |
| `A@B` | Run `A` within each level of `B`. |

| Option | Default | Notes |
| --- | --- | --- |
| `--condition-key` | none | Repeatable/comma-separated. Supports `A` and `A@B`. |
| `--condition-value` | none | Restrict context levels for `A@B`. |
| `--compare-level` | none | Optional levels of the primary condition variable to keep. |

## Cross-tissue Filtering

Cross-tissue mode is activated only when `--dataset-key` is supplied.

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset-key` | none | `adata.obs` column defining tissue/dataset origin. |
| `--source-level` | none | Allowed sender dataset levels. Required with `--dataset-key`. |
| `--target-level` | none | Allowed receiver dataset levels. Required with `--dataset-key`. |

```bash
scomnom markers-and-de ccc mebocost \
  --input-path results/adata.merged_dataset_A_dataset_B.zarr.tar.zst \
  --dataset-key dataset \
  --source-level dataset_A \
  --target-level dataset_B \
  --condition-key treatment \
  --compare-level treated \
  --compare-level vehicle \
  --input-mode lognorm
```

## MEBOCOST Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--organism` | `human` | Species passed to MEBOCOST resource loading. |
| `--input-mode` | `counts` | `counts` uses count-like input; `lognorm` builds/reuses a log-normalized layer. |
| `--lognorm-target-sum` | `10000` | Target sum for `--input-mode lognorm`. |
| `--n-shuffle` | `1000` | Permutation/shuffle count passed to MEBOCOST. |
| `--seed` | `42` | Random seed. |
| `--min-cell-number` | `10` | Minimum cells required by MEBOCOST for communication inference. |
| `--pval-cutoff` | `0.05` | P-value cutoff used for significant-event tables and plots. |
| `--plot-top-n` | `40` | Maximum events/classes used in plots. |

For count input, scOmnom prefers `counts_cb`, then `counts_raw`, then `adata.X`. `--input-mode lognorm` builds a cached log-normalized layer from `counts_cb` or `counts_raw`.

## Runtime Requirements

MEBOCOST uses the Python package `MEBOCOST` from the Chen Lab GitHub repository plus a local scOmnom cache of the upstream MEBOCOST resource database/config files.

| Option | Default | Notes |
| --- | --- | --- |
| `--install-missing-python-deps` / `--no-install-missing-python-deps` | `--no-install-missing-python-deps` | Let scOmnom bootstrap missing MEBOCOST Python dependencies/resources into the active environment/cache. |

If the package or resource bundle is missing and installation is not enabled, scOmnom stops with an explicit install hint.

## Pooled Outputs

Pooled MEBOCOST writes:

* figures: `figures/<fmt>/ccc_mebocost_<round>_roundN/`;
* tables: `tables/ccc_mebocost_<round>_roundN/`;
* saved AnnData: `adata.ccc_mebocost_<round>.zarr.tar.zst` by default;
* payloads under `adata.uns["markers_and_de"]["ccc"]["mebocost"]["runs"]`.

Key tables:

* `mebocost_commu_res.tsv`;
* `mebocost_sig_res.tsv`;
* `mebocost_source_target_summary.tsv`;
* `mebocost_metabolite_superclass_summary.tsv`;
* `mebocost_sensor_annotation_summary.tsv`;
* `mebocost_metabolite_superclass_by_source.tsv`;
* `mebocost_metabolite_superclass_by_target.tsv`;
* `mebocost_sensor_annotation_by_source.tsv`;
* `mebocost_sensor_annotation_by_target.tsv`;
* `mebocost_settings.tsv`.

Key figures include top metabolite-sensor events, source-target event-count and mean-score heatmaps, source/target breakdowns, metabolite superclass summaries, sensor annotation summaries, and condition-split comparison plots.

## Focused Donor-level Rescoring

`scomnom markers-and-de ccc mebocost-paired` rescales a focused candidate event table at donor or sample level. Use it after pooled MEBOCOST has identified candidate metabolite-sensor routes.

```bash
scomnom markers-and-de ccc mebocost-paired \
  --input-path results/adata.merged_dataset_A_dataset_B.zarr.tar.zst \
  --candidate-events results/tables/ccc_mebocost_r5_round1/mebocost_sig_res.tsv \
  --dataset-key dataset \
  --source-level dataset_A \
  --target-level dataset_B \
  --pairing-key sample_id \
  --condition-key treatment \
  --compare-level treated \
  --compare-level vehicle \
  --input-mode lognorm
```

### Paired Rescoring Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--candidate-events` | required | Candidate MEBOCOST table, usually `mebocost_sig_res.tsv` or a filtered derivative. |
| `--pairing-key` | `sample_id` | Donor/sample key for paired scoring. |
| `--condition-key` | none | Repeatable/comma-separated. Supports `A` and `A@B`. |
| `--condition-value` | none | Restrict context levels for `A@B`. |
| `--compare-level` | none | Optional levels of the primary condition variable to keep. |
| `--organism` | `human` | Species passed to MEBOCOST resource loading. |
| `--input-mode` | `counts` | `counts` or `lognorm`. |
| `--lognorm-target-sum` | `10000` | Target sum for log-normalized layer creation. |
| `--source-filter` | none | Filter candidate sender labels. |
| `--target-filter` | none | Filter candidate receiver labels. |
| `--metabolite-filter` | none | Filter candidate metabolites/HMDB IDs. |
| `--sensor-filter` | none | Filter candidate sensor genes. |
| `--superclass-filter` | none | Filter HMDB metabolite superclasses. |
| `--class-filter` | none | Filter HMDB metabolite classes. |
| `--subclass-filter` | none | Filter HMDB metabolite subclasses. |
| `--max-events` | `200` | Maximum candidate events retained for scoring. |
| `--score-method` | `mebocost-metabolite-sensor` | `mebocost-metabolite-sensor` or `associated-gene-proxy`. |
| `--min-sender-cells` | `5` | Minimum sender cells per donor/sample. |
| `--min-receiver-cells` | `5` | Minimum receiver cells per donor/sample. |
| `--min-scored-donors-per-group` | `3` | Minimum scored donors per group for effect summaries. |
| `--make-figures` / `--no-make-figures` | `--no-make-figures` | Paired mode defaults to tables only unless figures are requested. |

### Paired Outputs

Paired MEBOCOST writes:

* figures: `figures/<fmt>/ccc_mebocost_paired_<round>_roundN/`;
* tables: `tables/ccc_mebocost_paired_<round>_roundN/`;
* payloads under `adata.uns["markers_and_de"]["ccc"]["mebocost_paired_rescore"]`.

Key tables:

* `mebocost_paired_event_scores.tsv`;
* `mebocost_paired_route_scores.tsv`;
* `mebocost_paired_event_missingness.tsv`;
* `mebocost_paired_route_missingness.tsv`;
* `mebocost_paired_group_effects.tsv`;
* `mebocost_paired_settings.tsv`.

---
