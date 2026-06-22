# NicheNet CCC

`scomnom markers-and-de ccc nichenet` runs sender-focused NicheNet ligand activity analysis for one receiver cluster or, by default, every receiver cluster. It complements LIANA: LIANA proposes sender-receiver ligand-receptor structure, while NicheNet prioritizes ligands that best explain a receiver transcriptional program.

```bash
scomnom markers-and-de ccc nichenet \
  --input-path results/adata.clustered.annotated.zarr.tar.zst \
  --condition-key treatment \
  --compare-level treated \
  --compare-level vehicle
```

## Inputs And Defaults

| Option | Default | Notes |
| --- | --- | --- |
| `--input-path`, `-i` | required | AnnData object loaded through scOmnom IO. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | inferred from input, `ccc_nichenet`, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--n-jobs` | `1` | General command parallelism setting. |
| `--make-figures` / `--no-make-figures` | `--make-figures` | Create ligand and ligand-target plots. |
| `--round-id` | active clustering round | Selects receiver/sender labels. |
| `--group-key` | resolved from round | Override the group column directly. |
| `--label-source` | `pretty` | Use pretty labels where available. |

## Receiver And Sender Selection

| Option | Default | Notes |
| --- | --- | --- |
| `--receiver-cluster` | `all` | Receiver cluster to explain. Accepts raw ids or pretty labels. `all` batches over every cluster. |
| `--sender-cluster` | none | Optional sender cluster restriction. Repeatable/comma-separated. |
| `--dataset-key` | none | Enables cross-tissue sender/receiver restriction. |
| `--source-level` | none | Allowed sender dataset levels. Required with `--dataset-key`. |
| `--target-level` | none | Allowed receiver dataset levels. Required with `--dataset-key`. |
| `--signal-scope` | `all` | `all` or `secreted` for downstream ligand-receptor interpretation. |

```bash
scomnom markers-and-de ccc nichenet \
  --input-path results/adata.merged_dataset_A_dataset_B.zarr.tar.zst \
  --dataset-key dataset \
  --source-level dataset_A \
  --target-level dataset_B \
  --receiver-cluster macrophages \
  --condition-key treatment \
  --compare-level treated \
  --compare-level vehicle
```

## Receiver Gene Set

NicheNet needs a receiver gene set. scOmnom can get it in two ways:

| Mode | Required options | Notes |
| --- | --- | --- |
| Receiver DE | `--condition-key` and exactly two `--compare-level` values | Runs receiver-cluster DE and uses significant/upregulated genes after filtering. |
| Explicit gene list | `--gene-list-file` | One gene per line. Allows NicheNet without a condition contrast. |

Condition syntax:

| Syntax | Behavior |
| --- | --- |
| `A` | Compare two levels of `A` in each receiver cluster. |
| `A@B` | Compare two levels of `A` separately within each level of `B`. |

| Option | Default | Notes |
| --- | --- | --- |
| `--condition-key` | none | Repeatable/comma-separated. Supports `A` and `A@B`. |
| `--condition-value` | none | Restrict context levels for `A@B`. |
| `--compare-level` | required for receiver-DE mode | Exactly two levels of the primary condition variable. |
| `--gene-list-file` | none | One-gene-per-line receiver gene set. |
| `--min-logfc` | `0.25` | Minimum receiver-DE log fold-change for gene-set construction. |
| `--padj-threshold` | `0.05` | Adjusted p-value cutoff for receiver-DE gene-set construction. |

## Expression And Model Knobs

| Option | Default | Notes |
| --- | --- | --- |
| `--expression-pct` | `0.10` | Minimum fraction of cells expressing a gene in sender/receiver expressed-gene filters. |
| `--input-mode` | `counts` | `counts` uses count-like input; `lognorm` builds/reuses a log-normalized layer. |
| `--lognorm-target-sum` | `10000` | Target sum for `--input-mode lognorm`. |
| `--top-n-ligands` | `30` | Number of ligands retained in the NicheNet output/plots. |
| `--top-n-targets` | `200` | Number of ligand-target links retained. |
| `--organism` | `human` | Current scOmnom NicheNet support expects `human`. |

For count input, scOmnom uses the same count preference as LIANA: `counts_cb`, then `counts_raw`, then `adata.X`. `--input-mode lognorm` builds a cached log-normalized layer from `counts_cb` or `counts_raw`.

## Runtime Requirements

This backend shells out to `Rscript` and expects the R package `nichenetr` in a local cache library. On macOS the default cache is `~/Library/Caches/scOmnom/r-libs/nichenet/`; on Linux it is `~/.cache/scOmnom/r-libs/nichenet/`.

| Option | Default | Notes |
| --- | --- | --- |
| `--install-missing-r-deps` / `--no-install-missing-r-deps` | `--no-install-missing-r-deps` | Let scOmnom bootstrap missing NicheNet R dependencies into the cache library. |

The environment YAML files include the R runtime and helper packages, but `nichenetr` itself is not pinned there because available Conda builds are stale and not consistently cross-platform. The first NicheNet run may also need internet access to fetch official human model files.

## Outputs

NicheNet writes:

* figures: `figures/<fmt>/ccc_nichenet_<round>_roundN/`;
* tables: `tables/ccc_nichenet_<round>_roundN/`;
* saved AnnData: `adata.ccc_nichenet_<round>.zarr.tar.zst` by default;
* payloads under `adata.uns["markers_and_de"]["ccc"]["nichenet"]["runs"]`.

Each receiver run writes tables under a `receiver__<cluster>` folder:

* `nichenet_ligand_activity.tsv`;
* `nichenet_ligand_target_links.tsv`;
* `nichenet_ligand_receptor_links.tsv`;
* `nichenet_potential_ligands.tsv`;
* `nichenet_receiver_de.tsv`;
* `nichenet_settings.tsv`.

Key figures include top ligand plots and ligand-target heatmaps for each receiver cluster.

---
