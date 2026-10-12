# NicheNet CCC

`scomnom ccc nichenet` runs sender-focused NicheNet ligand activity analysis for one receiver cluster or, by default, every receiver cluster. It complements LIANA: LIANA proposes sender-receiver ligand-receptor structure, while NicheNet prioritizes ligands that best explain a receiver transcriptional program.

```bash
scomnom ccc nichenet \
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
scomnom ccc nichenet \
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

The environment YAML files include the R runtime and available helper packages.
On Apple Silicon, `fdrtool`, `mlrMBO`, and `emoa` are supplied through the separate
R cache along with `nichenetr`, because they are unavailable from the declared
Conda channels for that platform. The opt-in bootstrap follows upstream CRAN
versions and the NicheNet repository; it is not version-locked. For reproducible
analyses, provision a reviewed, revision-pinned library separately. See
[installation](../installation.md#nichenet-r-backend). The first NicheNet run may
also need internet access to fetch official human model files.

### Model Cache And Runtime Logs

The three human models are fixed to Zenodo record `7074291`. Downloads and cache
hits are checked against its published byte counts and MD5 checksums; an observed
SHA256 is also recorded for reproducibility. The SHA256 is not an independently
published reference checksum. R checks the local files again before reading them.
The model selection and inference settings are unchanged.

Set `SCOMNOM_NICHENET_CACHE_DIR` to a persistent, writable cache directory. Models
are stored in its `7074291/` subdirectory. Without this override, scOmnom uses
`$XDG_CACHE_HOME/scOmnom/nichenet/`, or `~/Library/Caches/scOmnom/nichenet/` on
macOS and `~/.cache/scOmnom/nichenet/` on Linux when `XDG_CACHE_HOME` is unset.
On HPC systems, use shared project scratch for this cache instead of temporary
job-local storage; respect the site's scratch retention policy.

Verified cached models work without network access. Downloads use temporary
files and are published to the cache only after verification. Transient network
failures receive at most three attempts, with each failed attempt logged. TLS
verification is never disabled. A checksum mismatch stops the run and is not
silently replaced; inspect the named file before removing it to allow a fresh
download.

Each receiver's table directory contains `nichenet_model_resources.tsv` with the
resource URLs, paths, sizes, reference MD5, observed SHA256 and cache-hit status.
On successful runs this information is also stored in the run payload under
`model_resources`. R stdout and stderr are retained as `nichenet_r.stdout.log`
and `nichenet_r.stderr.log`, including on R failure, and emitted to the pipeline
log. Stderr is logged at warning level even when R exits successfully.

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
