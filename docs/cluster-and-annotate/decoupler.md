# Decoupler configuration

Decoupler is enabled by default (`run_decoupler = True`). In `cluster-and-annotate`, scOmnom first builds a round-level pseudobulk expression matrix from the active clustering round, then runs decoupler resources on that matrix.

Pseudobulk is therefore the input representation for this module's decoupler pass. The resulting activity payloads are used for:

* cluster-level pathway and TF activity inference
* biological diagnostics
* decoupler plots and reports
* compaction decisions

The round-native payloads are stored under `adata.uns["cluster_rounds"][round_id]["decoupler"]`. For the active round, compatible copies may also be published to top-level `adata.uns["msigdb"]`, `adata.uns["progeny"]`, and `adata.uns["dorothea"]`.

## Resources

The default resources are:

| Resource | Default | Meaning |
| --- | --- | --- |
| MSigDB | `HALLMARK`, `REACTOME` | MSigDB pathway collections. `HALLMARK` and `REACTOME` are MSigDB collection keywords, not separate decoupler databases. |
| PROGENy | enabled | Pathway activity model from PROGENy. |
| DoRothEA | enabled | Transcription factor regulons from DoRothEA. |

`--msigdb-gene-sets` also accepts comma-separated custom `.gmt` paths, so project-specific signatures can be scored alongside or instead of the built-in MSigDB keywords.

## Settings

| CLI option | Config field | Default | Notes |
| --- | --- | --- | --- |
| `--run-decoupler` / `--no-run-decoupler` | `run_decoupler` | `True` | Enable or disable the whole cluster-level decoupler pass. |
| `--decoupler-pseudobulk-agg` | `decoupler_pseudobulk_agg` | `mean` | Aggregation used to build the round-level pseudobulk matrix; allowed values are `mean` and `median`. |
| `--decoupler-use-raw` / `--no-decoupler-use-raw` | `decoupler_use_raw` | `True` | Prefer raw-like count layers for pseudobulk input when available. |
| `--decoupler-method` | `decoupler_method` | `consensus` | Default decoupler method used by resources unless overridden below. |
| `--decoupler-consensus-methods` | `decoupler_consensus_methods` | `ulm`, `mlm`, `wsum` | Methods combined when a resource uses `consensus`. Supported values are `ulm`, `mlm`, `wsum`, and `aucell`. |
| `--decoupler-min-n-targets` | `decoupler_min_n_targets` | `5` | Generic minimum target count per source. |
| `--decoupler-bar-split-signed` / `--no-decoupler-bar-split-signed` | `decoupler_bar_split_signed` | `False` | Split decoupler barplots into positive and negative activities. |
| `--decoupler-bar-top-n-up` | `decoupler_bar_top_n_up` | `None` | Number of positive activities to show when signed barplots are split. |
| `--decoupler-bar-top-n-down` | `decoupler_bar_top_n_down` | `None` | Number of negative activities to show when signed barplots are split. |
| `--msigdb-gene-sets` | `msigdb_gene_sets` | `HALLMARK`, `REACTOME` | Comma-separated MSigDB collection keywords or `.gmt` files. |
| `--msigdb-method` | `msigdb_method` | `consensus` | Method override for MSigDB pathway activity. |
| `--msigdb-min-n-targets` | `msigdb_min_n_targets` | `5` | Minimum targets per MSigDB pathway. |
| `--run-progeny` / `--no-run-progeny` | `run_progeny` | `True` | Enable PROGENy activity inference. |
| `--progeny-method` | `progeny_method` | `consensus` | Method override for PROGENy. |
| `--progeny-min-n-targets` | `progeny_min_n_targets` | `5` | Minimum targets per PROGENy pathway. |
| `--progeny-top-n` | `progeny_top_n` | `100` | Number of top PROGENy target genes to use per pathway. |
| `--progeny-organism` | `progeny_organism` | `human` | Organism used when loading PROGENy resources. |
| `--run-dorothea` / `--no-run-dorothea` | `run_dorothea` | `True` | Enable DoRothEA TF activity inference. |
| `--dorothea-method` | `dorothea_method` | `consensus` | Method override for DoRothEA. |
| `--dorothea-min-n-targets` | `dorothea_min_n_targets` | `5` | Minimum targets per TF regulon. |
| `--dorothea-confidence` | `dorothea_confidence` | `A,B,C` | DoRothEA regulon confidence levels to include. |
| `--dorothea-organism` | `dorothea_organism` | `human` | Organism used when loading DoRothEA resources. |

Compaction has additional decoupler-derived similarity thresholds such as `--thr-progeny`, `--thr-dorothea`, and `--thr-msigdb-default`; those are documented on the [Compaction](compaction.md) page because they control merge decisions rather than activity inference itself.

For details on decoupler methods and resources, see:

[https://decoupler-py.readthedocs.io](https://decoupler-py.readthedocs.io)

---
