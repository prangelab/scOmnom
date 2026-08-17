# Cell-cell Communication

The `markers-and-de ccc` command group contains the cell-cell communication backends. These workflows are optional downstream analyses after clustering, annotation, marker/DE review, and final label cleanup.

| Backend | Command | Main question |
| --- | --- | --- |
| LIANA | `scomnom markers-and-de ccc liana` | Which ligand-receptor pairs connect sender and receiver populations? |
| NicheNet | `scomnom markers-and-de ccc nichenet` | Which sender ligands best explain a receiver transcriptional program? |
| MEBOCOST | `scomnom markers-and-de ccc mebocost` | Which metabolite-sensor routes connect sender and receiver populations? |

LIANA and MEBOCOST also have paired rescoring modes. Those do not rerun full pooled discovery; they take a focused candidate table and score candidate routes per donor or sample.

## Shared Concepts

CCC commands use the selected clustering round to resolve cell population labels. By default, scOmnom uses the active clustering round and pretty labels where possible.

| Option | Default | Notes |
| --- | --- | --- |
| `--round-id` | active clustering round | Selects the clustering round used for sender/receiver labels. |
| `--group-key` | resolved from round | Override the sender/receiver label column directly. |
| `--label-source` | `pretty` | Use pretty labels when available. |
| `--output-dir`, `-o` | inferred `results/` location | Output root. |
| `--output-name` | inferred from input, backend, and round | Saved AnnData name. |
| `--save-h5ad` / `--no-save-h5ad` | `--no-save-h5ad` | Also write h5ad output. |
| `--make-figures` / `--no-make-figures` | backend-specific | Pooled discovery defaults to figures on; MEBOCOST paired defaults to figures off. |
| `--figdir-name` | `figures` | Figure root directory name. |
| `--figure-formats`, `-F` | `png`, `pdf` | Repeatable output formats. |

## Condition Syntax

Condition syntax differs slightly by backend.

| Syntax | LIANA pooled | NicheNet | MEBOCOST pooled | Paired rescoring |
| --- | --- | --- | --- | --- |
| omitted | one full-object run | requires `--gene-list-file` if no condition is supplied | one full-object run | one run over the filtered object |
| `A` | one run per level of `A` | compare two levels of `A` | one run per level of `A` | subset/effect grouping by `A` |
| `A:B` | one run per composite level | not used | not used | not used |
| `A@B` | run `A` within each level of `B` | compare two levels of `A` within each level of `B` | run `A` within each level of `B` | subset by `B`, compare/group by `A` |

`--condition-value` filters the context levels for `A@B`. `--compare-level` filters or defines the compared levels of the primary condition variable. For NicheNet receiver-DE mode, exactly two `--compare-level` values are required.

## Cross-tissue Or Cross-dataset Mode

When an object contains multiple tissues, samples, or datasets, CCC can restrict sender and receiver populations by dataset origin.

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset-key` | none | `adata.obs` column defining tissue/dataset origin. |
| `--source-level` | none | Allowed sender dataset levels. Required when `--dataset-key` is set. |
| `--target-level` | none | Allowed receiver dataset levels. Required when `--dataset-key` is set. |
| `--signal-scope` | `all` for LIANA/NicheNet | `all` or `secreted`; LIANA uses CellChatDB route annotations for secreted filtering. |

For cross-tissue runs, consider `--input-mode lognorm` when datasets differ strongly in depth or chemistry.

## Backend Choice

Use LIANA first for broad ligand-receptor candidate discovery. Use NicheNet when the receiver transcriptional response is the focus and you either have a receiver gene list or a clean two-level receiver DE contrast. Use MEBOCOST for metabolite-sensor communication. Use paired rescoring after pooled discovery when donor/sample-level evidence matters.

Detailed pages:

* [LIANA CCC](liana.md)
* [NicheNet CCC](nichenet.md)
* [MEBOCOST CCC](mebocost.md)

---
