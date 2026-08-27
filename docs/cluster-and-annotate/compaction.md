# Compaction: consolidating redundant clusters

Compaction is an optional conservative step after BISC clustering and annotation. It tests whether parent clusters with the same trusted CellTypist label also agree in direct transcriptomic expression and across independent regulatory and pathway activity views. Supported clusters are consolidated in a new clustering round; the BISC parent round remains available.

Compaction does not infer missing evidence. A cluster with an `Unknown` label, a failed CellTypist confidence or purity gate, an incomplete required activity profile, or fewer than `compact_min_cells` cells remains a singleton and is recorded as ineligible.

## Required evidence

Compaction uses the round-specific CellTypist cluster-label audit and decoupler activities from:

* cluster pseudobulk expression, required;
* PROGENy, required;
* DoRothEA, required;
* available MSigDB GMT blocks, required by default.

The CellTypist gate is inherited from the annotation step. Compaction does not replace it with an unmasked vote over cell-level predictions.

The transcriptomic view uses `counts_cb` when available, then `counts_raw`, and finally `adata.X`. Count assays are summed by parent cluster, normalized to 10,000 counts, and log-transformed. Existing `adata.X` values are averaged by cluster. scOmnom retains the 2,000 genes with the largest variance across parent-cluster pseudobulks by default and stores the selected gene names and their SHA-256 digest in the round provenance.

Each numeric view is checked for duplicate axes, non-finite values, missing cluster rows, constant features, and inadequate dimensionality. A wholly unavailable required view stops compaction. A cluster missing required evidence remains a singleton.

## Similarity calculation

Parent-cluster transcriptomes are compared by Pearson correlation across the frozen variable-gene set. Valid activity features are z-scored across all clusters with complete evidence for that view. PROGENy and DoRothEA pairs are compared by cosine similarity.

For each MSigDB GMT block, scOmnom takes the union of the 25 features with the largest absolute z-score in either cluster and calculates cosine similarity on that union. When `msigdb_required=False`, available MSigDB similarities remain in the audit output but do not determine whether a pair passes.

## Threshold policy

Every view has an immutable evidence floor:

| View | Floor |
|---|---:|
| Transcriptome | 0.90 |
| PROGENy | 0.70 |
| DoRothEA | 0.60 |
| MSigDB HALLMARK | 0.60 |
| MSigDB REACTOME | 0.45 |
| Other MSigDB blocks | 0.50 |

For CellTypist groups containing two or three eligible clusters, scOmnom uses the floor directly. For groups containing at least four eligible clusters, it also calculates the configured within-group similarity quantile, 0.90 by default. The effective threshold is:

```text
min(user cap, max(immutable floor, adaptive quantile))
```

The cap limits how strict the adaptive threshold can become; it cannot lower the threshold below its floor. The relevant settings are:

* `compact_progeny_threshold_cap=0.98`;
* `compact_dorothea_threshold_cap=0.98`;
* `compact_msigdb_threshold_cap=0.98`;
* `compact_msigdb_threshold_cap_by_gmt` for per-GMT overrides;
* `compact_transcriptomic_source=auto`;
* `compact_transcriptomic_n_features=2000`;
* `compact_transcriptomic_threshold_cap=0.99`;
* `compact_adaptive_quantile=0.90`.

Legacy `thr_*` configuration keys and CLI flags are accepted as compatibility aliases for these caps.

## Pair and group decisions

A pair passes when transcriptomic concordance, PROGENy, and DoRothEA pass and, when required, at least `ceil(0.67 * n_blocks)` valid MSigDB blocks pass. Pairs are only evaluated within the same trusted CellTypist label.

The default `compact_grouping="complete_link"` forms deterministic groups in which every pair has passed. This prevents a chain such as A-B and B-C from merging A, B, and C when A-C failed. `connected_components` remains available only for replaying historical configurations. The legacy value `clique` maps to `complete_link`.

## Outputs

Compaction always creates and activates an explicit child round, including when no pair merges. A no-op child is stored with `did_merge=False`, so downstream stages retain one predictable state without implying that a biological merge occurred.

The child round stores:

* the compaction method identity and full configuration snapshot;
* transcriptomic source, normalization, feature selection, selected genes, and feature-set digest;
* upstream activity-method provenance;
* activity-view validation results;
* one eligibility record for every parent cluster;
* per-label floors, adaptive values, caps, and effective thresholds;
* all evaluated pairwise similarities and pass decisions;
* complete-link components, parent-to-child membership, and reverse mappings.

The CLI writes `view_audit.tsv`, `cluster_eligibility.tsv`, `thresholds_by_label.tsv`, `pairwise_evidence.tsv`, and `group_membership.tsv` under the round-specific `tables/cluster_and_annotate/` tree. The `compaction_review` figure summarizes candidate confidence and the pairs nearest the decision boundary; `compaction_flow` shows the parent-to-child mapping.
