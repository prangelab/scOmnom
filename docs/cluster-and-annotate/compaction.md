# Compaction: consolidating redundant clusters

Compaction is an optional conservative step after BISC clustering and annotation. It tests whether parent clusters with the same trusted CellTypist label lack substantial transcriptomic state divergence and agree across independent regulatory and pathway activity views. Supported clusters are consolidated in a new clustering round; the BISC parent round remains available.

Compaction does not infer missing evidence. A cluster with an `Unknown` label, a failed CellTypist confidence or purity gate, an incomplete required activity profile, or fewer than `compact_min_cells` cells remains a singleton and is recorded as ineligible.

## Required evidence

Compaction uses the round-specific CellTypist cluster-label audit and decoupler activities from:

* authoritative cluster pseudobulk counts, required;
* PROGENy, required;
* DoRothEA, required;
* available MSigDB GMT blocks, required by default.

The CellTypist gate is inherited from the annotation step. Compaction does not replace it with an unmasked vote over cell-level predictions.

With `compact_transcriptomic_source=auto`, the transcriptomic view uses `counts_cb` when available and otherwise `counts_raw`. Compaction stops if neither authoritative count layer is present. `adata.X` can be selected explicitly only when it contains nonnegative counts. Counts are summed by parent cluster and normalized to 10,000 counts. scOmnom also retains the 2,000 genes with the largest variance across log-transformed parent-cluster pseudobulks for diagnostic Pearson concordance and stores the selected gene names and their SHA-256 digest in the round provenance.

Each numeric view is checked for duplicate axes, non-finite values, missing cluster rows, constant features, and inadequate dimensionality. A wholly unavailable required view stops compaction. A cluster missing required evidence remains a singleton.

## Transcriptomic state-divergence veto

For each candidate pair, scOmnom calculates gene-wise absolute log2 fold changes and absolute detection-fraction differences from the count pseudobulks. A gene is eligible when it is detected in at least 5% of cells in either cluster and is not mitochondrial, ribosomal, `MTRNR*`, or `MALAT1`. The default medium envelope counts genes with both absolute log2 fold change at least 1.0 and detection-fraction difference at least 0.20. If more than 2% of eligible genes meet both criteria, the pair is vetoed. Exactly 2% passes. This is a one-sided safeguard: it can block a merge supported by the annotation and activity views, but it cannot create a merge.

Loose (0.75/0.15) and strict (1.50/0.25) envelopes are recorded as diagnostics. Pearson correlation over the frozen variable-gene set is also retained as a diagnostic and does not enter the merge decision. The default medium-envelope boundary was calibrated on development datasets and independently stress-tested; it is configurable rather than presented as a universal biological constant.

Valid activity features are z-scored across all clusters with complete evidence for that view. PROGENy and DoRothEA pairs are compared by cosine similarity.

For each MSigDB GMT block, scOmnom takes the union of the 25 features with the largest absolute z-score in either cluster and calculates cosine similarity on that union. When `msigdb_required=False`, available MSigDB similarities remain in the audit output but do not determine whether a pair passes.

## Threshold policy

Each activity view has an immutable evidence floor. Pearson retains its historical diagnostic floor for audit continuity:

| View | Floor |
|---|---:|
| Transcriptome Pearson (diagnostic only) | 0.90 |
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
* `compact_transcriptomic_threshold_cap=0.99` for the diagnostic Pearson threshold;
* `compact_state_divergence_log2fc_threshold=1.0`;
* `compact_state_divergence_detection_delta_threshold=0.20`;
* `compact_state_divergence_max_fraction=0.02`;
* `compact_adaptive_quantile=0.90`.

Legacy `thr_*` configuration keys and CLI flags are accepted as compatibility aliases for these caps.

## Pair and group decisions

A pair passes when it is not vetoed by transcriptomic state divergence, PROGENy and DoRothEA pass, and, when required, at least `ceil(0.67 * n_blocks)` valid MSigDB blocks pass. Pairs are only evaluated within the same trusted CellTypist label.

The default `compact_grouping="complete_link"` forms deterministic groups in which every pair has passed. This prevents a chain such as A-B and B-C from merging A, B, and C when A-C failed. `connected_components` remains available only for replaying historical configurations. The legacy value `clique` maps to `complete_link`.

## Outputs

Compaction always creates and activates an explicit child round, including when no pair merges. A no-op child is stored with `did_merge=False`, so downstream stages retain one predictable state without implying that a biological merge occurred.

The child round stores:

* the compaction method identity and full configuration snapshot;
* transcriptomic source, normalization, state-divergence thresholds, diagnostic feature selection, selected genes, and feature-set digest;
* upstream activity-method provenance;
* activity-view validation results;
* one eligibility record for every parent cluster;
* per-label floors, adaptive values, caps, and effective thresholds;
* all state-divergence envelopes, diagnostic similarities, vetoes, and pass decisions;
* complete-link components, parent-to-child membership, and reverse mappings.

The CLI writes `view_audit.tsv`, `cluster_eligibility.tsv`, `thresholds_by_label.tsv`, `pairwise_evidence.tsv`, and `group_membership.tsv` under the round-specific `tables/cluster_and_annotate/` tree. The `compaction_review` figure summarizes candidate confidence and the pairs nearest the decision boundary; `compaction_flow` shows the parent-to-child mapping.
