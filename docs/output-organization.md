# Output Organization

scOmnom uses two different kinds of "round" language:

* **Output rounds**: folders on disk that keep figures, reports, and tables from separate command runs from overwriting each other.
* **Clustering rounds**: internal AnnData state stored under `adata.uns["cluster_rounds"]`.

These are related by provenance, but they are not the same thing.

## Output rounds on disk

Many modules write figures and tables into module-specific run folders such as:

```text
results/
├── figures/
│   ├── png/
│   │   ├── integration_round1/
│   │   ├── de_r3_manual_rename_round1/
│   │   └── ccc_liana_r3_manual_rename_round1/
│   └── pdf/
│       ├── integration_round1/
│       ├── de_r3_manual_rename_round1/
│       └── ccc_liana_r3_manual_rename_round1/
└── tables/
    ├── de_r3_manual_rename_round1/
    ├── enrichment_r3_manual_rename_round1/
    └── ccc_liana_r3_manual_rename_round1/
```

The trailing `roundN` in these folder names is an **output run counter**. It is there so a second plotting, integration, DE, enrichment, or CCC run can write a new result folder instead of silently overwriting an earlier one.

Use output rounds to answer:

* Which command run produced these files?
* Did I rerun this module and keep the earlier figures/tables?
* Which folder should I inspect for a report or exported table?

## Clustering rounds inside AnnData

Clustering rounds live inside the saved AnnData object:

```text
adata.uns["cluster_rounds"]
adata.uns["cluster_round_order"]
adata.uns["active_cluster_round"]
```

They describe biological annotation state: raw cluster labels, pretty labels, parent/derived round relationships, decoupler payloads, compaction provenance, rename/subset-merge state, and which round is active by default for downstream commands.

Use clustering rounds to answer:

* Which labels should markers, DE, DA, enrichment, or CCC use?
* Was this label set produced by BISC, compaction, manual rename, annotation merge, or projection?
* What is the parent round for this derived annotation?

See [scOmnom AnnData Structure](adata-structure.md) for the full clustering-round schema.

## How they interact

Downstream table and figure folders often include the selected clustering round id. For example:

```text
tables/de_r3_manual_rename_round1/
figures/png/de_r3_manual_rename_round1/
```

In that name:

* `de` is the module/action.
* `r3_manual_rename` is the **AnnData clustering round** used for labels.
* `round1` is the **output run counter** for files on disk.

If you rerun DE on the same clustering round, the AnnData round id may stay `r3_manual_rename`, while the output folder can advance to `round2`. If you instead create a new manual rename or subset-annotation state, the AnnData clustering round id changes, and downstream output folders should reflect that new label source.

## Practical rule

When interpreting results, first identify the AnnData clustering round used for labels, then identify the output run folder that contains the files you want. The clustering round tells you **what biological label state** was analyzed; the output round tells you **which filesystem run** produced a particular table, figure, or report.

---
