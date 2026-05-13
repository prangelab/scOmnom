# Temporary Spec: DE Enrichment Storage and Plotting Stabilization

## Goal

Reduce `adata.uns` growth from DE-driven enrichment payloads, and harden the downstream plotting path so DE enrichment runs do not OOM during figure generation.

## Problem

The current `markers-and-de de` enrichment flow stores:

- `msigdb` decoupler payload
- `msigdb_gsea["results"]` full long-form table
- `msigdb_joint["results"]` full merged long-form concordance table

inside `adata.uns`.

In addition, DE-decoupler plotting currently appears to retain figure objects for too long and uses a plotting executor pattern that is nominally parallel but effectively serialized.

This is likely overloading `adata.uns` because:

- GSEA stores one row per `cluster x pathway`
- joint stores another full long table with overlapping pathway-level information
- runs may include multiple condition keys, contrasts, and both `cell` and `pseudobulk` sources
- string-heavy columns such as `leading_edge` and `leading_edge_preview` are expensive

## Desired Change

The implementation order should be:

1. Move large DE enrichment payloads out of `adata.uns`
2. Fix figure/artifact retention
3. Fix plotting parallelization

## 1. Move large DE enrichment payloads out of `adata.uns`

### Keep in `adata.uns`

Only compact summary payloads should remain in `adata.uns` for DE enrichment outputs.

For `msigdb_gsea`, keep only:

- `config`
- small summary metadata such as:
  - number of clusters
  - number of pathways
  - number of rows
  - selected MSigDB collections
  - significance threshold if relevant
  - maybe per-cluster top-hit summaries if needed for lightweight introspection

For `msigdb_joint`, keep only:

- `config`
- small summary metadata such as:
  - number of concordant rows
  - number of supported-by-both rows
  - per-cluster counts
  - maybe top supported pathways per cluster/family if genuinely useful

For DE-decoupler payloads such as `msigdb`, `progeny`, and `dorothea`:

- apply the same principle in one go wherever it makes sense
- keep only summary-scale payloads in `adata.uns`
- if a DE-decoupler payload carries large tables that are not needed for lightweight introspection, move those to disk as well

The intention is to treat this as a general DE enrichment storage cleanup, not a GSEA-only exception.

### Move to disk

Write the full long-form DE enrichment tables to the run output tree only.

These should live under the DE results tables directory, for example:

- `.../tables/msigdb_gsea/results.tsv`
- `.../tables/msigdb_joint/results.tsv`
- `.../tables/msigdb/...` for any large DE-decoupler tables if applicable
- corresponding locations for `progeny` / `dorothea` if they also benefit from disk-first storage

If needed later, optional family-split or cluster-split exports can also live there, but the primary requirement is that the full tables no longer live in `adata.uns`.

## Scope

This spec is primarily about DE-driven enrichment storage for:

- `markers-and-de de`
- `msigdb_gsea`
- `msigdb_joint`
- DE-decoupler payloads where large tables are currently retained unnecessarily

It does not require changing:

- cluster-level decoupler enrichment storage
- plotting outputs
- report generation semantics
- decoupler-only DE enrichment storage

## Behavioral Requirements

1. Full DE enrichment tables must still be written to disk in the normal results/tables output tree.
2. Plotting and reporting must continue to work from run outputs without relying on full GSEA/joint/decoupler tables being embedded in `adata.uns`.
3. Regenerate-figures workflows should continue to function, but may need to prefer loading from tables on disk instead of expecting full tables in `adata.uns`.
4. Saved AnnData size should drop materially for DE runs with GSEA enabled.

## Implementation Direction

Preferred direction:

1. Keep current computation flow unchanged.
2. Export full `msigdb_gsea`, `msigdb_joint`, and any large DE-decoupler tables to disk.
3. Replace in-memory stored payloads with summary-only payloads before final AnnData save.

This avoids changing the enrichment calculations themselves and limits the change to storage/export boundaries.

## 2. Fix figure/artifact retention

The current plotting path should be hardened so saved figures are fully released from memory immediately after persistence.

Preferred direction:

- fix this at the artifact/persistence machinery level if possible, so every module benefits at once
- after `persist_plot_artifacts(...)`, figure references should be dropped deterministically
- completed plot task results should not keep strong references to large `Figure` objects longer than necessary

This should be treated as a general infrastructure bug, not just a DE-enrichment-specific workaround, if the central artifact machinery can be improved safely.

## 3. Fix plotting parallelization

The current DE enrichment plotting executor should be made genuinely parallel rather than effectively serialized.

Requirements:

- remove the current broken pseudo-parallel pattern
- process plotting tasks in bounded batches
- cap at `8` in-flight tasks at a time
- after each batch or completed task, persist figures and release figure/object references immediately

`8` is the intended default cap unless later profiling suggests a smaller safe default is necessary for matplotlib stability.

## Open Questions

1. Should `adata.uns` retain a pointer-like record to the exported table path, or should it contain summaries only and let the report/plot layer rediscover the tables from the run folder?
2. For plot-only / regenerate-figures mode, should tables-on-disk become the canonical source for GSEA/joint rerendering?
3. Should `leading_edge_preview` stay in summaries if we keep a top-hit summary block, or should all leading-edge text be disk-only?
4. For DE-decoupler payloads, which exact pieces are worth preserving in-memory versus disk-only?
5. Can the artifact retention fix be implemented centrally without changing notebook/API semantics?
6. Is `8` the right long-term batch cap everywhere, or should DE-enrichment plotting own that cap specifically?

## Non-Goals

- changing GSEA statistics
- changing concordance logic
- changing DE computation
- changing CLI semantics
