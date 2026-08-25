# scOmnom Architecture Refactor Plan

## Status

Deferred until the current BISC validation campaign is complete. This document records the agreed direction only. No implementation should begin merely because this plan exists.

The work must be divided into small, independently validated changes. Dataset serialization correctness must be fixed before structural refactoring, and serializer changes must never be combined with module moves or unrelated cleanup.

## Primary Goals

1. Restore a lossless and fail-closed `save_dataset()` / `load_dataset()` contract without increasing matrix memory use.
2. Make the public API unambiguous.
3. Enforce the dataset-I/O and plot-persistence contracts automatically.
4. Split oversized workflow and utility modules along scientific and architectural boundaries without changing scientific behavior.
5. Remove dead backup code.

## 1. Dataset I/O Correctness

### Compound and normalized keys

AnnData backends may require unsafe dictionary keys to be normalized while writing. That normalization may remain, but it must be completely reversible on load.

Examples that must round-trip exactly include:

- `sex@MASLD`
- `female:male`
- `x/y`
- nested compound keys using the same characters

Normalization collisions must also remain lossless. For example, `sex@MASLD` and a literal `sex_MASLD` key must survive the same save/load cycle as two distinct keys with their original values.

Preferred implementation direction:

- Keep the existing save-side key map unless contract tests demonstrate that its representation is insufficient.
- Restore original dictionary keys in `load_dataset()` before normal payload rehydration.
- Apply restoration recursively to ordinary metadata dictionaries.
- Do not reinterpret internal tagged payload dictionaries as user metadata.
- Fail clearly if a corrupt or ambiguous key map would overwrite an existing restored key.
- Remove serializer-only key-map metadata from the user-facing loaded object after successful restoration.

This should be a metadata-only load-side correction. It must not copy, coerce, densify, or otherwise traverse expression matrices.

### Sidecar failures

A dataset save must not report success if any sidecar referenced by the stored metadata failed to write. The current warning-and-continue behavior can produce an unloadable dataset and must become fail-closed.

Preferred implementation direction:

- A failure to write any registered sidecar payload aborts the save.
- Successful sidecar writing is followed by lightweight reference/manifest verification.
- Verification inspects sidecar paths and metadata only; it must not call `load_dataset()` on the completed output or load `X` a second time.
- Archived Zarr output must retain its current staging behavior so an existing final archive is not replaced until the staged output is complete.
- Transactional behavior for unarchived Zarr directories should be evaluated separately rather than expanding the first correctness patch unnecessarily.
- The in-memory AnnData metadata must still be restored in `finally` paths after either success or failure.

### Mixed-DataFrame dtype fidelity

The current Zarr sidecar path converts a heterogeneous DataFrame to one string matrix. Reloading then restores each column with `astype(original_dtype)`. This is incorrect for Boolean columns because the non-empty string `"False"` converts to `True`. A minimal reproduction changed `[True, False]` into `[True, True]`, and archived compaction pairwise tables consequently reload with failing edges marked as passing.

The stored compaction merge maps were computed before serialization and remain correct; the persisted Boolean audit columns are not trustworthy until this is fixed. The correction must be generic rather than compaction-specific.

Preferred implementation direction:

- Store heterogeneous DataFrame columns independently, or use another typed representation that preserves each column without a shared string cast.
- Preserve Boolean, nullable Boolean, integer, nullable integer, floating, categorical, string, and missing-value semantics exactly.
- Never restore Boolean values through direct `astype(bool)` on strings.
- Version the sidecar representation if the on-disk contract changes.
- Retain a compatibility path for existing sidecars without pretending that unrecoverable historical Boolean values are reliable.

### Memory-safety invariants

The serializer was deliberately designed for large sparse datasets. All changes must preserve these invariants:

- No `adata.copy()` inside the save path.
- No copy of `adata.X` or count layers.
- No `toarray()`, implicit densification, or sparse-to-dense coercion.
- No post-save full reload performed as production validation.
- No second full in-memory AnnData representation.
- Temporary replacement of metadata is allowed only if object state is restored reliably.
- Additional temporary disk use is preferable to additional matrix RAM if staging is required.

### Approved internal I/O exception

The direct AnnData reads and writes used for controlled merge intermediates inside `io_utils` are approved internal scratch operations. They are confined to the I/O implementation and do not need to be routed recursively through the public wrappers.

### Required tests before implementation

Add contract tests before changing production behavior:

- Exact nested key round trips for Zarr directories, archived Zarr, and H5AD.
- Distinct round trips for unsafe keys and their normalized underscore equivalents.
- Legacy dataset loading with the current key-map representation.
- Sidecar write-failure injection that proves the save raises.
- Confirmation that a failed staged archive does not replace a valid existing archive.
- Sidecar manifest/reference completeness checks.
- Preservation of sparse matrix classes, shape, `nnz`, layers, `obsm`, `varm`, and metadata.
- Preservation of the original in-memory matrix and layer object identities across `save_dataset()`.
- A guard proving that the save path does not invoke `adata.copy()` or dense conversion.
- Representative large-object RSS comparison against the current serializer.
- Mixed string/Boolean/integer/float DataFrame round trips with both `True` and `False` values.
- Nullable Boolean and nullable integer round trips with missing values.
- A compaction edge-table round trip proving that passing and failing rows remain distinct.
- Cross-checks that stored decision summaries and their typed evidence tables remain mutually consistent.

## 2. Plot Persistence and Figure Rounds

### Immediate DE artifact persistence

The immediate persistence used by the two large marker-gene plotting workflows is deliberate and should remain. These routines can create many per-cluster figures; accumulating every figure until the workflow ends would create avoidable memory pressure.

This path is valid because it still routes artifacts through `persist_plot_artifacts()` and the central `save_multi()` machinery. It should be documented and enforced as an approved streaming-persistence path rather than treated as an accidental exception.

A future cleanup may expose this more explicitly as a generator or internal artifact sink, but it must preserve bounded figure memory. Do not replace it with collection of all figures in one list.

### Concurrent figure-round allocation

The current figure-round allocator assumes a single writer for a given results tree and run key. That assumption is acceptable for now.

A local atomic-directory fix would not solve two Snellius jobs operating in independent `$TMPDIR` snapshots and later merging their result trees. A real concurrency feature would require unique explicit run identifiers or coordination on the final shared filesystem.

Therefore:

- Do not add a misleading local-only lock.
- Treat concurrent writers using the same run key as unsupported.
- Use distinct explicit round IDs when concurrent jobs are intentionally launched.
- Revisit only if concurrent same-key output becomes a supported workflow.

## 3. Public API Cleanup

`scomnom.adata_ops` and `scomnom.markers_and_de` currently have confusing dual identities between public aliases and importable internal modules. Each public name should resolve to one real facade module.

Target behavior:

- `scomnom.adata_ops` is the canonical public adata-operations facade.
- `scomnom.markers_and_de` is the canonical public marker/DE/DA/enrichment/CCC facade.
- `scomnom.plotting` remains the canonical plotting facade.
- `scomnom.__init__` exposes real modules and explicit exports, not `SimpleNamespace` substitutes.
- CLI entry points import workflow implementations directly rather than reaching through public facades.
- Internal workflow functions are not documented or exported as public API accidentally.

The cleanup must update `API_REFERENCE.md` and affected notebooks in the same change. Import-identity and export-surface tests must prevent the ambiguity from returning.

## 4. Architectural Enforcement

Add focused architecture tests before or alongside module extraction. Prefer small AST-based checks with named exceptions over a disruptive repository-wide lint rewrite.

Required enforcement:

- Direct AnnData dataset reads/writes are forbidden outside the approved I/O implementation and named merge-intermediate functions.
- Direct `savefig()` is forbidden outside plot persistence core and explicit notebook/API `file=` handling.
- Direct `save_multi()` calls are confined to plot persistence core.
- The approved streaming DE plot workflows may call `persist_plot_artifacts()` immediately.
- Public module identity and `__all__` contents are tested.
- Backup modules such as `bk_load_and_filter.py` are not permitted after removal.

Add CI for architecture checks and the fast unit suite. Introduce stricter linting incrementally. Broad static typing should wait until the oversized configs and workflows have been separated enough to avoid a low-value flood of errors.

## 5. Module Decomposition

Module extraction must be mechanical first: move code, preserve signatures and behavior, add temporary compatibility re-exports where necessary, and run the existing test suite. Algorithm cleanup comes only after the move is stable.

### `markers_and_de`

Split the workflow into five scientific domains:

- `markers`: cluster-versus-rest marker discovery and marker plotting.
- `de`: within-cluster condition contrasts and pseudobulk DE.
- `enrichment`: gene-set enrichment and module-score workflows.
- `composition`: DA methods, including GLM, CLR, scCODA, Milo, and GraphDA.
- `ccc`: shared CCC orchestration plus LIANA, MEBOCOST, and NicheNet components.

`markers_and_de.py` should ultimately become a small public compatibility/facade module rather than a workflow implementation containing all five domains.

Configuration should eventually follow the same domains. Replace the single oversized configuration gradually with focused marker, DE, enrichment, composition, and CCC configurations. Do not combine this with the initial mechanical function moves.

### `plot_utils`

Keep central state and persistence contracts together:

- `PlotArtifact`
- artifact collection
- `persist_plot_artifacts()`
- `save_multi()`
- output path and format routing
- figure-round handling
- shared style primitives

Move domain-specific construction into grouped modules for QC, integration, clustering, markers, DE, enrichment, composition, and CCC. Preserve the public `plotting.py` facade.

### `io_utils`

Keep `load_dataset()` and `save_dataset()` in `io_utils.py` so the documented project-wide I/O boundary remains literal and easy to enforce.

Extract unrelated concerns into focused modules:

- 10x/raw/filtered/CellBender discovery and loading
- merge preparation and scratch intermediates
- CellTypist and other resource acquisition
- gene-set and MSigDB acquisition
- table and spreadsheet exports

Temporary re-exports from `io_utils.py` may be used during migration, then removed after internal imports and tests are updated.

### Dependency direction

The intended dependency flow is:

1. CLI and public facades
2. workflow orchestrators
3. scientific compute, plotting, and I/O helpers
4. low-level shared utilities

Lower layers must not import workflow modules. Shared QC calculations currently needed by plotting should move to a neutral QC utility module rather than creating reverse plotting-to-workflow dependencies.

## 6. Removal of Dead Code

Delete `src/scomnom/bk_load_and_filter.py` in a dedicated cleanup change after confirming that it has no imports or runtime references. Do not preserve source backups inside the package; Git history is the backup.

## Implementation Sequence

1. Freeze this plan until BISC validation is complete.
2. Add serializer contract tests without changing production code.
3. Implement exact compound-key restoration as an isolated patch.
4. Implement fail-closed sidecar handling and lightweight verification as a separate isolated patch.
5. Run synthetic, compatibility, large-object, and RSS validation for the serializer.
6. Add architecture contract tests and CI.
7. Remove `bk_load_and_filter.py` independently.
8. Resolve public API identity while establishing the new workflow module boundaries.
9. Extract markers, DE, enrichment, composition, and CCC workflows mechanically.
10. Extract domain plotting helpers while retaining the central persistence core.
11. Reduce `io_utils` to dataset serialization plus deliberate compatibility exports.
12. Split configurations and simplify internal APIs only after the structural moves are stable.

Each stage should be independently commit-ready and revertible. Serializer changes, scientific behavior changes, public API changes, and bulk module moves must not be mixed in one commit.

## Acceptance Criteria

The refactor is complete only when:

- Dataset keys round-trip exactly, including compound keys and normalization collisions.
- A successful save cannot contain missing referenced sidecars.
- Large sparse saves show no matrix-copy or densification regression.
- Existing supported datasets still load.
- Public modules have one unambiguous identity.
- CLI and public API behavior remain covered by tests.
- Every CLI figure path uses central format/path persistence, including approved streaming persistence.
- Oversized workflow modules have clear scientific ownership and one-way dependencies.
- The full behavior suite and representative end-to-end workflows pass.

## Non-Goals

- Changing scientific algorithms, thresholds, or defaults during mechanical extraction.
- Reworking the two approved internal merge intermediates merely to satisfy a superficial rule.
- Supporting concurrent figure writers with the same run key in this refactor.
- Replacing bounded streaming plot persistence with bulk figure collection.
- Performing dependency modernization at the same time.
- Removing legacy loader compatibility before affected datasets have been safely re-saved.
