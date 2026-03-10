# scOmnom project-wide agent instructions

These rules are persistent for all threads and apply across the whole project.

## Data IO (mandatory)
- All dataset loading must go through `load_dataset()` in `src/scomnom/io_utils.py`.
- All dataset saving must go through `save_dataset()` in `src/scomnom/io_utils.py`.
- Do not call `adata.write_zarr`, `adata.write_h5ad`, `ad.read_zarr`, or `ad.read_h5ad` directly in new or modified code.
- Temporary compatibility note: `load_dataset()` currently includes a legacy `_rehydrate` compatibility section to recover historically corrupted/stringified tagged payloads in `adata.uns` (notably old palette payloads).
- Once legacy affected datasets have been re-saved with the fixed serializer, remove that compatibility branch from `_rehydrate` to keep the loader minimal.

## Plot Saving (mandatory)
- Plot functions should emit `PlotArtifact` objects (from `src/scomnom/plot_utils.py`) and must not do direct filesystem saving themselves.
- `PlotArtifact` contains:
  - `stem`: output stem
  - `figdir`: target figure directory
  - `fig`: matplotlib figure handle
  - `savefig_kwargs`: optional kwargs for save behavior
- For CLI/pipeline code paths:
  - Collect artifacts (for example via `@collect_plot_artifacts` / `record_plot_artifact(...)`).
  - Persist them from orchestrator call sites using `persist_plot_artifacts(...)` (which routes to `save_multi(...)`).
  - This keeps file-format/path routing centralized and consistent.
- For API/notebook code paths (`src/scomnom/plotting_api_utils.py`):
  - Do not use CLI routing (`save_multi`) for user-facing file output.
  - Wrappers should display figures and only save when explicit `file=...` is provided.

## API Surface (mandatory)
- Public API must be exposed only via:
  - `src/scomnom/__init__.py`
  - `src/scomnom/plotting.py`
  - `src/scomnom/adata_public.py`
- Do not treat internal module functions as public unless explicitly exported in the files above.

## Plot Parameters (mandatory)
- New plotting parameters must default to current behavior.
- Adding parameters must not change existing CLI behavior unless explicitly requested.

## Plot Architecture (mandatory)
- Underlying plotting functions should construct/return artifacts, not perform direct persistence.
- Orchestrators/pipeline call sites are responsible for artifact persistence.
- API wrappers are responsible for notebook display behavior and optional explicit file saving.

## Public Signatures (mandatory)
- Do not expose CLI-internal arguments in public API signatures or docs.
- Hide/internalize parameters such as: `figdir`, `cfg`, `stage`, `artifact_stem`, `artifact_figdir`, `savefig_kwargs`, and similar routing/plumbing arguments.

## Documentation Sync (mandatory)
- Any public API change must update `API_REFERENCE.md` in the same change.

## Notebook Examples (mandatory)
- If public API surface changes, update notebooks under `notebooks/` that demonstrate the affected functions.

## Change Reporting (mandatory)
- Always include file paths and line numbers for any code changes you make, so I can verify quickly.
